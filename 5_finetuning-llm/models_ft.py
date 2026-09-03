from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


class CrossLayer(nn.Module):
    """Low-rank DCN-V2 cross: x_{l+1} = x0 * (U(V^T x_l)) + x_l.

    U has a bias term (matches tfrs.layers.dcn.Cross default with use_bias=True
    where the bias is on the inner U-projection).
    """

    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.V = nn.Linear(in_dim, proj_dim, bias=False)
        self.U = nn.Linear(proj_dim, in_dim, bias=True)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.zeros_(self.U.bias)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.U(self.V(x)) + x


class DCNHead(nn.Module):
    """Shared head module: ind/year/q embeddings + cross + dense + output.

    The question representation `q_h` is provided externally (either via frozen
    lookup or via LLM forward) so this module is identical across phases.
    """

    def __init__(self, n_individuals: int, n_years: int, hidden: int = 4096, dim: int = 50):
        super().__init__()
        self.ind_emb = nn.Embedding(n_individuals, dim)
        self.year_emb = nn.Embedding(n_years, dim)
        self.q_proj = nn.Linear(hidden, dim)
        nn.init.normal_(self.ind_emb.weight, std=0.05)
        nn.init.normal_(self.year_emb.weight, std=0.05)

        in_dim = dim * 3  # concat(ind, q, year) -> 150
        self.cross = nn.ModuleList([CrossLayer(in_dim, proj_dim=in_dim) for _ in range(3)])
        self.dense = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(3)])
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(in_dim, 1)

    def forward(self, ind_id: torch.Tensor, year_id: torch.Tensor, q_h: torch.Tensor) -> torch.Tensor:
        """Returns logits [B] (BCE-with-logits)."""
        x2 = self.q_proj(q_h)
        x1 = self.ind_emb(ind_id)
        x3 = self.year_emb(year_id)
        x0 = torch.cat([x1, x2, x3], dim=1)
        x = x0
        for cl in self.cross:
            x = self.drop(cl(x0, x))
        for dl in self.dense:
            x = self.drop(F.relu(dl(x)))
        return self.out(x).squeeze(-1)


class FrozenEmbDCN(nn.Module):
    """Stage 0: DCN head with FROZEN precomputed question embeddings.

    The embedding matrix is loaded once and stored as a non-trainable buffer.
    No LLM is ever instantiated.
    """

    def __init__(self, weights: np.ndarray, n_individuals: int, n_years: int, dim: int = 50):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer("question_table", w, persistent=False)
        self.head = DCNHead(n_individuals=n_individuals, n_years=n_years,
                            hidden=w.shape[1], dim=dim)

    @property
    def hidden(self) -> int:
        return int(self.question_table.shape[1])

    def forward(self, ind_id: torch.Tensor, year_id: torch.Tensor, q_id: torch.Tensor) -> torch.Tensor:
        q_h = self.question_table[q_id]
        return self.head(ind_id, year_id, q_h)


class LlamaWithDCN(nn.Module):
    """Stage 1: LoRA-wrapped Alpaca-7b + DCN head.

    Forward strategy (for unique-question batching):

        encode_questions(input_ids, attn) -> q_hidden [B_q, H]
        forward(ind_id [B], year_id [B], inv_idx [B], q_hidden [B_q, H]) -> logits [B]
    """

    def __init__(
        self,
        n_individuals: int,
        n_years: int,
        base_name: str = "maicomputer/alpaca-native",
        dim: int = 50,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Sequence[str] = ("q_proj", "v_proj"),
        torch_dtype=torch.bfloat16,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch_dtype)
        if gradient_checkpointing:
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.llm = get_peft_model(base, lora_cfg)
        self.hidden = base.config.hidden_size  # 4096 for LLaMA-7B

        self.head = DCNHead(n_individuals=n_individuals, n_years=n_years,
                            hidden=self.hidden, dim=dim)

    def encode_questions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]
        seq_lens = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), seq_lens]

    def forward(
        self,
        ind_id: torch.Tensor,
        year_id: torch.Tensor,
        inv_idx: torch.Tensor,
        q_hidden: torch.Tensor,
    ) -> torch.Tensor:
        q_h = q_hidden.to(self.head.q_proj.weight.dtype)
        return self.head(ind_id, year_id, q_h[inv_idx])

    def load_head_state(self, head_state: dict, strict: bool = True):
        """Load a head state_dict produced by FrozenEmbDCN (Stage 0)."""
        self.head.load_state_dict(head_state, strict=strict)

    def param_groups(self, lora_lr: float = 1e-4, head_lr: float = 1e-3, weight_decay: float = 0.01):
        lora_params, head_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in n:
                lora_params.append(p)
            else:
                head_params.append(p)
        return [
            {"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]
