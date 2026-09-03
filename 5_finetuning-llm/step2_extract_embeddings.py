from __future__ import annotations

import os
import pickle
import random
import sys
import time
from pathlib import Path

import fire
import numpy as np
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "1_data-preprocessing"))

from utils_ft import BASE_LLM, WEIGHTS_DIR, build_all_prompts

MODELS_DIR = HERE / "models"


def _seed_all(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_name_out(k: int, n: int, lora_targets: str, lora_r: int) -> str:
    """Filename used by Stage 3 as `model_name`. Avoid '/' so the TF code
    treats it as-is (no path mangling)."""
    return f"alpaca-ft-intersection-k{k}-{lora_targets}-r{lora_r}-n{n}"


def main(
    k: int = 0,
    n: int = 200_000,
    lora_targets: str = "all-linear",
    lora_r: int = 16,
    batch_size: int = 32,
    max_length: int = 192,
    seed: int = 42,
    sanity_check: bool = True,
):
    _seed_all(seed)
    tag = f"{lora_targets}_r{lora_r}"
    adapter_dir = MODELS_DIR / f"lora_intersection_k{k}_n{n}_{tag}"
    assert adapter_dir.exists(), f"adapter dir not found: {adapter_dir}"

    out_name = _model_name_out(k, n, lora_targets, lora_r)
    out_path = WEIGHTS_DIR / f"{out_name}.pkl"
    print(f"[stage2] adapter <- {adapter_dir}")
    print(f"[stage2] output  -> {out_path}")

    # --- load base + adapter, merge ---
    print("[stage2] loading base Alpaca-7b ...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(BASE_LLM, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[stage2] base+adapter ready in {time.time()-t0:.1f}s; device={device}")

    tok = AutoTokenizer.from_pretrained(BASE_LLM)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- build the SAME prompt set as step9_generate_embeddings ---
    print("[stage2] building prompts via step9._build_prompts ...")
    prompts, n_questions = build_all_prompts()
    print(f"[stage2] n_questions={n_questions}")

    # --- forward + last-token hidden state ---
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="extracting"):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to(device)
            out = model(**enc, output_hidden_states=True)
            seq_lens = enc["attention_mask"].sum(dim=1) - 1
            hidden = out.hidden_states[-1]
            emb = hidden[torch.arange(hidden.size(0), device=device), seq_lens].float().cpu().numpy()
            embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    print(f"[stage2] embeddings.shape = {embeddings.shape}, dtype={embeddings.dtype}")

    if sanity_check:
        ref_path = WEIGHTS_DIR / "maicomputer_alpaca-native.pkl"
        if ref_path.exists():
            ref = pickle.load(open(ref_path, "rb"))
            if isinstance(ref, list):
                ref = np.vstack(ref)
            assert ref.shape == embeddings.shape, f"shape mismatch: {ref.shape} vs {embeddings.shape}"
            # quick cosine vs baseline
            ref_norm = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
            ft_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            cos = (ref_norm * ft_norm).sum(axis=1)
            print(f"[stage2] cosine vs baseline: mean={cos.mean():.4f}, min={cos.min():.4f}, max={cos.max():.4f}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(embeddings, open(out_path, "wb"))
    print(f"[stage2] saved -> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
