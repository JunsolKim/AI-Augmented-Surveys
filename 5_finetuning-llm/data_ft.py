from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils_ft import build_prompt


@dataclass
class Batch:
    ind: torch.Tensor          # [B] long
    year: torch.Tensor         # [B] long
    label: torch.Tensor        # [B] float
    inv: torch.Tensor          # [B] long — index into input_ids/attn
    input_ids: torch.Tensor    # [B_q, T] long
    attention_mask: torch.Tensor  # [B_q, T] long

    def to(self, device, non_blocking=True):
        return Batch(
            ind=self.ind.to(device, non_blocking=non_blocking),
            year=self.year.to(device, non_blocking=non_blocking),
            label=self.label.to(device, non_blocking=non_blocking),
            inv=self.inv.to(device, non_blocking=non_blocking),
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(device, non_blocking=non_blocking),
        )


class SubsampledFoldDataset(Dataset):
    """Wraps a pandas DataFrame of (yearid_id, question_id, year_order, binarized) rows."""

    def __init__(self, df: pd.DataFrame):
        self.ind = df["yearid_id"].to_numpy(dtype=np.int64)
        self.year = df["year_order"].to_numpy(dtype=np.int64)
        self.q = df["question_id"].to_numpy(dtype=np.int64)
        self.lbl = df["binarized"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.ind)

    def __getitem__(self, idx):
        return self.ind[idx], self.year[idx], self.q[idx], self.lbl[idx]


class CollateUniqueQ:
    """Tokenizes ONLY unique question prompts in the batch and returns an inverse index."""

    def __init__(self, tokenizer, question_dict, max_length: int = 192):
        self.tok = tokenizer
        self.qdict = question_dict
        self.max_length = max_length

    def __call__(self, rows: List[tuple]) -> Batch:
        ind, year, qids, lbl = zip(*rows)
        ind = torch.as_tensor(ind, dtype=torch.long)
        year = torch.as_tensor(year, dtype=torch.long)
        lbl = torch.as_tensor(lbl, dtype=torch.float)
        qids_t = torch.as_tensor(qids, dtype=torch.long)
        uniq, inv = torch.unique(qids_t, return_inverse=True)
        prompts = [build_prompt(self.qdict.get(int(q), "")) for q in uniq.tolist()]
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return Batch(
            ind=ind,
            year=year,
            label=lbl,
            inv=inv,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
