from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2_model-finetuning"))

from utils_ft import BASE_LLM, load_question_dict, load_intersection_train
from models_ft import LlamaWithDCN
from data_ft import SubsampledFoldDataset, CollateUniqueQ

MODELS_DIR = HERE / "models"
LOGS_DIR = HERE / "logs"
DATA_DIR_FT = HERE / "data"

ALL_LINEAR_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
QV_TARGETS = ["q_proj", "v_proj"]


def _seed_all(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tag(lora_targets: str, lora_r: int) -> str:
    return f"{lora_targets}_r{lora_r}"


def _subsample(train_df: pd.DataFrame, n: int, seed: int = 42):
    """Sample n rows from the training pool.
    The monitoring val is val_intersection (loaded separately), shared with Stage 0.
    """
    if n >= len(train_df):
        return train_df.copy()
    return train_df.sample(n=n, random_state=seed).reset_index(drop=True)


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    losses, preds, labels = [], [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            q_h = model.encode_questions(batch.input_ids, batch.attention_mask)
            logits = model(batch.ind, batch.year, batch.inv, q_h)
            loss = F.binary_cross_entropy_with_logits(logits, batch.label)
        losses.append(float(loss))
        preds.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(batch.label.cpu().numpy())
    model.train()
    preds = np.concatenate(preds); labels = np.concatenate(labels)
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else float("nan")
    return float(np.mean(losses)), float(auc)


def main(
    k: int = 0,
    n_splits: int = 10,
    n: int = 200_000,
    lora_targets: str = "all-linear",   # "all-linear" or "qv"
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    batch_size: int = 128,
    epochs: int = 1,
    lora_lr: float = 1e-4,
    head_lr: float = 5e-5,            # smaller — head is already warm-started
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    max_length: int = 192,
    num_workers: int = 4,
    log_every: int = 50,
    eval_every: int = 500,
    pretrain_head_path: str = "",     # if empty, uses models/head_pretrain_intersection_k{k}.pt
    seed: int = 42,
):
    _seed_all(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR_FT.mkdir(parents=True, exist_ok=True)

    targets = ALL_LINEAR_TARGETS if lora_targets == "all-linear" else QV_TARGETS
    tag = _tag(lora_targets, lora_r)
    run_name = f"intersection_k{k}_n{n}_{tag}"
    adapter_dir = MODELS_DIR / f"lora_{run_name}"
    head_path = MODELS_DIR / f"head_ft_{run_name}.pt"
    log_path = LOGS_DIR / f"stage1_{run_name}.json"
    sample_path = DATA_DIR_FT / f"sample_intersection_k{k}_n{n}.parquet"
    if not pretrain_head_path:
        pretrain_head_path = str(MODELS_DIR / f"head_pretrain_intersection_k{k}.pt")

    print(f"[stage1] run_name = {run_name}")
    print(f"[stage1] LoRA targets = {targets}, r={lora_r}, alpha={lora_alpha}")
    print(f"[stage1] warm-start head <- {pretrain_head_path}")

    # ----------------------- data -----------------------
    print("[stage1] loading intersection train + val_intersection ...")
    t0 = time.time()
    df_analysis, intersection_train, val_intersection = load_intersection_train(k=k, n_splits=n_splits)
    print(f"[stage1] intersection train = {len(intersection_train):,}, "
          f"val_intersection = {len(val_intersection):,}  ({time.time()-t0:.1f}s)")
    sub_train = _subsample(intersection_train, n=n, seed=seed)
    sub_val = val_intersection  # shared with Stage 0 — same val cells used by both stages
    print(f"[stage1] subsample: train={len(sub_train):,}, val={len(sub_val):,} (val_intersection)")

    keep_cols = ["yearid_id", "question_id", "year_order", "binarized"]
    sub_train[keep_cols].to_parquet(sample_path)

    n_individuals = int(df_analysis["yearid_id"].max()) + 1
    n_years = int(df_analysis["year_order"].max()) + 1
    print(f"[stage1] n_individuals={n_individuals}, n_years={n_years}")

    qdict, _n_q = load_question_dict()

    tok = AutoTokenizer.from_pretrained(BASE_LLM)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    collate = CollateUniqueQ(tok, qdict, max_length=max_length)
    train_ds = SubsampledFoldDataset(sub_train)
    val_ds = SubsampledFoldDataset(sub_val)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate, pin_memory=True,
    )

    # ----------------------- model -----------------------
    print("[stage1] building LlamaWithDCN ...")
    t0 = time.time()
    model = LlamaWithDCN(
        n_individuals=n_individuals,
        n_years=n_years,
        base_name=BASE_LLM,
        dim=50,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,
        torch_dtype=torch.bfloat16,
        gradient_checkpointing=True,
    )
    print(f"[stage1] base+LoRA built in {time.time()-t0:.1f}s")

    # warm-start head from Stage 0
    if Path(pretrain_head_path).exists():
        head_state = torch.load(pretrain_head_path, map_location="cpu")
        model.head.load_state_dict(head_state, strict=True)
        print(f"[stage1] loaded warm-started head from {pretrain_head_path}")
    else:
        print(f"[stage1] WARNING: pretrain head not found at {pretrain_head_path} — head will be random-init")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lora = sum(p.numel() for n_, p in model.named_parameters() if p.requires_grad and "lora_" in n_)
    print(f"[stage1] trainable params = {n_trainable:,}  (lora={n_lora:,})")

    # ----------------------- optim -----------------------
    optim = torch.optim.AdamW(model.param_groups(lora_lr=lora_lr, head_lr=head_lr, weight_decay=weight_decay))
    total_steps = len(train_loader) * epochs
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    log = {
        "config": {
            "k": k, "n_splits": n_splits, "n": n, "lora_targets": lora_targets,
            "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
            "batch_size": batch_size, "epochs": epochs,
            "lora_lr": lora_lr, "head_lr": head_lr, "weight_decay": weight_decay,
            "warmup_steps": warmup_steps, "max_length": max_length, "seed": seed,
            "n_train": len(sub_train), "n_val": len(sub_val),
            "total_steps": total_steps,
            "pretrain_head": pretrain_head_path,
        },
        "steps": [], "evals": [],
    }
    best_auc = -1.0
    step = 0
    t_start = time.time()

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                q_h = model.encode_questions(batch.input_ids, batch.attention_mask)
                logits = model(batch.ind, batch.year, batch.inv, q_h)
                loss = F.binary_cross_entropy_with_logits(logits, batch.label)
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

            step += 1
            if step % log_every == 0:
                log["steps"].append({"step": step, "epoch": epoch, "loss": float(loss),
                                     "wall_s": time.time() - t_start})
                pbar.set_postfix(loss=f"{float(loss):.4f}")
            if eval_every and step % eval_every == 0:
                vloss, vauc = _evaluate(model, val_loader, device)
                log["evals"].append({"step": step, "epoch": epoch, "val_loss": vloss, "val_auc": vauc})
                print(f"  [eval @ step {step}] val_loss={vloss:.4f}  val_auc={vauc:.4f}")
                if vauc > best_auc:
                    best_auc = vauc
                    model.llm.save_pretrained(str(adapter_dir))
                    torch.save(model.head.state_dict(), str(head_path))
                with open(log_path, "w") as f:
                    json.dump(log, f, indent=2)

        # end-of-epoch eval (always)
        vloss, vauc = _evaluate(model, val_loader, device)
        log["evals"].append({"step": step, "epoch": epoch, "val_loss": vloss, "val_auc": vauc,
                              "end_of_epoch": True})
        print(f"  [eval end-of-epoch {epoch}] val_loss={vloss:.4f}  val_auc={vauc:.4f}")
        if vauc > best_auc:
            best_auc = vauc
            model.llm.save_pretrained(str(adapter_dir))
            torch.save(model.head.state_dict(), str(head_path))
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    log["best_val_auc"] = best_auc
    log["wall_s_total"] = time.time() - t_start
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[stage1] DONE.  best val_auc={best_auc:.4f}  wall={log['wall_s_total']/60:.1f} min")
    print(f"[stage1] adapter -> {adapter_dir}")
    print(f"[stage1] head    -> {head_path}")
    print(f"[stage1] log     -> {log_path}")


if __name__ == "__main__":
    fire.Fire(main)
