from __future__ import annotations

import json
import os
import pickle
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2_model-finetuning"))

from models_ft import FrozenEmbDCN
from utils_ft import WEIGHTS_DIR, load_intersection_train

MODELS_DIR = HERE / "models"
LOGS_DIR = HERE / "logs"


def _seed_all(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_tensor_dataset(df: pd.DataFrame) -> TensorDataset:
    ind = torch.as_tensor(df["yearid_id"].to_numpy(np.int64))
    year = torch.as_tensor(df["year_order"].to_numpy(np.int64))
    qid = torch.as_tensor(df["question_id"].to_numpy(np.int64))
    lbl = torch.as_tensor(df["binarized"].to_numpy(np.float32))
    return TensorDataset(ind, year, qid, lbl)


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    losses, preds, labels = [], [], []
    for ind, year, qid, lbl in loader:
        ind = ind.to(device); year = year.to(device); qid = qid.to(device); lbl = lbl.to(device)
        logits = model(ind, year, qid)
        loss = F.binary_cross_entropy_with_logits(logits, lbl)
        losses.append(float(loss))
        preds.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(lbl.cpu().numpy())
    model.train()
    preds = np.concatenate(preds); labels = np.concatenate(labels)
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else float("nan")
    return float(np.mean(losses)), float(auc)


def main(
    k: int = 0,
    n_splits: int = 10,
    epochs: int = 2,
    batch_size: int = 4096,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    dim: int = 50,
    base_emb: str = "maicomputer_alpaca-native",
    num_workers: int = 4,
    eval_every_epoch: bool = True,
    seed: int = 42,
):
    _seed_all(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_head = MODELS_DIR / f"head_pretrain_intersection_k{k}.pt"
    out_log = LOGS_DIR / f"stage0_intersection_k{k}.json"

    # --- load intersection train + val ---
    print(f"[stage0] loading intersection train (impute ∩ partial ∩ total), k={k} ...")
    t0 = time.time()
    df_analysis, train, val_fold = load_intersection_train(k=k, n_splits=n_splits)
    print(f"[stage0] loaded in {time.time()-t0:.1f}s.  train={len(train):,}  val={len(val_fold):,}")

    n_individuals = int(df_analysis["yearid_id"].max()) + 1
    n_years = int(df_analysis["year_order"].max()) + 1

    # --- frozen embedding table ---
    weights_path = WEIGHTS_DIR / f"{base_emb}.pkl"
    weights = pickle.load(open(weights_path, "rb"))
    if isinstance(weights, list):
        weights = np.vstack(weights)
    print(f"[stage0] loaded embeddings {weights.shape} from {weights_path.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenEmbDCN(weights=weights, n_individuals=n_individuals, n_years=n_years, dim=dim).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stage0] trainable params = {n_train:,}")

    # --- data loaders ---
    train_ds = _to_tensor_dataset(train)
    val_ds = _to_tensor_dataset(val_fold)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # --- optim ---
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- train ---
    log = {
        "config": {
            "split_type": "intersection", "k": k, "n_splits": n_splits, "epochs": epochs,
            "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay, "dim": dim,
            "base_emb": base_emb, "n_train": len(train), "n_val": len(val_fold),
            "n_individuals": n_individuals, "n_years": n_years,
        },
        "epochs": [],
    }
    best_auc = -1.0
    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        loss_sum, n_seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for ind, year, qid, lbl in pbar:
            ind = ind.to(device, non_blocking=True); year = year.to(device, non_blocking=True)
            qid = qid.to(device, non_blocking=True); lbl = lbl.to(device, non_blocking=True)
            logits = model(ind, year, qid)
            loss = F.binary_cross_entropy_with_logits(logits, lbl)
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            bs = lbl.size(0); loss_sum += float(loss) * bs; n_seen += bs
            if n_seen and (n_seen // batch_size) % 200 == 0:
                pbar.set_postfix(loss=f"{loss_sum/n_seen:.4f}")
        train_loss = loss_sum / max(n_seen, 1)
        vloss, vauc = _evaluate(model, val_loader, device) if eval_every_epoch else (float("nan"), float("nan"))
        log["epochs"].append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": vloss, "val_auc": vauc, "wall_s": time.time() - t_start,
        })
        print(f"  epoch {epoch}: train_loss={train_loss:.4f}  val_loss={vloss:.4f}  val_auc={vauc:.4f}"
              f"  wall={(time.time()-t_start)/60:.1f}m")
        if vauc > best_auc:
            best_auc = vauc
            torch.save(model.head.state_dict(), str(out_head))
        with open(out_log, "w") as f:
            json.dump(log, f, indent=2)

    log["best_val_auc"] = best_auc
    log["wall_s_total"] = time.time() - t_start
    with open(out_log, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[stage0] DONE.  best val_auc={best_auc:.4f}  wall={log['wall_s_total']/60:.1f}m")
    print(f"[stage0] head    -> {out_head}")
    print(f"[stage0] log     -> {out_log}")


if __name__ == "__main__":
    fire.Fire(main)
