"""5-seed ensemble of C3-Pro-Reg on C3.2 features.

Trains 5 seeds independently, averages sigmoid(logits) per (anchor, app),
and reports both per-seed metrics and ensemble metrics.

Output:
    artifacts/bg/checkpoints/task_c_c3pro_ens_s{seed}.pt
    artifacts/bg/results/task_c_c3pro_ens.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.bg import metrics_bg as MET


SEEDS = [7, 13, 31, 53, 91]
H_SEC = 3600
Y_COL = f"y_{H_SEC}"
R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)
LR = 1e-3
WD = 1e-4
BATCH = 512
EPOCHS = 30
PATIENCE = 5
DROPOUT = 0.3
LABEL_SMOOTH = 0.05


def load_grid():
    if "trainer_grid_36" in sys.modules:
        return sys.modules["trainer_grid_36"]
    spec = importlib.util.spec_from_file_location(
        "trainer_grid_36", ROOT / "scripts" / "36_train_c3_grid.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["trainer_grid_36"] = m
    spec.loader.exec_module(m)
    return m


class PairDS(Dataset):
    def __init__(self, d):
        self.f = torch.as_tensor(d["features"].copy(), dtype=torch.float32)
        self.ai = torch.as_tensor(d["app_idx"].copy(), dtype=torch.long)
        self.y = torch.as_tensor(d["y"].copy(), dtype=torch.float32)

    def __len__(self):
        return self.f.shape[0]

    def __getitem__(self, i):
        return {"features": self.f[i], "app_idx": self.ai[i], "y": self.y[i]}


def forward_probs(model, data_dict):
    model.eval()
    f = torch.as_tensor(data_dict["features"].copy(), dtype=torch.float32)
    a = torch.as_tensor(data_dict["app_idx"].copy(), dtype=torch.long)
    out = []
    with torch.no_grad():
        for i in range(0, len(f), 2048):
            logits = model(a[i:i + 2048], f[i:i + 2048])
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def metrics_for(bg_df, probs):
    df = bg_df.copy()
    df["score_ens"] = 1.0 - probs
    return MET.compute_metrics(df, "score_ens", "y_3600", r_values=R_SWEEP)


def train_seed(grid, d_tr, d_va, bg_val, vocab_size, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pos = float(d_tr["y"].mean())
    pw = torch.tensor([(1.0 - pos) / max(pos, 1e-6)])
    nf = int(d_tr["features"].shape[1])
    model = grid.make_baseline_model(num_features=nf, vocab_size=vocab_size, dropout=0.3)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    swa = torch.optim.swa_utils.AveragedModel(model)
    dl = DataLoader(PairDS(d_tr), batch_size=BATCH, shuffle=True)
    best_pr = -1.0
    best_state = None
    pat = 0
    swa_start = int(EPOCHS * 0.7)
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in dl:
            logit = model(b["app_idx"], b["features"])
            y = b["y"].float() * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
            loss = bce(logit, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep >= swa_start:
            swa.update_parameters(model)
        eval_model = swa.module if ep >= swa_start else model
        v_probs = forward_probs(eval_model, d_va)
        m = metrics_for(bg_val, v_probs)
        pr = float(m["pr_auc_mean"])
        if pr > best_pr + 1e-6:
            best_pr = pr
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    return best_state, best_pr


def main():
    grid = load_grid()
    print("[37] building features ...")
    ctx = grid.build_ctx()
    vocab = ctx["vocab"]
    bg_train = ctx["bg_train"]
    bg_val = ctx["bg_val"]
    bg_test = ctx["bg_test"]
    schema = sys.argv[1] if len(sys.argv) > 1 else "c3.3"
    d_tr = grid.build_schema_data(schema, bg_train, ctx)
    d_va = grid.build_schema_data(schema, bg_val, ctx)
    d_te = grid.build_schema_data(schema, bg_test, ctx)
    print(f"[37] schema={schema} feature_dim={d_tr['features'].shape[1]}")
    nf = int(d_tr["features"].shape[1])
    vocab_size = len(vocab)

    val_probs_list = []
    test_probs_list = []
    per_seed = []
    ckpt_dir = ROOT / "artifacts" / "bg" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        best_state, best_pr = train_seed(grid, d_tr, d_va, bg_val, vocab_size, seed)
        # Reload model and evaluate
        model = grid.make_baseline_model(num_features=nf, vocab_size=vocab_size, dropout=0.3)
        model.load_state_dict(best_state, strict=False)
        v_p = forward_probs(model, d_va)
        t_p = forward_probs(model, d_te)
        val_probs_list.append(v_p)
        test_probs_list.append(t_p)
        torch.save({"state_dict": best_state, "seed": seed},
                   ckpt_dir / f"task_c_c3pro_ens_s{seed}.pt")
        per_seed.append({
            "seed": seed,
            "best_val_pr": best_pr,
            "val": metrics_for(bg_val, v_p),
            "test": metrics_for(bg_test, t_p),
        })
        m = per_seed[-1]
        print(f"  seed {seed}: val_PR={m['val']['pr_auc_mean']:.4f}  test_PR={m['test']['pr_auc_mean']:.4f}  "
              f"test_FK@.5={m['test']['false_kill_rate']['0.5']:.4f}")

    val_probs = np.mean(np.stack(val_probs_list), axis=0)
    test_probs = np.mean(np.stack(test_probs_list), axis=0)
    val_m = metrics_for(bg_val, val_probs)
    test_m = metrics_for(bg_test, test_probs)
    print(f"\n[37] ENSEMBLE val:  PR={val_m['pr_auc_mean']:.4f} ROC={val_m['roc_auc_mean']:.4f} FK@.5={val_m['false_kill_rate']['0.5']:.4f}")
    print(f"[37] ENSEMBLE test: PR={test_m['pr_auc_mean']:.4f} ROC={test_m['roc_auc_mean']:.4f} FK@.5={test_m['false_kill_rate']['0.5']:.4f}")

    out_path = ROOT / "artifacts" / "bg" / "results" / "task_c_c3pro_ens.json"
    with open(out_path, "w") as f:
        json.dump({
            "tag": "c3pro_ens",
            "seeds": SEEDS,
            "per_seed": per_seed,
            "val": val_m,
            "test": test_m,
        }, f, indent=2)
    print(f"[37] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
