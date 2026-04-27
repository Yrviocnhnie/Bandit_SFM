"""Train GRU-64 with dual heads (Task A softmax + Task B sigmoid/Poisson)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import data as D
from lib import features as F
from lib import models as MOD
from lib import train as TR


HISTORY_K = 16
WINDOW_HORIZON_SEC = 900
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 25
PATIENCE = 5
SEED = 7
TASK_B_BCE_WEIGHT = 0.5
TASK_B_POISSON_WEIGHT = 0.2
LABEL_SMOOTHING = 0.05


def build_datasets(vocab_path, splits_dir):
    with open(vocab_path) as f:
        vocab = json.load(f)
    V = len(vocab)
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    scaler = F.fit_dt_scaler(train_df)

    horizon_ns = int(WINDOW_HORIZON_SEC * 1e9)

    def _prep(df_part):
        enc = F.encode_events(df_part, vocab, scaler["mean"], scaler["std"])
        hist = F.build_history_for_targets(enc, history_k=HISTORY_K)
        tgt_ts = enc.ts[enc.is_target].astype("datetime64[ns]").astype(np.int64)
        tgt_app = enc.app_idx[enc.is_target]
        win = TR.build_window_target_counts(tgt_ts, tgt_app, horizon_ns, V)
        return hist, win

    vocab = vocab
    return vocab, V, scaler, train_df, val_df, test_df


def main(model_kind="gru", n_epochs=MAX_EPOCHS):
    art = ROOT / "artifacts"
    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    V = len(vocab)

    train_df = pd.read_parquet(art / "splits" / "train.parquet")
    val_df = pd.read_parquet(art / "splits" / "val.parquet")
    scaler = F.fit_dt_scaler(train_df)

    horizon_ns = int(WINDOW_HORIZON_SEC * 1_000_000_000)

    def _build(df_part):
        enc = F.encode_events(df_part, vocab, scaler["mean"], scaler["std"])
        hist = F.build_history_for_targets(enc, history_k=HISTORY_K)
        t_ts = enc.ts[enc.is_target].astype("datetime64[ns]").astype(np.int64)
        t_app = enc.app_idx[enc.is_target]
        win = TR.build_window_target_counts(t_ts, t_app, horizon_ns, len(vocab))
        return hist, win

    hist_tr, win_tr = _build(train_df)
    hist_va, win_va = _build(pd.read_parquet(art / "splits" / "val.parquet"))

    ds_tr = TR.TargetWindowDataset(hist_tr, win_tr)
    ds_va = TR.TargetWindowDataset(hist_va, win_va)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    cw = TR.class_weights(hist_tr["target_app"].astype(np.int64), V)
    cw_t = torch.tensor(cw)

    torch.manual_seed(SEED)
    cfg = MOD.ModelConfig(vocab_size=V, numeric_dim=F.NUMERIC_FEAT_DIM)
    if model_kind == "gru":
        model = MOD.GRUModel(cfg)
    elif model_kind == "tgt":
        model = MOD.TGTLite(cfg)
    else:
        raise ValueError("unknown model kind: " + model_kind)
    device = torch.device("cpu")
    model = model.to(device)
    print(f"[train] model={model_kind} params={sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_hit5 = -1.0
    best_state = None
    patience_left = PATIENCE
    log = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        steps = 0
        for batch in dl_tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["history_app"], batch["history_feat"], batch["history_mask"], batch["side_hour_fourier"])
            L_A = Fn.cross_entropy(out["logits_a"], batch["target_app"], weight=cw_t, label_smoothing=0.05)
            y_bin = (batch["win_counts"] > 0).float()
            L_bce = Fn.binary_cross_entropy_with_logits(out["logits_b_sig"], y_bin)
            log_rate = out["log_rate_b"].clamp(max=8.0)
            rate = torch.exp(log_rate)
            L_pois = (rate - batch["win_counts"] * log_rate).mean()
            loss = L_A + 0.5 * L_bce + 0.2 * L_pois
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            steps += 1

        # Eval Task A
        model.eval()
        scores_all = []
        targets_all = []
        with torch.no_grad():
            for batch in dl_va:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["history_app"], batch["history_feat"], batch["history_mask"], batch["side_hour_fourier"])
                scores_all.append(out["logits_a"].softmax(dim=-1).cpu().numpy())
                targets_all.append(batch["target_app"].cpu().numpy())
        S = np.concatenate(scores_all, axis=0)
        T = np.concatenate(targets_all, axis=0)
        h1 = float((S.argmax(axis=1) == T).mean())
        order = np.argsort(-S, axis=1)
        h5 = float((order[:, :5] == T[:, None]).any(axis=1).mean())
        ranks = np.array([int(np.where(order[i] == T[i])[0][0]) for i in range(len(T))])
        mrr = float((1.0 / (ranks + 1.0)).mean())

        dt = time.time() - t0
        print(f"  ep{epoch:02d}  loss={loss_sum/max(steps,1):.4f}  val hit@1={h1:.3f}  hit@5={h5:.3f}  mrr={mrr:.3f}  ({dt:.1f}s)")
        log.append({"epoch": epoch, "loss": loss_sum / max(steps, 1), "val_hit@1": h1, "val_hit@5": h5, "val_mrr": mrr})

        if h5 > best_hit5:
            best_hit5 = h5
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("  early stop")
                break

    ckpt_path = art / "checkpoints" / "gru.pt" if model_kind == "gru" else art / "checkpoints" / "tgt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "cfg": vars(MOD.ModelConfig(V, F.NUMERIC_FEAT_DIM))}, ckpt_path)
    summary_path = ROOT / "artifacts" / "results" / (model_kind + "_train.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fp:
        json.dump({"best_val_hit@5": best_hit5, "log": log}, fp, indent=2)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gru", choices=["gru", "tgt"])
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    sys.exit(main(args.model, args.epochs) or 0)
