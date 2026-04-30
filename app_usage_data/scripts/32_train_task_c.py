"""Train + evaluate the per-pair MLP for Task C.

Two model schemas supported:
  * --schema v1   (default; legacy C1: cat_emb + 14 features, mirrors original)
  * --schema v2   (new C2: 15 Task-C-relevant features, no cat_emb,
                   optional app_emb via --no-app-emb)

Loss: dual-horizon BCE with pos_weight from train. Selection on val PR-AUC(H=5).

Outputs:
  artifacts/bg/checkpoints/task_c_<tag>.pt
  artifacts/bg/results/task_c_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.bg import metrics_bg as MET
from lib.bg.models_bg import (
    BgMLPConfig, BgPairMLP, FEATURE_NAMES, bce_dual,
    BgMLPv2Config, BgPairMLPv2, FEATURE_NAMES_V2,
)
from lib.v3 import categories as CAT
from lib.v3 import markov_prior as MK


SEED = 7
BATCH = 512
LR = 1e-3
WD = 1e-4
EPOCHS_DEFAULT = 30
PATIENCE = 5
R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)


def fit_hour_freq(train_df: pd.DataFrame, vocab: dict) -> np.ndarray:
    V = len(vocab)
    rare = vocab.get("<RARE>", 2)
    table = np.zeros((24, V), dtype=np.float64)
    mask = train_df["is_target_event"].astype(bool).to_numpy()
    hrs = pd.to_datetime(train_df.loc[mask, "event_ts"]).dt.hour.to_numpy()
    apps = train_df.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    for h, a in zip(hrs, idx):
        if 0 <= h < 24 and 3 <= a < V:
            table[int(h), int(a)] += 1.0
    table += 0.5
    table[:, :3] = 0.0
    s = table.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (table / s).astype(np.float32)


def build_features_v1(df, hour_freq, markov_probs, app_to_cat):
    """Legacy 14-feature build for C1 reproducibility."""
    n = len(df)
    tib = np.clip(df["time_in_bg_sec"].to_numpy(dtype=np.float64), 0, None)
    tsf = np.clip(df["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
    fg_ct = df["fg_count_today"].to_numpy(dtype=np.float64)
    bg_sz = df["bg_set_size"].to_numpy(dtype=np.float64)
    h = df["anchor_hour"].to_numpy(dtype=np.float64)
    w = df["anchor_weekday"].to_numpy(dtype=np.float64)
    dp_anchor = df["anchor_daypart"].to_numpy(dtype=np.int64)
    dp_last = df["last_fg_daypart"].to_numpy(dtype=np.int64)
    loc = df["last_fg_loc_id"].to_numpy(dtype=np.int64)
    last = np.clip(df["last_fg_app_idx"].to_numpy(dtype=np.int64), 0, markov_probs.shape[0] - 1)
    app = np.clip(df["app_idx"].to_numpy(dtype=np.int64), 0, markov_probs.shape[1] - 1)
    markov_col = markov_probs[last, app].astype(np.float32)
    hr_idx = np.clip(h.astype(np.int64) % 24, 0, 23)
    hour_col = hour_freq[hr_idx, app].astype(np.float32)

    cols = [
        np.log1p(tib),
        np.log1p(tsf),
        np.log1p(fg_ct),
        np.clip(tib / (6 * 3600), 0, 2),
        np.sin(2 * np.pi * h / 24.0),
        np.cos(2 * np.pi * h / 24.0),
        np.sin(2 * np.pi * w / 7.0),
        np.cos(2 * np.pi * w / 7.0),
        np.log1p(bg_sz),
        (w >= 5).astype(np.float64),
        (loc > 0).astype(np.float64),
        (dp_anchor == dp_last).astype(np.float64),
        markov_col.astype(np.float64),
        hour_col.astype(np.float64),
    ]
    feats = np.stack(cols, axis=1).astype(np.float32)
    app_idx = df["app_idx"].to_numpy(dtype=np.int64)
    cat_idx = app_to_cat[np.clip(app_idx, 0, len(app_to_cat) - 1)].astype(np.int64)
    return {
        "features": feats,
        "app_idx": app_idx,
        "cat_idx": cat_idx,
        "y_5": df["y_300"].to_numpy(dtype=np.int64),
        "y_10": df["y_600"].to_numpy(dtype=np.int64),
        "anchor_id": df["anchor_id"].to_numpy(dtype=np.int64),
    }


def build_features_v2(df, hour_freq, markov_probs):
    """Task-C-relevant 15-feature build (C2). See REPORT_bgkill_v2.md §4."""
    tib = np.clip(df["time_in_bg_sec"].to_numpy(dtype=np.float64), 0, None)
    tsf = np.clip(df["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
    fg_today = df["fg_count_today"].to_numpy(dtype=np.float64)
    fg_1h = df["fg_count_last_3600s"].to_numpy(dtype=np.float64)
    fg_6h = df["fg_count_last_21600s"].to_numpy(dtype=np.float64)
    rec_rank = df["recency_rank_in_bg"].to_numpy(dtype=np.float32)
    h = df["anchor_hour"].to_numpy(dtype=np.float64)
    w = df["anchor_weekday"].to_numpy(dtype=np.float64)
    dp_anchor = df["anchor_daypart"].to_numpy(dtype=np.int64)
    dp_last = df["last_fg_daypart"].to_numpy(dtype=np.int64)

    last = np.clip(df["last_fg_app_idx"].to_numpy(dtype=np.int64),
                   0, markov_probs.shape[0] - 1)
    app = np.clip(df["app_idx"].to_numpy(dtype=np.int64),
                  0, markov_probs.shape[1] - 1)
    markov_col = markov_probs[last, app].astype(np.float32)
    hr_idx = np.clip(h.astype(np.int64) % 24, 0, 23)
    hour_col = hour_freq[hr_idx, app].astype(np.float32)

    # bg_recency_min normalized by log1p(6h) → value in [0, 1]
    bg_rec_min = np.clip(df["bg_recency_min_sec"].to_numpy(dtype=np.float64), 0, None)
    bg_rec_min_norm = (np.log1p(bg_rec_min) / np.log1p(6 * 3600.0)).astype(np.float32)

    # time_since_screen_on_sec: clip to 1h, normalize by log1p(1h)
    tso = df["time_since_screen_on_sec"].to_numpy(dtype=np.float64)
    tso_clip = np.where(tso < 0, 3600.0, np.clip(tso, 0, 3600.0))
    tso_norm = (np.log1p(tso_clip) / np.log1p(3600.0)).astype(np.float32)

    # prev_killed_app_match: 1 if this row's app is the most recently OS-killed
    # (within the 10-min window already enforced by the state machine).
    pk_idx = df["prev_killed_app_idx"].to_numpy(dtype=np.int64)
    pk_age = df["prev_killed_age_sec"].to_numpy(dtype=np.float64)
    prev_kill_match = np.where((pk_idx > 0) & (pk_age >= 0) &
                                (df["app_idx"].to_numpy(dtype=np.int64) == pk_idx),
                                1.0, 0.0).astype(np.float32)

    cols = [
        np.log1p(tib),                               # log_time_in_bg
        np.log1p(tsf),                               # log_time_since_fg
        rec_rank.astype(np.float64),                 # recency_rank_in_bg
        np.log1p(fg_today),                          # log_fg_count_today
        np.log1p(fg_1h),                             # log_fg_count_last_1h
        np.log1p(fg_6h),                             # log_fg_count_last_6h
        markov_col.astype(np.float64),               # markov_prob
        hour_col.astype(np.float64),                 # hour_cond_prob
        bg_rec_min_norm.astype(np.float64),          # bg_recency_min_norm
        tso_norm.astype(np.float64),                 # time_since_screen_on_norm
        prev_kill_match.astype(np.float64),          # prev_killed_app_match
        np.sin(2 * np.pi * h / 24.0),                # hour_sin_24
        np.cos(2 * np.pi * h / 24.0),                # hour_cos_24
        (dp_anchor == dp_last).astype(np.float64),   # daypart_match_flag
        (w >= 5).astype(np.float64),                 # is_weekend
    ]
    feats = np.stack(cols, axis=1).astype(np.float32)
    app_idx = df["app_idx"].to_numpy(dtype=np.int64)
    return {
        "features": feats,
        "app_idx": app_idx,
        "y_5": df["y_300"].to_numpy(dtype=np.int64),
        "y_10": df["y_600"].to_numpy(dtype=np.int64),
        "anchor_id": df["anchor_id"].to_numpy(dtype=np.int64),
    }


class PairDS(Dataset):
    def __init__(self, d, schema):
        self.f = torch.as_tensor(d["features"].copy(), dtype=torch.float32)
        self.ai = torch.as_tensor(d["app_idx"].copy(), dtype=torch.long)
        self.y5 = torch.as_tensor(d["y_5"].copy(), dtype=torch.float32)
        self.y10 = torch.as_tensor(d["y_10"].copy(), dtype=torch.float32)
        self.schema = schema
        if schema == "v1":
            self.ci = torch.as_tensor(d["cat_idx"].copy(), dtype=torch.long)
        else:
            self.ci = None

    def __len__(self):
        return int(self.f.shape[0])

    def __getitem__(self, i):
        item = {
            "features": self.f[i],
            "app_idx": self.ai[i],
            "y_5": self.y5[i],
            "y_10": self.y10[i],
        }
        if self.ci is not None:
            item["cat_idx"] = self.ci[i]
        return item


def evaluate(model, dataset_df, split_d, schema):
    model.eval()
    loader = DataLoader(PairDS(split_d, schema=schema), batch_size=1024, shuffle=False)
    p5_list, p10_list = [], []
    with torch.no_grad():
        for b in loader:
            if schema == "v1":
                out = model(b["app_idx"], b["cat_idx"], b["features"])
            else:
                out = model(b["app_idx"], b["features"])
            p5_list.append(torch.sigmoid(out["logit_5"]).cpu().numpy())
            p10_list.append(torch.sigmoid(out["logit_10"]).cpu().numpy())
    p5 = np.concatenate(p5_list)
    p10 = np.concatenate(p10_list)
    merged = dataset_df.copy()
    merged["score_model_5"] = 1.0 - p5
    merged["score_model_10"] = 1.0 - p10
    return {
        "H_300": MET.compute_metrics(merged, "score_model_5", "y_300", r_values=R_SWEEP),
        "H_600": MET.compute_metrics(merged, "score_model_10", "y_600", r_values=R_SWEEP),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", choices=["v1", "v2"], default="v2",
                        help="v1=legacy C1 (14 features + cat_emb); v2=new C2 (15 features, no cat_emb)")
    parser.add_argument("--no-app-emb", dest="use_app_emb", action="store_false",
                        help="(v2 only) disable the 16-d app embedding")
    parser.set_defaults(use_app_emb=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--d-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    if args.tag is None:
        args.tag = "C2" if args.schema == "v2" else "C1_repro"
        if args.schema == "v2" and not args.use_app_emb:
            args.tag = "C2_noemb"

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    art = ROOT / "artifacts"
    bg_art = art / "bg"
    with open(art / "vocab.json") as f:
        vocab = json.load(f)

    train_raw = pd.read_parquet(art / "splits" / "train.parquet")
    hour_freq = fit_hour_freq(train_raw, vocab)
    mk_stats = MK.load_markov_prior(art / "v3" / "markov_prior.pkl")
    markov_probs = mk_stats["probs"].astype(np.float32)

    bg_train = pd.read_parquet(bg_art / "splits" / "bg_train.parquet")
    bg_val = pd.read_parquet(bg_art / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_art / "splits" / "bg_test.parquet")

    if args.schema == "v1":
        app_to_cat = CAT.build_app_to_cat_idx(vocab)
        build = lambda df: build_features_v1(df, hour_freq, markov_probs, app_to_cat)
    else:
        build = lambda df: build_features_v2(df, hour_freq, markov_probs)

    d_tr, d_va, d_te = build(bg_train), build(bg_val), build(bg_test)

    p_tr_5 = float(d_tr["y_5"].mean())
    p_tr_10 = float(d_tr["y_10"].mean())
    pw5 = torch.tensor([(1 - p_tr_5) / max(1e-6, p_tr_5)])
    pw10 = torch.tensor([(1 - p_tr_10) / max(1e-6, p_tr_10)])
    print(f"[task_c] schema={args.schema}  use_app_emb={args.use_app_emb}  "
          f"pos_rate_5={p_tr_5:.4f}  pos_weight_5={pw5.item():.2f}")

    if args.schema == "v1":
        cfg = BgMLPConfig(d_hidden=args.d_hidden)
        model = BgPairMLP(cfg)
    else:
        cfg = BgMLPv2Config(d_hidden=args.d_hidden, use_app_emb=args.use_app_emb)
        model = BgPairMLPv2(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[task_c] params={n_params}  feature_dim={d_tr['features'].shape[1]}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WD)
    dl_tr = DataLoader(PairDS(d_tr, schema=args.schema), batch_size=BATCH, shuffle=True)

    best_pr5 = -1.0
    best_state = None
    patience = 0
    log = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in dl_tr:
            if args.schema == "v1":
                out = model(b["app_idx"], b["cat_idx"], b["features"])
            else:
                out = model(b["app_idx"], b["features"])
            loss, _, _ = bce_dual(out, b["y_5"], b["y_10"],
                                   pos_weight_5=pw5, pos_weight_10=pw10)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()) * len(b["y_5"])
            n += len(b["y_5"])

        val_metrics = evaluate(model, bg_val, d_va, schema=args.schema)
        pr5 = val_metrics["H_300"]["pr_auc_mean"]
        roc5 = val_metrics["H_300"]["roc_auc_mean"]
        fk5 = val_metrics["H_300"]["false_kill_rate"]["0.5"]
        log.append({"epoch": epoch, "train_loss": tot / max(1, n),
                    "val_pr_auc_5": pr5, "val_roc_auc_5": roc5, "val_fk5_at_half": fk5})
        print(f"  ep{epoch:02d}  loss={tot/max(1,n):.4f}  val_PR@H5={pr5:.4f}  "
              f"val_ROC@H5={roc5:.4f}  val_FK@0.5={fk5:.4f}")
        if pr5 > best_pr5 + 1e-6:
            best_pr5 = pr5
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_m = evaluate(model, bg_val, d_va, schema=args.schema)
    test_m = evaluate(model, bg_test, d_te, schema=args.schema)

    out_ckpt = bg_art / "checkpoints" / f"task_c_{args.tag}.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "schema": args.schema,
                "use_app_emb": args.use_app_emb,
                "feature_names": (FEATURE_NAMES_V2 if args.schema == "v2" else FEATURE_NAMES)},
               out_ckpt)
    out_path = bg_art / "results" / f"task_c_{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"tag": args.tag, "schema": args.schema,
                   "use_app_emb": bool(args.use_app_emb),
                   "n_params": n_params,
                   "feature_dim": int(d_tr["features"].shape[1]),
                   "train_log": log,
                   "val": val_m, "test": test_m,
                   "r_sweep": list(R_SWEEP)}, f, indent=2)
    print(f"[task_c] saved {out_ckpt}  results -> {out_path}  elapsed {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
