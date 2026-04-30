"""Single-horizon (H=60 min) Task C training, three feature schemas.

Schemas:
    c1     14 legacy features + 16-d app_emb + 4-d cat_emb
    c2     15 v2 features + 16-d app_emb (no cat_emb)
    c3.1   c2 + 5 hourly-habit features (overdue, periodicity, counts)

Per-app inter-FG mean is fit on TRAIN events only (fit_split="train"). All
periodicity / count lookups use the full FG event timeline; this is causal
because every searchsorted query is strictly < the row's anchor_ts_ns.

Loss: BCE-with-logits + pos_weight on y_3600. Selection: best val PR-AUC.

Outputs:
    artifacts/bg/checkpoints/task_c_<tag>_h60.pt
    artifacts/bg/results/task_c_<tag>_h60.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.bg import metrics_bg as MET
from lib.bg.models_bg import FEATURE_NAMES, FEATURE_NAMES_V2
from lib.v3 import categories as CAT
from lib.v3 import markov_prior as MK


SEED = 7
LR = 1e-3
WD = 1e-4
BATCH = 512
EPOCHS_DEFAULT = 30
PATIENCE = 5
R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)
H_SEC = 3600
Y_COL = f"y_{H_SEC}"
NS = 1_000_000_000
NS_24H = 24 * 3600 * NS
NS_7D = 7 * NS_24H

C31_NAMES = (
    "overdue_ratio",
    "was_fg_24h_ago",
    "was_fg_7d_ago",
    "log_fg_count_last_24h",
    "log_fg_count_last_7d",
)


# ============================================================================
# Train-only stat fitters
# ============================================================================
def fit_hour_freq(train_events, vocab):
    V = len(vocab)
    rare = vocab.get("<RARE>", 2)
    table = np.zeros((24, V), dtype=np.float64)
    mask = train_events["is_target_event"].astype(bool).to_numpy()
    hrs = pd.to_datetime(train_events.loc[mask, "event_ts"]).dt.hour.to_numpy()
    apps = train_events.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    for h, a in zip(hrs, idx):
        if 0 <= h < 24 and 3 <= a < V:
            table[int(h), int(a)] += 1.0
    table += 0.5
    table[:, :3] = 0.0
    s = table.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (table / s).astype(np.float32)


def fit_per_app_inter_fg_mean(train_events, vocab):
    V = len(vocab)
    rare = vocab.get("<RARE>", 2)
    out = np.full(V, 24.0 * 3600.0, dtype=np.float64)
    mask = train_events["is_target_event"].astype(bool).to_numpy()
    ts_ns = (
        pd.to_datetime(train_events.loc[mask, "event_ts"])
        .to_numpy()
        .astype("datetime64[ns]")
        .view("int64")
    )
    apps = train_events.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    for a in np.unique(idx):
        if a < 3:
            continue
        ts_a = np.sort(ts_ns[idx == a])
        if len(ts_a) >= 2:
            out[a] = float(np.mean(np.diff(ts_a) / 1e9))
    return out.astype(np.float32)


def collect_fg_timeline(events_concat, vocab):
    rare = vocab.get("<RARE>", 2)
    fg_mask = events_concat["name_norm"].isin(["APP_FOREGROUND", "APP_START"])
    ts_ns = (
        pd.to_datetime(events_concat.loc[fg_mask, "event_ts"])
        .to_numpy()
        .astype("datetime64[ns]")
        .view("int64")
    )
    apps = events_concat.loc[fg_mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    out = {}
    for a in np.unique(idx):
        out[int(a)] = np.sort(ts_ns[idx == a])
    return out


# ============================================================================
# Feature builders (per schema)
# ============================================================================
def build_c1(bg_df, hour_freq, markov_probs, app_to_cat):
    tib = np.clip(bg_df["time_in_bg_sec"].to_numpy(dtype=np.float64), 0, None)
    tsf = np.clip(bg_df["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
    fg_ct = bg_df["fg_count_today"].to_numpy(dtype=np.float64)
    bg_sz = bg_df["bg_set_size"].to_numpy(dtype=np.float64)
    h = bg_df["anchor_hour"].to_numpy(dtype=np.float64)
    w = bg_df["anchor_weekday"].to_numpy(dtype=np.float64)
    dp_anchor = bg_df["anchor_daypart"].to_numpy(dtype=np.int64)
    dp_last = bg_df["last_fg_daypart"].to_numpy(dtype=np.int64)
    loc = bg_df["last_fg_loc_id"].to_numpy(dtype=np.int64)

    last = np.clip(bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64), 0, markov_probs.shape[0] - 1)
    app = np.clip(bg_df["app_idx"].to_numpy(dtype=np.int64), 0, markov_probs.shape[1] - 1)
    markov_col = markov_probs[last, app].astype(np.float64)
    hr_idx = np.clip(h.astype(np.int64) % 24, 0, 23)
    hour_col = hour_freq[hr_idx, app].astype(np.float64)

    cols = [
        np.log1p(tib),
        np.log1p(tsf),
        np.log1p(fg_ct),
        np.clip(tib / (6 * 3600.0), 0, 2.0),
        np.sin(2 * np.pi * h / 24.0),
        np.cos(2 * np.pi * h / 24.0),
        np.sin(2 * np.pi * w / 7.0),
        np.cos(2 * np.pi * w / 7.0),
        np.log1p(bg_sz),
        (w >= 5).astype(np.float64),
        (loc > 0).astype(np.float64),
        (dp_anchor == dp_last).astype(np.float64),
        markov_col,
        hour_col,
    ]
    feats = np.stack(cols, axis=1).astype(np.float32)
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    cat_idx = app_to_cat[np.clip(a_idx, 0, len(app_to_cat) - 1)].astype(np.int64)
    return {
        "features": feats,
        "app_idx": a_idx,
        "cat_idx": cat_idx,
        "y": bg_df[Y_COL].to_numpy(dtype=np.int64),
        "anchor_id": bg_df["anchor_id"].to_numpy(dtype=np.int64),
    }


def build_c2(bg_df, hour_freq, markov_probs):
    tib = np.clip(bg_df["time_in_bg_sec"].to_numpy(dtype=np.float64), 0, None)
    tsf = np.clip(bg_df["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
    fg_today = bg_df["fg_count_today"].to_numpy(dtype=np.float64)
    fg_1h = bg_df["fg_count_last_3600s"].to_numpy(dtype=np.float64)
    fg_6h = bg_df["fg_count_last_21600s"].to_numpy(dtype=np.float64)
    rec_rank = bg_df["recency_rank_in_bg"].to_numpy(dtype=np.float64)
    h = bg_df["anchor_hour"].to_numpy(dtype=np.float64)
    w = bg_df["anchor_weekday"].to_numpy(dtype=np.float64)
    dp_anchor = bg_df["anchor_daypart"].to_numpy(dtype=np.int64)
    dp_last = bg_df["last_fg_daypart"].to_numpy(dtype=np.int64)

    last = np.clip(bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64),
                   0, markov_probs.shape[0] - 1)
    app = np.clip(bg_df["app_idx"].to_numpy(dtype=np.int64),
                  0, markov_probs.shape[1] - 1)
    markov_col = markov_probs[last, app].astype(np.float64)
    hr_idx = np.clip(h.astype(np.int64) % 24, 0, 23)
    hour_col = hour_freq[hr_idx, app].astype(np.float64)

    bg_rec_min = np.clip(bg_df["bg_recency_min_sec"].to_numpy(dtype=np.float64), 0, None)
    bg_rec_norm = np.log1p(bg_rec_min) / np.log1p(6 * 3600.0)

    tso = bg_df["time_since_screen_on_sec"].to_numpy(dtype=np.float64)
    tso_clip = np.where(tso < 0, 3600.0, np.clip(tso, 0, 3600.0))
    tso_norm = np.log1p(tso_clip) / np.log1p(3600.0)

    pk_idx = bg_df["prev_killed_app_idx"].to_numpy(dtype=np.int64)
    pk_age = bg_df["prev_killed_age_sec"].to_numpy(dtype=np.float64)
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    pk_match = ((pk_idx == a_idx) & (pk_idx > 0) & (pk_age >= 0)).astype(np.float64)

    cols = [
        np.log1p(tib),
        np.log1p(tsf),
        rec_rank,
        np.log1p(fg_today),
        np.log1p(fg_1h),
        np.log1p(fg_6h),
        markov_col,
        hour_col,
        bg_rec_norm,
        tso_norm,
        pk_match,
        np.sin(2 * np.pi * h / 24.0),
        np.cos(2 * np.pi * h / 24.0),
        (dp_anchor == dp_last).astype(np.float64),
        (w >= 5).astype(np.float64),
    ]
    feats = np.stack(cols, axis=1).astype(np.float32)
    return {
        "features": feats,
        "app_idx": a_idx,
        "cat_idx": None,
        "y": bg_df[Y_COL].to_numpy(dtype=np.int64),
        "anchor_id": bg_df["anchor_id"].to_numpy(dtype=np.int64),
    }


def compute_c31_arrays(bg_df, fg_timeline, per_app_inter_mean, half_window_sec=1800):
    """Five hourly-habit arrays of length len(bg_df)."""
    n = len(bg_df)
    half_ns = int(half_window_sec) * NS
    ts = bg_df["anchor_ts_ns"].to_numpy(dtype=np.int64)
    aix = bg_df["app_idx"].to_numpy(dtype=np.int64)
    tib = bg_df["time_in_bg_sec"].to_numpy(dtype=np.float64)

    V = per_app_inter_mean.shape[0]
    a_clip = np.clip(aix, 0, V - 1)
    means = np.clip(per_app_inter_mean[a_clip], 1.0, None)
    overdue = np.clip(tib / means, 0.0, 5.0).astype(np.float32)

    fg_24 = np.zeros(n, dtype=np.float32)
    fg_7d = np.zeros(n, dtype=np.float32)
    cnt_24 = np.zeros(n, dtype=np.float32)
    cnt_7d = np.zeros(n, dtype=np.float32)

    for i in range(n):
        a = int(aix[i])
        t = int(ts[i])
        arr = fg_timeline.get(a)
        if arr is None or len(arr) == 0:
            continue

        c24 = t - NS_24H
        l = int(np.searchsorted(arr, c24 - half_ns, side="left"))
        r = int(np.searchsorted(arr, c24 + half_ns, side="right"))
        if r > l:
            fg_24[i] = 1.0

        c7 = t - NS_7D
        l = int(np.searchsorted(arr, c7 - half_ns, side="left"))
        r = int(np.searchsorted(arr, c7 + half_ns, side="right"))
        if r > l:
            fg_7d[i] = 1.0

        l24 = int(np.searchsorted(arr, t - NS_24H, side="left"))
        r24 = int(np.searchsorted(arr, t, side="left"))
        cnt_24[i] = float(max(0, r24 - l24))

        l7 = int(np.searchsorted(arr, t - NS_7D, side="left"))
        r7 = int(np.searchsorted(arr, t, side="left"))
        cnt_7d[i] = float(max(0, r7 - l7))

    return {
        "overdue_ratio": overdue,
        "was_fg_24h_ago": fg_24,
        "was_fg_7d_ago": fg_7d,
        "log_fg_count_last_24h": np.log1p(cnt_24).astype(np.float32),
        "log_fg_count_last_7d": np.log1p(cnt_7d).astype(np.float32),
    }


def build_c31(bg_df, hour_freq, markov_probs, fg_timeline, per_app_inter_mean,
              drop_feature: str | None = None):
    """C3.1 features = c2 + 5 hourly habit. drop_feature ∈ C31_NAMES zeros one column."""
    base = build_c2(bg_df, hour_freq, markov_probs)
    extras = compute_c31_arrays(bg_df, fg_timeline, per_app_inter_mean)
    cols = []
    for name in C31_NAMES:
        col = extras[name]
        if drop_feature is not None and name == drop_feature:
            col = np.zeros_like(col)
        cols.append(col)
    extra_arr = np.stack(cols, axis=1).astype(np.float32)
    base["features"] = np.concatenate([base["features"], extra_arr], axis=1)
    return base


C31_NAMES = (
    "overdue_ratio",
    "was_fg_24h_ago",
    "was_fg_7d_ago",
    "log_fg_count_last_24h",
    "log_fg_count_last_7d",
)


# ============================================================================
# Model: single-head MLP with optional cat_emb
# ============================================================================
@dataclass
class ModelCfg:
    vocab_size: int
    num_features: int
    use_cat_emb: bool = False
    num_categories: int = 11
    app_emb_dim: int = 16
    cat_emb_dim: int = 4
    d_hidden: int = 64
    dropout: float = 0.2


class SingleHeadMLP(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        if cfg.use_cat_emb:
            self.cat_emb = nn.Embedding(cfg.num_categories, cfg.cat_emb_dim, padding_idx=0)
        else:
            self.cat_emb = None
        in_dim = cfg.app_emb_dim + cfg.num_features
        if cfg.use_cat_emb:
            in_dim = in_dim + cfg.cat_emb_dim
        d_h = cfg.d_hidden
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, d_h),
            nn.ReLU(),
            nn.LayerNorm(d_h),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.LayerNorm(d_h // 2),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(d_h // 2, 1)

    def forward(self, app_idx, features, cat_idx=None):
        ae = self.app_emb(app_idx)
        parts = [ae, features]
        if self.cat_emb is not None and cat_idx is not None:
            ce = self.cat_emb(cat_idx)
            parts = [ae, ce, features]
        x = torch.cat(parts, dim=-1)
        h = self.trunk(x)
        return self.head(h).squeeze(-1)


def make_model(num_features, vocab_size, use_cat_emb, d_hidden=64, dropout=0.2):
    cfg = ModelCfg(
        vocab_size=vocab_size,
        num_features=num_features,
        d_hidden=d_hidden,
        dropout=dropout,
        use_cat_emb=use_cat_emb,
    )
    return SingleHeadMLP(cfg), cfg


# ============================================================================
# Dataset / training utilities
# ============================================================================
class PairDS(Dataset):
    def __init__(self, d):
        self.f = torch.as_tensor(d["features"].copy(), dtype=torch.float32)
        self.ai = torch.as_tensor(d["app_idx"].copy(), dtype=torch.long)
        self.y = torch.as_tensor(d["y"].copy(), dtype=torch.float32)
        ci = d.get("cat_idx")
        self.ci = torch.as_tensor(ci.copy(), dtype=torch.long) if ci is not None else None

    def __len__(self):
        return self.f.shape[0]

    def __getitem__(self, i):
        item = {"features": self.f[i], "app_idx": self.ai[i], "y": self.y[i]}
        if self.ci is not None:
            item["cat_idx"] = self.ci[i]
        return item


def forward_batch(model, batch):
    if "cat_idx" in batch:
        return model(batch["app_idx"], batch["features"], batch["cat_idx"])
    return model(batch["app_idx"], batch["features"])


def evaluate(model, bg_df, data_d):
    model.eval()
    loader = DataLoader(PairDS(data_d), batch_size=2048, shuffle=False)
    probs = []
    with torch.no_grad():
        for b in loader:
            logits = forward_batch(model, b)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(probs)
    merged = bg_df.copy()
    merged["score_model"] = 1.0 - p   # high score = more kill-worthy
    return MET.compute_metrics(merged, "score_model", Y_COL, r_values=R_SWEEP)


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", choices=["c1", "c2", "c3.1"], default="c2")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--d-hidden", type=int, default=64)
    ap.add_argument("--c31-drop", default=None,
                    help="(c3.1 only) name of one hourly-habit feature to zero out")
    args = ap.parse_args()

    if args.tag is None:
        args.tag = {"c1": "C1_h60", "c2": "C2_h60", "c3.1": "C3p1_h60"}[args.schema]

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    art = ROOT / "artifacts"
    bg_art = art / "bg"
    with open(art / "vocab.json") as f:
        vocab = json.load(f)

    train_events = pd.read_parquet(art / "splits" / "train.parquet")
    val_events = pd.read_parquet(art / "splits" / "val.parquet")
    test_events = pd.read_parquet(art / "splits" / "test.parquet")

    hour_freq = fit_hour_freq(train_events, vocab)
    markov_probs = MK.load_markov_prior(art / "v3" / "markov_prior.pkl")["probs"].astype(np.float32)

    bg_train = pd.read_parquet(bg_art / "splits" / "bg_train.parquet")
    bg_val = pd.read_parquet(bg_art / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_art / "splits" / "bg_test.parquet")

    if args.schema == "c1":
        app_to_cat = CAT.build_app_to_cat_idx(vocab)
        d_tr = build_c1(bg_train, hour_freq, markov_probs, app_to_cat)
        d_va = build_c1(bg_val, hour_freq, markov_probs, app_to_cat)
        d_te = build_c1(bg_test, hour_freq, markov_probs, app_to_cat)
        use_cat_emb = True
    elif args.schema == "c2":
        d_tr = build_c2(bg_train, hour_freq, markov_probs)
        d_va = build_c2(bg_val, hour_freq, markov_probs)
        d_te = build_c2(bg_test, hour_freq, markov_probs)
        use_cat_emb = False
    else:
        per_app_inter = fit_per_app_inter_fg_mean(train_events, vocab)
        all_events = pd.concat(
            [train_events, val_events, test_events], ignore_index=True
        ).sort_values("event_ts").reset_index(drop=True)
        fg_timeline = collect_fg_timeline(all_events, vocab)
        drop = getattr(args, "c31_drop", None)
        d_tr = build_c31(bg_train, hour_freq, markov_probs, fg_timeline, per_app_inter, drop)
        d_va = build_c31(bg_val, hour_freq, markov_probs, fg_timeline, per_app_inter, drop)
        d_te = build_c31(bg_test, hour_freq, markov_probs, fg_timeline, per_app_inter, drop)
        use_cat_emb = False

    pos = float(d_tr["y"].mean())
    pw = torch.tensor([(1.0 - pos) / max(1e-6, pos)])
    print(f"[task_c h60] schema={args.schema}  pos_rate={pos:.4f}  pos_weight={float(pw):.2f}")

    num_features = int(d_tr["features"].shape[1])
    model, cfg = make_model(
        num_features=num_features,
        vocab_size=len(vocab),
        use_cat_emb=use_cat_emb,
        d_hidden=args.d_hidden,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[task_c h60] params={n_params}  feature_dim={num_features}  use_cat_emb={use_cat_emb}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    dl_tr = DataLoader(PairDS(d_tr), batch_size=BATCH, shuffle=True)

    best_pr = -1.0
    best_state = None
    patience = 0
    log = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in dl_tr:
            logits = forward_batch(model, b)
            loss = bce(logits, b["y"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()) * len(b["y"])
            n += len(b["y"])
        val_m = evaluate(model, bg_val, d_va)
        cur_pr = float(val_m["pr_auc_mean"])
        log.append({
            "epoch": epoch,
            "train_loss": tot / max(1, n),
            "val_pr_auc": cur_pr,
            "val_roc_auc": float(val_m["roc_auc_mean"]),
            "val_fk_at_half": float(val_m["false_kill_rate"]["0.5"]),
        })
        print(f"  ep{epoch:02d}  loss={tot/max(1,n):.4f}  "
              f"val_PR={cur_pr:.4f}  "
              f"val_ROC={float(val_m['roc_auc_mean']):.4f}  "
              f"val_FK@.5={float(val_m['false_kill_rate']['0.5']):.4f}")
        if cur_pr > best_pr + 1e-6:
            best_pr = cur_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  early stop at ep{epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate(model, bg_val, d_va)
    final_test = evaluate(model, bg_test, d_te)

    out_ckpt = bg_art / "checkpoints" / f"task_c_{args.tag}.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "schema": args.schema,
        "feature_count": num_features,
        "use_cat_emb": use_cat_emb,
    }, out_ckpt)

    out_json = bg_art / "results" / f"task_c_{args.tag}.json"
    with open(out_json, "w") as f:
        json.dump({
            "tag": args.tag,
            "schema": args.schema,
            "feature_count": num_features,
            "use_cat_emb": use_cat_emb,
            "n_params": n_params,
            "pos_rate_train": float(pos),
            "train_log": log,
            "H_3600_val": final_val,
            "H_3600_test": final_test,
            "r_sweep": list(R_SWEEP),
            "best_val_pr_auc": float(best_pr),
        }, f, indent=2)
    print(f"[task_c h60] saved {out_ckpt}  results={out_json}  elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
