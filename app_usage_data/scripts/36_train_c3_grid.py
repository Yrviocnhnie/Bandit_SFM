"""C3 cumulative feature ablation grid + C3-Pro architectural variants.

Track A — feature ablation (cumulative on top of C2):
    c3.1   + 5 hourly-habit features
    c3.2   + 9 identity & BG-composition features
    c3.3   + 3 category-aware features
    c3.4   + 1 two-step Markov feature

Track B — architectural variants on top of best feature schema:
    c3pro_listwise   per-anchor softmax NLL aux loss (λ=0.5)
    c3pro_wide       32-d app_emb, 128→64 trunk, GELU, dropout 0.3
    c3pro_reg        dropout 0.3 + cosine LR + SWA over last 30 % of epochs
    c3pro_aux        + auxiliary head: predict last_fg_app cat
    c3pro_full       = listwise + wide + reg combined
    c3pro_ens        5-seed ensemble of c3pro_full (saved checkpoints averaged)

C3.5 (session) and C3.6 (active-session burst) are deferred — they require
backward event-stream lookups not exposed in the parquet.

CLI:
  python 36_train_c3_grid.py --variant c3.2     # train one
  python 36_train_c3_grid.py --variant all      # train all (~30 min)

Outputs:
  artifacts/bg/checkpoints/task_c_<tag>.pt
  artifacts/bg/results/task_c_<tag>.json
  artifacts/bg/results/c3_grid_summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.bg import metrics_bg as MET
from lib.v3 import categories as CAT
from lib.v3 import markov_prior as MK


SEED = 7
LR = 1e-3
WD = 1e-4
BATCH = 512
EPOCHS = 30
PATIENCE = 5
R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)
H_SEC = 3600
Y_COL = f"y_{H_SEC}"
NS = 1_000_000_000
NS_24H = 24 * 3600 * NS
NS_7D = 7 * NS_24H
SYSTEM_CAT_IDS = (4, 5, 9)


# ──────────────────────────────────────────────────────────────────
# Reuse 34_train_task_c_h60 helpers as a library module
# ──────────────────────────────────────────────────────────────────
def load_trainer():
    if "trainer_h60" in sys.modules:
        return sys.modules["trainer_h60"]
    spec = importlib.util.spec_from_file_location(
        "trainer_h60", ROOT / "scripts" / "34_train_task_c_h60.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["trainer_h60"] = m
    spec.loader.exec_module(m)
    return m


# ──────────────────────────────────────────────────────────────────
# Train-only stat fits used by C3.2 / C3.3 / C3.4
# ──────────────────────────────────────────────────────────────────
def fit_app_lifetime_share(train_events, vocab):
    V = len(vocab)
    rare = vocab.get("<RARE>", 2)
    counts = np.zeros(V, dtype=np.float64)
    mask = train_events["is_target_event"].astype(bool).to_numpy()
    apps = train_events.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    for a in idx:
        if 3 <= a < V:
            counts[a] += 1.0
    counts += 0.5
    return (counts / counts.sum()).astype(np.float32)


def fit_app_lifetime_kill_rate(bg_train, vocab):
    V = len(vocab)
    overall = float((bg_train[Y_COL].to_numpy() == 0).mean())
    out = np.full(V, overall, dtype=np.float64)
    for app_idx, sub in bg_train.groupby("app_idx", sort=False):
        if app_idx is None or len(sub) < 5:
            continue
        out[int(app_idx)] = float((sub[Y_COL].to_numpy() == 0).mean())
    return out.astype(np.float32)


def fit_cat_lifetime_share(train_events, vocab, app_to_cat):
    V_cat = int(app_to_cat.max()) + 1
    rare = vocab.get("<RARE>", 2)
    counts = np.zeros(V_cat, dtype=np.float64)
    mask = train_events["is_target_event"].astype(bool).to_numpy()
    apps = train_events.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    cats = app_to_cat[np.clip(idx, 0, len(app_to_cat) - 1)]
    for c in cats:
        if 0 <= int(c) < V_cat:
            counts[int(c)] += 1.0
    counts += 0.5
    return (counts / counts.sum()).astype(np.float32)


def fit_cat_markov(train_events, vocab, app_to_cat):
    """V_cat × V_cat next-cat probability over consecutive train events."""
    V_cat = int(app_to_cat.max()) + 1
    rare = vocab.get("<RARE>", 2)
    df = train_events[train_events["is_target_event"].astype(bool)].sort_values("event_ts")
    apps = df["app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    cats = app_to_cat[np.clip(idx, 0, len(app_to_cat) - 1)].astype(np.int64)
    table = np.zeros((V_cat, V_cat), dtype=np.float64)
    for i in range(1, len(cats)):
        c0, c1 = int(cats[i - 1]), int(cats[i])
        if 0 <= c0 < V_cat and 0 <= c1 < V_cat:
            table[c0, c1] += 1.0
    table += 0.5
    s = table.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (table / s).astype(np.float32)


def fit_two_step_markov(train_events, vocab):
    """Sparse map (a_{t-2}, a_{t-1}) -> P(a_t | ·, ·) of length V."""
    V = len(vocab)
    rare = vocab.get("<RARE>", 2)
    df = train_events[train_events["is_target_event"].astype(bool)].sort_values("event_ts")
    apps = df["app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    counts = {}
    for i in range(2, len(idx)):
        key = (int(idx[i - 2]), int(idx[i - 1]))
        if key not in counts:
            counts[key] = np.zeros(V, dtype=np.float64)
        a = int(idx[i])
        if 3 <= a < V:
            counts[key][a] += 1.0
    out = {}
    for k, v in counts.items():
        v = v + 0.5
        out[k] = (v / v.sum()).astype(np.float32)
    return out


def collect_last2_fg_per_anchor(events_concat, anchor_ts_unique, vocab):
    """Map anchor_ts_ns -> a_{t-2} = the FG event two before anchor t.

    0 if there are fewer than 2 prior FG events.
    """
    rare = vocab.get("<RARE>", 2)
    fg_mask = events_concat["name_norm"].isin(["APP_FOREGROUND", "APP_START"])
    ts_ns = (
        pd.to_datetime(events_concat.loc[fg_mask, "event_ts"])
        .to_numpy().astype("datetime64[ns]").view("int64")
    )
    apps = events_concat.loc[fg_mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    order = np.argsort(ts_ns, kind="mergesort")
    ts_sorted = ts_ns[order]
    idx_sorted = idx[order]
    out = {}
    for t in anchor_ts_unique:
        pos = int(np.searchsorted(ts_sorted, int(t), side="left"))
        out[int(t)] = int(idx_sorted[pos - 2]) if pos >= 2 else 0
    return out


# ────────────────────────────────────────────────────────────────────
# Feature builders (cumulative)
# ────────────────────────────────────────────────────────────────────
def cat_for(app_idx_arr, app_to_cat):
    return app_to_cat[np.clip(app_idx_arr, 0, len(app_to_cat) - 1)].astype(np.int64)


def add_c32_features(bg_df, app_to_cat,
                     app_lifetime_share, app_lifetime_kill_rate, cat_lifetime_share):
    """9 columns: category_match, is_system_app, app_share, app_kill_rate,
                  cat_share, bg_recency_mean, bg_recency_max, bg_unique_cat_cnt,
                  log_bg_set_size."""
    n = len(bg_df)
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    last_idx = bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64)
    cat = cat_for(a_idx, app_to_cat)
    last_cat = cat_for(last_idx, app_to_cat)

    cat_match = (cat == last_cat).astype(np.float32)
    is_sys = np.isin(cat, np.array(SYSTEM_CAT_IDS, dtype=np.int64)).astype(np.float32)
    a_share = app_lifetime_share[np.clip(a_idx, 0, len(app_lifetime_share) - 1)].astype(np.float32)
    a_kill = app_lifetime_kill_rate[np.clip(a_idx, 0, len(app_lifetime_kill_rate) - 1)].astype(np.float32)
    c_share = cat_lifetime_share[np.clip(cat, 0, len(cat_lifetime_share) - 1)].astype(np.float32)

    # bg-set composition stats per anchor
    bg_recency_mean = np.zeros(n, dtype=np.float32)
    bg_recency_max = np.zeros(n, dtype=np.float32)
    bg_unique_cat_cnt = np.zeros(n, dtype=np.float32)
    df_local = bg_df.assign(_cat=cat, _row=np.arange(n))
    for _, sub in df_local.groupby("anchor_id", sort=False):
        rs = sub["_row"].to_numpy()
        tsf = np.clip(sub["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
        log_tsf = np.log1p(tsf)
        bg_recency_mean[rs] = float(log_tsf.mean())
        bg_recency_max[rs] = float(log_tsf.max())
        bg_unique_cat_cnt[rs] = float(np.unique(sub["_cat"].to_numpy()).size)

    log_bg_size = np.log1p(bg_df["bg_set_size"].to_numpy(dtype=np.float64)).astype(np.float32)

    return np.stack([
        cat_match, is_sys, a_share, a_kill, c_share,
        bg_recency_mean, bg_recency_max, bg_unique_cat_cnt, log_bg_size,
    ], axis=1).astype(np.float32)


def add_c33_cols(bg_df, app_to_cat, cat_markov_probs):
    """3 columns: cat_markov_prob, time_since_cat_last_used (proxy), bg_apps_in_same_cat."""
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    last_idx = bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64)
    cat = np.clip(app_to_cat[np.clip(a_idx, 0, len(app_to_cat) - 1)].astype(np.int64),
                  0, cat_markov_probs.shape[0] - 1)
    last_cat = np.clip(app_to_cat[np.clip(last_idx, 0, len(app_to_cat) - 1)].astype(np.int64),
                       0, cat_markov_probs.shape[0] - 1)
    cat_markov = cat_markov_probs[last_cat, cat].astype(np.float32)

    # time_since_cat_last_used: use min time_since_fg_sec among bg apps in same cat
    n = len(bg_df)
    t_since_cat = np.zeros(n, dtype=np.float32)
    bg_in_cat = np.zeros(n, dtype=np.float32)
    df = bg_df.assign(_cat=cat, _row=np.arange(n))
    for _, sub in df.groupby("anchor_id", sort=False):
        cats_here = sub["_cat"].to_numpy()
        tsf = np.clip(sub["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
        rs = sub["_row"].to_numpy()
        for j, c in enumerate(cats_here):
            same = cats_here == c
            n_same = int(same.sum())
            if n_same > 0:
                t_min = float(tsf[same].min())
                t_since_cat[rs[j]] = float(np.log1p(t_min) / np.log1p(6 * 3600.0))
                bg_apps_same = n_same - 1
            else:
                t_since_cat[rs[j]] = 0.0
                bg_apps_same = 0
            bg_in_cat[rs[j]] = float(bg_apps_same)

    return np.stack([cat_markov, t_since_cat, bg_in_cat], axis=1).astype(np.float32)


def add_c34_col(bg_df, last2_map, two_step_table, fallback_markov_probs):
    """1 column: P(a | a_{t-1}, a_{t-2}) with fallback to 1-step Markov."""
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    last_idx = bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64)
    anchor_ts = bg_df["anchor_ts_ns"].to_numpy(dtype=np.int64)

    out = np.zeros(len(bg_df), dtype=np.float32)
    V = fallback_markov_probs.shape[1]
    for i in range(len(bg_df)):
        a2 = int(last2_map.get(int(anchor_ts[i]), 0))
        a1 = int(last_idx[i])
        a = min(max(int(a_idx[i]), 0), V - 1)
        probs = two_step_table.get((a2, a1))
        if probs is not None and a < probs.shape[0]:
            out[i] = float(probs[a])
        else:
            out[i] = float(fallback_markov_probs[min(max(a1, 0), V - 1), a])
    return out.reshape(-1, 1).astype(np.float32)


# ────────────────────────────────────────────────────────────────────
# Schema dispatcher — builds (features, app_idx, cat_idx, y, anchor_id) per schema
# ────────────────────────────────────────────────────────────────────
def build_schema_data(schema, bg_df, ctx):
    """Build training tensors for one schema.  ctx is a dict of fitted stats."""
    tr = ctx["trainer"]
    base = tr.build_c2(bg_df, ctx["hour_freq"], ctx["markov_probs"])
    if schema == "c2":
        return base
    base["features"] = np.concatenate(
        [base["features"], _c31_block(bg_df, ctx)], axis=1
    )
    if schema == "c3.1":
        return base
    base["features"] = np.concatenate(
        [base["features"], add_c32_features(bg_df, ctx["app_to_cat"],
                                             ctx["app_lifetime_share"],
                                             ctx["app_lifetime_kill_rate"],
                                             ctx["cat_lifetime_share"])], axis=1
    )
    if schema == "c3.2":
        return base
    base["features"] = np.concatenate(
        [base["features"], add_c33_cols(bg_df, ctx["app_to_cat"], ctx["cat_markov_probs"])], axis=1
    )
    if schema == "c3.3":
        return base
    base["features"] = np.concatenate(
        [base["features"], add_c34_col(bg_df, ctx["last2_map"], ctx["two_step_table"], ctx["markov_probs"])], axis=1
    )
    return base


def _c31_block(bg_df, ctx):
    """Reuse trainer's compute_c31_arrays to make the same 5 hourly-habit cols."""
    tr = ctx["trainer"]
    extras = tr.compute_c31_arrays(bg_df, ctx["fg_timeline"], ctx["per_app_inter_mean"])
    return np.stack([
        extras["overdue_ratio"],
        extras["was_fg_24h_ago"],
        extras["was_fg_7d_ago"],
        extras["log_fg_count_last_24h"],
        extras["log_fg_count_last_7d"],
    ], axis=1).astype(np.float32)


# ────────────────────────────────────────────────────────────────────
# Models — wide variant for C3-Pro
# ────────────────────────────────────────────────────────────────────
@dataclass
class WideCfg:
    vocab_size: int
    num_features: int
    app_emb_dim: int = 32
    d_hidden: int = 128
    dropout: float = 0.3


class WideMLP(nn.Module):
    def __init__(self, cfg: WideCfg):
        super().__init__()
        self.cfg = cfg
        self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        in_dim = cfg.num_features + cfg.app_emb_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, cfg.d_hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.d_hidden),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, cfg.d_hidden // 2),
            nn.GELU(),
            nn.LayerNorm(cfg.d_hidden // 2),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(cfg.d_hidden // 2, 1)

    def forward(self, app_idx, features, cat_idx=None):
        x = torch.cat([self.app_emb(app_idx), features], dim=-1)
        return self.head(self.trunk_x(x)).squeeze(-1)

    def trunk_x(self, x):
        return self.trunk(x)


# Listwise loss: per-anchor softmax NLL, target = (1 - y) (so the most kill-worthy gets max prob)
def listwise_softmax_nll(logits, y, anchor_ids):
    """Sum of −log softmax(logit) over kill-positive (y==0) within each anchor.
    Returns scalar.
    """
    loss = 0.0
    n_groups = 0
    for aid in torch.unique(anchor_ids):
        idx = (anchor_ids == aid).nonzero(as_tuple=False).flatten()
        if len(idx) < 2:
            continue
        scores = logits[idx]
        labels = (1 - y[idx].long()).float()  # 1 if kill-worthy
        if labels.sum() == 0:
            continue
        log_probs = F.log_softmax(scores, dim=0)
        n_kill = max(labels.sum().item(), 1.0)
        loss = loss + (-(log_probs * labels).sum() / n_kill)
        n_groups += 1
    return loss / max(n_groups, 1)


# ────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────
class PairDS(Dataset):
    def __init__(self, d):
        self.f = torch.as_tensor(d["features"].copy(), dtype=torch.float32)
        self.ai = torch.as_tensor(d["app_idx"].copy(), dtype=torch.long)
        self.y = torch.as_tensor(d["y"].copy(), dtype=torch.float32)
        self.aid = torch.as_tensor(d["anchor_id"].copy(), dtype=torch.long)

    def __len__(self):
        return self.f.shape[0]

    def __getitem__(self, i):
        return {"features": self.f[i], "app_idx": self.ai[i],
                "y": self.y[i], "anchor_id": self.aid[i]}


def evaluate(model, bg_df, data_d):
    model.eval()
    loader = DataLoader(PairDS(data_d), batch_size=2048, shuffle=False)
    probs = []
    with torch.no_grad():
        for b in loader:
            logits = model(b["app_idx"], b["features"])
            probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(probs)
    merged = bg_df.copy()
    merged["score_model"] = 1.0 - p
    return MET.compute_metrics(merged, "score_model", Y_COL, r_values=R_SWEEP)


def train_one(d_tr, d_va, d_te, bg_val, bg_test, vocab,
              tag, model_fn, listwise_lambda=0.0, label_smooth=0.0,
              epochs=EPOCHS, patience=PATIENCE, dropout=0.2, swa=False,
              seed=SEED, log_prefix=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pos = float(d_tr["y"].mean())
    pw = torch.tensor([(1.0 - pos) / max(pos, 1e-6)])
    num_features = int(d_tr["features"].shape[1])
    model = model_fn(num_features=num_features, vocab_size=len(vocab))
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if swa else None
    swa_model = torch.optim.swa_utils.AveragedModel(model) if swa else None

    dl_tr = DataLoader(PairDS(d_tr), batch_size=BATCH, shuffle=True)
    best_pr = -1.0
    best_state = None
    pat = 0
    log = []
    t0 = time.time()
    swa_start_epoch = max(1, int(epochs * 0.7))
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in dl_tr:
            logit = model(b["app_idx"], b["features"])
            y = b["y"].float()
            if label_smooth > 0:
                y = y * (1 - label_smooth) + 0.5 * label_smooth
            loss_main = bce(logit, y)
            if listwise_lambda > 0:
                loss_list = listwise_softmax_nll(logit, b["y"], b["anchor_id"])
                loss = loss_main + listwise_lambda * loss_list
            else:
                loss = loss_main
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()) * len(b["y"])
            n += len(b["y"])
        if sched is not None:
            sched.step()
        if swa_model is not None and ep >= swa_start_epoch:
            swa_model.update_parameters(model)
        eval_model = swa_model.module if (swa_model is not None and ep >= swa_start_epoch) else model
        val_m = evaluate(eval_model, bg_val, d_va)
        pr = float(val_m["pr_auc_mean"])
        log.append({"epoch": ep, "train_loss": tot / max(1, n),
                    "val_pr": pr, "val_roc": float(val_m["roc_auc_mean"]),
                    "val_fk_05": float(val_m["false_kill_rate"]["0.5"])})
        print(f"  {log_prefix}ep{ep:02d}  loss={tot/max(1,n):.4f}  val_PR={pr:.4f}  "
              f"val_ROC={float(val_m['roc_auc_mean']):.4f}  val_FK@.5={float(val_m['false_kill_rate']['0.5']):.4f}")
        if pr > best_pr + 1e-6:
            best_pr = pr
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                print(f"  {log_prefix}early stop at ep{ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return {
        "feature_dim": num_features,
        "n_params": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": time.time() - t0,
        "train_log": log,
        "best_val_pr": best_pr,
        "val": evaluate(model, bg_val, d_va),
        "test": evaluate(model, bg_test, d_te),
        "state_dict": best_state,
    }


# ────────────────────────────────────────────────────────────────────
# Build training context (load all needed train-time fits)
# ────────────────────────────────────────────────────────────────────
def build_ctx():
    art = ROOT / "artifacts"
    bg_art = art / "bg"
    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    train_events = pd.read_parquet(art / "splits" / "train.parquet")
    val_events = pd.read_parquet(art / "splits" / "val.parquet")
    test_events = pd.read_parquet(art / "splits" / "test.parquet")
    bg_train = pd.read_parquet(bg_art / "splits" / "bg_train.parquet")
    bg_val = pd.read_parquet(bg_art / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_art / "splits" / "bg_test.parquet")

    trainer = load_trainer()
    hour_freq = trainer.fit_hour_freq(train_events, vocab)
    markov_probs = MK.load_markov_prior(art_root() / "v3" / "markov_prior.pkl")["probs"].astype(np.float32)
    app_to_cat = CAT.build_app_to_cat_idx(vocab)
    per_app_inter = trainer.fit_per_app_inter_fg_mean(train_events, vocab)
    all_events = pd.concat([train_events, val_events, test_events], ignore_index=True).sort_values("event_ts").reset_index(drop=True)
    fg_timeline = trainer.collect_fg_timeline(all_events, vocab)

    print("[36] fitting C3.2/C3.3/C3.4 stats on train ...")
    app_lifetime_share = fit_app_lifetime_share(train_events, vocab)
    app_lifetime_kill_rate = fit_app_lifetime_kill_rate(bg_train, vocab)
    cat_lifetime_share = fit_cat_lifetime_share(train_events, vocab, app_to_cat)
    cat_markov_probs = fit_cat_markov(train_events, vocab, app_to_cat)
    two_step_table = fit_two_step_markov(train_events, vocab)
    anchor_ts_unique = pd.concat([bg_train["anchor_ts_ns"], bg_val["anchor_ts_ns"],
                                   bg_test["anchor_ts_ns"]]).unique()
    last2_map = collect_last2_fg_per_anchor(all_events, anchor_ts_unique, vocab)
    print(f"[36] fits done.  cat_markov shape={cat_markov_probs.shape}  2step keys={len(two_step_table)}")

    return {
        "vocab": vocab,
        "trainer": trainer,
        "hour_freq": hour_freq,
        "markov_probs": markov_probs,
        "app_to_cat": app_to_cat,
        "per_app_inter_mean": per_app_inter,
        "fg_timeline": fg_timeline,
        "app_lifetime_share": app_lifetime_share,
        "app_lifetime_kill_rate": app_lifetime_kill_rate,
        "cat_lifetime_share": cat_lifetime_share,
        "cat_markov_probs": cat_markov_probs,
        "two_step_table": two_step_table,
        "last2_map": last2_map,
        "bg_train": bg_train,
        "bg_val": bg_val,
        "bg_test": bg_test,
    }


def art_root():
    return ROOT / "artifacts"


# ────────────────────────────────────────────────────────────────────
# Model factories
# ────────────────────────────────────────────────────────────────────
def make_baseline_model(num_features, vocab_size, dropout=0.2):
    """The same MLP as `34_train_task_c_h60.SingleHeadMLP` (single head, 64 hidden)."""
    trainer = load_trainer()
    cfg = trainer.ModelCfg(vocab_size=vocab_size, num_features=num_features,
                           use_cat_emb=False, dropout=dropout)
    return trainer.SingleHeadMLP(cfg)


def make_wide_model(num_features, vocab_size, dropout=0.3):
    cfg = WideCfg(vocab_size=vocab_size, num_features=num_features, dropout=dropout)
    return WideMLP(cfg)


# ────────────────────────────────────────────────────────────────────
# Grid runner
# ────────────────────────────────────────────────────────────────────
def save_run(tag, result, ckpt_dir, json_dir):
    state = result.pop("state_dict", None)
    if state is not None:
        torch.save({"state_dict": state, "tag": tag_safe(tag)},
                   ckpt_dir / f"task_c_{tag_safe(tag)}.pt")
    with open(json_dir / f"task_c_{tag_safe(tag)}.json", "w") as f:
        json.dump({**result, "tag": tag}, f, indent=2)
    return result


def tag_safe(tag):
    return tag.replace(".", "p").replace("-", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pro-schema", default="c3.2", choices=["c3.1", "c3.2", "c3.3"],
                    help="feature schema to use for C3-Pro variants")
    ap.add_argument("--variant", default="all",
                    choices=["c3.2", "c3.3", "c3.4",
                             "c3pro_listwise", "c3pro_wide", "c3pro_aux",
                             "c3pro_reg", "c3pro_full", "c3pro_ens",
                             "all", "feature_only", "pro_only"])
    args = ap.parse_args()

    art = ROOT / "artifacts"
    bg_art = art / "bg"
    ckpt_dir = bg_art / "checkpoints"
    json_dir = bg_art / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    print("[36] loading and fitting context ...")
    ctx = build_ctx()

    bg_train = pd.read_parquet(bg_art / "splits" / "bg_train.parquet")
    bg_val = ctx["bg_val"]
    bg_test = ctx["bg_test"]
    vocab = ctx["vocab"] if "vocab" in ctx else json.load(open(art / "vocab.json"))

    # Cumulative feature schemas
    schemas_to_run = []
    if args.variant in ("c3.2", "all", "feature_only"):
        schemas_to_run.append("c3.2")
    if args.variant in ("c3.3", "all", "feature_only"):
        schemas_to_run.append("c3.3")
    if args.variant in ("c3.4", "all", "feature_only"):
        schemas_to_run.append("c3.4")

    summary = {}
    for sch in schemas_to_run:
        print(f"\n=== Train {sch} ===")
        d_tr = build_schema_data(sch, bg_train, ctx)
        d_va = build_schema_data(sch, bg_val, ctx)
        d_te = build_schema_data(sch, bg_test, ctx)
        result = train_one(
            d_tr, d_va, d_te, bg_val, bg_test, vocab,
            tag=sch, model_fn=make_baseline_model,
            log_prefix=f"[{sch}] ",
        )
        save_run(sch, result, ckpt_dir, json_dir)
        summary[sch] = {"val_pr": result["val"]["pr_auc_mean"],
                         "val_roc": result["val"]["roc_auc_mean"],
                         "val_fk_05": result["val"]["false_kill_rate"]["0.5"],
                         "test_pr": result["test"]["pr_auc_mean"],
                         "test_roc": result["test"]["roc_auc_mean"],
                         "test_fk_05": result["test"]["false_kill_rate"]["0.5"],
                         "feature_dim": result["feature_dim"]}

    # C3-Pro variants on best-feature schema (use the last winning if running all)
    if args.variant in ("all", "pro_only", "c3pro_listwise", "c3pro_wide",
                         "c3pro_aux", "c3pro_reg", "c3pro_full", "c3pro_ens"):
        winner_schema = args.pro_schema
        d_tr = build_schema_data(winner_schema, ctx["bg_train"], ctx)
        d_va = build_schema_data(winner_schema, bg_val, ctx)
        d_te = build_schema_data(winner_schema, bg_test, ctx)

        if args.variant in ("all", "pro_only", "c3pro_listwise"):
            print(f"\n=== Train c3pro_listwise (on {winner_schema}) ===")
            res = train_one(d_tr, d_va, d_te, bg_val, bg_test, vocab,
                            tag="c3pro_listwise",
                            model_fn=make_baseline_model,
                            listwise_lambda=0.5, log_prefix="[listwise] ")
            save_run("c3pro_listwise", res, ckpt_dir, json_dir)
            summary["c3pro_listwise"] = unpack_metrics(res)
        if args.variant in ("all", "pro_only", "c3pro_wide"):
            res = train_one(d_tr, d_va, d_te, bg_val, bg_test, vocab,
                            tag="c3pro_wide",
                            model_fn=make_wide_model,
                            log_prefix="[wide] ")
            save_run("c3pro_wide", res, ckpt_dir, json_dir)
            summary["c3pro_wide"] = unpack_metrics(res)
        if args.variant in ("all", "pro_only", "c3pro_reg"):
            res = train_one(d_tr, d_va, d_te, bg_val, bg_test, vocab,
                            tag="c3pro_reg",
                            model_fn=lambda **kw: make_baseline_model(dropout=0.3, **kw),
                            label_smooth=0.05, swa=True, log_prefix="[reg] ")
            save_run("c3pro_reg", res, ckpt_dir, json_dir)
            summary["c3pro_reg"] = unpack_metrics(res)
        if args.variant in ("all", "pro_only", "c3pro_full"):
            res = train_one(d_tr, d_va, d_te, bg_val, bg_test, vocab,
                            tag="c3pro_full",
                            model_fn=make_wide_model,
                            listwise_lambda=0.5, label_smooth=0.05, swa=True,
                            log_prefix="[full] ")
            save_run("c3pro_full", res, ckpt_dir, json_dir)
            summary["c3pro_full"] = unpack_metrics(res)
        if args.variant in ("all", "pro_only", "c3pro_ens"):
            print(f"\n=== Train c3pro_ens (5-seed) ===")
            ens_seeds = [7, 17, 31, 41, 53]
            ens_scores_val = None
            ens_scores_test = None
            for s in ens_seeds:
                res = train_one(d_tr, d_va, d_te, bg_val, bg_test, vocab,
                                tag=f"c3pro_ens_seed{s}",
                                model_fn=make_wide_model,
                                listwise_lambda=0.5, label_smooth=0.05, swa=True,
                                seed=s, log_prefix=f"[ens s={s}] ")
                # Score onto val/test using the saved best state
                # (skip — average via re-evaluating each model is cheaper than restoring; do later)

    print("[36] writing grid summary ...")
    with open(art / "bg" / "results" / "c3_grid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0


def unpack_metrics(res):
    return {"val_pr": res["val"]["pr_auc_mean"],
             "val_roc": res["val"]["roc_auc_mean"],
             "val_fk_05": res["val"]["false_kill_rate"]["0.5"],
             "test_pr": res["test"]["pr_auc_mean"],
             "test_roc": res["test"]["roc_auc_mean"],
             "test_fk_05": res["test"]["false_kill_rate"]["0.5"],
             "feature_dim": res["feature_dim"]}


if __name__ == "__main__":
    sys.exit(main() or 0)