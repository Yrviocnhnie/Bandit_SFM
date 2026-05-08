"""Train ONE global Task C model on pooled multi-user bg data.

Same architectures + recipes as the single-user `36_train_c3_grid.py`, but:
  - data is pooled across 22 users (artifacts/bg_multi/splits/bg_*.parquet)
  - vocab is the pooled global vocab (artifacts/bg_multi/vocab.json, V≈243)
  - per-user feature stats are looked up by user_id_idx at feature-build time
  - listwise loss / metric groupbys use a global anchor id (user_id_idx * 1e6 + anchor_id)

Trains 4 production picks:
  - task_c_multi_c3p3          (C3.3 baseline MLP, dropout 0.2)
  - task_c_multi_c3pro_reg     (C3.3 + dropout 0.3 + label smoothing + cosine LR + SWA)
  - task_c_multi_c3pro_listwise (C3.3 + listwise softmax NLL aux loss)
  - task_c_multi_c3pro_wide    (C3.3 + wider arch — 32-d emb, 128→64 trunk, GELU)

Outputs:
  artifacts/bg_multi/checkpoints/task_c_multi_<tag>.pt
  artifacts/bg_multi/results/task_c_multi_<tag>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg import metrics_bg as MET
from lib.bg_multi.helpers import (
    GLOBAL_ANCHOR_OFFSET, load_per_user_stats, make_global_anchor_id, stack_per_user,
)
from lib.v3 import categories as CAT


# ============================================================================
# Import single-user modules via importlib (mirrors 41_threshold_metrics.py)
# ============================================================================
def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_TRAINER = _load_module("trainer_h60_for_train_multi", ROOT / "scripts" / "34_train_task_c_h60.py")
_GRID = _load_module("grid_for_train_multi", ROOT / "scripts" / "36_train_c3_grid.py")


Y_COL = "y_3600"
R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)


# ============================================================================
# Multi-user feature builders
# ============================================================================
def build_c2_multi(bg_df: pd.DataFrame, hour_freq_stack: np.ndarray,
                   markov_probs_stack: np.ndarray) -> np.ndarray:
    """Same as `_TRAINER.build_c2`, but Markov + hour_freq lookup is per-user.

    hour_freq_stack: (U, 24, V)
    markov_probs_stack: (U, V, V)

    Returns just the (N, num_features) feature tensor (no other keys).
    """
    uid = bg_df["user_id_idx"].to_numpy(dtype=np.int64)
    n = len(bg_df)

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

    V = markov_probs_stack.shape[1]
    last = np.clip(bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64), 0, V - 1)
    app = np.clip(bg_df["app_idx"].to_numpy(dtype=np.int64), 0, V - 1)

    # Per-user gather: markov_probs_stack[uid_i, last_i, app_i]
    markov_col = markov_probs_stack[uid, last, app].astype(np.float64)
    hr_idx = np.clip(h.astype(np.int64) % 24, 0, 23)
    hour_col = hour_freq_stack[uid, hr_idx, app].astype(np.float64)

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
        np.sin(2 * np.pi * h / 24.0),
        np.cos(2 * np.pi * h / 24.0),
        np.sin(2 * np.pi * w / 7.0),
        np.cos(2 * np.pi * w / 7.0),
        (dp_anchor == dp_last).astype(np.float64),
        markov_col,
        hour_col,
        bg_rec_norm,
        tso_norm,
        pk_match,
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def compute_c31_arrays_multi(bg_df: pd.DataFrame,
                              fg_timeline_per_user: dict[int, dict],
                              per_app_inter_stack: np.ndarray) -> dict[str, np.ndarray]:
    """Multi-user version of `_TRAINER.compute_c31_arrays`.

    Loops per user, slices bg_df, calls the single-user function, scatters back.

    fg_timeline_per_user: {uid_idx: {app_idx: sorted ts_ns ndarray}}
    per_app_inter_stack:  (U, V) per-user inter-FG mean
    """
    n = len(bg_df)
    out_overdue = np.zeros(n, dtype=np.float32)
    out_fg24 = np.zeros(n, dtype=np.float32)
    out_fg7d = np.zeros(n, dtype=np.float32)
    out_cnt24 = np.zeros(n, dtype=np.float32)
    out_cnt7d = np.zeros(n, dtype=np.float32)

    uid = bg_df["user_id_idx"].to_numpy(dtype=np.int64)
    # iterate per user_id_idx to slice + dispatch
    for uid_idx in np.unique(uid):
        mask = uid == uid_idx
        if not mask.any():
            continue
        sub_idx = np.where(mask)[0]
        sub_df = bg_df.iloc[sub_idx].reset_index(drop=True)
        ftl = fg_timeline_per_user.get(int(uid_idx), {})
        per_app_inter = per_app_inter_stack[int(uid_idx)]
        extras = _TRAINER.compute_c31_arrays(sub_df, ftl, per_app_inter)
        out_overdue[sub_idx] = extras["overdue_ratio"]
        out_fg24[sub_idx] = extras["was_fg_24h_ago"]
        out_fg7d[sub_idx] = extras["was_fg_7d_ago"]
        out_cnt24[sub_idx] = extras["log_fg_count_last_24h"]
        out_cnt7d[sub_idx] = extras["log_fg_count_last_7d"]
    return {
        "overdue_ratio": out_overdue,
        "was_fg_24h_ago": out_fg24,
        "was_fg_7d_ago": out_fg7d,
        "log_fg_count_last_24h": out_cnt24,
        "log_fg_count_last_7d": out_cnt7d,
    }


def add_c32_features_multi(bg_df, app_to_cat,
                           app_lifetime_share_stack,    # (U, V)
                           app_lifetime_kill_rate_stack, # (U, V)
                           cat_lifetime_share_stack):    # (U, V_cat)
    """Same 9 columns as `_GRID.add_c32_features`, with per-user lookups for
    app_share / app_kill_rate / cat_share. The bg-set composition stats are
    intra-anchor and don't depend on per-user fits — reused via groupby."""
    n = len(bg_df)
    uid = bg_df["user_id_idx"].to_numpy(dtype=np.int64)
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    last_idx = bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64)
    cat = _GRID.cat_for(a_idx, app_to_cat)
    last_cat = _GRID.cat_for(last_idx, app_to_cat)

    cat_match = (cat == last_cat).astype(np.float32)
    is_sys = np.isin(cat, np.array(_GRID.SYSTEM_CAT_IDS, dtype=np.int64)).astype(np.float32)

    V = app_lifetime_share_stack.shape[1]
    a_clip = np.clip(a_idx, 0, V - 1)
    a_share = app_lifetime_share_stack[uid, a_clip].astype(np.float32)
    a_kill = app_lifetime_kill_rate_stack[uid, a_clip].astype(np.float32)
    V_cat = cat_lifetime_share_stack.shape[1]
    cat_clip = np.clip(cat, 0, V_cat - 1)
    c_share = cat_lifetime_share_stack[uid, cat_clip].astype(np.float32)

    # Bg-set composition (groupby per global anchor — uses pooled df's anchor groups)
    bg_recency_mean = np.zeros(n, dtype=np.float32)
    bg_recency_max = np.zeros(n, dtype=np.float32)
    bg_unique_cat_cnt = np.zeros(n, dtype=np.float32)
    df_local = bg_df.assign(_cat=cat, _row=np.arange(n))
    # Group by the pair (user_id_idx, anchor_id) so two users' anchor=0 don't merge.
    for _, sub in df_local.groupby(["user_id_idx", "anchor_id"], sort=False):
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


def add_c33_cols_multi(bg_df, app_to_cat, cat_markov_probs_stack):
    """Same as `_GRID.add_c33_cols`, with per-user cat_markov lookup."""
    n = len(bg_df)
    uid = bg_df["user_id_idx"].to_numpy(dtype=np.int64)
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    last_idx = bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64)
    V_cat = cat_markov_probs_stack.shape[1]
    cat = np.clip(_GRID.cat_for(a_idx, app_to_cat).astype(np.int64), 0, V_cat - 1)
    last_cat = np.clip(_GRID.cat_for(last_idx, app_to_cat).astype(np.int64), 0, V_cat - 1)
    cat_markov = cat_markov_probs_stack[uid, last_cat, cat].astype(np.float32)

    t_since_cat = np.zeros(n, dtype=np.float32)
    bg_in_cat = np.zeros(n, dtype=np.float32)
    df = bg_df.assign(_cat=cat, _row=np.arange(n))
    for _, sub in df.groupby(["user_id_idx", "anchor_id"], sort=False):
        cats_here = sub["_cat"].to_numpy()
        tsf = np.clip(sub["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
        rs = sub["_row"].to_numpy()
        for j, c in enumerate(cats_here):
            same = cats_here == c
            n_same = int(same.sum())
            if n_same > 0:
                t_min = float(tsf[same].min())
                t_since_cat[rs[j]] = float(np.log1p(t_min) / np.log1p(6 * 3600.0))
                bg_in_cat[rs[j]] = float(n_same - 1)
            else:
                t_since_cat[rs[j]] = 0.0
                bg_in_cat[rs[j]] = 0.0
    return np.stack([cat_markov, t_since_cat, bg_in_cat], axis=1).astype(np.float32)


# ============================================================================
# c3.4 STAGE 1 — cheap features (F3 / F5 / F6 / F7)
# ============================================================================
HOUR_SEGMENT_BINS = [(6, 11, 0),    # morning   (6 - 10)
                     (11, 14, 1),   # lunch     (11 - 13)
                     (14, 18, 2),   # afternoon (14 - 17)
                     (18, 22, 3),   # evening   (18 - 21)
                     (22, 6, 4)]    # night     (22 - 5, wraps)


def hour_segment_onehot(hours: np.ndarray) -> np.ndarray:
    """Convert hours-of-day → 5-class one-hot (morning/lunch/afternoon/evening/night)."""
    n = len(hours)
    out = np.zeros((n, 5), dtype=np.float32)
    h = hours.astype(np.int64) % 24
    for (a, b, k) in HOUR_SEGMENT_BINS:
        if a <= b:
            mask = (h >= a) & (h < b)
        else:  # wrap-around (night)
            mask = (h >= a) | (h < b)
        out[mask, k] = 1.0
    return out


def add_c34_cheap_features_multi(bg_df: pd.DataFrame, ctx: dict) -> np.ndarray:
    """8 columns total: F3 (1) + F5 (1) + F6 (1) + F7 (5)."""
    uid_idx = bg_df["user_id_idx"].to_numpy(dtype=np.int64)
    hr = bg_df["anchor_hour"].to_numpy(dtype=np.int64) % 24
    dow = bg_df["anchor_weekday"].to_numpy(dtype=np.int64) % 7
    app = bg_df["app_idx"].to_numpy(dtype=np.int64)
    V = ctx["dow_hour_freq_stack"].shape[3]
    app_clip = np.clip(app, 0, V - 1)

    # F3: per-user P(app | dow, hour)
    f3 = ctx["dow_hour_freq_stack"][uid_idx, dow, hr, app_clip].astype(np.float32)

    # F5: app popularity bucket (scalar 0/1/2)
    f5 = ctx["app_popularity_bucket"][app_clip].astype(np.float32)

    # F6: cross-user same-app prior (per-user)
    f6 = np.empty(len(bg_df), dtype=np.float32)
    cu = ctx["cross_user_app_prior_per_user"]
    uid_str = bg_df["user_uid"].to_numpy()
    for u in np.unique(uid_str):
        mask = uid_str == u
        prior = cu.get(str(u))
        if prior is None:
            f6[mask] = 0.5
        else:
            f6[mask] = prior[app_clip[mask]]

    # F7: hour-segment one-hot (5 cols)
    f7 = hour_segment_onehot(hr)

    return np.concatenate(
        [f3[:, None], f5[:, None], f6[:, None], f7],
        axis=1,
    ).astype(np.float32)


def build_c34n_cheap_multi(schema: str, bg_df: pd.DataFrame, ctx: dict) -> dict:
    """c3.3 schema + 8 cheap features (F3, F5, F6, F7). 33 + 8 = 41 numeric features."""
    base = build_schema_data_multi("c3.3", bg_df, ctx)
    extra = add_c34_cheap_features_multi(bg_df, ctx)
    base["features"] = np.concatenate([base["features"], extra], axis=1)
    return base


def _build_data_dispatch(schema: str, bg_df: pd.DataFrame, ctx: dict) -> dict:
    """Schema-aware feature builder dispatch."""
    if schema == "c3.4n_cheap":
        return build_c34n_cheap_multi(schema, bg_df, ctx)
    if schema == "c3.4n_full":
        return build_c34n_full_multi(schema, bg_df, ctx)
    return build_schema_data_multi(schema, bg_df, ctx)


# ============================================================================
# c3.4 STAGE 2 — heavier numeric features (F1 = 2-step Markov, F2 = co-FG)
# ============================================================================
def attach_last2_fg(bg_df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """Add a `last2_fg_app_idx` int64 column to bg_df via per-user lookup.

    Idempotent — overwrites if present. Uses ctx['flat_fg_per_user'][uid] which
    is a dict {'ts', 'app'} of sorted (ts_ns, app_idx) arrays.
    """
    from lib.bg_multi.c34_features import compute_last2_fg_for_anchors
    out = np.zeros(len(bg_df), dtype=np.int64)
    uid_str = bg_df["user_uid"].to_numpy()
    anchor_ts_ns = bg_df["anchor_ts_ns"].to_numpy(dtype=np.int64)
    for u in np.unique(uid_str):
        mask = uid_str == u
        flat = ctx["flat_fg_per_user"].get(str(u))
        if flat is None or len(flat["ts"]) == 0:
            continue
        out[mask] = compute_last2_fg_for_anchors(
            anchor_ts_ns[mask], flat["ts"], flat["app"],
        )
    df = bg_df.copy()
    df["last2_fg_app_idx"] = out
    return df


def add_c34_full_features_multi(bg_df: pd.DataFrame, ctx: dict) -> np.ndarray:
    """3 columns: F1 (1) + F2 (2). Requires `last2_fg_app_idx` in bg_df."""
    if "last2_fg_app_idx" not in bg_df.columns:
        raise KeyError("bg_df must have last2_fg_app_idx; call attach_last2_fg first")
    n = len(bg_df)
    uid_str = bg_df["user_uid"].to_numpy()
    last2 = bg_df["last2_fg_app_idx"].to_numpy(dtype=np.int64)
    last  = bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64)
    app   = bg_df["app_idx"].to_numpy(dtype=np.int64)

    markov_stack = ctx["markov_probs_stack"]   # (U, V, V)
    V = markov_stack.shape[1]
    uid_idx = bg_df["user_id_idx"].to_numpy(dtype=np.int64)

    # F1: 2-step Markov P(app | last2, last) — fall back to 1-step Markov on miss
    f1 = np.zeros(n, dtype=np.float32)
    for u in np.unique(uid_str):
        m_u = ctx["two_step_per_user"].get(str(u), {})
        mask = uid_str == u
        idxs = np.where(mask)[0]
        for i in idxs:
            key = (int(last2[i]), int(last[i]))
            probs = m_u.get(key)
            a = int(min(max(int(app[i]), 0), V - 1))
            if probs is not None and a < probs.shape[0]:
                f1[i] = float(probs[a])
            else:
                f1[i] = float(markov_stack[
                    uid_idx[i], min(max(int(last[i]), 0), V - 1), a,
                ])

    # F2: co-FG matrix lookup
    co_fg = ctx["co_fg_per_user"]
    f2_self = np.zeros(n, dtype=np.float32)   # co_fg[uid][app, last]
    f2_mean = np.zeros(n, dtype=np.float32)   # mean over peers in B(t)
    for u in np.unique(uid_str):
        mat = co_fg.get(str(u))
        mask = uid_str == u
        if mat is None:
            continue
        a_idx = np.clip(app[mask], 0, V - 1)
        b_idx = np.clip(last[mask], 0, V - 1)
        f2_self[mask] = mat[a_idx, b_idx]

    df = bg_df.assign(_row=np.arange(n))
    for (uid, anchor_id), grp in df.groupby(["user_uid", "anchor_id"], sort=False):
        mat = co_fg.get(str(uid))
        if mat is None:
            continue
        rows = grp["_row"].to_numpy()
        b = int(np.clip(grp["last_fg_app_idx"].iloc[0], 0, V - 1))
        peers = np.clip(grp["app_idx"].to_numpy(dtype=np.int64), 0, V - 1)
        peer_co = mat[peers, b]
        for j, r in enumerate(rows):
            others = np.delete(peer_co, j)
            f2_mean[r] = float(others.mean()) if others.size > 0 else float(peer_co[j])

    return np.stack([f1, f2_self, f2_mean], axis=1).astype(np.float32)


def build_c34n_full_multi(schema: str, bg_df: pd.DataFrame, ctx: dict) -> dict:
    """c3.4n_cheap + 3 features (F1 + F2). 41 + 3 = 44 numeric features."""
    bg_df = attach_last2_fg(bg_df, ctx)
    base = build_c34n_cheap_multi("c3.4n_cheap", bg_df, ctx)
    extra = add_c34_full_features_multi(bg_df, ctx)
    base["features"] = np.concatenate([base["features"], extra], axis=1)
    return base


# ============================================================================
# c3.3 schema (original)
# ============================================================================
def build_schema_data_multi(schema: str, bg_df: pd.DataFrame, ctx: dict) -> dict:
    """Multi-user equivalent of `_GRID.build_schema_data`. Currently supports c3.3."""
    if schema not in ("c2", "c3.1", "c3.2", "c3.3"):
        raise NotImplementedError(f"schema {schema!r} not supported in multi-user trainer")

    feats = build_c2_multi(bg_df, ctx["hour_freq_stack"], ctx["markov_probs_stack"])
    if schema != "c2":
        c31 = compute_c31_arrays_multi(
            bg_df, ctx["fg_timeline_per_user"], ctx["per_app_inter_stack"],
        )
        feats = np.concatenate([feats, np.stack([
            c31["overdue_ratio"], c31["was_fg_24h_ago"], c31["was_fg_7d_ago"],
            c31["log_fg_count_last_24h"], c31["log_fg_count_last_7d"],
        ], axis=1).astype(np.float32)], axis=1)
    if schema == "c3.1":
        return _wrap_features(feats, bg_df, ctx)
    feats = np.concatenate([feats, add_c32_features_multi(
        bg_df, ctx["app_to_cat"],
        ctx["app_lifetime_share_stack"],
        ctx["app_lifetime_kill_rate_stack"],
        ctx["cat_lifetime_share_stack"],
    )], axis=1)
    if schema == "c3.2":
        return _wrap_features(feats, bg_df, ctx)
    feats = np.concatenate([feats, add_c33_cols_multi(
        bg_df, ctx["app_to_cat"], ctx["cat_markov_probs_stack"],
    )], axis=1)
    return _wrap_features(feats, bg_df, ctx)


def _wrap_features(feats: np.ndarray, bg_df: pd.DataFrame, ctx: dict) -> dict:
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    cat_idx = ctx["app_to_cat"][np.clip(a_idx, 0, len(ctx["app_to_cat"]) - 1)].astype(np.int64)
    return {
        "features": feats,
        "app_idx": a_idx,
        "cat_idx": cat_idx,
        "y": bg_df[Y_COL].to_numpy(dtype=np.int64),
        # Global anchor id — collision-free across users (used by listwise loss).
        "anchor_id": make_global_anchor_id(bg_df).astype(np.int64),
    }


# ============================================================================
# Multi-user training context
# ============================================================================
def build_ctx_multi(art_dir: Path) -> dict:
    """Load pooled vocab + uid map, all per-user stats, and stack tensors."""
    bg_multi = art_dir / "bg_multi"
    with open(bg_multi / "vocab.json") as f:
        vocab = json.load(f)
    with open(bg_multi / "uid_to_idx.json") as f:
        uid_to_idx = json.load(f)

    # Load per-user stats
    per_user: dict[str, dict] = {}
    for uid in uid_to_idx.keys():
        p = bg_multi / "per_user" / uid / "stats.pkl"
        per_user[uid] = load_per_user_stats(p)

    app_to_cat = CAT.build_app_to_cat_idx(vocab)

    # Stack the regular per-user arrays
    hour_freq_stack             = stack_per_user(per_user, "hour_freq",              uid_to_idx)
    markov_probs_stack          = stack_per_user(per_user, "markov_probs",           uid_to_idx)
    per_app_inter_stack         = stack_per_user(per_user, "per_app_inter_fg_mean",  uid_to_idx)
    app_lifetime_share_stack    = stack_per_user(per_user, "app_lifetime_share",     uid_to_idx)
    app_lifetime_kill_rate_stack = stack_per_user(per_user, "app_lifetime_kill_rate", uid_to_idx)
    cat_lifetime_share_stack    = stack_per_user(per_user, "cat_lifetime_share",     uid_to_idx)
    cat_markov_probs_stack      = stack_per_user(per_user, "cat_markov_probs",       uid_to_idx)

    # fg_timeline is a per-user dict-of-arrays (variable shape) — keep as nested dict
    fg_timeline_per_user: dict[int, dict] = {}
    for uid, idx in uid_to_idx.items():
        fg_timeline_per_user[int(idx)] = per_user[uid]["fg_timeline"]

    # ===================================================================
    # c3.4 stage 1 stats — fit on the fly (cheap, ~20s for 22 users)
    # ===================================================================
    from lib.bg_multi.c34_features import (
        fit_dow_hour_freq, fit_app_popularity_bucket, fit_cross_user_app_prior,
    )
    print(f"[ctx] fitting c3.4 stage-1 stats (DOW × hour, popularity, cross-user prior)...")
    train_dfs_per_user: dict[str, pd.DataFrame] = {}
    dow_hour_freq_per_user: dict[str, dict] = {}
    for uid in uid_to_idx:
        # Reuse existing per-user enriched train.parquet from 40_prep_multiuser.py
        candidates = list((art_dir / "multiuser").glob(f"*/{uid}/splits/train.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"per-user train.parquet not found for uid={uid}. "
                f"Run scripts/40_prep_multiuser.py first."
            )
        tr = pd.read_parquet(candidates[0])
        train_dfs_per_user[uid] = tr
        dow_hour_freq_per_user[uid] = {"dow_hour_freq": fit_dow_hour_freq(tr, vocab)}
    dow_hour_freq_stack = stack_per_user(dow_hour_freq_per_user, "dow_hour_freq", uid_to_idx)
    app_popularity_bucket = fit_app_popularity_bucket(
        train_dfs_per_user, vocab, common_min=15, medium_min=5,
    )
    bg_train_pooled = pd.read_parquet(bg_multi / "splits" / "bg_train.parquet")
    cross_user_app_prior_per_user = fit_cross_user_app_prior(bg_train_pooled, vocab)
    print(f"[ctx]   dow_hour_freq_stack {dow_hour_freq_stack.shape}, "
          f"popularity_bucket V={len(app_popularity_bucket)}, "
          f"cross_user_app_prior n_users={len(cross_user_app_prior_per_user)}")

    # ===================================================================
    # c3.4 stage 2 stats — F1 (2-step Markov) + F2 (co-FG matrix)
    # Plus flat per-user FG timeline for last2 lookups.
    # ===================================================================
    from lib.bg_multi.c34_features import (
        fit_two_step_markov_per_user, fit_co_fg_matrix_per_user,
    )
    print(f"[ctx] fitting c3.4 stage-2 stats (2-step Markov, co-FG matrix, flat FG timeline)...")
    flat_fg_per_user: dict[str, dict] = {}
    for uid, idx in uid_to_idx.items():
        ftl = fg_timeline_per_user[int(idx)]
        all_ts: list[np.ndarray] = []
        all_app: list[np.ndarray] = []
        for ai, ts_arr in ftl.items():
            if len(ts_arr) == 0:
                continue
            all_ts.append(ts_arr)
            all_app.append(np.full(len(ts_arr), int(ai), dtype=np.int64))
        if all_ts:
            ts_cat = np.concatenate(all_ts)
            app_cat = np.concatenate(all_app)
            order = np.argsort(ts_cat, kind="mergesort")
            flat_fg_per_user[uid] = {"ts": ts_cat[order], "app": app_cat[order]}
        else:
            flat_fg_per_user[uid] = {"ts": np.zeros(0, dtype=np.int64),
                                      "app": np.zeros(0, dtype=np.int64)}
    two_step_per_user: dict[str, dict] = {}
    for uid, tr in train_dfs_per_user.items():
        two_step_per_user[uid] = fit_two_step_markov_per_user(tr, vocab)
    co_fg_per_user = fit_co_fg_matrix_per_user(bg_train_pooled, vocab)
    n_two_step_keys = sum(len(v) for v in two_step_per_user.values())
    print(f"[ctx]   flat_fg_per_user n_users={len(flat_fg_per_user)}, "
          f"two_step_per_user total_keys={n_two_step_keys}, "
          f"co_fg_per_user n_users={len(co_fg_per_user)}")

    return {
        "vocab": vocab,
        "uid_to_idx": uid_to_idx,
        "app_to_cat": app_to_cat,
        "hour_freq_stack": hour_freq_stack,
        "markov_probs_stack": markov_probs_stack,
        "per_app_inter_stack": per_app_inter_stack,
        "app_lifetime_share_stack": app_lifetime_share_stack,
        "app_lifetime_kill_rate_stack": app_lifetime_kill_rate_stack,
        "cat_lifetime_share_stack": cat_lifetime_share_stack,
        "cat_markov_probs_stack": cat_markov_probs_stack,
        "fg_timeline_per_user": fg_timeline_per_user,
        # c3.4 stage 1 additions
        "dow_hour_freq_stack": dow_hour_freq_stack,
        "app_popularity_bucket": app_popularity_bucket,
        "cross_user_app_prior_per_user": cross_user_app_prior_per_user,
        "_train_dfs_per_user": train_dfs_per_user,
        "_bg_train_pooled": bg_train_pooled,
        # c3.4 stage 2 additions
        "flat_fg_per_user": flat_fg_per_user,
        "two_step_per_user": two_step_per_user,
        "co_fg_per_user": co_fg_per_user,
    }


def evaluate_multi(model: torch.nn.Module, bg_df: pd.DataFrame, data_d: dict,
                    use_cat_emb: bool = False) -> dict:
    """Forward + per-(user, anchor) Track A metrics. Forwards cat_idx when use_cat_emb."""
    model.eval()
    f = torch.as_tensor(data_d["features"].copy(), dtype=torch.float32)
    a = torch.as_tensor(data_d["app_idx"].copy(), dtype=torch.long)
    c = torch.as_tensor(data_d["cat_idx"].copy(), dtype=torch.long) if use_cat_emb else None
    probs = []
    with torch.no_grad():
        for s in range(0, f.shape[0], 4096):
            if use_cat_emb:
                logits = model(a[s:s + 4096], f[s:s + 4096], c[s:s + 4096])
            else:
                logits = model(a[s:s + 4096], f[s:s + 4096])
            probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(probs)
    merged = bg_df.copy()
    merged["score_model"] = 1.0 - p
    # Use global anchor id so two users' anchor=k don't merge in groupby
    merged["_global_aid"] = make_global_anchor_id(merged)
    merged_for_metric = merged.rename(columns={"anchor_id": "_per_user_aid",
                                                 "_global_aid": "anchor_id"})
    return MET.compute_metrics(merged_for_metric, "score_model", Y_COL,
                                r_values=R_SWEEP)


# ============================================================================
# c3.4 STAGE 3 — cat_emb forward path (model-side wiring; numeric features
# unchanged). Forks _GRID.train_one with cat_idx forwarded into the model.
# ============================================================================
import torch.nn as _nn
from torch.utils.data import DataLoader as _DataLoader

# Default hparams for cat-emb path — mirrors _GRID
_LR = 1e-3
_WD = 1e-4
_BATCH = 256


def make_baseline_with_cat_emb(num_features: int, vocab_size: int,
                                num_categories: int = 11,
                                cat_emb_dim: int = 4,
                                d_hidden: int = 64,
                                dropout: float = 0.2):
    """Single-head MLP with explicit cat_emb. Re-uses _TRAINER.SingleHeadMLP
    which already supports use_cat_emb=True."""
    cfg = _TRAINER.ModelCfg(
        vocab_size=vocab_size,
        num_features=num_features,
        use_cat_emb=True,
        num_categories=num_categories,
        cat_emb_dim=cat_emb_dim,
        d_hidden=d_hidden,
        dropout=dropout,
    )
    return _TRAINER.SingleHeadMLP(cfg)


class _PairDSCat(torch.utils.data.Dataset):
    """Dataset that surfaces cat_idx alongside app_idx + features."""
    def __init__(self, d):
        self.f = torch.as_tensor(d["features"].copy(), dtype=torch.float32)
        self.ai = torch.as_tensor(d["app_idx"].copy(), dtype=torch.long)
        self.ci = torch.as_tensor(d["cat_idx"].copy(), dtype=torch.long)
        self.y = torch.as_tensor(d["y"].copy(), dtype=torch.float32)
        self.aid = torch.as_tensor(d["anchor_id"].copy(), dtype=torch.long)

    def __len__(self):
        return self.f.shape[0]

    def __getitem__(self, i):
        return {"features": self.f[i], "app_idx": self.ai[i],
                "cat_idx": self.ci[i], "y": self.y[i],
                "anchor_id": self.aid[i]}


def train_one_cat(d_tr: dict, d_va: dict, d_te: dict,
                   bg_val: pd.DataFrame, bg_test: pd.DataFrame, vocab: dict,
                   tag: str, model_fn,
                   listwise_lambda: float = 0.0, label_smooth: float = 0.0,
                   epochs: int = 30, patience: int = 4, dropout: float = 0.2,
                   swa: bool = False, seed: int = 7,
                   log_prefix: str = "") -> dict:
    """Forked _GRID.train_one with cat_idx forwarded into the model.

    Same loss, optimizer, scheduling, early-stop as _GRID.train_one. The only
    change is `model(b["app_idx"], b["features"], b["cat_idx"])`.
    """
    import time as _time
    torch.manual_seed(seed)
    np.random.seed(seed)
    pos = float(d_tr["y"].mean())
    pw = torch.tensor([(1.0 - pos) / max(pos, 1e-6)])
    num_features = int(d_tr["features"].shape[1])
    model = model_fn(num_features=num_features, vocab_size=len(vocab))
    bce = _nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if swa else None
    swa_model = torch.optim.swa_utils.AveragedModel(model) if swa else None

    dl_tr = _DataLoader(_PairDSCat(d_tr), batch_size=_BATCH, shuffle=True)
    best_pr = -1.0
    best_state = None
    pat = 0
    log = []
    t0 = _time.time()
    swa_start = max(1, int(epochs * 0.7))
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in dl_tr:
            logit = model(b["app_idx"], b["features"], b["cat_idx"])
            y = b["y"].float()
            if label_smooth > 0:
                y = y * (1 - label_smooth) + 0.5 * label_smooth
            loss_main = bce(logit, y)
            if listwise_lambda > 0:
                loss_list = _GRID.listwise_softmax_nll(logit, b["y"], b["anchor_id"])
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
        if swa_model is not None and ep >= swa_start:
            swa_model.update_parameters(model)
        eval_model = swa_model.module if (swa_model is not None and ep >= swa_start) else model
        val_m = evaluate_multi(eval_model, bg_val, d_va, use_cat_emb=True)
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
        "elapsed_sec": _time.time() - t0,
        "train_log": log,
        "best_val_pr": best_pr,
        "val": evaluate_multi(model, bg_val, d_va, use_cat_emb=True),
        "test": evaluate_multi(model, bg_test, d_te, use_cat_emb=True),
        "state_dict": best_state,
    }


# ============================================================================
# Recipe registry + main
# ============================================================================
RECIPES = {
    "c3p3":           {"schema": "c3.3", "wide": False, "dropout": 0.2,
                       "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.0},
    "c3pro_reg":      {"schema": "c3.3", "wide": False, "dropout": 0.3,
                       "label_smooth": 0.05, "swa": True, "listwise_lambda": 0.0},
    "c3pro_listwise": {"schema": "c3.3", "wide": False, "dropout": 0.2,
                       "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.5},
    "c3pro_wide":     {"schema": "c3.3", "wide": True,  "dropout": 0.3,
                       "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.0},
    # c3.4 stage-1 candidate (c3.3 + F3/F5/F6/F7)
    "c3p4_cheap":     {"schema": "c3.4n_cheap", "wide": False, "dropout": 0.2,
                       "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.0},
    # c3.4 stage-2 candidate (c3.4n_cheap + F1 2-step Markov + F2 co-FG)
    "c3p4_full":      {"schema": "c3.4n_full",  "wide": False, "dropout": 0.2,
                       "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.0},
    # c3.4 stage-3 candidate (architecture change): c3.3 numeric features +
    # explicit cat_emb. Tests whether category embedding alone helps,
    # isolated from any numeric feature additions.
    "c3p3_cat":       {"schema": "c3.3", "wide": False, "dropout": 0.2,
                       "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.0,
                       "use_cat_emb": True},
}


def make_model(recipe: dict, num_features: int, vocab_size: int) -> torch.nn.Module:
    if recipe.get("use_cat_emb"):
        return make_baseline_with_cat_emb(num_features=num_features,
                                            vocab_size=vocab_size,
                                            dropout=recipe["dropout"])
    if recipe["wide"]:
        return _GRID.make_wide_model(num_features=num_features, vocab_size=vocab_size,
                                      dropout=recipe["dropout"])
    return _GRID.make_baseline_model(num_features=num_features, vocab_size=vocab_size,
                                      dropout=recipe["dropout"])


def train_recipe(recipe_name: str, ctx: dict,
                 bg_train: pd.DataFrame, bg_val: pd.DataFrame, bg_test: pd.DataFrame,
                 epochs: int, patience: int, seed: int) -> dict:
    """Train one recipe end-to-end. Returns a result dict (without state_dict here)."""
    rec = RECIPES[recipe_name]
    schema = rec["schema"]
    print(f"[52] building features for schema={schema!r} ...")
    t0 = time.time()
    d_tr = _build_data_dispatch(schema, bg_train, ctx)
    d_va = _build_data_dispatch(schema, bg_val, ctx)
    d_te = _build_data_dispatch(schema, bg_test, ctx)
    print(f"[52]   feature build: {time.time() - t0:.1f}s   "
          f"feature_dim={d_tr['features'].shape[1]}  "
          f"train_rows={d_tr['features'].shape[0]:,}")

    # Dispatch on whether the recipe uses cat_emb (different forward signature).
    print(f"[52] training recipe={recipe_name} ...")
    use_cat = bool(rec.get("use_cat_emb"))
    train_fn = train_one_cat if use_cat else _GRID.train_one
    result = train_fn(
        d_tr, d_va, d_te, bg_val, bg_test, ctx["vocab"],
        tag=recipe_name,
        model_fn=lambda num_features, vocab_size: make_model(
            rec, num_features=num_features, vocab_size=vocab_size,
        ),
        listwise_lambda=rec["listwise_lambda"],
        label_smooth=rec["label_smooth"],
        epochs=epochs, patience=patience, dropout=rec["dropout"],
        swa=rec["swa"], seed=seed, log_prefix=f"[{recipe_name}] ",
    )
    # Replace the (single-user) val/test metrics with multi-user-grouped ones.
    # train_one_cat already evaluates on multi-user grouped metrics; the c3.x
    # path goes through _GRID.train_one and needs the override.
    state = result.get("state_dict")
    model = make_model(rec, num_features=d_tr["features"].shape[1], vocab_size=len(ctx["vocab"]))
    if state is not None:
        model.load_state_dict(state, strict=False)
    result["val"] = evaluate_multi(model, bg_val, d_va, use_cat_emb=use_cat)
    result["test"] = evaluate_multi(model, bg_test, d_te, use_cat_emb=use_cat)
    return result


def save_run(recipe_name: str, result: dict, art_dir: Path):
    state = result.pop("state_dict", None)
    ckpt_dir = art_dir / "bg_multi" / "checkpoints"
    res_dir = art_dir / "bg_multi" / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    if state is not None:
        torch.save({"state_dict": state, "tag": recipe_name},
                   ckpt_dir / f"task_c_multi_{recipe_name}.pt")
    with open(res_dir / f"task_c_multi_{recipe_name}.json", "w") as f:
        json.dump({**result, "recipe": recipe_name}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--art-dir", type=str, default=str(ROOT / "artifacts"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--recipes", type=str, default="all",
                        help='comma list of recipes or "all"; choices=' + ",".join(RECIPES))
    parser.add_argument("--smoke", action="store_true",
                        help="2-epoch run on a 30 %% sample for quick verification")
    args = parser.parse_args()

    art_dir = Path(args.art_dir)
    bg_multi = art_dir / "bg_multi"
    bg_train = pd.read_parquet(bg_multi / "splits" / "bg_train.parquet")
    bg_val = pd.read_parquet(bg_multi / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_multi / "splits" / "bg_test.parquet")
    print(f"[52] pooled rows: train={len(bg_train):,}  val={len(bg_val):,}  test={len(bg_test):,}")

    if args.smoke:
        print("[52] SMOKE — sampling 30 % of bg_train")
        bg_train = bg_train.sample(frac=0.3, random_state=7).reset_index(drop=True)
        args.epochs = 2

    print(f"[52] building multi-user ctx ...")
    ctx = build_ctx_multi(art_dir)
    print(f"[52]   V_pool={len(ctx['vocab'])}  "
          f"U={ctx['markov_probs_stack'].shape[0]}  "
          f"V_cat={ctx['cat_markov_probs_stack'].shape[1]}")

    if args.recipes == "all":
        recipes = list(RECIPES.keys())
    else:
        recipes = [r.strip() for r in args.recipes.split(",") if r.strip()]
    for r in recipes:
        if r not in RECIPES:
            print(f"[52] unknown recipe: {r}; skipping", file=sys.stderr)
            continue

    summary = {}
    for r in recipes:
        if r not in RECIPES:
            continue
        print(f"\n{'=' * 70}\n[52] === recipe: {r} ===\n{'=' * 70}")
        t0 = time.time()
        result = train_recipe(r, ctx, bg_train, bg_val, bg_test,
                              epochs=args.epochs, patience=args.patience, seed=args.seed)
        save_run(r, result, art_dir)
        summary[r] = {
            "elapsed_sec": time.time() - t0,
            "feature_dim": result.get("feature_dim"),
            "n_params": result.get("n_params"),
            "best_val_pr": result.get("best_val_pr"),
            "test_pr": float(result["test"]["pr_auc_mean"]),
            "test_roc": float(result["test"]["roc_auc_mean"]),
            "test_fk_05": float(result["test"]["false_kill_rate"]["0.5"]),
            "test_msr_05": float(result["test"]["memory_save_rate"]["0.5"]),
        }
        print(f"[52] recipe={r} done in {summary[r]['elapsed_sec']:.0f}s  "
              f"test PR={summary[r]['test_pr']:.4f}  ROC={summary[r]['test_roc']:.4f}  "
              f"FK@.5={summary[r]['test_fk_05']:.4f}  MSR@.5={summary[r]['test_msr_05']:.4f}")

    with open(bg_multi / "results" / "task_c_multi_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[52] DONE  summary -> {bg_multi / 'results' / 'task_c_multi_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
