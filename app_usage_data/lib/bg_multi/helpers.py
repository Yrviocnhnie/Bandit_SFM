"""Multi-user orchestration helpers for Task C.

Thin orchestration layer on top of the single-user pipeline:
  - list_cohort_users    : enumerate the 22 user XLSX files
  - build_pooled_vocab   : pool train target events across users, count >= min_count
  - fit_per_user_stats   : bundle every per-user fit (Markov, hour_freq, lifetime stats)
  - stack_per_user       : (U, *) tensor of per-user arrays, indexed by uid_idx
  - make_global_anchor_id: collision-free anchor IDs across users
  - attach_user_columns  : add user_uid + user_id_idx columns to a per-user df

All heavy lifting calls into existing single-user fitters (lib/v3, scripts/34, scripts/36).
"""
from __future__ import annotations

import importlib.util
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

GLOBAL_ANCHOR_OFFSET = 1_000_000  # max anchors-per-user before collision (we have ~6k)


# ============================================================================
# Cohort enumeration
# ============================================================================
def short_uid_from_path(path: str | Path) -> str:
    """Mirror `lib.multiuser.short_uid` — first 12 hex chars of the user-hash filename."""
    base = os.path.basename(str(path))
    m = re.match(r"([0-9A-Fa-f]+)_cleaned\.xlsx", base)
    return m.group(1)[:12] if m else base.replace(".xlsx", "")


def list_cohort_users(
    data_root: str | Path = "/data00/ruiqing/app_forecasting/data/cleaned",
    sets: Tuple[str, ...] = ("M_beta_Top30", "top2000"),
) -> List[Tuple[str, Path, str]]:
    """Return [(set_name, xlsx_path, short_uid), ...] for every user XLSX in the cohort.

    Sorted deterministically by (set_name, filename) so uid_to_idx is stable across runs.
    """
    out: List[Tuple[str, Path, str]] = []
    rp = Path(data_root)
    if not rp.exists():
        raise FileNotFoundError(f"data_root not found: {rp}")
    for set_name in sets:
        sub = rp / set_name
        if not sub.exists():
            continue
        for f in sorted(sub.glob("*_cleaned.xlsx")):
            uid = short_uid_from_path(f)
            out.append((set_name, f, uid))
    return out


# ============================================================================
# Pooled vocab
# ============================================================================
def build_pooled_vocab(
    per_user_train_dfs: Dict[str, pd.DataFrame],
    min_count: int = 5,
) -> Dict[str, int]:
    """Pool train target events across users, keep apps with global count >= min_count.

    Args:
        per_user_train_dfs: {uid: train_df} — each must have `is_target_event` and
                            `app_label_clean` columns.
        min_count: global threshold (default 5, matches single-user RARE_MIN_COUNT).

    Returns:
        {token: idx} dict with reserved <PAD>=0, <UNK>=1, <RARE>=2 followed by apps in
        descending count order.
    """
    counter: Dict[str, int] = {}
    for uid, df in per_user_train_dfs.items():
        if "is_target_event" not in df.columns or "app_label_clean" not in df.columns:
            continue
        mask = df["is_target_event"].astype(bool).to_numpy()
        apps = df.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str)
        for a, c in apps.value_counts().items():
            counter[a] = counter.get(a, 0) + int(c)

    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2}
    # Sort by descending count for determinism, then by app name
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    for app, c in items:
        if c < min_count or app in vocab:
            continue
        vocab[app] = len(vocab)
    return vocab


# ============================================================================
# Per-user stats fitting (orchestration over single-user fitters)
# ============================================================================
def _import_module(name: str, path: Path):
    """Import a script as a module via importlib (mirrors 41_threshold_metrics.py)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module spec for {path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def fit_per_user_stats(
    uid: str,
    train_events: pd.DataFrame,
    bg_train_user: pd.DataFrame,
    vocab: Dict[str, int],
    app_to_cat: np.ndarray,
    all_events_user: pd.DataFrame,
    scripts_dir: Optional[Path] = None,
) -> Dict:
    """Fit every per-user stat used by Task C feature builders.

    Inputs:
        uid                : short user id (12-hex)
        train_events       : enriched event stream for THIS user, train split only
        bg_train_user      : (anchor, app) bg rows for THIS user, train split only
                             — used to fit `app_lifetime_kill_rate`
        vocab              : pooled global vocab
        app_to_cat         : (V,) int array mapping app_idx -> category_idx
        all_events_user    : enriched event stream for THIS user, all 3 splits concatenated
                             — used for `fg_timeline` (causal at lookup time via searchsorted < t)
        scripts_dir        : path to app_usage_data/scripts/ (for importlib lookup)

    Returns:
        Dict with all per-user stats shaped against the pooled vocab.
        Includes {"fit_split": "train", "uid": uid} as a leakage assertion.
    """
    if scripts_dir is None:
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    trainer_h60 = _import_module("trainer_h60_for_bgmulti", scripts_dir / "34_train_task_c_h60.py")
    grid = _import_module("grid_for_bgmulti", scripts_dir / "36_train_c3_grid.py")
    from lib.v3 import markov_prior as MP

    # Train-only stats — must read train_events / bg_train_user only
    hour_freq = trainer_h60.fit_hour_freq(train_events, vocab)
    per_app_inter_fg_mean = trainer_h60.fit_per_app_inter_fg_mean(train_events, vocab)
    markov = MP.fit_markov_prior(train_events, vocab)  # has fit_split assertion
    app_lifetime_share = grid.fit_app_lifetime_share(train_events, vocab)
    # fit_app_lifetime_kill_rate fails on an empty df; fall back to a 0.5 prior.
    if bg_train_user is None or len(bg_train_user) == 0 or "y_3600" not in bg_train_user.columns:
        app_lifetime_kill_rate = np.full(int(len(vocab)), 0.5, dtype=np.float32)
    else:
        app_lifetime_kill_rate = grid.fit_app_lifetime_kill_rate(bg_train_user, vocab)
    cat_lifetime_share = grid.fit_cat_lifetime_share(train_events, vocab, app_to_cat)
    cat_markov_probs = grid.fit_cat_markov(train_events, vocab, app_to_cat)

    # Full-stream-causal: fg_timeline is sorted per-app FG timestamps for this user.
    # `compute_c31_arrays` does searchsorted < anchor_ts → causal at lookup.
    fg_timeline = trainer_h60.collect_fg_timeline(all_events_user, vocab)

    return {
        "fit_split": "train",
        "uid": str(uid),
        "V": int(len(vocab)),
        "hour_freq":              hour_freq,                # (24, V) float32
        "per_app_inter_fg_mean":  per_app_inter_fg_mean,    # (V,)    float32
        "markov_probs":           markov["probs"],          # (V, V)  float32
        "app_lifetime_share":     app_lifetime_share,       # (V,)    float32
        "app_lifetime_kill_rate": app_lifetime_kill_rate,   # (V,)    float32
        "cat_lifetime_share":     cat_lifetime_share,       # (V_cat,)
        "cat_markov_probs":       cat_markov_probs,         # (V_cat, V_cat)
        "fg_timeline":            fg_timeline,              # {int app_idx: int64 sorted ts_ns}
    }


def save_per_user_stats(stats: Dict, path: str | Path) -> None:
    """Pickle a per-user stats dict; create parent directory."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(stats, f)


def load_per_user_stats(path: str | Path) -> Dict:
    """Reload + assert fit_split=='train'."""
    with open(path, "rb") as f:
        stats = pickle.load(f)
    assert stats.get("fit_split") == "train", (
        f"per-user stats must be fit on train; got fit_split={stats.get('fit_split')!r} "
        f"in {path}"
    )
    return stats


# ============================================================================
# Stat-stacking for vectorized lookup
# ============================================================================
def stack_per_user(
    per_user: Dict[str, Dict],
    key: str,
    uid_to_idx: Dict[str, int],
) -> np.ndarray:
    """Stack per-user arrays into a (U, *shape_of_key) tensor ordered by uid_idx.

    For example, stack_per_user(per_user, "hour_freq", uid_to_idx) returns a
    `(U, 24, V)` float32 array such that stacked[idx, :, :] == per_user[uid][key]
    where uid_to_idx[uid] == idx.
    """
    U = len(uid_to_idx)
    if U == 0:
        raise ValueError("uid_to_idx is empty")
    # Build inverse lookup
    idx_to_uid: List[Optional[str]] = [None] * U
    for uid, i in uid_to_idx.items():
        if not (0 <= i < U):
            raise ValueError(f"uid_to_idx has out-of-range index: {uid} -> {i}")
        idx_to_uid[i] = uid
    if any(u is None for u in idx_to_uid):
        raise ValueError("uid_to_idx is not dense (some indices missing)")
    # Sample shape from first user
    sample = per_user[idx_to_uid[0]][key]
    if not isinstance(sample, np.ndarray):
        raise TypeError(f"key {key!r} is not an ndarray; got {type(sample)}")
    out = np.empty((U,) + sample.shape, dtype=sample.dtype)
    for i, uid in enumerate(idx_to_uid):
        arr = per_user[uid][key]
        if arr.shape != sample.shape:
            raise ValueError(
                f"shape mismatch for uid={uid} key={key}: "
                f"{arr.shape} vs {sample.shape}"
            )
        out[i] = arr
    return out


# ============================================================================
# Global anchor IDs (collision-free across users)
# ============================================================================
def make_global_anchor_id(df: pd.DataFrame, offset: int = GLOBAL_ANCHOR_OFFSET) -> np.ndarray:
    """Build a globally-unique anchor id per row.

    `df` must have `user_id_idx` (int) and `anchor_id` (int) columns. Returns
    `user_id_idx * offset + anchor_id` as int64.

    Used to disambiguate listwise-loss groupbys and metric groupbys when rows
    from multiple users are pooled into one DataFrame.
    """
    if "user_id_idx" not in df.columns or "anchor_id" not in df.columns:
        raise KeyError("make_global_anchor_id requires 'user_id_idx' and 'anchor_id' columns")
    uid = df["user_id_idx"].to_numpy(dtype=np.int64)
    aid = df["anchor_id"].to_numpy(dtype=np.int64)
    if (aid >= offset).any():
        raise ValueError(
            f"per-user anchor_id exceeds offset={offset}; raise GLOBAL_ANCHOR_OFFSET"
        )
    return (uid * int(offset) + aid).astype(np.int64)


def attach_user_columns(
    df: pd.DataFrame, uid: str, uid_idx: int,
) -> pd.DataFrame:
    """Add `user_uid` (str) and `user_id_idx` (int) columns to a per-user df.

    Idempotent — safe to call multiple times. Columns are overwritten if present.
    """
    out = df.copy()
    out["user_uid"] = str(uid)
    out["user_id_idx"] = int(uid_idx)
    return out
