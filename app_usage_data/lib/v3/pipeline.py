"""Shared v3 dataset-building pipeline used by both Task A and Task B trainers.

Rebuilds per-split tensor dictionaries that match the keys expected by
`TaskAModelV3` / `TaskBModelV3`. All train-only stats are assumed to live in
`artifacts/v3/`; this module loads them and applies them causally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from lib import data as D
from lib import features as F
from lib import train as TU
from lib.v2 import global_features as GF
from lib.v3 import categories as CAT
from lib.v3 import features_v3 as FV
from lib.v3 import location as LOC
from lib.v3.window_rollups import (
    DEFAULT_WINDOWS_SEC,
    PER_WINDOW_DIM,
    build_window_aggregates,
)
from lib.v3.daypart import NUM_DAYPARTS, daypart_onehot


def full_target_stream(df: pd.DataFrame, vocab: Dict[str, int],
                       app_to_cat: np.ndarray) -> Dict[str, np.ndarray]:
    """Return a sorted-by-ts TARGET-event stream with app idx, category idx,
    and dwell-time (proxied by seconds_to_next_event, clipped >=0)."""
    mask = df["is_target_event"].astype(bool).to_numpy()
    ts = pd.to_datetime(df.loc[mask, "event_ts"]).to_numpy().astype("datetime64[ns]").astype(np.int64)
    apps = np.array(
        [vocab_lookup_app(a) for a in df.loc[mask, "app_label_clean"].fillna("<UNK>")],
        dtype=np.int64,
    )
    dur_col = "seconds_to_next_event" if "seconds_to_next_event" in df.columns else None
    if dur_col is not None:
        dur = df.loc[mask, dur_col].fillna(0.0).to_numpy().astype(np.float32)
    else:
        dur = np.full(mask.sum(), 60.0, dtype=np.float32)
    dur = np.clip(dur, 0.0, 3600.0).astype(np.float32)
    order = np.argsort(ts, kind="mergesort")
    apps_idx_arr = _vocab_apply(df.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy(), vocab)
    apps = apps_idx_arr[order]
    return {
        "ts": ts[order],
        "app_idx": apps,
        "cat_idx": app_to_cat[apps],
        "dur_s": dur[order],
    }


def _vocab_apply(labels: np.ndarray, vocab: Dict[str, int]) -> np.ndarray:
    rare_idx = vocab.get("<RARE>", vocab.get("<UNK>", 1))
    out = np.empty(len(labels), dtype=np.int64)
    for i, v in enumerate(labels):
        out[i] = vocab.get(str(v), rare_idx)
    return out


def vocab_lookup_app(label: str) -> int:
    # not used; placeholder for backwards-compat signature above
    return 1


def load_vocab(art_dir: Path) -> Dict[str, int]:
    with open(art_dir / "vocab.json") as f:
        return json.load(f)


def category_per_row_from_vocab(app_idx: np.ndarray, app_to_cat: np.ndarray) -> np.ndarray:
    idx = np.asarray(app_idx, dtype=np.int64)
    safe = np.clip(idx, 0, len(app_to_cat) - 1)
    return app_to_cat[safe]


def build_target_stream_from_splits(train, val, test, vocab: Dict[str, int],
                                    app_to_cat: np.ndarray):
    """Concatenate target events across all splits into a single chronological
    stream for causal window rollups (val/test anchors may read backward into
    train, but never forward into their own split's target set — this is
    guaranteed by the anchor-time comparison)."""
    dfs = [train, val, test]
    full = pd.concat(dfs, ignore_index=True).sort_values("event_ts").reset_index(drop=True)
    mask = full["is_target_event"].astype(bool).to_numpy()
    ts = pd.to_datetime(full.loc[mask, "event_ts"]).to_numpy().astype("datetime64[ns]").astype(np.int64)
    labels = full.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    apps = _vocab_apply(labels, vocab)
    cats = category_per_row_from_apps(apps, app_to_cat)
    dur_col = "seconds_to_next_event" if "seconds_to_next_event" in full_df_cols(full) else None
    if dur_col:
        dur = np.clip(full.loc[mask, dur_col].fillna(60.0).to_numpy().astype(np.float32), 0.0, 3600.0) if False else np.clip(
            full_df_col(full, mask, dur_col), 0.0, 3600.0
        )
    else:
        dur = np.full(mask.sum(), 60.0, dtype=np.float32)
    return {"ts": ts, "app_idx": apps, "cat_idx": cats, "dur_s": dur}


def full_df_cols(df):
    return df.columns.tolist()


def full_df_col(df, mask, col):
    return df.loc[mask, col].fillna(60.0).to_numpy().astype(np.float32)


def category_per_row_from_apps(app_idx: np.ndarray, app_to_cat: np.ndarray) -> np.ndarray:
    idx = np.asarray(app_idx, dtype=np.int64)
    safe = np.clip(idx, 0, len(app_to_cat) - 1)
    return app_to_cat[safe]


def build_target_stream(all_df: pd.DataFrame, vocab: Dict[str, int],
                        app_to_cat: np.ndarray) -> Dict[str, np.ndarray]:
    """Target-only sorted stream across train+val+test."""
    d = all_df.sort_values("event_ts").reset_index(drop=True)
    mask = d["is_target_event"].astype(bool).to_numpy()
    ts = pd.to_datetime(d.loc[mask, "event_ts"]).to_numpy().astype("datetime64[ns]").astype(np.int64)
    labels = d.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    rare_idx = vocab.get("<RARE>", vocab.get("<UNK>", 1))
    app = np.array([vocab.get(a, rare_idx) for a in labels], dtype=np.int64)
    cat = app_to_cat[np.clip(app, 0, len(app_to_cat) - 1)]
    if "seconds_to_next_event" in d.columns:
        dur = np.clip(d.loc[mask, "seconds_to_next_event"].fillna(60.0).to_numpy().astype(np.float32), 0.0, 3600.0)
    else:
        dur = np.full(mask.sum(), 60.0, dtype=np.float32)
    return {"ts_ns": ts, "app_idx": app, "cat_idx": cat, "dur_s": dur}


def vocab_lookup_app(label):
    return 1


def last_target_app_before(anchor_ts_ns: np.ndarray,
                           stream_ts_ns: np.ndarray,
                           stream_app_idx: np.ndarray,
                           default_idx: int = 1) -> np.ndarray:
    """For each anchor, return the app-idx of the most recent TARGET event
    strictly before the anchor. Used for the Markov prior lookup."""
    ts = np.asarray(stream_ts_ns, dtype=np.int64)
    apps = np.asarray(stream_app_idx, dtype=np.int64)
    order = np.argsort(ts)
    ts_s = ts[order]
    apps_s = apps[order]
    insertion = np.searchsorted(ts_s, anchor_ts_ns, side="left")
    out = np.full(len(anchor_ts_ns), default_idx, dtype=np.int64)
    for i, ins in enumerate(insertion):
        ip = int(ins)
        if ip > 0:
            out[i] = int(apps_s[ip - 1])
    return out
