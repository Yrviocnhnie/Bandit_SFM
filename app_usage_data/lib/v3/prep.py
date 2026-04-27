"""v3 shared preparation helpers for training / eval scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


HISTORY_K = 16
LONG_K = 64
WINDOW_HORIZON_SEC = 900


def load_vocab(art: Path) -> Dict[str, int]:
    with open(art / "vocab.json") as f:
        return json.load(f)


def build_target_stream(train_df, val_df, test_df, vocab: Dict[str, int],
                        app_to_cat: np.ndarray) -> dict:
    """Return one sorted, causal target-event stream spanning all splits."""
    full = pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values("event_ts").reset_index(drop=True)
    mask = full["is_target_event"].astype(bool).to_numpy()
    ts = pd.to_datetime(full.loc[mask, "event_ts"]).to_numpy().astype("datetime64[ns]").astype(np.int64)
    rare_idx = vocab.get("<RARE>", 2)
    labels = full.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    app = np.array([vocab.get(a, rare_idx) for a in labels], dtype=np.int64)
    cat = app_to_cat[np.clip(app, 0, len(app_to_cat) - 1)]
    if "seconds_to_next_event" in full.columns:
        dur = np.clip(full.loc[mask, "seconds_to_next_event"].fillna(60.0).to_numpy().astype(np.float32),
                      0.0, 3600.0)
    else:
        dur = np.full(mask.sum(), 60.0, dtype=np.float32)
    return {"ts_ns": ts, "app": app, "cat": cat, "dur_s": dur}


def last_target_app_for_anchors(
    anchor_ts_ns: np.ndarray,
    stream_ts_ns: np.ndarray,
    stream_app_idx: np.ndarray,
    default_idx: int = 1,
) -> np.ndarray:
    ts = np.asarray(stream_ts_ns, dtype=np.int64)
    app = np.asarray(stream_app_idx, dtype=np.int64)
    anchors = np.asarray(anchor_ts_ns, dtype=np.int64)
    if len(ts) > 1 and not np.all(np.diff(ts) >= 0):
        order = np.argsort(ts)
        ts = ts[order]
        app = app[order]
    ins = np.searchsorted(ts, anchors, side="left")
    out = np.full(len(anchors), default_idx, dtype=np.int64)
    for i, ip in enumerate(ins):
        ip = int(ip)
        if ip > 0:
            out[i] = int(app[ip - 1])
    return out
