"""Background-state reconstruction from the event log.

For each target row, snapshot which apps are alive in memory just before.
Apps enter on any APP_*/PROCESS_START/PAGE_SWITCH/target event; they leave
on PROCESS_EXIT or after `idle_timeout_sec` (default 30 min) of inactivity.
Snapshots are taken BEFORE state update for the current event.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

DEFAULT_IDLE_TIMEOUT_SEC = 30 * 60
KILL_RECENT_WINDOW_SEC = 10 * 60


@dataclass
class BGSnapshot:
    bg_mask: np.ndarray            # (V,) bool
    bg_recency_sec: np.ndarray     # (V,) float32, inf if not in BG
    bg_count: int
    is_reuse: bool
    prev_killed_app: int
    time_since_screen_on: float
    target_app: int


def _label_to_idx(label, vocab):
    if label is None:
        return vocab.get("<UNK>", 1)
    if isinstance(label, float) and np.isnan(label):
        return vocab.get("<UNK>", 1)
    return vocab.get(str(label), vocab.get("<RARE>", 2))


def reconstruct_bg(df, vocab, idle_timeout_sec=DEFAULT_IDLE_TIMEOUT_SEC):
    V = len(vocab)
    df = df.sort_values("event_ts").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return []
    ts_ns = pd.to_datetime(df["event_ts"]).to_numpy().astype("datetime64[ns]").astype(np.int64)
    ts = ts_ns.astype(np.float64) / 1e9
    is_target = df["is_target_event"].astype(bool).to_numpy()
    name = df["name_norm"].fillna("").astype(str).to_numpy()
    label = df["app_label_clean"].astype(object).to_numpy()

    bg = {}
    last_screen_on = -1e18
    last_kill_app = -1
    last_kill_ts = -1e18
    out = []

    for i in range(n):
        t = float(ts[i])
        nm = name[i]
        a = _label_to_idx(label[i], vocab)

        if bg:
            doomed = [k for k, tk in bg.items() if (t - tk) > idle_timeout_sec]
            for k in doomed:
                bg.pop(k, None)

        if is_target[i]:
            mask = np.zeros(V, dtype=bool)
            rec = np.full(V, np.inf, dtype=np.float32)
            for k, tk in bg.items():
                if 0 <= k < V:
                    mask[k] = True
                    rec[k] = max(0.0, t - tk)
            kill_app = last_kill_app if (t - last_kill_ts) <= KILL_RECENT_WINDOW_SEC else -1
            tso = (t - last_screen_on) if last_screen_on > -1e17 else float("inf")
            is_reuse_flag = bool(mask[a]) if 0 <= a < V else False
            out.append(BGSnapshot(
                bg_mask=mask,
                bg_recency_sec=rec,
                bg_count=int(mask.sum()),
                is_reuse=is_reuse_flag,
                prev_killed_app=int(kill_app),
                time_since_screen_on=float(tso),
                target_app=int(a) if 0 <= a < V else -1,
            ))

        if nm == "SCREENON_EVENT":
            last_screen_on = t
        elif nm == "PROCESS_EXIT":
            if 0 <= a < V and a >= 3:
                bg.pop(a, None)
                last_kill_app = a
                last_kill_ts = t
        else:
            if (nm.startswith("APP_") or nm == "PROCESS_START"
                    or nm == "ABILITY_OR_PAGE_SWITCH" or is_target[i]):
                if 0 <= a < V and a >= 3:
                    bg[a] = t

    return out


def per_target_arrays(snapshots, V):
    """Convert list of snapshots into stacked numpy arrays for batched training."""
    M = len(snapshots)
    bg_mask = np.zeros((M, V), dtype=np.float32)
    bg_recency = np.zeros((M, V), dtype=np.float32)
    bg_count = np.zeros(M, dtype=np.float32)
    is_reuse = np.zeros(M, dtype=np.float32)
    prev_killed = np.full(M, -1, dtype=np.int64)
    tsos = np.zeros(M, dtype=np.float32)
    cap_sec = 30 * 60.0
    for i, s in enumerate(snapshots):
        bg_mask[i] = s.bg_mask.astype(np.float32)
        rec = np.where(np.isfinite(s.bg_recency_sec), s.bg_recency_sec, 0.0)
        rec = np.clip(rec, 0.0, cap_sec)
        rec_norm = np.log1p(rec) / np.log1p(cap_sec)
        bg_recency[i] = rec_norm * s.bg_mask.astype(np.float32)
        bg_count[i] = float(s.bg_count)
        is_reuse[i] = float(s.is_reuse)
        prev_killed[i] = s.prev_killed_app
        tso = s.time_since_screen_on
        if not np.isfinite(tso) or tso < 0:
            tso = 3600.0
        tsos[i] = float(np.log1p(min(tso, 3600.0)) / np.log1p(3600.0))
    return {
        "bg_mask": bg_mask,
        "bg_recency": bg_recency,
        "bg_count": (np.clip(bg_count, 0.0, 20.0) / 20.0).astype(np.float32),
        "is_reuse": is_reuse,
        "prev_killed": prev_killed,
        "time_since_screen_on": tsos,
    }
