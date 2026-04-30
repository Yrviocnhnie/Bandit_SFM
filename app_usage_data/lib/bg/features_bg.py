"""Feature / label builders for Task C.

Produces per-(anchor, app) rows from `BGSnapshot` lists. The v3 profile (153 d)
is NOT baked into the parquet — we rebuild what we need at training time so
Task C stays in sync with v3 feature logic without duplicating it on disk.

This module also contains derivation helpers used by `scripts/30_build_bg_data.py`:
  * `compute_labels`            — y_{H} per (anchor, app)
  * `compute_rolling_fg_counts` — per-(anchor, app) FG counts in last X seconds
  * `compute_recency_ranks`     — per-(anchor, app) within-anchor rank by time_in_bg
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .background_state import BGSnapshot, NS_PER_SEC


HORIZONS_SEC_DEFAULT: Sequence[int] = (300, 600)
ROLLING_WINDOWS_SEC_DEFAULT: Sequence[int] = (3600, 21600)  # 1 h, 6 h


def compute_labels(
    snapshots: List[BGSnapshot],
    fg_event_ts_ns: np.ndarray,
    fg_event_app: np.ndarray,
    horizons_sec: Sequence[int] = HORIZONS_SEC_DEFAULT,
) -> Dict[int, List[np.ndarray]]:
    """Per snapshot, one int8 vector per horizon of shape (|B(t)|,).

    ``y_h[i] == 1`` iff ``snap.apps[i]`` appears as a foreground event in
    the half-open window ``(snap.anchor_ts_ns, snap.anchor_ts_ns + h]``.
    """
    order = np.argsort(fg_event_ts_ns, kind="mergesort")
    ts_sorted = np.asarray(fg_event_ts_ns)[order]
    apps_sorted = np.asarray(fg_event_app)[order].astype(object)

    per_app_ts: Dict[str, np.ndarray] = {}
    for app in np.unique(apps_sorted):
        per_app_ts[str(app)] = ts_sorted[apps_sorted == app]

    per_anchor: Dict[int, List[np.ndarray]] = {int(h): [] for h in horizons_sec}
    for snap in snapshots:
        base = int(snap.anchor_ts_ns)
        for h in horizons_sec:
            end = base + int(h) * NS_PER_SEC
            vec = np.zeros(len(snap.apps), dtype=np.int8)
            for i, app in enumerate(snap.apps):
                arr = per_app_ts.get(app)
                if arr is None or arr.size == 0:
                    continue
                lo = int(np.searchsorted(arr, base, side="right"))
                hi = int(np.searchsorted(arr, end, side="right"))
                if hi > lo:
                    vec[i] = 1
            per_anchor[int(h)].append(vec)
    return per_anchor


def compute_rolling_fg_counts(
    snapshots: List[BGSnapshot],
    fg_event_ts_ns: np.ndarray,
    fg_event_app: np.ndarray,
    windows_sec: Sequence[int] = ROLLING_WINDOWS_SEC_DEFAULT,
) -> Dict[int, List[np.ndarray]]:
    """Per snapshot, per app in B(t), count strictly-prior FG events of that
    app in the lookback window `[anchor - W, anchor)`.

    Returns: {window_sec: list-of-(|B(t)|,) int32 arrays, one per snapshot}.
    """
    order = np.argsort(fg_event_ts_ns, kind="mergesort")
    ts_sorted = np.asarray(fg_event_ts_ns)[order]
    apps_sorted = np.asarray(fg_event_app)[order].astype(object)

    per_app_ts: Dict[str, np.ndarray] = {}
    for app in np.unique(apps_sorted):
        per_app_ts[str(app)] = ts_sorted[apps_sorted == app]

    out: Dict[int, List[np.ndarray]] = {int(w): [] for w in windows_sec}
    for snap in snapshots:
        base = int(snap.anchor_ts_ns)
        for w in windows_sec:
            lo_ts = base - int(w) * NS_PER_SEC
            counts = np.zeros(len(snap.apps), dtype=np.int32)
            for i, app in enumerate(snap.apps):
                arr = per_app_ts.get(app)
                if arr is None or arr.size == 0:
                    continue
                lo = int(np.searchsorted(arr, lo_ts, side="left"))
                hi = int(np.searchsorted(arr, base, side="left"))
                if hi > lo:
                    counts[i] = hi - lo
            out[int(w)].append(counts)
    return out


def compute_recency_ranks(snapshots: List[BGSnapshot]) -> List[np.ndarray]:
    """Per snapshot, return float ranks in [0, 1].

    Rank 0.0 = most recently backgrounded (smallest `time_in_bg_sec`).
    Rank 1.0 = oldest in B(t). For |B(t)| == 1 the single app gets rank 0.0.
    Apps tied in time_in_bg get tied (averaged) ranks.
    """
    out: List[np.ndarray] = []
    for snap in snapshots:
        n = len(snap.apps)
        if n == 0:
            out.append(np.zeros(0, dtype=np.float32))
            continue
        if n == 1:
            out.append(np.zeros(1, dtype=np.float32))
            continue
        # rankdata-equivalent with average ties
        order = np.argsort(snap.time_in_bg_sec, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i + 1
            while (j < n and
                   float(snap.time_in_bg_sec[order[j]]) ==
                   float(snap.time_in_bg_sec[order[i]])):
                j += 1
            avg = (i + j - 1) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
            i = j
        out.append((ranks / max(1, n - 1)).astype(np.float32))
    return out


def flatten_to_rows(
    snapshots: List[BGSnapshot],
    labels_by_horizon: Dict[int, List[np.ndarray]],
    vocab: Dict[str, int],
    horizons_sec: Sequence[int] = (300, 600),
    rolling_counts_by_window: Dict[int, List[np.ndarray]] = None,
    recency_ranks: List[np.ndarray] = None,
    rolling_windows_sec: Sequence[int] = ROLLING_WINDOWS_SEC_DEFAULT,
) -> pd.DataFrame:
    """Turn snapshots + labels into one row per (anchor, app ∈ B(t)).

    Optional inputs:
      rolling_counts_by_window — produced by compute_rolling_fg_counts; emitted as
                                 columns ``fg_count_last_<W>s``.
      recency_ranks            — produced by compute_recency_ranks; emitted as
                                 column ``recency_rank_in_bg``.
    """
    from lib.v3.daypart import daypart_bin  # local import to avoid heavy deps

    rare_idx = vocab.get("<RARE>", 2)
    rows: List[dict] = []
    for a_id, snap in enumerate(snapshots):
        ts = pd.Timestamp(int(snap.anchor_ts_ns), unit="ns")
        hr = int(ts.hour)
        wd = int(ts.weekday())
        dp = int(daypart_bin(hr, wd))
        last_fg_idx = int(vocab.get(snap.last_fg_app, rare_idx)) if snap.last_fg_app else 0
        # NEW v2 anchor-level scalars
        prev_killed_app_idx = int(vocab.get(snap.last_kill_app, rare_idx)) if snap.last_kill_app else 0
        bg_size = len(snap.apps)
        for i, app in enumerate(snap.apps):
            a_idx = int(vocab.get(app, rare_idx))
            row = {
                "anchor_id": a_id,
                "anchor_ts_ns": int(snap.anchor_ts_ns),
                "anchor_hour": hr,
                "anchor_weekday": wd,
                "anchor_daypart": dp,
                "last_fg_app_idx": last_fg_idx,
                "bg_set_size": bg_size,
                "app_label": app,
                "app_idx": a_idx,
                "time_in_bg_sec": float(snap.time_in_bg_sec[i]),
                "time_since_fg_sec": float(snap.time_since_fg_sec[i]),
                "fg_count_today": int(snap.fg_count_today[i]),
                "last_fg_loc_id": int(snap.last_fg_loc_id[i]),
                "last_fg_daypart": int(snap.last_fg_daypart[i]),
                # NEW v2: anchor-level scalars (broadcast across rows in this anchor)
                "time_since_screen_on_sec": float(snap.time_since_screen_on_sec),
                "prev_killed_app_idx": prev_killed_app_idx,
                "prev_killed_age_sec": float(snap.last_kill_age_sec),
                "bg_recency_min_sec": float(snap.bg_recency_min_sec),
            }
            if rolling_counts_by_window is not None:
                for w in rolling_windows_sec:
                    arr = rolling_counts_by_window.get(int(w))
                    if arr is not None:
                        row[f"fg_count_last_{int(w)}s"] = int(arr[a_id][i])
                    else:
                        row[f"fg_count_last_{int(w)}s"] = 0
            if recency_ranks is not None:
                row["recency_rank_in_bg"] = float(recency_ranks[a_id][i])
            for h in horizons_sec:
                row[f"y_{int(h)}"] = int(labels_by_horizon[int(h)][a_id][i])
            rows.append(row)
    return pd.DataFrame(rows)
