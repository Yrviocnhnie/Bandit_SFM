"""Feature / label builders for Task C.

Produces per-(anchor, app) rows from `BGSnapshot` lists. The v3 profile (153 d)
is NOT baked into the parquet — we rebuild that at training time from the
existing `lib.v2.global_features` + `lib.v3.*` helpers so Task C stays in sync
with v3 feature logic without duplicating it on disk.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .background_state import BGSnapshot, NS_PER_SEC


HORIZONS_SEC_DEFAULT: Sequence[int] = (300, 600)


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


def flatten_to_rows(
    snapshots: List[BGSnapshot],
    labels_by_horizon: Dict[int, List[np.ndarray]],
    vocab: Dict[str, int],
    horizons_sec: Sequence[int] = HORIZONS_SEC_DEFAULT if False else (300, 600),
) -> pd.DataFrame:
    """Turn snapshots + labels into one row per (anchor, app ∈ B(t))."""
    from lib.v3.daypart import daypart_bin  # local import avoids heavy deps

    rare_idx = vocab.get("<RARE>", 2)
    rows: List[dict] = []
    for a_id, snap in enumerate(snapshots):
        ts = pd.Timestamp(int(snap.anchor_ts_ns), unit="ns")
        hr = int(ts.hour)
        wd = int(ts.weekday())
        dp = int(daypart_bin(hr, wd))
        last_fg_idx = int(vocab.get(snap.last_fg_app, rare_idx := vocab.get("<RARE>", 2))) if snap.last_fg_app else 0
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
            }
            for h in horizons_sec:
                row_key = f"y_{int(h)}"
                row[row_key] = int(labels_by_horizon[int(h)][a_id := a_id][i])
            rows.append(row)
    return pd.DataFrame(rows)


HORIZONS_SEC_DEFAULT = (300, 600)  # re-export
