"""v3 feature builders: per-token category/loc and extended profile vector."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from lib.v2.global_features import PROFILE_DIM as V2_PROFILE_DIM

from .categories import NUM_CATEGORIES
from .daypart import NUM_DAYPARTS, daypart_onehot
from .window_rollups import (
    DEFAULT_WINDOWS_SEC,
    PER_WINDOW_DIM,
    build_window_aggregates,
)

PAD_CAT_IDX = 0
PAD_LOC_IDX = 0


def profile_v3_dim(use_daypart: bool = True,
                   use_windows: bool = True,
                   n_windows: int = len(DEFAULT_WINDOWS_SEC)) -> int:
    dim = V2_PROFILE_DIM
    if use_daypart:
        dim += NUM_DAYPARTS
    if use_windows:
        dim += n_windows * PER_WINDOW_DIM
    return dim


def extend_profile_v2_to_v3(
    v2_profile: np.ndarray,
    anchor_hours: np.ndarray,
    anchor_weekdays: np.ndarray,
    anchor_ts_ns: np.ndarray,
    stream_ts_ns: np.ndarray,
    stream_app_idx: np.ndarray,
    stream_cat_idx: np.ndarray,
    stream_dur_s: np.ndarray,
    windows_sec: Sequence[int] = DEFAULT_WINDOWS_SEC,
    use_daypart: bool = True,
    use_windows: bool = True,
) -> np.ndarray:
    parts = [v2_profile.astype(np.float32, copy=False)]
    if use_daypart:
        parts.append(daypart_onehot(anchor_hours, anchor_weekdays))
    if use_windows:
        parts.append(build_window_aggregates(
            anchor_ts_ns=anchor_ts_ns,
            stream_ts_ns=stream_ts_ns,
            stream_app_idx=stream_app_idx,
            stream_cat_idx=stream_cat_idx,
            stream_dur_s=stream_dur_s,
            windows_sec=windows_sec,
        ))
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_history_cat_loc_for_targets(
    enc,
    cat_per_row: np.ndarray,
    loc_per_row: np.ndarray,
    history_k: int = 16,
):
    """Return (history_category, history_loc) aligned with v1
    `build_history_for_targets`: last `history_k` rows strictly before each
    target row, filtered to the same session_id."""
    pos_targets = np.nonzero(enc.is_target)[0]
    M = pos_targets.size
    h_cat = np.full((M, history_k), PAD_CAT_IDX, dtype=np.int64)
    h_loc = np.full((M, history_k), PAD_LOC_IDX, dtype=np.int64)
    for r in range(M):
        pos = int(pos_targets[r])
        sid = int(enc.session_id[pos])
        start = max(0, pos - history_k)
        idxs = np.arange(start, pos)
        if len(idxs) > 0:
            idxs = idxs[enc.session_id[idxs] == sid]
        take = len(idxs)
        if take > 0:
            h_cat[r, -take:] = cat_per_row[idxs]
            h_loc[r, -take:] = loc_per_row[idxs]
    return h_cat, h_loc


def build_long_cat_loc_for_targets(
    enc,
    cat_per_row: np.ndarray,
    loc_per_row: np.ndarray,
    k_long: int = 64,
):
    """Parallel to `lib.v2.global_features.build_long_history_for_targets`:
    last `k_long` events strictly before each target, across session
    boundaries."""
    target_pos = np.nonzero(enc.is_target)[0]
    M = len(target_pos)
    h_cat = np.full((M, k_long), PAD_CAT_IDX, dtype=np.int64)
    h_loc = np.full((M, k_long), PAD_LOC_IDX, dtype=np.int64)
    for r in range(M):
        pos_i = int(target_pos[r])
        start = max(0, pos_i - k_long)
        take = pos_i - start
        if take > 0:
            h_cat[r, -take:] = cat_per_row_slice(cat_per_row, start, pos_i)
            h_loc[r, -take:] = loc_per_row_slice(loc_per_row := None, start, pos_i) if False else loc_per_row[start:pos_i]
    return {"long_category": h_cat, "long_loc": h_loc}


def build_history_cat_loc_for_anchors(
    enc,
    anchor_ts: np.ndarray,
    cat_per_row: np.ndarray,
    loc_per_row: np.ndarray,
    history_k: int = 16,
):
    ts_i64 = enc.ts.astype("datetime64[ns]").astype(np.int64)
    anchor_i64 = pd.to_datetime(anchor_ts).to_numpy().astype("datetime64[ns]").astype(np.int64)
    M = len(anchor_ts)
    h_cat = np.full((M, history_k), PAD_CAT_IDX, dtype=np.int64)
    h_loc = np.full((M, history_k), PAD_LOC_IDX, dtype=np.int64)
    ins = np.searchsorted(ts_i64 := ts_i64_from_enc(enc), anchor_i64, side="left")
    for i in range(M):
        ip = int(ins[i])
        start = max(0, ip - history_k)
        take = ip - start
        if take > 0:
            h_cat[i, -take:] = cat_per_row[start:ip]
            h_loc[i, -take:] = loc_per_row[start:ip]
    return h_cat, h_loc


def build_long_cat_loc_for_anchors(
    enc,
    anchor_ts: np.ndarray,
    cat_per_row: np.ndarray,
    loc_per_row: np.ndarray,
    k_long: int = 64,
):
    ts_i64 = enc.ts.astype("datetime64[ns]").astype(np.int64)
    anchor_i64 = pd.to_datetime(anchor_ts).to_numpy().astype("datetime64[ns]").astype(np.int64)
    M = len(anchor_ts)
    h_cat = np.full((M, k_long), PAD_CAT_IDX, dtype=np.int64)
    h_loc = np.full((M, k_long), PAD_LOC_IDX, dtype=np.int64)
    ins = np.searchsorted(ts_i64, anchor_i64, side="left")
    for i in range(M):
        ip = int(ins[i])
        start = max(0, ip - k_long)
        take = ip - start
        if take > 0:
            h_cat[i, -take:] = cat_per_row[start:ip]
            h_loc[i, -take:] = loc_per_row[start:ip]
    return {"long_category": h_cat, "long_loc": h_loc}


def ts_i64_from_enc(enc) -> np.ndarray:
    return enc.ts.astype("datetime64[ns]").astype(np.int64)


def cat_per_row_slice(cat_per_row, start, end):
    return cat_per_row[start:end]


def loc_per_row_slice(loc_per_row, start, end):
    return loc_per_row[start:end]


PAD_CAT_IDX = 0
PAD_LOC_IDX = 0
