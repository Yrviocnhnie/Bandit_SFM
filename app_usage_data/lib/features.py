"""Per-event feature encoder and batch-builder utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .data import (
    EVENT_TYPES,
    NET_CATS,
    NET_IDX,
    SCENE_CATS,
    SCENE_IDX,
    SESSION_MAX_EVENTS,
    app_to_idx,
)

HOUR_FOURIER_DIM = 4
NUMERIC_FEAT_DIM = (
    len(EVENT_TYPES)
    + HOUR_FOURIER_DIM
    + 7
    + 1
    + 1
    + len(SCENE_CATS)
    + len(NET_CATS)
    + 1
    + 1
)


def fourier_hour(hour: np.ndarray) -> np.ndarray:
    h = np.asarray(hour, dtype=float)
    return np.stack(
        [
            np.sin(2 * np.pi * h / 24.0),
            np.cos(2 * np.pi * h / 24.0),
            np.sin(2 * np.pi * h / 12.0),
            np.cos(2 * np.pi * h / 12.0),
        ],
        axis=-1,
    ).astype(np.float32)


def _one_hot(values: np.ndarray, cats: list, cat_idx: Dict[int, int]) -> np.ndarray:
    n = len(values)
    out = np.zeros((n, len(cats)), dtype=np.float32)
    for i, v in enumerate(values):
        out[i, cat_idx.get(int(v), 0)] = 1.0
    return out


@dataclass
class EncodedSplit:
    app_idx: np.ndarray
    feat: np.ndarray
    is_target: np.ndarray
    app_label: np.ndarray
    ts: np.ndarray
    session_id: np.ndarray
    hour: np.ndarray
    weekday: np.ndarray


def fit_dt_scaler(df: pd.DataFrame) -> dict:
    x = np.log1p(np.clip(df["sec_since_prev"].to_numpy(float), 0.0, None))
    return {"mean": float(x.mean()), "std": float(max(x.std(), 1e-6))}


def encode_events(df: pd.DataFrame, vocab: dict, dt_mean: float = 0.0, dt_std: float = 1.0) -> EncodedSplit:
    n = len(df)

    app_idx = np.array(
        [app_to_idx(a, vocab) for a in df["app_label_clean"].to_numpy()],
        dtype=np.int64,
    )

    et_oh = np.zeros((n, len(EVENT_TYPES)), dtype=np.float32)
    et_oh[np.arange(n), df["evt_type_idx"].to_numpy().astype(int)] = 1.0

    hours = df["hour"].to_numpy().astype(int)
    hour_f = fourier_hour(hours)

    weekday = df["weekday"].to_numpy().astype(int)
    wd_oh = np.zeros((n, 7), dtype=np.float32)
    wd_oh[np.arange(n), np.clip(weekday, 0, 6)] = 1.0

    dt_raw = np.clip(df["sec_since_prev"].to_numpy(float), 0.0, None)
    log_dt = ((np.log1p(dt_raw) - dt_mean) / max(dt_std, 1e-6)).astype(np.float32)
    log_dt_col = log_dt.reshape(-1, 1)

    pos = np.clip(
        df["session_position"].to_numpy().astype(np.float32) / float(SESSION_MAX_EVENTS),
        0.0,
        1.0,
    )
    pos_col = pos.reshape(-1, 1)

    scene_oh = _one_hot(df["scene"].to_numpy().astype(int), SCENE_CATS, SCENE_IDX)
    net_oh = _one_hot(df["net"].to_numpy().astype(int), NET_CATS, NET_IDX)
    miss = df["is_missing_state"].to_numpy().astype(np.float32).reshape(-1, 1)
    screen = df["screen_on"].to_numpy().astype(np.float32).reshape(-1, 1)

    feat = np.concatenate(
        [et_oh, hour_f, wd_oh, log_dt_col, pos_col, scene_oh, net_oh, miss, screen],
        axis=1,
    ).astype(np.float32)
    assert feat.shape[1] == NUMERIC_FEAT_DIM, f"feat dim {feat.shape[1]} != {NUMERIC_FEAT_DIM}"

    return EncodedSplit(
        app_idx=app_idx,
        feat=feat,
        is_target=df["is_target_event"].to_numpy().astype(bool),
        app_label=df["app_label_clean"].fillna("<UNK>").astype(str).to_numpy(),
        ts=df["event_ts"].to_numpy().astype("datetime64[ns]"),
        session_id=df["session_id"].to_numpy().astype(np.int64),
        hour=df["hour"].to_numpy().astype(np.int64),
        weekday=df["weekday"].to_numpy().astype(np.int64),
    )


def build_history_for_targets(enc: EncodedSplit, history_k: int = 16, pad_idx: int = 0) -> Dict[str, np.ndarray]:
    D = enc.feat.shape[1]
    pos_targets = np.nonzero(enc.is_target)[0]
    M = pos_targets.size

    h_app = np.full((M, history_k), pad_idx, dtype=np.int64)
    h_feat = np.zeros((M, history_k, D), dtype=np.float32)
    h_mask = np.zeros((M, history_k), dtype=bool)

    for r, pos in enumerate(pos_targets):
        sid = int(enc.session_id[pos])
        start = max(0, pos - history_k)
        idxs = np.arange(start, pos)
        idxs = idxs[enc.session_id[idxs] == sid]
        take = len(idxs)
        if take > 0:
            h_app[r, -take:] = enc.app_idx[idxs]
            h_feat[r, -take:, :] = enc.feat[idxs]
            h_mask[r, -take:] = True

    return {
        "history_app": h_app,
        "history_feat": h_feat,
        "history_mask": h_mask,
        "target_app": enc.app_idx[pos_targets],
        "target_label": enc.app_label[pos_targets],
        "target_hour": enc.hour[pos_targets],
        "target_weekday": enc.weekday[pos_targets],
        "target_ts": enc.ts[pos_targets],
        "target_hour_fourier": fourier_hour(enc.hour[pos_targets]),
    }


def build_history_for_anchors(
    enc: EncodedSplit, anchor_ts: np.ndarray, history_k: int = 16, pad_idx: int = 0
) -> Dict[str, np.ndarray]:
    D = enc.feat.shape[1]
    M = len(anchor_ts)
    h_app = np.full((M, history_k), pad_idx, dtype=np.int64)
    h_feat = np.zeros((M, history_k, D), dtype=np.float32)
    h_mask = np.zeros((M, history_k), dtype=bool)

    ts_i64 = enc.ts.astype("datetime64[ns]").astype(np.int64)
    anchor_i64 = pd.to_datetime(anchor_ts).to_numpy().astype("datetime64[ns]").astype(np.int64)
    ins = np.searchsorted(ts_i64, anchor_i64, side="left")

    for i, ip in enumerate(ins):
        start = max(0, int(ip) - history_k)
        take = int(ip) - start
        if take > 0:
            h_app[i, -take:] = enc.app_idx[start:int(ip)]
            h_feat[i, -take:, :] = enc.feat[start:int(ip)]
            h_mask[i, -take:] = True

    hours = np.array([pd.Timestamp(a).hour for a in anchor_ts], dtype=np.int64)
    wds = np.array([pd.Timestamp(a).weekday() for a in anchor_ts], dtype=np.int64)
    return {
        "history_app": h_app,
        "history_feat": h_feat,
        "history_mask": h_mask,
        "anchor_ts": pd.to_datetime(anchor_ts).to_numpy(),
        "anchor_hour": hours,
        "anchor_weekday": wds,
        "side_hour_fourier": fourier_hour(hours),
    }
