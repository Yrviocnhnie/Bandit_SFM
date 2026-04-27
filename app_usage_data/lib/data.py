"""Data loading, cleaning, sessionization, chronological split, and vocabulary.

Public API:
    load_xlsx(path) -> pd.DataFrame
    dedup(df) -> pd.DataFrame
    enrich(df) -> pd.DataFrame   # adds hour/weekday/session_id/evt_type/scene/net/screen
    build_vocab(df_train, min_count) -> Dict[str, int]
    split_chronological(df, train_end, val_end, embargo_minutes) -> Splits
    anchor_grid(df, stride_sec, hour_range) -> pd.DataFrame
    compute_window_gt(df, anchors, horizon_sec) -> List[Dict[str, int]]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
RARE_TOKEN = "<RARE>"
RESERVED_TOKENS = [PAD_TOKEN, UNK_TOKEN, RARE_TOKEN]

SESSION_IDLE_SEC = 300
SESSION_MAX_EVENTS = 32
RARE_MIN_COUNT = 5

EVENT_TYPES = ["TARGET", "BACKGROUND", "PAGE_SWITCH", "OTHER"]
EVT_IDX = {n: i for i, n in enumerate(EVENT_TYPES)}

SCENE_CATS = [-1, 1, 2, 3, 4]
SCENE_IDX = {s: i for i, s in enumerate(SCENE_CATS)}

NET_CATS = [-1, 0, 1, 2]
NET_IDX = {n: i for i, n in enumerate(NET_CATS)}


def load_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="final_compact", engine="openpyxl")
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    df = df.dropna(subset=["event_ts"]).reset_index(drop=True)
    for c in ("is_app_usage_event", "is_target_event"):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    return df


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    app_key = df["app_label_clean"].fillna("<NO_APP>").astype(str)
    key = pd.Series(list(zip(df["event_ts"], app_key, df["name_norm"])), index=df.index)
    counts = key.value_counts()
    df = df.copy()
    df["dup_count"] = key.map(counts).astype(int).values
    df["_k"] = key.values
    df = df.drop_duplicates(subset="_k", keep="first").drop(columns=["_k"])
    df = df.sort_values("event_ts", kind="mergesort").reset_index(drop=True)
    return df


def _event_type(name: str, is_target: bool) -> str:
    if is_target:
        return "TARGET"
    if name == "APP_BACKGROUND":
        return "BACKGROUND"
    if name == "ABILITY_OR_PAGE_SWITCH":
        return "PAGE_SWITCH"
    return "OTHER"


def _split_long_sessions(sid: np.ndarray, max_events: int) -> np.ndarray:
    out = np.empty_like(sid)
    cur = -1
    prev = -1
    count = 0
    for i, s in enumerate(sid):
        if int(s) != prev:
            cur += 1
            prev = int(s)
            count = 0
        if count >= max_events:
            cur += 1
            count = 0
        out[i] = cur
        count += 1
    return out


def _screen_state(name_norm: pd.Series) -> np.ndarray:
    state = np.ones(len(name_norm), dtype=np.int8)
    cur = 1
    for i, n in enumerate(name_norm.to_numpy()):
        if n == "SCREENON_EVENT":
            cur = 1
        elif n == "SCREENOFF_EVENT":
            cur = 0
        state[i] = cur
    return state


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["event_ts"])
    gap = np.concatenate([[0.0], (ts.values[1:] - ts.values[:-1]) / np.timedelta64(1, "s")])
    gap = np.clip(gap, 0.0, 3600.0)
    df["sec_since_prev"] = gap
    df["log1p_dt"] = np.log1p(gap)

    df["hour"] = ts.dt.hour.astype(int).values
    df["minute"] = ts.dt.minute.astype(int).values
    df["weekday"] = ts.dt.weekday.astype(int).values
    df["day"] = ts.dt.normalize().values

    raw_sid = (gap > SESSION_IDLE_SEC).astype(int).cumsum()
    df["session_id"] = _split_long_sessions(raw_sid, SESSION_MAX_EVENTS).astype(int)
    df["session_position"] = df.groupby("session_id").cumcount().astype(int)

    is_target = df["is_target_event"].values
    df["evt_type"] = [_event_type(n, bool(t)) for n, t in zip(df["name_norm"].values, is_target)]
    df["evt_type_idx"] = df["evt_type"].map(EVT_IDX).astype(int)

    for src, dst, default in (
        ("device_state_scene", "scene", -1),
        ("device_state_networktype", "net", -1),
    ):
        s = pd.to_numeric(df[src], errors="coerce") if src in df.columns else pd.Series([np.nan] * len(df))
        s = s.groupby(df["day"].values).ffill()
        df[dst] = s.fillna(default).astype(int).values
    df["is_missing_state"] = (df["scene"] == -1).astype(int).values
    df["screen_on"] = _screen_state(df["name_norm"])
    return df


def build_vocab(df_train: pd.DataFrame, min_count: int = RARE_MIN_COUNT) -> Dict[str, int]:
    mask = df_train["is_target_event"].values.astype(bool)
    apps = df_train.loc[mask, "app_label_clean"].fillna(UNK_TOKEN)
    counts = apps.value_counts()
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, RARE_TOKEN: 2}
    for app, c in counts.items():
        if c < min_count:
            continue
        if app in vocab:
            continue
        vocab[app] = len(vocab)
    return vocab


def app_to_idx(app, vocab: Dict[str, int]) -> int:
    if app is None or (isinstance(app, float) and np.isnan(app)):
        return vocab[UNK_TOKEN]
    return vocab.get(str(app), vocab[RARE_TOKEN])


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_chronological(
    df: pd.DataFrame,
    train_end: str = "2026-04-03",
    val_end: str = "2026-04-08",
    embargo_minutes: int = 60,
) -> Splits:
    ts = pd.to_datetime(df["event_ts"])
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    embargo = pd.Timedelta(minutes=embargo_minutes)
    train_mask = ts < (train_end_ts - embargo)
    val_mask = (ts >= train_end_ts) & (ts < val_end_ts)
    test_mask = ts >= val_end_ts
    return Splits(
        train=df.loc[train_mask].reset_index(drop=True),
        val=df.loc[val_mask].reset_index(drop=True),
        test=df.loc[test_mask].reset_index(drop=True),
    )


def anchor_grid(df: pd.DataFrame, stride_sec: int = 300,
                hour_start: int = 6, hour_end: int = 24) -> pd.DataFrame:
    ts = pd.to_datetime(df["event_ts"])
    days = pd.DatetimeIndex(ts.dt.normalize().unique()).sort_values()
    anchors: List[pd.Timestamp] = []
    for d in days:
        start = d + pd.Timedelta(hours=hour_start)
        end = d + pd.Timedelta(hours=hour_end)
        cur = start
        while cur < end:
            anchors.append(cur)
            cur += pd.Timedelta(seconds=stride_sec)
    return pd.DataFrame({"anchor_ts": pd.to_datetime(anchors)})


def compute_window_gt(
    df: pd.DataFrame,
    anchors: pd.DataFrame,
    horizon_sec: int = 900,
) -> List[Dict[str, int]]:
    ts = pd.to_datetime(df["event_ts"]).to_numpy()
    is_target = df["is_target_event"].to_numpy().astype(bool)
    apps = df["app_label_clean"].fillna(UNK_TOKEN).astype(str).to_numpy()
    target_ts = ts[is_target]
    target_apps = apps[is_target]
    order = np.argsort(target_ts)
    target_ts = target_ts[order]
    target_apps = target_apps[order]

    anchor_ts = pd.to_datetime(anchors["anchor_ts"]).to_numpy()
    # ensure sorted
    aord = np.argsort(anchor_ts)
    anchor_ts_sorted = anchor_ts[aord]
    horizon_td = np.timedelta64(int(horizon_sec), "s")

    out_sorted: List[Dict[str, int]] = [None] * len(anchor_ts_sorted)
    n = len(target_ts)
    left = 0
    for i, a in enumerate(anchor_ts_sorted):
        while left < n and target_ts[left] < a:
            left += 1
        right = left
        end = a + horizon_td
        while right < n and target_ts[right] < end:
            right += 1
        win = target_apps[left:right]
        if len(win) == 0:
            out_sorted_entry = {}
        else:
            vals, counts = np.unique(win, return_counts=True)
            out_sorted_entry = {str(v): int(c) for v, c in zip(vals, counts)}
        out_sorted[i] = out_sorted_entry

    # Undo sort
    out = [None] * len(anchors)
    for i_sorted, i_orig in enumerate(aord):
        out[int(i_orig)] = out_sorted[i_sorted]
    return out
