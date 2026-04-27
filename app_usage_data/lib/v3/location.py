"""v3 location proxy: parse `device_state_update_payload` into a stable loc_id.

Strategy:
  1. regex-extract `WifiSsid` (primary) or `CellId` (fallback) from each payload
  2. forward-fill within the same calendar day (devices stay on the same WiFi
     until a state change is logged)
  3. bucket into a train-only vocab of {`<PAD>`, `<NONE>`, `<OTHER>`, + top-K labels}
"""
from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PAD_LOC = "<PAD>"
NONE_LOC = "<NONE>"
OTHER_LOC = "<OTHER>"

_SSID_RE = re.compile(r'"?WifiSsid"?\s*[:=]\s*"([^"]+)"')
_CELL_RE = re.compile(r'"?CellId"?\s*[:=]\s*(\d+)')


def _parse_one_payload(payload) -> Optional[str]:
    if payload is None:
        return None
    if isinstance(payload, float) and np.isnan(payload):
        return None
    text = str(payload)
    if not text:
        return None
    m = _SSID_RE.search(text)
    if m:
        ssid = m.group(1).strip()
        if ssid and ssid.lower() not in ("null", "none", "unknown"):
            return f"wifi:{ssid}"
    m = _CELL_RE.search(text)
    if m:
        try:
            return f"cell:{int(m.group(1))}"
        except ValueError:
            return None
    return None


def _forward_fill_within_day(raw: np.ndarray, days: np.ndarray) -> np.ndarray:
    out = np.empty_like(raw)
    last = None
    last_day = None
    for i in range(len(raw)):
        d = days[i]
        if d != last_day:
            last_day = d
            last = None
        if raw[i] is not None:
            last = raw[i]
        out[i] = last
    return out


def _raw_labels(df: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(df), dtype=object)
    if "device_state_update_payload" not in df.columns:
        out[:] = None
        return out
    arr = df["device_state_update_payload"].to_numpy()
    for i, p in enumerate(arr):
        out[i] = _parse_one_payload(p)
    return out


def fit_location_vocab(df_train: pd.DataFrame, top_k: int = 15) -> Dict:
    raw = _raw_labels(df_train)
    days = pd.to_datetime(df_train["event_ts"]).dt.normalize().values
    filled = _forward_fill_within_day(raw, days)
    counts = Counter(x for x in filled if x is not None)
    top = [lab for lab, _ in counts.most_common(top_k)]
    vocab: Dict[str, int] = {PAD_LOC: 0, NONE_LOC: 1, OTHER_LOC: 2}
    for i, lbl in enumerate(top):
        vocab[lbl] = 3 + i
    return {
        "fit_split": "train",
        "vocab": vocab,
        "top_k": top_k,
        "top_counts": dict(counts.most_common(min(top_k * 3, 60))),
    }


def save_location_vocab(stats: Dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def load_location_vocab(path) -> Dict:
    with open(path, "rb") as f:
        stats = pickle.load(f)
    assert stats.get("fit_split") == "train", (
        f"location vocab must be fit on train; got fit_split={stats.get('fit_split')!r}"
    )
    return stats


def attach_loc_id(df: pd.DataFrame, loc_stats: Dict) -> np.ndarray:
    vocab = loc_stats["vocab"]
    none_idx = vocab[NONE_LOC]
    other_idx = vocab[OTHER_LOC]
    raw = _raw_labels(df)
    days = pd.to_datetime(df["event_ts"]).dt.normalize().values
    filled = _forward_fill_within_day(raw, days)
    out = np.full(len(filled), none_idx, dtype=np.int64)
    for i, label in enumerate(filled):
        if label is None:
            continue
        out[i] = vocab.get(label, other_idx)
    return out
