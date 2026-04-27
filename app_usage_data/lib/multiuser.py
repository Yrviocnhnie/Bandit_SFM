"""Multi-user data utilities.

Adapts the single-user pipeline (lib/data.py) to the multi-user data under
/data00/ruiqing/app_forecasting/data/cleaned/. The multi-user files lack the
device_state_scene / device_state_networktype columns, which we fill with null
markers so the existing encoder (lib/features.py) sees consistent input.

Per-user output: train/val/test parquets + vocab.json + (optionally) Markov
prior + profile_stats. Stored under artifacts/multiuser/<short_uid>/.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from lib import data as D
from lib import features as F
from lib import baselines as BL


REQUIRED_RAW_COLS = (
    "raw_row_id",
    "event_ts",
    "app_label_clean",
    "name_norm",
    "is_app_usage_event",
    "is_target_event",
    "process_name_norm",
    "bundle_name_norm",
    "has_any_app_identity",
    "device_state_update_payload",
    "seconds_since_prev_event",
    "seconds_to_next_event",
)
DEFAULT_DATA_ROOT = "/data00/ruiqing/app_forecasting/data/cleaned"


def short_uid(path: str | Path) -> str:
    base = os.path.basename(str(path))
    m = re.match(r"([0-9A-Fa-f]+)_cleaned\.xlsx", base)
    return m.group(1)[:12] if m else base.replace(".xlsx", "")


def list_user_files(root: str | Path = DEFAULT_DATA_ROOT) -> List[Tuple_str_str]:
    """Return [(set_name, file_path), ...] for every user XLSX under root.
    set_name is the subdirectory ('M_beta_Top30' or 'top2000')."""
    out = []
    rp = Path(root)
    for sub in sorted(p for p in rp.iterdir() if p.is_dir()):
        for f in sorted(sub.glob("*_cleaned.xlsx")):
            out.append((sub.name, str(f)))
    return out


from typing import Tuple as Tuple_str_str


def load_user_xlsx(path: str | Path) -> pd.DataFrame:
    """Load a multi-user XLSX, fill missing scene/network columns, return ready
    for `lib.data.dedup` + `enrich`."""
    df = pd.read_excel(path, sheet_name="final_compact", engine="openpyxl")
    # Some columns the single-user pipeline expects may be missing; add as None
    for col in [
        "device_state_scene",
        "device_state_networktype",
        "device_state_has_wifi_info",
        "device_state_has_cellular_info",
        "device_state_has_data_switch_info",
        "device_state_has_common_info",
        "device_state_source_event",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    df = df.dropna(subset=["event_ts"]).reset_index(drop=True)
    for c in ("is_app_usage_event", "is_target_event"):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    return df


def per_user_split(df: pd.DataFrame, val_days: int = 3, test_days: int = 3,
                   embargo_minutes: int = 60):
    """Chronological split: last `test_days` = test; preceding `val_days` = val;
    rest = train. Embargo of `embargo_minutes` between adjacent splits."""
    ts = pd.to_datetime(df["event_ts"])
    end = ts.max()
    test_start = end - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    embargo = pd.Timedelta(minutes=embargo_minutes)
    train_mask = ts < (val_start - embargo)
    val_mask = (ts >= val_start) & (ts < test_start)
    test_mask = ts >= test_start
    return (df.loc[train_mask].reset_index(drop=True),
            df.loc[val_mask].reset_index(drop=True),
            df.loc[test_mask].reset_index(drop=True))


def build_user_vocab(train_df: pd.DataFrame, min_count: int = 3) -> dict:
    """Build a per-user vocab from train target events. Apps with <min_count are <RARE>."""
    mask = train_df["is_target_event"].astype(bool).values
    apps = train_df.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str)
    counts = apps.value_counts()
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2}
    for app, c in counts.items():
        if c < min_count or app in vocab:
            continue
        vocab[app] = len(vocab)
    return vocab


def prep_user(path: str | Path, out_dir: Path,
              val_days: int = 3, test_days: int = 3,
              min_vocab_count: int = 3) -> dict:
    """Process one user's XLSX into ready-to-train parquets + vocab.

    Returns a small stats dict used by the runner.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_xlsx_with_padding(str(path))
    raw = D.dedup(raw)
    enriched = D.enrich(raw)

    train, val, test = per_user_split(enriched, val_days=val_days, test_days=test_days)

    vocab = build_user_vocab(train, min_count=min_vocab_count)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(splits_dir / "train.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet")
    test.to_parquet(splits_dir / "test.parquet", index=False)

    return {
        "n_total": int(len(train) + len(val) + len(test)),
        "train_targets": int(train["is_target_event"].astype(bool).sum()),
        "val_targets": int(val["is_target_event"].astype(bool).sum()),
        "test_targets": int(test["is_target_event"].astype(bool).sum()),
        "vocab_size": len(vocab),
        "first_ts": str(min(train["event_ts"].min(), val["event_ts"].min(), test["event_ts"].min())),
        "last_ts": str(max(train["event_ts"].max(), val["event_ts"].max(), test["event_ts"].max())),
    }


def load_xlsx_with_padding(path: str) -> pd.DataFrame:
    """Like lib.data.load_xlsx but tolerant to missing optional columns."""
    df = pd.read_excel(path, sheet_name="final_compact", engine="openpyxl")
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    for col in [
        "device_state_scene",
        "device_state_networktype",
        "device_state_has_wifi_info",
        "device_state_has_cellular_info",
        "device_state_has_data_switch_info",
        "device_state_has_common_info",
        "device_state_source_event",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.dropna(subset=["event_ts"]).reset_index(drop=True)
    for c in ("is_app_usage_event", "is_target_event"):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    return df
