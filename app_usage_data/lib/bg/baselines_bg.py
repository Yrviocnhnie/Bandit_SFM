"""Task-C baselines (zero training).

Each baseline takes a Task-C parquet (one row per (anchor, app in B(t))) and
adds a ``score_<baseline>`` column. Higher = more kill-worthy.

Implemented:
  * LRU            -- score = time_since_last_fg_sec  (older -> more kill-worthy)
  * LFU-hour       -- score = 1 - HourMFU.P(a | anchor_hour)
  * Markov-inverse -- score = 1 - P(a | last_fg_app) from train-only Markov
  * TaskB-inverse  -- score = 1 - sigmoid(TaskBModelV3 logit) (loaded from R6)
  * Random         -- score ~ Uniform(0,1) per row, seeded
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------- cheap closed-form baselines ----------------

def add_lru_score(df: pd.DataFrame, col: str = "score_lru") -> pd.DataFrame:
    """LRU: the older the last FG, the higher the kill-score."""
    tfg = df["time_since_fg_sec"].to_numpy(dtype=np.float64)
    # time_since_fg_sec = -1 when we've never seen it go FG in our record; treat
    # as maximally kill-worthy (never used recently).
    tfg = np.where(tfg < 0, 1e9, tfg)
    df = df.copy()
    df[col] = tfg
    return df


def add_time_in_bg_score(df: pd.DataFrame, col: str = "score_tibg") -> pd.DataFrame:
    """Even simpler: the longer it's been in BG, the more kill-worthy."""
    df = df.copy()
    df[col] = df["time_in_bg_sec"].to_numpy(dtype=np.float64)
    return df


def add_random_score(df: pd.DataFrame, seed: int = 7, col: str = "score_random") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df[col] = rng.random(len(df))
    return df


def add_lfu_hour_score(
    df: pd.DataFrame,
    hour_freq: np.ndarray,   # (24, V) fit on train
    col: str = "score_lfu_hour",
) -> pd.DataFrame:
    hrs = df["anchor_hour"].to_numpy().astype(int) % 24
    apps = df["app_idx"].to_numpy().astype(int)
    p = hour_freq[hrs, apps]
    df = df.copy()
    df[col] = 1.0 - p
    return df


def add_markov_inverse(
    df: pd.DataFrame,
    markov_prior: np.ndarray,  # (V, V) P(next | prev) - train-fit
    col: str = "score_markov_inv",
) -> pd.DataFrame:
    last = df["last_fg_app_idx"].to_numpy().astype(int)
    apps = df["app_idx"].to_numpy().astype(int)
    # clamp out-of-range indices defensively
    V = markov_prior.shape[0]
    last = np.clip(last, 0, V - 1)
    apps = np.clip(apps, 0, V - 1)
    p = markov_prior[last, apps]
    df = df.copy()
    df[col] = 1.0 - p
    return df


# ---------------- heavy-baseline: TaskB-inverse using v3 R6 ----------------

def add_taskb_inverse_score(
    df: pd.DataFrame,
    taskb_probs: np.ndarray,   # (n_anchors, V) sigmoid(TaskBModelV3)
    col: str = "score_taskb_inv",
) -> pd.DataFrame:
    """Caller is responsible for producing a per-anchor×app probability matrix
    using the v3 model (see scripts/31_run_baselines_bg.py for the glue)."""
    df = df.copy()
    idx_anchor = df["anchor_id"].to_numpy().astype(int)
    idx_app = df["app_idx"].to_numpy().astype(int)
    df[col] = 1.0 - taskb_probs[idx_anchor, idx_app]
    return df


import numpy as np
