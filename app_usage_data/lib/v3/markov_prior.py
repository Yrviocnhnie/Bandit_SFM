"""v3 Markov-1 prior, fit on train targets only. Saved as a `(V, V)` log-prob table."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

RESERVED_END = 3  # <PAD>, <UNK>, <RARE>


def fit_markov_prior(df_train: pd.DataFrame, vocab: Dict[str, int], alpha: float = 0.5) -> Dict:
    V = len(vocab)
    mask = df_train["is_target_event"].astype(bool).values
    apps_raw = df_train.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str)
    rare_idx = vocab.get("<RARE>", vocab.get("<UNK>", 1))
    seq = np.array(
        [vocab.get(a, rare_idx) for a in apps_raw],
        dtype=np.int64,
    )

    counts = np.zeros((V, V), dtype=np.float64)
    for i in range(1, len(seq)):
        a = int(seq[i - 1])
        b = int(seq[i])
        if a >= RESERVED_END and b >= RESERVED_END:
            counts[a, b] += 1.0
    smoothed = counts + float(alpha)
    smoothed[:, :RESERVED_END] = 0.0
    smoothed[:RESERVED_END, :] = 0.0
    row_sum = smoothed.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    probs = smoothed / row_sum
    eps = 1e-8
    log_probs = np.log(np.clip(probs, eps, 1.0)).astype(np.float32)
    return {
        "fit_split": "train",
        "V": int(V),
        "alpha": float(alpha),
        "probs": probs.astype(np.float32),
        "log_prior": log_probs,
    }


RESERVED = RESERVED_END  # backward alias


def save_markov_prior(stats: Dict, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(stats, f)


def load_markov_prior(path) -> Dict:
    with open(path, "rb") as f:
        stats = pickle.load(f)
    assert stats.get("fit_split") == "train", (
        f"markov prior must be fit on train; got fit_split={stats.get('fit_split')!r}"
    )
    return stats
