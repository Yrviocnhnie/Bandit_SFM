"""Baselines for next-app (Task A) and 15-min window (Task B) prediction.

Each baseline returns a score vector of shape (V,) per query. Scores are probabilities
(row sum to 1 excluding reserved tokens 0,1,2) so they can be used for both:

  Task A: argmax / topk for next-app prediction
  Task B: feed through multi-label top-K set prediction (same ranking; no count estimate)

Reserved token indices (0=PAD, 1=UNK, 2=RARE) always get zero score so they never win top-K.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


RESERVED_END = 3  # indices [0..2] are reserved


def _zero_reserved(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    x[:RESERVED_END] = 0.0
    return x


def _norm(x: np.ndarray) -> np.ndarray:
    x = _zero_reserved(x)
    s = x.sum()
    return (x / s) if s > 0 else x


class MFU:
    """Most-Frequently-Used (global target-app frequency)."""

    def __init__(self, vocab_size: int):
        self.vocab_size = int(vocab_size)
        self.probs = np.zeros(vocab_size, dtype=np.float32)

    def fit(self, target_app_indices: np.ndarray) -> "MFU":
        counts = np.bincount(np.asarray(target_app_indices), minlength=self.vocab_size).astype(np.float32)
        self.probs = _norm(counts)
        return self

    def predict_task_a(self, **_unused) -> np.ndarray:
        return self.probs.copy()

    def predict_task_b(self, **_unused) -> np.ndarray:
        return self.probs.copy()


class MRU:
    """Most-Recently-Used: predict the last target-type app seen in history."""

    def __init__(self, vocab_size: int):
        self.vocab_size = int(vocab_size)

    def fit(self, *_a, **_k):
        return self

    def predict_task_a(self, history_app: np.ndarray, history_mask: np.ndarray, **_unused) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        for i in range(len(history_app) - 1, -1, -1):
            if not history_mask[i]:
                continue
            a = int(history_app[i])
            if a >= RESERVED_SIZE:
                scores[a] = 1.0
                return scores
        return scores

    def predict_task_b(self, history_app: np.ndarray, history_mask: np.ndarray, **_unused) -> np.ndarray:
        return self.predict_task_a(history_app=history_app, history_mask=history_mask)


RESERVED_SIZE = RESERVED_END


class HourMFU:
    """P(app | hour). Dirichlet-smoothed 24 × V table."""

    def __init__(self, vocab_size: int, alpha: float = 0.5):
        self.vocab_size = int(vocab_size)
        self.alpha = float(alpha)
        self.table = np.zeros((24, vocab_size), dtype=np.float32)

    def fit(self, hours: np.ndarray, apps: np.ndarray) -> "HourMFU":
        for h, a in zip(hours.astype(int), apps.astype(int)):
            if 0 <= h < 24 and a >= RESERVED_END and a < self.vocab_size:
                self.table[h, a] += 1.0
        # Dirichlet smoothing on valid vocab cells
        self.table[:, RESERVED_END:] += 0.5
        self.table[:, :RESERVED_END] = 0.0
        row_sum = self.table.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        self.table = self.table / row_sum
        return self

    def predict_task_a(self, hour: int) -> np.ndarray:
        return self.table[int(hour) % 24].copy()

    def predict_task_b(self, hour: int) -> np.ndarray:
        return self.predict_task_a(hour)


class Markov1:
    """First-order Markov transition matrix on target-event app sequence."""

    def __init__(self, vocab_size: int, alpha: float = 0.5):
        self.vocab_size = int(vocab_size)
        self.alpha = float(alpha)
        self.counts = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        self.prior = np.zeros(vocab_size, dtype=np.float32)

    def fit(self, target_sequence: np.ndarray) -> "Markov1":
        seq = np.asarray(target_sequence, dtype=np.int64)
        for i in range(1, len(seq)):
            a, b = int(seq[i - 1]), int(seq[i])
            if a >= RESERVED_SIZE and b >= RESERVED_SIZE and a < self.vocab_size and b < self.vocab_size:
                self.counts[a, b] += 1.0
        # Smooth rows on the valid vocab columns
        table = self.counts + self.alpha
        table[:, :RESERVED_SIZE] = 0.0
        row_sums = table.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.transition = (table / row_sums).astype(np.float32)
        # Global prior
        pr = np.bincount(seq, minlength=self.vocab_size).astype(np.float32)
        self.prior = _norm(pr)
        return self

    def predict_task_a(self, last_app: int | None) -> np.ndarray:
        if last_app is None or last_app < RESERVED_SIZE or last_app >= self.vocab_size:
            return self.prior.copy()
        return self.transition[int(last_app)].copy()

    def predict_task_b(self, last_app: int | None) -> np.ndarray:
        return self.predict_task_a(last_app)


# Expose RESERVED_SIZE for external use
RESERVED_SIZE = 3


__all__ = ["MFU", "MRU", "HourMFU", "Markov1", "RESERVED_SIZE"]
