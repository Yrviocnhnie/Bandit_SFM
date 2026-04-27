"""Task-C metrics: FalseKillRate, MemorySaveRate, PR-AUC, ROC-AUC, NDCG, Pareto.

All metrics are computed per anchor and then averaged with equal anchor weight.

Conventions:
  * ``score``: higher = more kill-worthy.
  * ``y``:    1 if the app was foregrounded in (t, t+H] (= bad to kill).
  * Eviction policy: top-K by ``score`` where K = ceil(r * |B(t)|).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


R_DEFAULT: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def _topk_kill_mask(scores: np.ndarray, r: float) -> np.ndarray:
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, min(int(np.ceil(r * n)), n))
    order = np.argsort(-scores, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    n = s.size
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float((precision * (recall - recall_prev)).sum())


def ndcg_at_k(relevance: np.ndarray, score: np.ndarray, k: int) -> float:
    rel = np.asarray(relevance, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    n = len(rel)
    if n == 0 or k <= 0 or rel.sum() == 0:
        return 0.0
    k_eff = min(k, n)
    discounts = 1.0 / np.log2(np.arange(2, 2 + k_eff))
    order = np.argsort(-s, kind="mergesort")[:k_eff]
    dcg = float((rel[order] * discounts).sum())
    ideal_order = np.argsort(-rel, kind="mergesort")[:k_eff]
    idcg = float((rel[ideal_order] * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    df: pd.DataFrame,
    score_col: str,
    y_col: str,
    r_values: Sequence[float] = R_DEFAULT,
) -> dict:
    """Compute all per-anchor metrics, aggregated with equal anchor weight."""
    r_list = list(r_values)
    fk = {r: [] for r in r_list}
    msr = {r: [] for r in r_list}
    roc_list: List[float] = []
    pr_list: List[float] = []
    ndcg_list: List[float] = []

    for _, g in df.groupby("anchor_id", sort=False):
        scores = g[score_col].to_numpy(dtype=np.float64)
        y = g[y_col].to_numpy(dtype=np.int64)
        n = len(y)
        if n == 0:
            continue
        for r in r_list:
            mask = _topk_kill_mask(scores, r)
            killed = int(mask.sum())
            fk_num = int((mask & (y == 1)).sum())
            save_num = int((mask & (y == 0)).sum())
            neg_total = int((y == 0).sum())
            fk[r].append(fk_num / killed if killed > 0 else 0.0)
            msr[r].append(save_num / neg_total if neg_total > 0 else 0.0)
        n_kill = int((y == 0).sum())
        if 0 < n_kill < n:
            kill_label = (1 - y).astype(np.int64)
            roc_val = roc_auc(kill_label, scores)
            pr_val = pr_auc(kill_label, scores)
            if not np.isnan(roc_val):
                roc_list.append(roc_val)
            if not np.isnan(pr_val):
                pr_list.append(pr_val)
        ndcg_list.append(ndcg_at_k((1 - y).astype(np.float64), scores,
                                   k=max(1, int(np.ceil(0.5 * n)))))

    return {
        "false_kill_rate": {str(r): float(np.mean(fk[r])) if fk[r] else 0.0 for r in r_values},
        "memory_save_rate": {str(r): float(np.mean(msr[r])) if msr[r] else 0.0 for r in r_values},
        "ndcg_half": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        "roc_auc_mean": float(np.mean(roc_list)) if roc_list else float("nan"),
        "pr_auc_mean": float(np.mean(pr_list)) if pr_list else float("nan"),
        "n_anchors": int(df["anchor_id"].nunique()),
        "n_rows": int(len(df)),
    }
