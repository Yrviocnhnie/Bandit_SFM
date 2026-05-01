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


def compute_positive_only_metrics(
    df: pd.DataFrame,
    score_col: str,
    y_col: str,
    r_values: Sequence[float] = R_DEFAULT,
) -> dict:
    """Metrics computed using ONLY the positive-labeled (y == 1) rows.

    Three metrics:

    * **WAKR@r — Wanted-App Kill Rate at r** *(lower is better)*
      For each anchor with ≥ 1 positive: fraction of positives that fall in
      the top-⌈r·|B(t)|⌉ kill list. Anchor-mean.

      Differs from FK@r in the *denominator*:
        - FK@r   = positives_in_kill_list / size_of_kill_list
        - WAKR@r = positives_in_kill_list / total_positives_in_anchor
      → WAKR@r is the "missed safety" rate from the user's perspective:
      "of the apps you wanted, what fraction would the model have killed?"

    * **PosRank — mean rank-percentile of positives within B(t)**
      *(higher is better; range [0, 1])*
      For each positive (anchor, app):
        PosRank_per = (# apps in B(t) with strictly higher kill-score
                       than this positive  +  0.5 × # ties) / (|B(t)| − 1)
      Higher PosRank = positive sits low in kill priority = model thinks it
      is "safe to keep". Random baseline ≈ 0.5; perfect = 1.0.
      Scale-free (works for any kill-score units) and anchor-aware.

    * **PosScoreNorm — anchor-normalised mean kill-score on positives**
      *(lower is better; range [0, 1])*
      For each anchor: min-max scale scores to [0, 1]. Average that over the
      positives in the anchor. Then average across anchors. This is the
      "literal mean kill-score on positives" idea, made comparable across
      models with different score scales.

    Anchors with 0 positives or with |B(t)| < 2 are skipped (PosRank /
    PosScoreNorm are undefined). Anchors with all-equal scores are skipped
    from PosScoreNorm (zero-range).
    """
    r_list = list(r_values)
    pos_ranks: List[float] = []
    pos_score_norms: List[float] = []
    wakr = {r: [] for r in r_list}
    n_anchors_with_pos = 0

    for _, g in df.groupby("anchor_id", sort=False):
        scores = g[score_col].to_numpy(dtype=np.float64)
        y = g[y_col].to_numpy(dtype=np.int64)
        n = len(y)
        n_pos = int((y == 1).sum())
        if n_pos == 0:
            continue
        n_anchors_with_pos += 1

        # WAKR@r — fraction of positives ending up in top-r kill list
        order = np.argsort(-scores, kind="mergesort")
        for r in r_list:
            k = max(1, min(int(np.ceil(r * n)), n))
            top_k = order[:k]
            killed_pos = int((y[top_k] == 1).sum())
            wakr[r].append(killed_pos / n_pos)

        # PosRank and PosScoreNorm need |B(t)| ≥ 2
        if n < 2:
            continue
        s_min = float(scores.min())
        s_range = float(scores.max() - s_min)
        for i in np.where(y == 1)[0]:
            s_i = float(scores[i])
            others_mask = np.ones(n, dtype=bool)
            others_mask[i] = False
            others = scores[others_mask]
            n_higher = int((others > s_i).sum())
            n_equal  = int((others == s_i).sum())
            pos_ranks.append((n_higher + 0.5 * n_equal) / (n - 1))
            if s_range > 0:
                pos_score_norms.append((s_i - s_min) / s_range)

    return {
        "pos_rank_mean":      float(np.mean(pos_ranks))      if pos_ranks      else float("nan"),
        "pos_score_norm_mean": float(np.mean(pos_score_norms)) if pos_score_norms else float("nan"),
        "wakr": {str(r): float(np.mean(wakr_r)) if wakr_r else float("nan")
                 for r, wakr_r in wakr.items()},
        "n_positive_samples": int(len(pos_ranks)),
        "n_anchors_with_positives": int(n_anchors_with_pos),
    }


# ────────────────────────────────────────────────────────────────────
# Track B — threshold-based metrics
# ────────────────────────────────────────────────────────────────────
def compute_threshold_metrics(
    df: pd.DataFrame,
    score_col: str,
    y_col: str,
    tau: float,
) -> dict:
    """Threshold-based binary-classification metrics at a single τ.

    Per (anchor, app) row:
        predicted_kill = 1 iff score > τ else 0
        kill_label     = 1 iff y == 0 (safe to kill / OS would benefit from kill)

    Returns:
        kill_precision, kill_recall, f1, accuracy, mcc, fpr, fnr,
        tp, fp, tn, fn, n, tau
    """
    s = df[score_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.int64)
    pred_kill  = (s > tau).astype(np.int64)
    kill_label = (1 - y).astype(np.int64)

    tp = int(((pred_kill == 1) & (kill_label == 1)).sum())
    fp = int(((pred_kill == 1) & (kill_label == 0)).sum())
    tn = int(((pred_kill == 0) & (kill_label == 0)).sum())
    fn = int(((pred_kill == 0) & (kill_label == 1)).sum())
    n = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / float(np.sqrt(denom_sq)) if denom_sq > 0 else 0.0

    return {
        "tau": float(tau),
        "kill_precision": float(precision),
        "kill_recall":    float(recall),
        "f1":             float(f1),
        "accuracy":       float(accuracy),
        "mcc":            float(mcc),
        "fpr":            float(fpr),
        "fnr":            float(fnr),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n,
    }


def find_best_tau_by_f1(
    df: pd.DataFrame,
    score_col: str,
    y_col: str,
    n_grid: int = 49,
) -> float:
    """Return the τ that maximises F1 on this df. Use on val, freeze for test."""
    s = df[score_col].to_numpy(dtype=np.float64)
    qs = np.linspace(0.02, 0.98, n_grid)
    tau_grid = np.unique(np.quantile(s, qs))
    best_tau, best_f1 = float(tau_grid[0]), -1.0
    for tau in tau_grid:
        m = compute_threshold_metrics(df, score_col, y_col, tau=float(tau))
        if m["f1"] > best_f1 + 1e-9:
            best_f1 = m["f1"]
            best_tau = float(tau)
    return best_tau
