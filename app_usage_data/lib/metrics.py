"""Evaluation metrics for Task A (next app) and Task B (15-min window set prediction).

All functions are numpy-based and return Python scalars or dicts of scalars. Vectorized
where possible; confidence-interval helpers sit at the bottom.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.stats import beta


# =============================================================================
# Task A: next-app classification metrics
# =============================================================================

def hit_at_k(logits_or_probs: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    """Top-k accuracy. `targets` are class indices (shape (N,))."""
    topk = np.argsort(-logits_or_probs, axis=1)[:, :k]
    hits = (topk == targets[:, None]).any(axis=1)
    return float(hits.mean())


def mrr(logits_or_probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean reciprocal rank of the true class."""
    order = np.argsort(-logits_or_probs, axis=1)
    # For each row, find rank of target
    ranks = np.array([np.where(order[i] == t)[0][0] for i, t in enumerate(targets)])
    return float((1.0 / (ranks + 1.0)).mean())


def macro_f1(logits_or_probs: np.ndarray, targets: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 over predicted classes."""
    preds = logits_or_probs.argmax(axis=1)
    f1s = []
    for c in range(n_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def per_class_hit_at_1(
    logits_or_scores: np.ndarray, targets: np.ndarray, n_classes: int
) -> Dict[int, float]:
    preds = logits_or_scores.argmax(axis=1)
    out: Dict[int, float] = {}
    for c in range(n_classes):
        mask = targets == c
        if mask.sum() == 0:
            continue
        out[int(c)] = float((preds[mask] == c).mean())
    return out


# =============================================================================
# Task B: 15-min window set metrics
# =============================================================================
def topk_set(scores: np.ndarray, k: int) -> np.ndarray:
    """Return top-k indices per row, sorted by score desc. (N, k)."""
    if k <= 0:
        return np.empty((scores.shape[0], 0), dtype=np.int64)
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    scores_at = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-scores_at, axis=1)
    return np.take_along_axis(part, order, axis=1)


def precision_at_k(topk: np.ndarray, gt_sets: list, k: int) -> np.ndarray:
    p = np.zeros(len(topk), dtype=np.float32)
    for i, tk in enumerate(topk):
        g = gt_sets[i]
        if len(g) == 0:
            p[i] = np.nan
            continue
        hits = sum(1 for x in tk[:k] if x in g)
        p[i] = hits / k
    return p


def recall_at_k(topk: np.ndarray, gt_sets: list, k: int) -> np.ndarray:
    r = np.zeros(len(topk), dtype=np.float32)
    for i, tk in enumerate(topk):
        g = gt_sets[i]
        if len(g) == 0:
            r[i] = np.nan
            continue
        hits = sum(1 for x in tk[:k] if x in g)
        r[i] = hits / len(g)
    return r


def f1_at_k(topk: np.ndarray, gt_sets: list, k: int) -> np.ndarray:
    p = precision_at_k(topk, gt_sets, k)
    r = recall_at_k(topk, gt_sets, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return f


def jaccard_at_k(topk: np.ndarray, gt_sets: list, k: int) -> np.ndarray:
    j = np.zeros(len(topk), dtype=np.float32)
    for i, tk in enumerate(topk):
        g = set(map(int, gt_sets[i]))
        if not g:
            j[i] = np.nan
            continue
        t = set(int(x) for x in tk[:k])
        inter = len(t & g)
        union = len(t | g)
        j[i] = inter / union if union else 0.0
    return j


def coverage_at_k(topk: np.ndarray, gt_sets: Sequence[Iterable[int] | set[int]], k: int) -> np.ndarray:
    cov = np.zeros(len(gt_sets), dtype=np.float32)
    for i in range(len(gt_sets)):
        g = set(int(x) for x in gt_sets[i])
        if not g:
            cov[i] = np.nan
            continue
        t = set(int(x) for x in topk[i, :k])
        cov[i] = 1.0 if g.issubset(t) else 0.0
    return cov


def event_hit_at_k(topk: np.ndarray, gt_multisets: list, k: int) -> np.ndarray:
    """Frequency-weighted hit rate: fraction of events whose app is in top-K."""
    out = np.zeros(len(gt_multisets), dtype=np.float32)
    for i, ms in enumerate(gt_multisets):
        if not ms:
            out[i] = np.nan
            continue
        total = sum(ms.values())
        if total == 0:
            out[i] = np.nan
            continue
        t = set(int(x) for x in topk[i, :k])
        hit = sum(c for a, c in ms.items() if int(a) in t)
        out[i] = hit / total
    return out


def map_at_k(topk: np.ndarray, gt_sets: list, k: int) -> np.ndarray:
    out = np.zeros(len(gt_sets), dtype=np.float32)
    for i, tk in enumerate(topk):
        g = gt_sets[i]
        if not g:
            out[i] = np.nan
            continue
        ap = 0.0
        hits = 0
        for rank, x in enumerate(tk[:k], start=1):
            if int(x) in g:
                hits += 1
                ap += hits / rank
        denom = min(len(g), k)
        out[i] = ap / denom if denom else 0.0
    return out


def ndcg_at_k(topk: np.ndarray, gt_sets: list, k: int) -> np.ndarray:
    out = np.zeros(len(gt_sets), dtype=np.float32)
    for i, tk in enumerate(topk):
        g = gt_sets[i]
        if not g:
            out[i] = np.nan
            continue
        dcg = 0.0
        for rank, x in enumerate(tk[:k], start=1):
            if int(x) in g:
                dcg += 1.0 / np.log2(rank + 1)
        ideal = min(len(g), k)
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal + 1))
        out[i] = dcg / idcg if idcg > 0 else 0.0
    return out


# =============================================================================
# Calibration
# =============================================================================
def brier_multiclass(probs: np.ndarray, targets: np.ndarray, n_classes: int) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(targets)), targets] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def brier_binary(probs: np.ndarray, y: np.ndarray) -> float:
    return float(((probs - y) ** 2).mean())


def ece(probs_top1: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error on the top-1 probability."""
    assert len(probs_top1) == len(correct)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    n = len(probs_top1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs_top1 >= lo) & (probs_top1 < hi if i < n_bins - 1 else probs_top1 <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        conf = probs_top1[mask].mean()
        err += abs(acc - conf) * (mask.sum() / n)
    return float(err)


# =============================================================================
# Confidence intervals
# =============================================================================
def wilson_ci(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score 95 % CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    from scipy.stats import norm  # lazy
    z = norm.ppf(1 - (1 - conf) / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 1000, conf: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI + point estimate for an aggregate statistic (default: mean)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n = len(vals)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = stat(vals[idx])
    lo, hi = np.quantile(boot, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return (float(stat(vals)), float(lo), float(hi))


# =============================================================================
# Aggregation helpers
# =============================================================================
def macro_aggregate(per_anchor_values: np.ndarray) -> float:
    vals = per_anchor_values[~np.isnan(per_anchor_values)]
    return float(vals.mean()) if len(vals) else float("nan")


# Name alias used throughout the codebase
event_hit_rate = event_hit_at_k


def event_hit_micro(gt_multisets: list, topk: np.ndarray, k: int) -> float:
    num = 0
    den = 0
    for i, ms in enumerate(gt_multisets):
        if not ms:
            continue
        t = set(int(x) for x in topk[i, :k])
        num += sum(c for a, c in ms.items() if int(a) in t)
        den += sum(ms.values())
    return float(num / den) if den else float("nan")
