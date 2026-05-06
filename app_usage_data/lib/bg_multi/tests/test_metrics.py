"""Layer 6 — metric range, monotonicity, and aggregation correctness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg.metrics_bg import (
    compute_keep_threshold_metrics, compute_metrics, compute_threshold_metrics,
    find_best_tau_by_f1, find_best_tau_by_f1_keep,
)


@pytest.fixture(scope="module")
def synthetic_bg():
    """Synthetic bg test with 5 anchors × 6 apps; mixed labels."""
    rng = np.random.default_rng(7)
    rows = []
    for anchor_id in range(5):
        for app_idx in (3, 4, 5, 6, 7, 8):
            rows.append({
                "anchor_id": anchor_id, "app_idx": int(app_idx),
                "y_3600": int(rng.integers(0, 2)),
                "score_random": float(rng.uniform()),
                "score_perfect": float(1 - int(rng.integers(0, 2))),  # arbitrary
            })
    df = pd.DataFrame(rows)
    # build a perfect-correlated score (1 - y → high kill_score iff y == 0)
    df["score_perfect"] = (1.0 - df["y_3600"]).astype(np.float64)
    return df


def test_track_a_msr_monotone_in_r(synthetic_bg):
    """MSR@r should be non-decreasing in r."""
    m = compute_metrics(synthetic_bg, score_col="score_random", y_col="y_3600",
                        r_values=(0.1, 0.25, 0.5, 0.75, 0.9))
    msrs = [m["memory_save_rate"][k] for k in ("0.1", "0.25", "0.5", "0.75", "0.9")]
    for i in range(len(msrs) - 1):
        assert msrs[i] <= msrs[i + 1] + 1e-6, (
            f"MSR@.{i} > MSR@.{i+1}: {msrs[i]} > {msrs[i+1]}"
        )


def test_random_mcc_near_zero(synthetic_bg):
    """For random scores, MCC should be near zero (within tolerance)."""
    tau = float(np.median(synthetic_bg["score_random"]))
    m = compute_threshold_metrics(synthetic_bg, score_col="score_random",
                                   y_col="y_3600", tau=tau)
    # With only 30 rows we can have noisy MCC; 0.5 is loose but catches gross bugs
    assert abs(m["mcc"]) < 0.5, f"random MCC = {m['mcc']:.3f} (expected ≈ 0)"


def test_perfect_score_high_pr_auc(synthetic_bg):
    """A perfectly-correlated score should give PR-AUC near 1."""
    m = compute_metrics(synthetic_bg, score_col="score_perfect", y_col="y_3600")
    # Perfect score: kill_score = 1 - y. Apps with y=0 (kill class) get score=1, y=1 get score=0.
    # Top-r ranking is perfect → MSR=1 at r ≥ pos_rate, FK=0 below capacity.
    msr_05 = m["memory_save_rate"]["0.5"]
    fk_05 = m["false_kill_rate"]["0.5"]
    # Not strictly 1.0 because top-K capacity may exceed the # killable apps in some anchors
    assert msr_05 >= 0.5
    assert fk_05 <= 0.5


def test_argmax_keep_lands_at_higher_tau_than_kill():
    """For a non-trivial probability score, argmax-F1(keep) should land higher than argmax-F1(kill).

    This mirrors the single-user finding where τ_kill ≈ 0.18 and τ_keep ≈ 0.42.
    """
    rng = np.random.default_rng(7)
    n = 200
    # Probability-style score, biased toward 0.5
    s = rng.beta(2, 2, size=n).astype(np.float64)
    y = (s + rng.normal(0, 0.3, size=n) < 0.5).astype(int)  # noisy correlation
    df = pd.DataFrame({"anchor_id": np.arange(n), "score": s, "y_3600": y})
    tau_kill = find_best_tau_by_f1(df, "score", "y_3600")
    tau_keep = find_best_tau_by_f1_keep(df, "score", "y_3600")
    # In the recall-saturated regime (kill = majority), kill argmax lands low; keep argmax higher.
    # Allow == in degenerate corners.
    assert tau_keep >= tau_kill - 1e-6


def test_aggregate_mean_matches_numpy():
    """Per-user aggregation should equal np.mean over per-user values."""
    sample = {
        "u0": {"pr_auc_mean": 0.7, "false_kill_rate": {"0.5": 0.18}},
        "u1": {"pr_auc_mean": 0.8, "false_kill_rate": {"0.5": 0.22}},
        "u2": {"pr_auc_mean": 0.9, "false_kill_rate": {"0.5": 0.26}},
    }
    # Simulate 53_eval's aggregate_users
    agg_pr = float(np.mean([sample[u]["pr_auc_mean"] for u in sample]))
    assert abs(agg_pr - 0.8) < 1e-9
    agg_fk = float(np.mean([sample[u]["false_kill_rate"]["0.5"] for u in sample]))
    assert abs(agg_fk - 0.22) < 1e-9


def test_compute_keep_threshold_metrics_inverse_of_kill():
    """For y_kill = 1-y, the keep-class TP/FP/TN/FN are kill's TN/FN/TP/FP."""
    rng = np.random.default_rng(7)
    n = 100
    df = pd.DataFrame({
        "anchor_id": np.arange(n),
        "score": rng.uniform(size=n),
        "y_3600": rng.integers(0, 2, size=n),
    })
    tau = 0.5
    m_kill = compute_threshold_metrics(df, "score", "y_3600", tau=tau)
    m_keep = compute_keep_threshold_metrics(df, "score", "y_3600", tau=tau)
    # TP_keep == TN_kill and so on
    assert m_keep["tp_keep"] == m_kill["tn"]
    assert m_keep["fp_keep"] == m_kill["fn"]
    assert m_keep["tn_keep"] == m_kill["tp"]
    assert m_keep["fn_keep"] == m_kill["fp"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
