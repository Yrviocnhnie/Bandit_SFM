"""Layer 4 — closed-form baselines correctness on synthetic per-user bg data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg.baselines_bg import (
    add_lru_score, add_markov_inverse, add_random_score, add_time_in_bg_score,
)
from lib.bg.metrics_bg import compute_metrics


@pytest.fixture(scope="module")
def synthetic_user_bg():
    """A small bg df for one user (4 anchors × 4 apps), with sane labels."""
    rng = np.random.default_rng(7)
    rows = []
    for anchor_id in range(4):
        ts_ns = int((1700000000 + anchor_id * 600) * 1_000_000_000)
        for app_idx in (3, 4, 5, 6):
            rows.append({
                "user_uid": "u0", "user_id_idx": 0,
                "anchor_id": anchor_id, "anchor_ts_ns": ts_ns,
                "anchor_hour": int(rng.integers(0, 24)),
                "app_label": f"app{app_idx}", "app_idx": int(app_idx),
                "last_fg_app_idx": int(rng.integers(3, 7)),
                "time_in_bg_sec": float(rng.uniform(60, 6 * 3600)),
                "time_since_fg_sec": float(rng.uniform(60, 6 * 3600)),
                "fg_count_today": int(rng.integers(0, 10)),
                "y_3600": int(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


def test_random_score_uniform(synthetic_user_bg):
    out = add_random_score(synthetic_user_bg, seed=7)
    s = out["score_random"].to_numpy()
    assert (s >= 0).all() and (s <= 1).all()
    # With seed=7 we get the same sequence
    out2 = add_random_score(synthetic_user_bg, seed=7)
    np.testing.assert_array_equal(s, out2["score_random"].to_numpy())


def test_lru_score_monotone_with_time(synthetic_user_bg):
    """For each anchor, the row with the larger `time_since_fg_sec` should get a larger LRU score."""
    out = add_lru_score(synthetic_user_bg)
    # For row pairs within the same anchor, score order matches time_since_fg_sec order
    for _, sub in out.groupby("anchor_id"):
        ts = sub["time_since_fg_sec"].to_numpy()
        sc = sub["score_lru"].to_numpy()
        # rank by time and rank by score should match (no negative tfg in this fixture)
        rank_t = np.argsort(ts)
        rank_s = np.argsort(sc)
        np.testing.assert_array_equal(rank_t, rank_s)


def test_time_in_bg_score(synthetic_user_bg):
    out = add_time_in_bg_score(synthetic_user_bg)
    np.testing.assert_array_equal(
        out["score_tibg"].to_numpy(),
        synthetic_user_bg["time_in_bg_sec"].to_numpy(),
    )


def test_markov_inverse_lookup(synthetic_user_bg):
    """For known markov_prob[last, app], score = 1 - that prob."""
    V = 8
    rng = np.random.default_rng(0)
    markov_prior = rng.dirichlet([0.5] * V, size=V).astype(np.float32)
    out = add_markov_inverse(synthetic_user_bg, markov_prior=markov_prior)
    last = synthetic_user_bg["last_fg_app_idx"].to_numpy(dtype=int)
    app = synthetic_user_bg["app_idx"].to_numpy(dtype=int)
    expected = 1.0 - markov_prior[np.clip(last, 0, V - 1), np.clip(app, 0, V - 1)]
    np.testing.assert_allclose(out["score_markov_inv"].to_numpy(), expected, atol=1e-6)


def test_baseline_metric_bounds(synthetic_user_bg):
    """All Track A metrics should be in [0, 1]."""
    out = add_random_score(synthetic_user_bg, seed=7)
    m = compute_metrics(out, score_col="score_random", y_col="y_3600",
                        r_values=(0.25, 0.5, 0.75))
    for r in ("0.25", "0.5", "0.75"):
        assert 0.0 <= m["false_kill_rate"][r] <= 1.0
        assert 0.0 <= m["memory_save_rate"][r] <= 1.0
    assert (m["roc_auc_mean"] != m["roc_auc_mean"]) or 0.0 <= m["roc_auc_mean"] <= 1.0
    assert (m["pr_auc_mean"] != m["pr_auc_mean"]) or 0.0 <= m["pr_auc_mean"] <= 1.0
    assert 0.0 <= m["ndcg_half"] <= 1.0


def test_aggregate_mean_matches_numpy_mean():
    """Per-user aggregation in 51_run_baselines should equal np.mean over per-user values."""
    # Synthesize a few per-user metric dicts and verify the aggregate computes correctly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "baselines_multi", ROOT / "scripts" / "51_run_baselines_bg_multiuser.py",
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["baselines_multi"] = m
    spec.loader.exec_module(m)

    pu = {"u0": {"pr_auc_mean": 0.7, "false_kill_rate": {"0.5": 0.18}},
          "u1": {"pr_auc_mean": 0.8, "false_kill_rate": {"0.5": 0.22}},
          "u2": {"pr_auc_mean": 0.9, "false_kill_rate": {"0.5": 0.26}}}
    block = {"per_user": pu, "n_users": 3}
    agg = m.aggregate(block)
    assert abs(agg["mean"]["pr_auc_mean"] - 0.8) < 1e-6
    assert abs(agg["mean"]["false_kill_rate"]["0.5"] - 0.22) < 1e-6
    assert abs(agg["median"]["pr_auc_mean"] - 0.8) < 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
