"""Layer 3 — per-user stat lookup tests.

Verifies that `build_schema_data_multi`'s vectorized gather matches the per-user
stats it indexes into. Synthetic 3-user setup, no model training.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg_multi.helpers import GLOBAL_ANCHOR_OFFSET, make_global_anchor_id, stack_per_user
from lib.v3 import categories as CAT


# Lazy-import the trainer module (heavy importlib path)
def _load_train_multi():
    spec = importlib.util.spec_from_file_location(
        "train_multi_for_test", ROOT / "scripts" / "52_train_task_c_multiuser.py",
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["train_multi_for_test"] = m
    spec.loader.exec_module(m)
    return m


# ────────────────────────────────────────────────────────────────────────────
# stack_per_user: row-wise gather correctness
# ────────────────────────────────────────────────────────────────────────────
def test_stack_lookup_random_correctness():
    """50 random gather indices match per_user[uid][...]."""
    rng = np.random.default_rng(7)
    U, V = 4, 8
    per_user = {f"u{i}": {"hf": rng.standard_normal((24, V)).astype(np.float32)} for i in range(U)}
    uid_to_idx = {f"u{i}": i for i in range(U)}
    stacked = stack_per_user(per_user, "hf", uid_to_idx)
    for _ in range(50):
        i = int(rng.integers(U))
        h = int(rng.integers(24))
        a = int(rng.integers(V))
        np.testing.assert_allclose(stacked[i, h, a], per_user[f"u{i}"]["hf"][h, a])


def test_user_specific_markov_distinct():
    """Two users get distinct markov rows (since stats are computed per-user)."""
    rng = np.random.default_rng(7)
    V = 6
    per_user = {
        "u0": {"mp": rng.dirichlet([0.1] * V, size=V).astype(np.float32)},
        "u1": {"mp": rng.dirichlet([0.1] * V, size=V).astype(np.float32)},
    }
    uid_to_idx = {"u0": 0, "u1": 1}
    stacked = stack_per_user(per_user, "mp", uid_to_idx)
    diff = float(np.linalg.norm(stacked[0] - stacked[1]))
    assert diff > 1e-3, "two users' markov tables should differ"


# ────────────────────────────────────────────────────────────────────────────
# build_schema_data_multi: shape + per-user consistency
# ────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def synthetic_pooled_bg():
    """Build a tiny synthetic pooled bg DataFrame with 2 users × 3 anchors × 4 apps."""
    rng = np.random.default_rng(7)
    rows = []
    for uid_idx in (0, 1):
        for anchor_id in range(3):
            ts_ns = int((1700000000 + uid_idx * 1000 + anchor_id) * 1_000_000_000)
            for app_idx in (3, 4, 5, 6):  # avoid reserved indices
                rows.append({
                    "user_uid": f"u{uid_idx}",
                    "user_id_idx": uid_idx,
                    "anchor_id": anchor_id,
                    "anchor_ts_ns": ts_ns,
                    "anchor_hour": int(rng.integers(0, 24)),
                    "anchor_weekday": int(rng.integers(0, 7)),
                    "anchor_daypart": int(rng.integers(0, 10)),
                    "last_fg_app_idx": int(rng.integers(3, 7)),
                    "last_fg_daypart": int(rng.integers(0, 10)),
                    "last_fg_loc_id": 0,
                    "bg_set_size": 4,
                    "app_label": f"app{app_idx}",
                    "app_idx": int(app_idx),
                    "time_in_bg_sec": float(rng.uniform(60, 6 * 3600)),
                    "time_since_fg_sec": float(rng.uniform(60, 6 * 3600)),
                    "fg_count_today": int(rng.integers(0, 10)),
                    "time_since_screen_on_sec": float(rng.uniform(0, 1800)),
                    "prev_killed_app_idx": 0,
                    "prev_killed_age_sec": -1.0,
                    "bg_recency_min_sec": float(rng.uniform(60, 600)),
                    "fg_count_last_3600s": int(rng.integers(0, 5)),
                    "fg_count_last_21600s": int(rng.integers(0, 10)),
                    "recency_rank_in_bg": float(rng.uniform(0, 1)),
                    "y_3600": int(rng.integers(0, 2)),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_ctx(synthetic_pooled_bg):
    """Synthetic per-user stats stack matching the bg fixture."""
    V = 8
    V_cat = 11  # CAT.NUM_CATEGORIES is 11
    U = 2
    rng = np.random.default_rng(0)
    # Build a per-user dict as if produced by fit_per_user_stats
    per_user = {}
    uid_to_idx = {f"u{i}": i for i in range(U)}
    for uid in uid_to_idx:
        per_user[uid] = {
            "hour_freq": rng.uniform(0, 1, size=(24, V)).astype(np.float32),
            "markov_probs": rng.dirichlet([0.5] * V, size=V).astype(np.float32),
            "per_app_inter_fg_mean": rng.uniform(60, 86400, size=(V,)).astype(np.float32),
            "app_lifetime_share": rng.dirichlet([0.5] * V).astype(np.float32),
            "app_lifetime_kill_rate": rng.uniform(0, 1, size=(V,)).astype(np.float32),
            "cat_lifetime_share": rng.dirichlet([0.5] * V_cat).astype(np.float32),
            "cat_markov_probs": rng.dirichlet([0.5] * V_cat, size=V_cat).astype(np.float32),
            "fg_timeline": {a: np.array([], dtype=np.int64) for a in range(V)},  # empty timeline
        }
    # We make app 3 in user 0 have a marker timestamp so c3.1 features can pick it up
    per_user["u0"]["fg_timeline"][3] = np.array([1700000000_000_000_000], dtype=np.int64)
    # Fake vocab/app_to_cat
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2, **{f"app{i}": i for i in range(3, V)}}
    app_to_cat = CAT.build_app_to_cat_idx(vocab)
    return {
        "vocab": vocab, "uid_to_idx": uid_to_idx, "app_to_cat": app_to_cat,
        "hour_freq_stack":              stack_per_user(per_user, "hour_freq", uid_to_idx),
        "markov_probs_stack":           stack_per_user(per_user, "markov_probs", uid_to_idx),
        "per_app_inter_stack":          stack_per_user(per_user, "per_app_inter_fg_mean", uid_to_idx),
        "app_lifetime_share_stack":     stack_per_user(per_user, "app_lifetime_share", uid_to_idx),
        "app_lifetime_kill_rate_stack": stack_per_user(per_user, "app_lifetime_kill_rate", uid_to_idx),
        "cat_lifetime_share_stack":     stack_per_user(per_user, "cat_lifetime_share", uid_to_idx),
        "cat_markov_probs_stack":       stack_per_user(per_user, "cat_markov_probs", uid_to_idx),
        "fg_timeline_per_user": {
            i: per_user[uid]["fg_timeline"] for uid, i in uid_to_idx.items()
        },
    }


def test_build_schema_data_multi_c33_shape(synthetic_pooled_bg, synthetic_ctx):
    """c3.3 schema should produce 33 columns (16 c2 + 5 c31 + 9 c32 + 3 c33)."""
    train_multi = _load_train_multi()
    out = train_multi.build_schema_data_multi("c3.3", synthetic_pooled_bg, synthetic_ctx)
    assert "features" in out
    assert "app_idx" in out
    assert "y" in out
    assert "anchor_id" in out
    n = len(synthetic_pooled_bg)
    assert out["features"].shape == (n, 33)
    assert out["features"].dtype == np.float32
    # No NaN/inf
    assert np.isfinite(out["features"]).all()


def test_build_schema_data_multi_anchor_id_is_global(synthetic_pooled_bg, synthetic_ctx):
    """anchor_id in the data dict should be the global anchor id (collision-free)."""
    train_multi = _load_train_multi()
    out = train_multi.build_schema_data_multi("c3.3", synthetic_pooled_bg, synthetic_ctx)
    expected = make_global_anchor_id(synthetic_pooled_bg)
    np.testing.assert_array_equal(out["anchor_id"], expected.astype(np.int64))
    # And — no collision: 2 users × 3 anchors = 6 unique IDs
    assert int(np.unique(out["anchor_id"]).size) == 6


def test_per_user_consistency_markov_col(synthetic_pooled_bg, synthetic_ctx):
    """For each row, the markov col equals per-user markov[last_app, app_idx]."""
    train_multi = _load_train_multi()
    feats = train_multi.build_c2_multi(
        synthetic_pooled_bg, synthetic_ctx["hour_freq_stack"], synthetic_ctx["markov_probs_stack"],
    )
    # Markov col is index 11 in build_c2_multi (after the 11 simpler features)
    # Actually it's the 12th column (index 11) per the column list — let me just verify
    # by re-computing the expected value from the stack and comparing.
    n = len(synthetic_pooled_bg)
    uid = synthetic_pooled_bg["user_id_idx"].to_numpy(dtype=np.int64)
    last = synthetic_pooled_bg["last_fg_app_idx"].to_numpy(dtype=np.int64)
    app = synthetic_pooled_bg["app_idx"].to_numpy(dtype=np.int64)
    expected_markov = synthetic_ctx["markov_probs_stack"][uid, last, app]
    # Find which column matches expected_markov (one of them must)
    for col in range(feats.shape[1]):
        if np.allclose(feats[:, col], expected_markov.astype(np.float32), atol=1e-5):
            return
    pytest.fail("no column in c2_multi features matches per-user markov gather")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
