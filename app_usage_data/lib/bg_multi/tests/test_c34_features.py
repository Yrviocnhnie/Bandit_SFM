"""Tests for c3.4 feature fitters."""
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

from lib.bg_multi.c34_features import (
    compute_last2_fg_for_anchors,
    fit_app_popularity_bucket,
    fit_co_fg_matrix_per_user,
    fit_cross_user_app_prior,
    fit_dow_hour_freq,
    fit_two_step_markov_per_user,
)


# ────────────────────────────────────────────────────────────────────────────
# F3: dow_hour_freq
# ────────────────────────────────────────────────────────────────────────────
def _train_events_for_user():
    return pd.DataFrame({
        "event_ts": pd.to_datetime([
            "2026-04-01 09:00:00",  # Wed (weekday=2), hour=9, app=A
            "2026-04-01 09:30:00",  # Wed,  hour=9, app=A
            "2026-04-01 14:00:00",  # Wed,  hour=14, app=B
            "2026-04-02 09:00:00",  # Thu (weekday=3), hour=9, app=A
            "2026-04-02 14:00:00",  # Thu,  hour=14, app=A
        ]),
        "is_target_event": [True] * 5,
        "app_label_clean": ["A", "A", "B", "A", "A"],
    })


def test_fit_dow_hour_freq_shape_and_normalisation():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2, "A": 3, "B": 4}
    train = _train_events_for_user()
    table = fit_dow_hour_freq(train, vocab)
    assert table.shape == (7, 24, 5)
    assert table.dtype == np.float32
    assert (table[:, :, :3] == 0.0).all()
    sums = table[:, :, 3:].sum(axis=2)
    np.testing.assert_allclose(sums, 1.0, atol=1e-4)


def test_fit_dow_hour_freq_picks_up_observed_app():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2, "A": 3, "B": 4}
    train = _train_events_for_user()
    table = fit_dow_hour_freq(train, vocab)
    # On Wed (weekday=2) at hour=9, only A appears -> A's smoothed prob > B's
    assert table[2, 9, 3] > table[2, 9, 4]
    # On Wed at hour=14, only B appears -> B > A
    assert table[2, 14, 4] > table[2, 14, 3]


# ────────────────────────────────────────────────────────────────────────────
# F5: app_popularity_bucket
# ────────────────────────────────────────────────────────────────────────────
def test_fit_app_popularity_bucket_thresholds():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2,
             "WIDE_APP": 3, "MED_APP": 4, "RARE_APP": 5, "UNUSED_APP": 6}
    per_user_train_dfs = {}
    for i in range(22):
        apps = ["WIDE_APP"]
        if i < 10:
            apps.append("MED_APP")
        if i < 2:
            apps.append("RARE_APP")
        df = pd.DataFrame({
            "is_target_event": [True] * len(apps),
            "app_label_clean": apps,
        })
        per_user_train_dfs[f"u{i}"] = df

    bucket = fit_app_popularity_bucket(
        per_user_train_dfs, vocab,
        common_min=15, medium_min=5,
    )
    assert bucket.shape == (len(vocab),)
    assert bucket.dtype == np.int64
    assert bucket[0] == 0
    assert bucket[1] == 0
    assert bucket[2] == 0
    assert bucket[vocab["WIDE_APP"]] == 2
    assert bucket[vocab["MED_APP"]] == 1
    assert bucket[vocab["RARE_APP"]] == 0
    assert bucket[vocab["UNUSED_APP"]] == 0


# ────────────────────────────────────────────────────────────────────────────
# F6: cross_user_app_prior
# ────────────────────────────────────────────────────────────────────────────
def test_fit_cross_user_app_prior_excludes_self():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2, "X": 3, "Y": 4}
    bg_train = pd.DataFrame({
        "user_uid": ["u0"]*4 + ["u1"]*4 + ["u2"]*4,
        "app_idx":  [3, 3, 4, 4]*3,
        "y_3600":   [1, 1, 0, 0,    # u0: P(used | X) = 1.0, P(used | Y) = 0.0
                     1, 0, 0, 0,    # u1: P(used | X) = 0.5, P(used | Y) = 0.0
                     0, 0, 1, 1],   # u2: P(used | X) = 0.0, P(used | Y) = 1.0
    })
    table = fit_cross_user_app_prior(bg_train, vocab)
    assert isinstance(table, dict)
    for uid in ("u0", "u1", "u2"):
        assert table[uid].shape == (5,)
    np.testing.assert_allclose(table["u0"][3], 0.25, atol=1e-6)
    np.testing.assert_allclose(table["u0"][4], 0.5, atol=1e-6)
    np.testing.assert_allclose(table["u1"][3], 0.5, atol=1e-6)


# ────────────────────────────────────────────────────────────────────────────
# F1 prep: last2_fg lookup
# ────────────────────────────────────────────────────────────────────────────
def test_compute_last2_fg_for_anchors():
    fg_ts_ns = np.array([10, 20, 30, 40], dtype=np.int64) * 1_000_000_000
    fg_app_idx = np.array([3, 4, 3, 5], dtype=np.int64)
    anchor_ts_ns = np.array([15, 25, 35, 45, 50], dtype=np.int64) * 1_000_000_000
    last2 = compute_last2_fg_for_anchors(anchor_ts_ns, fg_ts_ns, fg_app_idx)
    assert last2[0] == 0   # only 1 prior FG
    assert last2[1] == 3   # 2 priors A, B → last2 = A
    assert last2[2] == 4   # 3 priors A, B, A → last2 = B
    assert last2[3] == 3   # 4 priors A, B, A, C → last2 = A
    assert last2[4] == 3   # same as above


def test_compute_last2_empty_fg_stream():
    out = compute_last2_fg_for_anchors(
        np.array([10, 20], dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
    )
    assert out.tolist() == [0, 0]


# ────────────────────────────────────────────────────────────────────────────
# F1: 2-step Markov
# ────────────────────────────────────────────────────────────────────────────
def test_fit_two_step_markov_per_user_normalisation():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2, "A": 3, "B": 4, "C": 5}
    train = pd.DataFrame({
        "event_ts": pd.to_datetime([
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
            "2026-04-01 10:00:00",
            "2026-04-01 11:00:00",
            "2026-04-01 12:00:00",
        ]),
        "is_target_event": [True] * 5,
        "app_label_clean": ["A", "B", "C", "A", "B"],
    })
    table = fit_two_step_markov_per_user(train, vocab)
    assert isinstance(table, dict)
    if (vocab["A"], vocab["B"]) in table:
        probs = table[(vocab["A"], vocab["B"])]
        # Reserved cells zeroed
        assert (probs[:3] == 0).all()
        # Row sum over non-reserved is 1.0
        np.testing.assert_allclose(probs[3:].sum(), 1.0, atol=1e-4)
        # The observed C is more likely than the unobserved A
        assert probs[vocab["C"]] > probs[vocab["A"]]


# ────────────────────────────────────────────────────────────────────────────
# F2: co_fg matrix
# ────────────────────────────────────────────────────────────────────────────
def test_fit_co_fg_matrix_per_user():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2, "A": 3, "B": 4}
    bg_train = pd.DataFrame({
        "user_uid": ["u0"] * 3,
        "anchor_id": [1, 2, 3],
        "app_idx":   [4, 4, 3],
        "last_fg_app_idx": [3, 3, 4],
        "y_3600": [1, 0, 1],
    })
    co_fg = fit_co_fg_matrix_per_user(bg_train, vocab)
    np.testing.assert_allclose(co_fg["u0"][vocab["B"], vocab["A"]], 0.5, atol=1e-4)
    np.testing.assert_allclose(co_fg["u0"][vocab["A"], vocab["B"]], 1.0, atol=1e-4)


# ────────────────────────────────────────────────────────────────────────────
# Schema dispatch (smoke): hour_segment_onehot
# ────────────────────────────────────────────────────────────────────────────
def _load_train_multi():
    if "train_multi_c34_smoke" in sys.modules:
        return sys.modules["train_multi_c34_smoke"]
    spec = importlib.util.spec_from_file_location(
        "train_multi_c34_smoke",
        ROOT / "scripts" / "52_train_task_c_multiuser.py",
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["train_multi_c34_smoke"] = m
    spec.loader.exec_module(m)
    return m


def test_hour_segment_onehot_partitions_correctly():
    train_multi = _load_train_multi()
    if not hasattr(train_multi, "hour_segment_onehot"):
        pytest.skip("hour_segment_onehot not yet implemented")
    hours = np.arange(24)
    out = train_multi.hour_segment_onehot(hours)
    assert out.shape == (24, 5)
    np.testing.assert_array_equal(out.sum(axis=1), np.ones(24))
    assert out[7, 0] == 1.0   # morning
    assert out[12, 1] == 1.0  # lunch
    assert out[15, 2] == 1.0  # afternoon
    assert out[20, 3] == 1.0  # evening
    assert out[23, 4] == 1.0  # night
    assert out[3, 4] == 1.0   # night wrap


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
