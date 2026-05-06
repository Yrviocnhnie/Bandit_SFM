"""Layer 1 — unit tests for lib/bg_multi/helpers.py orchestration helpers."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `lib.*` importable from the test runner regardless of cwd
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg_multi.helpers import (
    GLOBAL_ANCHOR_OFFSET,
    attach_user_columns,
    build_pooled_vocab,
    list_cohort_users,
    make_global_anchor_id,
    short_uid_from_path,
    stack_per_user,
)


# ────────────────────────────────────────────────────────────────────────────
# short_uid_from_path
# ────────────────────────────────────────────────────────────────────────────
def test_short_uid_stable():
    p = "/data00/x/y/1E1480118A703377325564F77308F87DCC482296AF40969F903403B81FCAAE89_cleaned.xlsx"
    assert short_uid_from_path(p) == "1E1480118A70"
    # idempotent / deterministic
    assert short_uid_from_path(p) == short_uid_from_path(p)


def test_short_uid_distinct_paths():
    p1 = "/x/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA_cleaned.xlsx"
    p2 = "/x/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB_cleaned.xlsx"
    assert short_uid_from_path(p1) != short_uid_from_path(p2)


def test_short_uid_fallback_for_non_hex():
    """Non-matching name falls back to filename stem (no .xlsx)."""
    assert short_uid_from_path("/x/some_other_file.xlsx") == "some_other_file"


# ────────────────────────────────────────────────────────────────────────────
# list_cohort_users
# ────────────────────────────────────────────────────────────────────────────
def test_list_cohort_users_synthetic(tmp_path):
    """Build a fake cohort dir with 2 sets, 3 users each, and verify enumeration."""
    counter = 0
    for set_name in ("M_beta_Top30", "top2000"):
        sub = tmp_path / set_name
        sub.mkdir()
        for _ in range(3):
            # Put the unique 12-hex prefix FIRST (short_uid takes the first 12 chars),
            # then pad zeros on the right.
            prefix = f"{0xABCDEF000000 + counter:012X}"
            hex_hash = (prefix + "0" * 64)[:64]
            (sub / f"{hex_hash}_cleaned.xlsx").touch()
            counter += 1
    out = list_cohort_users(tmp_path, sets=("M_beta_Top30", "top2000"))
    assert len(out) == 6
    sets = {row[0] for row in out}
    assert sets == {"M_beta_Top30", "top2000"}
    uids = [row[2] for row in out]
    assert len(uids) == len(set(uids)), "uids should be unique across the cohort"
    # All paths exist
    for set_name, p, uid in out:
        assert Path(p).exists()
        assert len(uid) == 12


def test_list_cohort_users_missing_set(tmp_path):
    """Missing set is silently skipped (not an error)."""
    (tmp_path / "M_beta_Top30").mkdir()
    out = list_cohort_users(tmp_path, sets=("M_beta_Top30", "top2000"))
    assert out == []


def test_list_cohort_users_data_root_missing():
    with pytest.raises(FileNotFoundError):
        list_cohort_users("/this/path/does/not/exist", sets=())


# ────────────────────────────────────────────────────────────────────────────
# build_pooled_vocab
# ────────────────────────────────────────────────────────────────────────────
def _make_train_df(events: list[tuple[str, bool]]) -> pd.DataFrame:
    """Tiny helper: events = [(app, is_target), ...]."""
    return pd.DataFrame(
        {"app_label_clean": [a for a, _ in events],
         "is_target_event": [t for _, t in events]}
    )


def test_build_pooled_vocab_reserved_tokens():
    """Reserved tokens always at indices 0/1/2 even with empty input."""
    vocab = build_pooled_vocab({}, min_count=5)
    assert vocab["<PAD>"] == 0
    assert vocab["<UNK>"] == 1
    assert vocab["<RARE>"] == 2
    assert len(vocab) == 3


def test_build_pooled_vocab_min_count_pooling():
    """An app with 3 events in user A and 2 in user B (total 5) makes the cut at min_count=5."""
    user_a = _make_train_df([("WeChat", True)] * 3 + [("LITE", True)] * 4)
    user_b = _make_train_df([("WeChat", True)] * 2 + [("LITE", True)] * 1)
    vocab = build_pooled_vocab({"a": user_a, "b": user_b}, min_count=5)
    # WeChat = 3+2 = 5  → in vocab
    # LITE   = 4+1 = 5  → in vocab
    assert "WeChat" in vocab
    assert "LITE" in vocab
    assert len(vocab) == 5  # 3 reserved + 2 apps


def test_build_pooled_vocab_min_count_below_threshold():
    """An app appearing 2 in A and 2 in B (total 4) is < min_count=5 → excluded."""
    user_a = _make_train_df([("WeChat", True)] * 2)
    user_b = _make_train_df([("WeChat", True)] * 2)
    vocab = build_pooled_vocab({"a": user_a, "b": user_b}, min_count=5)
    assert "WeChat" not in vocab
    assert len(vocab) == 3  # only reserved


def test_build_pooled_vocab_only_target_events_counted():
    """Non-target events are not counted toward the vocab threshold."""
    df = _make_train_df([("WeChat", False)] * 10 + [("WeChat", True)] * 2)
    vocab = build_pooled_vocab({"a": df}, min_count=5)
    assert "WeChat" not in vocab  # only 2 target-event occurrences


def test_build_pooled_vocab_descending_count_order():
    """Apps in vocab are ordered by descending pooled count (after reserved)."""
    user_a = _make_train_df([("LITE", True)] * 7 + [("WeChat", True)] * 2)
    user_b = _make_train_df([("LITE", True)] * 1 + [("WeChat", True)] * 4)
    vocab = build_pooled_vocab({"a": user_a, "b": user_b}, min_count=5)
    # LITE = 8 (most), WeChat = 6 (less). LITE should get smaller idx (higher rank)
    assert vocab["LITE"] < vocab["WeChat"]


# ────────────────────────────────────────────────────────────────────────────
# make_global_anchor_id
# ────────────────────────────────────────────────────────────────────────────
def test_global_anchor_id_no_collision():
    """Two users with overlapping per-user anchor_ids get unique global IDs."""
    df = pd.DataFrame({
        "user_id_idx": [0, 0, 0, 1, 1, 1, 2, 2],
        "anchor_id":   [0, 1, 2, 0, 1, 2, 0, 1],
    })
    g = make_global_anchor_id(df)
    assert len(np.unique(g)) == 8
    # Verify expected formula
    assert g[0] == 0 * GLOBAL_ANCHOR_OFFSET + 0
    assert g[3] == 1 * GLOBAL_ANCHOR_OFFSET + 0
    assert g[6] == 2 * GLOBAL_ANCHOR_OFFSET + 0


def test_global_anchor_id_dtype():
    df = pd.DataFrame({"user_id_idx": [0, 1], "anchor_id": [0, 99]})
    assert make_global_anchor_id(df).dtype == np.int64


def test_global_anchor_id_overflow_guard():
    """If a per-user anchor_id exceeds the offset, raise."""
    df = pd.DataFrame({"user_id_idx": [0], "anchor_id": [GLOBAL_ANCHOR_OFFSET + 1]})
    with pytest.raises(ValueError, match="exceeds offset"):
        make_global_anchor_id(df)


def test_global_anchor_id_missing_columns():
    df = pd.DataFrame({"anchor_id": [0]})
    with pytest.raises(KeyError):
        make_global_anchor_id(df)


# ────────────────────────────────────────────────────────────────────────────
# attach_user_columns
# ────────────────────────────────────────────────────────────────────────────
def test_attach_user_columns_basic():
    df = pd.DataFrame({"app_idx": [0, 1, 2]})
    out = attach_user_columns(df, uid="abc123", uid_idx=7)
    assert out["user_uid"].tolist() == ["abc123"] * 3
    assert out["user_id_idx"].tolist() == [7, 7, 7]
    # Original df unchanged
    assert "user_uid" not in df.columns


def test_attach_user_columns_idempotent():
    df = pd.DataFrame({"app_idx": [0, 1]})
    out1 = attach_user_columns(df, uid="x", uid_idx=3)
    out2 = attach_user_columns(out1, uid="x", uid_idx=3)
    pd.testing.assert_frame_equal(out1, out2)


def test_attach_user_columns_overwrites_existing():
    """Calling on a df that already has the columns should overwrite (not append)."""
    df = pd.DataFrame({"app_idx": [0], "user_uid": ["old"], "user_id_idx": [99]})
    out = attach_user_columns(df, uid="new", uid_idx=1)
    assert out["user_uid"].tolist() == ["new"]
    assert out["user_id_idx"].tolist() == [1]


# ────────────────────────────────────────────────────────────────────────────
# stack_per_user
# ────────────────────────────────────────────────────────────────────────────
def test_stack_per_user_2d():
    per_user = {
        "u0": {"hour_freq": np.full((24, 5), 0.0, dtype=np.float32)},
        "u1": {"hour_freq": np.full((24, 5), 1.0, dtype=np.float32)},
        "u2": {"hour_freq": np.full((24, 5), 2.0, dtype=np.float32)},
    }
    uid_to_idx = {"u0": 0, "u1": 1, "u2": 2}
    stacked = stack_per_user(per_user, "hour_freq", uid_to_idx)
    assert stacked.shape == (3, 24, 5)
    assert stacked.dtype == np.float32
    # Order respects uid_to_idx
    assert (stacked[0] == 0.0).all()
    assert (stacked[1] == 1.0).all()
    assert (stacked[2] == 2.0).all()


def test_stack_per_user_respects_idx_order():
    """If uid_to_idx maps u0 -> 1 and u1 -> 0, stacking should follow that order."""
    per_user = {
        "u0": {"v": np.array([10, 20, 30], dtype=np.float32)},
        "u1": {"v": np.array([40, 50, 60], dtype=np.float32)},
    }
    uid_to_idx = {"u0": 1, "u1": 0}
    stacked = stack_per_user(per_user, "v", uid_to_idx)
    np.testing.assert_array_equal(stacked[0], [40, 50, 60])  # u1 -> idx 0
    np.testing.assert_array_equal(stacked[1], [10, 20, 30])  # u0 -> idx 1


def test_stack_per_user_shape_mismatch_raises():
    per_user = {
        "u0": {"v": np.zeros((3,), dtype=np.float32)},
        "u1": {"v": np.zeros((4,), dtype=np.float32)},  # wrong shape
    }
    uid_to_idx = {"u0": 0, "u1": 1}
    with pytest.raises(ValueError, match="shape mismatch"):
        stack_per_user(per_user, "v", uid_to_idx)


def test_stack_per_user_non_dense_idx_raises():
    """Two uids mapping to indices 0 and 2 (skipping 1) is not dense."""
    per_user = {
        "u0": {"v": np.zeros((3,), dtype=np.float32)},
        "u1": {"v": np.zeros((3,), dtype=np.float32)},
    }
    uid_to_idx = {"u0": 0, "u1": 2}  # idx=1 is missing → not dense
    # The current impl rejects out-of-range first when U == len(uid_to_idx) < max(idx)
    with pytest.raises(ValueError):
        stack_per_user(per_user, "v", uid_to_idx)


def test_stack_per_user_lookup_by_idx():
    """Row-wise gather: stacked[uid, hr, app] == per_user[uid_str][hr, app]."""
    np.random.seed(7)
    V, U = 8, 4
    per_user = {f"u{i}": {"hf": np.random.randn(24, V).astype(np.float32)} for i in range(U)}
    uid_to_idx = {f"u{i}": i for i in range(U)}
    stacked = stack_per_user(per_user, "hf", uid_to_idx)
    # 50 random gather indices
    for _ in range(50):
        i = int(np.random.randint(U))
        h = int(np.random.randint(24))
        a = int(np.random.randint(V))
        np.testing.assert_allclose(
            stacked[i, h, a], per_user[f"u{i}"]["hf"][h, a]
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
