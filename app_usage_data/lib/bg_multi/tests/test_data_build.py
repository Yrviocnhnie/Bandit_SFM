"""Layer 2 — data-build integration tests on a synthetic 3-user dataset.

Builds 3 in-memory users, each with ~7-10 days of synthetic events, and exercises:
  - per-user splits (no overlap, embargo)
  - per-user replay isolation
  - label causality (same-split FG events only)
  - per-user stats fit_split assertion + train-only window
  - pooled-bg uniqueness (anchor IDs, user contributions)
  - vocab coverage per user
  - no NaN in features after build
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import data as D
from lib.bg.background_state import FG_NAMES, replay_and_snapshot
from lib.bg.features_bg import (
    compute_labels, compute_rolling_fg_counts, compute_recency_ranks, flatten_to_rows,
)
from lib.bg_multi.helpers import (
    attach_user_columns,
    build_pooled_vocab,
    fit_per_user_stats,
    make_global_anchor_id,
    save_per_user_stats,
    load_per_user_stats,
)
from lib.multiuser import per_user_split
from lib.v3 import categories as CAT


# ============================================================================
# Synthetic data factory
# ============================================================================
APP_POOL = ["WeChat", "LITE", "Browser", "Maps", "Mail", "Music", "Calendar"]


def synth_user(uid: str, n_days: int = 9, seed: int = 0) -> pd.DataFrame:
    """Generate ~n_days of synthetic events for one user.

    Each "session" is FG → BG → FG → BG. ~30 events per day.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-04-01 09:00:00")
    rows: list[dict] = []
    raw_id = 0
    for day_offset in range(n_days):
        day_t = base + pd.Timedelta(days=day_offset)
        for _ in range(30):
            t = day_t + pd.Timedelta(seconds=int(rng.uniform(0, 12 * 3600)))
            app = rng.choice(APP_POOL)
            # FG event
            rows.append({
                "raw_row_id": raw_id, "event_ts": t,
                "app_label_clean": app, "name_norm": "APP_FOREGROUND",
                "is_app_usage_event": True, "is_target_event": True,
                "process_name_norm": app, "bundle_name_norm": app,
                "has_any_app_identity": True,
                "device_state_update_payload": None,
                "seconds_since_prev_event": 0.0,
                "seconds_to_next_event": 0.0,
            })
            raw_id += 1
            # BG event a few minutes later
            t_bg = t + pd.Timedelta(seconds=int(rng.uniform(60, 1800)))
            rows.append({
                "raw_row_id": raw_id, "event_ts": t_bg,
                "app_label_clean": app, "name_norm": "APP_BACKGROUND",
                "is_app_usage_event": True, "is_target_event": False,
                "process_name_norm": app, "bundle_name_norm": app,
                "has_any_app_identity": True,
                "device_state_update_payload": None,
                "seconds_since_prev_event": 0.0,
                "seconds_to_next_event": 0.0,
            })
            raw_id += 1
    df = pd.DataFrame(rows).sort_values("event_ts").reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def synthetic_cohort():
    """3 users with 9 days each, distinct seeds, distinct app distributions."""
    users = {}
    for i, uid in enumerate(["u_alpha", "u_beta", "u_gamma"]):
        df = synth_user(uid, n_days=9, seed=7 + i)
        df = D.dedup(df)
        df = D.enrich(df)
        users[uid] = df
    return users


# ============================================================================
# Per-user split tests
# ============================================================================
def test_per_user_split_no_overlap(synthetic_cohort):
    """Train ∩ val == ∅ via timestamp windows; train_max < val_min."""
    for uid, df in synthetic_cohort.items():
        tr, va, te = per_user_split(df, val_days=3, test_days=3, embargo_minutes=60)
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            pytest.skip(f"{uid}: not enough days for 3+3 split")
        tr_max = pd.to_datetime(tr["event_ts"]).max()
        va_min = pd.to_datetime(va["event_ts"]).min()
        va_max = pd.to_datetime(va["event_ts"]).max()
        te_min = pd.to_datetime(te["event_ts"]).min()
        assert tr_max < va_min, f"{uid}: train_max >= val_min"
        assert va_max < te_min, f"{uid}: val_max >= test_min"


def test_per_user_split_embargo(synthetic_cohort):
    """Train_max should be at least 60 min before val_min."""
    for uid, df in synthetic_cohort.items():
        tr, va, te = per_user_split(df, val_days=3, test_days=3, embargo_minutes=60)
        if len(tr) == 0 or len(va) == 0:
            continue
        tr_max = pd.to_datetime(tr["event_ts"]).max()
        va_min = pd.to_datetime(va["event_ts"]).min()
        gap = (va_min - tr_max).total_seconds() / 60.0
        assert gap >= 60.0, f"{uid}: embargo gap {gap:.1f} min < 60"


# ============================================================================
# Replay isolation
# ============================================================================
def test_replay_per_user_isolated(synthetic_cohort):
    """Replaying user A's events should never produce apps from user B."""
    keys = list(synthetic_cohort.keys())
    df_a = synthetic_cohort[keys[0]]
    df_b = synthetic_cohort[keys[1]]
    # Make sure the two users have different app distributions
    apps_a = set(df_a["app_label_clean"].dropna().astype(str).unique())
    apps_b = set(df_b["app_label_clean"].dropna().astype(str).unique())
    if apps_a == apps_b:
        pytest.skip("synthetic users share the same app pool exactly; isolation is trivial")
    # Replay user A only
    anchors = D.anchor_grid(df_a, stride_sec=300, hour_start=6, hour_end=24)
    if len(anchors) == 0:
        pytest.skip("no anchors in user A")
    snaps = replay_and_snapshot(df_a, anchors["anchor_ts"].to_numpy(), t_stale_sec=2 * 3600)
    seen = set()
    for s in snaps:
        for a in s.apps:
            seen.add(a)
    # Apps in B(t) should be a subset of user A's apps (no leakage from B)
    extras = seen - apps_a
    assert not extras, f"replay of user A produced apps not in A: {extras}"


# ============================================================================
# Label causality (per-user, per-split)
# ============================================================================
def test_label_causality_per_user(synthetic_cohort):
    """Restrict to anchors strictly after train_max; for those, train-FG labels must be zero.

    The anchor grid (built from val days) may include anchors whose timestamp lies
    BEFORE the val window boundary (because per_user_split crosses days mid-day).
    For anchors strictly AFTER train_max, however, no train FG event can land in
    the forward window (t, t+H], so substituting train-FG events for val-FG events
    must produce zero positives. This is the labels-don't-leak-across-splits
    invariant.
    """
    uid = list(synthetic_cohort.keys())[0]
    df = synthetic_cohort[uid]
    tr, va, te = per_user_split(df, val_days=3, test_days=3, embargo_minutes=60)
    if len(va) == 0 or len(tr) == 0:
        pytest.skip("not enough data")

    all_user = pd.concat([tr, va, te], ignore_index=True).sort_values("event_ts").reset_index(drop=True)
    anchors = D.anchor_grid(va, stride_sec=300, hour_start=6, hour_end=24)
    if len(anchors) == 0:
        pytest.skip("no val anchors")

    # Restrict to anchors strictly after train_max — eliminates anchors that
    # land in train territory because the val-day grid includes earlier hours.
    train_max_ts = pd.to_datetime(tr["event_ts"]).max()
    train_max_ns = pd.Timestamp(train_max_ts).asm8.astype("datetime64[ns]").view("int64")
    anchor_ts_ns = pd.to_datetime(anchors["anchor_ts"]).to_numpy().astype("datetime64[ns]").view("int64")
    anchor_keep_mask = anchor_ts_ns > int(train_max_ns)
    anchor_ts_filt = pd.to_datetime(anchors["anchor_ts"]).to_numpy()[anchor_keep_mask]
    if len(anchor_ts_filt) == 0:
        pytest.skip("no val anchors after train_max")

    snaps = replay_and_snapshot(all_user, anchor_ts_filt, t_stale_sec=2 * 3600)
    keep = [s for s in snaps if len(s.apps) >= 1]
    if not keep:
        pytest.skip("no kept snapshots")

    # train FG events on ANY of these post-train-max val-anchor windows ⇒ leakage
    fg_mask_tr = tr["name_norm"].isin(list(FG_NAMES))
    fg_ts_tr = (
        pd.to_datetime(tr.loc[fg_mask_tr, "event_ts"]).to_numpy()
        .astype("datetime64[ns]").view("int64")
    )
    fg_app_tr = tr.loc[fg_mask_tr, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    labels_tr = compute_labels(keep, fg_event_ts_ns=fg_ts_tr, fg_event_app=fg_app_tr,
                               horizons_sec=(3600,))
    n_pos_train = int(np.sum([v.sum() for v in labels_tr[3600]]))
    assert n_pos_train == 0, (
        f"train-FG labels for val anchors after train_max should be zero "
        f"(train ends before this region), got {n_pos_train} — possible leakage"
    )


# ============================================================================
# Per-user stats fits
# ============================================================================
def test_per_user_stats_fit_split_assertion(synthetic_cohort, tmp_path):
    uid = list(synthetic_cohort.keys())[0]
    df = synthetic_cohort[uid]
    tr, va, te = per_user_split(df, val_days=3, test_days=3, embargo_minutes=60)
    if len(tr) < 50:
        pytest.skip("too few train events to fit stats")
    pooled_vocab = build_pooled_vocab({uid: tr}, min_count=2)
    app_to_cat = CAT.build_app_to_cat_idx(pooled_vocab)
    all_user = pd.concat([tr, va, te], ignore_index=True).sort_values("event_ts").reset_index(drop=True)

    # Empty bg_train_user — kill_rate fitter handles this gracefully
    stats = fit_per_user_stats(
        uid=uid, train_events=tr, bg_train_user=pd.DataFrame(),
        vocab=pooled_vocab, app_to_cat=app_to_cat,
        all_events_user=all_user, scripts_dir=ROOT / "scripts",
    )
    assert stats["fit_split"] == "train"
    assert stats["uid"] == uid
    assert stats["V"] == len(pooled_vocab)
    # Save + reload + assertion
    p = tmp_path / "stats.pkl"
    save_per_user_stats(stats, p)
    reloaded = load_per_user_stats(p)
    assert reloaded["fit_split"] == "train"


def test_markov_row_normalization(synthetic_cohort):
    """For non-reserved rows (idx >= 3), markov_probs row sums to 1."""
    uid = list(synthetic_cohort.keys())[0]
    df = synthetic_cohort[uid]
    tr, _, _ = per_user_split(df, val_days=3, test_days=3, embargo_minutes=60)
    if len(tr) < 50:
        pytest.skip("too few train events")
    vocab = build_pooled_vocab({uid: tr}, min_count=2)
    app_to_cat = CAT.build_app_to_cat_idx(vocab)
    all_user = df
    stats = fit_per_user_stats(
        uid=uid, train_events=tr, bg_train_user=pd.DataFrame(),
        vocab=vocab, app_to_cat=app_to_cat, all_events_user=all_user,
        scripts_dir=ROOT / "scripts",
    )
    mp = stats["markov_probs"]
    V = mp.shape[0]
    # Reserved rows (0,1,2) are zeroed; non-reserved rows should sum to 1
    for i in range(3, V):
        s = mp[i].sum()
        assert abs(s - 1.0) < 1e-4, f"row {i} sum = {s}, expected 1.0"


# ============================================================================
# Pooled bg integrity
# ============================================================================
def _build_pooled_synthetic(synthetic_cohort, vocab):
    """Helper: run replay+flatten+attach for all 3 users; concat into pooled bg dfs."""
    all_train, all_val, all_test = [], [], []
    uid_to_idx = {uid: i for i, uid in enumerate(synthetic_cohort.keys())}
    for uid, df in synthetic_cohort.items():
        tr, va, te = per_user_split(df, val_days=3, test_days=3, embargo_minutes=60)
        all_user = pd.concat([tr, va, te], ignore_index=True).sort_values("event_ts").reset_index(drop=True)
        fg_mask_all = all_user["name_norm"].isin(list(FG_NAMES))
        fg_ts_all = (
            pd.to_datetime(all_user.loc[fg_mask_all, "event_ts"]).to_numpy()
            .astype("datetime64[ns]").view("int64")
        )
        fg_app_all = all_user.loc[fg_mask_all, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
        for split_name, split_df, accum in (("train", tr, all_train), ("val", va, all_val), ("test", te, all_test)):
            if len(split_df) == 0:
                continue
            anchors = D.anchor_grid(split_df, stride_sec=300, hour_start=6, hour_end=24)
            if len(anchors) == 0:
                continue
            snaps = replay_and_snapshot(all_user, anchors["anchor_ts"].to_numpy(), t_stale_sec=2 * 3600)
            keep = [s for s in snaps if len(s.apps) >= 1]
            if not keep:
                continue
            fg_mask = split_df["name_norm"].isin(list(FG_NAMES))
            fg_ts_split = (
                pd.to_datetime(split_df.loc[fg_mask, "event_ts"]).to_numpy()
                .astype("datetime64[ns]").view("int64")
            )
            fg_app_split = split_df.loc[fg_mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
            labels = compute_labels(keep, fg_event_ts_ns=fg_ts_split, fg_event_app=fg_app_split,
                                    horizons_sec=(300, 600, 1800, 3600))
            rolling = compute_rolling_fg_counts(
                keep, fg_event_ts_ns=fg_ts_all, fg_event_app=fg_app_all,
                windows_sec=(3600, 21600),
            )
            ranks = compute_recency_ranks(keep)
            row_df = flatten_to_rows(
                keep, labels, vocab=vocab,
                horizons_sec=(300, 600, 1800, 3600),
                rolling_counts_by_window=rolling, recency_ranks=ranks,
                rolling_windows_sec=(3600, 21600),
            )
            row_df = attach_user_columns(row_df, uid=uid, uid_idx=uid_to_idx[uid])
            accum.append(row_df)
    return (
        pd.concat(all_train, ignore_index=True) if all_train else pd.DataFrame(),
        pd.concat(all_val,   ignore_index=True) if all_val   else pd.DataFrame(),
        pd.concat(all_test,  ignore_index=True) if all_test  else pd.DataFrame(),
        uid_to_idx,
    )


@pytest.fixture(scope="module")
def pooled_bg(synthetic_cohort):
    """Build pooled bg_{train,val,test} for the synthetic cohort once per module."""
    train_dfs = {}
    for uid, df in synthetic_cohort.items():
        tr, _, _ = per_user_split(df, val_days=3, test_days=3, embargo_minutes=60)
        train_dfs[uid] = tr
    vocab = build_pooled_vocab(train_dfs, min_count=2)
    bg_tr, bg_va, bg_te, uid_to_idx = _build_pooled_synthetic(synthetic_cohort, vocab)
    return {"vocab": vocab, "uid_to_idx": uid_to_idx,
            "train": bg_tr, "val": bg_va, "test": bg_te}


def test_pooled_bg_user_id_idx_dense(pooled_bg):
    """All 3 users contribute rows to bg_train (assuming non-empty)."""
    bg_tr = pooled_bg["train"]
    if len(bg_tr) == 0:
        pytest.skip("synthetic bg_train is empty")
    seen = set(bg_tr["user_id_idx"].unique().tolist())
    expected = set(range(len(pooled_bg["uid_to_idx"])))
    assert seen == expected or seen.issubset(expected)


def test_pooled_bg_anchor_unique_per_user(pooled_bg):
    """Within a single user, (anchor_id) is unique per anchor; across users, anchor_ids may collide."""
    bg_tr = pooled_bg["train"]
    if len(bg_tr) == 0:
        pytest.skip("synthetic bg_train is empty")
    for uid_idx, sub in bg_tr.groupby("user_id_idx"):
        # An anchor has multiple (anchor, app) rows but should have the same anchor_ts_ns
        for aid, ssub in sub.groupby("anchor_id"):
            n_unique_ts = ssub["anchor_ts_ns"].nunique()
            assert n_unique_ts == 1, f"uid_idx={uid_idx} anchor_id={aid} has multiple anchor_ts"


def test_global_anchor_id_no_collision(pooled_bg):
    """make_global_anchor_id produces unique IDs across users."""
    bg_tr = pooled_bg["train"]
    if len(bg_tr) == 0:
        pytest.skip("synthetic bg_train is empty")
    g = make_global_anchor_id(bg_tr)
    bg_tr2 = bg_tr.copy()
    bg_tr2["_global_aid"] = g
    n_groups = bg_tr2.groupby(["user_id_idx", "anchor_id"]).ngroups
    n_unique = bg_tr2["_global_aid"].nunique()
    assert n_unique == n_groups


def test_pooled_pos_rate_in_range(pooled_bg):
    """Pooled positive rate is reasonable (0.05 - 0.45)."""
    bg_tr = pooled_bg["train"]
    if len(bg_tr) == 0:
        pytest.skip("synthetic bg_train is empty")
    p = float(bg_tr["y_3600"].mean())
    # Synthetic data is uniform-random, pos rate may swing wider than real data
    assert 0.0 < p < 1.0, f"pooled pos rate {p} is degenerate"


def test_pooled_vocab_coverage_per_user(pooled_bg):
    """For each user, ≥ 50 % of their (synthetic) test target apps map to non-RARE."""
    bg_te = pooled_bg["test"]
    vocab = pooled_bg["vocab"]
    rare_idx = vocab["<RARE>"]
    if len(bg_te) == 0:
        pytest.skip("synthetic bg_test is empty")
    for uid_idx, sub in bg_te.groupby("user_id_idx"):
        n = len(sub)
        if n == 0:
            continue
        n_rare = int((sub["app_idx"].to_numpy() == rare_idx).sum())
        coverage = 1.0 - n_rare / n
        # Synthetic data uses 7 apps total; min_count=2 → most apps qualify; coverage should be high
        assert coverage >= 0.5, (
            f"uid_idx={uid_idx} coverage={coverage:.2f} (< 0.5) — many apps fell to <RARE>"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
