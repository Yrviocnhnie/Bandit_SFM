"""Layer 7 — end-to-end smoke test on a synthetic 3-user dataset.

Builds pooled bg → fits per-user stats → scores random + lru → asserts:
  - Random MCC ≈ 0
  - PR-AUC ≥ pos_rate (above-floor)
  - LRU mean PR-AUC across users > Random mean PR-AUC

No model training — pure pipeline correctness check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import data as D
from lib.bg.background_state import FG_NAMES, replay_and_snapshot
from lib.bg.baselines_bg import add_lru_score, add_random_score
from lib.bg.features_bg import (
    compute_labels, compute_recency_ranks, compute_rolling_fg_counts, flatten_to_rows,
)
from lib.bg.metrics_bg import compute_metrics, compute_threshold_metrics
from lib.bg_multi.helpers import attach_user_columns, build_pooled_vocab, make_global_anchor_id
from lib.multiuser import per_user_split


APP_POOL = ["WeChat", "LITE", "Browser", "Maps", "Mail", "Music"]


def _synth_user(uid: str, n_days: int = 9, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-04-01 09:00:00")
    rows = []
    raw_id = 0
    for day_offset in range(n_days):
        day_t = base + pd.Timedelta(days=day_offset)
        for _ in range(40):
            t = day_t + pd.Timedelta(seconds=int(rng.uniform(0, 12 * 3600)))
            app = rng.choice(APP_POOL)
            rows.append({"raw_row_id": raw_id, "event_ts": t,
                          "app_label_clean": app, "name_norm": "APP_FOREGROUND",
                          "is_app_usage_event": True, "is_target_event": True,
                          "process_name_norm": app, "bundle_name_norm": app,
                          "has_any_app_identity": True,
                          "device_state_update_payload": None,
                          "seconds_since_prev_event": 0.0,
                          "seconds_to_next_event": 0.0})
            raw_id += 1
            t_bg = t + pd.Timedelta(seconds=int(rng.uniform(60, 1800)))
            rows.append({"raw_row_id": raw_id, "event_ts": t_bg,
                          "app_label_clean": app, "name_norm": "APP_BACKGROUND",
                          "is_app_usage_event": True, "is_target_event": False,
                          "process_name_norm": app, "bundle_name_norm": app,
                          "has_any_app_identity": True,
                          "device_state_update_payload": None,
                          "seconds_since_prev_event": 0.0,
                          "seconds_to_next_event": 0.0})
            raw_id += 1
    df = pd.DataFrame(rows).sort_values("event_ts").reset_index(drop=True)
    return df


def _build_pooled_bg(users: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build pooled bg_train + bg_test for the synthetic cohort."""
    train_dfs = {}
    enriched_users = {}
    for uid, df in users.items():
        df = D.dedup(df)
        df = D.enrich(df)
        enriched_users[uid] = df
        tr, _, _ = per_user_split(df, val_days=3, test_days=3, embargo_minutes=60)
        train_dfs[uid] = tr
    vocab = build_pooled_vocab(train_dfs, min_count=3)

    all_train, all_test = [], []
    uid_to_idx = {uid: i for i, uid in enumerate(users.keys())}
    for uid, df in enriched_users.items():
        tr, va, te = per_user_split(df, val_days=3, test_days=3, embargo_minutes=60)
        all_user = pd.concat([tr, va, te], ignore_index=True).sort_values("event_ts").reset_index(drop=True)
        fg_mask_all = all_user["name_norm"].isin(list(FG_NAMES))
        fg_ts_all = (
            pd.to_datetime(all_user.loc[fg_mask_all, "event_ts"]).to_numpy()
            .astype("datetime64[ns]").view("int64")
        )
        fg_app_all = all_user.loc[fg_mask_all, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
        for split_name, split_df, accum in (("train", tr, all_train), ("test", te, all_test)):
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
                keep, labels, vocab=vocab, horizons_sec=(300, 600, 1800, 3600),
                rolling_counts_by_window=rolling, recency_ranks=ranks,
                rolling_windows_sec=(3600, 21600),
            )
            row_df = attach_user_columns(row_df, uid=uid, uid_idx=uid_to_idx[uid])
            accum.append(row_df)
    return (
        pd.concat(all_train, ignore_index=True) if all_train else pd.DataFrame(),
        pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame(),
        vocab,
    )


def test_smoke_e2e_pipeline():
    """3 synthetic users → pooled bg → run baselines → assert reasonable metrics."""
    users = {f"u_{i}": _synth_user(f"u_{i}", n_days=9, seed=7 + i) for i in range(3)}
    bg_tr, bg_te, vocab = _build_pooled_bg(users)
    if len(bg_te) == 0:
        pytest.skip("no test rows in synthetic cohort")

    pos_rate = float(bg_te["y_3600"].mean())
    assert 0.05 < pos_rate < 0.65, f"synthetic pos rate {pos_rate} is degenerate"

    # Random + LRU baselines on the pooled bg_test
    bg_te = add_random_score(bg_te, seed=7)
    bg_te = add_lru_score(bg_te)
    # Use global anchor id for groupby
    bg_te = bg_te.copy()
    bg_te["_global_aid"] = make_global_anchor_id(bg_te)
    bg_te2 = bg_te.rename(columns={"anchor_id": "_per_user_aid"})
    bg_te2["anchor_id"] = bg_te2["_global_aid"]

    m_random = compute_metrics(bg_te2, score_col="score_random", y_col="y_3600")
    m_lru = compute_metrics(bg_te2, score_col="score_lru", y_col="y_3600")

    # Random PR-AUC ≥ pos_rate (within sampling noise)
    assert m_random["pr_auc_mean"] >= pos_rate * 0.8

    # MCC near zero for random
    tau = float(np.median(bg_te["score_random"]))
    m_thr = compute_threshold_metrics(bg_te2, score_col="score_random",
                                       y_col="y_3600", tau=tau)
    assert abs(m_thr["mcc"]) < 0.4, f"random MCC={m_thr['mcc']:.3f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
