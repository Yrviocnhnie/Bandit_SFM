"""Build pooled multi-user Task C training data.

For each of the 22 users at /data00/ruiqing/app_forecasting/data/cleaned/:

  1. Load + enrich the XLSX (or reuse an existing per-user enriched parquet).
  2. Per-user chronological split: last 3 days = test, prior 3 days = val,
     rest = train, 60-min embargo at train↔val boundary.
  3. Build a pooled global vocab from train target events across all users
     (count >= POOLED_VOCAB_MIN_COUNT, default 5).
  4. Per-user replay_and_snapshot + label compute + flatten_to_rows. Same-split
     FG events for label causality, full-stream for rolling counts (causal).
  5. Per-user feature-stat fits (Markov, hour_freq, lifetime stats, etc.) on
     train rows of that user only. Saved as artifacts/bg_multi/per_user/<uid>/stats.pkl.
  6. Concat per-user bg dfs → pooled artifacts/bg_multi/splits/bg_{train,val,test}.parquet.

Outputs:
  artifacts/bg_multi/vocab.json
  artifacts/bg_multi/uid_to_idx.json
  artifacts/bg_multi/splits/bg_{train,val,test}.parquet
  artifacts/bg_multi/per_user/<uid>/stats.pkl
  artifacts/bg_multi/stats/bg_data_stats.json

CLI:
  python scripts/50_build_bg_data_multiuser.py
  python scripts/50_build_bg_data_multiuser.py --reuse-existing-splits
  python scripts/50_build_bg_data_multiuser.py --max-users 3   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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
    list_cohort_users,
    save_per_user_stats,
)
from lib.multiuser import load_xlsx_with_padding, per_user_split
from lib.v3 import categories as CAT


# ============================================================================
# Constants
# ============================================================================
POOLED_VOCAB_MIN_COUNT = 5
ANCHOR_STRIDE_SEC = 300
HOUR_START, HOUR_END = 6, 24
HORIZONS_SEC = (300, 600, 1800, 3600)
ROLLING_WINDOWS_SEC = (3600, 21600)
T_STALE_SEC = 2 * 3600
VAL_DAYS = 3
TEST_DAYS = 3
EMBARGO_MINUTES = 60


# ============================================================================
# Per-user data loading
# ============================================================================
def load_user_events(set_name: str, xlsx_path: Path, uid: str,
                     reuse_existing_splits: bool, art_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load + enrich one user's events; apply per-user split.

    If `reuse_existing_splits`, try `artifacts/multiuser/<set>/<uid>/splits/*.parquet`
    first (already-enriched, already-split — saves ~3 min over re-loading XLSX).
    """
    existing_dir = art_dir / "multiuser" / set_name / uid / "splits"
    if reuse_existing_splits and existing_dir.exists():
        try:
            train = pd.read_parquet(existing_dir / "train.parquet")
            val = pd.read_parquet(existing_dir / "val.parquet")
            test = pd.read_parquet(existing_dir / "test.parquet")
            return train, val, test
        except Exception as e:
            print(f"  ⚠ couldn't reuse existing splits at {existing_dir}: {e}")
            print(f"  ⚠ falling back to XLSX")

    raw = load_xlsx_with_padding(str(xlsx_path))
    raw = D.dedup(raw)
    enriched = D.enrich(raw)
    train, val, test = per_user_split(
        enriched, val_days=VAL_DAYS, test_days=TEST_DAYS,
        embargo_minutes=EMBARGO_MINUTES,
    )
    return train, val, test


def build_user_bg_split(
    user_train: pd.DataFrame,
    user_val: pd.DataFrame,
    user_test: pd.DataFrame,
    vocab: dict,
) -> dict[str, pd.DataFrame]:
    """Replay BG state + compute labels + flatten to rows, per split.

    Mirrors single-user `30_build_bg_data.py` but per user. Returns
    {"train": bg_train_df, "val": bg_val_df, "test": bg_test_df}.
    """
    all_user = (
        pd.concat([user_train, user_val, user_test], ignore_index=True)
        .sort_values("event_ts").reset_index(drop=True)
    )
    fg_mask_all = all_user["name_norm"].isin(list(FG_NAMES))
    fg_ts_all = (
        pd.to_datetime(all_user.loc[fg_mask_all, "event_ts"]).to_numpy()
        .astype("datetime64[ns]").view("int64")
    )
    fg_app_all = (
        all_user.loc[fg_mask_all, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    )

    out: dict[str, pd.DataFrame] = {}
    for split_name, split_df in (("train", user_train), ("val", user_val), ("test", user_test)):
        if len(split_df) == 0:
            out[split_name] = pd.DataFrame()
            continue
        anchors = D.anchor_grid(
            split_df, stride_sec=ANCHOR_STRIDE_SEC, hour_start=HOUR_START, hour_end=HOUR_END,
        )
        anchor_ts = anchors["anchor_ts"].to_numpy()

        fg_mask = split_df["name_norm"].isin(list(FG_NAMES))
        fg_ts_split = (
            pd.to_datetime(split_df.loc[fg_mask, "event_ts"]).to_numpy()
            .astype("datetime64[ns]").view("int64")
        )
        fg_app_split = (
            split_df.loc[fg_mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
        )

        # Replay state on full per-user stream → snapshot at split anchors
        snaps = replay_and_snapshot(all_user, anchor_ts, t_stale_sec=T_STALE_SEC)
        keep = [s for s in snaps if len(s.apps) >= 1]
        if len(keep) == 0:
            out[split_name] = pd.DataFrame()
            continue

        labels = compute_labels(
            keep, fg_event_ts_ns=fg_ts_split, fg_event_app=fg_app_split,
            horizons_sec=HORIZONS_SEC,
        )
        rolling = compute_rolling_fg_counts(
            keep, fg_event_ts_ns=fg_ts_all, fg_event_app=fg_app_all,
            windows_sec=ROLLING_WINDOWS_SEC,
        )
        ranks = compute_recency_ranks(keep)
        df = flatten_to_rows(
            keep, labels, vocab=vocab,
            horizons_sec=HORIZONS_SEC,
            rolling_counts_by_window=rolling,
            recency_ranks=ranks,
            rolling_windows_sec=ROLLING_WINDOWS_SEC,
        )
        out[split_name] = df
    return out


# ============================================================================
# Main pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=str,
        default="/data00/ruiqing/app_forecasting/data/cleaned",
    )
    parser.add_argument(
        "--art-dir", type=str,
        default=str(ROOT / "artifacts"),
        help="Root artifacts dir; bg_multi/ goes inside.",
    )
    parser.add_argument(
        "--reuse-existing-splits", action="store_true",
        help="Reuse artifacts/multiuser/<set>/<uid>/splits/*.parquet if available.",
    )
    parser.add_argument(
        "--max-users", type=int, default=None,
        help="Smoke-test mode: only process the first N users (sorted).",
    )
    parser.add_argument(
        "--min-count", type=int, default=POOLED_VOCAB_MIN_COUNT,
        help="Pooled-vocab threshold (default 5).",
    )
    args = parser.parse_args()

    art_dir = Path(args.art_dir)
    bg_multi = art_dir / "bg_multi"
    bg_multi.mkdir(parents=True, exist_ok=True)
    (bg_multi / "splits").mkdir(parents=True, exist_ok=True)
    (bg_multi / "stats").mkdir(parents=True, exist_ok=True)
    (bg_multi / "per_user").mkdir(parents=True, exist_ok=True)

    # 0. Enumerate cohort
    cohort = list_cohort_users(args.data_root)
    if args.max_users is not None:
        cohort = cohort[: args.max_users]
    if not cohort:
        print(f"[50] no users found under {args.data_root}", file=sys.stderr)
        return 1
    print(f"[50] cohort: {len(cohort)} users")

    uid_to_idx: dict[str, int] = {uid: i for i, (_, _, uid) in enumerate(cohort)}
    with open(bg_multi / "uid_to_idx.json", "w") as f:
        json.dump(uid_to_idx, f, indent=2)

    # 1. Pass 1 — load + per-user splits
    print(f"[50] Pass 1 — loading + per-user splits ...")
    per_user_data: dict[str, dict[str, pd.DataFrame]] = {}
    set_name_for_uid: dict[str, str] = {}
    t0 = time.time()
    for set_name, xlsx_path, uid in cohort:
        try:
            tr, va, te = load_user_events(
                set_name, xlsx_path, uid,
                reuse_existing_splits=args.reuse_existing_splits, art_dir=art_dir,
            )
        except Exception as e:
            print(f"  ⚠ failed to load uid={uid}: {e}; skipping", file=sys.stderr)
            continue
        if len(tr) == 0:
            print(f"  ⚠ uid={uid} has 0 train events after split; skipping", file=sys.stderr)
            continue
        per_user_data[uid] = {"train": tr, "val": va, "test": te}
        set_name_for_uid[uid] = set_name
        train_days = (
            (pd.to_datetime(tr["event_ts"]).max() - pd.to_datetime(tr["event_ts"]).min())
            .total_seconds() / 86400.0
        )
        print(
            f"  uid={uid} set={set_name}  train={len(tr):>7} ({train_days:.1f}d)  "
            f"val={len(va):>5}  test={len(te):>5}"
        )
    print(f"[50] Pass 1 done in {time.time() - t0:.1f}s; {len(per_user_data)} users kept")

    if not per_user_data:
        print("[50] no users with train data; aborting", file=sys.stderr)
        return 1

    # Re-build uid_to_idx in case we dropped anyone
    uid_to_idx = {uid: i for i, uid in enumerate(per_user_data.keys())}
    with open(bg_multi / "uid_to_idx.json", "w") as f:
        json.dump(uid_to_idx, f, indent=2)

    # 2. Pass 2 — pooled vocab from train target events
    print(f"[50] Pass 2 — building pooled vocab (min_count={args.min_count}) ...")
    train_dfs = {uid: d["train"] for uid, d in per_user_data.items()}
    vocab = build_pooled_vocab(train_dfs, min_count=args.min_count)
    with open(bg_multi / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"[50] pooled vocab size = {len(vocab)} (incl. PAD/UNK/RARE)")

    app_to_cat = CAT.build_app_to_cat_idx(vocab)

    # 3. Pass 3 — per-user replay + flatten
    print(f"[50] Pass 3 — replay + label + flatten per user ...")
    all_bg_train: list[pd.DataFrame] = []
    all_bg_val: list[pd.DataFrame] = []
    all_bg_test: list[pd.DataFrame] = []
    per_user_summary: dict[str, dict] = {}
    t1 = time.time()
    for uid, d in per_user_data.items():
        uid_idx = uid_to_idx[uid]
        bg = build_user_bg_split(d["train"], d["val"], d["test"], vocab)
        for split_name, accum in (("train", all_bg_train),
                                  ("val", all_bg_val),
                                  ("test", all_bg_test)):
            df = bg[split_name]
            if len(df) == 0:
                continue
            df = attach_user_columns(df, uid=uid, uid_idx=uid_idx)
            accum.append(df)
        per_user_summary[uid] = {
            "set": set_name_for_uid[uid],
            "uid_idx": uid_idx,
            "n_train_rows": int(len(bg.get("train", pd.DataFrame()))),
            "n_val_rows":   int(len(bg.get("val",   pd.DataFrame()))),
            "n_test_rows":  int(len(bg.get("test",  pd.DataFrame()))),
        }
        print(
            f"  uid={uid} bg rows: train={per_user_summary[uid]['n_train_rows']:>6} "
            f"val={per_user_summary[uid]['n_val_rows']:>4} "
            f"test={per_user_summary[uid]['n_test_rows']:>4}"
        )
    print(f"[50] Pass 3 done in {time.time() - t1:.1f}s")

    # Concat pooled
    bg_train = pd.concat(all_bg_train, ignore_index=True) if all_bg_train else pd.DataFrame()
    bg_val   = pd.concat(all_bg_val,   ignore_index=True) if all_bg_val   else pd.DataFrame()
    bg_test  = pd.concat(all_bg_test,  ignore_index=True) if all_bg_test  else pd.DataFrame()

    # 4. Pass 4 — per-user feature stats (need the user's bg_train rows for kill_rate)
    print(f"[50] Pass 4 — per-user feature stats ...")
    t2 = time.time()
    for uid, d in per_user_data.items():
        uid_idx = uid_to_idx[uid]
        all_user = pd.concat([d["train"], d["val"], d["test"]], ignore_index=True) \
                     .sort_values("event_ts").reset_index(drop=True)
        bg_train_user = (
            bg_train[bg_train["user_id_idx"] == uid_idx]
            if "user_id_idx" in bg_train.columns else pd.DataFrame()
        )
        try:
            stats = fit_per_user_stats(
                uid=uid,
                train_events=d["train"],
                bg_train_user=bg_train_user,
                vocab=vocab,
                app_to_cat=app_to_cat,
                all_events_user=all_user,
                scripts_dir=ROOT / "scripts",
            )
        except Exception as e:
            print(f"  ⚠ failed per-user fit for uid={uid}: {e}", file=sys.stderr)
            continue
        save_per_user_stats(stats, bg_multi / "per_user" / uid / "stats.pkl")
        # Causality assertion
        max_train_ts = pd.to_datetime(d["train"]["event_ts"]).max()
        per_user_summary[uid]["train_max_ts"] = str(max_train_ts)
    print(f"[50] Pass 4 done in {time.time() - t2:.1f}s")

    # 5. Write pooled bg parquets
    if len(bg_train):
        bg_train.to_parquet(bg_multi / "splits" / "bg_train.parquet", index=False)
    if len(bg_val):
        bg_val.to_parquet(bg_multi / "splits" / "bg_val.parquet", index=False)
    if len(bg_test):
        bg_test.to_parquet(bg_multi / "splits" / "bg_test.parquet", index=False)
    print(
        f"[50] pooled rows: train={len(bg_train):,}  "
        f"val={len(bg_val):,}  test={len(bg_test):,}"
    )

    # 6. Sanity stats
    def _per_user_pos_rate(df: pd.DataFrame) -> dict[str, float]:
        if len(df) == 0:
            return {}
        return df.groupby("user_id_idx")["y_3600"].mean().to_dict()

    def _per_user_anchor_count(df: pd.DataFrame) -> dict[str, int]:
        if len(df) == 0:
            return {}
        return df.groupby("user_id_idx")["anchor_id"].nunique().to_dict()

    def _vocab_coverage_per_user(df: pd.DataFrame, vocab_size: int) -> dict[int, float]:
        if len(df) == 0:
            return {}
        out = {}
        rare_idx = vocab.get("<RARE>", 2)
        for uid_idx, sub in df.groupby("user_id_idx"):
            n = len(sub)
            if n == 0:
                continue
            n_rare = int((sub["app_idx"].to_numpy() == rare_idx).sum())
            out[int(uid_idx)] = float(1.0 - n_rare / n)
        return out

    stats_out = {
        "n_users":             int(len(per_user_data)),
        "vocab_size":          int(len(vocab)),
        "min_count":           int(args.min_count),
        "anchor_stride_sec":   ANCHOR_STRIDE_SEC,
        "hour_range":          [HOUR_START, HOUR_END],
        "horizons_sec":        list(HORIZONS_SEC),
        "t_stale_sec":         T_STALE_SEC,
        "val_days":            VAL_DAYS,
        "test_days":           TEST_DAYS,
        "embargo_minutes":     EMBARGO_MINUTES,
        "splits": {
            "train": {
                "n_rows": int(len(bg_train)),
                "pos_rate_3600": float(bg_train["y_3600"].mean()) if len(bg_train) else None,
                "per_user_pos_rate": {str(k): float(v) for k, v in _per_user_pos_rate(bg_train).items()},
                "per_user_anchor_count": {str(k): int(v) for k, v in _per_user_anchor_count(bg_train).items()},
                "per_user_vocab_coverage": {str(k): float(v) for k, v in _vocab_coverage_per_user(bg_train, len(vocab)).items()},
            },
            "val": {
                "n_rows": int(len(bg_val)),
                "pos_rate_3600": float(bg_val["y_3600"].mean()) if len(bg_val) else None,
                "per_user_pos_rate": {str(k): float(v) for k, v in _per_user_pos_rate(bg_val).items()},
                "per_user_anchor_count": {str(k): int(v) for k, v in _per_user_anchor_count(bg_val).items()},
            },
            "test": {
                "n_rows": int(len(bg_test)),
                "pos_rate_3600": float(bg_test["y_3600"].mean()) if len(bg_test) else None,
                "per_user_pos_rate": {str(k): float(v) for k, v in _per_user_pos_rate(bg_test).items()},
                "per_user_anchor_count": {str(k): int(v) for k, v in _per_user_anchor_count(bg_test).items()},
                "per_user_vocab_coverage": {str(k): float(v) for k, v in _vocab_coverage_per_user(bg_test, len(vocab)).items()},
            },
        },
        "per_user": per_user_summary,
    }
    with open(bg_multi / "stats" / "bg_data_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"[50] DONE  stats -> {bg_multi / 'stats' / 'bg_data_stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
