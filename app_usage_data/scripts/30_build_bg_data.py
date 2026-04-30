"""Build Task C training / eval data from the enriched event stream.

Pipeline:
  1. Reload existing train/val/test split parquets (v1 artifacts).
  2. Concatenate them back to one chronologically sorted event stream so the
     state machine can be seeded from full history when building snapshots
     for val/test anchors. (This preserves zero leakage: we use the full
     event stream only for state reconstruction; labels for each split come
     from that split's FG events only.)
  3. For each split, build a 5-min anchor grid restricted to that split's
     calendar day range (inherits from v1 Task B anchor protocol).
  4. For each anchor: compute B(t), per-app features, labels at H=5 / H=10.
  5. Drop anchors with |B(t)| == 0; write flat (anchor, app) rows to
     artifacts/bg/splits/bg_{train,val,test}.parquet.
  6. Emit artifacts/bg/stats.json with |B(t)| distribution and positive-rate
     sanity checks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg.background_state import (
    FG_NAMES, replay_and_snapshot, T_STALE_SEC_DEFAULT,
)
from lib.bg.features_bg import (
    compute_labels, compute_rolling_fg_counts, compute_recency_ranks,
    flatten_to_rows,
)
from lib import data as D


ANCHOR_STRIDE_SEC = 300
HOUR_START = 6
HOUR_END = 24
# Multi-horizon: 5/10/30/60 min. H=3600 (1 hour) is the new headline.
HORIZONS_SEC = (300, 600, 1800, 3600)
ROLLING_WINDOWS_SEC = (3600, 21600)   # 1 h, 6 h per-app FG counts
# 2-hour staleness window (was 6h in v1) — see REPORT_bgkill_data.md §0.
T_STALE_SEC = 2 * 3600


def main() -> int:
    art = ROOT / "artifacts"
    bg_art = art / "bg"
    bg_art.mkdir(parents=True, exist_ok=True)
    (bg_art / "splits").mkdir(parents=True, exist_ok=True)

    with open(art / "vocab.json") as f:
        vocab = json.load(f)

    # Load the three splits (already enriched) and concatenate for state replay.
    train = pd.read_parquet(art / "splits" / "train.parquet")
    val = pd.read_parquet(art / "splits" / "val.parquet")
    test = pd.read_parquet(art / "splits" / "test.parquet")
    all_df = pd.concat([train, val, test], ignore_index=True).sort_values("event_ts").reset_index(drop=True)

    print(f"[bg/30] events: train={len(train)}  val={len(val)}  test={len(test)}  all={len(all_df)}")
    print(f"[bg/30] BG events: {(all_df['name_norm'] == 'APP_BACKGROUND').sum()}")
    print(f"[bg/30] FG events: {all_df['name_norm'].isin(FG_NAMES).sum()}")
    print(f"[bg/30] PROCESS_EXIT events: {(all_df['name_norm'] == 'PROCESS_EXIT').sum()}")

    # NOTE: B(t) state machine uses the FULL event stream (all_df) for cross-split
    # context. Labels are computed using SAME-SPLIT FG events only to avoid
    # cross-split leakage at H=60 (horizon equals embargo width).
    # Rolling-count features use the full FG stream (always backward-looking, safe).
    fg_mask_all = all_df["name_norm"].isin(list(FG_NAMES))
    fg_ts_all = pd.to_datetime(all_df.loc[fg_mask_all, "event_ts"]).to_numpy().astype("datetime64[ns]").view("int64")
    fg_app_all = all_df.loc[fg_mask_all, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()

    split_summary = {}
    for split_name, split_df in (("train", train), ("val", val), ("test", test)):
        print(f"[bg/30] building split={split_name} ...")
        # 5-min anchor grid on the split's own day range
        anchors = D.anchor_grid(split_df, stride_sec=300, hour_start=HOUR_START, hour_end=HOUR_END)
        anchor_ts = anchors["anchor_ts"].to_numpy()
        print(f"[bg/30]   raw anchors = {len(anchor_ts)}")

        # Per-split FG events for label computation — avoids cross-split leakage.
        fg_mask = split_df["name_norm"].isin(list(FG_NAMES))
        fg_ts_split = pd.to_datetime(split_df.loc[fg_mask, "event_ts"]).to_numpy().astype("datetime64[ns]").view("int64")
        fg_app_split = split_df.loc[fg_mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()

        # Replay state using the FULL stream (no leakage — only past), snapshot at split anchors.
        snaps = replay_and_snapshot(all_df, anchor_ts, t_stale_sec=T_STALE_SEC)

        # Keep only anchors with |B(t)| >= 1
        keep = [s for s in snaps if len(s.apps) >= 1]
        dropped = len(snaps) - len(keep)
        print(f"[bg/30]   dropped {dropped} anchors with |B(t)| == 0; kept {len(keep)}")

        # Labels at H=5 / H=10 / H=30 / H=60 — same-split FG events only
        labels = compute_labels(keep, fg_event_ts_ns=fg_ts_split,
                                fg_event_app=fg_app_split,
                                horizons_sec=HORIZONS_SEC)

        # Per-app rolling FG counts: full stream (backward-looking is safe)
        rolling_counts = compute_rolling_fg_counts(
            keep, fg_event_ts_ns=fg_ts_all, fg_event_app=fg_app_all,
            windows_sec=ROLLING_WINDOWS_SEC,
        )
        ranks = compute_recency_ranks(keep)

        df = flatten_to_rows(keep, labels, vocab=load_vocab(art),
                             horizons_sec=HORIZONS_SEC,
                             rolling_counts_by_window=rolling_counts,
                             recency_ranks=ranks,
                             rolling_windows_sec=ROLLING_WINDOWS_SEC)
        out_path = bg_art / "splits" / f"bg_{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[bg/30]   wrote {out_path}  rows={len(df):,}")

        split_summary[split_name] = {
            "n_anchors_kept": int(len(keep)),
            "n_rows": int(len(df)),
            "bg_size_mean": float(np.mean([len(s.apps) for s in keep])),
            "bg_size_p50": int(np.median([len(s.apps) for s in keep])),
            "bg_size_p95": int(np.quantile([len(s.apps) for s in keep], 0.95)),
            "bg_size_max": int(max((len(s.apps) for s in keep), default=0)),
            "pos_rate_300": float(df["y_300"].mean()),
            "pos_rate_600": float(df["y_600"].mean()),
            "pos_rate_1800": float(df["y_1800"].mean()),
            "pos_rate_3600": float(df["y_3600"].mean()),
            "dropped_empty_anchors": int(dropped),
        }
        print(f"[bg/30]   summary: {split_summary[split_name]}")

    stats_path = bg_art / "stats" / "bg_data_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({
            "t_stale_sec": T_STALE_SEC,
            "anchor_stride_sec": ANCHOR_STRIDE_SEC,
            "hour_range": [HOUR_START, HOUR_END],
            "splits": split_summary,
        }, f, indent=2)
    print(f"[bg/30] DONE  stats -> {stats_path}")
    return 0


def load_vocab(art_dir):
    with open(art_dir / "vocab.json") as f:
        return json.load(f)


if __name__ == "__main__":
    import json
    sys.exit(main() or 0)
