"""Run closed-form baselines on the Task-C parquets.

Baselines:
  * Random          — sanity floor
  * LRU             — score = time_since_last_fg_sec  (oldest FG -> most killable)
  * TimeInBG        — score = time_in_bg_sec          (oldest BG -> most killable)
  * LFU-hour        — score = 1 - P(app | anchor_hour); table fit on train
  * Markov-inverse  — score = 1 - P(app | last_fg_app); v3 Markov prior (train)

TaskB-inverse is handled in 32_train_task_c.py C0 because it needs full
v3 R6 inference machinery — easier to share code paths with training.

Output: artifacts/bg/results/baselines_bg.json
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

from lib.bg import metrics_bg as MET
from lib.bg.baselines_bg import (
    add_lru_score,
    add_random_score,
    add_time_in_bg_score,
    add_lfu_hour_score,
    add_markov_inverse,
)
from lib.v3 import markov_prior as MK


HORIZONS_SEC = (300, 600)
R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)


def fit_hour_freq(train_df: pd.DataFrame, vocab: dict) -> np.ndarray:
    """P(app | hour), Dirichlet-smoothed, fit on train target events only."""
    V = len(vocab)
    rare_idx = vocab.get("<RARE>", 2)
    table = np.zeros((24, V), dtype=np.float64)
    mask = train_df["is_target_event"].astype(bool).to_numpy()
    hrs = pd.to_datetime(train_df.loc[mask, "event_ts"]).dt.hour.to_numpy()
    apps = train_df.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare_idx) for a in apps], dtype=np.int64)
    for h, a in zip(hrs, idx):
        if 0 <= h < 24 and 3 <= a < V:
            table[int(h), int(a)] += 1.0
    table += 0.5
    table[:, :3] = 0.0
    row_sum = table.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return (table / row_sum).astype(np.float32)


def summarize(df: pd.DataFrame, score_cols: list) -> dict:
    out = {}
    for h in (300, 600):
        y_col = f"y_{h}"
        out[f"H_{h}"] = {}
        for sc in score_cols:
            m = _compute_metrics_ex(df, sc, y_col)
            out[f"H_{h}"][sc.replace("score_", "")] = m
    return out


def _compute_metrics_ex(df, score_col, y_col):
    from lib.bg import metrics_bg as M2
    return M2.compute_metrics(df, score_col, y_col, r_values=R_SWEEP)


def main() -> int:
    art = ROOT / "artifacts"
    bg_art = art / "bg"

    with open(art / "vocab.json") as f:
        vocab = json.load(f)

    train_df = pd.read_parquet(art / "splits" / "train.parquet")
    hour_freq = fit_hour_freq(train_df, vocab)

    mk_stats = MK.load_markov_prior(art / "v3" / "markov_prior.pkl")
    markov_probs = mk_stats["probs"].astype(np.float32)

    results = {}
    for split in ("train", "val", "test"):
        p = bg_art / "splits" / f"bg_{split}.parquet"
        df = pd.read_parquet(p)
        df = add_random_score(df, seed=7)
        df = add_lru_score(df)
        df = add_time_in_bg_score(df)
        df = add_lfu_hour_score(df, hour_freq=hour_freq)
        df = add_markov_inverse(df, markov_prior=markov_probs)

        score_cols = ["score_random", "score_lru", "score_tibg",
                      "score_lfu_hour", "score_markov_inv"]
        s = summarize(df, score_cols)
        results[split] = {
            "n_anchors": int(df["anchor_id"].nunique()),
            "n_rows": int(len(df)),
            "metrics": s,
        }

    out_path = bg_art / "results" / "baselines_bg.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "r_sweep": list(R_SWEEP),
            "horizons_sec": list(HORIZONS_SEC),
            "splits": results,
        }, f, indent=2)
    print(f"[bg/31] wrote {out_path}")

    # Headline print (test-only)
    print("\n=== TEST ===")
    for h in HORIZONS_SEC:
        print(f"  H={h} sec")
        for name, m in results["test"]["metrics"][f"H_{h}"].items():
            fk = m["false_kill_rate"]["0.5"]
            msr = m["memory_save_rate"]["0.5"]
            print(f"    {name:16s}  FK@0.5={fk:.4f}  MSR@0.5={msr:.4f}  "
                  f"PR-AUC={m['pr_auc_mean']:.4f}  ROC-AUC={m['roc_auc_mean']:.4f}  "
                  f"NDCG@half={m['ndcg_half']:.4f}")
    return 0


if __name__ == "__main__":
    art = ROOT / "artifacts"
    with open(art / "vocab.json") as f_vocab:
        vocab_g = json.load(f_vocab)
    train_df_g = pd.read_parquet(art / "splits" / "train.parquet")
    # hour-freq is cheap — fit here for import-time test
    sys.exit(main() or 0)
