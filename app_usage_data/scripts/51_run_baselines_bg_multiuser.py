"""Per-user closed-form baselines on the multi-user pooled bg parquets.

For each user, runs Random / LRU / TimeInBG / LFU-hour / Markov-inverse /
Hybrid LRU+Markov on that user's val + test rows, then aggregates as
mean / median / std / per-user list.

Per-user stats (Markov / hour_freq) come from artifacts/bg_multi/per_user/<uid>/stats.pkl.
LRU and TimeInBG are structural — no fit.

Output: artifacts/bg_multi/results/baselines_bg.json with the standard structure.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg.baselines_bg import (
    add_hybrid_lru_markov, add_lfu_hour_score, add_lru_score,
    add_markov_inverse, add_random_score, add_time_in_bg_score,
)
from lib.bg.metrics_bg import compute_metrics
from lib.bg_multi.helpers import load_per_user_stats


# ============================================================================
R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)

BASELINES = [
    ("random",        "score_random"),
    ("lru",           "score_lru"),
    ("tibg",          "score_tibg"),
    ("lfu_hour",      "score_lfu_hour"),
    ("markov_inv",    "score_markov_inv"),
    ("hybrid_lru_mk", "score_hybrid_lru_mk"),
]


def attach_baseline_scores(
    df_user: pd.DataFrame,
    stats_user: dict,
    seed: int = 7,
) -> pd.DataFrame:
    """Attach all 6 baseline score columns using THIS USER's per-user stats."""
    if len(df_user) == 0:
        return df_user
    # Reset index — `add_hybrid_lru_markov` uses df.index values to slot output
    # back into a positional array, which breaks when the input has a non-zero-
    # based index (after slicing the pooled bg df).
    out = df_user.reset_index(drop=True).copy()
    out = add_random_score(out, seed=seed)
    out = add_lru_score(out)
    out = add_time_in_bg_score(out)
    out = add_lfu_hour_score(out, hour_freq=stats_user["hour_freq"])
    out = add_markov_inverse(out, markov_prior=stats_user["markov_probs"])
    out = add_hybrid_lru_markov(
        out, markov_prior=stats_user["markov_probs"], alpha=0.5,
    )
    return out


def per_user_metrics(
    bg_split: pd.DataFrame,
    art_dir: Path,
    seed: int = 7,
) -> dict:
    """For each user in bg_split, attach baseline scores + compute Track A metrics.

    Returns: {baseline: {"per_user": {uid: metrics_dict}, "n_users": int}}.
    """
    if "user_uid" not in bg_split.columns:
        raise KeyError("bg_split must have a 'user_uid' column")

    out: dict = {b: {"per_user": {}} for b, _ in BASELINES}
    user_ids = sorted(bg_split["user_uid"].unique().tolist())
    for uid in user_ids:
        sub = bg_split[bg_split["user_uid"] == uid].copy()
        if len(sub) == 0:
            continue
        stats_path = art_dir / "per_user" / uid / "stats.pkl"
        if not stats_path.exists():
            print(f"  ⚠ stats.pkl missing for uid={uid}; skipping", file=sys.stderr)
            continue
        stats_user = load_per_user_stats(stats_path)
        scored = attach_baseline_scores(sub, stats_user, seed=seed)
        for label, score_col in BASELINES:
            try:
                m = compute_metrics(scored, score_col=score_col, y_col="y_3600",
                                    r_values=R_SWEEP)
            except Exception as e:
                print(f"  ⚠ {label} failed for uid={uid}: {e}", file=sys.stderr)
                continue
            out[label]["per_user"][uid] = m
    for b in out:
        out[b]["n_users"] = len(out[b]["per_user"])
    return out


_FLAT_SEP = "\x1f"  # unit separator — won't collide with metric-name characters


def aggregate(per_user_block: dict) -> dict:
    """For one baseline's per-user block, compute mean/median/std summaries."""
    pu = per_user_block["per_user"]
    if not pu:
        return {**per_user_block, "mean": {}, "median": {}, "std": {}}
    # Collect every numeric metric across users; nested dicts of r-values flattened
    flat: dict[str, list[float]] = {}

    def _walk(uid_metrics, prefix=""):
        for k, v in uid_metrics.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}{_FLAT_SEP}{k}"
            if isinstance(v, dict):
                _walk(v, prefix=key)
            elif isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                flat.setdefault(key, []).append(float(v))

    for uid, m in pu.items():
        _walk(m, prefix="")

    def _build(reducer):
        out = {}
        for k, vals in flat.items():
            if not vals:
                continue
            parts = k.split(_FLAT_SEP)
            cur = out
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = float(reducer(vals))
        return out

    return {
        **per_user_block,
        "mean":   _build(np.mean),
        "median": _build(np.median),
        "std":    _build(np.std),
    }


def main():
    bg_multi = ROOT / "artifacts" / "bg_multi"
    bg_val = pd.read_parquet(bg_multi / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_multi / "splits" / "bg_test.parquet")

    print(f"[51] bg_val rows={len(bg_val):,}  bg_test rows={len(bg_test):,}")

    out = {"splits": {}}
    for split_name, df in (("val", bg_val), ("test", bg_test)):
        print(f"[51] {split_name} — running baselines per user ...")
        t0 = time.time()
        block = per_user_metrics(df, art_dir=bg_multi, seed=7)
        for b in block:
            block[b] = aggregate(block[b])
        out["splits"][split_name] = block
        print(f"[51]   done in {time.time() - t0:.1f}s")

    res_dir = bg_multi / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = res_dir / "baselines_bg.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Pretty print
    print(f"\n=== Test — per-user mean Track A metrics ===")
    print(f"{'baseline':<16}{'PR-AUC':>9}{'ROC-AUC':>9}{'FK@.25':>9}{'FK@.5':>9}"
          f"{'MSR@.25':>9}{'MSR@.5':>9}{'NDCG':>9}{'n_users':>9}")
    print("-" * 90)
    for label, _ in BASELINES:
        m = out["splits"]["test"][label].get("mean", {})
        n = out["splits"]["test"][label].get("n_users", 0)
        print(
            f"{label:<16}"
            f"{m.get('pr_auc_mean', 0.0):>9.4f}"
            f"{m.get('roc_auc_mean', 0.0):>9.4f}"
            f"{m.get('false_kill_rate', {}).get('0.25', 0.0):>9.4f}"
            f"{m.get('false_kill_rate', {}).get('0.5',  0.0):>9.4f}"
            f"{m.get('memory_save_rate', {}).get('0.25', 0.0):>9.4f}"
            f"{m.get('memory_save_rate', {}).get('0.5',  0.0):>9.4f}"
            f"{m.get('ndcg_half', 0.0):>9.4f}"
            f"{n:>9d}"
        )
    print(f"\n[51] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
