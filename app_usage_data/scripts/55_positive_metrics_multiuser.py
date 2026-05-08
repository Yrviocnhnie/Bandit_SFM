"""Positives-only metrics (WAKR / PosRank / PosScoreNorm) for the multi-user pipeline.

Aggregates per-user `compute_positive_only_metrics` across the 22 users.

Output: artifacts/bg_multi/results/positive_metrics.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg.baselines_bg import (
    add_hybrid_lru_markov, add_lfu_hour_score, add_lru_score,
    add_markov_inverse, add_random_score, add_time_in_bg_score,
)
from lib.bg.metrics_bg import compute_positive_only_metrics
from lib.bg_multi.helpers import load_per_user_stats, make_global_anchor_id


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_TRAIN_MULTI = _load("train_multi_for_pos", ROOT / "scripts" / "52_train_task_c_multiuser.py")


R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)
TRAINED_PICKS = ("c3p3", "c3pro_reg", "c3pro_listwise", "c3pro_wide",
                  "c3p4_cheap", "c3p4_full", "c3p3_cat")
RECIPE_LABELS = {
    "c3p3":           "C3.3",
    "c3pro_reg":      "Pro-Reg/c3.3",
    "c3pro_listwise": "Pro-List/c3.3",
    "c3pro_wide":     "Pro-Wide/c3.3",
    "c3p4_cheap":     "C3.4n-cheap",
    "c3p4_full":      "C3.4n-full",
    "c3p3_cat":       "C3.3+catEmb",
}
BASELINES = [
    ("random",        "score_random"),
    ("lru",           "score_lru"),
    ("tibg",          "score_tibg"),
    ("lfu_hour",      "score_lfu_hour"),
    ("markov_inv",    "score_markov_inv"),
    ("hybrid_lru_mk", "score_hybrid_lru_mk"),
]


def attach_baselines_per_user(df: pd.DataFrame, art_dir: Path) -> pd.DataFrame:
    parts = []
    for uid, sub in df.groupby("user_uid", sort=False):
        sub_r = sub.reset_index(drop=True).copy()
        stats = load_per_user_stats(art_dir / "per_user" / uid / "stats.pkl")
        sub_r = add_random_score(sub_r, seed=7)
        sub_r = add_lru_score(sub_r)
        sub_r = add_time_in_bg_score(sub_r)
        sub_r = add_lfu_hour_score(sub_r, hour_freq=stats["hour_freq"])
        sub_r = add_markov_inverse(sub_r, markov_prior=stats["markov_probs"])
        sub_r = add_hybrid_lru_markov(sub_r, markov_prior=stats["markov_probs"], alpha=0.5)
        parts.append(sub_r)
    return pd.concat(parts, ignore_index=True)


def score_trained(recipe: str, df: pd.DataFrame, ctx_multi: dict,
                  art_dir: Path) -> np.ndarray:
    ckpt = torch.load(
        art_dir / "bg_multi" / "checkpoints" / f"task_c_multi_{recipe}.pt",
        map_location="cpu", weights_only=False,
    )
    rec = _TRAIN_MULTI.RECIPES[recipe]
    use_cat = bool(rec.get("use_cat_emb"))
    data_d = _TRAIN_MULTI._build_data_dispatch(rec["schema"], df, ctx_multi)
    model = _TRAIN_MULTI.make_model(rec, num_features=int(data_d["features"].shape[1]),
                                     vocab_size=len(ctx_multi["vocab"]))
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    f = torch.as_tensor(data_d["features"].copy(), dtype=torch.float32)
    a = torch.as_tensor(data_d["app_idx"].copy(), dtype=torch.long)
    c = torch.as_tensor(data_d["cat_idx"].copy(), dtype=torch.long) if use_cat else None
    out = []
    with torch.no_grad():
        for s in range(0, f.shape[0], 4096):
            if use_cat:
                logits = model(a[s:s + 4096], f[s:s + 4096], c[s:s + 4096])
            else:
                logits = model(a[s:s + 4096], f[s:s + 4096])
            out.append(torch.sigmoid(logits).cpu().numpy())
    return 1.0 - np.concatenate(out)


def per_user_pos_metrics(df: pd.DataFrame, score_col: str,
                          y_col: str = "y_3600") -> dict:
    """Per-user positives-only metrics."""
    out: dict = {}
    df = df.copy()
    if "user_id_idx" in df.columns and "anchor_id" in df.columns:
        df["_global_aid"] = make_global_anchor_id(df)
    for uid, sub in df.groupby("user_uid", sort=False):
        sub2 = sub.copy()
        sub2 = sub2.rename(columns={"anchor_id": "_per_user_aid"})
        sub2["anchor_id"] = sub2.get("_global_aid", sub2["_per_user_aid"])
        out[uid] = compute_positive_only_metrics(sub2, score_col=score_col, y_col=y_col,
                                                  r_values=R_SWEEP)
    return out


def aggregate(per_user: dict) -> dict:
    if not per_user:
        return {}
    sample = next(iter(per_user.values()))
    out: dict = {}
    for k, v in sample.items():
        if isinstance(v, dict):
            out[k] = {}
            for r in v.keys():
                vals = [per_user[u][k].get(r, np.nan) for u in per_user]
                vals = [x for x in vals if isinstance(x, (int, float)) and not np.isnan(x)]
                if vals:
                    out[k][r] = {"mean": float(np.mean(vals)),
                                  "median": float(np.median(vals)),
                                  "std": float(np.std(vals))}
        elif isinstance(v, (int, float)):
            vals = [per_user[u].get(k, np.nan) for u in per_user]
            vals = [x for x in vals if isinstance(x, (int, float)) and not np.isnan(x)]
            if vals:
                out[k] = {"mean": float(np.mean(vals)),
                           "median": float(np.median(vals)),
                           "std": float(np.std(vals))}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--art-dir", type=str, default=str(ROOT / "artifacts"))
    args = parser.parse_args()

    art_dir = Path(args.art_dir)
    bg_multi = art_dir / "bg_multi"

    bg_val = pd.read_parquet(bg_multi / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_multi / "splits" / "bg_test.parquet")
    print(f"[55] bg_val={len(bg_val):,}  bg_test={len(bg_test):,}")

    bg_val = attach_baselines_per_user(bg_val, bg_multi)
    bg_test = attach_baselines_per_user(bg_test, bg_multi)

    ctx_multi = _TRAIN_MULTI.build_ctx_multi(art_dir)
    for r in TRAINED_PICKS:
        ckpt = bg_multi / "checkpoints" / f"task_c_multi_{r}.pt"
        if not ckpt.exists():
            continue
        bg_val[f"score_{r}"] = score_trained(r, bg_val, ctx_multi, art_dir)
        bg_test[f"score_{r}"] = score_trained(r, bg_test, ctx_multi, art_dir)

    out = {"val": {}, "test": {}}
    rows = list(BASELINES) + [(RECIPE_LABELS[r], f"score_{r}")
                                 for r in TRAINED_PICKS
                                 if f"score_{r}" in bg_test.columns]
    for split_name, df in (("val", bg_val), ("test", bg_test)):
        for label, col in rows:
            if col not in df.columns:
                continue
            per_u = per_user_pos_metrics(df, score_col=col)
            agg = aggregate(per_u)
            out[split_name][label] = {"per_user": per_u, "agg": agg, "n_users": len(per_u)}

    res_dir = bg_multi / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = res_dir / "positive_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Pretty print test
    print(f"\n=== TEST — positives-only mean across users ===")
    print(f"{'baseline':<18}{'PosRank↑':>10}{'PosScoreNorm↓':>14}{'WAKR@.25↓':>11}{'WAKR@.5↓':>11}")
    print("-" * 70)
    for label, _ in rows:
        if label not in out["test"]:
            continue
        agg = out["test"][label]["agg"]
        pr = agg.get("pos_rank_mean", {}).get("mean", float("nan"))
        psn = agg.get("pos_score_norm_mean", {}).get("mean", float("nan"))
        w25 = agg.get("wakr", {}).get("0.25", {}).get("mean", float("nan"))
        w50 = agg.get("wakr", {}).get("0.5",  {}).get("mean", float("nan"))
        print(f"{label:<18}{pr:>10.4f}{psn:>14.4f}{w25:>11.4f}{w50:>11.4f}")
    print(f"\n[55] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
