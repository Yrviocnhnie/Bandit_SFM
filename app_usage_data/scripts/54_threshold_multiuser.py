"""Track B — threshold-based eval for the multi-user pipeline.

For each model (random + 4 trained picks):
  - Per-user τ\* via argmax F1(keep) on val; apply per-user τ\* to that user's test.
  - Global τ\* via argmax F1(keep) on pooled val; apply pooled τ\* to pooled test.
  - Both kill-class argmax-F1 (legacy) and keep-class argmax-F1 (recommended).
  - Dense τ-sweep at τ ∈ {0.02, 0.04, …, 1.00} on pooled val (top-5 by mean MCC).

Aggregates per-user metrics as mean / median / std across users.

Output: artifacts/bg_multi/results/threshold_metrics.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg.baselines_bg import add_random_score
from lib.bg.metrics_bg import (
    compute_keep_threshold_metrics, compute_threshold_metrics,
    find_best_tau_by_f1, find_best_tau_by_f1_keep,
)


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_TRAIN_MULTI = _load("train_multi_for_thr", ROOT / "scripts" / "52_train_task_c_multiuser.py")


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
DENSE_TAUS = [round(0.02 * (i + 1), 2) for i in range(50)]


def score_trained_into_df(recipe: str, df: pd.DataFrame, ctx_multi: dict,
                           art_dir: Path) -> pd.DataFrame:
    """Forward `recipe` checkpoint over df; add column `score_<recipe>`."""
    ckpt_path = art_dir / "bg_multi" / "checkpoints" / f"task_c_multi_{recipe}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    df = df.copy()
    df[f"score_{recipe}"] = 1.0 - np.concatenate(out)
    return df


def per_user_argmax_keep(df_val: pd.DataFrame, df_test: pd.DataFrame,
                          score_col: str, y_col: str = "y_3600") -> dict:
    """For each user, pick τ_keep on their val rows, evaluate on their test rows."""
    out: dict = {"per_user": {}}
    for uid, sub_val in df_val.groupby("user_uid", sort=False):
        sub_test = df_test[df_test["user_uid"] == uid]
        if len(sub_val) == 0 or len(sub_test) == 0:
            continue
        sub_val_r = sub_val.reset_index(drop=True)
        sub_test_r = sub_test.reset_index(drop=True)
        try:
            tau_keep = find_best_tau_by_f1_keep(sub_val_r, score_col, y_col)
            tau_kill = find_best_tau_by_f1(sub_val_r, score_col, y_col)
        except Exception as e:
            print(f"  ⚠ tau search failed for uid={uid}: {e}")
            continue
        m_keep = compute_keep_threshold_metrics(sub_test_r, score_col, y_col, tau_keep)
        m_kill_at_keep = compute_threshold_metrics(sub_test_r, score_col, y_col, tau_keep)
        m_kill_at_kill = compute_threshold_metrics(sub_test_r, score_col, y_col, tau_kill)
        out["per_user"][uid] = {
            "tau_keep": float(tau_keep), "tau_kill": float(tau_kill),
            "metrics_at_tau_keep": {
                "keep": m_keep,
                "kill_side": m_kill_at_keep,
            },
            "metrics_at_tau_kill": m_kill_at_kill,
        }
    out["n_users"] = len(out["per_user"])
    return out


def aggregate_per_user_thr(per_user_block: dict) -> dict:
    """Mean/median across users on the kill-side metrics at τ_keep + at τ_kill."""
    pu = per_user_block.get("per_user", {})
    if not pu:
        return per_user_block
    keys = ("kill_precision", "kill_recall", "f1", "accuracy", "mcc")
    agg = {"at_tau_keep": {}, "at_tau_kill": {}}
    tau_keep_vals = [m["tau_keep"] for m in pu.values()]
    tau_kill_vals = [m["tau_kill"] for m in pu.values()]
    agg["tau_keep_mean"] = float(np.mean(tau_keep_vals))
    agg["tau_keep_median"] = float(np.median(tau_keep_vals))
    agg["tau_kill_mean"] = float(np.mean(tau_kill_vals))
    agg["tau_kill_median"] = float(np.median(tau_kill_vals))
    for k in keys:
        vals_keep = [m["metrics_at_tau_keep"]["kill_side"].get(k, np.nan) for m in pu.values()]
        vals_kill = [m["metrics_at_tau_kill"].get(k, np.nan) for m in pu.values()]
        vals_keep = [x for x in vals_keep if not (isinstance(x, float) and np.isnan(x))]
        vals_kill = [x for x in vals_kill if not (isinstance(x, float) and np.isnan(x))]
        if vals_keep:
            agg["at_tau_keep"][k] = {"mean": float(np.mean(vals_keep)),
                                     "median": float(np.median(vals_keep)),
                                     "std": float(np.std(vals_keep))}
        if vals_kill:
            agg["at_tau_kill"][k] = {"mean": float(np.mean(vals_kill)),
                                     "median": float(np.median(vals_kill)),
                                     "std": float(np.std(vals_kill))}
    return {**per_user_block, "agg": agg}


def global_argmax(df_val: pd.DataFrame, df_test: pd.DataFrame,
                  score_col: str, y_col: str = "y_3600") -> dict:
    """Global τ\* on pooled val (kill + keep variants), apply to pooled test."""
    df_val_r = df_val.reset_index(drop=True)
    df_test_r = df_test.reset_index(drop=True)
    tau_kill = float(find_best_tau_by_f1(df_val_r, score_col, y_col))
    tau_keep = float(find_best_tau_by_f1_keep(df_val_r, score_col, y_col))
    return {
        "tau_kill": tau_kill,
        "tau_keep": tau_keep,
        "test_at_tau_kill": compute_threshold_metrics(df_test_r, score_col, y_col, tau_kill),
        "test_at_tau_keep_kill_side": compute_threshold_metrics(df_test_r, score_col, y_col, tau_keep),
        "test_at_tau_keep_keep_side": compute_keep_threshold_metrics(df_test_r, score_col, y_col, tau_keep),
    }


def dense_sweep_pooled(df_test: pd.DataFrame, score_col: str,
                        y_col: str = "y_3600") -> dict:
    """Dense τ ∈ {0.02, …, 1.0} sweep on pooled test."""
    df_r = df_test.reset_index(drop=True)
    out: dict = {}
    for tau in DENSE_TAUS:
        out[f"{tau:.2f}"] = compute_threshold_metrics(df_r, score_col, y_col, tau)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--art-dir", type=str, default=str(ROOT / "artifacts"))
    args = parser.parse_args()

    art_dir = Path(args.art_dir)
    bg_multi = art_dir / "bg_multi"
    res_dir = bg_multi / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    bg_val = pd.read_parquet(bg_multi / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_multi / "splits" / "bg_test.parquet")
    print(f"[54] bg_val={len(bg_val):,}  bg_test={len(bg_test):,}")

    # Random baseline: same RNG sequence on val and test (per pooled df)
    bg_val = add_random_score(bg_val, seed=7)
    bg_test = add_random_score(bg_test, seed=7)

    print("[54] building multi-user ctx ...")
    ctx_multi = _TRAIN_MULTI.build_ctx_multi(art_dir)

    # Score every trained pick
    for r in TRAINED_PICKS:
        ckpt = bg_multi / "checkpoints" / f"task_c_multi_{r}.pt"
        if not ckpt.exists():
            print(f"  ⚠ ckpt missing: {ckpt}; skipping {r}")
            continue
        bg_val = score_trained_into_df(r, bg_val, ctx_multi, art_dir)
        bg_test = score_trained_into_df(r, bg_test, ctx_multi, art_dir)

    out: dict = {}
    rows = [("random", "score_random")] + [(RECIPE_LABELS[r], f"score_{r}")
                                              for r in TRAINED_PICKS
                                              if (bg_multi / "checkpoints" / f"task_c_multi_{r}.pt").exists()]
    for label, col in rows:
        print(f"[54] {label} — per-user argmax-F1(keep) ...")
        t0 = time.time()
        block = per_user_argmax_keep(bg_val, bg_test, score_col=col)
        block = aggregate_per_user_thr(block)
        block["global_argmax"] = global_argmax(bg_val, bg_test, score_col=col)
        # Dense sweep only for trained models
        if label != "random":
            block["dense_test"] = dense_sweep_pooled(bg_test, score_col=col)
        out[label] = block
        agg = block["agg"]
        print(
            f"  {label:<18}"
            f" τ_keep={agg['tau_keep_mean']:.3f} (med={agg['tau_keep_median']:.3f})"
            f"  MCC@τ_keep={agg['at_tau_keep'].get('mcc', {}).get('mean', 0.0):.4f}"
            f"  F1={agg['at_tau_keep'].get('f1', {}).get('mean', 0.0):.4f}"
            f"  ({time.time() - t0:.1f}s)"
        )

    out_path = res_dir / "threshold_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[54] DONE  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
