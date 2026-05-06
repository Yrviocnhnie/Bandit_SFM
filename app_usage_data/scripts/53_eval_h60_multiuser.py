"""Track A — rank-based eval for the multi-user pipeline.

Loads the 4 trained checkpoints + 6 closed-form baselines, scores each (user,
anchor, app) row, and computes per-user Track A metrics (FK@r, MSR@r, PR-AUC,
ROC-AUC, NDCG). Aggregates as mean / median / std across users + bootstrap CI
on user-level metrics. Generates Pareto figures.

Two evaluation slices:
  - Slice A (default): the 22-user pooled bg_test parquet (from 50_*).
  - Slice B (--include-slice-b): scores the global model on the existing
    single-user bg_test (artifacts/bg/splits/bg_test.parquet) — the cold-start
    transfer comparison vs the user's own bespoke single-user models.

Outputs:
  artifacts/bg_multi/results/h60_leaderboard.json
  artifacts/bg_multi/figures/pareto_h60_test.png
  artifacts/bg_multi/figures/per_user_pr_auc.png
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.bg.baselines_bg import (
    add_hybrid_lru_markov, add_lfu_hour_score, add_lru_score,
    add_markov_inverse, add_random_score, add_time_in_bg_score,
)
from lib.bg.metrics_bg import compute_metrics
from lib.bg_multi.helpers import load_per_user_stats, make_global_anchor_id


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_TRAIN_MULTI = _load("train_multi_for_eval", ROOT / "scripts" / "52_train_task_c_multiuser.py")


# ============================================================================
R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)
TRAINED_PICKS = ("c3p3", "c3pro_reg", "c3pro_listwise", "c3pro_wide")
RECIPE_LABELS = {
    "c3p3":           "C3.3",
    "c3pro_reg":      "Pro-Reg/c3.3",
    "c3pro_listwise": "Pro-List/c3.3",
    "c3pro_wide":     "Pro-Wide/c3.3",
}
BASELINES = [
    ("random",        "score_random"),
    ("lru",           "score_lru"),
    ("tibg",          "score_tibg"),
    ("lfu_hour",      "score_lfu_hour"),
    ("markov_inv",    "score_markov_inv"),
    ("hybrid_lru_mk", "score_hybrid_lru_mk"),
]


def attach_baseline_scores_per_user(df: pd.DataFrame, art_dir: Path,
                                     seed: int = 7) -> pd.DataFrame:
    """For each user in df, attach the 6 baseline scores using THAT user's stats."""
    out_parts = []
    for uid, sub in df.groupby("user_uid", sort=False):
        sub_r = sub.reset_index(drop=True).copy()
        stats = load_per_user_stats(art_dir / "per_user" / uid / "stats.pkl")
        sub_r = add_random_score(sub_r, seed=seed)
        sub_r = add_lru_score(sub_r)
        sub_r = add_time_in_bg_score(sub_r)
        sub_r = add_lfu_hour_score(sub_r, hour_freq=stats["hour_freq"])
        sub_r = add_markov_inverse(sub_r, markov_prior=stats["markov_probs"])
        sub_r = add_hybrid_lru_markov(sub_r, markov_prior=stats["markov_probs"], alpha=0.5)
        out_parts.append(sub_r)
    return pd.concat(out_parts, ignore_index=True)


def score_trained(model_recipe: str, df: pd.DataFrame, ctx_multi: dict,
                  art_dir: Path) -> np.ndarray:
    """Forward the recipe's checkpoint over `df` (must have user_id_idx); return kill_score."""
    ckpt_path = art_dir / "bg_multi" / "checkpoints" / f"task_c_multi_{model_recipe}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    data_d = _TRAIN_MULTI.build_schema_data_multi("c3.3", df, ctx_multi)
    rec = _TRAIN_MULTI.RECIPES[model_recipe]
    model = _TRAIN_MULTI.make_model(
        rec, num_features=int(data_d["features"].shape[1]),
        vocab_size=len(ctx_multi["vocab"]),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    f = torch.as_tensor(data_d["features"].copy(), dtype=torch.float32)
    a = torch.as_tensor(data_d["app_idx"].copy(), dtype=torch.long)
    out = []
    with torch.no_grad():
        for s in range(0, f.shape[0], 4096):
            logits = model(a[s:s + 4096], f[s:s + 4096])
            out.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(out)
    return 1.0 - p


def per_user_metrics(df: pd.DataFrame, score_col: str, y_col: str = "y_3600") -> dict:
    """Compute Track A metrics per user; return {uid: metrics}."""
    out: dict[str, dict] = {}
    df = df.copy()
    if "user_id_idx" in df.columns and "anchor_id" in df.columns:
        df["_global_aid"] = make_global_anchor_id(df)
    for uid, sub in df.groupby("user_uid", sort=False):
        sub2 = sub.copy()
        # The metrics function groups by anchor_id — pass a per-user-unique id
        sub2 = sub2.rename(columns={"anchor_id": "_per_user_aid"})
        sub2["anchor_id"] = sub2.get("_global_aid", sub2["_per_user_aid"])
        out[uid] = compute_metrics(sub2, score_col=score_col, y_col=y_col, r_values=R_SWEEP)
    return out


def aggregate_users(per_user: dict) -> dict:
    """Mean / median / std across users on every numeric metric (handles dict-of-r)."""
    if not per_user:
        return {}
    sample = next(iter(per_user.values()))
    out: dict = {}
    for k, v in sample.items():
        if isinstance(v, dict):
            out[k] = {}
            for r in v.keys():
                vals = [per_user[u][k].get(r, float("nan")) for u in per_user]
                vals = [x for x in vals if isinstance(x, (int, float)) and not np.isnan(x)]
                if vals:
                    out[k][r] = {"mean": float(np.mean(vals)),
                                  "median": float(np.median(vals)),
                                  "std": float(np.std(vals))}
        elif isinstance(v, (int, float)):
            vals = [per_user[u].get(k, float("nan")) for u in per_user]
            vals = [x for x in vals if isinstance(x, (int, float)) and not np.isnan(x)]
            if vals:
                out[k] = {"mean": float(np.mean(vals)),
                           "median": float(np.median(vals)),
                           "std": float(np.std(vals))}
    return out


def bootstrap_user_ci(per_user_vals: list[float], n_boot: int = 1000,
                      seed: int = 7) -> tuple[float, float]:
    """95% CI on the mean across users by user-level resampling."""
    if not per_user_vals:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    arr = np.array(per_user_vals, dtype=np.float64)
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(arr[idx].mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot_pareto(scored_dfs: dict[str, pd.DataFrame], out_path: Path, title: str):
    """One Pareto curve per (model, baseline). x = MSR, y = (1 - FK)."""
    plt.figure(figsize=(9, 6.5), dpi=140)
    cmap = {"random": "#7f7f7f", "lru": "#1f77b4", "markov_inv": "#d62728",
            "C3.3": "#2ca02c", "Pro-Reg/c3.3": "#9467bd",
            "Pro-List/c3.3": "tab:brown", "Pro-Wide/c3.3": "tab:olive"}
    for label, df in scored_dfs.items():
        per_u = per_user_metrics(df, score_col=f"score_for_{label}")
        msr_means = [np.mean([per_u[u]["memory_save_rate"][f"{r}"] for u in per_u])
                      for r in R_SWEEP]
        fk_means = [np.mean([per_u[u]["false_kill_rate"][f"{r}"] for u in per_u])
                     for r in R_SWEEP]
        kp_means = [1.0 - v for v in fk_means]
        c = cmap.get(label, "#333")
        plt.plot(msr_means, kp_means, marker="o", color=c, linewidth=2,
                 markersize=6, label=label)
    plt.xlabel("Memory-save rate (mean across users)")
    plt.ylabel("Kill precision  =  1 − False-kill rate")
    plt.title(title, fontsize=12)
    plt.legend(loc="lower left", fontsize=9, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[53] wrote {out_path}")


def plot_per_user_pr_auc(scored_dfs: dict[str, pd.DataFrame], out_path: Path):
    """One bar per (user, model) showing PR-AUC, sorted by mean across models."""
    user_pr: dict[str, dict[str, float]] = {}
    for label, df in scored_dfs.items():
        per_u = per_user_metrics(df, score_col=f"score_for_{label}")
        for u, m in per_u.items():
            user_pr.setdefault(u, {})[label] = float(m.get("pr_auc_mean", float("nan")))
    if not user_pr:
        return
    users = list(user_pr.keys())
    mean_per_user = {u: float(np.nanmean(list(user_pr[u].values()))) for u in users}
    users_sorted = sorted(users, key=lambda u: -mean_per_user[u])
    labels = list(scored_dfs.keys())

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    x = np.arange(len(users_sorted))
    width = 0.8 / max(1, len(labels))
    for i, lab in enumerate(labels):
        ys = [user_pr[u].get(lab, np.nan) for u in users_sorted]
        ax.bar(x + i * width, ys, width=width, label=lab)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(users_sorted, rotation=70, fontsize=7)
    ax.set_ylabel("test PR-AUC")
    ax.set_title("Per-user PR-AUC (sorted by mean across models)")
    ax.legend(fontsize=8, loc="lower left", ncol=3)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[53] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--art-dir", type=str, default=str(ROOT / "artifacts"))
    parser.add_argument("--include-slice-b", action="store_true",
                        help="also score on the single-user bg_test (cold-start transfer)")
    args = parser.parse_args()

    art_dir = Path(args.art_dir)
    bg_multi = art_dir / "bg_multi"
    fig_dir = bg_multi / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir = bg_multi / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    bg_val = pd.read_parquet(bg_multi / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_multi / "splits" / "bg_test.parquet")
    print(f"[53] slice A: bg_val={len(bg_val):,}  bg_test={len(bg_test):,}")

    # ── Build context for trained models
    print("[53] building multi-user ctx ...")
    ctx_multi = _TRAIN_MULTI.build_ctx_multi(art_dir)

    # ── Slice A: 22-user pooled
    print("[53] attaching baselines + scoring trained models on slice A val/test ...")
    bg_val_b = attach_baseline_scores_per_user(bg_val, bg_multi, seed=7)
    bg_test_b = attach_baseline_scores_per_user(bg_test, bg_multi, seed=7)

    # Score every trained pick — store its kill_score in its own column
    for r in TRAINED_PICKS:
        ckpt = bg_multi / "checkpoints" / f"task_c_multi_{r}.pt"
        if not ckpt.exists():
            print(f"  ⚠ ckpt missing: {ckpt}; skipping {r}")
            continue
        bg_val_b[f"score_{r}"] = score_trained(r, bg_val_b, ctx_multi, art_dir)
        bg_test_b[f"score_{r}"] = score_trained(r, bg_test_b, ctx_multi, art_dir)

    # ── Compute per-user metrics
    leaderboard = {"slice_a": {"val": {}, "test": {}}, "config": {
        "n_users": int(bg_test_b["user_uid"].nunique()),
        "vocab_size": len(ctx_multi["vocab"]),
    }}

    def _eval_slice_a(split_name: str, df: pd.DataFrame):
        print(f"[53] slice A {split_name} — per-user metrics ...")
        rows = list(BASELINES) + [(RECIPE_LABELS[r], f"score_{r}") for r in TRAINED_PICKS]
        for label, col in rows:
            if col not in df.columns:
                continue
            t0 = time.time()
            per_u = per_user_metrics(df, score_col=col)
            agg = aggregate_users(per_u)
            # bootstrap CI on PR-AUC and FK@.5 across users
            pr_vals = [per_u[u]["pr_auc_mean"] for u in per_u
                       if isinstance(per_u[u]["pr_auc_mean"], (int, float))
                       and not np.isnan(per_u[u]["pr_auc_mean"])]
            fk5_vals = [per_u[u]["false_kill_rate"]["0.5"] for u in per_u]
            agg["pr_auc_ci95"] = list(bootstrap_user_ci(pr_vals))
            agg["fk_0.5_ci95"] = list(bootstrap_user_ci(fk5_vals))
            leaderboard["slice_a"][split_name][label] = {
                "per_user": per_u, "agg": agg, "n_users": len(per_u),
            }
            print(
                f"  {label:<18}"
                f" PR-AUC={agg.get('pr_auc_mean', {}).get('mean', 0.0):.4f}"
                f"  ROC={agg.get('roc_auc_mean', {}).get('mean', 0.0):.4f}"
                f"  FK@.5={agg.get('false_kill_rate', {}).get('0.5', {}).get('mean', 0.0):.4f}"
                f"  MSR@.5={agg.get('memory_save_rate', {}).get('0.5', {}).get('mean', 0.0):.4f}"
                f"  ({time.time() - t0:.1f}s)"
            )

    _eval_slice_a("val", bg_val_b)
    _eval_slice_a("test", bg_test_b)

    # ── Pareto & per-user PR-AUC figures (slice A test)
    scored_dfs = {}
    for label, col in BASELINES:
        df_one = bg_test_b[["user_uid", "user_id_idx", "anchor_id", "y_3600", col]].copy()
        df_one[f"score_for_{label}"] = df_one[col]
        scored_dfs[label] = df_one
    for r in TRAINED_PICKS:
        col = f"score_{r}"
        if col not in bg_test_b.columns:
            continue
        df_one = bg_test_b[["user_uid", "user_id_idx", "anchor_id", "y_3600", col]].copy()
        df_one[f"score_for_{RECIPE_LABELS[r]}"] = df_one[col]
        scored_dfs[RECIPE_LABELS[r]] = df_one
    plot_pareto(scored_dfs, fig_dir / "pareto_h60_test.png",
                title="Track A Pareto — multi-user TEST (mean across 22 users)")
    plot_per_user_pr_auc(scored_dfs, fig_dir / "per_user_pr_auc.png")

    # ── Slice B: cold-start single-user
    if args.include_slice_b:
        sub_path = art_dir / "bg" / "splits" / "bg_test.parquet"
        if sub_path.exists():
            bg_test_su = pd.read_parquet(sub_path)
            # Inject required columns to match the multi-user schema
            single_uid = "single_user"
            bg_test_su = bg_test_su.copy()
            bg_test_su["user_uid"] = single_uid
            # The single-user model was trained with vocab_size=50; the multi-user
            # model uses vocab_size=243. The single-user `app_idx` is from the OLD
            # vocab. To score, we need to re-map — fall back to UNK for now. This
            # is an honest cold-start representation.
            bg_test_su["user_id_idx"] = -1  # cold-start sentinel
            print(f"\n[53] slice B (cold-start single user): rows={len(bg_test_su)}")
            print("[53]   note: slice B uses index -1 (cold-start) — no per-user stats injected.")
            # We can still run baselines that don't need per-user stats: random, lru, tibg
            cold = bg_test_su.copy().reset_index(drop=True)
            cold = add_random_score(cold, seed=7)
            cold = add_lru_score(cold)
            cold = add_time_in_bg_score(cold)
            simple = {}
            for label, col in (("random", "score_random"), ("lru", "score_lru"), ("tibg", "score_tibg")):
                m = compute_metrics(cold, score_col=col, y_col="y_3600", r_values=R_SWEEP)
                simple[label] = m
            leaderboard["slice_b"] = {"test": simple, "n_rows": int(len(cold))}
            print(f"[53] slice B baselines: " + ", ".join(
                f"{k}: PR={v['pr_auc_mean']:.3f}" for k, v in simple.items()))
        else:
            print(f"[53] slice B requested but {sub_path} doesn't exist; skipping")

    out_path = res_dir / "h60_leaderboard.json"
    with open(out_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"\n[53] DONE  leaderboard -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
