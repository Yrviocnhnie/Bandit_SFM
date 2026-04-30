"""H=60 evaluation: bootstrap CIs + Pareto figures.

Reads:
  artifacts/bg/splits/bg_{val,test}.parquet
  artifacts/bg/checkpoints/task_c_C{1,2,31}_h60.pt

Writes:
  artifacts/bg/results/h60_ci_table.json
  figures/bg/pareto_h60_{val,test}.png

For each closed-form baseline + trained model, computes per-anchor metrics
(PR-AUC, ROC-AUC, NDCG, FK@r, MSR@r), then bootstrap-resamples anchors B=1000
times to get 95% CIs on the mean of each metric.
"""
from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.bg import metrics_bg as MET
from lib.bg.baselines_bg import (
    add_lru_score, add_random_score, add_time_in_bg_score,
    add_lfu_hour_score, add_markov_inverse, add_hybrid_lru_markov,
)
from lib.v3 import markov_prior as MK
from lib.v3 import categories as CAT

SEED = 7
N_BOOT = 1000
R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)
H_SEC = 3600
Y_COL = f"y_{H_SEC}"


def fit_hour_freq(train_events, vocab):
    V = len(vocab)
    rare = vocab.get("<RARE>", 2)
    table = np.zeros((24, V), dtype=np.float64)
    mask = train_events["is_target_event"].astype(bool).to_numpy()
    hrs = pd.to_datetime(train_events.loc[mask, "event_ts"]).dt.hour.to_numpy()
    apps = train_events.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    for h, a in zip(hrs, idx):
        if 0 <= h < 24 and 3 <= a < V:
            table[int(h), int(a)] += 1.0
    table += 0.5
    table[:, :3] = 0.0
    s = table.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (table / s).astype(np.float32)


def topk_mask(scores, r):
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, min(int(np.ceil(r * n)), n))
    order = np.argsort(-scores, kind="mergesort")
    m = np.zeros(n, dtype=bool)
    m[order[:k]] = True
    return m


def per_anchor_arrays(df, score_col):
    pr, roc, ndcg = [], [], []
    fk = {r: [] for r in R_SWEEP}
    msr = {r: [] for r in R_SWEEP}
    for _, g in df.groupby("anchor_id", sort=False):
        scores = g[score_col].to_numpy(dtype=np.float64)
        y = g[Y_COL].to_numpy(dtype=np.int64)
        n = len(y)
        if n == 0:
            continue
        n_pos = int(y.sum())
        if 0 < n_pos < n:
            kill_label = (1 - y).astype(np.int64)
            pr.append(float(MET.pr_auc(kill_label, scores)))
            roc.append(float(MET.roc_auc(kill_label, scores)))
        else:
            pr.append(np.nan)
            roc.append(np.nan)
        ndcg.append(float(MET.ndcg_at_k((1 - y).astype(np.float64), scores,
                                         k=max(1, int(np.ceil(0.5 * n))))))
        for r in R_SWEEP:
            mask = topk_mask(scores, r)
            killed = int(mask.sum())
            fk_num = int((mask & (y == 1)).sum())
            save_num = int((mask & (y == 0)).sum())
            n_neg = int((y == 0).sum())
            fk[r].append(fk_num / killed if killed > 0 else 0.0)
            msr[r].append(save_num / n_neg if n_neg > 0 else 0.0)

    out = {
        "pr": np.asarray(pr, dtype=np.float64),
        "roc": np.asarray(roc, dtype=np.float64),
        "ndcg": np.asarray(ndcg, dtype=np.float64),
    }
    for r in R_SWEEP:
        out[f"fk_{r}"] = np.asarray(fk[r], dtype=np.float64)
        out[f"msr_{r}"] = np.asarray(msr[r], dtype=np.float64)
    return out


def bootstrap_ci(arr, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = arr[~np.isnan(arr)]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    return {
        "mean": float(np.mean(vals)),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(n),
    }


def load_trainer():
    if "trainer_h60" in sys.modules:
        return sys.modules["trainer_h60"]
    s = importlib.util.spec_from_file_location("trainer_h60", ROOT / "scripts" / "34_train_task_c_h60.py")
    m = importlib.util.module_from_spec(s)
    sys.modules["trainer_h60"] = m
    s.loader.exec_module(m)
    return m


import importlib.util


def score_trained(tag, data_dict):
    """Forward through saved checkpoint, return per-row kill score = 1 - sigmoid(logit)."""
    trainer = load_trainer()
    payload = torch.load(ROOT / "artifacts" / "bg" / "checkpoints" / f"task_c_{tag}.pt",
                         map_location="cpu", weights_only=False)
    use_cat = bool(payload.get("schema") == "c1") or bool(payload.get("use_cat_emb", False))
    nf = data_dict["features"].shape[1]
    vsize = int(payload["state_dict"]["app_emb.weight"].shape[0])
    model, _ = trainer.make_model(num_features=nf, vocab_size=vsize, use_cat_emb=use_cat)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    ds = trainer.PairDS(data_dict)
    loader = DataLoader(ds, batch_size=2048, shuffle=False)
    probs = []
    with torch.no_grad():
        for b in loader:
            logits = trainer.forward_batch(model, b)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(probs)
    return 1.0 - p


def main():
    art = ROOT / "artifacts"
    bg_art = art / "bg"
    fig_dir = ROOT / "figures" / "bg"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    train_events = pd.read_parquet(art / "splits" / "train.parquet")
    val_events = pd.read_parquet(art / "splits" / "val.parquet")
    test_events = pd.read_parquet(art / "splits" / "test.parquet")
    hour_freq = fit_hour_freq(train_events, vocab)
    markov_probs = MK.load_markov_prior(art / "v3" / "markov_prior.pkl")["probs"].astype(np.float32)
    app_to_cat = __import__("lib.v3.categories", fromlist=["build_app_to_cat_idx"]).build_app_to_cat_idx(vocab)

    bg_val = pd.read_parquet(bg_art / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_art / "splits" / "bg_test.parquet")

    # ───── score baselines onto each split (returns new df, must reassign) ─────
    def attach_baselines(df):
        df = add_random_score(df, seed=7)
        df = add_lru_score(df)
        df = add_time_in_bg_score(df)
        df = add_lfu_hour_score(df, hour_freq=hour_freq)
        df = add_markov_inverse(df, markov_prior=markov_probs)
        df = add_hybrid_lru_markov(df, markov_prior=markov_probs, alpha=0.5)
        return df

    bg_val = attach_baselines(bg_val)
    bg_test = attach_baselines(bg_test)

    # ── Trained model scoring ──
    trainer = load_trainer()
    per_app_inter = trainer.fit_per_app_inter_fg_mean(
        pd.read_parquet(art / "splits" / "train.parquet"), vocab,
    )
    all_events = pd.concat(
        [pd.read_parquet(art / "splits" / "train.parquet"),
         val_events_df := pd.read_parquet(art / "splits" / "val.parquet"),
         pd.read_parquet(art / "splits" / "test.parquet")],
        ignore_index=True,
    ).sort_values("event_ts").reset_index(drop=True)
    fg_timeline = trainer.collect_fg_timeline(all_events, vocab)

    splits = {"val": bg_val, "test": bg_test}
    schema_data = {"val": {}, "test": {}}
    for sp_name, df in splits.items():
        schema_data[sp_name]["c1"] = trainer.build_c1(df, hour_freq, markov_probs, app_to_cat)
        schema_data[sp_name]["c2"] = trainer.build_c2(df, hour_freq, markov_probs)
        schema_data[sp_name]["c3.1"] = trainer.build_c31(df, hour_freq, markov_probs,
                                                          fg_timeline, per_app_inter)

    # Score trained models onto df
    for sp_name, df in splits.items():
        for tag, schema in [("C1_h60", "c1"), ("C2_h60", "c2"), ("C31_h60", "c3.1")]:
            kill_score = score_trained(tag, schema_data[sp_name][schema])
            df[f"score_{tag.lower()}"] = kill_score

    # ───── per-anchor metrics for everyone, then bootstrap ─────
    BASELINE_NAMES = [
        ("random", "score_random"),
        ("lru", "score_lru"),
        ("tibg", "score_tibg"),
        ("lfu_hour", "score_lfu_hour"),
        ("markov_inv", "score_markov_inv"),
        ("hybrid_lru_mk", "score_hybrid_lru_mk"),
    ]
    TRAINED_NAMES = [(tag, f"score_{tag.lower()}") for tag in ("C1_h60", "C2_h60", "C31_h60")]
    ALL_NAMES = BASELINE_NAMES + TRAINED_NAMES

    leaderboard = {"r_sweep": list(R_SWEEP), "splits": {}}
    for sp_name, df in splits.items():
        sp = {}
        for name, col in ALL_NAMES:
            arrs = per_anchor_arrays(df, col)
            ent = {
                "n_anchors": int(df["anchor_id"].nunique()),
                "pr_auc": bootstrap_ci(arrs["pr"]),
                "roc_auc": bootstrap_ci(arrs["roc"]),
                "ndcg": bootstrap_ci(arrs["ndcg"]),
                "fk": {str(r): bootstrap_ci(arrs[f"fk_{r}"]) for r in R_SWEEP},
                "msr": {str(r): bootstrap_ci(arrs[f"msr_{r}"]) for r in R_SWEEP},
            }
            sp[name] = ent
        leaderboard["splits"][sp_name] = sp

    out_path = bg_art / "results" / "h60_leaderboard_ci.json"
    with open(out_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"[35] wrote {out_path}")

    # ───── Pareto curves ─────
    for sp_name, df in splits.items():
        plt.figure(figsize=(8, 5.5))
        for label, score_col in [("Random", "score_random"),
                                  ("LRU", "score_lru"),
                                  ("LFU-hour", "score_lfu_hour"),
                                  ("Markov-inv", "score_markov_inv"),
                                  ("C1", "score_c1_h60"),
                                  ("C2", "score_c2_h60"),
                                  ("C3.1", "score_c31_h60")]:
            arr = per_anchor_arrays(df, score_col)
            fk_means = [np.nanmean(arr[f"fk_{r}"]) for r in R_SWEEP]
            msr_means = [np.nanmean(arr[f"msr_{r}"]) for r in R_SWEEP]
            plt.plot(fk_means, msr_means, marker="o", label=label)
        plt.xlabel("False kill rate (lower better)")
        plt.ylabel("Memory save rate (higher better)")
        plt.title(f"H = 60 min Pareto — {sp_name.upper()}  (B(t) ≥ 1)")
        plt.legend(loc="lower left", fontsize=9)
        plt.grid(True, alpha=0.3)
        out = ROOT / "figures" / "bg" / f"pareto_h60_{sp_name}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"[35] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
