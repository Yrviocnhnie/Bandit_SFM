"""Threshold-based (Track B) metrics for Task C H=60.

For every closed-form baseline + 4 trained picks (C3.3, Pro-Reg/c3.3,
Pro-List/c3.3, Pro-Wide/c3.3):

  1. Find τ* = arg max F1 on val
  2. Evaluate KillPrecision / KillRecall / F1 / MCC / Accuracy at τ* on val and test
  3. Save to artifacts/bg/results/threshold_metrics.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.bg import metrics_bg as MET
from lib.bg.baselines_bg import (
    add_lru_score, add_random_score, add_time_in_bg_score,
    add_lfu_hour_score, add_markov_inverse, add_hybrid_lru_markov,
)
from lib.v3 import categories as CAT
from lib.v3 import markov_prior as MK


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


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


def forward_probs(model, data_d):
    model.eval()
    f = torch.as_tensor(data_d["features"].copy(), dtype=torch.float32)
    a = torch.as_tensor(data_d["app_idx"].copy(), dtype=torch.long)
    out = []
    with torch.no_grad():
        for s in range(0, f.shape[0], 2048):
            logits = model(a[s:s + 2048], f[s:s + 2048])
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def main():
    art = Path("artifacts")
    bg_art = art / "bg"
    out_dir = bg_art / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    train_events = pd.read_parquet(art / "splits" / "train.parquet")
    val_events = pd.read_parquet(art / "splits" / "val.parquet")
    test_events = pd.read_parquet(art / "splits" / "test.parquet")
    bg_train = pd.read_parquet(bg_art / "splits" / "bg_train.parquet")
    bg_val = pd.read_parquet(bg_art / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_art / "splits" / "bg_test.parquet")

    hour_freq = fit_hour_freq(train_events, vocab)
    markov_probs = MK.load_markov_prior(art / "v3" / "markov_prior.pkl")["probs"].astype(np.float32)

    def attach(df):
        df = add_random_score(df, seed=7)
        df = add_lru_score(df)
        df = add_time_in_bg_score(df)
        df = add_lfu_hour_score(df, hour_freq=hour_freq)
        df = add_markov_inverse(df, markov_prior=markov_probs)
        df = add_hybrid_lru_markov(df, markov_prior=markov_probs, alpha=0.5)
        return df

    bg_val = attach(bg_val)
    bg_test = attach(bg_test)

    trainer = load_module("trainer_h60", ROOT / "scripts" / "34_train_task_c_h60.py")
    grid = load_module("trainer_grid", ROOT / "scripts" / "36_train_c3_grid.py")
    app_to_cat = CAT.build_app_to_cat_idx(vocab)
    per_app_inter = trainer.fit_per_app_inter_fg_mean(train_events, vocab)
    all_events = pd.concat([train_events, val_events, test_events], ignore_index=True) \
        .sort_values("event_ts").reset_index(drop=True)
    fg_timeline = trainer.collect_fg_timeline(all_events, vocab)
    grid_ctx = {
        "trainer": trainer, "grid": grid, "vocab": vocab,
        "hour_freq": hour_freq, "markov_probs": markov_probs,
        "app_to_cat": app_to_cat, "per_app_inter_mean": per_app_inter,
        "fg_timeline": fg_timeline,
        "app_lifetime_share":     grid.fit_app_lifetime_share(train_events, vocab),
        "app_lifetime_kill_rate": grid.fit_app_lifetime_kill_rate(bg_train, vocab),
        "cat_lifetime_share":     grid.fit_cat_lifetime_share(train_events, vocab, app_to_cat),
        "cat_markov_probs":       grid.fit_cat_markov(train_events, vocab, app_to_cat),
    }

    PICKS = [
        ("C3.3",          "task_c_c3p3.pt",          "score_c33",     False),
        ("Pro-Reg/c3.3",  "task_c_c3pro_reg.pt",     "score_proreg",  False),
        ("Pro-List/c3.3", "task_c_c3pro_listwise.pt", "score_prolist", False),
        ("Pro-Wide/c3.3", "task_c_c3pro_wide.pt",    "score_prowide", True),
    ]
    for sp_name, sp_df in (("val", bg_val), ("test", bg_test)):
        d = grid.build_schema_data("c3.3", sp_df, grid_ctx)
        for label, fn, col, is_wide in PICKS:
            ckpt = torch.load(art / "bg" / "checkpoints" / fn,
                              map_location="cpu", weights_only=False)
            nf = int(d["features"].shape[1])
            if is_wide:
                model = grid.make_wide_model(num_features=nf, vocab_size=len(vocab))
            else:
                model = grid.make_baseline_model(num_features=nf, vocab_size=len(vocab),
                                                  dropout=0.2)
            model.load_state_dict(ckpt["state_dict"], strict=False)
            sp_df[col] = 1.0 - forward_probs(model, d)

    REPORT_ROWS = [
        ("random",         "score_random"),
        ("lru",            "score_lru"),
        ("tibg",           "score_tibg"),
        ("lfu_hour",       "score_lfu_hour"),
        ("markov_inv",     "score_markov_inv"),
        ("hybrid_lru_mk",  "score_hybrid_lru_mk"),
        ("C3.3",           "score_c33"),
        ("Pro-Reg/c3.3",   "score_proreg"),
        ("Pro-List/c3.3",  "score_prolist"),
        ("Pro-Wide/c3.3",  "score_prowide"),
    ]

    FIXED_TAUS = [0.30, 0.40, 0.50, 0.60, 0.70]
    TRAINED_LABELS = {"random", "C3.3", "Pro-Reg/c3.3", "Pro-List/c3.3", "Pro-Wide/c3.3"}
    # 50-point dense sweep, 0.02 step, τ ∈ {0.02, 0.04, ..., 1.00}
    DENSE_TAUS = [round(0.02 * (i + 1), 2) for i in range(50)]

    out: dict = {}
    for label, col in REPORT_ROWS:
        tau_star = MET.find_best_tau_by_f1(bg_val, col, "y_3600")
        entry = {
            "tau_star": float(tau_star),
            "val":  MET.compute_threshold_metrics(bg_val,  col, "y_3600", tau_star),
            "test": MET.compute_threshold_metrics(bg_test, col, "y_3600", tau_star),
        }
        # Keep-class argmax F1 (positive class flipped to the rare 22 % "keep" class)
        tau_star_keep = MET.find_best_tau_by_f1_keep(bg_val, col, "y_3600")
        entry["tau_star_keep"] = float(tau_star_keep)
        entry["val_keep"]  = MET.compute_keep_threshold_metrics(bg_val,  col, "y_3600", tau_star_keep)
        entry["test_keep"] = MET.compute_keep_threshold_metrics(bg_test, col, "y_3600", tau_star_keep)
        # Also compute the *kill-side* metrics at the keep-tuned τ for direct comparison
        entry["test_kill_at_keep_tau"] = MET.compute_threshold_metrics(bg_test, col, "y_3600", tau_star_keep)
        if label in TRAINED_LABELS:
            entry["fixed"] = {}
            for tau in FIXED_TAUS:
                entry["fixed"][f"{tau:.2f}"] = {
                    "val":  MET.compute_threshold_metrics(bg_val,  col, "y_3600", tau),
                    "test": MET.compute_threshold_metrics(bg_test, col, "y_3600", tau),
                }
            # 50-point dense sweep on test (only the trained-comparable rows)
            entry["dense"] = {}
            for tau in DENSE_TAUS:
                entry["dense"][f"{tau:.2f}"] = MET.compute_threshold_metrics(
                    bg_test, col, "y_3600", tau)
        out[label] = entry

    out_path = art / "bg" / "results" / "threshold_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Pretty-print τ* (all models)
    for split in ("val", "test"):
        print(f"\n=== {split.upper()}   Track B — threshold-based metrics  (τ* = arg max F1 on val) ===")
        print(f"{'Model':<18}{'τ*':<7}{'KillPrec':<10}{'KillRec':<10}"
              f"{'F1':<8}{'Accuracy':<10}{'MCC':<8}")
        print("-" * 75)
        for label, _ in REPORT_ROWS:
            row = out[label]
            m = row[split]
            print(f"{label:<18}{row['tau_star']:<7.3f}"
                  f"{m['kill_precision']:<10.4f}{m['kill_recall']:<10.4f}"
                  f"{m['f1']:<8.4f}{m['accuracy']:<10.4f}{m['mcc']:<8.4f}")

    # Pretty-print keep-class argmax τ*
    print(f"\n=== TEST   Track B — argmax-F1 on KEEP class (rare 22 %) ===")
    print(f"{'Model':<18}{'τ_keep':<8}{'F1_keep':<10}{'F1_kill':<10}"
          f"{'MCC':<8}{'FKR':<8}{'SKR':<8}{'Acc':<8}")
    print("-" * 85)
    for label, _ in REPORT_ROWS:
        row = out[label]
        mk = row["test_keep"]
        mki = row["test_kill_at_keep_tau"]
        fkr = 1.0 - mki["kill_precision"]
        print(f"{label:<18}{row['tau_star_keep']:<8.3f}"
              f"{mk['keep_f1']:<10.4f}{mki['f1']:<10.4f}"
              f"{mki['mcc']:<8.4f}{fkr:<8.4f}{mki['kill_recall']:<8.4f}"
              f"{mki['accuracy']:<8.4f}")

    # Pretty-print fixed-τ sweep (trained models only)
    print(f"\n=== TEST   Track B — fixed-τ sweep (trained models only) ===")
    print(f"{'Model':<18}{'τ':<7}{'KillPrec':<10}{'KillRec':<10}"
          f"{'F1':<8}{'Accuracy':<10}{'MCC':<8}")
    print("-" * 75)
    for label in ("random", "C3.3", "Pro-Reg/c3.3", "Pro-List/c3.3", "Pro-Wide/c3.3"):
        for tau in FIXED_TAUS:
            m = out[label]["fixed"][f"{tau:.2f}"]["test"]
            print(f"{label:<18}{tau:<7.2f}"
                  f"{m['kill_precision']:<10.4f}{m['kill_recall']:<10.4f}"
                  f"{m['f1']:<8.4f}{m['accuracy']:<10.4f}{m['mcc']:<8.4f}")
        print()

    # Dense 50-point sweep — find the 5 τ with the highest mean MCC
    # across the 4 trained models (sanity-check across the full grid).
    trained = ("C3.3", "Pro-Reg/c3.3", "Pro-List/c3.3", "Pro-Wide/c3.3")
    mean_mcc = []
    for tau in DENSE_TAUS:
        vals = [out[m]["dense"][f"{tau:.2f}"]["mcc"] for m in trained]
        mean_mcc.append((tau, float(sum(vals) / len(vals))))
    mean_mcc_sorted = sorted(mean_mcc, key=lambda kv: -kv[1])
    top5_taus = sorted([t for t, _ in mean_mcc_sorted[:5]])

    print(f"\n=== TEST   Track B — 50-point dense sweep (τ step = 0.02) ===")
    print("Top 5 τ by mean MCC across the 4 trained models:")
    print(f"  {top5_taus}")
    print(f"\n{'Model':<18}{'τ':<7}{'FKR↓':<8}{'SKR↑':<8}"
          f"{'F1↑':<8}{'Acc↑':<8}{'MCC↑':<8}")
    print("-" * 65)
    for tau in top5_taus:
        for label in ("random",) + trained:
            m = out[label]["dense"][f"{tau:.2f}"]
            fkr = 1.0 - m["kill_precision"]
            print(f"{label:<18}{tau:<7.2f}"
                  f"{fkr:<8.4f}{m['kill_recall']:<8.4f}"
                  f"{m['f1']:<8.4f}{m['accuracy']:<8.4f}{m['mcc']:<8.4f}")
        print()

    print(f"[41] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

