"""Compute positive-only metrics (WAKR@r, PosRank, PosScoreNorm) for Task C.

For every closed-form baseline + the 4 trained picks (C3.3, Pro-Reg/c3.3,
Pro-List/c3.3, Pro-Wide/c3.3), score each (anchor, app) row of the val and
test bg parquets and run `compute_positive_only_metrics`.

Outputs:
    artifacts/bg/results/positive_metrics.json
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


R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)


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
    """Re-score (anchor, app) rows with a trained model."""
    import torch as _t
    model.eval()
    f = _t.as_tensor(data_d["features"].copy(), dtype=_t.float32)
    a = _t.as_tensor(data_d["app_idx"].copy(), dtype=_t.long)
    out = []
    with _t.no_grad():
        for i in range(0, f.shape[0], 2048):
            logits = model(a[i:i + 2048], f[i:i + 2048])
            out.append(_t.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def main():
    art = ROOT / "artifacts"
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

    # Closed-form scores
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

    # Set up trainer + grid for the c3.3 schema
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

    # Build c3.3 features and score the 4 picks onto each split
    PICKS = [
        ("C3.3",            "task_c_c3p3.pt",          "score_c33",          False),
        ("Pro-Reg/c3.3",    "task_c_c3pro_reg.pt",     "score_proreg",       False),
        ("Pro-List/c3.3",   "task_c_c3pro_listwise.pt", "score_prolist",     False),
        ("Pro-Wide/c3.3",   "task_c_c3pro_wide.pt",    "score_prowide",     True),
    ]
    for sp_name, df in (("val", bg_val), ("test", bg_test)):
        d = grid.build_schema_data("c3.3", df, grid_ctx)
        for label, ckpt_name, col, is_wide in PICKS:
            ckpt = torch.load(art / "bg" / "checkpoints" / ckpt_name,
                              map_location="cpu", weights_only=False)
            nf = int(d["features"].shape[1])
            if is_wide:
                model = grid.make_wide_model(num_features=nf, vocab_size=len(vocab),
                                              dropout=0.3) if hasattr(grid, "make_wide_model") else grid.make_baseline_model(num_features=nf, vocab_size=len(vocab), dropout=0.3)
            else:
                model = grid.make_baseline_model(num_features=nf, vocab_size=len(vocab),
                                                  dropout=0.2)
            model.load_state_dict(ckpt_pull(ckpt), strict=False)
            df[col] = 1.0 - forward_probs(model, d)

    # Compute positive-only metrics
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

    out = {"val": {}, "test": {}}
    for sp_name, df in (("val", bg_val), ("test", bg_test)):
        for label, col in REPORT_ROWS:
            out[sp_name][label] = MET.compute_positive_only_metrics(
                df, col, "y_3600", r_values=R_SWEEP)

    # Print
    print(f"\n{'='*120}")
    print(f"{'='*40} POSITIVE-ONLY METRICS H = 60 min {'='*40}")
    print(f"{'='*120}")
    for sp in ("val", "test"):
        print(f"\n=== {sp.upper()} ===")
        print(f"{'Model':<22}{'PosRank↑':<12}{'PosScoreNorm↓':<15}{'WAKR@.25↓':<12}{'WAKR@.5↓':<12}{'WAKR@.75↓':<12}")
        print("-" * 100)
        for label, *_ in REPORT_ROWS:
            m = out[sp][label]
            print(f"{label:<22}"
                  f"{m['pos_rank_mean']:<12.4f}"
                  f"{m['pos_score_norm_mean']:<15.4f}"
                  f"{m['wakr']['0.25']:<12.4f}"
                  f"{m['wakr']['0.5']:<12.4f}"
                  f"{m['wakr']['0.75']:<12.4f}")

    # Save
    out_path = ROOT / "artifacts" / "bg" / "results" / "positive_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[40] wrote {out_path}")
    return 0


def ckpt_pull(ckpt):
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


if __name__ == "__main__":
    sys.exit(main() or 0)
