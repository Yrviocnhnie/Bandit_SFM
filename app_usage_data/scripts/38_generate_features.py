"""End-to-end Task-C feature generator with sanity checks.

Walks the full chain from raw events to the C3.x feature matrices and emits a
JSON sanity report. Useful as a reproducibility recipe and as a regression
test that all features remain valid after pipeline changes.

Pipeline:
    1. Load split parquets (artifacts/splits/{train,val,test}.parquet).
    2. Load BG parquets (artifacts/bg/splits/bg_{train,val,test}.parquet).
    3. Fit train-only stats: hour-freq, app/cat lifetime priors, Markov tables,
       fg_timeline, per-app inter-FG mean.
    4. For each schema in {c1, c2, c3.1, c3.2, c3.3}, build the feature matrix
       and emit shape/dtype/range/NaN/Inf/sample-row diagnostics.
    5. Save artifacts/bg/results/feature_validation_report.json.

Usage:
    python scripts/38_generate_features.py
    python scripts/38_generate_features.py --schemas c3.2 c3.3
    python scripts/38_generate_features.py --splits train val test
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.v3 import categories as CAT
from lib.v3 import markov_prior as MK


SCHEMA_EXPECTED_DIM = {
    "c1": 14,
    "c2": 15,
    "c3.1": 20,
    "c3.2": 29,
    "c3.3": 32,
}
H_SEC = 3600
Y_COL = f"y_{H_SEC}"


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_trainer():
    return load_module("trainer_h60", ROOT / "scripts" / "34_train_task_c_h60.py")


def load_grid():
    return load_module("trainer_grid_36", ROOT / "scripts" / "36_train_c3_grid.py")


def load_event_splits(art):
    paths = {n: art / "splits" / f"{n}.parquet" for n in ("train", "val", "test")}
    for n, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p} — run `python scripts/01_prep_data.py` first"
            )
    train = pd.read_parquet(paths["train"])
    val = pd.read_parquet(paths["val"])
    test = pd.read_parquet(paths["test"])
    print(f"[38] events: train={len(train):,} val={len(val):,} test={len(test):,}")
    return train, val, test


def load_bg_splits(art):
    paths = {n: art / "bg" / "splits" / f"bg_{n}.parquet"
             for n in ("train", "val", "test")}
    for n, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p} — run `python scripts/30_build_bg_data.py` first"
            )
    bg_train = pd.read_parquet(paths["train"])
    bg_val = pd.read_parquet(paths["val"])
    bg_test = pd.read_parquet(paths["test"])
    print(f"[38] bg rows : train={len(bg_train):,} val={len(bg_val):,} test={len(bg_test):,}")
    print(f"[38] anchors : train={bg_train['anchor_id'].nunique()} "
          f"val={bg_val['anchor_id'].nunique()} test={bg_test['anchor_id'].nunique()}")
    print(f"[38] pos-rate y_3600 : train={bg_train['y_3600'].mean():.4f} "
          f"val={bg_val['y_3600'].mean():.4f} test={bg_test['y_3600'].mean():.4f}")
    return bg_train, bg_val, bg_test


def fit_all_stats(art, train_events, val_events, test_events, bg_train):
    """Fit every train-only stat used by C1/C2/C3.x feature builders."""
    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    trainer = load_trainer()
    grid = load_grid()

    print("[38] fitting hour_freq, markov_probs, app_to_cat ...")
    hour_freq = trainer.fit_hour_freq(train_events, vocab)
    markov_probs = MK.load_markov_prior(art / "v3" / "markov_prior.pkl")["probs"].astype(np.float32)
    app_to_cat = CAT.build_app_to_cat_idx(vocab)

    print("[38] fitting per-app inter-FG mean + full-stream FG timeline ...")
    per_app_inter_mean = trainer.fit_per_app_inter_fg_mean(train_events, vocab)
    all_events = pd.concat([train_events, val_events, test_events], ignore_index=True) \
        .sort_values("event_ts").reset_index(drop=True)
    fg_timeline = trainer.collect_fg_timeline(all_events, vocab)

    print("[38] fitting C3.2/C3.3 stats (lifetime shares, kill rates, cat-Markov) ...")
    app_lifetime_share = grid.fit_app_lifetime_share(train_events, vocab)
    app_lifetime_kill_rate = grid.fit_app_lifetime_kill_rate(bg_train, vocab)
    cat_lifetime_share = grid.fit_cat_lifetime_share(train_events, vocab, app_to_cat)
    cat_markov_probs = grid.fit_cat_markov(train_events, vocab, app_to_cat)

    return {
        "trainer": trainer,
        "grid": grid,
        "vocab": vocab,
        "hour_freq": hour_freq,
        "markov_probs": markov_probs,
        "app_to_cat": app_to_cat,
        "per_app_inter_mean": per_app_inter_mean,
        "fg_timeline": fg_timeline,
        "app_lifetime_share": app_lifetime_share,
        "app_lifetime_kill_rate": app_lifetime_kill_rate,
        "cat_lifetime_share": cat_lifetime_share,
        "cat_markov_probs": cat_markov_probs,
    }


def build_schema(schema, bg_df, ctx):
    """Dispatch to the right feature builder for a given schema name."""
    trainer = ctx["trainer"]
    if schema == "c1":
        return trainer.build_c1(bg_df, ctx["hour_freq"], ctx["markov_probs"], ctx["app_to_cat"])
    if schema == "c2":
        return trainer.build_c2(bg_df, ctx["hour_freq"], ctx["markov_probs"])
    if schema == "c3.1":
        return trainer.build_c31(
            bg_df, ctx["hour_freq"], ctx["markov_probs"],
            ctx["fg_timeline"], ctx["per_app_inter_mean"],
        )
    return ctx["grid"].build_schema_data(schema, bg_df, ctx)


def array_summary(arr):
    a = np.asarray(arr)
    if a.dtype.kind in "fc":
        n_nan = int(np.isnan(a).sum())
        n_inf = int((np.isinf(a)).sum())
    else:
        n_nan = n_inf = 0
    finite = np.isfinite(a) if a.dtype.kind in "fc" else np.ones_like(a, dtype=bool)
    sub = a[finite] if finite.any() else np.array([0.0])
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": float(sub.min()),
        "max": float(sub.max()),
        "mean": float(sub.mean()),
        "n_nan": n_nan,
        "n_inf": n_inf,
    }


def schema_report(schema, data, expected_dim, sample_n=2):
    feats = data["features"]
    rep = array_summary(feats)
    rep["schema"] = schema
    rep["n_rows"] = int(feats.shape[0])
    rep["n_cols"] = int(feats.shape[1])
    rep["expected_dim"] = expected_dim
    rep["dim_ok"] = (rep["n_cols"] == expected_dim)
    rep["y_pos_rate"] = float(np.asarray(data["y"]).mean())
    rep["n_anchors"] = int(np.unique(data["anchor_id"]).size)
    rep["sample_rows"] = []
    for i in range(min(sample_n, feats.shape[0])):
        rec = {
            "anchor_id": int(data["anchor_id"][i]),
            "app_idx": int(data["app_idx"][i]),
            "y": int(data["y"][i]),
            "feat_first6": [round(float(v), 4) for v in feats[i, :6]],
            "feat_last6": [round(float(v), 4) for v in feats[i, -6:]],
        }
        if data.get("cat_idx") is not None:
            rec["cat_idx"] = int(data["cat_idx"][i])
        rep["sample_rows"].append(rec)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemas", nargs="+", default=list(SCHEMA_EXPECTED_DIM),
                    choices=list(SCHEMA_EXPECTED_DIM))
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    choices=["train", "val", "test"])
    args = ap.parse_args()

    art = ROOT / "artifacts"
    t0 = time.time()
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schemas_run": args.schemas,
        "splits_run": args.splits,
        "expected_dim": SCHEMA_EXPECTED_DIM,
        "issues": [],
    }

    print("\n=== Stage 1: load event splits ===")
    train_events, val_events, test_events = load_event_splits(art)
    report["row_counts_events"] = {
        "train": len(train_events), "val": len(val_events), "test": len(test_events),
    }

    print("\n=== Stage 2: load BG splits ===")
    bg_train, bg_val, bg_test = load_bg_splits(art)
    report["row_counts_bg"] = {
        "train": len(bg_train), "val": len(bg_val), "test": len(bg_test),
    }
    report["pos_rate_y_3600"] = {
        "train": float(bg_train["y_3600"].mean()),
        "val": float(bg_val["y_3600"].mean()),
        "test": float(bg_test["y_3600"].mean()),
    }
    bg_map = {"train": bg_train, "val": bg_val, "test": bg_test}

    print("\n=== Stage 3: train-only stat fits ===")
    ctx = fit_all_stats(art, train_events, val_events, test_events, bg_train)
    print(f"[38] hour_freq.shape={ctx['hour_freq'].shape}, "
          f"markov.shape={ctx['markov_probs'].shape}, "
          f"cat_markov.shape={ctx['cat_markov_probs'].shape}")
    print(f"[38] app_lifetime_share top-5 apps: "
          f"{[round(float(x), 4) for x in np.sort(ctx['app_lifetime_share'])[::-1][:5]]}")
    report["stat_shapes"] = {
        "hour_freq": list(ctx["hour_freq"].shape),
        "markov_probs": list(ctx["markov_probs"].shape),
        "cat_markov_probs": list(ctx["cat_markov_probs"].shape),
        "app_to_cat": list(ctx["app_to_cat"].shape),
    }

    print("\n=== Stage 4: build feature schemas ===")
    report["schema_reports"] = {}
    for sch in args.schemas:
        print(f"\n[38] schema={sch}")
        for sp in args.splits:
            df = bg_map[sp]
            data = build_schema(sch, df, ctx)
            sr = schema_report(sch, data, SCHEMA_EXPECTED_DIM[sch])
            report["schema_reports"][f"{sch}/{sp}"] = sr
            ok = "OK" if (sr["dim_ok"] and sr["n_nan"] == 0 and sr["n_inf"] == 0) else "FAIL"
            print(f"   {sp:<5} {ok}  rows={sr['n_rows']}  cols={sr['n_cols']}/{sr['expected_dim']}  "
                  f"NaN={sr['n_nan']} Inf={sr['n_inf']}  pos={sr['y_pos_rate']:.4f}  "
                  f"min={sr['min']:.3f}  max={sr['max']:.3f}")
            if ok == "FAIL":
                report["issues"].append(
                    f"{sch}/{sp}: cols={sr['n_cols']}/exp={sr['expected_dim']}, "
                    f"NaN={sr['n_nan']}, Inf={sr['n_inf']}"
                )

    out_path = art / "bg" / "results" / "feature_validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    elapsed = time.time() - t0
    print(f"\n[38] wrote {out_path}  (elapsed {elapsed:.1f}s)")
    if report["issues"]:
        print(f"[38] {len(report['issues'])} issues:")
        for issue in report["issues"]:
            print(f"  - {issue}")
        return 1
    print("[38] all schemas passed ✓")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    sys.exit(main() or 0)
