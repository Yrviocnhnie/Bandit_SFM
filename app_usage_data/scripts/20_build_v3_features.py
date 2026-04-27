"""Build v3 feature artifacts (all stats fit on train only).

Outputs under `artifacts/v3/`:
  - category_map.json        : vocab + taxonomy provenance
  - loc_vocab.pkl            : train-only location vocab (top-K + PAD/NONE/OTHER)
  - markov_prior.pkl         : (V,V) log P(next | last) fit on train
  - loc_ids_{train,val,test}.npy : per-row loc_id aligned with each split
  - provenance.json          : coverage + sizes for sanity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.v3 import categories as CAT
from lib.v3 import location as LOC
from lib.v3 import markov_prior as MK


def load_raw_payload(root: Path) -> pd.DataFrame:
    raw_path = root / "app_usage_cleaned_dictionary_mapped.xlsx"
    df = pd.read_excel(raw_path, sheet_name="final_compact", engine="openpyxl")
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    return df[["raw_row_id", "event_ts", "device_state_update_payload"]].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k-loc", type=int, default=15)
    parser.add_argument("--alpha-markov", type=float, default=0.5)
    args = parser.parse_args()

    art = Path("artifacts")
    v3_dir = art / "v3"
    v3_dir.mkdir(parents=True, exist_ok=True)

    # --- load splits + vocab
    train = pd.read_parquet(art / "splits" / "train.parquet")
    val = pd.read_parquet(art / "splits" / "val.parquet")
    test = pd.read_parquet(art / "splits" / "test.parquet")
    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    V = len(vocab)
    print(f"[v3] train={len(train)}  val={len(val)}  test={len(test)}  V={V}")

    # --- category map
    with open(v3_dir / "category_map.json", "w") as f:
        json.dump({
            "num_categories": CAT.NUM_CATEGORIES,
            "categories": CAT.CATEGORIES,
            "app_to_category": CAT.APP_TO_CATEGORY,
        }, f, indent=2)
    print(f"[v3] category_map.json written ({CAT.NUM_CATEGORIES} categories)")

    # --- location vocab — fit on train only (with payload merged in)
    print("[v3] loading raw Excel for payload …")
    raw = load_raw_payload(ROOT)
    # Merge payload into the parquet splits via raw_row_id
    train_p = train.merge(raw[["raw_row_id", "device_state_update_payload"]],
                          on="raw_row_id", how="left").sort_values("event_ts").reset_index(drop=True)
    val_p = val.merge(raw[["raw_row_id", "device_state_update_payload"]],
                      on="raw_row_id", how="left").sort_values("event_ts").reset_index(drop=True)
    test_p = test.merge(raw[["raw_row_id", "device_state_update_payload"]],
                        on="raw_row_id", how="left").sort_values("event_ts").reset_index(drop=True)
    n_nn_train = int(train_p["device_state_update_payload"].notna().sum())
    print(f"[v3] train rows with non-null payload: {n_nn_train} / {len(train_p)}")

    loc_stats = LOC.fit_location_vocab(train_p, top_k=args.top_k_loc)
    LOC.save_location_vocab(loc_stats, ROOT / "artifacts" / "v3" / "loc_vocab.pkl")
    print(f"[v3] loc vocab size {len(loc_stats['vocab'])}; top: {list(loc_stats['top_counts'].items())[:5]}")

    # Attach per-row loc_id for each split
    loc_train = LOC.attach_loc_id(train_p, loc_stats)
    loc_val = LOC.attach_loc_id(val_p, loc_stats)
    loc_test = LOC.attach_loc_id(test_p, loc_stats)
    np.save(v3_dir / "loc_ids_train.npy", loc_train)
    np.save(v3_dir / "loc_ids_val.npy", loc_val)
    np.save(v3_dir / "loc_ids_test.npy", loc_test)

    none_idx = loc_stats["vocab"]["<NONE>"]
    frac_resolved_train = float((loc_train_arr := loc_train) is not None and (loc_train != none_idx).mean())
    frac_resolved_val = float((loc_val != none_idx).mean())
    frac_resolved_test = float((loc_test != none_idx).mean())
    print(
        f"[v3] loc resolved — train {frac_resolved_train:.1%}  val {frac_resolved_val:.1%}  test {frac_resolved_test:.1%}"
    )

    # --- Markov prior — train only
    mk = MK.fit_markov_prior(train, vocab=_load_vocab(art), alpha=args.alpha_markov)
    MK.save_markov_prior(mk, v3_dir / "markov_prior.pkl")
    print(f"[v3] markov prior fit: V={mk['V']} alpha={mk['alpha']}")

    # --- category map coverage
    vocab_apps = set(vocab.keys())
    mapped = sum(1 for a in vocab_apps if a in CAT.APP_TO_CATEGORY)
    print(f"[v3] categories: {mapped}/{len(vocab_apps)} vocab apps explicitly mapped")

    # --- provenance
    with open(v3_dir / "provenance.json", "w") as f:
        json.dump({
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "V": len(vocab),
            "num_categories": CAT.__dict__.get("NUM_CATEGORIES", 11),
            "loc_vocab_size": len(mk["log_prior"]) if False else len(loc_stats["vocab"]),
            "markov_alpha": mk["alpha"],
            "train_loc_resolved_frac": frac_resolved_train,
            "val_loc_resolved_frac": frac_resolved_val,
            "test_loc_resolved_frac": frac_resolved_test,
        }, f, indent=2)
    print("[v3] DONE")
    return 0


def _load_vocab(art: Path) -> dict:
    with open(art / "vocab.json") as f:
        return json.load(f)


if __name__ == "__main__":
    sys.exit(main() or 0)
