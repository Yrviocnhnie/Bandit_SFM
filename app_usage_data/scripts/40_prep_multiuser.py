"""Prep per-user data into artifacts/multiuser/<set>/<uid>/."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import data as D
from lib.multiuser import list_user_files, load_user_xlsx, per_user_split, build_user_vocab, short_uid


def prep_one(set_name, fp, out_root):
    uid = short_uid(fp)
    out_dir = out_root / set_name / uid
    splits = out_dir / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    raw = load_user_xlsx(fp)
    raw = D.dedup(raw)
    enr = D.enrich(raw)
    train, val, test = per_user_split(enr, val_days=3, test_days=3)
    vocab = build_user_vocab(train, min_count=3)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    train.to_parquet(splits / "train.parquet")
    val.to_parquet(splits / "val.parquet")
    test.to_parquet(splits / "test.parquet")
    return {
        "uid": Path(fp).stem.split("_")[0][:12],
        "set": set_name,
        "V": len(vocab),
        "rows_train": len(train),
        "rows_val": len(val),
        "rows_test": len(test),
        "targets_train": int(train["is_target_event"].astype(bool).sum()),
        "targets_val": int(val["is_target_event"].astype(bool).sum()),
        "targets_test": int(test["is_target_event"].astype(bool).sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-files-per-set", type=int, default=None)
    args = p.parse_args()

    files = list_user_files()
    if args.max_files_per_set:
        from collections import defaultdict
        seen = defaultdict(int)
        kept = []
        for s, f in files:
            if seen[s] < args.max_files_per_set:
                kept.append((s, f))
                seen[s] += 1
        files = kept

    out_root = Path("artifacts/multiuser")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[prep] {len(files)} users")
    summary = []
    for i, (s, fp) in enumerate(files):
        t0 = time.time()
        try:
            stats = prep_one(s, fp, out_root)
            summary.append(stats)
            print(f"  {i+1:2d}/{len(files):2d} {s} {stats['uid']} V={stats['V']:3d} "
                  f"tr={stats['rows_train']:>6} va={stats['rows_val']:>5} te={stats['rows_test']:>5} "
                  f"({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  ERROR on {s}/{fp}: {e}")
    with open(out_root / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE. Summary at {out_root}/prep_summary.json")
    return 0


from lib.multiuser import list_user_files


if __name__ == "__main__":
    sys.exit(main() or 0)
