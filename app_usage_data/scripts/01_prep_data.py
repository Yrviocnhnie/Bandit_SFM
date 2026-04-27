"""Load xlsx -> dedup -> enrich -> chronological split -> vocab -> parquet artifacts.

Outputs:
    artifacts/splits/{train,val,test}.parquet
    artifacts/vocab.json
    artifacts/session_stats.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow "python scripts/01_prep_data.py" from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import data as D  # noqa: E402
from lib import features as F  # noqa: E402


XLSX_PATH = ROOT / "app_usage_cleaned_dictionary_mapped.xlsx"
SPLITS_DIR = ROOT / "artifacts" / "splits"
VOCAB_JSON = ROOT / "artifacts" / "vocab.json"
STATS_JSON = ROOT / "artifacts" / "session_stats.json"


def main() -> int:
    print(f"[prep] loading {XLSX_PATH}")
    df = D.load_xlsx(str(XLSX_PATH))
    n_raw = len(df)

    print(f"[prep] dedup from {n_raw} rows ...")
    df = D.dedup(df)
    n_dedup = len(df)
    print(f"[prep] dedup kept {n_dedup}/{n_raw} rows ({n_raw - n_dedup} dropped)")

    print(f"[prep] enriching ...")
    df = D.enrich(df)
    n_sessions = int(df["session_id"].nunique())
    n_target = int(df["is_target_event"].sum())
    print(f"[prep]   sessions (capped at {D.SESSION_MAX_EVENTS}): {n_sessions}")
    print(f"[prep]   target events: {n_target}")

    # ----- Split chronologically -----
    splits = D.split_chronological(df)
    print(
        f"[prep] split: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}"
    )
    for name, part in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        nt = int(part["is_target_event"].sum())
        print(f"[prep]   {name}: rows={len(part)}  targets={nt}")

    # ----- Vocab -----
    vocab = D.build_vocab(splits.train, min_count=D.RARE_MIN_COUNT)
    print(f"[prep] vocab size: {len(vocab)} (reserved: {D.RESERVED_TOKENS})")

    # coverage check on test
    test_target_apps = splits.test.loc[splits.test["is_target_event"], "app_label_clean"].fillna("<UNK>")
    covered = test_target_apps.apply(lambda a: a in vocab).mean() if len(test_target_apps) else 1.0
    print(f"[prep] test target-app direct-hit coverage (exact vocab membership): {covered:.3%}")

    # ----- Write artifacts -----
    SPLITS_DIR = ROOT / "artifacts" / "splits"
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        out = SPLITS_DIR / f"{name}.parquet"
        # Drop the raw JSON payload column that bloats parquet / can trip pyarrow
        part.drop(columns=["device_state_update_payload"], errors="ignore").to_parquet(out, index=False)
        print(f"[prep] wrote {out}")

    VOCAB_JSON.parent.mkdir(parents=True, exist_ok=True)
    vocab = D.build_vocab(splits.train if False else splits.train, min_count=D.RARE_MIN_COUNT)
    with open(VOCAB_JSON, "w") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[prep] wrote {VOCAB_JSON}")

    # ----- Session stats -----
    stats = {
        "rows_raw": int(n_raw),
        "rows_deduped": int(n_dedup),
        "targets": int(n_target),
        "sessions": int(n_sessions),
        "vocab_size": int(len(vocab)),
        "train_targets": int(splits.train["is_target_event"].sum()),
        "val_targets": int(splits.val["is_target_event"].sum()),
        "test_targets": int(splits.test["is_target_event"].sum()),
        "test_vocab_coverage": float(covered),
        "session_idle_sec": D.SESSION_IDLE_SEC,
        "session_max_events": D.SESSION_MAX_EVENTS,
        "rare_min_count": D.RARE_MIN_COUNT,
    }
    STATS_PATH = ROOT / "artifacts" / "session_stats.json"
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # ----- Asserts -----
    assert n_target >= 7000, f"unexpected target count {n_target}"
    assert len(vocab) >= 20, f"vocab too small: {len(vocab)}"
    assert covered >= 0.90, f"test vocab coverage too low: {covered:.2%}"

    print(f"[prep] OK | targets={n_target} vocab={len(vocab)} coverage={covered:.2%}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
