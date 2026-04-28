"""Build the unified extended-metrics summary across all baselines + neural configs.

Reads:
  - artifacts/multiuser/task_b_baselines_extended.json (4 closed-form baselines)
  - artifacts/multiuser/<config>_aggregate.json for each neural config

Writes:
  - artifacts/multiuser/task_b_extended_summary.json (single JSON keyed by config)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "multiuser"

KEYS = [
    "event_hit_at_1",
    "event_hit_at_3",
    "event_hit_at_5",
    "event_hit_at_dyn",
    "recall_at_5",
    "recall_at_dyn",
    "coverage_at_5",
]


def summarize(rows):
    out = {"n_users": len(rows)}
    for k in KEYS:
        vals = [r.get(k, np.nan) for r in rows]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        if vals:
            out[k + "_mean"] = float(np.mean(vals))
            out[k + "_median"] = float(np.median(vals))
        else:
            out[k + "_mean"] = 0.0
            out[k + "_median"] = 0.0
    g = [r.get("avg_g_size", 0) for r in rows]
    g = [v for v in g if v]
    out["avg_g_size_mean"] = float(np.mean(g)) if g else 0.0
    return out


def main():
    summary = {}

    # Closed-form baselines (keyed as task_b_test_MFU, task_b_test_MRU, etc.)
    cf = json.load(open(ART / "task_b_baselines_extended.json"))
    label_to_key = {
        "MFU": "task_b_test_MFU",
        "MRU": "task_b_test_MRU",
        "HourMFU": "task_b_test_HourMFU",
        "Markov-1": "task_b_test_Markov1",
    }
    for base, key in label_to_key.items():
        rows = [u[key] for u in cf if key in u]
        summary[base] = summarize(rows)
        print(f"  [{base}] n={len(rows)}  EH@5={summary[base]['event_hit_at_5_mean']:.3f}  EH@3={summary[base]['event_hit_at_3_mean']:.3f}  EH@dyn={summary[base]['event_hit_at_dyn_mean']:.3f}")

    # Neural configs
    configs = ["v1_gru", "v1_tgt", "v1_gru_markov", "v2",
               "v3_r4", "v3_r6", "v3_r6_lite", "v3_r6_arch_trim",
               "v4", "v4_trim", "v5_e1", "v5_e2", "v5_e4"]
    for cfg in configs:
        path = ART / f"{cfg}_aggregate.json"
        if not path.exists():
            print(f"  [skip] {cfg}: aggregate file missing")
            continue
        agg = json.load(open(path))
        rows = []
        for u in agg:
            if "error" in u:
                continue
            tb = u.get("test_b", {})
            if tb:
                rows.append(tb)
        summary[cfg] = summarize(rows)
        print(f"  [{cfg}] n={len(rows)}  EH@5={summary[cfg]['event_hit_at_5_mean']:.3f}  EH@3={summary[cfg]['event_hit_at_3_mean']:.3f}  EH@dyn={summary[cfg]['event_hit_at_dyn_mean']:.3f}")

    out = ART / "task_b_extended_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
