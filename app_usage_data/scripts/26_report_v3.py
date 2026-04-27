"""Assemble v3 result tables from artifacts/results/task_?_v3_R*.json and emit CSVs + pretty print.

Writes:
  artifacts/results/task_a_v3_table.csv
  artifacts/results/task_b_v3_table.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_round(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    results_dir = ROOT / "artifacts" / "results"
    # --- Task A
    a_rows = []
    for p in sorted(results_dir.glob("task_a_v3_R*.json")):
        if "smoke" in p.name:
            continue
        d = load_round(p)
        v = d.get("val_final_reloaded", d.get("val", {}))
        t = d.get("test", {})
        f = d.get("flags", {})
        a_rows.append({
            "round": p.stem.replace("task_a_v3_", ""),
            "flags": ",".join(k for k, val in f.items() if val) or "-",
            "val_hit1": round(v.get("hit_at_1", 0.0), 4),
            "val_hit5": round(v.get("hit_at_5", 0.0), 4),
            "val_mrr": round(v.get("mrr", 0.0), 4),
            "test_hit1": round(t1 := t.get("hit_at_1", 0.0), 4),
            "test_hit5": round(t.get("hit_at_5", 0.0), 4),
            "test_mrr": round(t.get("mrr", 0.0), 4),
            "params": d.get("n_params", 0),
            "profile_dim": d.get("profile_dim", 0),
        })
    # --- Task B
    b_rows = []
    for p in sorted([x for x in results_dir.glob("task_b_v3_R*.json") if "_smoke" not in x.name] if False else [x for x in sorted(results_dir.glob("task_b_v3_R*.json")) if "_smoke" not in x.name]):
        d = __import__("json").load(open(p))
        v = d.get("val_final_reloaded", d.get("val", {}))
        t = d.get("test", {})
        f = d.get("flags", {})
        b_rows.append({
            "round": p.stem.replace("task_b_v3_", ""),
            "flags": ",".join(k for k, val in f.items() if val) or "-",
            "val_eh5": round(v.get("event_hit_at_5", 0.0), 4),
            "val_r5": round(v.get("recall_at_5", 0.0), 4),
            "val_cov5": round(v.get("coverage_at_5", 0.0), 4),
            "test_eh5": round(t.get("event_hit_at_5", 0.0), 4),
            "test_r5": round(t.get("recall_at_5", 0.0), 4),
            "test_cov5": round(t_cov := t.get("coverage_at_5", 0.0), 4),
            "alpha_markov": d.get("alpha_markov_final"),
            "params": d.get("n_params", 0),
            "profile_dim": d.get("profile_dim", 0),
        })

    def write_csv(path, rows):
        if not rows:
            return
        with open(path, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {path} ({len(rows)} rows)")

    art_res = Path("artifacts") / "results"
    write_csv(art_res / "task_a_v3_table.csv", a_rows)
    write_csv(art_res / "task_b_v3_table.csv", b_rows := b_rows)

    # pretty-print
    print("\n=== Task A (v3) ===")
    for r in a_rows:
        print(f"  {r['round']:>4}  val h1={r['val_hit1']:.3f} h5={r['val_hit5']:.3f} mrr={r['val_mrr']:.3f}  test h1={r['test_hit1']:.3f} h5={r['test_hit5']:.3f}  ({r['params']}p)")
    print("\n=== Task B (v3) ===")
    for r in b_rows:
        a = r.get("alpha_markov")
        a_str = f" α={a:.3f}" if a is not None else ""
        print(f"  {r['round']:>6}  val EH@5={r['val_eh5']:.3f}  test EH@5={r['test_eh5']:.3f}  cov5={r['test_cov5']:.3f}  ({r['params']}p){a_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
