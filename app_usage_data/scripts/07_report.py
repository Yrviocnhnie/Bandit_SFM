"""Aggregate baselines.json, task_a.json, task_b.json into tables and figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    art = ROOT / "artifacts" / "results"
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base = json.load(open(art / "baselines.json"))
    task_a = json.load(open(art / "task_a.json"))
    task_b = json.load(open(art / "task_b.json"))

    # ----- Task A table -----
    rows_a = []
    for mname in ("MFU", "MRU", "HourMFU", "Markov1"):
        for split in ("val", "test"):
            r = base["models"][mname].get(f"task_a_{split}", {})
            rows_a.append({
                "model": mname,
                "split": split,
                "hit@1": r.get("hit_at_1"),
                "hit@5": r.get("hit_at_5"),
                "hit@10": r.get("hit_at_10"),
                "mrr": r.get("mrr"),
                "macro_f1": r.get("macro_f1"),
            })
    for mname in ("GRU", "TGT"):
        for split in ("val", "test"):
            r = task_a["models"].get(mname, {}).get(split, {})
            rows_a_entry = {
                "model": mname,
                "split": split,
                "hit@1": r.get("hit_at_1"),
                "hit@5": r.get("hit_at_5"),
                "hit@10": r.get("hit_at_10"),
                "mrr": r.get("mrr"),
                "macro_f1": r.get("macro_f1"),
            }
            rows_a.append(rows_a_entry)

    # Ensure baseline rows also have hit@5 key name
    for row in rows_a:
        if "hit@1" not in row:
            row["hit@1"] = row.pop("hit_at_1", None)
        if "hit@5" not in row:
            row["hit@5"] = row.pop("hit_at_5", None)
        if "hit@10" not in row:
            row["hit@10"] = row.pop("hit_at_10", None)

    df_a = pd.DataFrame(rows_a)
    df_a.to_csv(art / "task_a_table.csv", index=False)

    # ----- Task B table -----
    rows_b = []
    for mname in ("MFU", "MRU", "HourMFU", "Markov1"):
        for split in ("val", "test"):
            r = base["models"][mname].get(f"task_b_{split}", {})
            rows_b.append({
                "model": mname,
                "split": split,
                "precision@5": r.get("precision@5"),
                "recall@5": r.get("recall@5"),
                "f1@5": r.get("f1@5"),
                "event_hit@5": r.get("event_hit@5"),
                "coverage@5": r.get("coverage@5"),
                "jaccard@5": r.get("jaccard@5"),
            })
    for mname in ("GRU", "TGT"):
        for split in ("val", "test"):
            r = task_b["models"][mname].get(split, {})
            rows_b_entry = {
                "model": mname,
                "split": split,
                "precision@5": r.get("precision@5"),
                "recall@5": r.get("recall@5"),
                "f1@5": r.get("f1@5"),
                "event_hit@5": r.get("event_hit@5"),
                "coverage@5": r.get("coverage@5"),
                "jaccard@5": r.get("jaccard@5"),
            }
            rows_b.append(rows_b_entry)

    df_b = pd.DataFrame(rows_b)
    df_b.to_csv(art / "task_b_table.csv", index=False)

    # Print headline
    print("\n=== Task A (hit@1 / hit@5 / mrr) ===")
    for (mname, split), grp in df_a.groupby(["model", "split"]):
        r = grp.iloc[0]
        print(f"  {mname:8s} {split:4s}  hit@1={r['hit@1']:.3f}  hit@5={r['hit@5']:.3f}  mrr={r['mrr']:.3f}  f1={r['macro_f1']:.3f}")

    print("\n=== Task B (precision@5 / recall@5 / event_hit@5 / coverage@5) ===")
    for (mname, split), g in df_b.groupby(["model", "split"]):
        r = g.iloc[0]
        print(f"  {mname:8s}/{split:4s}  P@5={r['precision@5']:.3f}  R@5={r['recall@5']:.3f}  EH@5={r['event_hit@5']:.3f}  Cov@5={r['coverage@5']:.3f}")

    # ----- Figure: Hit@1 bar chart -----
    fig, ax = plt.subplots(figsize=(8, 4))
    order = ["MFU", "MRU", "HourMFU", "Markov1", "GRU", "TGT"]
    val = []
    test = []
    for m in order:
        g_val = df_a[(df_a["model"] == m) & (df_a["split"] == "val")]
        g_test = df_a[(df_a["model"] == m) & (df_a["split"] == "test")]
        val.append(g_val["hit@1"].iloc[0] if len(g_val) else 0)
        test.append(g_test["hit@1"].iloc[0] if len(g_test) else 0)
    x = np.arange(len(order))
    w = 0.35
    ax.bar(x - w/2, val, width=w, label="val", color="#4c72b0")
    ax.bar(x + w/2, test, width=w, label="test", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Hit@1")
    ax.set_title("Task A — Next-app Hit@1 per model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_fig = ROOT / "figures" / "task_a_hit1.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_fig}")

    # Task B figure
    fig, ax = plt.subplots(figsize=(8, 4))
    val_b = []
    test_b = []
    for m in order:
        g_val = df_b[(df_b["model"] == m) & (df_b["split"] == "val")]
        g_test = df_b[(df_b["model"] == m) & (df_b["split"] == "test")]
        val_b.append(g_val["event_hit@5"].iloc[0] if len(g_val) else 0)
        test_b.append(g_test["event_hit@5"].iloc[0] if len(g_test) else 0)
    ax.bar(np.arange(len(order)) - 0.2, val_b, 0.35, label="val", color="#4c72b0")
    ax.bar(np.arange(len(order)) + 0.2, test_b, 0.35, label="test", color="#dd8452")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("EventHit@5")
    ax.set_title("Task B — 15-min window EventHit@5")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_fig2 = ROOT / "figures" / "task_b_eventhit5.png"
    fig.savefig(out_fig2)
    plt.close(fig)
    print(f"  wrote {out_fig2}")

    print(f"\nTables: {art}/task_a_table.csv and task_b_table.csv")
    return 0


def task_a_rows():
    return None


if __name__ == "__main__":
    sys.exit(main() or 0)
