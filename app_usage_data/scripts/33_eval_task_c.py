"""Aggregate baselines + learned-model results and render Pareto figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "artifacts" / "bg" / "results"
FIG_DIR = ROOT / "figures" / "bg"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def collect_all() -> dict:
    out = {s: {"H_300": {}, "H_600": {}} for s in ("train", "val", "test")}
    baseline_path = RESULTS_DIR / "baselines_bg.json"
    if baseline_path.exists():
        bl = json.load(open(baseline_path))
        for split, payload in bl["splits"].items():
            for h, models in payload["metrics"].items():
                for name, m in models.items():
                    out[split][h][name] = m
    for path in sorted(RESULTS_DIR.glob("task_c_*.json")):
        if path.name == "task_c_summary.json":
            continue
        r = json.load(open(path))
        for split in ("val", "test"):
            payload = r.get(split)
            if not payload:
                continue
            for h in ("H_300", "H_600"):
                if h in payload:
                    out[split][h][r["tag"]] = payload[h]
    return out


def flatten_rows(collected: dict) -> pd.DataFrame:
    rows = []
    for split in ("val", "test"):
        for h, per_model in collected[split].items():
            for name, m in per_model.items():
                fk = m.get("false_kill_rate", {})
                msr = m.get("memory_save_rate", {})
                rows.append({
                    "split": split,
                    "horizon_sec": int(h.split("_")[1]),
                    "model": name,
                    "FK_at_0.5": fk.get("0.5", float("nan")),
                    "MSR_at_0.5": msr.get("0.5", float("nan")),
                    "FK_at_0.25": fk.get("0.25", float("nan")),
                    "FK_at_0.75": fk.get("0.75", float("nan")),
                    "PR_AUC": m.get("pr_auc_mean", float("nan")),
                    "ROC_AUC": m.get("roc_auc_mean", float("nan")),
                    "NDCG_half": m.get("ndcg_half", float("nan")),
                    "n_anchors": m.get("n_anchors", 0),
                })
    return pd.DataFrame(rows)


def render_pareto(collected: dict, split: str, h_key: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_list = [0.1, 0.25, 0.5, 0.75, 0.9]
    models = sorted(collected[split][h_key].keys())
    annotate_model = "C1" if "C1" in models else (models[0] if models else None)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for name in models:
        m = collected[split][h_key][name]
        fk = m.get("false_kill_rate", {})
        sr = m.get("memory_save_rate", {})  # legacy key = SafeKillRecall
        xs = [sr.get(str(r), 0.0) for r in r_list]
        ys = [1.0 - fk.get(str(r), 0.0) for r in r_list]
        line, = ax.plot(xs, ys, marker="o", linewidth=1.4,
                         label=name, markersize=5)
        if name == annotate_model:
            for r, x, y in zip(r_list, xs, ys):
                ax.annotate(f"r={r}", xy=(x, y),
                             xytext=(6, -10), textcoords="offset points",
                             fontsize=8, color=line.get_color())

    ax.set_xlabel(
        "SafeKillRecall@r  =  |killed ∩ safe-to-kill| / |safe-to-kill|\n"
        "(higher → more of the reclaimable RAM is actually reclaimed)",
        fontsize=9,
    )
    ax.set_ylabel(
        "1 − FalseKillRate@r  =  1 − |killed ∩ will-be-used| / |killed|\n"
        "(higher → fewer apps killed that the user actually opened)",
        fontsize=9,
    )
    h_sec = int(h_key_local_strip(h_key))
    ax.set_title(
        f"Task C Pareto — {split_upper(split)} split, H = {h_sec} s\n"
        "r = fraction of |B(t)| evicted per anchor; upper-right dominates",
        fontsize=10,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.92, 1.005)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def h_key_local_strip(h_key: str) -> int:
    return int(h_key.split("_")[1])


def split_upper(s: str) -> str:
    return s.upper()


def main() -> int:
    import matplotlib  # imported here so headless works
    matplotlib.use("Agg")
    collected = collect_all()

    df = flatten_rows(collected)
    df.sort_values(["split", "horizon_sec", "ROC_AUC"],
                   ascending=[True, True, False], inplace=True)
    csv_path = RESULTS_DIR / "task_c_summary.csv"
    json_path = RESULTS_DIR / "task_c_summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({"rows": df.to_dict(orient="records")}, f, indent=2)
    print(f"[bg/33] wrote {json_path}")
    print(f"[bg/33] wrote {csv_path}")

    for split in ("val", "test"):
        for h in ("H_300", "H_600"):
            out = FIG_DIR / f"pareto_{h}_{split}.png"
            try:
                render_pareto(collected, split=split, h_key=h, out_path=out)
                print(f"[bg/33] wrote {out}")
            except Exception as exc:
                print(f"[bg/33] figure skipped ({out.name}): {exc}")

    print("\n=== TEST SPLIT HEADLINE ===")
    test_df = df[df["split"] == "test"].sort_values(["horizon_sec", "ROC_AUC"],
                                                      ascending=[True, False])
    cols = ["horizon_sec", "model", "FK_at_0.5", "MSR_at_0.5",
            "PR_AUC", "ROC_AUC", "NDCG_half"]
    print(test_df[cols].to_string(index=False))
    return 0


RESULTS_DIR = ROOT / "artifacts" / "bg" / "results"


import json  # keep a stable import for json_path usage above
json_path = RESULTS_DIR / "task_c_summary.json"
csv_path = RESULTS_DIR / "task_c_summary.csv"


if __name__ == "__main__":
    sys.exit(main() or 0)
