"""Aggregate baselines + learned-model results and render Pareto figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

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


def render_pareto(collected: dict, split: str, h_key: str, out_path: Path,
                   annotate_model: Optional[str] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_list = [0.1, 0.25, 0.5, 0.75, 0.9]
    models = sorted(collected[split][h_key].keys())
    if annotate_model is None:
        # Default policy: prefer the latest learned model for r-value annotations.
        if "C2" in models:
            annotate_model = "C2"
        elif "C1" in models:
            annotate_model = "C1"
        else:
            annotate_model = models[0] if models else None

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


def render_pareto_focused(collected: dict, split: str, h_key: str,
                           out_path: Path,
                           include_models=("C2", "markov_inv", "lru", "random")):
    """Compact Pareto with only the headline model + key baselines for clarity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_list = [0.1, 0.25, 0.5, 0.75, 0.9]
    avail = collected[split][h_key]
    models = [m for m in include_models if m in avail]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    color_map = {"C2": "tab:orange", "markov_inv": "tab:olive",
                 "lru": "tab:gray", "random": "tab:purple"}
    for name in models:
        m = avail[name]
        fk = m.get("false_kill_rate", {})
        sr = m.get("memory_save_rate", {})
        xs = [sr.get(str(r), 0.0) for r in r_list]
        ys = [1.0 - fk.get(str(r), 0.0) for r in r_list]
        line, = ax.plot(xs, ys, marker="o", linewidth=2.0,
                         label=name, markersize=7,
                         color=color_map.get(name))
        # annotate r-values on the headline model only
        if name == "C2" or (name == models[0] and "C2" not in models):
            for r, x, y in zip(r_list, xs, ys):
                ax.annotate(f"r={r}", xy=(x, y),
                             xytext=(7, -11), textcoords="offset points",
                             fontsize=9, color=line.get_color(),
                             fontweight="bold")

    ax.set_xlabel(
        "SafeKillRecall@r  =  |killed ∩ safe-to-kill| / |safe-to-kill|\n"
        "(higher → more reclaimable RAM is actually reclaimed)",
        fontsize=9,
    )
    ax.set_ylabel(
        "1 − FalseKillRate@r  =  1 − |killed ∩ will-be-used| / |killed|\n"
        "(higher → fewer apps killed that the user actually opened)",
        fontsize=9,
    )
    h_sec = int(h_key.split("_")[1])
    ax.set_title(
        f"Task C — C2 vs strongest baselines | {split.upper()} split, H = {h_sec} s",
        fontsize=11,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.92, 1.005)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    import matplotlib.pyplot as plt_close
    plt_close.close(fig)


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
            # Default Pareto: annotate C2 (current headline). Filename
            # `pareto_<H>_<split>.png` matches the v2 report's references.
            out_c2 = FIG_DIR / f"pareto_{h}_{split}.png"
            try:
                render_pareto(collected, split=split, h_key=h, out_path=out_c2,
                               annotate_model="C2")
                print(f"[bg/33] wrote {out_c2}")
            except Exception as exc:
                print(f"[bg/33] figure skipped ({out_c2.name}): {exc}")
            # Legacy Pareto preserved: annotate C1, save under `_c1annot` suffix.
            out_c1 = FIG_DIR / f"pareto_{h}_{split}_c1annot.png"
            try:
                render_pareto(collected, split=split, h_key=h, out_path=out_c1,
                               annotate_model="C1")
                print(f"[bg/33] wrote {out_c1}")
            except Exception as exc:
                print(f"[bg/33] legacy figure skipped ({out_c1.name}): {exc}")
            # Focused 4-model variant (always C2-annotated; nothing to keep "old" of).
            out_focus = FIG_DIR / f"pareto_focused_{h}_{split}.png"
            try:
                render_pareto_focused(collected, split=split, h_key=h, out_path=out_focus)
                print(f"[bg/33] wrote {out_focus}")
            except Exception as exc:
                print(f"[bg/33] focused figure skipped ({out_focus.name}): {exc}")

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
