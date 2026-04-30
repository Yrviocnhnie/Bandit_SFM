"""Plot focused H=60 Pareto from saved JSON results.

Reads the per-r FK / MSR means from baselines_bg.json and the trained-model
JSONs (no checkpoint reloading needed). Plots:
    x-axis: Memory-save rate  (higher = better)
    y-axis: False-kill rate   (lower = better, axis inverted so good = down)
    Ideal corner = bottom-right.

Picks (4 best trained models + 3 reference baselines):
    Random, LRU, Markov-inverse,
    C3.3, Pro-Reg/c3.3, Pro-Wide/c3.3, Pro-List/c3.3

Outputs:
    figures/bg/pareto_h60_focused_{val,test}.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "bg" / "results"
FIG = ROOT / "figures" / "bg"

R_SWEEP = ("0.1", "0.25", "0.5", "0.75", "0.9")
H = "H_3600"

ROWS = [
    ("Random",          "baseline", "random",          "#7f7f7f", ":"),
    ("LRU",             "baseline", "lru",             "#1f77b4", "--"),
    ("Markov-inverse",  "baseline", "markov_inv",      "#d62728", "--"),
    ("C3.3",            "trained",  "task_c_c3p3.json",                  "#2ca02c", "-"),
    ("Pro-Reg / c3.3",  "trained",  "task_c_c3pro_reg_c3p3.json",        "#9467bd", "-"),
    ("Pro-Wide / c3.3", "trained",  "task_c_c3pro_wide_c3p3.json",       "tab:olive", "-"),
    ("Pro-List / c3.3", "trained",  "task_c_c3pro_listwise_c3p3.json",   "tab:brown", "-"),
]


def metrics_for(kind, key, split):
    if kind == "baseline":
        bj = json.load(open(ART / "baselines_bg.json"))
        return bj["splits"][split]["metrics"][H][key]
    j = json.load(open(ART / key))
    if "H_3600_val" in j:
        return j[f"H_3600_{split}"]
    return j[split]


def points(m):
    msr = [m["memory_save_rate"][r] for r in R_SWEEP]
    fk  = [m["false_kill_rate"][r]  for r in R_SWEEP]
    return msr, fk


def plot_one(split):
    plt.figure(figsize=(8.5, 6.0))
    for label, kind, key, color, ls in ROWS:
        m = metrics_for(kind, key, split)
        msr, fk = points(m)
        # Plot 1 - FK on the y-axis so both axes are "higher is better"
        kp = [1.0 - v for v in fk]   # kill-precision = fraction of kills that were safe
        plt.plot(msr, kp, marker="o", color=color, linestyle=ls,
                 linewidth=2.0, markersize=6, label=label)
        if label == "C3.3":
            for r, x, y in zip(R_SWEEP, msr, kp):
                plt.annotate(f"r={r}", xy=(x, y),
                             xytext=(6, -10), textcoords="offset points",
                             fontsize=8, color="dimgray")
    plt.xlabel("Memory-save rate (MSR)\n"
               "fraction of safely-killable apps actually killed  (higher = better)",
               fontsize=10)
    plt.ylabel("Kill precision  =  1 − False-kill rate\n"
               "fraction of killed apps the user does NOT need  (higher = better)",
               fontsize=10)
    plt.title(f"H = 60 min Pareto frontier — {split.upper()}\n"
              f"4 best trained models vs Random / LRU / Markov-inverse  ·  ideal corner = upper-right",
              fontsize=11)
    plt.legend(loc="lower left", fontsize=9, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG / f"pareto_h60_focused_{split}.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"[39] wrote {out}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plot_one_check = [(label, kind, ref) for label, kind, ref, _, _ in ROWS
                      if kind == "trained" and not (ART / ref).exists()]
    if plot_one_check:
        print(f"[39] WARNING: missing trained JSONs: {plot_one_check}")
    for split in ("val", "test"):
        plot_one(split)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
