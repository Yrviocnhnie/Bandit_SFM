"""Create the 'always trigger' vs 'surprise trigger' comparison chart."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEMO = Path(__file__).parent
ART = DEMO / "artifacts"
PLOTS = ART / "plots"


def main():
    s = json.loads((ART / "scores.json").read_text())
    records = s["records"]

    # Use ROUTINE + MILD + MODERATE + STRONG + EXTREME test rows as a
    # 'simulated stream of events the rules engine would have fired on'.
    stream = [r for r in records if r["category"] in
              ("ROUTINE", "MILD", "MODERATE", "STRONG", "EXTREME")]
    total = len(stream)
    baseline = total  # always-trigger would fire on every event
    surprise = sum(1 for r in stream if r["novelty"] > 1.0)

    # break down surprise hits by tier
    tier_total = {}
    tier_hits = {}
    for r in stream:
        tier_total[r["category"]] = tier_total.get(r["category"], 0) + 1
        if r["novelty"] > 1.0:
            tier_hits[r["category"]] = tier_hits.get(r["category"], 0) + 1

    # Plot: stacked counts per tier, with hit rate
    tiers = ["ROUTINE", "MILD", "MODERATE", "STRONG", "EXTREME"]
    colors = ["limegreen", "gold", "orange", "red", "darkred"]
    always = [tier_total.get(t, 0) for t in tiers]
    only_surprise = [tier_hits.get(t, 0) for t in tiers]

    x = np.arange(len(tiers))
    w = 0.38
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, always, width=w, color="lightgrey", edgecolor="black", label="Always trigger")
    ax.bar(x + w/2, only_surprise, width=w, color=colors, edgecolor="black",
           label="Surprise trigger (score > 1.0)")
    for i, (a, s2) in enumerate(zip(always, only_surprise)):
        ax.text(x[i] - w/2, a + 0.5, str(a), ha="center")
        ax.text(x[i] + w/2, s2 + 0.5, str(s2), ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("# events / triggers")
    ax.set_title(f"Total events: {total}  |  Always-trigger: {baseline}  |  Surprise-trigger: {surprise}  "
                 f"({surprise/max(1,baseline)*100:.0f}% of always)")
    ax.legend()
    fig = ax.get_figure()
    fig.tight_layout()
    out = PLOTS / "trigger_summary.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)
    print(f"Always trigger: {baseline}  |  Surprise trigger: {surprise}  ({surprise*100/max(1,baseline):.1f}% of events)")
    for t in tiers:
        print(f"  tier {t}: {tier_hits.get(t, 0)}/{tier_total.get(t, 0)} triggered")


if __name__ == "__main__":
    main()
