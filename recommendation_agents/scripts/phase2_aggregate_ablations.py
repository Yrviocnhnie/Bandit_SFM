#!/usr/bin/env python3
"""Aggregate phase-2 ablation results into a single comparison markdown."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("artifacts/phase2/ablations")
VANILLA_PATH = Path("artifacts/phase2/feedback_propagation_phase2_v1/results_phase2.json")

ALL_CONTEXTS = [
    "ARRIVE_OFFICE_A",
    "ARRIVE_OFFICE_B",
    "ARRIVE_OFFICE_C",
    "CAFE_QUIET_A",
    "CAFE_QUIET_B",
    "LEAVE_OFFICE_A",
    "LEAVE_OFFICE_B",
    "OFFICE_LUNCH_OUT_A",
    "OFFICE_LUNCH_OUT_B",
]

NEEDS_PROMO = [
    "ARRIVE_OFFICE_C",
    "CAFE_QUIET_A",
    "CAFE_QUIET_B",
    "LEAVE_OFFICE_A",
    "OFFICE_LUNCH_OUT_A",
    "OFFICE_LUNCH_OUT_B",
]


def safe_num(val: Any) -> float:
    try:
        v = float(val)
        if math.isnan(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def fmt_signed(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:+.3f}"


def load_ablations(root: Path, include_vanilla: bool) -> dict[str, dict]:
    out = {}
    if include_vanilla and VANILLA_PATH.exists():
        out["vanilla_baseline"] = json.loads(VANILLA_PATH.read_text())
    if root.exists():
        for sub in sorted(root.iterdir()):
            if not sub.is_dir() or sub.name == "logs":
                continue
            f = sub / "results_phase2.json"
            if f.exists():
                out[sub.name] = json.loads(f.read_text())
    return out


def pick_best_condition(results: dict) -> dict:
    """Return the condition with the best overall improvement.

    Priority order:
      1. higher avg_anchor_rank_delta
      2. higher num_anchors_in_top3_after
      3. higher V6 most_relevant top-3 coverage delta (after - before)
      4. higher num_anchors_improved
    """
    best = None
    best_key = (-math.inf, -1, -math.inf, -1)
    for cond in results["conditions"]:
        agg = cond["aggregate"]
        anchor = safe_num(agg["avg_anchor_rank_delta"])
        top3 = agg.get("num_anchors_in_top3_after", 0) or 0
        improved = agg.get("num_anchors_improved", 0) or 0
        q_before = cond["selected_scenarios_quality_before"]["avg_most_relevant_covered_in_topk"]
        q_after = cond["selected_scenarios_quality_after"]["avg_most_relevant_covered_in_topk"]
        v6_delta = q_after - q_before
        key = (anchor, top3, v6_delta, improved)
        if key > best_key:
            best_key = key
            best = cond
    return best


def context_rank_after(cond: dict) -> dict[str, int]:
    return {fr["phase2_context_id"]: fr["anchor"]["after"]["rank"] for fr in cond["feedback_results"]}


def render_markdown(ablations: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("# Phase-2 Ablation Comparison")
    lines.append("")
    lines.append(f"Total ablations: {len(ablations)}")
    lines.append("")

    any_data = next(iter(ablations.values()))
    baseline_rank = {
        item["phase2_context_id"]: item["baseline_anchor_gt_rank"]
        for item in any_data["feedback_items"]
    }

    # Precompute best condition per ablation
    best_map: dict[str, dict] = {}
    for name, data in ablations.items():
        if data.get("conditions"):
            best_map[name] = pick_best_condition(data)

    # -------- Table 1: global aggregates --------
    lines.append("## 1. Best condition per ablation (sorted by anchor delta)")
    lines.append("")
    lines.append(
        "For each ablation, the condition with the highest `avg_anchor_rank_delta` is selected "
        "(ties broken by smallest |cross_scenario_delta|)."
    )
    lines.append("")
    headers = ["ablation", "mode", "N", "anchor Δ", "improved/9", "top3/9",
               "same Δ", "cross Δ", "max reg", "V6 before", "V6 after"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    rows: list[tuple[float, list[str]]] = []
    for name, best in best_map.items():
        agg = best["aggregate"]
        q_b = best["selected_scenarios_quality_before"]["avg_most_relevant_covered_in_topk"]
        q_a = best["selected_scenarios_quality_after"]["avg_most_relevant_covered_in_topk"]
        anchor_d = safe_num(agg["avg_anchor_rank_delta"])
        row = [
            f"`{name}`",
            f"`{best['mode']}`",
            str(best["requested_n"]),
            fmt_signed(anchor_d),
            str(agg.get("num_anchors_improved", "-")),
            str(agg.get("num_anchors_in_top3_after", "-")),
            fmt_signed(safe_num(agg["avg_same_scenario_rank_delta"])),
            fmt_signed(safe_num(agg["avg_cross_scenario_rank_delta"])),
            f"{agg.get('max_anchor_regression', 0):+.0f}",
            f"{q_b:.3f}",
            f"{q_a:.3f}",
        ]
        rows.append((anchor_d, row))

    rows.sort(key=lambda x: -x[0])
    for _, row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # -------- Table 2: per-context rank after --------
    lines.append("## 2. Per-context anchor rank after best condition")
    lines.append("")
    lines.append("Baseline row shows the frozen model's anchor rank of `phase2_gt_ro`. "
                 "**Bold** = improved, ~~strike~~ = regressed.")
    lines.append("")
    hdr = ["ablation"] + [c.replace("_", "&nbsp;") for c in ALL_CONTEXTS]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("| " + " | ".join(["---"] * len(hdr)) + " |")
    # baseline row
    baseline_cells = ["**BASELINE**"] + [str(baseline_rank[c]) for c in ALL_CONTEXTS]
    lines.append("| " + " | ".join(baseline_cells) + " |")
    for name, best in sorted(best_map.items(), key=lambda x: -safe_num(x[1]["aggregate"]["avg_anchor_rank_delta"])):
        after_map = context_rank_after(best)
        cells = [f"`{name}`"]
        for ctx in ALL_CONTEXTS:
            r = after_map.get(ctx)
            b = baseline_rank.get(ctx)
            if r is None or b is None:
                cells.append("-")
            elif r < b:
                cells.append(f"**{r}**")
            elif r > b:
                cells.append(f"~~{r}~~")
            else:
                cells.append(str(r))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # -------- Table 3: success criteria --------
    lines.append("## 3. Success criteria")
    lines.append("")
    lines.append("- **P1:** at least 3 of the 6 needs-promotion contexts land in rank ≤ 2")
    lines.append("- **P2:** OFFICE_LUNCH_OUT_B (plausible-tier diagnostic) reaches rank ≤ 3")
    lines.append("- **S1:** |avg cross-scenario rank delta| ≤ 0.2")
    lines.append("- **S2:** V6 most-relevant top-3 coverage not damaged (delta ≥ 0)")
    lines.append("")
    lines.append("| ablation | P1 | P2 | S1 | S2 | overall |")
    lines.append("| --- | :---: | :---: | :---: | :---: | :---: |")
    for name, best in sorted(best_map.items(), key=lambda x: -safe_num(x[1]["aggregate"]["avg_anchor_rank_delta"])):
        after_map = context_rank_after(best)
        agg = best["aggregate"]
        promo_hits = sum(1 for c in NEEDS_PROMO if after_map.get(c, 99) <= 2)
        p1 = promo_hits >= 3
        p2 = after_map.get("OFFICE_LUNCH_OUT_B", 99) <= 3
        cross_abs = abs(safe_num(agg["avg_cross_scenario_rank_delta"]))
        s1 = cross_abs <= 0.2
        q_b = best["selected_scenarios_quality_before"]["avg_most_relevant_covered_in_topk"]
        q_a = best["selected_scenarios_quality_after"]["avg_most_relevant_covered_in_topk"]
        s2 = q_a + 1e-9 >= q_b
        ok = "**PASS**" if all([p1, p2, s1, s2]) else "—"
        lines.append(
            f"| `{name}` | {'OK' if p1 else '✗'} ({promo_hits}/6) "
            f"| {'OK' if p2 else '✗'} "
            f"| {'OK' if s1 else '✗'} ({cross_abs:.3f}) "
            f"| {'OK' if s2 else '✗'} | {ok} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default="artifacts/phase2/ablations/ablations_comparison.md")
    parser.add_argument("--no-vanilla", action="store_true")
    args = parser.parse_args()

    ablations = load_ablations(Path(args.ablation_root), include_vanilla=not args.no_vanilla)
    if not ablations:
        print(f"No ablations loaded from {args.ablation_root}")
        return

    text = render_markdown(ablations)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Wrote {out} ({len(ablations)} ablations)")


if __name__ == "__main__":
    main()
