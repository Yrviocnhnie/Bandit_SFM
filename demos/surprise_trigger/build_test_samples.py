"""Build a tiered evaluation set for the surprise-trigger demo.

Tiers:
  ROUTINE   = on the user's routine envelope
  MILD      = routine except exactly one of {ps_phone, ps_light, ps_sound} is outside
  MODERATE  = routine except precondition or hour is outside
  STRONG    = HOME_EVENING substate (dark/lying/noisy) or weekend/holiday dayType
  EXTREME   = ARRIVE_OFFICE with office_late_overtime / office_rest_day / office_overtime
              (or HOME_EVENING with holiday+noisy substate etc.)
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from user_persona import ROUTINES

DEMO_DIR = Path(__file__).parent
INPUT = DEMO_DIR / "data" / "demo_two_scenarios.jsonl"
OUTPUT = DEMO_DIR / "data" / "test_samples.jsonl"
REPORT = DEMO_DIR / "data" / "test_samples_report.txt"

N_PER_TIER = 40
SEED = 17

EXTREME_STATES = {
    "ARRIVE_OFFICE": {"office_late_overtime", "office_rest_day", "office_overtime"},
    "HOME_EVENING": set(),
}
SUBSTATES = {
    "HOME_EVENING": {"home_evening_dark", "home_evening_lying", "home_evening_noisy"},
    "ARRIVE_OFFICE": set(),
}


def classify(row) -> str:
    sid = row["scenario_id"]
    spec = ROUTINES[sid]
    f = row["features"]
    allowed = spec.allowed_cat

    if spec.matches(f):
        return "ROUTINE"

    if f.get("state_current") in EXTREME_STATES.get(sid, set()):
        return "EXTREME"

    if f.get("state_current") in SUBSTATES.get(sid, set()):
        return "STRONG"

    if f.get("ps_dayType") in {"weekend", "holiday"}:
        return "STRONG"

    try:
        hour = int(f.get("hour", -1))
    except (TypeError, ValueError):
        hour = -1
    lo, hi = spec.hour_range

    precond_off = f.get("precondition") not in allowed["precondition"]
    hour_off = not (lo <= hour <= hi)
    if precond_off or hour_off:
        return "MODERATE"

    phone_off = f.get("ps_phone") not in allowed["ps_phone"]
    light_off = f.get("ps_light") not in allowed["ps_light"]
    sound_off = f.get("ps_sound") not in allowed["ps_sound"]
    off_axes = int(phone_off) + int(light_off) + int(sound_off)
    if off_axes == 1:
        return "MILD"
    if off_axes >= 2:
        return "MODERATE"

    return "MILD"


def main():
    random.seed(SEED)
    buckets = {
        "ARRIVE_OFFICE": {t: [] for t in ("ROUTINE", "MILD", "MODERATE", "STRONG", "EXTREME")},
        "HOME_EVENING": {t: [] for t in ("ROUTINE", "MILD", "MODERATE", "STRONG", "EXTREME")},
    }
    with INPUT.open() as f:
        for line in f:
            r = json.loads(line)
            sid = r["scenario_id"]
            tier = classify(r)
            if sid in buckets and tier in buckets[sid]:
                buckets[sid][tier].append(r)

    # also save the raw bucket counts for the report
    report_lines = ["Tier counts per scenario (full dataset):", ""]
    for sid, tiers in buckets.items():
        report_lines.append(f"{sid}:")
        for tier, rows in tiers.items():
            report_lines.append(f"  {tier}: {len(rows)}")
        report_lines.append("")

    selected = []
    for sid, tiers in buckets.items():
        for tier, rows in tiers.items():
            random.shuffle(rows)
            take = rows[:N_PER_TIER]
            for r in take:
                r2 = dict(r)
                r2["tier"] = tier
                selected.append(r2)

    report_lines.append(f"Selected {len(selected)} rows (up to {N_PER_TIER} per scenario x tier).")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")
    REPORT.write_text("\n".join(report_lines) + "\n")
    print("wrote", OUTPUT, "and", REPORT)
    print("\n".join(report_lines))


SEED = 17
N_PER_TIER_DEFAULT = 40


if __name__ == "__main__":
    main()
