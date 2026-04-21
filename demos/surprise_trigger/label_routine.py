"""Read the two-scenario JSONL, add `is_routine` label using the Alex persona,
and write a labeled JSONL + a small report of counts.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from user_persona import ROUTINES

INPUT = Path(__file__).parent / "data" / "demo_two_scenarios.jsonl"
OUTPUT = Path(__file__).parent / "data" / "demo_two_scenarios_labeled.jsonl"
REPORT = Path(__file__).parent / "data" / "labeling_report.txt"


def main() -> None:
    counts: Counter = Counter()
    with INPUT.open() as fin, OUTPUT.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            sid = row["scenario_id"]
            spec = ROUTINES.get(sid)
            if spec is None:
                continue
            is_routine = spec.matches(row["features"])
            row["is_routine"] = bool(is_routine)
            fout.write(json.dumps(row) + "\n")
            counts[(sid, is_routine)] += 1

    lines = ["Routine vs non-routine counts (Alex persona):", ""]
    for sid in ("ARRIVE_OFFICE", "HOME_EVENING"):
        r = counts[(sid, True)]
        n = counts[(sid, False)]
        total = r + n
        pct = 100 * r / total if total else 0.0
        lines.append(f"  {sid}: routine={r} non-routine={n} total={total} ({pct:.1f}% routine)")
    out = "\n".join(lines) + "\n"
    REPORT = Path(__file__).parent / "data" / "labeling_report.txt"
    REPORT.write_text(out)
    print(out)


if __name__ == "__main__":
    main()
