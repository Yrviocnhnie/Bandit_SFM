"""Extract ARRIVE_OFFICE and HOME_EVENING rows from the full 65-scenario JSONL."""
from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path("/home/mohan/Bandit_SFM/data/bandit_train_data_unique_samples/bandit_v0_65_scenaroios_unique_support_to_2k_samples.jsonl")
DEST = Path(__file__).parent / "data" / "demo_two_scenarios.jsonl"
SCENARIOS = {"ARRIVE_OFFICE", "HOME_EVENING"}


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    counts: dict[str, int] = {s: 0 for s in SCENARIOS}
    with SOURCE.open() as src, DEST.open("w") as dst:
        for line in src:
            row = json.loads(line)
            if row["scenario_id"] in SCENARIOS:
                dst.write(line)
                kept += 1
                counts[row["scenario_id"]] += 1
    print(f"Kept {kept} rows -> {DEST}")
    for sid, c in counts.items():
        print(f"  {sid}: {c}")


if __name__ == "__main__":
    main()
