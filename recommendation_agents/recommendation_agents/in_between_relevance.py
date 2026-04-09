"""Parser for evaluation-only in-between scenario recommendation specs."""

from __future__ import annotations

from pathlib import Path
import re


def parse_in_between_scenarios_markdown(markdown_path: str | Path) -> dict[str, dict[str, tuple[str, ...]]]:
    text = Path(markdown_path).read_text()
    if "## " not in text:
        raise ValueError(f"No scenario sections found in {markdown_path}")

    parsed: dict[str, dict[str, tuple[str, ...]]] = {}
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    for section in sections[1:]:
        lines = section.splitlines()
        if not lines:
            continue
        header = lines[0].strip()
        if not header:
            continue
        scenario_id = header.split("—", 1)[0].strip()
        body = "\n".join(lines[1:])

        ro_match = re.search(r"^- \*\*Expanded RO ranking \(top 10\):\*\* (.+)$", body, flags=re.MULTILINE)
        app_match = re.search(r"^- \*\*Expanded App ranking \(top 5\):\*\* (.+)$", body, flags=re.MULTILINE)
        if ro_match is None or app_match is None:
            raise ValueError(f"Scenario {scenario_id!r} is missing Expanded RO/App ranking lines in {markdown_path}")

        ro_actions = tuple(re.findall(r"`([^`]+)`", ro_match.group(1)))
        app_actions = tuple(re.findall(r"`([^`]+)`", app_match.group(1)))
        if len(ro_actions) < 10:
            raise ValueError(f"Scenario {scenario_id!r} does not provide 10 R/O recommendations")
        if len(app_actions) < 5:
            raise ValueError(f"Scenario {scenario_id!r} does not provide 5 App recommendations")

        parsed[scenario_id] = {
            "ro_top10": ro_actions[:10],
            "app_top5": app_actions[:5],
        }

    if not parsed:
        raise ValueError(f"No in-between scenario rows were parsed from {markdown_path}")
    return parsed
