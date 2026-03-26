"""Shared parser for the V6 scenario relevance catalog."""

from __future__ import annotations

from pathlib import Path
import re


def parse_v6_relevance_markdown(markdown_path: str | Path) -> dict[str, dict[str, dict[str, tuple[str, ...]]]]:
    text = Path(markdown_path).read_text()
    if "## Scenario Defaults" not in text:
        raise ValueError(f"Could not find '## Scenario Defaults' in {markdown_path}")

    section = text.split("## Scenario Defaults", 1)[1]
    parsed: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {"ro": {}, "app": {}}

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line.startswith("| `"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 8:
            continue

        scenario_id = parts[0].strip("`")
        ro_most = tuple(re.findall(r"`([^`]+)`", parts[2]))
        ro_plausible = tuple(re.findall(r"`([^`]+)`", parts[3]))
        ro_irrelevant = tuple(re.findall(r"`([^`]+)`", parts[4]))
        app_most = tuple(re.findall(r"`([^`]+)`", parts[5]))
        app_plausible = tuple(re.findall(r"`([^`]+)`", parts[6]))
        app_irrelevant = tuple(re.findall(r"`([^`]+)`", parts[7]))

        if len(ro_most) < 3 or len(ro_plausible) < 3 or len(ro_irrelevant) < 2:
            raise ValueError(f"Scenario {scenario_id!r} does not satisfy the required R/O 3+3+2 structure")
        if len(app_most) < 3 or len(app_plausible) < 3 or len(app_irrelevant) < 2:
            raise ValueError(f"Scenario {scenario_id!r} does not satisfy the required App 3+3+2 structure")

        parsed["ro"][scenario_id] = {
            "most_relevant_3": ro_most[:3],
            "other_plausible_3": ro_plausible[:3],
            "irrelevant_2": ro_irrelevant[:2],
        }
        parsed["app"][scenario_id] = {
            "most_relevant_3": app_most[:3],
            "other_plausible_3": app_plausible[:3],
            "irrelevant_2": app_irrelevant[:2],
        }

    if not parsed["ro"] or not parsed["app"]:
        raise ValueError(f"No scenario relevance rows were parsed from {markdown_path}")
    return parsed
