"""Parse the current scenario/action catalog document into training metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from recommendation_agents.taxonomies import APP_CATEGORIES


@dataclass(frozen=True)
class CatalogSummary:
    total_scenarios: int
    total_actions: int
    min_actions_per_scenario: int
    max_actions_per_scenario: int


@dataclass(frozen=True)
class AppCatalogSummary:
    total_scenarios: int
    total_actions: int


def _strip_code_ticks(value: str) -> str:
    text = value.strip()
    if text.startswith("`") and text.endswith("`"):
        return text[1:-1]
    return text


def build_ro_metadata_from_catalog_markdown(
    markdown_path: str | Path,
    output_metadata_path: str | Path,
) -> CatalogSummary:
    lines = Path(markdown_path).read_text().splitlines()

    in_summary = False
    scenarios = []
    action_type_by_id: dict[str, str] = {}
    per_scenario_counts: list[int] = []

    for line in lines:
        if line.startswith("## Scenario Summary"):
            in_summary = True
            continue
        if in_summary and line.startswith("## R&O Actions For Each Scenario"):
            break
        if not in_summary:
            continue
        if not line.startswith("|"):
            continue
        if "scenarioId" in line or "---" in line:
            continue

        columns = [part.strip() for part in line.split("|")[1:-1]]
        if len(columns) != 8:
            raise ValueError(f"Unexpected Scenario Summary row shape: {line}")

        scenario_id = _strip_code_ticks(columns[0])
        action_rows = json.loads(_strip_code_ticks(columns[2]))
        default_row = json.loads(_strip_code_ticks(columns[3]))
        action_ids = [str(item["id"]) for item in action_rows]
        default_action_id = str(default_row["id"])
        if default_action_id not in action_ids:
            raise ValueError(f"Default action {default_action_id!r} is not in actionIDs for {scenario_id!r}")

        scenario_actions = []
        for item in action_rows:
            raw_action_id = str(item["id"])
            arm_id = f"{scenario_id}::{raw_action_id}"
            scenario_actions.append(
                {
                    "arm_id": arm_id,
                    "action_id": raw_action_id,
                    "action_type": str(item["type"]),
                    "display_name": raw_action_id,
                }
            )

        scenarios.append(
            {
                "scenario_id": scenario_id,
                "default_action_id": default_action_id,
                "default_arm_id": f"{scenario_id}::{default_action_id}",
                "actions": scenario_actions,
            }
        )
        per_scenario_counts.append(len(action_ids))
        for item in action_rows:
            arm_id = f"{scenario_id}::{item['id']}"
            action_type_by_id[arm_id] = str(item["type"])

    if not scenarios:
        raise ValueError("Failed to parse any scenarios from the Scenario Summary table")

    actions = [
        {
            "action_id": action_id,
            "raw_action_id": action_id.split("::", 1)[1],
            "display_name": action_id.split("::", 1)[1],
            "action_type": action_type_by_id[action_id],
        }
        for action_id in sorted(action_type_by_id)
    ]
    payload = {
        "schema_version": "ro-catalog-doc",
        "source": str(markdown_path),
        "scenarios": scenarios,
        "actions": actions,
    }

    output_path = Path(output_metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return CatalogSummary(
        total_scenarios=len(scenarios),
        total_actions=len(actions),
        min_actions_per_scenario=min(per_scenario_counts),
        max_actions_per_scenario=max(per_scenario_counts),
    )


def build_app_metadata_from_catalog_markdown(
    markdown_path: str | Path,
    output_metadata_path: str | Path,
) -> AppCatalogSummary:
    lines = Path(markdown_path).read_text().splitlines()

    in_default_a = False
    scenarios = []
    for line in lines:
        if line.startswith("## Default A For Each Scenario"):
            in_default_a = True
            continue
        if not in_default_a:
            continue
        if not line.startswith("|"):
            continue
        if "scenarioId" in line or "---" in line:
            continue

        columns = [part.strip() for part in line.split("|")[1:-1]]
        if len(columns) != 9:
            raise ValueError(f"Unexpected Default A row shape: {line}")

        scenario_id = _strip_code_ticks(columns[0])
        payload_zh = json.loads(columns[7])
        app_category = str(payload_zh["app_category"])
        if app_category not in APP_CATEGORIES:
            raise ValueError(f"Unknown app_category {app_category!r} for scenario {scenario_id!r}")

        scenarios.append(
            {
                "scenario_id": scenario_id,
                "default_action_id": app_category,
                "action_ids": list(APP_CATEGORIES),
            }
        )

    if not scenarios:
        raise ValueError("Failed to parse any scenarios from the Default A table")

    payload = {
        "schema_version": "app-catalog-doc",
        "source": str(markdown_path),
        "scenarios": scenarios,
        "actions": [{"action_id": category, "display_name": category} for category in APP_CATEGORIES],
    }

    output_path = Path(output_metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return AppCatalogSummary(
        total_scenarios=len(scenarios),
        total_actions=len(APP_CATEGORIES),
    )
