"""Parse the current shared action catalog into training metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

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


def _load_global_catalog_payload(markdown_path: str | Path) -> dict[str, Any]:
    markdown_file = Path(markdown_path)
    sibling_json = markdown_file.with_name('global_action_space.json')
    if sibling_json.exists():
        return json.loads(sibling_json.read_text())
    raise ValueError(
        f'Expected sibling global_action_space.json next to {markdown_file}, but none was found'
    )


def build_ro_metadata_from_catalog_markdown(
    markdown_path: str | Path,
    output_metadata_path: str | Path,
) -> CatalogSummary:
    payload = _load_global_catalog_payload(markdown_path)
    global_actions = payload.get('globalActions', [])
    scenario_defaults = payload.get('scenarioDefaults', [])
    if not global_actions or not scenario_defaults:
        raise ValueError('global_action_space.json must contain globalActions and scenarioDefaults')

    actions = []
    global_action_ids: list[str] = []
    for row in global_actions:
        action_id = str(row['actionId'])
        global_action_ids.append(action_id)
        actions.append(
            {
                'action_id': action_id,
                'raw_action_id': action_id,
                'display_name': row.get('nameEn', action_id),
                'action_type': row.get('actionType'),
            }
        )

    scenario_rankings = []
    per_scenario_counts: list[int] = []
    for row in scenario_defaults:
        default_action_ids = [str(value) for value in row.get('defaultActionIds', [])]
        if not default_action_ids:
            raise ValueError(f"Scenario {row.get('scenarioId')!r} is missing defaultActionIds")
        unknown_actions = [action_id for action_id in default_action_ids if action_id not in global_action_ids]
        if unknown_actions:
            raise ValueError(
                f"Scenario {row.get('scenarioId')!r} references unknown actions: {unknown_actions}"
            )
        scenario_rankings.append(
            {
                'scenario_id': str(row['scenarioId']),
                'default_action_ids': default_action_ids,
            }
        )
        per_scenario_counts.append(len(default_action_ids))

    output_path = Path(output_metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                'schema_version': 'ro-global-catalog',
                'source': str(markdown_path),
                'global_action_ids': global_action_ids,
                'actions': actions,
                'scenario_default_rankings': scenario_rankings,
            },
            indent=2,
            sort_keys=True,
        )
    )

    return CatalogSummary(
        total_scenarios=len(scenario_rankings),
        total_actions=len(global_action_ids),
        min_actions_per_scenario=min(per_scenario_counts),
        max_actions_per_scenario=max(per_scenario_counts),
    )


def build_app_metadata_from_catalog_markdown(
    markdown_path: str | Path,
    output_metadata_path: str | Path,
) -> AppCatalogSummary:
    payload = _load_global_catalog_payload(markdown_path)
    scenario_defaults = payload.get('scenarioDefaults', [])
    if not scenario_defaults:
        raise ValueError('global_action_space.json must contain scenarioDefaults')

    scenario_rankings = []
    for row in scenario_defaults:
        default_app_categories = [str(value) for value in row.get('defaultAppCategories', [])]
        if not default_app_categories:
            raise ValueError(f"Scenario {row.get('scenarioId')!r} is missing defaultAppCategories")
        unknown_categories = [value for value in default_app_categories if value not in APP_CATEGORIES]
        if unknown_categories:
            raise ValueError(
                f"Scenario {row.get('scenarioId')!r} references unknown app categories: {unknown_categories}"
            )
        scenario_rankings.append(
            {
                'scenario_id': str(row['scenarioId']),
                'default_action_ids': default_app_categories,
            }
        )

    output_path = Path(output_metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                'schema_version': 'app-global-catalog',
                'source': str(markdown_path),
                'global_action_ids': list(APP_CATEGORIES),
                'actions': [
                    {'action_id': category, 'raw_action_id': category, 'display_name': category}
                    for category in APP_CATEGORIES
                ],
                'scenario_default_rankings': scenario_rankings,
            },
            indent=2,
            sort_keys=True,
        )
    )

    return AppCatalogSummary(
        total_scenarios=len(scenario_rankings),
        total_actions=len(APP_CATEGORIES),
    )
