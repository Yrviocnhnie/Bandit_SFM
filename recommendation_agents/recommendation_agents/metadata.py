"""Scenario and action metadata contract for the V0 model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from recommendation_agents.taxonomies import CANONICAL_SCENARIO_IDS


@dataclass(frozen=True)
class ActionMetadata:
    action_id: str
    raw_action_id: str | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class ScenarioMetadata:
    scenario_id: str
    default_action_id: str
    default_arm_id: str
    action_ids: tuple[str, ...]
    raw_action_ids: tuple[str, ...]
    arm_to_raw: dict[str, str]
    raw_to_arms: dict[str, tuple[str, ...]]

    def resolve_action_token(self, token: str) -> str:
        if token in self.arm_to_raw:
            return token
        matching_arms = self.raw_to_arms.get(token)
        if not matching_arms:
            raise KeyError(f"Action {token!r} is not valid for scenario {self.scenario_id!r}")
        if len(matching_arms) > 1:
            raise ValueError(
                f"Action token {token!r} is ambiguous in scenario {self.scenario_id!r}; "
                "use the full scenario-scoped arm id"
            )
        return matching_arms[0]

    def raw_action_id(self, arm_id: str) -> str:
        try:
            return self.arm_to_raw[arm_id]
        except KeyError as exc:
            raise KeyError(f"Arm {arm_id!r} is not valid for scenario {self.scenario_id!r}") from exc


@dataclass(frozen=True)
class BanditMetadata:
    schema_version: str
    scenarios: dict[str, ScenarioMetadata]
    actions: dict[str, ActionMetadata]

    @classmethod
    def load(cls, path: str | Path) -> "BanditMetadata":
        payload = json.loads(Path(path).read_text())
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BanditMetadata":
        schema_version = str(payload.get("schema_version", "v0"))
        scenario_rows = payload.get("scenarios")
        if not isinstance(scenario_rows, list) or not scenario_rows:
            raise ValueError("metadata.scenarios must be a non-empty list")

        scenarios: dict[str, ScenarioMetadata] = {}
        seen_actions: dict[str, ActionMetadata] = {}
        canonical_ids = set(CANONICAL_SCENARIO_IDS)

        for row in scenario_rows:
            scenario_id = str(row["scenario_id"])
            if scenario_id not in canonical_ids:
                raise ValueError(f"Unknown scenario_id {scenario_id!r}; use canonical IDs from the feature spec")
            action_rows = row.get("actions")
            if action_rows is not None:
                if not isinstance(action_rows, list) or not action_rows:
                    raise ValueError(f"Scenario {scenario_id!r} actions must be a non-empty list")
                arm_to_raw: dict[str, str] = {}
                raw_to_arms_lists: dict[str, list[str]] = {}
                action_ids_list: list[str] = []
                raw_action_ids_list: list[str] = []
                for action_row in action_rows:
                    arm_id = str(action_row["arm_id"])
                    raw_action_id = str(action_row.get("action_id", arm_id))
                    action_ids_list.append(arm_id)
                    raw_action_ids_list.append(raw_action_id)
                    arm_to_raw[arm_id] = raw_action_id
                    raw_to_arms_lists.setdefault(raw_action_id, []).append(arm_id)
                    seen_actions.setdefault(
                        arm_id,
                        ActionMetadata(
                            action_id=arm_id,
                            raw_action_id=raw_action_id,
                            display_name=action_row.get("display_name", raw_action_id),
                        ),
                    )
                action_ids = tuple(dict.fromkeys(action_ids_list))
                raw_action_ids = tuple(dict.fromkeys(raw_action_ids_list))
                raw_to_arms = {key: tuple(value) for key, value in raw_to_arms_lists.items()}
                default_action_id = str(row.get("default_action_id", row.get("default_arm_id")))
                default_arm_id = str(row.get("default_arm_id", default_action_id))
                if default_arm_id not in arm_to_raw:
                    if default_action_id in raw_to_arms and len(raw_to_arms[default_action_id]) == 1:
                        default_arm_id = raw_to_arms[default_action_id][0]
                    else:
                        raise ValueError(
                            f"Scenario {scenario_id!r} default arm {default_arm_id!r} "
                            "must be present in actions"
                        )
                default_action_id = arm_to_raw[default_arm_id]
            else:
                action_ids = tuple(dict.fromkeys(str(value) for value in row["action_ids"]))
                if not action_ids:
                    raise ValueError(f"Scenario {scenario_id!r} must declare at least one action")
                default_action_id = str(row["default_action_id"])
                if default_action_id not in action_ids:
                    raise ValueError(
                        f"Scenario {scenario_id!r} default_action_id {default_action_id!r} "
                        "must be present in action_ids"
                    )
                default_arm_id = default_action_id
                raw_action_ids = action_ids
                arm_to_raw = {action_id: action_id for action_id in action_ids}
                raw_to_arms = {action_id: (action_id,) for action_id in action_ids}
                for action_id in action_ids:
                    seen_actions.setdefault(action_id, ActionMetadata(action_id=action_id, raw_action_id=action_id))
            if scenario_id in scenarios:
                raise ValueError(f"Duplicate scenario_id {scenario_id!r}")
            scenarios[scenario_id] = ScenarioMetadata(
                scenario_id=scenario_id,
                default_action_id=default_action_id,
                default_arm_id=default_arm_id,
                action_ids=action_ids,
                raw_action_ids=raw_action_ids,
                arm_to_raw=arm_to_raw,
                raw_to_arms=raw_to_arms,
            )

        for row in payload.get("actions", []):
            action_id = str(row["action_id"])
            seen_actions[action_id] = ActionMetadata(
                action_id=action_id,
                raw_action_id=row.get("raw_action_id"),
                display_name=row.get("display_name"),
            )

        return cls(schema_version=schema_version, scenarios=scenarios, actions=seen_actions)

    def scenario(self, scenario_id: str) -> ScenarioMetadata:
        try:
            return self.scenarios[scenario_id]
        except KeyError as exc:
            raise KeyError(f"Scenario {scenario_id!r} is not present in metadata") from exc

    def all_action_ids(self) -> list[str]:
        return list(self.actions)
