"""Converters for colleague-provided synthetic sequence data."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from recommendation_agents.feature_space import V0FeatureSpace


RAW_TO_V0_CONTEXT_FIELDS = (
    "state_current",
    "state_duration_sec",
    "ps_time",
    "hour",
    "cal_hasUpcoming",
    "ps_dayType",
    "ps_motion",
    "wifiLost",
    "wifiLostCategory",
    "cal_eventCount",
    "cal_inMeeting",
    "cal_nextLocation",
    "ps_sound",
    "sms_delivery_pending",
    "sms_train_pending",
    "sms_flight_pending",
    "sms_hotel_pending",
    "sms_movie_pending",
    "sms_hospital_pending",
    "sms_ride_pending",
    "timestep",
    "ps_location",
    "ps_phone",
    "batteryLevel",
    "isCharging",
    "networkType",
    "transportMode",
    "activityState",
    "activityDuration",
    "user_id_hash_bucket",
    "age_bucket",
    "sex",
    "has_kids",
)


@dataclass(frozen=True)
class RawConversionSummary:
    input_rows: int
    kept_rows: int
    skipped_none_rows: int
    unique_scenarios: int
    unique_actions: int


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
    return rows


def _build_context(row: dict[str, Any]) -> dict[str, Any]:
    feature_payload = row.get("features")
    source = feature_payload if isinstance(feature_payload, dict) else row
    return {field: source.get(field) for field in RAW_TO_V0_CONTEXT_FIELDS}


def _scenario_id(row: dict[str, Any]) -> str:
    feature_payload = row.get("features")
    if "scenarioId" in row:
        return str(row["scenarioId"])
    if "scenario_id" in row:
        return str(row["scenario_id"])
    if isinstance(feature_payload, dict) and "scenarioId" in feature_payload:
        return str(feature_payload["scenarioId"])
    raise KeyError("Raw row is missing scenario id; expected scenarioId, scenario_id, or features.scenarioId")


def _event_id(row: dict[str, Any], scenario_id: str) -> str:
    episode_id = row.get("episode_id", scenario_id)
    offset = row.get("scenario_elapsed_sec", row.get("t_in_scenario_sec", 0))
    return f"{episode_id}:{offset}"


def _raw_label(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return "NONE"


def convert_raw_sequence_to_v0(
    input_path: str | Path,
    output_samples_path: str | Path,
    output_metadata_path: str | Path | None = None,
    reward: float = 1.0,
) -> RawConversionSummary:
    raw_rows = _load_jsonl(input_path)
    kept_rows: list[dict[str, Any]] = []
    scenario_action_counts: dict[str, Counter[str]] = defaultdict(Counter)

    feature_space = V0FeatureSpace()
    for row in raw_rows:
        action_id = _raw_label(row, "gt_ro_action_id", "gt_ro")
        if action_id == "NONE":
            continue
        if "emit_recommendation" in row and int(row["emit_recommendation"]) != 1:
            raise ValueError(
                "Found a non-NONE gt_ro_action_id on a row where emit_recommendation != 1; "
                "this would violate the one-call-per-emission assumption"
            )
        scenario_id = _scenario_id(row)
        context = _build_context(row)
        feature_space.encode(scenario_id, context)

        kept_rows.append(
            {
                "event_id": _event_id(row, scenario_id),
                "scenario_id": scenario_id,
                "context": context,
                "selected_action": action_id,
                "reward": float(reward),
                "propensity": 1.0,
                "source_episode_id": row.get("episode_id"),
                "source_reason": row.get("gt_reason"),
            }
        )
        scenario_action_counts[scenario_id][action_id] += 1

    if not kept_rows:
        raise ValueError("No trainable rows found after filtering out gt_ro_action_id == NONE")

    output_samples = Path(output_samples_path)
    output_samples.parent.mkdir(parents=True, exist_ok=True)
    with output_samples.open("w") as handle:
        for event in kept_rows:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")

    if output_metadata_path is not None:
        actions = sorted({event["selected_action"] for event in kept_rows})
        metadata = {
            "schema_version": "v0",
            "scenarios": [
                {
                    "scenario_id": scenario_id,
                    "default_action_id": counts.most_common(1)[0][0],
                    "action_ids": sorted(counts),
                }
                for scenario_id, counts in sorted(scenario_action_counts.items())
            ],
            "actions": [{"action_id": action_id, "display_name": action_id} for action_id in actions],
        }
        output_metadata = Path(output_metadata_path)
        output_metadata.parent.mkdir(parents=True, exist_ok=True)
        output_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return RawConversionSummary(
        input_rows=len(raw_rows),
        kept_rows=len(kept_rows),
        skipped_none_rows=len(raw_rows) - len(kept_rows),
        unique_scenarios=len(scenario_action_counts),
        unique_actions=len({event["selected_action"] for event in kept_rows}),
    )


def convert_raw_sequence_to_v0_app(
    input_path: str | Path,
    output_samples_path: str | Path,
    output_metadata_path: str | Path | None = None,
    reward: float = 1.0,
) -> RawConversionSummary:
    raw_rows = _load_jsonl(input_path)
    kept_rows: list[dict[str, Any]] = []
    scenario_action_counts: dict[str, Counter[str]] = defaultdict(Counter)

    feature_space = V0FeatureSpace()
    for row in raw_rows:
        app_category = _raw_label(row, "gt_app_category", "gt_app")
        if app_category == "NONE":
            continue
        if "emit_recommendation" in row and int(row["emit_recommendation"]) != 1:
            raise ValueError(
                "Found a non-NONE gt_app_category on a row where emit_recommendation != 1; "
                "this would violate the one-call-per-emission assumption"
            )
        scenario_id = _scenario_id(row)
        context = _build_context(row)
        feature_space.encode(scenario_id, context)

        kept_rows.append(
            {
                "event_id": _event_id(row, scenario_id),
                "scenario_id": scenario_id,
                "context": context,
                "selected_action": app_category,
                "reward": float(reward),
                "propensity": 1.0,
                "source_episode_id": row.get("episode_id"),
                "source_reason": row.get("gt_reason"),
                "source_app_action_id": row.get("gt_app_action_id"),
            }
        )
        scenario_action_counts[scenario_id][app_category] += 1

    if not kept_rows:
        raise ValueError("No trainable app rows found after filtering out gt_app_category == NONE")

    output_samples = Path(output_samples_path)
    output_samples.parent.mkdir(parents=True, exist_ok=True)
    with output_samples.open("w") as handle:
        for event in kept_rows:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")

    if output_metadata_path is not None:
        actions = sorted({event["selected_action"] for event in kept_rows})
        metadata = {
            "schema_version": "v0-app",
            "scenarios": [
                {
                    "scenario_id": scenario_id,
                    "default_action_id": counts.most_common(1)[0][0],
                    "action_ids": actions,
                }
                for scenario_id, counts in sorted(scenario_action_counts.items())
            ],
            "actions": [{"action_id": action_id, "display_name": action_id} for action_id in actions],
        }
        output_metadata = Path(output_metadata_path)
        output_metadata.parent.mkdir(parents=True, exist_ok=True)
        output_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return RawConversionSummary(
        input_rows=len(raw_rows),
        kept_rows=len(kept_rows),
        skipped_none_rows=len(raw_rows) - len(kept_rows),
        unique_scenarios=len(scenario_action_counts),
        unique_actions=len({event["selected_action"] for event in kept_rows}),
    )
