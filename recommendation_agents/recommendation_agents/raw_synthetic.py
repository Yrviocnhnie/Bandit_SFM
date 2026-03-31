"""Converters for colleague-provided synthetic sequence data."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from recommendation_agents.feature_space import V0FeatureSpace
from recommendation_agents.v6_relevance import parse_v6_relevance_markdown


RAW_TO_V0_CONTEXT_FIELDS = (
    "state_current",
    "precondition",
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


@dataclass(frozen=True)
class ExpandedRawConversionSummary:
    input_rows: int
    kept_rows: int
    skipped_none_rows: int
    unique_scenarios: int
    unique_actions: int
    emitted_samples: int
    tier_counts: dict[str, int]


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
    context = {field: source.get(field) for field in RAW_TO_V0_CONTEXT_FIELDS}
    if context.get("precondition") is None:
        for key in ("state_prev", "previous_state_current", "prev_state_current", "state_previous"):
            if key in source:
                context["precondition"] = source.get(key)
                break
    return context


def _scenario_id(row: dict[str, Any]) -> str | None:
    feature_payload = row.get("features")
    if "scenarioId" in row:
        return str(row["scenarioId"])
    if "scenario_id" in row:
        return str(row["scenario_id"])
    if isinstance(feature_payload, dict) and "scenarioId" in feature_payload:
        return str(feature_payload["scenarioId"])
    return None


def _event_id(row: dict[str, Any], scenario_id: str | None) -> str:
    episode_id = row.get("episode_id", scenario_id or "event")
    offset = row.get("scenario_elapsed_sec", row.get("t_in_scenario_sec", 0))
    return f"{episode_id}:{offset}"


def _raw_label(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return "NONE"


def _write_global_inferred_metadata(
    output_metadata_path: str | Path,
    schema_version: str,
    actions: list[str],
    scenario_action_counts: dict[str, Counter[str]],
) -> None:
    metadata = {
        "schema_version": schema_version,
        "global_action_ids": actions,
        "actions": [{"action_id": action_id, "display_name": action_id} for action_id in actions],
        "scenario_default_rankings": [
            {
                "scenario_id": scenario_id,
                "default_action_ids": [action_id for action_id, _ in counts.most_common()],
            }
            for scenario_id, counts in sorted(scenario_action_counts.items())
        ],
    }
    output_metadata = Path(output_metadata_path)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _write_events(output_samples_path: str | Path, events: list[dict[str, Any]]) -> None:
    output_samples = Path(output_samples_path)
    output_samples.parent.mkdir(parents=True, exist_ok=True)
    with output_samples.open("w") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")


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
        feature_space.encode(context)

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
        if scenario_id is not None:
            scenario_action_counts[scenario_id][action_id] += 1

    if not kept_rows:
        raise ValueError("No trainable rows found after filtering out gt_ro_action_id == NONE")

    _write_events(output_samples_path, kept_rows)

    if output_metadata_path is not None:
        actions = sorted({event["selected_action"] for event in kept_rows})
        _write_global_inferred_metadata(
            output_metadata_path=output_metadata_path,
            schema_version="v0-ro-inferred-global",
            actions=actions,
            scenario_action_counts=scenario_action_counts,
        )

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
        feature_space.encode(context)

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
        if scenario_id is not None:
            scenario_action_counts[scenario_id][app_category] += 1

    if not kept_rows:
        raise ValueError("No trainable app rows found after filtering out gt_app_category == NONE")

    _write_events(output_samples_path, kept_rows)

    if output_metadata_path is not None:
        actions = sorted({event["selected_action"] for event in kept_rows})
        _write_global_inferred_metadata(
            output_metadata_path=output_metadata_path,
            schema_version="v0-app-inferred-global",
            actions=actions,
            scenario_action_counts=scenario_action_counts,
        )

    return RawConversionSummary(
        input_rows=len(raw_rows),
        kept_rows=len(kept_rows),
        skipped_none_rows=len(raw_rows) - len(kept_rows),
        unique_scenarios=len(scenario_action_counts),
        unique_actions=len({event["selected_action"] for event in kept_rows}),
    )


def _expand_raw_sequence_with_v6_relevance(
    input_path: str | Path,
    output_samples_path: str | Path,
    relevance_markdown: str | Path,
    label_namespace: str,
    most_relevant_reward: float,
    plausible_reward: float,
    irrelevant_reward: float,
    most_relevant_repeat: int,
    plausible_repeat: int,
    irrelevant_repeat: int,
    all_action_ids: list[str] | tuple[str, ...] | None = None,
    other_zero_mode: str = "none",
) -> ExpandedRawConversionSummary:
    if label_namespace not in {"ro", "app"}:
        raise ValueError("label_namespace must be 'ro' or 'app'")
    if most_relevant_repeat < 0:
        raise ValueError("most_relevant_repeat must be non-negative")
    if plausible_repeat < 0:
        raise ValueError("plausible_repeat must be non-negative")
    if irrelevant_repeat < 0:
        raise ValueError("irrelevant_repeat must be non-negative")
    if other_zero_mode not in {"none", "exclude-most-plausible", "exclude-most-only", "exclude-all-labeled"}:
        raise ValueError(
            "other_zero_mode must be one of: none, exclude-most-plausible, exclude-most-only, exclude-all-labeled"
        )
    if other_zero_mode != "none" and not all_action_ids:
        raise ValueError("all_action_ids must be provided when other_zero_mode is not 'none'")

    raw_rows = _load_jsonl(input_path)
    relevance_catalog = parse_v6_relevance_markdown(relevance_markdown)[label_namespace]
    feature_space = V0FeatureSpace()
    kept_input_rows = 0
    tier_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    action_ids: set[str] = set()
    emitted_events: list[dict[str, Any]] = []

    for row in raw_rows:
        if label_namespace == "ro":
            label = _raw_label(row, "gt_ro_action_id", "gt_ro")
        else:
            label = _raw_label(row, "gt_app_category", "gt_app")
        if label == "NONE":
            continue
        if "emit_recommendation" in row and int(row["emit_recommendation"]) != 1:
            raise ValueError(
                "Found a non-NONE label on a row where emit_recommendation != 1; "
                "this would violate the one-call-per-emission assumption"
            )
        scenario_id = _scenario_id(row)
        if scenario_id is None:
            raise ValueError("Expanded V6 training requires scenario_id on every raw row")
        if scenario_id not in relevance_catalog:
            raise ValueError(f"Scenario {scenario_id!r} is not present in the V6 relevance catalog")
        context = _build_context(row)
        context_vector = feature_space.encode(context).tolist()
        base_event_id = _event_id(row, scenario_id)
        source_episode_id = row.get("episode_id")
        most_relevant_ids = tuple(relevance_catalog[scenario_id]["most_relevant_3"])
        plausible_ids = tuple(relevance_catalog[scenario_id]["other_plausible_3"])
        irrelevant_ids = tuple(relevance_catalog[scenario_id]["irrelevant_2"])
        if label_namespace == "ro":
            irrelevant_ids = irrelevant_ids + tuple(relevance_catalog[scenario_id].get("extra_hard_negative_ro", ()))
        tiers = [
            (
                "most_relevant",
                most_relevant_ids,
                float(most_relevant_reward),
                most_relevant_repeat,
            ),
            (
                "plausible",
                plausible_ids,
                float(plausible_reward),
                plausible_repeat,
            ),
            (
                "irrelevant",
                irrelevant_ids,
                float(irrelevant_reward),
                irrelevant_repeat,
            ),
        ]
        if other_zero_mode != "none":
            excluded = set(most_relevant_ids)
            if other_zero_mode == "exclude-most-plausible":
                excluded.update(plausible_ids)
            elif other_zero_mode == "exclude-all-labeled":
                excluded.update(plausible_ids)
                excluded.update(irrelevant_ids)
            extra_zero_actions = tuple(action_id for action_id in all_action_ids if action_id not in excluded)
            tiers.append(
                (
                    "other_zero",
                    extra_zero_actions,
                    0.0,
                    1,
                )
            )
        kept_input_rows += 1
        scenario_counts[scenario_id] += 1
        for tier_name, action_ids_for_tier, reward, repeat_count in tiers:
            for position, action_id in enumerate(action_ids_for_tier, start=1):
                for repeat_index in range(1, repeat_count + 1):
                    tier_counts[tier_name] += 1
                    action_ids.add(action_id)
                    emitted_events.append(
                        {
                            "event_id": f"{base_event_id}::{tier_name}:{position}:r{repeat_index}",
                            "scenario_id": scenario_id,
                            "context": context,
                            "context_vector": context_vector,
                            "selected_action": action_id,
                            "reward": reward,
                            "propensity": 1.0,
                            "source_episode_id": source_episode_id,
                            "source_relevance_tier": tier_name,
                            "source_base_event_id": base_event_id,
                        }
                    )

    if not emitted_events:
        raise ValueError("No expanded trainable rows were produced from the raw input")

    _write_events(output_samples_path, emitted_events)

    return ExpandedRawConversionSummary(
        input_rows=len(raw_rows),
        kept_rows=kept_input_rows,
        skipped_none_rows=len(raw_rows) - kept_input_rows,
        unique_scenarios=len(scenario_counts),
        unique_actions=len(action_ids),
        emitted_samples=len(emitted_events),
        tier_counts=dict(tier_counts),
    )


def convert_raw_sequence_to_v6_expanded_ro(
    input_path: str | Path,
    output_samples_path: str | Path,
    relevance_markdown: str | Path,
    most_relevant_reward: float = 1.0,
    plausible_reward: float = 0.6,
    irrelevant_reward: float = 0.0,
    most_relevant_repeat: int = 1,
    plausible_repeat: int = 1,
    irrelevant_repeat: int = 1,
    all_action_ids: list[str] | tuple[str, ...] | None = None,
    other_zero_mode: str = "none",
) -> ExpandedRawConversionSummary:
    return _expand_raw_sequence_with_v6_relevance(
        input_path=input_path,
        output_samples_path=output_samples_path,
        relevance_markdown=relevance_markdown,
        label_namespace="ro",
        most_relevant_reward=most_relevant_reward,
        plausible_reward=plausible_reward,
        irrelevant_reward=irrelevant_reward,
        most_relevant_repeat=most_relevant_repeat,
        plausible_repeat=plausible_repeat,
        irrelevant_repeat=irrelevant_repeat,
        all_action_ids=all_action_ids,
        other_zero_mode=other_zero_mode,
    )


def convert_raw_sequence_to_v6_expanded_app(
    input_path: str | Path,
    output_samples_path: str | Path,
    relevance_markdown: str | Path,
    most_relevant_reward: float = 1.0,
    plausible_reward: float = 0.6,
    irrelevant_reward: float = 0.0,
    most_relevant_repeat: int = 1,
    plausible_repeat: int = 1,
    irrelevant_repeat: int = 1,
    all_action_ids: list[str] | tuple[str, ...] | None = None,
    other_zero_mode: str = "none",
) -> ExpandedRawConversionSummary:
    return _expand_raw_sequence_with_v6_relevance(
        input_path=input_path,
        output_samples_path=output_samples_path,
        relevance_markdown=relevance_markdown,
        label_namespace="app",
        most_relevant_reward=most_relevant_reward,
        plausible_reward=plausible_reward,
        irrelevant_reward=irrelevant_reward,
        most_relevant_repeat=most_relevant_repeat,
        plausible_repeat=plausible_repeat,
        irrelevant_repeat=irrelevant_repeat,
        all_action_ids=all_action_ids,
        other_zero_mode=other_zero_mode,
    )
