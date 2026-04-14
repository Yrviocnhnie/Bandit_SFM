"""Parsers and helpers for simulated online feedback specs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from recommendation_agents.raw_synthetic import RAW_TO_V0_CONTEXT_FIELDS


_TIME_SLOT_TO_HOUR = {
    "dawn": 6,
    "morning": 9,
    "forenoon": 10,
    "lunch": 12,
    "afternoon": 15,
    "evening": 18,
    "night": 22,
    "late_night": 23,
    "sleeping": 2,
}


def build_phase2_anchor_context(partial_features: dict[str, Any]) -> dict[str, Any]:
    """Fill missing V0 fields with stable defaults for explicit feedback anchors."""

    defaults: dict[str, Any] = {
        "state_current": "unknown",
        "precondition": "unknown",
        "state_duration_sec": 1800,
        "ps_time": "morning",
        "hour": 9,
        "cal_hasUpcoming": 0,
        "ps_dayType": "workday",
        "ps_motion": "unknown",
        "wifiLost": 0,
        "wifiLostCategory": "unknown",
        "cal_eventCount": 0,
        "cal_inMeeting": 0,
        "cal_nextLocation": "unknown",
        "ps_sound": "unknown",
        "sms_delivery_pending": 0,
        "sms_train_pending": 0,
        "sms_flight_pending": 0,
        "sms_hotel_pending": 0,
        "sms_movie_pending": 0,
        "sms_hospital_pending": 0,
        "sms_ride_pending": 0,
        "timestep": 43200,
        "ps_location": "unknown",
        "ps_phone": "unknown",
        "batteryLevel": 60,
        "isCharging": 0,
        "networkType": "wifi",
        "activityState": "unknown",
        "activityDuration": 1800,
        "user_id_hash_bucket": "b00",
        "age_bucket": "25_34",
        "sex": "unknown",
        "has_kids": 0,
    }
    context = dict(defaults)
    context.update(partial_features)
    if context.get("hour") in (None, ""):
        context["hour"] = _TIME_SLOT_TO_HOUR.get(str(context.get("ps_time") or "morning"), 9)
    return {field: context.get(field) for field in RAW_TO_V0_CONTEXT_FIELDS}


def parse_phase2_feedback_markdown(markdown_path: str | Path) -> list[dict[str, Any]]:
    """Parse phase-2 multi-anchor markdown into explicit feedback specs."""

    text = Path(markdown_path).read_text()
    if "## " not in text or "### Context " not in text:
        raise ValueError(f"No scenario/context sections found in {markdown_path}")

    parsed: list[dict[str, Any]] = []
    scenario_sections = re.split(r"^##\s+\d+\.\s+SCENARIO:\s+", text, flags=re.MULTILINE)
    for scenario_section in scenario_sections[1:]:
        lines = scenario_section.splitlines()
        if not lines:
            continue
        scenario_id = lines[0].strip()
        if not scenario_id:
            continue
        body = "\n".join(lines[1:])
        context_sections = re.split(r"^###\s+Context\s+", body, flags=re.MULTILINE)
        for context_section in context_sections[1:]:
            context_lines = context_section.splitlines()
            if not context_lines:
                continue
            header = context_lines[0].strip()
            header_match = re.match(r"([A-Za-z0-9]+):\s+(.+)$", header)
            if header_match is None:
                raise ValueError(f"Malformed context header {header!r} in {markdown_path}")
            context_code = header_match.group(1)
            context_title = header_match.group(2).strip()
            context_body = "\n".join(context_lines[1:])

            features_match = re.search(
                r"\*\*Features\*\*\n(?P<body>.*?)(?:\n\*\*|\Z)",
                context_body,
                flags=re.DOTALL,
            )
            feedback_match = re.search(
                r"\*\*Simulated feedback\*\*\n(?P<body>.*?)(?:\n\*\*|\Z)",
                context_body,
                flags=re.DOTALL,
            )
            if features_match is None or feedback_match is None:
                raise ValueError(
                    f"Context {scenario_id} / {context_code} is missing Features or Simulated feedback in {markdown_path}"
                )

            partial_features: dict[str, Any] = {}
            for line in features_match.group("body").splitlines():
                stripped = line.strip()
                if not stripped.startswith("- "):
                    continue
                key, _, raw_value = stripped[2:].partition(":")
                if not _:
                    continue
                value_text = raw_value.strip()
                if value_text in {"0", "1"}:
                    value: Any = int(value_text)
                else:
                    value = value_text
                partial_features[key.strip()] = value

            feedback_payload: dict[str, str] = {}
            for line in feedback_match.group("body").splitlines():
                stripped = line.strip()
                if not stripped.startswith("- "):
                    continue
                key, _, raw_value = stripped[2:].partition(":")
                if not _:
                    continue
                feedback_payload[key.strip().lower()] = raw_value.strip().strip("`")

            missing_feedback = {"like", "dislike"} - set(feedback_payload)
            if missing_feedback:
                raise ValueError(
                    f"Context {scenario_id} / {context_code} is missing feedback types {sorted(missing_feedback)} in {markdown_path}"
                )

            anchor_id = f"{scenario_id.lower()}__context_{context_code.lower()}"
            anchor_context = build_phase2_anchor_context(partial_features)
            for feedback_type in ("like", "dislike"):
                parsed.append(
                    {
                        "feedback_id": f"{anchor_id}__{feedback_type}",
                        "anchor_id": anchor_id,
                        "scenario_id": scenario_id,
                        "feedback_type": feedback_type,
                        "target_action_id": feedback_payload[feedback_type],
                        "anchor_context": anchor_context,
                        "anchor_context_id": context_code,
                        "anchor_context_title": context_title,
                        "source_markdown": str(Path(markdown_path)),
                    }
                )

    if not parsed:
        raise ValueError(f"No feedback items were parsed from {markdown_path}")
    return parsed
