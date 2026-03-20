"""App-aligned feature-space definitions for the Python Dual-UCB prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ENCODER_PATH = REPO_ROOT / "entry" / "src" / "main" / "ets" / "service" / "context" / "dispatch" / "DualBanditFeatureEncoder.ets"
APP_CATALOG_PATH = REPO_ROOT / "entry" / "src" / "main" / "ets" / "service" / "context" / "dispatch" / "BanditActionCatalog.ets"

BANDIT_FEATURE_DIM = 726
CHAIN_MAX_LEN = 10


def _extract_string_array(text: str, const_name: str) -> List[str]:
    pattern = re.compile(rf"const {re.escape(const_name)}: string\[\] = \[(.*?)\];", re.S)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"{const_name} not found in {APP_ENCODER_PATH}")
    return re.findall(r"'([^']+)'", match.group(1))


def _fallback_state_codes() -> List[str]:
    return [
        "home_sleeping",
        "home_morning_workday",
        "home_morning_rest",
        "home_daytime_workday",
        "home_daytime_rest",
        "home_evening",
        "home_active",
        "office_arriving",
        "office_lunch_break",
        "office_working",
        "office_overtime",
        "office_late_overtime",
        "office_rest_day",
        "commuting_walk_out",
        "commuting_walk_home",
        "commuting_cycle_out",
        "commuting_cycle_home",
        "commuting_drive_out",
        "commuting_drive_home",
        "commuting_transit_out",
        "commuting_transit_home",
        "driving",
        "in_transit",
        "at_metro",
        "at_rail_station",
        "at_airport",
        "at_transit_hub",
        "outdoor_walking",
        "outdoor_running",
        "outdoor_cycling",
        "outdoor_resting",
        "at_restaurant_lunch",
        "at_restaurant_dinner",
        "at_restaurant_other",
        "at_cafe",
        "at_gym_exercising",
        "at_gym",
        "at_shopping",
        "at_health",
        "at_social",
        "at_education",
        "at_custom",
        "stationary_unknown",
        "walking_unknown",
        "home_sleeping_lying",
        "home_morning_workday_lying",
        "home_morning_rest_lying",
        "home_daytime_workday_dark",
        "home_daytime_workday_lying",
        "home_daytime_rest_dark",
        "home_daytime_rest_lying",
        "home_evening_dark",
        "home_evening_lying",
        "home_evening_noisy",
        "office_working_focused",
        "office_working_noisy",
        "at_cafe_quiet",
        "at_education_class",
        "at_education_break",
        "at_health_inpatient",
        "unknown_noisy",
        "unknown_dark",
        "unknown_settled",
        "unknown_lying",
    ]


def _load_encoder_vocab() -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    if APP_ENCODER_PATH.exists():
        text = APP_ENCODER_PATH.read_text(encoding="utf-8")
        return (
            _extract_string_array(text, "STATE_CODES"),
            _extract_string_array(text, "LOCATION_CLASSES"),
            _extract_string_array(text, "TIME_SLOTS"),
            _extract_string_array(text, "DAY_TYPES"),
            _extract_string_array(text, "MOTION_CLASSES"),
            _extract_string_array(text, "PHONE_CLASSES"),
            _extract_string_array(text, "SOUND_CLASSES"),
            _extract_string_array(text, "ACTIVITY_CLASSES"),
            _extract_string_array(text, "NETWORK_CLASSES"),
        )

    return (
        _fallback_state_codes(),
        ["unknown", "home", "work", "restaurant", "cafe", "gym", "metro", "rail_station", "airport", "transit", "shopping", "outdoor", "health", "social", "education", "custom", "en_route"],
        ["sleeping", "dawn", "morning", "forenoon", "lunch", "afternoon", "evening", "night", "late_night"],
        ["workday", "weekend", "holiday"],
        ["stationary", "walking", "running", "cycling", "driving", "transit", "unknown"],
        ["in_use", "holding_lying", "on_desk", "face_up", "in_pocket", "face_down", "charging", "unknown"],
        ["silent", "quiet", "normal", "noisy", "unknown"],
        ["sitting", "sleeping", "standing", "active", "unknown"],
        ["wifi", "cellular", "none", "other"],
    )


def _load_bandit_catalog() -> Tuple[List[str], List[str]]:
    if APP_CATALOG_PATH.exists():
        text = APP_CATALOG_PATH.read_text(encoding="utf-8")
        scenario_ids = re.findall(r"scenarioId: '([^']+)'", text)
        ro_actions = re.findall(r"\{ type: '[RO]', id: '([^']+)' \}", text)
        unique_ro: List[str] = []
        for action_id in ro_actions:
            if action_id not in unique_ro:
                unique_ro.append(action_id)
        if scenario_ids and unique_ro:
            unique_scenarios: List[str] = []
            for scenario_id in scenario_ids:
                if scenario_id not in unique_scenarios:
                    unique_scenarios.append(scenario_id)
            return unique_scenarios, unique_ro

    return [], []


STATE_CODES, LOCATION_CLASSES, TIME_SLOTS, DAY_TYPES, MOTION_CLASSES, PHONE_CLASSES, SOUND_CLASSES, ACTIVITY_CLASSES, NETWORK_CLASSES = _load_encoder_vocab()
SCENARIO_IDS, RO_ACTIONS = _load_bandit_catalog()
APP_ACTIONS: List[str] = [
    "social",
    "productivity",
    "entertainment",
    "navigation",
    "shopping",
    "news",
    "health",
    "music",
    "reading",
    "game",
]

TRI_STATE_CLASSES: List[str] = ["true", "false", "unknown"]
RECENT_RO_ACTION_CLASSES: List[str] = RO_ACTIONS + ["none"]
RECENT_APP_CATEGORY_CLASSES: List[str] = APP_ACTIONS + ["none"]
HOUR_CLASSES: List[str] = [str(i) for i in range(24)]
USER_ID_BUCKETS: List[str] = [f"user_bucket_{i}" for i in range(32)]
SEX_CLASSES: List[str] = ["unknown", "female", "male", "other"]
KIDS_CLASSES: List[str] = ["unknown", "no", "yes"]


FEATURE_GROUPS: List[Tuple[str, int, Sequence[str] | None]] = [
    ("scenarioId", len(SCENARIO_IDS), SCENARIO_IDS),
    ("state_current", len(STATE_CODES), STATE_CODES),
    ("state_duration_sec", 1, None),
    ("hour", len(HOUR_CLASSES), HOUR_CLASSES),
    ("ps_time", len(TIME_SLOTS), TIME_SLOTS),
    ("cal_hasUpcoming", len(TRI_STATE_CLASSES), TRI_STATE_CLASSES),
    ("cal_nextMinutes", 1, None),
    ("ps_dayType", len(DAY_TYPES), DAY_TYPES),
    ("ps_motion", len(MOTION_CLASSES), MOTION_CLASSES),
    ("wifiLost", len(TRI_STATE_CLASSES), TRI_STATE_CLASSES),
    ("wifiLostCategory", len(LOCATION_CLASSES), LOCATION_CLASSES),
    ("cal_eventCount", 1, None),
    ("cal_inMeeting", len(TRI_STATE_CLASSES), TRI_STATE_CLASSES),
    ("cal_nextLocation", len(LOCATION_CLASSES), LOCATION_CLASSES),
    ("ps_sound", len(SOUND_CLASSES), SOUND_CLASSES),
    ("sms_delivery_pending", len(TRI_STATE_CLASSES), TRI_STATE_CLASSES),
    ("sms_train_pending", len(TRI_STATE_CLASSES), TRI_STATE_CLASSES),
    ("sms_flight_pending", len(TRI_STATE_CLASSES), TRI_STATE_CLASSES),
    ("sms_hotel_pending", len(TRI_STATE_CLASSES), TRI_STATE_CLASSES),
    ("sms_movie_pending", len(TRI_STATE_CLASSES), TRI_STATE_CLASSES),
    ("sms_hospital_pending", len(TRI_STATE_CLASSES), TRI_STATE_CLASSES),
    ("sms_ride_pending", len(TRI_STATE_CLASSES), TRI_STATE_CLASSES),
    ("timestep", 1, None),
    ("ps_location", len(LOCATION_CLASSES), LOCATION_CLASSES),
    ("ps_phone", len(PHONE_CLASSES), PHONE_CLASSES),
    ("batteryLevel", 1, None),
    ("isCharging", len(TRI_STATE_CLASSES), TRI_STATE_CLASSES),
    ("networkType", len(NETWORK_CLASSES), NETWORK_CLASSES),
    ("transportMode", len(MOTION_CLASSES), MOTION_CLASSES),
    ("activityState", len(ACTIVITY_CLASSES), ACTIVITY_CLASSES),
    ("activityDuration", 1, None),
    ("hist_state_1", len(STATE_CODES), STATE_CODES),
    ("hist_state_2", len(STATE_CODES), STATE_CODES),
    ("hist_state_3", len(STATE_CODES), STATE_CODES),
    ("prev_location", len(LOCATION_CLASSES), LOCATION_CLASSES),
    ("prev_activityState", len(ACTIVITY_CLASSES), ACTIVITY_CLASSES),
    ("recent_accepted_ro_action_same_scenario", len(RECENT_RO_ACTION_CLASSES), RECENT_RO_ACTION_CLASSES),
    ("recent_accepted_app_category_same_scenario", len(RECENT_APP_CATEGORY_CLASSES), RECENT_APP_CATEGORY_CLASSES),
    ("recent_ro_feedback_score_same_scenario", 1, None),
    ("recent_app_feedback_score_same_scenario", 1, None),
    ("user_id_hash_bucket", len(USER_ID_BUCKETS), USER_ID_BUCKETS),
    ("age", 1, None),
    ("sex", len(SEX_CLASSES), SEX_CLASSES),
    ("kids", len(KIDS_CLASSES), KIDS_CLASSES),
    ("padding_reserved", 1, None),
]


def build_feature_names() -> List[str]:
    names: List[str] = []
    for group_name, dim, classes in FEATURE_GROUPS:
        if classes is None:
            if dim == 1:
                names.append(group_name)
            else:
                for idx in range(dim):
                    names.append(f"{group_name}[{idx}]")
        else:
            for class_name in classes:
                names.append(f"{group_name}={class_name}")
    return names


BANDIT_FEATURE_NAMES: List[str] = build_feature_names()


@dataclass(frozen=True)
class FeatureSpaceSpec:
    bandit_feature_dim: int = BANDIT_FEATURE_DIM
    bandit_feature_name_count: int = len(BANDIT_FEATURE_NAMES)
    scenario_count: int = len(SCENARIO_IDS)
    ro_action_dim: int = len(RO_ACTIONS)
    app_action_dim: int = len(APP_ACTIONS)


FEATURE_SPACE_SPEC = FeatureSpaceSpec()

if FEATURE_SPACE_SPEC.bandit_feature_name_count != BANDIT_FEATURE_DIM:
    raise ValueError(
        f"feature layout mismatch: expected {BANDIT_FEATURE_DIM}, got {FEATURE_SPACE_SPEC.bandit_feature_name_count}"
    )

