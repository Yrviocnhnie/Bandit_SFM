"""App-aligned 726-d feature encoder for the Python Dual-UCB prototype."""

from __future__ import annotations

import json
import math
from typing import List, Optional, Sequence

try:
    from .feature_space import (
        ACTIVITY_CLASSES,
        APP_ACTIONS,
        BANDIT_FEATURE_DIM,
        DAY_TYPES,
        HOUR_CLASSES,
        LOCATION_CLASSES,
        MOTION_CLASSES,
        NETWORK_CLASSES,
        PHONE_CLASSES,
        RECENT_APP_CATEGORY_CLASSES,
        RECENT_RO_ACTION_CLASSES,
        SCENARIO_IDS,
        SOUND_CLASSES,
        STATE_CODES,
        TIME_SLOTS,
    )
    from .state_and_scenario import ContextSnapshot, StateChainEntry
except ImportError:
    from feature_space import (
        ACTIVITY_CLASSES,
        APP_ACTIONS,
        BANDIT_FEATURE_DIM,
        DAY_TYPES,
        HOUR_CLASSES,
        LOCATION_CLASSES,
        MOTION_CLASSES,
        NETWORK_CLASSES,
        PHONE_CLASSES,
        RECENT_APP_CATEGORY_CLASSES,
        RECENT_RO_ACTION_CLASSES,
        SCENARIO_IDS,
        SOUND_CLASSES,
        STATE_CODES,
        TIME_SLOTS,
    )
    from state_and_scenario import ContextSnapshot, StateChainEntry


def _parse_tri_state(raw: str) -> int:
    if raw == "true":
        return 0
    if raw == "false":
        return 1
    return 2


def _push_one_hot(vec: List[float], vocab: Sequence[str], raw: Optional[str], fallback: str) -> None:
    value = raw or fallback
    idx = list(vocab).index(value) if value in vocab else list(vocab).index(fallback)
    for i in range(len(vocab)):
        vec.append(1.0 if i == idx else 0.0)


def _push_one_hot_allow_zero(vec: List[float], vocab: Sequence[str], raw: Optional[str]) -> None:
    idx = list(vocab).index(raw) if raw in vocab else -1
    for i in range(len(vocab)):
        vec.append(1.0 if i == idx else 0.0)


def _push_tri_state(vec: List[float], raw: Optional[str]) -> None:
    idx = _parse_tri_state(raw or "")
    for i in range(3):
        vec.append(1.0 if i == idx else 0.0)


def _push_scalar(vec: List[float], value: float) -> None:
    vec.append(value if math.isfinite(value) else 0.0)


def _parse_number(raw: Optional[str]) -> float:
    try:
        return float(raw) if raw not in (None, "") else 0.0
    except ValueError:
        return 0.0


def _normalize_log1p(value: float, max_value: float) -> float:
    clamped = max(0.0, min(value, max_value))
    return math.log1p(clamped) / math.log1p(max_value)


def normalize_location_class(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    normalized = raw.lower().replace(" ", "_")
    if normalized == "commute":
        return "en_route"
    if normalized == "subway":
        return "metro"
    if normalized in ("bus_stop", "ferry", "train_station"):
        return "transit"
    if normalized in LOCATION_CLASSES:
        return normalized
    if "airport" in normalized:
        return "airport"
    if "cafe" in normalized:
        return "cafe"
    if "restaurant" in normalized or "dining" in normalized:
        return "restaurant"
    if "shop" in normalized or "mall" in normalized:
        return "shopping"
    if "health" in normalized or "hospital" in normalized or "clinic" in normalized:
        return "health"
    if "school" in normalized or "education" in normalized or "class" in normalized:
        return "education"
    if "social" in normalized or "bar" in normalized or "ktv" in normalized:
        return "social"
    return "unknown"


def normalize_network(raw: Optional[str]) -> str:
    if not raw:
        return "other"
    normalized = raw.lower()
    return normalized if normalized in NETWORK_CLASSES else "other"


def normalize_activity(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    normalized = raw.lower()
    if normalized in ("walking", "running", "cycling"):
        return "active"
    return normalized if normalized in ACTIVITY_CLASSES else "unknown"


def parse_state_chain_json(raw: str) -> List[StateChainEntry]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    entries: List[StateChainEntry] = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            entries.append(
                StateChainEntry(
                    code=str(item.get("code", "")),
                    start_time=int(item.get("startTime", 0) or 0),
                    duration_ms=int(item.get("durationMs", 0) or 0),
                )
            )
    return entries


def location_from_state_code(state_code: str, fallback: str = "unknown") -> str:
    if not state_code:
        return fallback
    if state_code.startswith("home_"):
        return "home"
    if state_code.startswith("office_"):
        return "work"
    if state_code.startswith("commuting_"):
        return "en_route"
    if state_code.startswith("at_restaurant_"):
        return "restaurant"
    if state_code.startswith("at_cafe"):
        return "cafe"
    if state_code.startswith("at_gym"):
        return "gym"
    if state_code.startswith("at_shopping"):
        return "shopping"
    if state_code.startswith("at_health"):
        return "health"
    if state_code.startswith("at_social"):
        return "social"
    if state_code.startswith("at_education"):
        return "education"
    if state_code.startswith("at_custom"):
        return "custom"
    if state_code == "at_metro":
        return "metro"
    if state_code == "at_rail_station":
        return "rail_station"
    if state_code == "at_airport":
        return "airport"
    if state_code in ("at_transit_hub", "in_transit"):
        return "transit"
    if state_code.startswith("outdoor_"):
        return "outdoor"
    return fallback


def hash_bucket(raw: str, modulo: int) -> int:
    value = 2166136261
    for ch in raw:
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return value % modulo if modulo > 0 else 0


class DualBanditFeatureEncoderPy:
    FEATURE_DIM = BANDIT_FEATURE_DIM
    APP_CATEGORIES = APP_ACTIONS
    STATE_CODES = STATE_CODES
    LOCATION_CLASSES = LOCATION_CLASSES

    def encode_shared(self, snapshot: ContextSnapshot, scenario_id: Optional[str] = None) -> List[float]:
        vec: List[float] = []
        chain = parse_state_chain_json(snapshot.state_chain_json)
        hist1 = chain[1].code if len(chain) > 1 else snapshot.state_prev
        hist2 = chain[2].code if len(chain) > 2 else ""
        hist3 = chain[3].code if len(chain) > 3 else ""
        active_scenario_id = scenario_id or snapshot.ps_scenario
        hour = int(_parse_number(snapshot.hour))
        minute_of_day = max(0, min(hour * 3600, 86400))

        _push_one_hot_allow_zero(vec, SCENARIO_IDS, active_scenario_id)
        _push_one_hot(vec, STATE_CODES, snapshot.state_current, "stationary_unknown")
        _push_scalar(vec, _normalize_log1p(_parse_number(snapshot.state_duration_sec), 12 * 60 * 60))
        _push_one_hot(vec, HOUR_CLASSES, snapshot.hour, "0")
        _push_one_hot(vec, TIME_SLOTS, snapshot.ps_time, "sleeping")
        _push_tri_state(vec, snapshot.cal_hasUpcoming)
        _push_scalar(vec, _normalize_log1p(_parse_number(snapshot.cal_nextMinutes), 24 * 60))
        _push_one_hot(vec, DAY_TYPES, snapshot.ps_dayType, "workday")
        _push_one_hot(vec, MOTION_CLASSES, snapshot.ps_motion, "unknown")
        _push_tri_state(vec, snapshot.wifiLost)
        _push_one_hot(vec, LOCATION_CLASSES, normalize_location_class(snapshot.wifiLostCategory), "unknown")
        _push_scalar(vec, max(0.0, min(_parse_number(snapshot.cal_eventCount), 10.0)) / 10.0)
        _push_tri_state(vec, snapshot.cal_inMeeting)
        _push_one_hot(vec, LOCATION_CLASSES, normalize_location_class(snapshot.cal_nextLocation), "unknown")
        _push_one_hot(vec, SOUND_CLASSES, snapshot.ps_sound, "unknown")
        _push_tri_state(vec, snapshot.sms_delivery_pending)
        _push_tri_state(vec, snapshot.sms_train_pending)
        _push_tri_state(vec, snapshot.sms_flight_pending)
        _push_tri_state(vec, snapshot.sms_hotel_pending)
        _push_tri_state(vec, snapshot.sms_movie_pending)
        _push_tri_state(vec, snapshot.sms_hospital_pending)
        _push_tri_state(vec, snapshot.sms_ride_pending)

        _push_scalar(vec, minute_of_day / 86400.0)
        _push_one_hot(vec, LOCATION_CLASSES, normalize_location_class(snapshot.ps_location), "unknown")
        _push_one_hot(vec, PHONE_CLASSES, snapshot.ps_phone, "unknown")
        _push_scalar(vec, max(0.0, min(_parse_number(snapshot.batteryLevel), 100.0)) / 100.0)
        _push_tri_state(vec, snapshot.isCharging)
        _push_one_hot(vec, NETWORK_CLASSES, normalize_network(snapshot.networkType), "other")
        _push_one_hot(vec, MOTION_CLASSES, snapshot.transportMode, "unknown")
        _push_one_hot(vec, ACTIVITY_CLASSES, normalize_activity(snapshot.activityState), "unknown")
        _push_scalar(vec, _normalize_log1p(_parse_number(snapshot.activityDuration), 12 * 60 * 60))

        _push_one_hot(vec, STATE_CODES, hist1, "stationary_unknown")
        _push_one_hot(vec, STATE_CODES, hist2, "stationary_unknown")
        _push_one_hot(vec, STATE_CODES, hist3, "stationary_unknown")
        _push_one_hot(vec, LOCATION_CLASSES, normalize_location_class(location_from_state_code(hist1, snapshot.ps_location)), "unknown")
        _push_one_hot(vec, ACTIVITY_CLASSES, normalize_activity(snapshot.prevActivityState), "unknown")

        _push_one_hot(vec, RECENT_RO_ACTION_CLASSES, "none", "none")
        _push_one_hot(vec, RECENT_APP_CATEGORY_CLASSES, "none", "none")
        _push_scalar(vec, 0.0)
        _push_scalar(vec, 0.0)

        bucket = hash_bucket("anonymous", 32)
        for idx in range(32):
            vec.append(1.0 if idx == bucket else 0.0)
        _push_scalar(vec, 0.0)
        vec.extend([0.0, 0.0, 0.0, 0.0])
        vec.extend([0.0, 0.0, 0.0])

        if len(vec) < BANDIT_FEATURE_DIM:
            vec.extend([0.0] * (BANDIT_FEATURE_DIM - len(vec)))
        elif len(vec) > BANDIT_FEATURE_DIM:
            vec = vec[:BANDIT_FEATURE_DIM]
        return vec

    def encode_ro_input(self, snapshot: ContextSnapshot, scenario_id: Optional[str] = None) -> List[float]:
        return self.encode_shared(snapshot, scenario_id)

    def encode_app_input(self, snapshot: ContextSnapshot, scenario_id: Optional[str] = None) -> List[float]:
        return self.encode_shared(snapshot, scenario_id)

