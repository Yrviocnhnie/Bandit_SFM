"""Feature encoder for the surprise-trigger demo.

Turns a `features` dict (from the generator JSONL) into a fixed-size
numpy float32 vector and exposes a structured `Encoding` object so the
VAE loss can apply:
  - cross-entropy per one-hot group
  - BCE per binary bit
  - MSE per scalar

User-profile features (`user_id_hash_bucket`, `age_bucket`, `sex`,
`has_kids`) are constant-per-user in this demo and excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


# ----- canonical vocabularies -----

STATE_CODES: List[str] = [
    # 44 base states
    "home_sleeping", "home_morning_workday", "home_morning_rest",
    "home_daytime_workday", "home_daytime_rest", "home_evening", "home_active",
    "office_arriving", "office_lunch_break", "office_working",
    "office_overtime", "office_late_overtime", "office_rest_day",
    "commuting_walk_out", "commuting_walk_home",
    "commuting_cycle_out", "commuting_cycle_home",
    "commuting_drive_out", "commuting_drive_home",
    "commuting_transit_out", "commuting_transit_home",
    "driving", "in_transit",
    "at_metro", "at_rail_station", "at_airport", "at_transit_hub",
    "outdoor_walking", "outdoor_running", "outdoor_cycling", "outdoor_resting",
    "at_restaurant_lunch", "at_restaurant_dinner", "at_restaurant_other",
    "at_cafe", "at_gym_exercising", "at_gym", "at_shopping", "at_health",
    "at_social", "at_education", "at_custom",
    "stationary_unknown", "walking_unknown",
    # 20 substates
    "home_sleeping_lying",
    "home_morning_workday_lying", "home_morning_rest_lying",
    "home_daytime_workday_dark", "home_daytime_workday_lying",
    "home_daytime_rest_dark", "home_daytime_rest_lying",
    "home_evening_dark", "home_evening_lying", "home_evening_noisy",
    "office_working_focused", "office_working_noisy",
    "at_cafe_quiet",
    "at_education_class", "at_education_break",
    "at_health_inpatient",
    "unknown_noisy", "unknown_dark", "unknown_settled", "unknown_lying",
]

PRECOND_CODES: List[str] = STATE_CODES + ["none"]

TIME_SLOTS: List[str] = [
    "sleeping", "dawn", "morning", "forenoon", "lunch",
    "afternoon", "evening", "night", "late_night",
]
DAY_TYPES: List[str] = ["workday", "weekend", "holiday"]
MOTION_CATS: List[str] = ["stationary", "walking", "running", "cycling",
                          "driving", "transit", "unknown"]
ACTIVITY_CATS: List[str] = ["sitting", "sleeping", "standing", "active", "unknown"]
PHONE_CATS: List[str] = ["in_use", "on_desk", "face_up", "face_down",
                         "in_pocket", "holding_lying", "charging", "unknown"]
LIGHT_CATS: List[str] = ["dark", "dim", "normal", "bright"]
SOUND_CATS: List[str] = ["silent", "quiet", "normal", "noisy", "unknown"]
LOCATION_CATS: List[str] = [
    "home", "work", "restaurant", "cafe", "gym", "metro", "rail_station",
    "airport", "transit", "shopping", "outdoor", "health", "social",
    "education", "custom", "en_route", "unknown",
]
TRANSPORT_CATS: List[str] = ["walking", "running", "cycling", "driving",
                             "transit", "stationary", "unknown"]
NETWORK_CATS: List[str] = ["wifi", "cellular", "none"]

SMS_FIELDS: List[str] = [
    "sms_delivery_pending", "sms_train_pending", "sms_flight_pending",
    "sms_hotel_pending", "sms_movie_pending", "sms_hospital_pending",
    "sms_ride_pending",
]

# order matters: all encoders index into this exactly.
ONE_HOT_GROUPS: List[Tuple[str, List[str]]] = [
    ("state_current", STATE_CODES),
    ("precondition", STATE_CODES + ["none"]),
    ("ps_time", TIME_SLOTS),
    ("ps_dayType", DAY_TYPES),
    ("ps_motion", MOTION_CATS),
    ("activityState", ACTIVITY_CATS),
    ("ps_phone", PHONE_CATS),
    ("ps_light", LIGHT_CATS),
    ("ps_sound", SOUND_CATS),
    ("ps_location", LOCATION_CATS),
    ("transportMode", TRANSPORT_CATS),
    ("networkType", NETWORK_CATS),
    ("wifiLostCategory", LOCATION_CATS),
    ("cal_nextLocation", LOCATION_CATS),
]

BINARY_FIELDS: List[str] = [
    "isCharging", "wifiLost", "cal_hasUpcoming", "cal_inMeeting",
] + SMS_FIELDS

# (name, lo, hi) for linear normalization to [0, 1], clipped
NUMERIC_FIELDS: List[Tuple[str, float, float]] = [
    ("hour", 0.0, 23.0),
    ("timestep", 0.0, 86400.0),
    ("state_duration_sec", 0.0, 28800.0),
    ("cal_eventCount", 0.0, 5.0),
    ("batteryLevel", 0.0, 100.0),
    ("activityDuration", 0.0, 28800.0),
]


@dataclass(frozen=True)
class GroupSpec:
    name: str
    kind: str          # "onehot" | "binary" | "scalar"
    offset: int
    size: int


@dataclass(frozen=True)
class Encoding:
    groups: Tuple[GroupSpec, ...]
    total_dim: int
    vocabs: Dict[str, Tuple[str, ...]]


def build_encoding() -> Encoding:
    groups: List[GroupSpec] = []
    offset = 0
    vocabs: Dict[str, Tuple[str, ...]] = {}
    for name, vocab in ONE_HOT_GROUPS:
        groups.append(GroupSpec(name=name, kind="onehot", offset=offset, size=len(vocab)))
        vocabs[name] = tuple(vocab)
        offset += len(vocab)
    for name in BINARY_FIELDS:
        groups.append(GroupSpec(name=name, kind="binary", offset=offset, size=1))
        offset += 1
    for name, _lo, _hi in NUMERIC_FIELDS:
        groups.append(GroupSpec(name=name, kind="scalar", offset=offset, size=1))
        offset += 1
    return Encoding(groups=tuple(groups), total_dim=offset, vocabs=vocabs)


ENCODING = build_encoding()


def _norm_scalar(raw: Any, lo: float, hi: float) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        v = lo
    span = hi - lo if hi > lo else 1.0
    return max(0.0, min(1.0, (v - lo) / span))


def encode_features(features: Dict[str, Any]) -> np.ndarray:
    vec = np.zeros(ENCODING.total_dim, dtype=np.float32)
    # one-hot
    off = 0
    for name, vocab in ONE_HOT_GROUPS:
        val = features.get(name)
        if val in vocab:
            vec[off + vocab.index(val)] = 1.0
        off += len(vocab)
    # binary
    for name in BINARY_FIELDS:
        raw = features.get(name, 0)
        try:
            vec[off] = 1.0 if int(raw) == 1 else 0.0
        except (TypeError, ValueError):
            vec[off] = 0.0
        off += 1
    # scalar
    for name, lo, hi in NUMERIC_FIELDS:
        vec[off] = _safe_scalar(features.get(name), lo, hi)
        off += 1
    return vec


def _safe_scalar(x: Any, lo: float, hi: float) -> float:
    return _norm_scalar(x, lo, hi)


def get_encoding() -> Encoding:
    return ENCODING


if __name__ == "__main__":
    enc = build_encoding()
    print(f"total dim: {enc.total_dim}")
    print("groups:")
    for g in enc.groups:
        print(f"  {g.kind:7s} {g.name:30s} offset={g.offset:4d} size={g.size}")
