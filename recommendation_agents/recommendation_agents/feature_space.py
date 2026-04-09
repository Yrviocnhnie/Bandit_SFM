"""V0 feature encoding for the shared-action LinUCB model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from recommendation_agents.taxonomies import (
    ACTIVITY_STATES,
    AGE_BUCKETS,
    CANONICAL_STATE_CODES,
    DAY_TYPES,
    LOCATION_CATEGORIES,
    MOTION_CATEGORIES,
    NETWORK_TYPES,
    PHONE_CATEGORIES,
    SEX_VALUES,
    SOUND_CATEGORIES,
    TIME_SLOTS,
    USER_HASH_BUCKETS,
)


def _clip_non_negative(value: Any) -> float:
    if value is None:
        return 0.0
    return max(0.0, float(value))


def _normalize_linear(value: Any, maximum: float) -> float:
    clipped = min(_clip_non_negative(value), maximum)
    return clipped / maximum if maximum else clipped


def _normalize_log(value: Any, maximum: float) -> float:
    clipped = min(_clip_non_negative(value), maximum)
    return float(np.log1p(clipped) / np.log1p(maximum))


def _normalize_user_hash_bucket(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.isdigit():
        return f"b{int(text)}"
    if text.startswith("b") and text[1:].isdigit():
        return f"b{int(text[1:])}"
    return text


def _resolve_state_previous(context: dict[str, Any]) -> Any:
    for key in ("precondition", "state_prev", "previous_state_current", "prev_state_current", "state_previous"):
        if key in context:
            return context.get(key)
    return None


@dataclass(frozen=True)
class FeatureDef:
    name: str
    kind: str
    categories: tuple[str, ...] = ()
    unknown_token: str | None = None
    scale: str = "linear"
    maximum: float = 1.0

    @property
    def dimension(self) -> int:
        if self.kind == "categorical":
            return len(self.categories)
        return 1


class V0FeatureSpace:
    """Encodes the V0 context vector without explicit scenarioId input."""

    def __init__(self) -> None:
        self.feature_defs = (
            FeatureDef("state_current", "categorical", CANONICAL_STATE_CODES, unknown_token="unknown"),
            FeatureDef("precondition", "categorical", CANONICAL_STATE_CODES, unknown_token="unknown"),
            FeatureDef("state_duration_sec", "numeric", scale="log", maximum=86400.0),
            FeatureDef("ps_time", "categorical", TIME_SLOTS),
            FeatureDef("hour", "categorical", tuple(str(i) for i in range(24))),
            FeatureDef("cal_hasUpcoming", "binary"),
            FeatureDef("ps_dayType", "categorical", DAY_TYPES),
            FeatureDef("ps_motion", "categorical", MOTION_CATEGORIES, unknown_token="unknown"),
            FeatureDef("wifiLost", "binary"),
            FeatureDef("wifiLostCategory", "categorical", LOCATION_CATEGORIES, unknown_token="unknown"),
            FeatureDef("cal_eventCount", "numeric", scale="log", maximum=20.0),
            FeatureDef("cal_inMeeting", "binary"),
            FeatureDef("cal_nextLocation", "categorical", LOCATION_CATEGORIES, unknown_token="unknown"),
            FeatureDef("ps_sound", "categorical", SOUND_CATEGORIES, unknown_token="unknown"),
            FeatureDef("sms_delivery_pending", "binary"),
            FeatureDef("sms_train_pending", "binary"),
            FeatureDef("sms_flight_pending", "binary"),
            FeatureDef("sms_hotel_pending", "binary"),
            FeatureDef("sms_movie_pending", "binary"),
            FeatureDef("sms_hospital_pending", "binary"),
            FeatureDef("sms_ride_pending", "binary"),
            FeatureDef("timestep", "numeric", scale="linear", maximum=86400.0),
            FeatureDef("ps_location", "categorical", LOCATION_CATEGORIES, unknown_token="unknown"),
            FeatureDef("ps_phone", "categorical", PHONE_CATEGORIES, unknown_token="unknown"),
            FeatureDef("batteryLevel", "numeric", scale="linear", maximum=100.0),
            FeatureDef("isCharging", "binary"),
            FeatureDef("networkType", "categorical", NETWORK_TYPES),
            FeatureDef("activityState", "categorical", ACTIVITY_STATES, unknown_token="unknown"),
            FeatureDef("activityDuration", "numeric", scale="log", maximum=86400.0),
            FeatureDef("user_id_hash_bucket", "categorical", USER_HASH_BUCKETS),
            FeatureDef("age_bucket", "categorical", AGE_BUCKETS, unknown_token="unknown"),
            FeatureDef("sex", "categorical", SEX_VALUES, unknown_token="unknown"),
            FeatureDef("has_kids", "binary"),
        )
        self._categorical_index = {
            feature.name: {value: index for index, value in enumerate(feature.categories)}
            for feature in self.feature_defs
            if feature.kind == "categorical"
        }

    @property
    def dimension(self) -> int:
        return sum(feature.dimension for feature in self.feature_defs)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for feature in self.feature_defs:
            if feature.kind == "categorical":
                names.extend(f"{feature.name}={value}" for value in feature.categories)
            else:
                names.append(feature.name)
        return names

    def encode(self, context: dict[str, Any]) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        cursor = 0
        for feature in self.feature_defs:
            raw_value = _resolve_state_previous(context) if feature.name == "precondition" else context.get(feature.name)
            if feature.kind == "categorical":
                encoded = self._encode_categorical(feature, raw_value)
            elif feature.kind == "binary":
                encoded = np.array([self._encode_binary(raw_value)], dtype=np.float32)
            else:
                encoded = np.array([self._encode_numeric(feature, raw_value)], dtype=np.float32)
            width = encoded.shape[0]
            vector[cursor : cursor + width] = encoded
            cursor += width
        return vector

    def _encode_categorical(self, feature: FeatureDef, value: Any) -> np.ndarray:
        if feature.name == "user_id_hash_bucket":
            category = _normalize_user_hash_bucket(value)
        else:
            category = str(value) if value is not None else feature.unknown_token
        index_map = self._categorical_index[feature.name]
        if category not in index_map:
            if feature.unknown_token is None:
                raise ValueError(f"Unknown value for {feature.name}: {value!r}")
            category = feature.unknown_token
        encoded = np.zeros(feature.dimension, dtype=np.float32)
        encoded[index_map[category]] = 1.0
        return encoded

    def _encode_binary(self, value: Any) -> float:
        if value in (None, "", False):
            return 0.0
        if value in (True, 1, "1"):
            return 1.0
        if value in (0, "0"):
            return 0.0
        numeric = float(value)
        if numeric not in (0.0, 1.0):
            raise ValueError(f"Binary feature expects 0/1, got {value!r}")
        return numeric

    def _encode_numeric(self, feature: FeatureDef, value: Any) -> float:
        if feature.scale == "log":
            return _normalize_log(value, feature.maximum)
        return _normalize_linear(value, feature.maximum)
