#!/usr/bin/env python3
"""
State-aligned synthetic data generator for the v0 contextual bandit setup.

Key behavior:
- first prediction timestep only
- includes all scenarios by default
- uses updated rules with preconditions + state_current
- uses v5/global action space ranked labels
- ensures state_code-dependent features align with the chosen state_current
- outputs v0 model features only (no long history features)

GT policy:
- RO: first ranked action gets 50% probability; remaining 50% split uniformly across the rest
- App: first ranked app category gets 50% probability; remaining 50% split uniformly across the rest
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

APP_CATEGORIES: List[str] = [
    "social", "productivity", "entertainment", "navigation", "shopping",
    "news", "health", "music", "reading", "game",
]
TIME_SLOTS = ["sleeping", "dawn", "morning", "forenoon", "lunch", "afternoon", "evening", "night", "late_night"]
TIME_SLOT_HOURS = {
    "sleeping": list(range(0, 5)),
    "dawn": [5, 6],
    "morning": [7, 8],
    "forenoon": [9, 10, 11],
    "lunch": [12, 13],
    "afternoon": [14, 15, 16, 17],
    "evening": [18, 19],
    "night": [20, 21],
    "late_night": [22, 23],
}
LOCATION_CATEGORIES = [
    "home", "work", "restaurant", "cafe", "gym", "metro", "rail_station", "airport",
    "transit", "shopping", "outdoor", "health", "social", "education", "custom", "en_route", "unknown",
]
MOTION_CATEGORIES = ["stationary", "walking", "running", "cycling", "driving", "transit", "unknown"]
PHONE_CATEGORIES = ["in_use", "holding_lying", "on_desk", "face_up", "in_pocket", "face_down", "charging", "unknown"]
SOUND_CATEGORIES = ["silent", "quiet", "normal", "noisy", "unknown"]
LIGHT_CATEGORIES = ["dark", "dim", "normal", "bright"]
DAY_TYPES = ["workday", "weekend", "holiday"]
NETWORK_TYPES = ["wifi", "cellular", "none"]
TRANSPORT_MODES = ["walking", "running", "cycling", "driving", "transit", "stationary", "unknown"]
ACTIVITY_STATES = ["sitting", "sleeping", "standing", "active", "unknown"]
AGE_BUCKETS = ["0_17", "18_24", "25_34", "35_44", "45_54", "55_64", "65_plus", "unknown"]
SEX_BUCKETS = ["female", "male", "other", "unknown"]
USER_BUCKETS = [f"b{i:02d}" for i in range(32)]
SMS_FIELDS = [
    "sms_delivery_pending", "sms_train_pending", "sms_flight_pending", "sms_hotel_pending",
    "sms_movie_pending", "sms_hospital_pending", "sms_ride_pending",
]
LEGACY_STATE_MAP = {
    "at_education_quiet": "at_education_class",
    "at_education_noisy": "at_education_break",
}
SCENARIO_ALIAS_TO_V5 = {
    "OFFICE_AFTERNOON": "OFFICE_WORKING",
}
DEFAULT_IGNORED_TRIGGER_SCENARIOS = set()

SUBSTATE_TRIGGER = {
    "home_sleeping_lying": ("phone", "holding_lying"),
    "home_morning_workday_lying": ("phone", "holding_lying"),
    "home_morning_rest_lying": ("phone", "holding_lying"),
    "home_daytime_workday_dark": ("light", "dark"),
    "home_daytime_workday_lying": ("phone", "holding_lying"),
    "home_daytime_rest_dark": ("light", "dark"),
    "home_daytime_rest_lying": ("phone", "holding_lying"),
    "home_evening_dark": ("light", "dark"),
    "home_evening_lying": ("phone", "holding_lying"),
    "home_evening_noisy": ("sound", "noisy"),
    "office_working_focused": ("phone", "face_down"),
    "office_working_noisy": ("sound", "noisy"),
    "at_cafe_quiet": ("sound", "quiet"),
    "at_education_class": ("sound", "quiet"),
    "at_education_break": ("sound", "noisy"),
    "at_health_inpatient": ("phone", "holding_lying"),
    "unknown_noisy": ("sound", "noisy"),
    "unknown_dark": ("light", "dark"),
    "unknown_settled": ("phone", "charging"),
    "unknown_lying": ("phone", "holding_lying"),
}
BASE_AVOID_TRIGGERS = {
    "home_sleeping": {("phone", "holding_lying")},
    "home_morning_workday": {("phone", "holding_lying")},
    "home_morning_rest": {("phone", "holding_lying")},
    "home_daytime_workday": {("phone", "holding_lying"), ("light", "dark")},
    "home_daytime_rest": {("phone", "holding_lying"), ("light", "dark")},
    "home_evening": {("phone", "holding_lying"), ("light", "dark"), ("sound", "noisy")},
    "office_working": {("phone", "face_down"), ("sound", "noisy")},
    "at_cafe": {("sound", "quiet")},
    "at_education": {("sound", "quiet"), ("sound", "noisy")},
    "at_health": {("phone", "holding_lying")},
    "stationary_unknown": {("sound", "noisy"), ("light", "dark"), ("phone", "charging"), ("phone", "holding_lying")},
}

@dataclass
class ScenarioSpec:
    scenario_id: str
    scenario_name: str
    category: str
    conditions: List[Dict[str, Any]]
    preconditions: List[Dict[str, Any]]
    default_action_ids: List[str]
    default_app_categories: List[str]
    v5_scenario_id: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--global-action-space", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--episodes-per-scenario", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scenario-ids", type=str, default="")
    p.add_argument("--ignore-trigger-scenarios", type=str, default=",".join(sorted(DEFAULT_IGNORED_TRIGGER_SCENARIOS)))
    p.add_argument("--debug-examples", type=Path, default=None)
    p.add_argument("--run-smoke-tests", action="store_true")
    return p.parse_args()


def normalize_state(value: str) -> str:
    return LEGACY_STATE_MAP.get(str(value).strip(), str(value).strip())


def split_csv_values(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [normalize_state(x) for x in str(value).split(",") if str(x).strip()]


def parse_rules(rules_json: Path) -> Dict[str, Dict[str, Any]]:
    rules = json.loads(rules_json.read_text(encoding="utf-8"))
    return {rule["scenarioId"]: rule for rule in rules}


def parse_global_action_space(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    action_type_by_id = {a["actionId"]: a["actionType"] for a in data["globalActions"]}
    scenario_defaults = {s["scenarioId"]: s for s in data["scenarioDefaults"]}
    return scenario_defaults, action_type_by_id


def build_scenario_specs(rules_by_sid: Dict[str, Dict[str, Any]], scenario_defaults: Dict[str, Dict[str, Any]]) -> Dict[str, ScenarioSpec]:
    specs: Dict[str, ScenarioSpec] = {}
    for sid, rule in rules_by_sid.items():
        v5_sid = SCENARIO_ALIAS_TO_V5.get(sid, sid)
        if v5_sid not in scenario_defaults:
            continue
        sd = scenario_defaults[v5_sid]
        specs[sid] = ScenarioSpec(
            scenario_id=sid,
            scenario_name=rule.get("scenarioNameEn", sd.get("scenarioName", sid)),
            category=rule.get("category", sd.get("category", "unknown")),
            conditions=rule.get("conditions", []),
            preconditions=rule.get("preconditions", []),
            default_action_ids=list(sd["defaultActionIds"]),
            default_app_categories=list(sd["defaultAppCategories"]),
            v5_scenario_id=v5_sid,
        )
    return specs


def choose_current_state(rule: Dict[str, Any]) -> List[str]:
    current_state_opts: List[str] = []
    for cond in rule.get("conditions", []):
        if cond.get("key") == "state_current":
            current_state_opts.extend(split_csv_values(cond.get("value")))
    deduped: List[str] = []
    for s in current_state_opts:
        if s not in deduped:
            deduped.append(s)
    return deduped or []


def choose_precondition(rule: Dict[str, Any]) -> List[str]:
    opts: List[str] = []
    for cond in rule.get("preconditions", []):
        if cond.get("key") == "state_current":
            opts.extend(split_csv_values(cond.get("value")))
    deduped: List[str] = []
    for s in opts:
        if s not in deduped:
            deduped.append(s)
    return deduped


def state_constraints(state: str) -> Dict[str, Any]:
    s = normalize_state(state)
    # Substates inherit from base then apply a trigger.
    if s in SUBSTATE_TRIGGER:
        if s.startswith("home_sleeping_lying"):
            base = "home_sleeping"
        elif s.startswith("home_morning_workday_lying"):
            base = "home_morning_workday"
        elif s.startswith("home_morning_rest_lying"):
            base = "home_morning_rest"
        elif s.startswith("home_daytime_workday_"):
            base = "home_daytime_workday"
        elif s.startswith("home_daytime_rest_"):
            base = "home_daytime_rest"
        elif s.startswith("home_evening_"):
            base = "home_evening"
        elif s.startswith("office_working_"):
            base = "office_working"
        elif s == "at_cafe_quiet":
            base = "at_cafe"
        elif s in {"at_education_class", "at_education_break"}:
            base = "at_education"
        elif s == "at_health_inpatient":
            base = "at_health"
        elif s.startswith("unknown_"):
            base = "stationary_unknown"
        else:
            base = s
        base_c = state_constraints(base)
        trigger_field, trigger_val = SUBSTATE_TRIGGER[s]
        base_c["substate_trigger"] = {trigger_field: trigger_val}
        return base_c

    # Canonical base states.
    if s == "home_sleeping":
        return dict(location=["home"], motion=["stationary"], time=["sleeping", "late_night"], day=DAY_TYPES, transport=["stationary"], activity=["sleeping"])
    if s == "home_morning_workday":
        return dict(location=["home"], motion=["stationary", "walking"], time=["dawn", "morning"], day=["workday"], transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "home_morning_rest":
        return dict(location=["home"], motion=["stationary", "walking"], time=["dawn", "morning"], day=["weekend", "holiday"], transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "home_daytime_workday":
        return dict(location=["home"], motion=["stationary", "walking"], time=["forenoon", "lunch", "afternoon"], day=["workday"], transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "home_daytime_rest":
        return dict(location=["home"], motion=["stationary", "walking"], time=["forenoon", "lunch", "afternoon"], day=["weekend", "holiday"], transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "home_evening":
        return dict(location=["home"], motion=["stationary", "walking"], time=["evening", "night"], day=DAY_TYPES, transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "home_active":
        return dict(location=["home"], motion=["running", "cycling"], time=TIME_SLOTS, day=DAY_TYPES, transport=["running", "cycling"], activity=["active"])

    if s == "office_arriving":
        return dict(location=["work"], motion=["stationary", "walking"], time=["dawn", "morning"], day=["workday"], transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "office_lunch_break":
        return dict(location=["work"], motion=["stationary", "walking"], time=["lunch"], day=["workday"], transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "office_working":
        return dict(location=["work"], motion=["stationary", "walking"], time=["forenoon", "afternoon"], day=["workday"], transport=["stationary", "walking"], activity=["sitting", "standing"])
    if s == "office_overtime":
        return dict(location=["work"], motion=["stationary", "walking"], time=["evening", "night"], day=["workday"], transport=["stationary", "walking"], activity=["sitting", "standing"])
    if s == "office_late_overtime":
        return dict(location=["work"], motion=["stationary", "walking"], time=["sleeping", "late_night"], day=["workday"], transport=["stationary", "walking"], activity=["sitting", "standing"])
    if s == "office_rest_day":
        return dict(location=["work"], motion=["stationary", "walking"], time=TIME_SLOTS, day=["weekend", "holiday"], transport=["stationary", "walking"], activity=["sitting", "standing"])

    if s == "commuting_walk_out":
        return dict(location=["en_route"], motion=["walking"], time=["dawn", "morning"], day=["workday"], transport=["walking"], activity=["active", "standing"])
    if s == "commuting_walk_home":
        return dict(location=["en_route"], motion=["walking"], time=["evening", "night"], day=["workday"], transport=["walking"], activity=["active", "standing"])
    if s == "commuting_cycle_out":
        return dict(location=["en_route"], motion=["cycling"], time=["dawn", "morning"], day=["workday"], transport=["cycling"], activity=["active"])
    if s == "commuting_cycle_home":
        return dict(location=["en_route"], motion=["cycling"], time=["evening", "night"], day=["workday"], transport=["cycling"], activity=["active"])
    if s == "commuting_drive_out":
        return dict(location=["en_route"], motion=["driving"], time=["dawn", "morning"], day=["workday"], transport=["driving"], activity=["sitting"])
    if s == "commuting_drive_home":
        return dict(location=["en_route"], motion=["driving"], time=["evening", "night"], day=["workday"], transport=["driving"], activity=["sitting"])
    if s == "commuting_transit_out":
        return dict(location=["en_route"], motion=["transit"], time=["dawn", "morning"], day=["workday"], transport=["transit"], activity=["standing", "sitting"])
    if s == "commuting_transit_home":
        return dict(location=["en_route"], motion=["transit"], time=["evening", "night"], day=["workday"], transport=["transit"], activity=["standing", "sitting"])
    if s == "driving":
        return dict(location=["en_route"], motion=["driving"], time=TIME_SLOTS, day=DAY_TYPES, transport=["driving"], activity=["sitting"])
    if s == "in_transit":
        return dict(location=["en_route", "transit"], motion=["transit"], time=TIME_SLOTS, day=DAY_TYPES, transport=["transit"], activity=["standing", "sitting"])

    if s == "at_metro":
        return dict(location=["metro"], motion=["stationary", "walking", "transit"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking", "transit"], activity=["standing"])
    if s == "at_rail_station":
        return dict(location=["rail_station"], motion=["stationary", "walking", "transit"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking", "transit"], activity=["standing"])
    if s == "at_airport":
        return dict(location=["airport"], motion=["stationary", "walking", "transit"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking", "transit"], activity=["standing"])
    if s == "at_transit_hub":
        return dict(location=["transit"], motion=["stationary", "walking", "driving"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking", "driving"], activity=["standing"])

    if s == "outdoor_walking":
        return dict(location=["outdoor"], motion=["walking"], time=TIME_SLOTS, day=DAY_TYPES, transport=["walking"], activity=["active", "standing"])
    if s == "outdoor_running":
        return dict(location=["outdoor", "unknown"], motion=["running"], time=TIME_SLOTS, day=DAY_TYPES, transport=["running"], activity=["active"])
    if s == "outdoor_cycling":
        return dict(location=["outdoor", "unknown"], motion=["cycling"], time=TIME_SLOTS, day=DAY_TYPES, transport=["cycling"], activity=["active"])
    if s == "outdoor_resting":
        return dict(location=["outdoor"], motion=["stationary"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary"], activity=["standing", "sitting"])

    if s == "at_restaurant_lunch":
        return dict(location=["restaurant"], motion=["stationary", "walking"], time=["lunch"], day=DAY_TYPES, transport=["stationary", "walking"], activity=["sitting", "standing"])
    if s == "at_restaurant_dinner":
        return dict(location=["restaurant"], motion=["stationary", "walking"], time=["evening", "night"], day=DAY_TYPES, transport=["stationary", "walking"], activity=["sitting", "standing"])
    if s == "at_restaurant_other":
        return dict(location=["restaurant"], motion=["stationary", "walking"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking"], activity=["sitting", "standing"])

    if s == "at_cafe":
        return dict(location=["cafe"], motion=["stationary", "walking"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking"], activity=["sitting", "standing"])
    if s == "at_gym_exercising":
        return dict(location=["gym"], motion=["running", "cycling"], time=TIME_SLOTS, day=DAY_TYPES, transport=["running", "cycling"], activity=["active"])
    if s == "at_gym":
        return dict(location=["gym"], motion=["stationary", "walking"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "at_shopping":
        return dict(location=["shopping"], motion=["stationary", "walking"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "at_health":
        return dict(location=["health"], motion=["stationary", "walking"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "at_social":
        return dict(location=["social"], motion=["stationary", "walking"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "at_education":
        return dict(location=["education"], motion=["stationary", "walking"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking"], activity=["standing", "sitting"])
    if s == "at_custom":
        return dict(location=["custom"], motion=["stationary", "walking"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "walking"], activity=["standing", "sitting"])

    if s == "stationary_unknown":
        return dict(location=["unknown"], motion=["stationary", "unknown"], time=TIME_SLOTS, day=DAY_TYPES, transport=["stationary", "unknown"], activity=["standing", "sitting", "unknown"])
    if s == "walking_unknown":
        return dict(location=["unknown"], motion=["walking"], time=TIME_SLOTS, day=DAY_TYPES, transport=["walking"], activity=["standing", "active"])

    return dict(location=["unknown"], motion=["unknown"], time=TIME_SLOTS, day=DAY_TYPES, transport=["unknown"], activity=["unknown"])


def valid_phone_choices(state: str) -> List[str]:
    s = normalize_state(state)
    if s in SUBSTATE_TRIGGER and SUBSTATE_TRIGGER[s][0] == "phone":
        return [SUBSTATE_TRIGGER[s][1]]
    base = s
    if s in SUBSTATE_TRIGGER:
        base = state_constraints(s).get("base", s)
    avoid = {v for f, v in BASE_AVOID_TRIGGERS.get(s, set()) if f == "phone"}
    loc = state_constraints(s)["location"][0]
    if loc == "work":
        options = ["in_use", "on_desk", "face_up", "in_pocket"]
    elif loc == "home":
        options = ["in_use", "face_up", "on_desk", "in_pocket"]
    elif loc in {"restaurant", "shopping", "outdoor", "en_route", "airport", "rail_station", "metro", "transit", "social"}:
        options = ["in_pocket", "in_use", "face_up"]
    elif loc in {"cafe", "education", "health"}:
        options = ["in_use", "on_desk", "face_up", "in_pocket"]
    else:
        options = ["unknown", "in_use", "face_up"]
    return [x for x in options if x not in avoid] or [x for x in PHONE_CATEGORIES if x not in avoid]


def valid_sound_choices(state: str) -> List[str]:
    s = normalize_state(state)
    if s in SUBSTATE_TRIGGER and SUBSTATE_TRIGGER[s][0] == "sound":
        return [SUBSTATE_TRIGGER[s][1]]
    avoid = {v for f, v in BASE_AVOID_TRIGGERS.get(s, set()) if f == "sound"}
    loc = state_constraints(s)["location"][0]
    if loc == "cafe":
        options = ["normal", "noisy"]  # avoid quiet for base at_cafe
    elif loc in {"restaurant", "shopping", "airport", "rail_station", "metro", "transit", "social"}:
        options = ["normal", "noisy"]
    elif loc in {"work", "education"}:
        options = ["quiet", "normal", "silent"]
    elif loc == "home":
        options = ["quiet", "normal", "silent"]
    elif loc == "unknown":
        options = ["normal", "quiet", "unknown"]
    else:
        options = ["normal", "quiet", "unknown"]
    return [x for x in options if x not in avoid] or [x for x in SOUND_CATEGORIES if x not in avoid]


def valid_light_choices(state: str) -> List[str]:
    s = normalize_state(state)
    if s in SUBSTATE_TRIGGER and SUBSTATE_TRIGGER[s][0] == "light":
        return [SUBSTATE_TRIGGER[s][1]]
    avoid = {v for f, v in BASE_AVOID_TRIGGERS.get(s, set()) if f == "light"}
    loc = state_constraints(s)["location"][0]
    if loc == "home":
        if s.startswith("home_evening") or s.startswith("home_sleeping"):
            options = ["dim", "normal"]
        elif s.startswith("home_daytime"):
            options = ["normal", "bright"]
        else:
            options = ["dim", "normal", "bright"]
    elif loc in {"work", "education", "health"}:
        options = ["normal", "bright"]
    elif loc in {"cafe", "restaurant", "shopping", "social"}:
        options = ["dim", "normal", "bright"]
    elif loc in {"outdoor", "airport", "rail_station", "metro", "transit", "en_route"}:
        options = ["normal", "bright"]
    elif loc == "unknown":
        options = ["dim", "normal"]
    else:
        options = ["dim", "normal", "bright"]
    return [x for x in options if x not in avoid] or [x for x in LIGHT_CATEGORIES if x not in avoid]


def motion_activity_candidates(motion: str, state: str) -> List[str]:
    c = state_constraints(state)
    if motion == "stationary":
        preferred = ["sleeping", "sitting", "standing"]
    elif motion == "walking":
        preferred = ["standing", "active"]
    elif motion in {"running", "cycling"}:
        preferred = ["active"]
    elif motion in {"driving", "transit"}:
        preferred = ["sitting", "standing"]
    else:
        preferred = list(c["activity"])
    out = [x for x in preferred if x in c["activity"]]
    return out or list(c["activity"])


def enforce_motion_alignment(features: Dict[str, Any], state: str, rng: random.Random) -> None:
    c = state_constraints(state)
    motion = features["ps_motion"]
    mapped_transport = motion if motion in TRANSPORT_MODES else "unknown"
    if mapped_transport in c["transport"]:
        features["transportMode"] = mapped_transport
    else:
        features["transportMode"] = rng.choice(c["transport"])
    act_candidates = motion_activity_candidates(motion, state)
    features["activityState"] = rng.choice(act_candidates)


def enforce_explicit_rule_conditions(features: Dict[str, Any], spec: ScenarioSpec, state: str, rng: random.Random) -> None:
    # apply direct rule restrictions that should affect sampled context
    c = state_constraints(state)
    for cond in spec.conditions:
        key = cond.get("key")
        op = cond.get("op")
        value = cond.get("value")

        if key == "ps_motion":
            allowed = [str(value)] if op == "eq" else [str(x).strip() for x in str(value).split(",") if str(x).strip()]
            allowed = [x for x in allowed if x in c["motion"]]
            if allowed:
                features["ps_motion"] = rng.choice(allowed)

        elif key == "ps_dayType":
            allowed = [str(value)] if op == "eq" else [str(x).strip() for x in str(value).split(",") if str(x).strip()]
            allowed = [x for x in allowed if x in c["day"]]
            if allowed:
                features["ps_dayType"] = rng.choice(allowed)

        elif key == "ps_time":
            allowed = [str(value)] if op == "eq" else [str(x).strip() for x in str(value).split(",") if str(x).strip()]
            allowed = [x for x in allowed if x in c["time"]]
            if allowed:
                features["ps_time"] = rng.choice(allowed)
                features["hour"] = rng.choice(TIME_SLOT_HOURS[features["ps_time"]])

        elif key == "ps_sound":
            allowed = [str(value)] if op == "eq" else [str(x).strip() for x in str(value).split(",") if str(x).strip()]
            allowed = [x for x in allowed if x in valid_sound_choices(state)]
            if allowed:
                features["ps_sound"] = rng.choice(allowed)

        elif key == "ps_phone":
            allowed = [str(value)] if op == "eq" else [str(x).strip() for x in str(value).split(",") if str(x).strip()]
            allowed = [x for x in allowed if x in valid_phone_choices(state)]
            if allowed:
                features["ps_phone"] = rng.choice(allowed)

        elif key == "ps_light":
            allowed = [str(value)] if op == "eq" else [str(x).strip() for x in str(value).split(",") if str(x).strip()]
            allowed = [x for x in allowed if x in valid_light_choices(state)]
            if allowed:
                features["ps_light"] = rng.choice(allowed)

        elif key == "wifiLost":
            features["wifiLost"] = 1 if str(value).lower() == "true" else 0
        elif key == "wifiLostCategory":
            candidate = str(value)
            if candidate in LOCATION_CATEGORIES:
                features["wifiLostCategory"] = candidate
        elif key == "cal_hasUpcoming":
            features["cal_hasUpcoming"] = 1 if str(value).lower() == "true" else 0
        elif key == "cal_inMeeting":
            features["cal_inMeeting"] = 1 if str(value).lower() == "true" else 0
        elif key == "cal_nextLocation" and op == "eq":
            loc = str(value).strip().lower()
            if loc in LOCATION_CATEGORIES:
                features["cal_nextLocation"] = loc
        elif key == "state_duration_sec":
            iv = int(value)
            if op == "gte":
                features["state_duration_sec"] = max(int(features["state_duration_sec"]), iv)
            elif op == "gt":
                features["state_duration_sec"] = max(int(features["state_duration_sec"]), iv + 1)
            elif op == "lte":
                features["state_duration_sec"] = min(int(features["state_duration_sec"]), iv)
            elif op == "lt":
                features["state_duration_sec"] = min(int(features["state_duration_sec"]), iv - 1)
        elif key == "cal_eventCount":
            iv = int(value)
            if op == "gte":
                features["cal_eventCount"] = max(int(features["cal_eventCount"]), iv)
            elif op == "gt":
                features["cal_eventCount"] = max(int(features["cal_eventCount"]), iv + 1)
            elif op == "lte":
                features["cal_eventCount"] = min(int(features["cal_eventCount"]), iv)
            elif op == "lt":
                features["cal_eventCount"] = min(int(features["cal_eventCount"]), iv - 1)
        elif key == "cal_nextMinutes":
            iv = int(value)
            cur = int(features.get("__latent_cal_nextMinutes", rng.randint(0, 240)))
            if op == "gte":
                features["__latent_cal_nextMinutes"] = max(cur, iv)
            elif op == "gt":
                features["__latent_cal_nextMinutes"] = max(cur, iv + 1)
            elif op == "lte":
                features["__latent_cal_nextMinutes"] = min(cur, iv)
            elif op == "lt":
                features["__latent_cal_nextMinutes"] = min(cur, iv - 1)

    enforce_motion_alignment(features, state, rng)


def choose_hour_and_time(spec: ScenarioSpec, state: str, rng: random.Random) -> Tuple[int, str]:
    c = state_constraints(state)
    allowed_slots = list(c["time"])
    # Optional explicit hour restrictions from rules.
    explicit_hours = set(range(24))
    gtes = [int(x["value"]) for x in spec.conditions if x.get("key") == "hour" and x.get("op") == "gte" and str(x.get("value")).isdigit()]
    ltes = [int(x["value"]) for x in spec.conditions if x.get("key") == "hour" and x.get("op") == "lte" and str(x.get("value")).isdigit()]
    if gtes:
        explicit_hours = {h for h in explicit_hours if h >= max(gtes)}
    if ltes:
        explicit_hours = {h for h in explicit_hours if h <= min(ltes)}
    candidates: List[Tuple[int, str]] = []
    for slot in allowed_slots:
        for h in TIME_SLOT_HOURS[slot]:
            if h in explicit_hours:
                candidates.append((h, slot))
    if not candidates:
        for slot in allowed_slots:
            for h in TIME_SLOT_HOURS[slot]:
                candidates.append((h, slot))
    hour, slot = rng.choice(candidates)
    return hour, slot


def choose_profile(rng: random.Random) -> Dict[str, Any]:
    return {
        "user_id_hash_bucket": rng.choice(USER_BUCKETS),
        "age_bucket": rng.choices(AGE_BUCKETS[:-1], weights=[0.03, 0.10, 0.28, 0.22, 0.17, 0.13, 0.07])[0],
        "sex": rng.choices(SEX_BUCKETS, weights=[0.48, 0.48, 0.02, 0.02])[0],
        "has_kids": rng.choices([0, 1], weights=[0.62, 0.38])[0],
    }


def choose_network(location: str, rng: random.Random) -> str:
    if location in {"home", "work", "cafe", "education", "health"}:
        return rng.choices(NETWORK_TYPES, weights=[0.80, 0.18, 0.02])[0]
    if location in {"airport", "rail_station", "metro", "transit", "en_route", "outdoor"}:
        return rng.choices(NETWORK_TYPES, weights=[0.15, 0.80, 0.05])[0]
    return rng.choices(NETWORK_TYPES, weights=[0.40, 0.55, 0.05])[0]


def choose_battery_and_charging(location: str, hour: int, phone: str, rng: random.Random) -> Tuple[int, int]:
    is_charging = 1 if phone == "charging" else 0
    if not is_charging and location in {"home", "work"} and hour in {0, 1, 2, 3, 4, 5, 6, 22, 23} and rng.random() < 0.25:
        is_charging = 1
    level = rng.randint(35, 95)
    if is_charging:
        level = max(level, 55)
    return level, is_charging


def pick_ranked_label(items: List[str], rng: random.Random) -> str:
    if not items:
        raise ValueError("Expected non-empty ranked item list")
    if len(items) == 1:
        return items[0]
    if rng.random() < 0.5:
        return items[0]
    return rng.choice(items[1:])


def activity_duration_for_state(state: str, rng: random.Random) -> int:
    m = state_constraints(state)["motion"]
    if "running" in m or "cycling" in m:
        return rng.randint(120, 1800)
    if "walking" in m:
        return rng.randint(60, 1800)
    return rng.randint(0, 7200)


def state_duration_for_state(state: str, rng: random.Random) -> int:
    s = normalize_state(state)
    if s in SUBSTATE_TRIGGER:
        return rng.randint(600, 7200)
    return rng.randint(0, 7200)


def choose_precondition_value(spec: ScenarioSpec, rng: random.Random) -> str:
    opts = choose_precondition({"preconditions": spec.preconditions})
    return rng.choice(opts) if opts else "none"


def apply_rule_overrides(features: Dict[str, Any], spec: ScenarioSpec, rng: random.Random) -> None:
    # explicit condition overrides from rules
    for cond in spec.conditions:
        key = cond.get("key")
        value = cond.get("value")
        if key == "wifiLost":
            features["wifiLost"] = 1 if str(value).lower() == "true" else 0
        elif key == "wifiLostCategory":
            candidate = str(value)
            if candidate in LOCATION_CATEGORIES:
                features["wifiLostCategory"] = candidate
        elif key == "cal_hasUpcoming":
            features["cal_hasUpcoming"] = 1 if str(value).lower() == "true" else 0
        elif key == "cal_inMeeting":
            features["cal_inMeeting"] = 1 if str(value).lower() == "true" else 0
        elif key == "cal_eventCount" and str(value).isdigit():
            features["cal_eventCount"] = int(value)
        elif key == "cal_nextLocation":
            loc = str(value).strip().lower()
            if loc in LOCATION_CATEGORIES:
                features["cal_nextLocation"] = loc
        elif key in SMS_FIELDS:
            features[key] = 1 if str(value).lower() == "true" else 0

    sid = spec.scenario_id
    # scenario-specific realism
    if sid == "OFFICE_LUNCH_OUT":
        features["wifiLost"] = 1
        features["wifiLostCategory"] = "work"
        features["ps_time"] = "lunch"
        features["hour"] = rng.choice(TIME_SLOT_HOURS["lunch"])
    elif sid == "LEAVE_OFFICE":
        features["wifiLost"] = 1
        features["wifiLostCategory"] = "work"
    elif sid == "IN_MEETING":
        features["cal_inMeeting"] = 1
        features["cal_hasUpcoming"] = 1
        features["cal_eventCount"] = max(features["cal_eventCount"], 1)
        if features["cal_nextLocation"] == "unknown":
            features["cal_nextLocation"] = "work"
    elif sid == "REMOTE_MEETING":
        features["cal_hasUpcoming"] = 1
        features["cal_eventCount"] = max(features["cal_eventCount"], 1)
        if features["cal_nextLocation"] == "unknown":
            features["cal_nextLocation"] = rng.choice(["work", "cafe", "education", "custom"])
    elif sid == "MEETING_UPCOMING":
        features["cal_hasUpcoming"] = 1
        features["cal_eventCount"] = max(features["cal_eventCount"], 1)
        if features["cal_nextLocation"] == "unknown":
            features["cal_nextLocation"] = "work"
    elif sid == "NO_MEETINGS":
        features["cal_hasUpcoming"] = 0
        features["cal_eventCount"] = 0
    elif sid == "CAL_HAS_EVENTS":
        features["cal_hasUpcoming"] = 1
        features["cal_eventCount"] = max(features["cal_eventCount"], 2)
        if features["cal_nextLocation"] == "unknown":
            features["cal_nextLocation"] = rng.choice(["work", "cafe", "custom"])
    elif sid == "DELIVERY_AT_OFFICE":
        features["sms_delivery_pending"] = 1
        features["cal_nextLocation"] = features["cal_nextLocation"] if features["cal_nextLocation"] != "unknown" else "work"
    elif sid == "TRAIN_DEPARTURE":
        features["sms_train_pending"] = 1
        features["cal_nextLocation"] = "rail_station"
    elif sid == "FLIGHT_BOARDING":
        features["sms_flight_pending"] = 1
        features["cal_nextLocation"] = "airport"
    elif sid == "HOTEL_CHECKIN":
        features["sms_hotel_pending"] = 1
    elif sid == "MOVIE_TICKET":
        features["sms_movie_pending"] = 1
    elif sid == "HOSPITAL_APPOINTMENT":
        features["sms_hospital_pending"] = 1
        if features["cal_nextLocation"] == "unknown":
            features["cal_nextLocation"] = "health"
    elif sid == "RIDESHARE_PICKUP":
        features["sms_ride_pending"] = 1

    if features["wifiLost"] == 0:
        features["wifiLostCategory"] = "unknown"


def _cond_value_to_python(key: str, value: Any) -> Any:
    if key in {"wifiLost", "cal_hasUpcoming", "cal_inMeeting"} or key in SMS_FIELDS:
        return 1 if str(value).lower() == "true" else 0
    if key == "hour" or key == "cal_eventCount":
        try:
            return int(value)
        except Exception:
            return value
    return normalize_state(value) if key in {"state_current", "precondition"} else str(value)


def validate_rule_conditions(row: Dict[str, Any], spec: ScenarioSpec) -> List[str]:
    f = row["features"]
    errors: List[str] = []
    for cond in spec.conditions:
        key = cond.get("key")
        op = cond.get("op")
        value = cond.get("value")
        actual = f.get(key)
        if key == "cal_nextMinutes":
            actual = f.get("__latent_cal_nextMinutes")
        if key == "state_current":
            actual = normalize_state(actual)
        if op == "eq":
            expected = _cond_value_to_python(key, value)
            if actual != expected:
                errors.append(f"rule eq failed: {key}={actual} != {expected}")
        elif op == "in":
            allowed = [_cond_value_to_python(key, x) for x in str(value).split(",") if str(x).strip()]
            if actual not in allowed:
                errors.append(f"rule in failed: {key}={actual} not in {allowed}")
        elif op == "gte":
            if int(actual) < int(value):
                errors.append(f"rule gte failed: {key}={actual} < {value}")
        elif op == "gt":
            if int(actual) <= int(value):
                errors.append(f"rule gt failed: {key}={actual} <= {value}")
        elif op == "lte":
            if int(actual) > int(value):
                errors.append(f"rule lte failed: {key}={actual} > {value}")
        elif op == "lt":
            if int(actual) >= int(value):
                errors.append(f"rule lt failed: {key}={actual} >= {value}")
        elif op == "neq":
            # In this project, neq "" on cal_nextLocation means a meaningful non-empty location.
            if key == "cal_nextLocation" and (actual in {"", "unknown", None}):
                errors.append("rule neq failed: cal_nextLocation is empty/unknown")
            elif str(actual) == str(value):
                errors.append(f"rule neq failed: {key} == {value}")
    # precondition membership
    allowed_pre = choose_precondition({"preconditions": spec.preconditions})
    if allowed_pre:
        if f["precondition"] not in allowed_pre:
            errors.append(f"precondition {f['precondition']} not in {allowed_pre}")
    else:
        if f["precondition"] != "none":
            errors.append(f"precondition should be none, got {f['precondition']}")
    return errors


def validate_state_alignment(row: Dict[str, Any]) -> List[str]:
    f = row["features"]
    s = normalize_state(f["state_current"])
    c = state_constraints(s)
    errors: List[str] = []
    if f["ps_location"] not in c["location"]:
        errors.append(f"ps_location={f['ps_location']} not in {c['location']}")
    if f["ps_motion"] not in c["motion"]:
        errors.append(f"ps_motion={f['ps_motion']} not in {c['motion']}")
    if f["ps_time"] not in c["time"]:
        errors.append(f"ps_time={f['ps_time']} not in {c['time']}")
    if f["ps_dayType"] not in c["day"]:
        errors.append(f"ps_dayType={f['ps_dayType']} not in {c['day']}")
    if f["transportMode"] not in c["transport"]:
        errors.append(f"transportMode={f['transportMode']} not in {c['transport']}")
    if f["activityState"] not in c["activity"]:
        errors.append(f"activityState={f['activityState']} not in {c['activity']}")
    if f["hour"] not in TIME_SLOT_HOURS[f["ps_time"]]:
        errors.append(f"hour={f['hour']} incompatible with ps_time={f['ps_time']}")
    # validate chosen categorical supports
    if f["ps_phone"] not in valid_phone_choices(s):
        errors.append(f"ps_phone={f['ps_phone']} not valid for {s}")
    if f["ps_sound"] not in valid_sound_choices(s):
        errors.append(f"ps_sound={f['ps_sound']} not valid for {s}")
    if f["ps_light"] not in valid_light_choices(s):
        errors.append(f"ps_light={f['ps_light']} not valid for {s}")
    trig = c.get("substate_trigger")
    if trig:
        for k, v in trig.items():
            got = f.get(f"ps_{k}", f.get(k))
            if got != v:
                errors.append(f"substate trigger {k}={v} required but got {got}")
        if f["state_duration_sec"] < 600:
            errors.append("substate requires state_duration_sec >= 600")
    else:
        # Ensure base state does not accidentally satisfy a refined trigger that should have produced a substate
        for field, disallowed in BASE_AVOID_TRIGGERS.get(s, set()):
            got = f.get(f"ps_{field}", f.get(field))
            if got == disallowed:
                errors.append(f"base state {s} should avoid trigger {field}={disallowed}")
    return errors


def generate_firststep_row(spec: ScenarioSpec, episode_idx: int, rng: random.Random, action_type_by_id: Dict[str, str], include_debug: bool=False) -> Dict[str, Any]:
    state_options = choose_current_state({"conditions": spec.conditions})
    if not state_options:
        raise ValueError(f"Scenario {spec.scenario_id} has no state_current options in rules")
    chosen_state = rng.choice(state_options)
    chosen_precondition = choose_precondition_value(spec, rng)

    hour, ps_time = choose_hour_and_time(spec, chosen_state, rng)
    minute = rng.randint(0, 59)
    second = rng.choice([0,5,10,15,20,25,30,35,40,45,50,55])
    timestep = hour * 3600 + minute * 60 + second

    c = state_constraints(chosen_state)
    ps_location = rng.choice(c["location"])
    ps_motion = rng.choice(c["motion"])
    ps_dayType = rng.choice(c["day"])

    phone_choices = valid_phone_choices(chosen_state)
    ps_phone = rng.choice(phone_choices)
    sound_choices = valid_sound_choices(chosen_state)
    ps_sound = rng.choice(sound_choices)
    light_choices = valid_light_choices(chosen_state)
    ps_light = rng.choice(light_choices)

    battery, is_charging = choose_battery_and_charging(ps_location, hour, ps_phone, rng)
    networkType = choose_network(ps_location, rng)
    profile = choose_profile(rng)

    features: Dict[str, Any] = {
        "precondition": chosen_precondition,
        "state_current": chosen_state,
        "state_duration_sec": state_duration_for_state(chosen_state, rng),
        "ps_time": ps_time,
        "hour": hour,
        "cal_hasUpcoming": 1 if spec.category in {"work", "education", "travel"} else 0,
        "ps_dayType": ps_dayType,
        "ps_motion": ps_motion,
        "wifiLost": 0,
        "wifiLostCategory": "unknown",
        "cal_eventCount": 0 if spec.scenario_id == "NO_MEETINGS" else rng.randint(0, 4),
        "cal_inMeeting": 0,
        "cal_nextLocation": "unknown",
        "__latent_cal_nextMinutes": rng.randint(0, 240),
        "ps_sound": ps_sound,
        "timestep": timestep,
        "ps_location": ps_location,
        "ps_phone": ps_phone,
        "ps_light": ps_light,
        "batteryLevel": battery,
        "isCharging": is_charging,
        "networkType": networkType,
        "transportMode": "unknown",
        "activityState": "unknown",
        "activityDuration": activity_duration_for_state(chosen_state, rng),
        **profile,
    }
    enforce_motion_alignment(features, chosen_state, rng)
    for sms_field in SMS_FIELDS:
        features[sms_field] = 0

    apply_rule_overrides(features, spec, rng)
    enforce_explicit_rule_conditions(features, spec, chosen_state, rng)

    gt_ro = pick_ranked_label(spec.default_action_ids, rng)
    gt_app = pick_ranked_label(spec.default_app_categories, rng)

    row = {
        "episode_id": f"{spec.scenario_id.lower()}_ep{episode_idx:04d}",
        "scenario_id": spec.scenario_id,
        "scenario_name": spec.scenario_name,
        "t_in_scenario_sec": 0,
        "dt_sec": None,
        "features": features,
        "gt_ro": gt_ro,
        "gt_ro_type": action_type_by_id.get(gt_ro, "unknown"),
        "gt_app": gt_app,
    }
    errors = validate_state_alignment(row) + validate_rule_conditions(row, spec)
    if errors:
        raise ValueError(f"Alignment failed for {spec.scenario_id}: {'; '.join(errors)}")
    return row


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    ignored = {s for s in args.ignore_trigger_scenarios.split(",") if s}
    requested = {s for s in args.scenario_ids.split(",") if s} if args.scenario_ids else None

    rules_by_sid = parse_rules(args.rules)
    scenario_defaults, action_type_by_id = parse_global_action_space(args.global_action_space)
    specs = build_scenario_specs(rules_by_sid, scenario_defaults)

    included_specs: List[ScenarioSpec] = []
    ignored_specs: List[str] = []
    for sid, spec in sorted(specs.items()):
        if requested and sid not in requested:
            continue
        if sid in ignored:
            ignored_specs.append(sid)
            continue
        included_specs.append(spec)

    rows: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    validation_summary: Dict[str, Any] = {"checked_rows": 0, "checked_scenarios": [], "all_passed": True}
    for spec in included_specs:
        for i in range(1, args.episodes_per_scenario + 1):
            include_debug = args.debug_examples is not None and len(examples) < 12
            row = generate_firststep_row(spec, i, rng, action_type_by_id, include_debug=include_debug)
            validation_summary["checked_rows"] += 1
            if spec.scenario_id not in validation_summary["checked_scenarios"]:
                validation_summary["checked_scenarios"].append(spec.scenario_id)
            rows.append(row)
            if include_debug:
                examples.append(row)

    args.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    metadata = {
        "rules_source": str(args.rules),
        "global_action_space_source": str(args.global_action_space),
        "episodes_per_scenario": args.episodes_per_scenario,
        "ignored_trigger_scenarios": sorted(list(ignored)),
        "included_scenarios": [s.scenario_id for s in included_specs],
        "ignored_scenarios": ignored_specs,
        "rows_written": len(rows),
        "gt_policy": {
            "ro_default_prob": 0.5,
            "ro_remaining_policy": "uniform over remaining ranked actions",
            "app_default_prob": 0.5,
            "app_remaining_policy": "uniform over remaining ranked app categories",
        },
        "state_alignment": {
            "enabled": True,
            "description": "ps_location, ps_motion, ps_time, ps_dayType, transportMode, activityState, ps_phone/ps_sound/ps_light plus transport/activity are generated to align with state_current base/substate logic; all scenarios are included by default unless manually excluded via --ignore-trigger-scenarios",
        },
        "validation_summary": validation_summary,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.debug_examples:
        args.debug_examples.write_text("\n".join(json.dumps(r, ensure_ascii=False, indent=2) for r in examples), encoding="utf-8")


if __name__ == "__main__":
    main()
