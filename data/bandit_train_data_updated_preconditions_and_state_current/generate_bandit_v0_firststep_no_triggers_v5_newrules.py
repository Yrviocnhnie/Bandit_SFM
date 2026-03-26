#!/usr/bin/env python3
"""Generate synthetic v0/v1 bandit samples using the updated rules file with added preconditions and state_current.

Current mode kept from the approved pipeline:
- ignore trigger-based scenarios via a maintained ignore list
- emit exactly one row per episode (first prediction timestep only)
- v0 feature set by default (no history features)
- include precondition
- exclude transportMode
- exclude cal_nextMinutes
- use v5 global action space for GT labels
- choose GT labels with 50% probability on the first ranked item and the remaining 50% spread uniformly across the remaining ranked items
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
LOCATION_CATEGORIES = [
    "home", "work", "restaurant", "cafe", "gym", "metro", "rail_station", "airport",
    "transit", "shopping", "outdoor", "health", "social", "education", "custom", "en_route", "unknown",
]
MOTION_CATEGORIES = ["stationary", "walking", "running", "cycling", "driving", "transit", "unknown"]
PHONE_CATEGORIES = ["in_use", "holding_lying", "on_desk", "face_up", "in_pocket", "face_down", "charging", "unknown"]
SOUND_CATEGORIES = ["silent", "quiet", "normal", "noisy", "unknown"]
DAY_TYPES = ["workday", "weekend", "holiday"]
NETWORK_TYPES = ["wifi", "cellular", "none"]
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
DEFAULT_IGNORED_TRIGGER_SCENARIOS = {
    "OFFICE_AFTERNOON", "WEEKEND_OVERTIME", "HOME_EVENING", "GYM_WORKOUT"
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

@dataclass
class ScenarioPlan:
    scenario_id: str
    scenario_name: str
    category: str
    time_window: str
    state_options: List[str]
    precondition_options: List[str]
    notes: str

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--actions-md", type=Path, required=False, default=None)
    p.add_argument("--global-action-space", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--plan-output", type=Path, default=None)
    p.add_argument("--episodes-per-scenario", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scenario-ids", type=str, default="")
    p.add_argument("--ignore-trigger-scenarios", type=str, default=",".join(sorted(DEFAULT_IGNORED_TRIGGER_SCENARIOS)))
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

def infer_location_from_state(state: str) -> str:
    state = normalize_state(state)
    if state.startswith("home_"):
        return "home"
    if state.startswith("office_"):
        return "work"
    if state.startswith("commuting_"):
        return "en_route"
    if state in {"driving", "in_transit"}:
        return "en_route"
    if state.startswith("at_restaurant_"):
        return "restaurant"
    if state.startswith("at_cafe"):
        return "cafe"
    if state.startswith("at_gym"):
        return "gym"
    if state.startswith("at_metro"):
        return "metro"
    if state.startswith("at_rail_station"):
        return "rail_station"
    if state.startswith("at_airport"):
        return "airport"
    if state.startswith("at_transit_hub"):
        return "transit"
    if state.startswith("at_shopping"):
        return "shopping"
    if state.startswith("at_health"):
        return "health"
    if state.startswith("at_social"):
        return "social"
    if state.startswith("at_education"):
        return "education"
    if state.startswith("at_custom"):
        return "custom"
    if state.startswith("outdoor_"):
        return "outdoor"
    if state.startswith("stationary_unknown") or state.startswith("walking_unknown") or state.startswith("unknown_"):
        return "unknown"
    return "unknown"

def infer_motion_from_state(state: str) -> str:
    state = normalize_state(state)
    if "sleeping" in state:
        return "stationary"
    if state.startswith("commuting_walk") or state == "outdoor_walking" or state == "walking_unknown":
        return "walking"
    if state.startswith("commuting_cycle") or state == "outdoor_cycling":
        return "cycling"
    if state.startswith("commuting_drive") or state == "driving":
        return "driving"
    if state.startswith("commuting_transit") or state == "in_transit":
        return "transit"
    if state == "outdoor_running":
        return "running"
    return "stationary"

def infer_activity_from_state(state: str) -> str:
    state = normalize_state(state)
    if "sleeping" in state:
        return "sleeping"
    if infer_motion_from_state(state) in {"walking", "running", "cycling"}:
        return "active"
    if state.startswith("office_") or state.startswith("at_cafe") or state.startswith("at_education") or state.startswith("at_restaurant"):
        return "sitting"
    if "lying" in state:
        return "sitting"
    if state.startswith("home_"):
        return "standing"
    return "standing"

def infer_phone_from_state(state: str, rng: random.Random) -> str:
    loc = infer_location_from_state(state)
    if loc in {"work", "education"}:
        return rng.choices(["in_use", "on_desk", "in_pocket", "face_up"], weights=[0.35, 0.35, 0.1, 0.2])[0]
    if loc in {"home", "cafe"}:
        return rng.choices(["in_use", "face_up", "holding_lying", "on_desk"], weights=[0.4, 0.2, 0.15, 0.25])[0]
    if loc in {"restaurant", "shopping", "outdoor", "en_route", "airport", "rail_station", "metro", "transit", "social"}:
        return rng.choices(["in_pocket", "in_use", "face_up"], weights=[0.45, 0.4, 0.15])[0]
    return rng.choice(PHONE_CATEGORIES)

def infer_sound(state: str, scenario_id: str, rng: random.Random) -> str:
    state = normalize_state(state)
    if "noisy" in state:
        return "noisy"
    loc = infer_location_from_state(state)
    if scenario_id in {"NOISY_GATHERING", "HOME_EVENING_NOISY", "LATE_NIGHT_NOISY"}:
        return rng.choice(["normal", "noisy"])
    if loc in {"cafe", "restaurant", "shopping", "transit", "airport", "rail_station", "metro", "social"}:
        return rng.choices(["normal", "noisy", "quiet"], weights=[0.5, 0.3, 0.2])[0]
    if loc in {"work", "education"}:
        return rng.choices(["quiet", "normal", "silent"], weights=[0.4, 0.5, 0.1])[0]
    if loc == "home":
        return rng.choices(["quiet", "normal", "silent"], weights=[0.45, 0.4, 0.15])[0]
    return "normal"

def time_slot_from_hour(hour: int) -> str:
    if 0 <= hour <= 4:
        return "sleeping"
    if hour == 5:
        return "dawn"
    if 6 <= hour <= 8:
        return "morning"
    if 9 <= hour <= 11:
        return "forenoon"
    if 12 <= hour <= 13:
        return "lunch"
    if 14 <= hour <= 17:
        return "afternoon"
    if 18 <= hour <= 20:
        return "evening"
    if 21 <= hour <= 22:
        return "night"
    return "late_night"

def hour_candidates_for_window(window: str) -> List[int]:
    mapping = {
        "sleeping": list(range(0, 5)),
        "dawn": [5],
        "morning": list(range(6, 9)),
        "forenoon": list(range(9, 12)),
        "lunch": [12, 13],
        "afternoon": list(range(14, 18)),
        "evening": list(range(18, 21)),
        "night": [21, 22],
        "late_night": [23],
        "workday_day": list(range(9, 18)),
        "office_day": list(range(8, 18)),
        "home_evening": list(range(18, 23)),
        "weekend_day": list(range(9, 19)),
        "travel_day": list(range(7, 21)),
        "any": list(range(0, 24)),
    }
    return mapping.get(window, mapping["any"])

def choose_current_state(rule: Dict[str, Any]) -> List[str]:
    current_state_opts: List[str] = []
    for cond in rule.get("conditions", []):
        if cond.get("key") == "state_current":
            current_state_opts.extend(split_csv_values(cond.get("value")))
    deduped: List[str] = []
    for s in current_state_opts:
        if s not in deduped:
            deduped.append(s)
    return deduped or ["unknown"]

def infer_time_window(spec: ScenarioSpec, state_options: List[str]) -> str:
    sid = spec.scenario_id
    joined = " ".join(state_options)
    # First respect explicit hour conditions when they exist.
    gte_vals = [int(c["value"]) for c in spec.conditions if c.get("key") == "hour" and c.get("op") == "gte" and str(c.get("value")).isdigit()]
    lte_vals = [int(c["value"]) for c in spec.conditions if c.get("key") == "hour" and c.get("op") == "lte" and str(c.get("value")).isdigit()]
    if gte_vals or lte_vals:
        lo = max(gte_vals) if gte_vals else 0
        hi = min(lte_vals) if lte_vals else 23
        mid = (lo + hi) / 2
        if hi <= 5:
            return "sleeping"
        if lo == 5 and hi == 5:
            return "dawn"
        if hi <= 8:
            return "morning"
        if hi <= 11:
            return "forenoon"
        if lo <= 13 and hi <= 14:
            return "lunch"
        if hi <= 17:
            return "afternoon"
        if hi <= 20:
            return "evening"
        if hi <= 22:
            return "night"
        return "late_night"
    if any(k in sid for k in ["LATE_NIGHT", "HOME_LATE_NIGHT"]):
        return "late_night"
    if any(k in sid for k in ["MORNING", "WAKE_UP", "COMMUTE_MORNING"]):
        return "morning"
    if "LUNCH" in sid:
        return "lunch"
    if any(k in sid for k in ["AFTERNOON", "AFTER_LUNCH"]):
        return "afternoon"
    if any(k in sid for k in ["EVENING", "COMMUTE_EVENING", "LATE_RETURN_HOME"]):
        return "evening"
    if any(x in joined for x in ["office_", "home_daytime_workday", "at_cafe_quiet"]):
        return "office_day"
    if spec.category in {"work", "education"}:
        return "workday_day"
    if spec.category in {"travel", "commute"}:
        return "travel_day"
    return "any"

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

def build_scenario_plan(spec: ScenarioSpec) -> ScenarioPlan:
    state_options = choose_current_state({"conditions": spec.conditions})
    precondition_options: List[str] = []
    for cond in spec.preconditions:
        if cond.get("key") == "state_current":
            precondition_options.extend(split_csv_values(cond.get("value")))
        elif cond.get("key"):
            precondition_options.append(f"{cond['key']}={cond.get('value')}")
    if not precondition_options:
        precondition_options = ["none"]

    return ScenarioPlan(
        scenario_id=spec.scenario_id,
        scenario_name=spec.scenario_name,
        category=spec.category,
        time_window=infer_time_window(spec, state_options),
        state_options=state_options,
        precondition_options=precondition_options,
        notes="first-timestep-only generator using updated rules + v5 global action space",
    )

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

def choose_battery_and_charging(location: str, hour: int, rng: random.Random) -> Tuple[int, int]:
    is_charging = 1 if location in {"home", "work"} and hour in {0,1,2,3,4,5,6,22,23} and rng.random() < 0.35 else 0
    level = rng.randint(35, 95)
    if is_charging:
        level = max(level, 55)
    return level, is_charging

def choose_start_timestep(plan: ScenarioPlan, rng: random.Random) -> int:
    hour = rng.choice(hour_candidates_for_window(plan.time_window))
    minute = rng.randint(0, 59)
    second = rng.choice([0,5,10,15,20,25,30,35,40,45,50,55])
    return hour * 3600 + minute * 60 + second

def pick_ranked_label(items: List[str], rng: random.Random) -> str:
    if not items:
        raise ValueError("Expected non-empty ranked item list")
    if len(items) == 1:
        return items[0]
    if rng.random() < 0.5:
        return items[0]
    return rng.choice(items[1:])

def apply_rule_overrides(features: Dict[str, Any], spec: ScenarioSpec, chosen_state: str, chosen_precondition: str, rng: random.Random) -> None:
    # generic rule-derived overrides
    for cond in spec.conditions:
        key = cond.get("key")
        value = cond.get("value")
        if key == "state_current":
            features["state_current"] = chosen_state
        elif key == "wifiLost":
            features["wifiLost"] = 1 if str(value).lower() == "true" else 0
        elif key == "wifiLostCategory":
            candidate = str(value)
            features["wifiLostCategory"] = candidate if candidate in LOCATION_CATEGORIES else features["wifiLostCategory"]
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

    features["precondition"] = chosen_precondition
    features["state_current"] = chosen_state

    sid = spec.scenario_id
    # scenario-specific realism fixes
    if sid == "OFFICE_LUNCH_OUT":
        features["wifiLost"] = 1
        features["wifiLostCategory"] = "work"
        features["ps_time"] = "lunch"
    elif sid == "LEAVE_OFFICE":
        features["wifiLost"] = 1
        features["wifiLostCategory"] = "work"
        if infer_location_from_state(chosen_state) == "en_route":
            features["ps_location"] = "work" if chosen_state == "outdoor_walking" else "en_route"
    elif sid == "IN_MEETING":
        features["cal_inMeeting"] = 1
        features["cal_hasUpcoming"] = 1
        features["cal_eventCount"] = max(features["cal_eventCount"], 1)
        if features["cal_nextLocation"] == "unknown":
            features["cal_nextLocation"] = "work" if features["ps_location"] in {"work", "cafe", "home"} else features["ps_location"]
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
        features["ps_location"] = "work"
    elif sid == "TRAIN_DEPARTURE":
        features["sms_train_pending"] = 1
        features["ps_location"] = "rail_station" if "rail_station" in chosen_state else infer_location_from_state(chosen_state)
    elif sid == "FLIGHT_BOARDING":
        features["sms_flight_pending"] = 1
        features["ps_location"] = "airport"
    elif sid == "HOTEL_CHECKIN":
        features["sms_hotel_pending"] = 1
    elif sid == "MOVIE_TICKET":
        features["sms_movie_pending"] = 1
    elif sid == "HOSPITAL_APPOINTMENT":
        features["sms_hospital_pending"] = 1
        if features["ps_location"] == "en_route":
            features["cal_nextLocation"] = "health"
    elif sid == "RIDESHARE_PICKUP":
        features["sms_ride_pending"] = 1

    # recompute derived fields
    features["ps_location"] = infer_location_from_state(features["state_current"])
    features["ps_motion"] = infer_motion_from_state(features["state_current"])
    features["activityState"] = infer_activity_from_state(features["state_current"])
    features["ps_phone"] = infer_phone_from_state(features["state_current"], rng)
    features["ps_sound"] = infer_sound(features["state_current"], sid, rng)
    if features["wifiLost"] == 0:
        features["wifiLostCategory"] = "unknown"

def generate_firststep_row(spec: ScenarioSpec, plan: ScenarioPlan, episode_idx: int, rng: random.Random, action_type_by_id: Dict[str, str]) -> Dict[str, Any]:
    start_timestep = choose_start_timestep(plan, rng)
    hour = start_timestep // 3600
    chosen_state = rng.choice(plan.state_options)
    chosen_precondition = rng.choice(plan.precondition_options) if plan.precondition_options else "none"
    loc = infer_location_from_state(chosen_state)
    battery, is_charging = choose_battery_and_charging(loc, hour, rng)
    profile = choose_profile(rng)

    features: Dict[str, Any] = {
        "scenarioId": spec.scenario_id,
        "precondition": "none",
        "state_current": chosen_state,
        "state_duration_sec": 0,
        "ps_time": time_slot_from_hour(hour),
        "hour": hour,
        "cal_hasUpcoming": 1 if spec.category in {"work", "education", "travel"} else 0,
        "ps_dayType": "weekend" if "WEEKEND" in spec.scenario_id else "workday",
        "ps_motion": infer_motion_from_state(chosen_state),
        "wifiLost": 0,
        "wifiLostCategory": "unknown",
        "cal_eventCount": 0 if spec.scenario_id == "NO_MEETINGS" else rng.randint(0, 4),
        "cal_inMeeting": 0,
        "cal_nextLocation": "unknown",
        "ps_sound": infer_sound(chosen_state, spec.scenario_id, rng),
        "timestep": start_timestep,
        "ps_location": loc,
        "ps_phone": infer_phone_from_state(chosen_state, rng),
        "batteryLevel": battery,
        "isCharging": is_charging,
        "networkType": choose_network(loc, rng),
        "activityState": infer_activity_from_state(chosen_state),
        "activityDuration": 0,
        **profile,
    }
    for sms_field in SMS_FIELDS:
        features[sms_field] = 0

    apply_rule_overrides(features, spec, chosen_state, chosen_precondition, rng)

    gt_ro = pick_ranked_label(spec.default_action_ids, rng)
    gt_app = pick_ranked_label(spec.default_app_categories, rng)
    return {
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

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    scenario_defaults, action_type_by_id = parse_global_action_space(args.global_action_space)
    rules_by_sid = parse_rules(args.rules)
    specs_all = build_scenario_specs(rules_by_sid, scenario_defaults)

    ignored = {x.strip() for x in args.ignore_trigger_scenarios.split(",") if x.strip()}
    included_specs = {sid: spec for sid, spec in specs_all.items() if sid not in ignored}

    scenario_subset = None
    if args.scenario_ids.strip():
        scenario_subset = {x.strip() for x in args.scenario_ids.split(",") if x.strip()}
        included_specs = {sid: spec for sid, spec in included_specs.items() if sid in scenario_subset}

    plans = {sid: build_scenario_plan(spec) for sid, spec in included_specs.items()}
    if args.plan_output is not None:
        args.plan_output.write_text(json.dumps({sid: asdict(plan) for sid, plan in sorted(plans.items())}, indent=2), encoding="utf-8")

    metadata: Dict[str, Any] = {
        "generator": Path(__file__).name,
        "mode": "v0_first_timestep_only_no_trigger_scenarios_using_updated_rules_and_v5_global_action_space",
        "rules_source": str(args.rules),
        "actions_md_source": str(args.actions_md) if args.actions_md is not None else None,
        "global_action_space_source": str(args.global_action_space),
        "episodes_per_scenario": args.episodes_per_scenario,
        "gt_policy": {
            "ro": "first ranked actionId gets 50% probability; remaining 50% is distributed uniformly across the remaining actionIds",
            "app": "first ranked app category gets 50% probability; remaining 50% is distributed uniformly across the remaining app categories",
        },
        "seed": args.seed,
        "scenario_alias_to_v5": SCENARIO_ALIAS_TO_V5,
        "ignored_scenarios_count": len(ignored),
        "ignored_scenarios": sorted(ignored),
        "included_scenarios_count": len(included_specs),
        "included_scenarios": [],
    }

    total_rows = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for sid in sorted(included_specs):
            spec = included_specs[sid]
            plan = plans[sid]
            per_scenario_summary = {
                "scenario_id": sid,
                "v5_scenario_id": spec.v5_scenario_id,
                "scenario_name": spec.scenario_name,
                "category": spec.category,
                "ranked_ro_action_ids": spec.default_action_ids,
                "ranked_app_categories": spec.default_app_categories,
                "plan": asdict(plan),
                "episode_count": args.episodes_per_scenario,
            }
            for i in range(args.episodes_per_scenario):
                row = generate_firststep_row(spec, plan, i + 1, rng, action_type_by_id)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1
            metadata["included_scenarios"].append(per_scenario_summary)

    metadata["total_rows"] = total_rows
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Ignored scenarios: {len(ignored)} / {len(specs_all)}")
    print(f"Included scenarios: {len(included_specs)}")
    print(f"Wrote {total_rows} rows -> {args.output}")
    print(f"Metadata -> {args.metadata}")
    if args.plan_output is not None:
        print(f"Scenario plan -> {args.plan_output}")

if __name__ == "__main__":
    main()
