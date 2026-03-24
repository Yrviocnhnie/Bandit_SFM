#!/usr/bin/env python3
"""Generate synthetic v0 contextual-bandit data using the precondition-enhanced rules file.

Mode implemented here matches the latest agreed setup:
- ignore scenarios that have any non-`none` recommendation trigger
- emit exactly one row per episode (the first prediction timestep only)
- keep only v0 features
- include `precondition`
- exclude `transportMode`
- exclude `cal_nextMinutes`
- use 70/30 default-vs-alternative GT mix

Output:
- one JSON object per episode in JSONL
- metadata JSON describing included / ignored scenarios and generation settings
- optional scenario plan JSON for inspection
"""
from __future__ import annotations

import argparse
import ast
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

APP_CATEGORIES: List[str] = [
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


@dataclass
class ActionRow:
    scenario_id: str
    scenario_name: str
    trigger: str
    action_id: str
    is_default: bool
    action_type: str
    payload: Dict[str, Any]


@dataclass
class ScenarioSpec:
    scenario_id: str
    scenario_name: str
    category: str
    conditions: List[Dict[str, Any]]
    preconditions: List[Dict[str, Any]]
    default_ro_id: str
    default_ro_type: str
    default_app_category: str
    ro_action_ids: List[str]
    alt_ro_action_ids: List[str]
    all_triggers: List[str]


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
    p = argparse.ArgumentParser(description="Generate first-step-only v0 bandit data from the updated rules file (with added preconditions)")
    p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--actions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--plan-output", type=Path, default=None)
    p.add_argument("--episodes-per-scenario", type=int, default=1000)
    p.add_argument("--default-ratio", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scenario-ids", type=str, default="", help="Comma-separated subset after trigger filtering")
    return p.parse_args()


def parse_action_payload(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return {"raw": text}


def parse_actions(actions_md: Path) -> Tuple[Dict[str, Dict[str, Any]], List[ActionRow]]:
    lines = actions_md.read_text(encoding="utf-8").splitlines()

    summary_start = None
    for i, line in enumerate(lines):
        if line.startswith("| scenarioId | scenarioName | actionIDs | actionIDDefault |"):
            summary_start = i + 2
            break
    if summary_start is None:
        raise ValueError("Could not find scenario summary table")

    summary: Dict[str, Dict[str, Any]] = {}
    for line in lines[summary_start:]:
        if not line.startswith("|"):
            continue
        if line.startswith("| scenarioId | scenarioName | trigger |"):
            break
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 8:
            continue
        sid = parts[0].strip("`")
        if not sid:
            continue
        action_ids = ast.literal_eval(parts[2].strip("`"))
        default_obj = ast.literal_eval(parts[3].strip("`"))
        summary[sid] = {
            "scenario_name": parts[1],
            "action_ids": action_ids,
            "default_ro": default_obj,
            "default_app_category": parts[7].strip("`"),
        }

    detail_start = None
    for i, line in enumerate(lines):
        if line.startswith("| scenarioId | scenarioName | trigger | isNewlyAdded | actionId | isDefault | actionType | actionPayload |"):
            detail_start = i + 2
            break
    if detail_start is None:
        raise ValueError("Could not find detailed action table")

    detail_rows: List[ActionRow] = []
    cur_sid = None
    cur_name = None
    for line in lines[detail_start:]:
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 9:
            continue
        scenario_id, scenario_name, trigger, _is_new, action_id, is_default, action_type, action_payload, _payload_en = parts[:9]
        if scenario_id == "scenarioId" or trigger == "trigger" or trigger == "---":
            continue
        if scenario_id:
            cur_sid = scenario_id.strip("`")
            cur_name = scenario_name
        if cur_sid is None:
            continue
        detail_rows.append(ActionRow(
            scenario_id=cur_sid,
            scenario_name=cur_name or "",
            trigger=trigger.strip("`"),
            action_id=action_id.strip("`"),
            is_default=is_default.strip("`") == "true",
            action_type=action_type.strip("`"),
            payload=parse_action_payload(action_payload),
        ))
    return summary, detail_rows


def parse_rules(rules_json: Path) -> Dict[str, Dict[str, Any]]:
    rules = json.loads(rules_json.read_text(encoding="utf-8"))
    return {rule["scenarioId"]: rule for rule in rules}


def normalize_state(value: str) -> str:
    return LEGACY_STATE_MAP.get(value.strip(), value.strip())


def split_csv_values(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [normalize_state(x) for x in str(value).split(",") if x.strip()]


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
    if state.startswith("office_") or state.startswith("at_cafe") or state.startswith("at_education"):
        return "sitting"
    if any(x in state for x in ["lying", "dark"]):
        return "sitting"
    return "standing"


def infer_phone_from_state(state: str, rng: random.Random) -> str:
    loc = infer_location_from_state(state)
    if loc in {"work", "education"}:
        return rng.choices(["in_use", "on_desk", "in_pocket", "face_up"], weights=[0.35, 0.35, 0.1, 0.2])[0]
    if loc in {"home", "cafe"}:
        return rng.choices(["in_use", "face_up", "holding_lying", "on_desk"], weights=[0.4, 0.2, 0.15, 0.25])[0]
    if loc in {"restaurant", "shopping", "outdoor", "en_route", "airport", "rail_station", "metro", "transit"}:
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


def choose_current_state(rule: Dict[str, Any], scenario_id: str) -> List[str]:
    """Choose candidate current states for the generated row.

    Important policy:
    - If a scenario has explicit `state_current` in its conditions, use those.
    - If a scenario has no current-state condition but does have preconditions,
      keep current state as `unknown` because the state evidence is only in the
      preceding context, not in the current rule body.
    - If a scenario has neither current-state conditions nor preconditions,
      fall back to a small scenario-specific plausible set.
    """
    current_state_opts: List[str] = []
    for cond in rule.get("conditions", []):
        if cond.get("key") == "state_current":
            current_state_opts.extend(split_csv_values(cond.get("value")))

    if current_state_opts:
        deduped: List[str] = []
        for s in current_state_opts:
            s = normalize_state(s)
            if s not in deduped:
                deduped.append(s)
        return deduped

    if rule.get("preconditions"):
        return ["unknown"]

    if scenario_id in {"CAL_HAS_EVENTS", "MEETING_UPCOMING", "NO_MEETINGS"}:
        current_state_opts = ["office_working", "home_daytime_workday", "at_cafe_quiet"]
    elif scenario_id == "IN_MEETING":
        current_state_opts = ["office_working", "home_daytime_workday"]
    elif scenario_id == "REMOTE_MEETING":
        current_state_opts = ["home_daytime_workday", "at_cafe_quiet"]
    elif scenario_id == "DELIVERY_AT_OFFICE":
        current_state_opts = ["office_working", "office_arriving"]
    elif scenario_id in {"TRAIN_DEPARTURE", "FLIGHT_BOARDING", "HOTEL_CHECKIN"}:
        current_state_opts = ["at_rail_station", "at_airport", "at_transit_hub"]
    elif scenario_id == "MOVIE_TICKET":
        current_state_opts = ["at_social", "at_restaurant_other"]
    elif scenario_id == "HOSPITAL_APPOINTMENT":
        current_state_opts = ["at_health", "at_health_inpatient"]
    elif scenario_id == "RIDESHARE_PICKUP":
        current_state_opts = ["outdoor_walking", "stationary_unknown"]
    else:
        current_state_opts = ["unknown"]

    deduped: List[str] = []
    for s in current_state_opts:
        s = normalize_state(s)
        if s not in deduped:
            deduped.append(s)
    return deduped

def infer_time_window(scenario_id: str, state_options: List[str], category: str) -> str:
    sid = scenario_id
    joined = " ".join(state_options)
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
    if category in {"work", "education"}:
        return "workday_day"
    if category in {"travel", "commute"}:
        return "travel_day"
    return "any"


def build_scenario_specs(rules_by_sid: Dict[str, Dict[str, Any]], summary: Dict[str, Dict[str, Any]], detail_rows: List[ActionRow]) -> Dict[str, ScenarioSpec]:
    detail_by_sid: Dict[str, List[ActionRow]] = {}
    for row in detail_rows:
        detail_by_sid.setdefault(row.scenario_id, []).append(row)

    specs: Dict[str, ScenarioSpec] = {}
    for sid, rule in rules_by_sid.items():
        if sid not in summary:
            continue
        summ = summary[sid]
        rows = detail_by_sid.get(sid, [])
        default_ro = summ["default_ro"]
        ro_ids = [r.action_id for r in rows if r.action_type in {"R", "O"}]
        all_triggers = sorted({(r.trigger or "none").strip() for r in rows if r.action_type in {"R", "O"}})
        alt_ro = [x for x in ro_ids if x != default_ro["id"]]
        specs[sid] = ScenarioSpec(
            scenario_id=sid,
            scenario_name=summ["scenario_name"],
            category=rule.get("category", "unknown"),
            conditions=rule.get("conditions", []),
            preconditions=rule.get("preconditions", []),
            default_ro_id=default_ro["id"],
            default_ro_type=default_ro["type"],
            default_app_category=summ["default_app_category"],
            ro_action_ids=ro_ids,
            alt_ro_action_ids=alt_ro,
            all_triggers=all_triggers,
        )
    return specs


def build_scenario_plan(spec: ScenarioSpec) -> ScenarioPlan:
    state_options = choose_current_state({"conditions": spec.conditions, "preconditions": spec.preconditions}, spec.scenario_id)
    precondition_options: List[str] = []
    for cond in spec.preconditions:
        if cond.get("key") == "state_current":
            precondition_options.extend(split_csv_values(cond.get("value")))
        elif cond.get("key"):
            precondition_options.append(f"{cond['key']}={cond.get('value')}")
    if not precondition_options:
        precondition_options = ["none"]

    notes = "first-timestep-only v0 episode"
    return ScenarioPlan(
        scenario_id=spec.scenario_id,
        scenario_name=spec.scenario_name,
        category=spec.category,
        time_window=infer_time_window(spec.scenario_id, state_options, spec.category),
        state_options=state_options,
        precondition_options=precondition_options,
        notes=notes,
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


def exact_default_flags(n: int, default_ratio: float, rng: random.Random) -> List[bool]:
    n_default = int(round(n * default_ratio))
    n_default = max(0, min(n, n_default))
    flags = [True] * n_default + [False] * (n - n_default)
    rng.shuffle(flags)
    return flags


def apply_rule_overrides(features: Dict[str, Any], spec: ScenarioSpec, chosen_state: str, chosen_precondition: str) -> None:
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
    if sid == "OFFICE_LUNCH_OUT":
        features["wifiLost"] = 1
        features["wifiLostCategory"] = "work"
        features["ps_time"] = "lunch"
        features["ps_location"] = "work"
    elif sid == "LEAVE_OFFICE":
        features["wifiLost"] = 1
        features["wifiLostCategory"] = "work"
    elif sid == "IN_MEETING":
        features["cal_inMeeting"] = 1
        features["cal_hasUpcoming"] = 1
        features["cal_eventCount"] = max(features["cal_eventCount"], 1)
    elif sid == "REMOTE_MEETING":
        features["cal_hasUpcoming"] = 1
        features["cal_eventCount"] = max(features["cal_eventCount"], 1)
        if features["ps_location"] == "home":
            features["cal_nextLocation"] = "home"
    elif sid == "MEETING_UPCOMING":
        features["cal_hasUpcoming"] = 1
        features["cal_eventCount"] = max(features["cal_eventCount"], 1)
    elif sid == "NO_MEETINGS":
        features["cal_hasUpcoming"] = 0
        features["cal_eventCount"] = 0
    elif sid == "CAL_HAS_EVENTS":
        features["cal_hasUpcoming"] = 1
        features["cal_eventCount"] = max(features["cal_eventCount"], 2)
    elif sid == "DELIVERY_AT_OFFICE":
        features["sms_delivery_pending"] = 1
        features["ps_location"] = "work"
    elif sid == "TRAIN_DEPARTURE":
        features["sms_train_pending"] = 1
    elif sid == "FLIGHT_BOARDING":
        features["sms_flight_pending"] = 1
        features["ps_location"] = "airport"
    elif sid == "HOTEL_CHECKIN":
        features["sms_hotel_pending"] = 1
    elif sid == "MOVIE_TICKET":
        features["sms_movie_pending"] = 1
    elif sid == "HOSPITAL_APPOINTMENT":
        features["sms_hospital_pending"] = 1
        features["ps_location"] = "health"
    elif sid == "RIDESHARE_PICKUP":
        features["sms_ride_pending"] = 1

    features["ps_motion"] = infer_motion_from_state(features["state_current"])
    features["activityState"] = infer_activity_from_state(features["state_current"])
    if features["wifiLost"] == 0:
        features["wifiLostCategory"] = "unknown"


def choose_labels(spec: ScenarioSpec, use_default_ro: bool, use_default_app: bool, rng: random.Random) -> Tuple[str, str]:
    if use_default_ro or not spec.alt_ro_action_ids:
        ro = spec.default_ro_id
    else:
        ro = rng.choice(spec.alt_ro_action_ids)

    if use_default_app:
        app = spec.default_app_category
    else:
        alt_apps = [x for x in APP_CATEGORIES if x != spec.default_app_category]
        app = rng.choice(alt_apps)
    return ro, app


def generate_firststep_row(spec: ScenarioSpec, plan: ScenarioPlan, episode_idx: int, use_default_ro: bool, use_default_app: bool, rng: random.Random) -> Dict[str, Any]:
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
        "cal_nextLocation": loc if loc in LOCATION_CATEGORIES else "unknown",
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
    apply_rule_overrides(features, spec, chosen_state, chosen_precondition)

    gt_ro, gt_app = choose_labels(spec, use_default_ro, use_default_app, rng)
    return {
        "episode_id": f"{spec.scenario_id.lower()}_ep{episode_idx:04d}",
        "scenario_id": spec.scenario_id,
        "scenario_name": spec.scenario_name,
        "t_in_scenario_sec": 0,
        "dt_sec": None,
        "features": features,
        "gt_ro": gt_ro,
        "gt_app": gt_app,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    summary, detail_rows = parse_actions(args.actions)
    rules_by_sid = parse_rules(args.rules)
    specs_all = build_scenario_specs(rules_by_sid, summary, detail_rows)

    # Ignore any scenario with any non-none trigger in its R/O rows.
    ignored_trigger_scenarios = sorted([
        sid for sid, spec in specs_all.items()
        if any((tr or "none").strip() != "none" for tr in spec.all_triggers)
    ])
    included_specs = {sid: spec for sid, spec in specs_all.items() if sid not in ignored_trigger_scenarios}

    scenario_subset = None
    if args.scenario_ids.strip():
        scenario_subset = {x.strip() for x in args.scenario_ids.split(",") if x.strip()}
        included_specs = {sid: spec for sid, spec in included_specs.items() if sid in scenario_subset}

    plans = {sid: build_scenario_plan(spec) for sid, spec in included_specs.items()}
    if args.plan_output is not None:
        args.plan_output.write_text(json.dumps({sid: asdict(plan) for sid, plan in sorted(plans.items())}, indent=2), encoding="utf-8")

    metadata: Dict[str, Any] = {
        "generator": "generate_bandit_v0_firststep_no_triggers_newrules.py",
        "mode": "v0_first_timestep_only_no_trigger_scenarios",
        "episodes_per_scenario": args.episodes_per_scenario,
        "default_ratio": args.default_ratio,
        "seed": args.seed,
        "ignored_trigger_scenarios_count": len(ignored_trigger_scenarios),
        "ignored_trigger_scenarios": ignored_trigger_scenarios,
        "included_scenarios_count": len(included_specs),
        "included_scenarios": [],
    }

    total_rows = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for sid in sorted(included_specs):
            spec = included_specs[sid]
            plan = plans[sid]
            ro_default_flags = exact_default_flags(args.episodes_per_scenario, args.default_ratio, rng)
            app_default_flags = exact_default_flags(args.episodes_per_scenario, args.default_ratio, rng)

            per_scenario_summary = {
                "scenario_id": sid,
                "scenario_name": spec.scenario_name,
                "category": spec.category,
                "candidate_ro_action_ids": spec.ro_action_ids,
                "default_ro_id": spec.default_ro_id,
                "default_app_category": spec.default_app_category,
                "plan": asdict(plan),
                "episode_count": args.episodes_per_scenario,
                "ro_default_episode_count": 0,
                "ro_alternative_episode_count": 0,
                "app_default_episode_count": 0,
                "app_alternative_episode_count": 0,
            }

            for i in range(args.episodes_per_scenario):
                row = generate_firststep_row(
                    spec=spec,
                    plan=plan,
                    episode_idx=i + 1,
                    use_default_ro=ro_default_flags[i],
                    use_default_app=app_default_flags[i],
                    rng=rng,
                )
                per_scenario_summary["ro_default_episode_count" if ro_default_flags[i] else "ro_alternative_episode_count"] += 1
                per_scenario_summary["app_default_episode_count" if app_default_flags[i] else "app_alternative_episode_count"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1
            metadata["included_scenarios"].append(per_scenario_summary)

    metadata["total_rows"] = total_rows
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Ignored trigger-based scenarios: {len(ignored_trigger_scenarios)} / {len(specs_all)}")
    print(f"Included scenarios: {len(included_specs)}")
    print(f"Wrote {total_rows} rows -> {args.output}")
    print(f"Metadata -> {args.metadata}")
    if args.plan_output is not None:
        print(f"Scenario plan -> {args.plan_output}")


if __name__ == "__main__":
    main()
