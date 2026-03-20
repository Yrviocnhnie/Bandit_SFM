"""App-aligned state, snapshot, and scenario definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

try:
    from .feature_space import APP_ACTIONS, APP_CATALOG_PATH, REPO_ROOT, RO_ACTIONS
except ImportError:
    from feature_space import APP_ACTIONS, APP_CATALOG_PATH, REPO_ROOT, RO_ACTIONS


DEFAULT_RULES_PATH = REPO_ROOT / "entry" / "src" / "main" / "resources" / "rawfile" / "config" / "default_rules.json"
CONTEXT_MODELS_PATH = REPO_ROOT / "entry" / "src" / "main" / "ets" / "service" / "context" / "ContextModels.ets"
STATE_FIELDS = ("time", "location", "motion", "phone", "light", "sound", "day_type")


@dataclass(frozen=True)
class PhysicalState:
    time: str = "sleeping"
    location: str = "unknown"
    motion: str = "stationary"
    phone: str = "unknown"
    light: str = "normal"
    sound: str = "unknown"
    day_type: str = "workday"


@dataclass(frozen=True)
class LocInstanceAttr:
    familiarity: float = 0.0
    ownership: float = 0.0
    routine_level: float = 0.0


@dataclass(frozen=True)
class StateChainEntry:
    code: str
    start_time: int
    duration_ms: int

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "startTime": self.start_time,
            "durationMs": self.duration_ms,
        }


@dataclass
class ContextSnapshot:
    hour: str = "0"
    ps_time: str = "sleeping"
    ps_location: str = "unknown"
    ps_motion: str = "stationary"
    ps_phone: str = "unknown"
    ps_sound: str = "unknown"
    ps_dayType: str = "workday"
    state_current: str = "stationary_unknown"
    state_prev: str = ""
    state_duration_sec: str = "0"
    state_chain_json: str = "[]"
    ps_scenario: str = ""
    cal_hasUpcoming: str = "unknown"
    cal_nextMinutes: str = "0"
    cal_inMeeting: str = "unknown"
    cal_nextLocation: str = "unknown"
    cal_eventCount: str = "0"
    wifiLost: str = "unknown"
    wifiLostCategory: str = "unknown"
    batteryLevel: str = "50"
    isCharging: str = "false"
    networkType: str = "wifi"
    transportMode: str = "unknown"
    activityState: str = "unknown"
    activityDuration: str = "0"
    prevActivityState: str = "unknown"
    sms_delivery_pending: str = "unknown"
    sms_train_pending: str = "unknown"
    sms_flight_pending: str = "unknown"
    sms_hotel_pending: str = "unknown"
    sms_movie_pending: str = "unknown"
    sms_hospital_pending: str = "unknown"
    sms_ride_pending: str = "unknown"

    def to_dict(self) -> Dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class RuleCondition:
    key: str
    op: str
    value: str


@dataclass(frozen=True)
class RulePrecondition:
    key: str
    op: str
    value: str
    within_ms: int


@dataclass(frozen=True)
class ScenarioDefinition:
    rule_id: str
    scenario_id: str
    scenario_name: str
    scenario_name_en: str
    category: str
    conditions: List[RuleCondition]
    preconditions: List[RulePrecondition] = field(default_factory=list)
    ro_actions: List[str] = field(default_factory=list)
    default_app_category: str = "productivity"


@dataclass(frozen=True)
class ScenarioMatch:
    scenario_id: str
    name: str
    confidence: float
    category: str
    ro_actions: List[str]
    app_action: str
    rule_id: str


def _extract_ro_actions_from_rule(rule_obj: Dict[str, Any]) -> List[str]:
    actions: List[str] = []
    for suggestion in rule_obj.get("suggestions", []):
        action_id = suggestion.get("id")
        if isinstance(action_id, str) and action_id in RO_ACTIONS and action_id not in actions:
            actions.append(action_id)
    return actions


def _load_bandit_catalog() -> Dict[str, Dict[str, Any]]:
    if not APP_CATALOG_PATH.exists():
        return {}
    text = APP_CATALOG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\{ scenarioId: '([^']+)', scenarioName: '([^']+)', defaultAppCategory: '([^']+)', roActions: \[(.*?)\] \}",
        re.S,
    )
    catalog: Dict[str, Dict[str, Any]] = {}
    for scenario_id, scenario_name, app_category, actions_blob in pattern.findall(text):
        ro_actions = re.findall(r"id: '([^']+)'", actions_blob)
        catalog[scenario_id] = {
            "scenario_name": scenario_name,
            "default_app_category": app_category if app_category in APP_ACTIONS else "productivity",
            "ro_actions": ro_actions,
        }
    return catalog


def load_app_scenarios(default_rules_path: Path = DEFAULT_RULES_PATH) -> List[ScenarioDefinition]:
    raw = json.loads(default_rules_path.read_text(encoding="utf-8"))
    bandit_catalog = _load_bandit_catalog()
    scenarios: List[ScenarioDefinition] = []
    for item in raw:
        scenario_id = item.get("scenarioId")
        if not isinstance(scenario_id, str) or not scenario_id:
            continue
        conditions = [
            RuleCondition(key=str(cond.get("key", "")), op=str(cond.get("op", "")), value=str(cond.get("value", "")))
            for cond in item.get("conditions", [])
        ]
        preconditions = [
            RulePrecondition(
                key=str(cond.get("key", "")),
                op=str(cond.get("op", "")),
                value=str(cond.get("value", "")),
                within_ms=int(cond.get("withinMs", 0) or 0),
            )
            for cond in item.get("preconditions", [])
        ]
        bandit_entry = bandit_catalog.get(scenario_id, {})
        ro_actions = list(bandit_entry.get("ro_actions", _extract_ro_actions_from_rule(item)))
        default_app_category = str(bandit_entry.get("default_app_category", "productivity"))
        scenarios.append(
            ScenarioDefinition(
                rule_id=str(item.get("id", "")),
                scenario_id=scenario_id,
                scenario_name=str(bandit_entry.get("scenario_name", item.get("scenarioName", item.get("name", scenario_id)))),
                scenario_name_en=str(item.get("scenarioNameEn", scenario_id)),
                category=str(item.get("category", "unknown")),
                conditions=conditions,
                preconditions=preconditions,
                ro_actions=ro_actions,
                default_app_category=default_app_category,
            )
        )
    return scenarios


APP_SCENARIOS: List[ScenarioDefinition] = load_app_scenarios()
APP_SCENARIO_BY_ID: Dict[str, ScenarioDefinition] = {item.scenario_id: item for item in APP_SCENARIOS}


def _humanize_code(raw: str) -> str:
    if not raw:
        return ""
    return raw.replace("_", " ").title()


def _load_state_labels() -> Dict[str, Dict[str, str]]:
    labels: Dict[str, Dict[str, str]] = {}
    if not CONTEXT_MODELS_PATH.exists():
        return labels
    text = CONTEXT_MODELS_PATH.read_text(encoding="utf-8")
    for zh, en, code in re.findall(r"v = \{ zh: '([^']+)', en: '([^']+)' \}; m\.set\('([^']+)', v\);", text):
        labels[code] = {"zh": zh, "en": en}
    return labels


STATE_LABELS: Dict[str, Dict[str, str]] = _load_state_labels()


def get_state_name(code: str, language: str = "en") -> str:
    label = STATE_LABELS.get(code)
    if label is None:
        return _humanize_code(code)
    return label.get(language, label.get("en", _humanize_code(code)))


def get_scenario_name(scenario_id: str, language: str = "en") -> str:
    scenario = APP_SCENARIO_BY_ID.get(scenario_id)
    if scenario is None:
        return _humanize_code(scenario_id)
    return scenario.scenario_name_en if language == "en" else scenario.scenario_name


def build_debug_labels(state_code: str, scenario_id: Optional[str]) -> Dict[str, Optional[str]]:
    return {
        "state_code": state_code,
        "state_name_en": get_state_name(state_code, "en"),
        "state_name_zh": get_state_name(state_code, "zh"),
        "scenario_id": scenario_id,
        "scenario_name_en": get_scenario_name(scenario_id, "en") if scenario_id else None,
        "scenario_name_zh": get_scenario_name(scenario_id, "zh") if scenario_id else None,
    }

