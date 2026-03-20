"""App-aligned state encoder and scenario matcher."""

from __future__ import annotations

import json
from typing import Iterable, List, Optional, Sequence

try:
    from .feature_space import APP_ACTIONS, RO_ACTIONS
    from .state_and_scenario import APP_SCENARIOS, ContextSnapshot, PhysicalState, ScenarioDefinition, ScenarioMatch, StateChainEntry, build_debug_labels
except ImportError:
    from feature_space import APP_ACTIONS, RO_ACTIONS
    from state_and_scenario import APP_SCENARIOS, ContextSnapshot, PhysicalState, ScenarioDefinition, ScenarioMatch, StateChainEntry, build_debug_labels


def one_hot_distribution(items: Sequence[str], winner: str, smooth: float = 0.05) -> List[float]:
    if not items:
        return []
    item_list = list(items)
    base = smooth / len(item_list)
    out = [base] * len(item_list)
    if winner in item_list:
        out[item_list.index(winner)] += 1.0 - smooth
    total = sum(out)
    return [v / total for v in out]


def ranked_distribution(items: Sequence[str], winners: Sequence[str], smooth: float = 0.05) -> List[float]:
    if not items:
        return []
    item_list = list(items)
    out = [smooth / len(item_list)] * len(item_list)
    rank_weights = [0.65, 0.25, 0.10]
    for rank, code in enumerate(list(winners)[: len(rank_weights)]):
        if code in item_list:
            out[item_list.index(code)] += rank_weights[rank]
    total = sum(out)
    return [v / total for v in out]


class RuleEngine:
    def __init__(self, scenarios: Sequence[ScenarioDefinition] = APP_SCENARIOS) -> None:
        self.scenarios = list(scenarios)

    def encode_state(self, state: PhysicalState) -> str:
        loc = state.location
        mot = state.motion
        time = state.time
        day = state.day_type

        if loc == "metro":
            base = "at_metro"
        elif loc == "rail_station":
            base = "at_rail_station"
        elif loc == "airport":
            base = "at_airport"
        elif loc == "transit":
            base = "at_transit_hub"
        elif loc == "home":
            if time in ("sleeping", "late_night"):
                base = "home_sleeping"
            elif mot in ("running", "cycling"):
                base = "home_active"
            elif time in ("dawn", "morning"):
                base = "home_morning_workday" if day == "workday" else "home_morning_rest"
            elif time in ("evening", "night"):
                base = "home_evening"
            else:
                base = "home_daytime_workday" if day == "workday" else "home_daytime_rest"
        elif loc == "work":
            if day in ("weekend", "holiday"):
                base = "office_rest_day"
            elif time in ("dawn", "morning"):
                base = "office_arriving"
            elif time == "lunch":
                base = "office_lunch_break"
            elif time in ("forenoon", "afternoon"):
                base = "office_working"
            elif time in ("evening", "night"):
                base = "office_overtime"
            elif time in ("sleeping", "late_night"):
                base = "office_late_overtime"
            else:
                base = "office_working"
        elif loc == "restaurant":
            if time == "lunch":
                base = "at_restaurant_lunch"
            elif time in ("evening", "night"):
                base = "at_restaurant_dinner"
            else:
                base = "at_restaurant_other"
        elif loc == "cafe":
            base = "at_cafe"
        elif loc == "gym":
            base = "at_gym_exercising" if mot in ("walking", "running", "cycling") else "at_gym"
        elif loc == "shopping":
            base = "at_shopping"
        elif loc == "health":
            base = "at_health"
        elif loc == "social":
            base = "at_social"
        elif loc == "education":
            base = "at_education"
        elif loc == "custom":
            base = "at_custom"
        elif loc == "outdoor":
            if mot == "running":
                base = "outdoor_running"
            elif mot == "cycling":
                base = "outdoor_cycling"
            elif mot == "walking":
                base = "outdoor_walking"
            else:
                base = "outdoor_resting"
        elif mot == "cycling":
            if day == "workday" and time in ("dawn", "morning"):
                base = "commuting_cycle_out"
            elif day == "workday" and time in ("evening", "night"):
                base = "commuting_cycle_home"
            else:
                base = "outdoor_cycling"
        elif mot == "driving":
            if day == "workday" and time in ("dawn", "morning"):
                base = "commuting_drive_out"
            elif day == "workday" and time in ("evening", "night"):
                base = "commuting_drive_home"
            else:
                base = "driving"
        elif mot == "transit":
            if day == "workday" and time in ("dawn", "morning"):
                base = "commuting_transit_out"
            elif day == "workday" and time in ("evening", "night"):
                base = "commuting_transit_home"
            else:
                base = "in_transit"
        elif mot == "running":
            base = "outdoor_running"
        elif mot == "walking":
            if loc == "en_route" and day == "workday" and time in ("dawn", "morning"):
                base = "commuting_walk_out"
            elif loc == "en_route" and day == "workday" and time in ("evening", "night"):
                base = "commuting_walk_home"
            else:
                base = "walking_unknown"
        else:
            base = "stationary_unknown"

        return self.refine_substate(base, state)

    def refine_substate(self, base: str, state: PhysicalState) -> str:
        phone = state.phone
        light = state.light
        sound = state.sound

        if base == "home_sleeping" and phone == "holding_lying":
            return "home_sleeping_lying"
        if base == "home_morning_workday" and phone == "holding_lying":
            return "home_morning_workday_lying"
        if base == "home_morning_rest" and phone == "holding_lying":
            return "home_morning_rest_lying"
        if base == "home_daytime_workday":
            if phone == "holding_lying":
                return "home_daytime_workday_lying"
            if light == "dark":
                return "home_daytime_workday_dark"
        if base == "home_daytime_rest":
            if phone == "holding_lying":
                return "home_daytime_rest_lying"
            if light == "dark":
                return "home_daytime_rest_dark"
        if base == "home_evening":
            if phone == "holding_lying":
                return "home_evening_lying"
            if light == "dark":
                return "home_evening_dark"
            if sound == "noisy":
                return "home_evening_noisy"
        if base == "office_working":
            if phone == "face_down":
                return "office_working_focused"
            if sound == "noisy":
                return "office_working_noisy"
        if base == "at_cafe" and sound == "quiet":
            return "at_cafe_quiet"
        if base == "at_education":
            if sound == "quiet":
                return "at_education_class"
            if sound == "noisy":
                return "at_education_break"
        if base == "at_health" and phone == "holding_lying":
            return "at_health_inpatient"
        if base == "stationary_unknown":
            if phone == "holding_lying":
                return "unknown_lying"
            if phone == "charging":
                return "unknown_settled"
            if light == "dark":
                return "unknown_dark"
            if sound == "noisy":
                return "unknown_noisy"
        return base

    def build_snapshot(
        self,
        state: PhysicalState,
        hour: int = 0,
        state_chain: Optional[Sequence[StateChainEntry]] = None,
        **overrides: str,
    ) -> ContextSnapshot:
        chain = list(state_chain or [])
        snapshot = ContextSnapshot(
            hour=str(hour),
            ps_time=state.time,
            ps_location=state.location,
            ps_motion=state.motion,
            ps_phone=state.phone,
            ps_sound=state.sound,
            ps_dayType=state.day_type,
            state_current=self.encode_state(state),
            state_prev=chain[1].code if len(chain) > 1 else "",
            state_duration_sec=str(int(chain[0].duration_ms / 1000)) if chain else "0",
            state_chain_json=json.dumps([entry.to_json_dict() for entry in chain], ensure_ascii=False),
        )
        for key, value in overrides.items():
            if hasattr(snapshot, key):
                setattr(snapshot, key, value)
        return snapshot

    def _compare(self, raw_value: str, op: str, expected: str) -> bool:
        if op == "in":
            return raw_value in [item.strip() for item in expected.split(",") if item.strip()]
        if op == "eq":
            return raw_value == expected
        if op == "neq":
            return raw_value != expected
        try:
            left = float(raw_value)
            right = float(expected)
        except ValueError:
            left = None
            right = None
        if op == "gt" and left is not None and right is not None:
            return left > right
        if op == "gte" and left is not None and right is not None:
            return left >= right
        if op == "lt" and left is not None and right is not None:
            return left < right
        if op == "lte" and left is not None and right is not None:
            return left <= right
        if op == "range" and left is not None:
            parts = [part.strip() for part in expected.split(",")]
            if len(parts) == 2:
                return float(parts[0]) <= left <= float(parts[1])
        return False

    def _check_conditions(self, snapshot: ContextSnapshot, conditions: Iterable) -> bool:
        values = snapshot.to_dict()
        for cond in conditions:
            if not self._compare(values.get(cond.key, ""), cond.op, cond.value):
                return False
        return True

    def _check_preconditions(self, snapshot: ContextSnapshot, chain: Sequence[StateChainEntry], scenario: ScenarioDefinition) -> bool:
        if not scenario.preconditions:
            return True
        now_ms = chain[0].start_time + chain[0].duration_ms if chain else 0
        snapshot_values = snapshot.to_dict()
        for cond in scenario.preconditions:
            matched = False
            if cond.key == "state_current":
                for entry in chain:
                    if now_ms and entry.start_time + entry.duration_ms < now_ms - cond.within_ms:
                        continue
                    if self._compare(entry.code, cond.op, cond.value):
                        matched = True
                        break
            else:
                matched = self._compare(snapshot_values.get(cond.key, ""), cond.op, cond.value)
            if not matched:
                return False
        return True

    def match_snapshot(self, snapshot: ContextSnapshot, state_chain: Optional[Sequence[StateChainEntry]] = None) -> List[ScenarioMatch]:
        chain = list(state_chain or [])
        matches: List[ScenarioMatch] = []
        for scenario in self.scenarios:
            if not self._check_conditions(snapshot, scenario.conditions):
                continue
            if not self._check_preconditions(snapshot, chain, scenario):
                continue
            matches.append(
                ScenarioMatch(
                    scenario_id=scenario.scenario_id,
                    name=scenario.scenario_name,
                    confidence=1.0,
                    category=scenario.category,
                    ro_actions=list(scenario.ro_actions),
                    app_action=scenario.default_app_category,
                    rule_id=scenario.rule_id,
                )
            )
        return matches

    def match(
        self,
        state: PhysicalState,
        snapshot: Optional[ContextSnapshot] = None,
        state_chain: Optional[Sequence[StateChainEntry]] = None,
    ) -> Optional[ScenarioMatch]:
        active_snapshot = snapshot or self.build_snapshot(state, state_chain=state_chain)
        matches = self.match_snapshot(active_snapshot, state_chain=state_chain)
        return matches[0] if matches else None

    def ro_targets(self, scenario_match: Optional[ScenarioMatch]) -> List[float]:
        return ranked_distribution(RO_ACTIONS, [] if scenario_match is None else scenario_match.ro_actions)

    def app_targets(self, scenario_match: Optional[ScenarioMatch]) -> List[float]:
        return one_hot_distribution(APP_ACTIONS, "productivity" if scenario_match is None else scenario_match.app_action)

    def build_debug_info(
        self,
        state: PhysicalState,
        snapshot: Optional[ContextSnapshot] = None,
        state_chain: Optional[Sequence[StateChainEntry]] = None,
    ) -> dict:
        active_snapshot = snapshot or self.build_snapshot(state, state_chain=state_chain)
        match = self.match(state, snapshot=active_snapshot, state_chain=state_chain)
        return build_debug_labels(active_snapshot.state_current, match.scenario_id if match else None)
