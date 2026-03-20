"""Synthetic app-aligned samples for quick Dual-UCB training runs."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional

try:
    from .feature_space import DAY_TYPES, LOCATION_CLASSES, MOTION_CLASSES, PHONE_CLASSES, SOUND_CLASSES, TIME_SLOTS
    from .rule_engine import RuleEngine
    from .state_and_scenario import ContextSnapshot, LocInstanceAttr, PhysicalState, ScenarioMatch, StateChainEntry, get_scenario_name, get_state_name
    from .ucb_input_encoding import DualBanditFeatureEncoderPy
except ImportError:
    from feature_space import DAY_TYPES, LOCATION_CLASSES, MOTION_CLASSES, PHONE_CLASSES, SOUND_CLASSES, TIME_SLOTS
    from rule_engine import RuleEngine
    from state_and_scenario import ContextSnapshot, LocInstanceAttr, PhysicalState, ScenarioMatch, StateChainEntry, get_scenario_name, get_state_name
    from ucb_input_encoding import DualBanditFeatureEncoderPy


LIGHT_CLASSES = ["dark", "dim", "normal", "bright"]
TRI_STATES = ["true", "false", "unknown"]
NETWORK_TYPES = ["wifi", "cellular", "none", "other"]
ACTIVITY_STATES = ["sitting", "sleeping", "standing", "active", "unknown"]


@dataclass
class SyntheticSample:
    physical_state: PhysicalState
    state_chain: List[StateChainEntry]
    snapshot: ContextSnapshot
    inst_attr: LocInstanceAttr
    feature: List[float]
    ro_target: List[float]
    app_target: List[float]
    scenario: Optional[ScenarioMatch]
    state_name_en: str
    state_name_zh: str
    scenario_name_en: Optional[str]
    scenario_name_zh: Optional[str]


class RandomFeatureFactory:
    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.rule_engine = RuleEngine()
        self.encoder = DualBanditFeatureEncoderPy()

    def random_state(self) -> PhysicalState:
        return PhysicalState(
            time=self.rng.choice(TIME_SLOTS),
            location=self.rng.choice(LOCATION_CLASSES),
            motion=self.rng.choice(MOTION_CLASSES),
            phone=self.rng.choice(PHONE_CLASSES),
            light=self.rng.choice(LIGHT_CLASSES),
            sound=self.rng.choice(SOUND_CLASSES),
            day_type=self.rng.choice(DAY_TYPES),
        )

    def random_inst_attr(self) -> LocInstanceAttr:
        return LocInstanceAttr(
            familiarity=self.rng.random(),
            ownership=self.rng.random(),
            routine_level=self.rng.random(),
        )

    def scenario_shaped_state(self) -> PhysicalState:
        return self.rng.choice(
            [
                PhysicalState("morning", "work", "stationary", "on_desk", "normal", "normal", "workday"),
                PhysicalState("lunch", "work", "stationary", "in_use", "normal", "normal", "workday"),
                PhysicalState("evening", "home", "stationary", "on_desk", "dim", "quiet", "workday"),
                PhysicalState("late_night", "home", "stationary", "holding_lying", "dark", "quiet", "weekend"),
                PhysicalState("forenoon", "cafe", "stationary", "on_desk", "normal", "quiet", "workday"),
                PhysicalState("afternoon", "gym", "running", "in_pocket", "bright", "normal", "weekend"),
                PhysicalState("morning", "en_route", "transit", "in_pocket", "normal", "normal", "workday"),
            ]
        )

    def build_chain(self, current_state: PhysicalState) -> List[StateChainEntry]:
        chain_len = self.rng.randint(1, 4)
        now_ms = 1_700_000_000_000 + self.rng.randint(0, 500_000)
        entries: List[StateChainEntry] = []
        for idx in range(chain_len):
            state = current_state if idx == 0 else self.random_state()
            code = self.rule_engine.encode_state(state)
            duration_ms = self.rng.randint(60_000, 40 * 60_000)
            entries.append(
                StateChainEntry(
                    code=code,
                    start_time=now_ms - sum(item.duration_ms for item in entries) - duration_ms,
                    duration_ms=duration_ms,
                )
            )
        return entries

    def build_snapshot(self, state: PhysicalState, chain: List[StateChainEntry]) -> ContextSnapshot:
        hour_map = {
            "sleeping": 1,
            "dawn": 6,
            "morning": 8,
            "forenoon": 10,
            "lunch": 12,
            "afternoon": 15,
            "evening": 18,
            "night": 21,
            "late_night": 23,
        }
        snapshot = self.rule_engine.build_snapshot(
            state,
            hour=hour_map.get(state.time, 12),
            state_chain=chain,
            cal_hasUpcoming=self.rng.choice(TRI_STATES),
            cal_nextMinutes=str(self.rng.randint(0, 180)),
            cal_inMeeting=self.rng.choice(TRI_STATES),
            cal_nextLocation=self.rng.choice(LOCATION_CLASSES),
            cal_eventCount=str(self.rng.randint(0, 8)),
            wifiLost=self.rng.choice(TRI_STATES),
            wifiLostCategory=self.rng.choice(LOCATION_CLASSES),
            batteryLevel=str(self.rng.randint(15, 100)),
            isCharging=self.rng.choice(TRI_STATES[:2]),
            networkType=self.rng.choice(NETWORK_TYPES),
            transportMode=self.rng.choice(MOTION_CLASSES),
            activityState=self.rng.choice(ACTIVITY_STATES),
            activityDuration=str(self.rng.randint(0, 4 * 60 * 60)),
            prevActivityState=self.rng.choice(ACTIVITY_STATES),
            sms_delivery_pending=self.rng.choice(TRI_STATES),
            sms_train_pending=self.rng.choice(TRI_STATES),
            sms_flight_pending=self.rng.choice(TRI_STATES),
            sms_hotel_pending=self.rng.choice(TRI_STATES),
            sms_movie_pending=self.rng.choice(TRI_STATES),
            sms_hospital_pending=self.rng.choice(TRI_STATES),
            sms_ride_pending=self.rng.choice(TRI_STATES),
        )
        return snapshot

    def build_sample(self) -> SyntheticSample:
        inst_attr = self.random_inst_attr()
        current_state = self.scenario_shaped_state() if self.rng.random() < 0.7 else self.random_state()
        chain = self.build_chain(current_state)
        snapshot = self.build_snapshot(current_state, chain)
        scenario = self.rule_engine.match(current_state, snapshot=snapshot, state_chain=chain)
        if scenario is not None:
            snapshot.ps_scenario = scenario.scenario_id
        feature = self.encoder.encode_shared(snapshot, scenario_id=scenario.scenario_id if scenario else None)
        ro_target = self.rule_engine.ro_targets(scenario)
        app_target = self.rule_engine.app_targets(scenario)
        return SyntheticSample(
            physical_state=current_state,
            state_chain=chain,
            snapshot=snapshot,
            inst_attr=inst_attr,
            feature=feature,
            ro_target=ro_target,
            app_target=app_target,
            scenario=scenario,
            state_name_en=get_state_name(snapshot.state_current, "en"),
            state_name_zh=get_state_name(snapshot.state_current, "zh"),
            scenario_name_en=get_scenario_name(scenario.scenario_id, "en") if scenario else None,
            scenario_name_zh=get_scenario_name(scenario.scenario_id, "zh") if scenario else None,
        )

