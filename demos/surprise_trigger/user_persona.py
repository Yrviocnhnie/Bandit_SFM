"""User persona definition for the surprise-trigger demo.

A sample is ROUTINE for the user if every feature in the persona's allowed set
matches and the hour falls in the allowed inclusive range. Everything else is
non-routine (used as the triplet negative pool).

Alex - regular office worker:
  ARRIVE_OFFICE: transit/walk commute, 7-9am workday, base office_arriving state.
  HOME_EVENING:  18-20h workday, base home_evening state (no dark/lying/noisy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class RoutineSpec:
    scenario_id: str
    allowed_cat: Dict[str, frozenset]
    hour_range: Tuple[int, int] = (0, 23)

    def matches(self, features: Dict[str, Any]) -> bool:
        for key, allowed in self.allowed_cat.items():
            if features.get(key) not in allowed:
                return False
        try:
            h = int(features.get("hour", -1))
        except (TypeError, ValueError):
            return False
        lo, hi = self.hour_range
        return lo <= h <= hi


ALEX_ARRIVE_OFFICE = RoutineSpec(
    scenario_id="ARRIVE_OFFICE",
    allowed_cat={
        "state_current": frozenset({"office_arriving"}),
        "precondition": frozenset({"commuting_transit_out", "commuting_walk_out"}),
        "ps_time": frozenset({"morning", "dawn"}),
        "ps_dayType": frozenset({"workday"}),
        "ps_motion": frozenset({"stationary", "walking"}),
        "ps_phone": frozenset({"in_pocket", "on_desk", "face_up"}),
        "ps_light": frozenset({"normal", "bright"}),
        "ps_sound": frozenset({"normal", "quiet"}),
        "networkType": frozenset({"wifi", "cellular"}),
        "isCharging": frozenset({0}),
    },
    hour_range=(7, 9),
)

ALEX_HOME_EVENING = RoutineSpec(
    scenario_id="HOME_EVENING",
    allowed_cat={
        "state_current": frozenset({"home_evening"}),
        "precondition": frozenset(
            {"commuting_transit_home", "commuting_walk_home", "commuting_drive_home"}
        ),
        "ps_time": frozenset({"evening", "night"}),
        "ps_dayType": frozenset({"workday"}),
        "ps_motion": frozenset({"stationary", "walking"}),
        "ps_phone": frozenset({"in_use", "face_up", "on_desk"}),
        "ps_light": frozenset({"dim", "normal"}),
        "ps_sound": frozenset({"quiet", "normal"}),
        "networkType": frozenset({"wifi"}),
        "isCharging": frozenset({0, 1}),
    },
    hour_range=(18, 20),
)

ROUTINES: Dict[str, RoutineSpec] = {
    "ARRIVE_OFFICE": ALEX_ARRIVE_OFFICE,
    "HOME_EVENING": ALEX_HOME_EVENING,
}
