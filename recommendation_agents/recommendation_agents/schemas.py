"""Runtime data schemas for training and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingEvent:
    scenario_id: str
    context: dict[str, Any]
    selected_action: str
    reward: float
    shown_actions: tuple[str, ...] | None = None
    propensity: float | None = None
    event_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingEvent":
        context = payload.get("context")
        if not isinstance(context, dict):
            raise ValueError("event.context must be an object")
        shown_actions_raw = payload.get("shown_actions")
        shown_actions = None
        if shown_actions_raw is not None:
            if not isinstance(shown_actions_raw, list) or not shown_actions_raw:
                raise ValueError("event.shown_actions must be a non-empty list when provided")
            shown_actions = tuple(dict.fromkeys(str(value) for value in shown_actions_raw))
        propensity = payload.get("propensity")
        if propensity is not None:
            propensity = float(propensity)
            if not 0.0 < propensity <= 1.0:
                raise ValueError("event.propensity must be in (0, 1]")
        return cls(
            scenario_id=str(payload["scenario_id"]),
            context=context,
            selected_action=str(payload["selected_action"]),
            reward=float(payload["reward"]),
            shown_actions=shown_actions,
            propensity=propensity,
            event_id=payload.get("event_id"),
        )


@dataclass(frozen=True)
class ScoreRequest:
    scenario_id: str
    context: dict[str, Any]
    shown_actions: tuple[str, ...] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoreRequest":
        context = payload.get("context")
        if not isinstance(context, dict):
            raise ValueError("score sample context must be an object")
        shown_actions_raw = payload.get("shown_actions")
        shown_actions = None
        if shown_actions_raw is not None:
            shown_actions = tuple(dict.fromkeys(str(value) for value in shown_actions_raw))
        return cls(
            scenario_id=str(payload["scenario_id"]),
            context=context,
            shown_actions=shown_actions,
        )

