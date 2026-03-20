"""Compatibility entrypoint for the split Python UCB prototype."""

from __future__ import annotations

import json

try:
    from .dual_ucb_model import DualUCBModel, load_cpp_weight_arrays
    from .random_training_features import RandomFeatureFactory
    from .rule_engine import RuleEngine
    from .state_and_scenario import ContextSnapshot, LocInstanceAttr, PhysicalState, ScenarioDefinition, ScenarioMatch, StateChainEntry
    from .ucb_input_encoding import DualBanditFeatureEncoderPy, parse_state_chain_json
except ImportError:
    from dual_ucb_model import DualUCBModel, load_cpp_weight_arrays
    from random_training_features import RandomFeatureFactory
    from rule_engine import RuleEngine
    from state_and_scenario import ContextSnapshot, LocInstanceAttr, PhysicalState, ScenarioDefinition, ScenarioMatch, StateChainEntry
    from ucb_input_encoding import DualBanditFeatureEncoderPy, parse_state_chain_json


__all__ = [
    "DualUCBModel",
    "DualBanditFeatureEncoderPy",
    "RuleEngine",
    "ContextSnapshot",
    "LocInstanceAttr",
    "PhysicalState",
    "ScenarioDefinition",
    "ScenarioMatch",
    "StateChainEntry",
    "parse_state_chain_json",
    "load_cpp_weight_arrays",
]


def demo() -> None:
    factory = RandomFeatureFactory(seed=7)
    sample = factory.build_sample()
    model = DualUCBModel(seed=7)
    predictions = model.predict(sample.feature, top_k=3)
    print(
        json.dumps(
            {
                "state_code": sample.snapshot.state_current,
                "state_name_en": sample.state_name_en,
                "state_name_zh": sample.state_name_zh,
                "scenario": sample.scenario.scenario_id if sample.scenario else None,
                "scenario_name_en": sample.scenario_name_en,
                "scenario_name_zh": sample.scenario_name_zh,
                "ro": [item.action for item in predictions["ro"]],
                "app": [item.action for item in predictions["app"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    demo()
