"""Small training runner for the Python Dual-UCB prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .dual_ucb_model import DualUCBModel
    from .random_training_features import RandomFeatureFactory
except ImportError:
    from dual_ucb_model import DualUCBModel
    from random_training_features import RandomFeatureFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Python Dual-UCB prototype with synthetic data.")
    parser.add_argument("--steps", type=int, default=200, help="Number of synthetic training steps.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "dual_ucb_random_model.json"),
        help="Where to save the trained model JSON.",
    )
    parser.add_argument("--topk", type=int, default=3, help="How many predictions to print at the end.")
    args = parser.parse_args()

    factory = RandomFeatureFactory(seed=args.seed)
    model = DualUCBModel(seed=args.seed)

    for step in range(args.steps):
        sample = factory.build_sample()
        model.train_step(sample.feature, sample.ro_target, sample.app_target, learning_rate=0.01)

        # Give the target winner a light bandit reward to warm up UCB statistics.
        ro_idx = max(range(len(sample.ro_target)), key=sample.ro_target.__getitem__)
        app_idx = max(range(len(sample.app_target)), key=sample.app_target.__getitem__)
        model.reward(
            ro_action=model.ro_head.action_space[ro_idx],
            app_action=model.app_head.action_space[app_idx],
            reward_value=1.0,
            feature=sample.feature,
        )

        if (step + 1) % 50 == 0:
            print(
                "step="
                f"{step + 1} "
                f"state={sample.snapshot.state_current} "
                f"stateName={sample.state_name_en} "
                f"scenario={sample.scenario.scenario_id if sample.scenario else 'none'} "
                f"scenarioName={sample.scenario_name_en or 'none'}"
            )

    model.save_json(args.output)
    print(f"saved model to {args.output}")

    sample = factory.build_sample()
    predictions = model.predict(sample.feature, top_k=args.topk)
    print(
        f"debug state={sample.snapshot.state_current} "
        f"stateName={sample.state_name_en} "
        f"scenario={sample.scenario.scenario_id if sample.scenario else 'none'} "
        f"scenarioName={sample.scenario_name_en or 'none'}"
    )
    print("final RO predictions:")
    for item in predictions["ro"]:
        print(f"  {item.action} score={item.score:.4f} prior={item.prior_prob:.4f} ucb={item.ucb_score:.4f}")
    print("final APP predictions:")
    for item in predictions["app"]:
        print(f"  {item.action} score={item.score:.4f} prior={item.prior_prob:.4f} ucb={item.ucb_score:.4f}")


if __name__ == "__main__":
    main()

