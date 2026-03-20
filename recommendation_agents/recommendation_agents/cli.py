"""Command-line entrypoints for the V0 recommendation agent scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommendation_agents.catalog import build_app_metadata_from_catalog_markdown, build_ro_metadata_from_catalog_markdown
from recommendation_agents.raw_synthetic import convert_raw_sequence_to_v0, convert_raw_sequence_to_v0_app
from recommendation_agents.schemas import ScoreRequest
from recommendation_agents.trainer import choose_v0, score_v0, train_v0


def _display_action_id(model_action_id: str) -> str:
    if "::" in model_action_id:
        return model_action_id.split("::", 1)[1]
    return model_action_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recommendation-agents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-v0", help="Replay-train the V0 masked LinUCB model")
    train_parser.add_argument("--metadata", required=True, help="Path to metadata JSON")
    train_parser.add_argument("--samples", required=True, help="Path to JSONL training samples")
    train_parser.add_argument("--output", required=True, help="Directory to write model artifacts")
    train_parser.add_argument("--alpha", type=float, default=0.15, help="LinUCB exploration coefficient")
    train_parser.add_argument(
        "--default-bonus",
        type=float,
        default=0.75,
        help="Additive prior bonus for the scenario default action",
    )
    train_parser.add_argument("--l2", type=float, default=1.0, help="L2 regularization for LinUCB")
    train_parser.add_argument(
        "--device",
        default="auto",
        help="Execution device. Examples: auto, cpu, cuda, cuda:0. Uses CPU when GPU is unavailable.",
    )

    score_parser = subparsers.add_parser("score-v0", help="Rank actions for one sample with a trained artifact")
    score_parser.add_argument("--artifact", required=True, help="Artifact directory produced by train-v0")
    score_parser.add_argument(
        "--metadata",
        required=False,
        help="Path to metadata JSON. Defaults to metadata.snapshot.json inside the artifact directory.",
    )
    score_parser.add_argument("--sample", required=True, help="Path to one JSON score request")
    score_parser.add_argument("--top-k", type=int, default=5, help="How many ranked actions to return")
    score_parser.add_argument("--device", default="auto", help="Execution device for scoring, e.g. cpu or cuda")

    choose_parser = subparsers.add_parser("choose-v0", help="Choose one action for one sample with optional epsilon-randomness")
    choose_parser.add_argument("--artifact", required=True, help="Artifact directory produced by train-v0")
    choose_parser.add_argument(
        "--metadata",
        required=False,
        help="Path to metadata JSON. Defaults to metadata.snapshot.json inside the artifact directory.",
    )
    choose_parser.add_argument("--sample", required=True, help="Path to one JSON score request")
    choose_parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="With probability epsilon, choose uniformly from the masked candidate set instead of the top score",
    )
    choose_parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible smoke tests")
    choose_parser.add_argument("--device", default="auto", help="Execution device for choose, e.g. cpu or cuda")

    convert_parser = subparsers.add_parser(
        "convert-v0-raw",
        help="Convert colleague synthetic sequence JSONL into one-row-per-recommendation V0 events",
    )
    convert_parser.add_argument("--input", required=True, help="Path to the raw synthetic JSONL")
    convert_parser.add_argument("--output-samples", required=True, help="Path to write converted V0 JSONL samples")
    convert_parser.add_argument(
        "--output-metadata",
        required=False,
        help="Optional path to write metadata inferred from observed actions in the raw file",
    )
    convert_parser.add_argument(
        "--reward",
        type=float,
        default=1.0,
        help="Reward assigned to each kept synthetic recommendation row",
    )

    convert_app_parser = subparsers.add_parser(
        "convert-v0-raw-app",
        help="Convert colleague synthetic sequence JSONL into one-row-per-recommendation V0 app events",
    )
    convert_app_parser.add_argument("--input", required=True, help="Path to the raw synthetic JSONL")
    convert_app_parser.add_argument("--output-samples", required=True, help="Path to write converted V0 app JSONL samples")
    convert_app_parser.add_argument(
        "--output-metadata",
        required=False,
        help="Optional path to write app metadata inferred from observed app categories in the raw file",
    )
    convert_app_parser.add_argument(
        "--reward",
        type=float,
        default=1.0,
        help="Reward assigned to each kept synthetic app recommendation row",
    )

    ro_metadata_parser = subparsers.add_parser(
        "build-ro-metadata",
        help="Parse the current scenario/action catalog markdown into R/O training metadata",
    )
    ro_metadata_parser.add_argument(
        "--input-markdown",
        required=True,
        help="Path to the current scenario/action catalog markdown",
    )
    ro_metadata_parser.add_argument("--output-metadata", required=True, help="Path to write the parsed metadata JSON")

    app_metadata_parser = subparsers.add_parser(
        "build-app-metadata",
        help="Parse the current scenario/action catalog markdown into app-agent metadata",
    )
    app_metadata_parser.add_argument(
        "--input-markdown",
        required=True,
        help="Path to the current scenario/action catalog markdown",
    )
    app_metadata_parser.add_argument("--output-metadata", required=True, help="Path to write the parsed app metadata JSON")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train-v0":
        metrics = train_v0(
            metadata_path=args.metadata,
            samples_path=args.samples,
            output_dir=args.output,
            alpha=args.alpha,
            default_bonus=args.default_bonus,
            l2=args.l2,
            device=args.device,
        )
        print(json.dumps(metrics.__dict__, indent=2, sort_keys=True))
        return

    if args.command == "score-v0":
        request = ScoreRequest.from_dict(json.loads(Path(args.sample).read_text()))
        ranked = score_v0(
            artifact_dir=args.artifact,
            metadata_path=args.metadata,
            request=request,
            top_k=args.top_k,
            device=args.device,
        )
        print(
            json.dumps(
                [
                    {
                        "action_id": _display_action_id(item.action_id),
                        "model_action_id": item.action_id,
                        "score": item.score,
                        "mean_reward": item.mean_reward,
                        "uncertainty": item.uncertainty,
                        "default_bonus": item.default_bonus,
                    }
                    for item in ranked
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "choose-v0":
        request = ScoreRequest.from_dict(json.loads(Path(args.sample).read_text()))
        chosen = choose_v0(
            artifact_dir=args.artifact,
            metadata_path=args.metadata,
            request=request,
            epsilon=args.epsilon,
            seed=args.seed,
            device=args.device,
        )
        print(
            json.dumps(
                {
                    "action_id": _display_action_id(chosen.action_id),
                    "model_action_id": chosen.action_id,
                    "score": chosen.score,
                    "mean_reward": chosen.mean_reward,
                    "uncertainty": chosen.uncertainty,
                    "default_bonus": chosen.default_bonus,
                    "epsilon": args.epsilon,
                    "seed": args.seed,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "convert-v0-raw":
        summary = convert_raw_sequence_to_v0(
            input_path=args.input,
            output_samples_path=args.output_samples,
            output_metadata_path=args.output_metadata,
            reward=args.reward,
        )
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
        return

    if args.command == "convert-v0-raw-app":
        summary = convert_raw_sequence_to_v0_app(
            input_path=args.input,
            output_samples_path=args.output_samples,
            output_metadata_path=args.output_metadata,
            reward=args.reward,
        )
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
        return

    if args.command == "build-ro-metadata":
        summary = build_ro_metadata_from_catalog_markdown(
            markdown_path=args.input_markdown,
            output_metadata_path=args.output_metadata,
        )
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
        return

    if args.command == "build-app-metadata":
        summary = build_app_metadata_from_catalog_markdown(
            markdown_path=args.input_markdown,
            output_metadata_path=args.output_metadata,
        )
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
