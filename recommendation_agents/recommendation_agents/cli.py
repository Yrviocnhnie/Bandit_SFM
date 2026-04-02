"""Command-line entrypoints for the V0 recommendation agent scaffold."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from recommendation_agents.catalog import build_app_metadata_from_catalog_markdown, build_ro_metadata_from_catalog_markdown
from recommendation_agents.metadata import BanditMetadata
from recommendation_agents.raw_synthetic import (
    convert_raw_sequence_to_v0,
    convert_raw_sequence_to_v0_app,
    convert_raw_sequence_to_v6_expanded_app,
    convert_raw_sequence_to_v6_expanded_ro,
)
from recommendation_agents.schemas import ScoreRequest
from recommendation_agents.trainer import choose_v0, score_v0, train_v0, train_v0_dual_from_raw, train_v0_from_raw
from recommendation_agents.workflows import (
    eval_soft_scenarios_both,
    eval_v0_both,
    mine_v6_hard_negative_candidates,
    prepare_v0_data,
    render_hard_negative_candidates_markdown,
    run_v6_plan_a,
    run_v6_plan_all_data,
    run_v6_plan_b,
    train_v0_both,
)


def _display_action_id(model_action_id: str) -> str:
    if "::" in model_action_id:
        return model_action_id.split("::", 1)[1]
    return model_action_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recommendation-agents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-v0", help="Replay-train the V0 shared-action LinUCB model")
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
    train_parser.add_argument("--epochs", type=int, default=1, help="How many times to replay the sample file")
    train_parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Log interval training metrics every N samples",
    )
    train_parser.add_argument(
        "--device",
        default="auto",
        help="Execution device. Examples: auto, cpu, cuda, cuda:0. Uses CPU when GPU is unavailable.",
    )
    train_parser.add_argument(
        "--track-train-hit-rate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to rank before each update to compute pre-update train top-1 metrics. Slower when enabled.",
    )
    train_parser.add_argument(
        "--model-type",
        choices=("disjoint", "shared-linear", "neural-linear", "neural-scorer", "neural-ucb-lite", "neural-ucb", "neural-ucb-direct"),
        default="disjoint",
        help="Bandit model type to train",
    )

    train_raw_parser = subparsers.add_parser(
        "train-v0-raw",
        help="Convert raw JSONL and train the V0 shared-action LinUCB model in one step",
    )
    train_raw_parser.add_argument("--input", required=True, help="Path to the raw synthetic JSONL")
    train_raw_parser.add_argument("--output", required=True, help="Directory to write model artifacts")
    train_raw_parser.add_argument(
        "--metadata",
        required=False,
        help="Optional metadata JSON. If omitted, metadata is inferred from the raw file.",
    )
    train_raw_parser.add_argument(
        "--label-type",
        choices=("ro", "app"),
        default="ro",
        help="Which label column to train on: R/O actions or app categories",
    )
    train_raw_parser.add_argument(
        "--reward",
        type=float,
        default=1.0,
        help="Reward assigned to each kept raw row during conversion",
    )
    train_raw_parser.add_argument("--alpha", type=float, default=0.15, help="LinUCB exploration coefficient")
    train_raw_parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Log interval training metrics every N samples",
    )
    train_raw_parser.add_argument(
        "--default-bonus",
        type=float,
        default=0.75,
        help="Additive prior bonus for the scenario default action",
    )
    train_raw_parser.add_argument("--l2", type=float, default=1.0, help="L2 regularization for LinUCB")
    train_raw_parser.add_argument("--epochs", type=int, default=1, help="How many times to replay the sample file")
    train_raw_parser.add_argument(
        "--device",
        default="auto",
        help="Execution device. Examples: auto, cpu, cuda, cuda:0. Uses CPU when GPU is unavailable.",
    )
    train_raw_parser.add_argument(
        "--track-train-hit-rate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to rank before each update to compute pre-update train top-1 metrics. Slower when enabled.",
    )
    train_raw_parser.add_argument(
        "--model-type",
        choices=("disjoint", "shared-linear", "neural-linear", "neural-scorer", "neural-ucb-lite", "neural-ucb", "neural-ucb-direct"),
        default="disjoint",
        help="Bandit model type to train",
    )

    train_dual_raw_parser = subparsers.add_parser(
        "train-v0-raw-dual",
        help="Train interleaved R/O and app LinUCB models from one shuffled raw JSONL",
    )
    train_dual_raw_parser.add_argument("--input", required=True, help="Path to the raw synthetic JSONL")
    train_dual_raw_parser.add_argument("--output", required=True, help="Directory to write the dual model artifacts")
    train_dual_raw_parser.add_argument(
        "--ro-metadata",
        required=True,
        help="Metadata JSON for the R/O model action space",
    )
    train_dual_raw_parser.add_argument(
        "--app-metadata",
        required=True,
        help="Metadata JSON for the app model action space",
    )
    train_dual_raw_parser.add_argument(
        "--reward",
        type=float,
        default=1.0,
        help="Reward assigned to each kept raw row during conversion",
    )
    train_dual_raw_parser.add_argument("--alpha", type=float, default=0.15, help="Starting LinUCB exploration coefficient")
    train_dual_raw_parser.add_argument(
        "--alpha-end",
        type=float,
        default=0.01,
        help="Final LinUCB exploration coefficient after linear decay",
    )
    train_dual_raw_parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log after every N train samples during the interleaved loop",
    )
    train_dual_raw_parser.add_argument("--train-window", type=int, default=100, help="How many shuffled rows to train on per cycle")
    train_dual_raw_parser.add_argument("--eval-window", type=int, default=10, help="How many shuffled rows to eval on per cycle")
    train_dual_raw_parser.add_argument("--shuffle-seed", type=int, default=0, help="Seed used to shuffle the shared raw rows")
    train_dual_raw_parser.add_argument(
        "--tensorboard-logdir",
        required=False,
        help="Optional TensorBoard log directory. Defaults to <output>/tensorboard",
    )
    train_dual_raw_parser.add_argument(
        "--default-bonus",
        type=float,
        default=0.75,
        help="Additive prior bonus for the scenario default action",
    )
    train_dual_raw_parser.add_argument("--l2", type=float, default=1.0, help="L2 regularization for LinUCB")
    train_dual_raw_parser.add_argument(
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
        help="With probability epsilon, choose uniformly from the candidate set instead of the top score",
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

    convert_v6_ro_parser = subparsers.add_parser(
        "convert-v6-raw-ro",
        help="Expand raw JSONL into V6 multi-action R/O training samples using most-relevant/plausible/irrelevant tiers",
    )
    convert_v6_ro_parser.add_argument("--input", required=True, help="Path to the raw synthetic JSONL")
    convert_v6_ro_parser.add_argument("--output-samples", required=True, help="Path to write expanded R/O samples")
    convert_v6_ro_parser.add_argument(
        "--relevance-markdown",
        default="docs/scenario_recommendation_actions_v6.md",
        help="Path to the V6 scenario relevance markdown",
    )
    convert_v6_ro_parser.add_argument("--most-relevant-reward", type=float, default=1.0)
    convert_v6_ro_parser.add_argument("--plausible-reward", type=float, default=0.6)
    convert_v6_ro_parser.add_argument("--irrelevant-reward", type=float, default=0.0)
    convert_v6_ro_parser.add_argument("--most-relevant-repeat", type=int, default=1)
    convert_v6_ro_parser.add_argument("--plausible-repeat", type=int, default=1)
    convert_v6_ro_parser.add_argument("--irrelevant-repeat", type=int, default=1)
    convert_v6_ro_parser.add_argument("--all-actions-metadata", required=False, help="Optional metadata JSON used to enumerate the full global action space")
    convert_v6_ro_parser.add_argument(
        "--other-zero-mode",
        choices=("none", "exclude-most-plausible", "exclude-most-only", "exclude-all-labeled"),
        default="none",
        help="Optionally add all remaining global actions as 0-reward samples after excluding most-relevant+plausible or excluding only most-relevant",
    )

    convert_v6_app_parser = subparsers.add_parser(
        "convert-v6-raw-app",
        help="Expand raw JSONL into V6 multi-action app training samples using most-relevant/plausible/irrelevant tiers",
    )
    convert_v6_app_parser.add_argument("--input", required=True, help="Path to the raw synthetic JSONL")
    convert_v6_app_parser.add_argument("--output-samples", required=True, help="Path to write expanded app samples")
    convert_v6_app_parser.add_argument(
        "--relevance-markdown",
        default="docs/scenario_recommendation_actions_v6.md",
        help="Path to the V6 scenario relevance markdown",
    )
    convert_v6_app_parser.add_argument("--most-relevant-reward", type=float, default=1.0)
    convert_v6_app_parser.add_argument("--plausible-reward", type=float, default=0.6)
    convert_v6_app_parser.add_argument("--irrelevant-reward", type=float, default=0.0)
    convert_v6_app_parser.add_argument("--most-relevant-repeat", type=int, default=1)
    convert_v6_app_parser.add_argument("--plausible-repeat", type=int, default=1)
    convert_v6_app_parser.add_argument("--irrelevant-repeat", type=int, default=1)
    convert_v6_app_parser.add_argument("--all-actions-metadata", required=False, help="Optional metadata JSON used to enumerate the full global app action space")
    convert_v6_app_parser.add_argument(
        "--other-zero-mode",
        choices=("none", "exclude-most-plausible", "exclude-most-only", "exclude-all-labeled"),
        default="none",
        help="Optionally add all remaining global actions as 0-reward samples after excluding most-relevant+plausible or excluding only most-relevant",
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

    prepare_parser = subparsers.add_parser(
        "prepare-v0-data",
        help="One-line workflow: build metadata, split raw JSONL, and convert train/test samples for both agents",
    )
    prepare_parser.add_argument("--input", required=True, help="Path to the raw synthetic JSONL")
    prepare_parser.add_argument(
        "--catalog-markdown",
        required=True,
        help="Path to the v5 scenario/action catalog markdown",
    )
    prepare_parser.add_argument("--output-dir", required=True, help="Directory to write prepared data artifacts")
    prepare_parser.add_argument("--reward", type=float, default=1.0, help="Reward assigned during conversion")
    prepare_parser.add_argument("--test-ratio", type=float, default=0.2, help="Episode-level test split ratio")

    train_both_parser = subparsers.add_parser(
        "train-v0-both",
        help="One-line workflow: train both R/O and App V0 models from a prepared data directory",
    )
    train_both_parser.add_argument("--data-dir", required=True, help="Prepared data directory from prepare-v0-data")
    train_both_parser.add_argument("--alpha", type=float, default=0.05, help="LinUCB exploration coefficient")
    train_both_parser.add_argument(
        "--default-bonus",
        type=float,
        default=0.75,
        help="Additive prior bonus for the scenario default action",
    )
    train_both_parser.add_argument("--l2", type=float, default=1.0, help="L2 regularization for LinUCB")
    train_both_parser.add_argument("--epochs", type=int, default=1, help="How many times to replay the train samples")
    train_both_parser.add_argument("--progress-every", type=int, default=1000, help="Training log interval")
    train_both_parser.add_argument("--device", default="auto", help="Execution device. Examples: auto, cpu, cuda")
    train_both_parser.add_argument(
        "--track-train-hit-rate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to rank before each update to compute pre-update train top-1 metrics. Slower when enabled.",
    )
    train_both_parser.add_argument(
        "--model-type",
        choices=("disjoint", "shared-linear", "neural-linear", "neural-scorer", "neural-ucb-lite", "neural-ucb", "neural-ucb-direct"),
        default="disjoint",
        help="Bandit model type to train",
    )

    eval_both_parser = subparsers.add_parser(
        "eval-v0-both",
        help="One-line workflow: evaluate both R/O and App models using the v6 relevance-style top-k metrics",
    )
    eval_both_parser.add_argument("--data-dir", required=True, help="Prepared data directory containing trained artifacts")
    eval_both_parser.add_argument(
        "--catalog-markdown",
        default="docs/scenario_recommendation_actions_v6.md",
        help="Path to the v6 scenario relevance catalog used for evaluation",
    )
    eval_both_parser.add_argument("--top-k", type=int, default=3, help="Top-k used for evaluation")
    eval_both_parser.add_argument("--progress-every", type=int, default=1000, help="Evaluation log interval")
    eval_both_parser.add_argument("--device", default="auto", help="Execution device. Examples: auto, cpu, cuda")

    eval_soft_parser = subparsers.add_parser(
        "eval-soft-scenarios",
        help="Evaluate a trained dual model on evaluation-only in-between scenarios using top-3 and top-5 hit counts",
    )
    eval_soft_parser.add_argument("--data-dir", required=True, help="Prepared/trained data directory containing ro_model/app_model")
    eval_soft_parser.add_argument("--input-raw", required=True, help="Raw JSONL containing in-between soft scenarios")
    eval_soft_parser.add_argument(
        "--spec-markdown",
        required=True,
        help="Markdown defining the soft scenarios and their RO top10 / App top5 recommendation lists",
    )
    eval_soft_parser.add_argument("--progress-every", type=int, default=1000, help="Evaluation log interval")
    eval_soft_parser.add_argument("--device", default="auto", help="Execution device. Examples: auto, cpu, cuda")

    plan_a_parser = subparsers.add_parser(
        "run-v6-plan-a",
        help="Reuse an existing prepared split, expand train only with V6 relevance labels, then train and evaluate both models",
    )
    plan_a_parser.add_argument("--input-data-dir", required=True, help="Existing prepared data directory to reuse")
    plan_a_parser.add_argument("--output-dir", required=True, help="Directory to write the new Plan A artifacts")
    plan_a_parser.add_argument(
        "--relevance-markdown",
        default="docs/scenario_recommendation_actions_v6.md",
        help="Path to the V6 scenario relevance markdown",
    )
    plan_a_parser.add_argument("--most-relevant-reward", type=float, default=1.0)
    plan_a_parser.add_argument("--plausible-reward", type=float, default=0.6)
    plan_a_parser.add_argument("--irrelevant-reward", type=float, default=0.0)
    plan_a_parser.add_argument("--most-relevant-repeat", type=int, default=1)
    plan_a_parser.add_argument("--plausible-repeat", type=int, default=1)
    plan_a_parser.add_argument("--irrelevant-repeat", type=int, default=1)
    plan_a_parser.add_argument(
        "--other-zero-mode",
        choices=("none", "exclude-most-plausible", "exclude-most-only", "exclude-all-labeled"),
        default="none",
        help="Optionally expand all remaining global actions as 0-reward samples",
    )
    plan_a_parser.add_argument("--alpha", type=float, default=0.05)
    plan_a_parser.add_argument("--default-bonus", type=float, default=0.0)
    plan_a_parser.add_argument("--l2", type=float, default=1.0)
    plan_a_parser.add_argument("--epochs", type=int, default=1)
    plan_a_parser.add_argument("--top-k", type=int, default=3)
    plan_a_parser.add_argument("--progress-every", type=int, default=1000)
    plan_a_parser.add_argument("--device", default="auto")
    plan_a_parser.add_argument("--model-type", choices=("disjoint", "shared-linear", "neural-linear", "neural-scorer", "neural-ucb-lite", "neural-ucb", "neural-ucb-direct"), default="disjoint")
    plan_a_parser.add_argument(
        "--track-train-hit-rate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to rank before each update to compute pre-update train top-1 metrics. Slower when enabled.",
    )

    plan_b_parser = subparsers.add_parser(
        "run-v6-plan-b",
        help="Create a scenario-stratified raw split, expand train only with V6 relevance labels, then train and evaluate both models",
    )
    plan_b_parser.add_argument("--input", required=True, help="Path to the raw synthetic JSONL")
    plan_b_parser.add_argument(
        "--catalog-markdown",
        required=True,
        help="Path to the catalog markdown used to build metadata",
    )
    plan_b_parser.add_argument("--output-dir", required=True, help="Directory to write the new Plan B artifacts")
    plan_b_parser.add_argument(
        "--relevance-markdown",
        default="docs/scenario_recommendation_actions_v6.md",
        help="Path to the V6 scenario relevance markdown",
    )
    plan_b_parser.add_argument("--test-ratio", type=float, default=0.2)
    plan_b_parser.add_argument("--most-relevant-reward", type=float, default=1.0)
    plan_b_parser.add_argument("--plausible-reward", type=float, default=0.6)
    plan_b_parser.add_argument("--irrelevant-reward", type=float, default=0.0)
    plan_b_parser.add_argument("--most-relevant-repeat", type=int, default=1)
    plan_b_parser.add_argument("--plausible-repeat", type=int, default=1)
    plan_b_parser.add_argument("--irrelevant-repeat", type=int, default=1)
    plan_b_parser.add_argument(
        "--other-zero-mode",
        choices=("none", "exclude-most-plausible", "exclude-most-only", "exclude-all-labeled"),
        default="none",
        help="Optionally expand all remaining global actions as 0-reward samples",
    )
    plan_b_parser.add_argument("--alpha", type=float, default=0.05)
    plan_b_parser.add_argument("--default-bonus", type=float, default=0.0)
    plan_b_parser.add_argument("--l2", type=float, default=1.0)
    plan_b_parser.add_argument("--epochs", type=int, default=1)
    plan_b_parser.add_argument("--top-k", type=int, default=3)
    plan_b_parser.add_argument("--progress-every", type=int, default=1000)
    plan_b_parser.add_argument("--device", default="auto")
    plan_b_parser.add_argument("--model-type", choices=("disjoint", "shared-linear", "neural-linear", "neural-scorer", "neural-ucb-lite", "neural-ucb", "neural-ucb-direct"), default="disjoint")
    plan_b_parser.add_argument(
        "--track-train-hit-rate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to rank before each update to compute pre-update train top-1 metrics. Slower when enabled.",
    )

    plan_all_data_parser = subparsers.add_parser(
        "run-v6-plan-all-data",
        help="Use the full raw dataset for both train and test, expand train only with V6 relevance labels, then train and evaluate both models",
    )
    plan_all_data_parser.add_argument("--input", required=True, help="Path to the raw synthetic JSONL")
    plan_all_data_parser.add_argument(
        "--catalog-markdown",
        required=True,
        help="Path to the catalog markdown used to build metadata",
    )
    plan_all_data_parser.add_argument("--output-dir", required=True, help="Directory to write the all-data artifacts")
    plan_all_data_parser.add_argument(
        "--relevance-markdown",
        default="docs/scenario_recommendation_actions_v6.md",
        help="Path to the V6 scenario relevance markdown",
    )
    plan_all_data_parser.add_argument("--most-relevant-reward", type=float, default=1.0)
    plan_all_data_parser.add_argument("--plausible-reward", type=float, default=0.6)
    plan_all_data_parser.add_argument("--irrelevant-reward", type=float, default=0.0)
    plan_all_data_parser.add_argument("--most-relevant-repeat", type=int, default=1)
    plan_all_data_parser.add_argument("--plausible-repeat", type=int, default=1)
    plan_all_data_parser.add_argument("--irrelevant-repeat", type=int, default=1)
    plan_all_data_parser.add_argument(
        "--other-zero-mode",
        choices=("none", "exclude-most-plausible", "exclude-most-only", "exclude-all-labeled"),
        default="none",
        help="Optionally expand all remaining global actions as 0-reward samples",
    )
    plan_all_data_parser.add_argument("--alpha", type=float, default=0.05)
    plan_all_data_parser.add_argument("--default-bonus", type=float, default=0.0)
    plan_all_data_parser.add_argument("--l2", type=float, default=1.0)
    plan_all_data_parser.add_argument("--epochs", type=int, default=1)
    plan_all_data_parser.add_argument("--top-k", type=int, default=3)
    plan_all_data_parser.add_argument("--progress-every", type=int, default=1000)
    plan_all_data_parser.add_argument("--device", default="auto")
    plan_all_data_parser.add_argument("--model-type", choices=("disjoint", "shared-linear", "neural-linear", "neural-scorer", "neural-ucb-lite", "neural-ucb", "neural-ucb-direct"), default="disjoint")
    plan_all_data_parser.add_argument(
        "--track-train-hit-rate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to rank before each update to compute pre-update train top-1 metrics. Slower when enabled.",
    )

    mine_hardneg_parser = subparsers.add_parser(
        "mine-v6-hard-negatives",
        help="Run a trained model over raw train/dev rows, then propose per-scenario hard-negative candidates from high-frequency non-acceptable top-k predictions",
    )
    mine_hardneg_parser.add_argument("--artifact-dir", required=True, help="Trained model artifact directory")
    mine_hardneg_parser.add_argument("--metadata", required=True, help="Metadata JSON matching the artifact")
    mine_hardneg_parser.add_argument("--input-raw", required=True, help="Raw JSONL to mine on, usually train.raw.jsonl or a held-out dev.raw.jsonl")
    mine_hardneg_parser.add_argument(
        "--catalog-markdown",
        default="docs/scenario_recommendation_actions_v6.md",
        help="Path to the V6 relevance catalog",
    )
    mine_hardneg_parser.add_argument("--label-namespace", choices=("ro", "app"), required=True)
    mine_hardneg_parser.add_argument("--output-json", required=True, help="Path to write the mining summary JSON")
    mine_hardneg_parser.add_argument("--output-markdown", required=True, help="Path to write the candidate markdown table")
    mine_hardneg_parser.add_argument("--top-k", type=int, default=6, help="Mine mistakes from the model's top-k predictions")
    mine_hardneg_parser.add_argument("--max-candidates-per-scenario", type=int, default=5)
    mine_hardneg_parser.add_argument("--min-count", type=int, default=1)
    mine_hardneg_parser.add_argument("--min-rate", type=float, default=0.0)
    mine_hardneg_parser.add_argument("--progress-every", type=int, default=1000)
    mine_hardneg_parser.add_argument("--device", default="auto")

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
            epochs=args.epochs,
            device=args.device,
            progress_every=args.progress_every,
            track_train_hit_rate=args.track_train_hit_rate,
            model_type=args.model_type,
        )
        print(json.dumps(metrics.__dict__, indent=2, sort_keys=True))
        return

    if args.command == "train-v0-raw":
        summary = train_v0_from_raw(
            input_path=args.input,
            output_dir=args.output,
            metadata_path=args.metadata,
            reward=args.reward,
            label_type=args.label_type,
            alpha=args.alpha,
            default_bonus=args.default_bonus,
            l2=args.l2,
            epochs=args.epochs,
            device=args.device,
            progress_every=args.progress_every,
            track_train_hit_rate=args.track_train_hit_rate,
            model_type=args.model_type,
        )
        print(
            json.dumps(
                {
                    "input_path": summary.input_path,
                    "metadata_path": summary.metadata_path,
                    "samples_path": summary.samples_path,
                    "label_type": summary.label_type,
                    "reward": summary.reward,
                    "conversion": summary.conversion,
                    "training": summary.training.__dict__,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "train-v0-raw-dual":
        summary = train_v0_dual_from_raw(
            input_path=args.input,
            output_dir=args.output,
            ro_metadata_path=args.ro_metadata,
            app_metadata_path=args.app_metadata,
            reward=args.reward,
            alpha=args.alpha,
            alpha_end=args.alpha_end,
            default_bonus=args.default_bonus,
            l2=args.l2,
            epochs=args.epochs,
            device=args.device,
            progress_every=args.progress_every,
            train_window=args.train_window,
            eval_window=args.eval_window,
            shuffle_seed=args.shuffle_seed,
            tensorboard_logdir=args.tensorboard_logdir,
        )
        print(
            json.dumps(
                {
                    "input_path": summary.input_path,
                    "output_dir": summary.output_dir,
                    "tensorboard_logdir": summary.tensorboard_logdir,
                    "reward": summary.reward,
                    "alpha_start": summary.alpha_start,
                    "alpha_end": summary.alpha_end,
                    "train_window": summary.train_window,
                    "eval_window": summary.eval_window,
                    "shuffle_seed": summary.shuffle_seed,
                    "ro": {
                        "metadata_path": summary.ro.metadata_path,
                        "artifact_dir": summary.ro.artifact_dir,
                        "metrics": summary.ro.metrics.__dict__,
                    },
                    "app": {
                        "metadata_path": summary.app.metadata_path,
                        "artifact_dir": summary.app.artifact_dir,
                        "metrics": summary.app.metrics.__dict__,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
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

    if args.command == "convert-v6-raw-ro":
        all_action_ids = None
        if args.all_actions_metadata:
            all_action_ids = list(BanditMetadata.load(args.all_actions_metadata).global_action_ids)
        summary = convert_raw_sequence_to_v6_expanded_ro(
            input_path=args.input,
            output_samples_path=args.output_samples,
            relevance_markdown=args.relevance_markdown,
            most_relevant_reward=args.most_relevant_reward,
            plausible_reward=args.plausible_reward,
            irrelevant_reward=args.irrelevant_reward,
            most_relevant_repeat=args.most_relevant_repeat,
            plausible_repeat=args.plausible_repeat,
            irrelevant_repeat=args.irrelevant_repeat,
            all_action_ids=all_action_ids,
            other_zero_mode=args.other_zero_mode,
        )
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
        return

    if args.command == "convert-v6-raw-app":
        all_action_ids = None
        if args.all_actions_metadata:
            all_action_ids = list(BanditMetadata.load(args.all_actions_metadata).global_action_ids)
        summary = convert_raw_sequence_to_v6_expanded_app(
            input_path=args.input,
            output_samples_path=args.output_samples,
            relevance_markdown=args.relevance_markdown,
            most_relevant_reward=args.most_relevant_reward,
            plausible_reward=args.plausible_reward,
            irrelevant_reward=args.irrelevant_reward,
            most_relevant_repeat=args.most_relevant_repeat,
            plausible_repeat=args.plausible_repeat,
            irrelevant_repeat=args.irrelevant_repeat,
            all_action_ids=all_action_ids,
            other_zero_mode=args.other_zero_mode,
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

    if args.command == "prepare-v0-data":
        summary = prepare_v0_data(
            input_path=args.input,
            catalog_markdown=args.catalog_markdown,
            output_dir=args.output_dir,
            reward=args.reward,
            test_ratio=args.test_ratio,
        )
        print(json.dumps({
            "input_path": summary.input_path,
            "output_dir": summary.output_dir,
            "train_raw_path": summary.train_raw_path,
            "test_raw_path": summary.test_raw_path,
            "reward": summary.reward,
            "test_ratio": summary.test_ratio,
            "ro_metadata": summary.ro_metadata,
            "app_metadata": summary.app_metadata,
            "ro_train_conversion": summary.ro_train_conversion,
            "ro_test_conversion": summary.ro_test_conversion,
            "app_train_conversion": summary.app_train_conversion,
            "app_test_conversion": summary.app_test_conversion,
        }, indent=2, sort_keys=True))
        return

    if args.command == "train-v0-both":
        summary = train_v0_both(
            data_dir=args.data_dir,
            alpha=args.alpha,
            default_bonus=args.default_bonus,
            l2=args.l2,
            epochs=args.epochs,
            device=args.device,
            progress_every=args.progress_every,
            track_train_hit_rate=args.track_train_hit_rate,
            model_type=args.model_type,
        )
        print(json.dumps({
            "data_dir": summary.data_dir,
            "ro_model_dir": summary.ro_model_dir,
            "app_model_dir": summary.app_model_dir,
            "alpha": summary.alpha,
            "default_bonus": summary.default_bonus,
            "l2": summary.l2,
            "epochs": summary.epochs,
            "device": summary.device,
            "ro": summary.ro.__dict__,
            "app": summary.app.__dict__,
        }, indent=2, sort_keys=True))
        return

    if args.command == "eval-v0-both":
        summary = eval_v0_both(
            data_dir=args.data_dir,
            catalog_markdown=args.catalog_markdown,
            top_k=args.top_k,
            device=args.device,
            progress_every=args.progress_every,
        )
        print(json.dumps({
            "data_dir": summary.data_dir,
            "catalog_markdown": summary.catalog_markdown,
            "top_k": summary.top_k,
            "device": summary.device,
            "ro": summary.ro.__dict__,
            "app": summary.app.__dict__,
        }, indent=2))
        return

    if args.command == "eval-soft-scenarios":
        summary = eval_soft_scenarios_both(
            data_dir=args.data_dir,
            raw_input_path=args.input_raw,
            spec_markdown=args.spec_markdown,
            device=args.device,
            progress_every=args.progress_every,
        )
        print(json.dumps({
            "data_dir": summary.data_dir,
            "raw_input_path": summary.raw_input_path,
            "spec_markdown": summary.spec_markdown,
            "device": summary.device,
            "ro": asdict(summary.ro),
            "app": asdict(summary.app),
        }, indent=2))
        return

    if args.command == "run-v6-plan-a":
        summary = run_v6_plan_a(
            input_data_dir=args.input_data_dir,
            output_dir=args.output_dir,
            relevance_markdown=args.relevance_markdown,
            most_relevant_reward=args.most_relevant_reward,
            plausible_reward=args.plausible_reward,
            irrelevant_reward=args.irrelevant_reward,
            most_relevant_repeat=args.most_relevant_repeat,
            plausible_repeat=args.plausible_repeat,
            irrelevant_repeat=args.irrelevant_repeat,
            other_zero_mode=args.other_zero_mode,
            alpha=args.alpha,
            default_bonus=args.default_bonus,
            l2=args.l2,
            epochs=args.epochs,
            top_k=args.top_k,
            device=args.device,
            progress_every=args.progress_every,
            track_train_hit_rate=args.track_train_hit_rate,
            model_type=args.model_type,
        )
        print(json.dumps({
            "workflow_name": summary.workflow_name,
            "output_dir": summary.output_dir,
            "top_k": summary.top_k,
            "relevance_markdown": summary.relevance_markdown,
            "train_raw_path": summary.train_raw_path,
            "test_raw_path": summary.test_raw_path,
            "ro_train_samples_path": summary.ro_train_samples_path,
            "app_train_samples_path": summary.app_train_samples_path,
            "ro_expansion": summary.ro_expansion,
            "app_expansion": summary.app_expansion,
            "ro_training": summary.ro_training.__dict__,
            "app_training": summary.app_training.__dict__,
            "evaluation_report_path": summary.evaluation_report_path,
            "extra_summary": summary.extra_summary,
        }, indent=2))
        return

    if args.command == "run-v6-plan-b":
        summary = run_v6_plan_b(
            input_path=args.input,
            catalog_markdown=args.catalog_markdown,
            output_dir=args.output_dir,
            relevance_markdown=args.relevance_markdown,
            test_ratio=args.test_ratio,
            most_relevant_reward=args.most_relevant_reward,
            plausible_reward=args.plausible_reward,
            irrelevant_reward=args.irrelevant_reward,
            most_relevant_repeat=args.most_relevant_repeat,
            plausible_repeat=args.plausible_repeat,
            irrelevant_repeat=args.irrelevant_repeat,
            other_zero_mode=args.other_zero_mode,
            alpha=args.alpha,
            default_bonus=args.default_bonus,
            l2=args.l2,
            epochs=args.epochs,
            top_k=args.top_k,
            device=args.device,
            progress_every=args.progress_every,
            track_train_hit_rate=args.track_train_hit_rate,
            model_type=args.model_type,
        )
        print(json.dumps({
            "workflow_name": summary.workflow_name,
            "output_dir": summary.output_dir,
            "top_k": summary.top_k,
            "relevance_markdown": summary.relevance_markdown,
            "train_raw_path": summary.train_raw_path,
            "test_raw_path": summary.test_raw_path,
            "ro_train_samples_path": summary.ro_train_samples_path,
            "app_train_samples_path": summary.app_train_samples_path,
            "ro_expansion": summary.ro_expansion,
            "app_expansion": summary.app_expansion,
            "ro_training": summary.ro_training.__dict__,
            "app_training": summary.app_training.__dict__,
            "evaluation_report_path": summary.evaluation_report_path,
            "extra_summary": summary.extra_summary,
        }, indent=2))
        return

    if args.command == "run-v6-plan-all-data":
        summary = run_v6_plan_all_data(
            input_path=args.input,
            catalog_markdown=args.catalog_markdown,
            output_dir=args.output_dir,
            relevance_markdown=args.relevance_markdown,
            most_relevant_reward=args.most_relevant_reward,
            plausible_reward=args.plausible_reward,
            irrelevant_reward=args.irrelevant_reward,
            most_relevant_repeat=args.most_relevant_repeat,
            plausible_repeat=args.plausible_repeat,
            irrelevant_repeat=args.irrelevant_repeat,
            other_zero_mode=args.other_zero_mode,
            alpha=args.alpha,
            default_bonus=args.default_bonus,
            l2=args.l2,
            epochs=args.epochs,
            top_k=args.top_k,
            device=args.device,
            progress_every=args.progress_every,
            track_train_hit_rate=args.track_train_hit_rate,
            model_type=args.model_type,
        )
        print(json.dumps({
            "workflow_name": summary.workflow_name,
            "output_dir": summary.output_dir,
            "top_k": summary.top_k,
            "relevance_markdown": summary.relevance_markdown,
            "train_raw_path": summary.train_raw_path,
            "test_raw_path": summary.test_raw_path,
            "ro_train_samples_path": summary.ro_train_samples_path,
            "app_train_samples_path": summary.app_train_samples_path,
            "ro_expansion": summary.ro_expansion,
            "app_expansion": summary.app_expansion,
            "ro_training": summary.ro_training.__dict__,
            "app_training": summary.app_training.__dict__,
            "evaluation_report_path": summary.evaluation_report_path,
            "extra_summary": summary.extra_summary,
        }, indent=2))
        return

    if args.command == "mine-v6-hard-negatives":
        summary = mine_v6_hard_negative_candidates(
            artifact_dir=args.artifact_dir,
            metadata_path=args.metadata,
            raw_input_path=args.input_raw,
            catalog_markdown=args.catalog_markdown,
            label_namespace=args.label_namespace,
            top_k=args.top_k,
            max_candidates_per_scenario=args.max_candidates_per_scenario,
            device=args.device,
            progress_every=args.progress_every,
            min_count=args.min_count,
            min_rate=args.min_rate,
        )
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps({
            "raw_input_path": summary.raw_input_path,
            "artifact_dir": summary.artifact_dir,
            "metadata_path": summary.metadata_path,
            "catalog_markdown": summary.catalog_markdown,
            "label_namespace": summary.label_namespace,
            "top_k": summary.top_k,
            "sample_count": summary.sample_count,
            "scenarios_with_samples": summary.scenarios_with_samples,
            "max_candidates_per_scenario": summary.max_candidates_per_scenario,
            "per_scenario": summary.per_scenario,
        }, indent=2))
        Path(args.output_markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_markdown).write_text(render_hard_negative_candidates_markdown(summary))
        print(json.dumps({
            "output_json": str(Path(args.output_json)),
            "output_markdown": str(Path(args.output_markdown)),
            "label_namespace": summary.label_namespace,
            "top_k": summary.top_k,
            "sample_count": summary.sample_count,
            "scenarios_with_samples": summary.scenarios_with_samples,
            "max_candidates_per_scenario": summary.max_candidates_per_scenario,
        }, indent=2))
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
