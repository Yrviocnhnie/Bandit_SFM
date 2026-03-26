"""High-level one-line workflows for V0 data prep, training, and evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Any

from recommendation_agents.catalog import (
    AppCatalogSummary,
    CatalogSummary,
    build_app_metadata_from_catalog_markdown,
    build_ro_metadata_from_catalog_markdown,
)
from recommendation_agents.feature_space import V0FeatureSpace
from recommendation_agents.linucb import MaskedDisjointLinUCB
from recommendation_agents.metadata import BanditMetadata
from recommendation_agents.raw_synthetic import (
    ExpandedRawConversionSummary,
    convert_raw_sequence_to_v0,
    convert_raw_sequence_to_v0_app,
    convert_raw_sequence_to_v6_expanded_app,
    convert_raw_sequence_to_v6_expanded_ro,
)
from recommendation_agents.schemas import TrainingEvent
from recommendation_agents.trainer import TrainingMetrics, _format_duration, train_v0
from recommendation_agents.v6_relevance import parse_v6_relevance_markdown


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedDatasetSummary:
    input_path: str
    output_dir: str
    train_raw_path: str
    test_raw_path: str
    reward: float
    test_ratio: float
    ro_metadata: dict[str, Any]
    app_metadata: dict[str, Any]
    ro_train_conversion: dict[str, int]
    ro_test_conversion: dict[str, int]
    app_train_conversion: dict[str, int]
    app_test_conversion: dict[str, int]


@dataclass(frozen=True)
class TrainBothSummary:
    data_dir: str
    ro_model_dir: str
    app_model_dir: str
    alpha: float
    default_bonus: float
    l2: float
    epochs: int
    device: str
    ro: TrainingMetrics
    app: TrainingMetrics


@dataclass(frozen=True)
class EvalTopKSummary:
    sample_count: int
    top_k: int
    avg_most_relevant_covered_in_topk: float
    avg_most_relevant_covered_in_topk_ratio: float
    avg_acceptable_covered_in_topk: float
    avg_acceptable_covered_in_topk_ratio: float
    avg_irrelevant_in_topk: float
    avg_irrelevant_in_topk_ratio: float
    top6_predicted_action_distribution: list[dict[str, Any]]
    scenarios_with_test_samples: int
    scenarios_without_test_samples: list[str]
    per_scenario: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class EvalBothSummary:
    data_dir: str
    catalog_markdown: str
    top_k: int
    device: str
    ro: EvalTopKSummary
    app: EvalTopKSummary


@dataclass(frozen=True)
class StratifiedSplitSummary:
    train_rows: int
    test_rows: int
    train_scenarios: int
    test_scenarios: int
    per_scenario: dict[str, dict[str, int]]


@dataclass(frozen=True)
class V6WorkflowSummary:
    workflow_name: str
    output_dir: str
    top_k: int
    relevance_markdown: str
    train_raw_path: str
    test_raw_path: str
    ro_train_samples_path: str
    app_train_samples_path: str
    ro_expansion: dict[str, Any]
    app_expansion: dict[str, Any]
    ro_training: TrainingMetrics
    app_training: TrainingMetrics
    evaluation_report_path: str
    extra_summary: dict[str, Any]


def _write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_json_unsorted(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def _copy_file(src: str | Path, dst: str | Path) -> None:
    destination = Path(dst)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def split_raw_by_episode(
    input_path: str | Path,
    train_output_path: str | Path,
    test_output_path: str | Path,
    test_ratio: float = 0.2,
) -> dict[str, int]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    modulo = 10000
    threshold = int(test_ratio * modulo)
    train_count = 0
    test_count = 0

    train_path = Path(train_output_path)
    test_path = Path(test_output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(input_path).open() as fin, train_path.open("w") as ftrain, test_path.open("w") as ftest:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            scenario_id = row.get("scenario_id") or row.get("scenarioId")
            episode_id = row.get("episode_id")
            key = f"{scenario_id}::{episode_id}"
            bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % modulo
            if bucket < threshold:
                ftest.write(line)
                test_count += 1
            else:
                ftrain.write(line)
                train_count += 1

    return {
        "train_rows": train_count,
        "test_rows": test_count,
    }


def split_raw_by_scenario_episode(
    input_path: str | Path,
    train_output_path: str | Path,
    test_output_path: str | Path,
    test_ratio: float = 0.2,
) -> StratifiedSplitSummary:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")

    rows_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl_for_split(input_path):
        scenario_id = str(row.get("scenario_id") or row.get("scenarioId") or "")
        if not scenario_id:
            raise ValueError("Scenario-stratified splitting requires scenario_id on every raw row")
        rows_by_scenario[scenario_id].append(row)

    train_path = Path(train_output_path)
    test_path = Path(test_output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    per_scenario: dict[str, dict[str, int]] = {}
    train_rows = 0
    test_rows = 0
    train_scenarios = 0
    test_scenarios = 0

    with train_path.open("w") as ftrain, test_path.open("w") as ftest:
        for scenario_id in sorted(rows_by_scenario):
            scenario_rows = rows_by_scenario[scenario_id]
            episode_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in scenario_rows:
                episode_id = str(row.get("episode_id") or f"{scenario_id}::{row.get('scenario_elapsed_sec', 0)}")
                episode_groups[episode_id].append(row)

            ordered_episode_ids = sorted(
                episode_groups,
                key=lambda episode_id: hashlib.md5(f"{scenario_id}::{episode_id}".encode("utf-8")).hexdigest(),
            )
            total_groups = len(ordered_episode_ids)
            if total_groups <= 1:
                test_group_count = 0
            else:
                test_group_count = max(1, int(round(total_groups * test_ratio)))
                test_group_count = min(test_group_count, total_groups - 1)
            test_episode_ids = set(ordered_episode_ids[:test_group_count])

            scenario_train_rows = 0
            scenario_test_rows = 0
            for episode_id in ordered_episode_ids:
                target = ftest if episode_id in test_episode_ids else ftrain
                group_rows = episode_groups[episode_id]
                for row in group_rows:
                    target.write(json.dumps(row, sort_keys=True))
                    target.write("\n")
                if episode_id in test_episode_ids:
                    scenario_test_rows += len(group_rows)
                else:
                    scenario_train_rows += len(group_rows)

            if scenario_train_rows > 0:
                train_scenarios += 1
            if scenario_test_rows > 0:
                test_scenarios += 1
            train_rows += scenario_train_rows
            test_rows += scenario_test_rows
            per_scenario[scenario_id] = {
                "train_rows": scenario_train_rows,
                "test_rows": scenario_test_rows,
                "episode_groups": total_groups,
            }

    return StratifiedSplitSummary(
        train_rows=train_rows,
        test_rows=test_rows,
        train_scenarios=train_scenarios,
        test_scenarios=test_scenarios,
        per_scenario=per_scenario,
    )


def _load_jsonl_for_split(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prepare_v0_data(
    input_path: str | Path,
    catalog_markdown: str | Path,
    output_dir: str | Path,
    reward: float = 1.0,
    test_ratio: float = 0.2,
) -> PreparedDatasetSummary:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ro_metadata_path = output_path / "ro_metadata.json"
    app_metadata_path = output_path / "app_metadata.json"
    train_raw_path = output_path / "train.raw.jsonl"
    test_raw_path = output_path / "test.raw.jsonl"

    ro_catalog = build_ro_metadata_from_catalog_markdown(catalog_markdown, ro_metadata_path)
    app_catalog = build_app_metadata_from_catalog_markdown(catalog_markdown, app_metadata_path)
    split_summary = split_raw_by_episode(input_path, train_raw_path, test_raw_path, test_ratio=test_ratio)

    ro_train = convert_raw_sequence_to_v0(
        input_path=train_raw_path,
        output_samples_path=output_path / "ro_train_samples.jsonl",
        output_metadata_path=output_path / "ro_metadata_inferred_train.json",
        reward=reward,
    )
    ro_test = convert_raw_sequence_to_v0(
        input_path=test_raw_path,
        output_samples_path=output_path / "ro_test_samples.jsonl",
        output_metadata_path=output_path / "ro_metadata_inferred_test.json",
        reward=reward,
    )
    app_train = convert_raw_sequence_to_v0_app(
        input_path=train_raw_path,
        output_samples_path=output_path / "app_train_samples.jsonl",
        output_metadata_path=output_path / "app_metadata_inferred_train.json",
        reward=reward,
    )
    app_test = convert_raw_sequence_to_v0_app(
        input_path=test_raw_path,
        output_samples_path=output_path / "app_test_samples.jsonl",
        output_metadata_path=output_path / "app_metadata_inferred_test.json",
        reward=reward,
    )

    summary = PreparedDatasetSummary(
        input_path=str(Path(input_path)),
        output_dir=str(output_path),
        train_raw_path=str(train_raw_path),
        test_raw_path=str(test_raw_path),
        reward=float(reward),
        test_ratio=float(test_ratio),
        ro_metadata=asdict(ro_catalog),
        app_metadata=asdict(app_catalog),
        ro_train_conversion=asdict(ro_train) | split_summary,
        ro_test_conversion=asdict(ro_test),
        app_train_conversion=asdict(app_train),
        app_test_conversion=asdict(app_test),
    )
    _write_json(output_path / "prepare_summary.json", asdict(summary))
    return summary


def _run_v6_training_bundle(
    output_dir: str | Path,
    relevance_markdown: str | Path,
    most_relevant_reward: float,
    plausible_reward: float,
    irrelevant_reward: float,
    most_relevant_repeat: int,
    plausible_repeat: int,
    irrelevant_repeat: int,
    alpha: float,
    default_bonus: float,
    l2: float,
    epochs: int,
    top_k: int,
    device: str,
    progress_every: int,
    workflow_name: str,
    extra_summary: dict[str, Any],
    track_train_hit_rate: bool,
) -> V6WorkflowSummary:
    output_path = Path(output_dir)
    ro_expansion = convert_raw_sequence_to_v6_expanded_ro(
        input_path=output_path / "train.raw.jsonl",
        output_samples_path=output_path / "ro_train_samples_expanded.jsonl",
        relevance_markdown=relevance_markdown,
        most_relevant_reward=most_relevant_reward,
        plausible_reward=plausible_reward,
        irrelevant_reward=irrelevant_reward,
        most_relevant_repeat=most_relevant_repeat,
        plausible_repeat=plausible_repeat,
        irrelevant_repeat=irrelevant_repeat,
    )
    app_expansion = convert_raw_sequence_to_v6_expanded_app(
        input_path=output_path / "train.raw.jsonl",
        output_samples_path=output_path / "app_train_samples_expanded.jsonl",
        relevance_markdown=relevance_markdown,
        most_relevant_reward=most_relevant_reward,
        plausible_reward=plausible_reward,
        irrelevant_reward=irrelevant_reward,
        most_relevant_repeat=most_relevant_repeat,
        plausible_repeat=plausible_repeat,
        irrelevant_repeat=irrelevant_repeat,
    )
    training_summary = train_v0_both(
        data_dir=output_path,
        alpha=alpha,
        default_bonus=default_bonus,
        l2=l2,
        epochs=epochs,
        device=device,
        progress_every=progress_every,
        track_train_hit_rate=track_train_hit_rate,
    )
    eval_summary = eval_v0_both(
        data_dir=output_path,
        catalog_markdown=relevance_markdown,
        top_k=top_k,
        device=device,
        progress_every=progress_every,
    )
    summary = V6WorkflowSummary(
        workflow_name=workflow_name,
        output_dir=str(output_path),
        top_k=top_k,
        relevance_markdown=str(Path(relevance_markdown)),
        train_raw_path=str(output_path / "train.raw.jsonl"),
        test_raw_path=str(output_path / "test.raw.jsonl"),
        ro_train_samples_path=str(output_path / "ro_train_samples_expanded.jsonl"),
        app_train_samples_path=str(output_path / "app_train_samples_expanded.jsonl"),
        ro_expansion=asdict(ro_expansion),
        app_expansion=asdict(app_expansion),
        ro_training=training_summary.ro,
        app_training=training_summary.app,
        evaluation_report_path=str(output_path / f"eval_both_top{top_k}.json"),
        extra_summary=extra_summary,
    )
    _write_json_unsorted(output_path / f"{workflow_name}_summary.json", {
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
        "ro_training": asdict(summary.ro_training),
        "app_training": asdict(summary.app_training),
        "evaluation_report_path": summary.evaluation_report_path,
        "extra_summary": summary.extra_summary,
        "evaluation": {
            "ro": asdict(eval_summary.ro),
            "app": asdict(eval_summary.app),
        },
    })
    return summary


def run_v6_plan_a(
    input_data_dir: str | Path,
    output_dir: str | Path,
    relevance_markdown: str | Path = "docs/scenario_recommendation_actions_v6.md",
    most_relevant_reward: float = 1.0,
    plausible_reward: float = 0.6,
    irrelevant_reward: float = 0.0,
    most_relevant_repeat: int = 1,
    plausible_repeat: int = 1,
    irrelevant_repeat: int = 1,
    alpha: float = 0.05,
    default_bonus: float = 0.0,
    l2: float = 1.0,
    epochs: int = 1,
    top_k: int = 3,
    device: str = "auto",
    progress_every: int = 1000,
    track_train_hit_rate: bool = False,
) -> V6WorkflowSummary:
    input_path = Path(input_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name in (
        "train.raw.jsonl",
        "test.raw.jsonl",
        "ro_metadata.json",
        "app_metadata.json",
        "ro_test_samples.jsonl",
        "app_test_samples.jsonl",
    ):
        _copy_file(input_path / name, output_path / name)

    return _run_v6_training_bundle(
        output_dir=output_path,
        relevance_markdown=relevance_markdown,
        most_relevant_reward=most_relevant_reward,
        plausible_reward=plausible_reward,
        irrelevant_reward=irrelevant_reward,
        most_relevant_repeat=most_relevant_repeat,
        plausible_repeat=plausible_repeat,
        irrelevant_repeat=irrelevant_repeat,
        alpha=alpha,
        default_bonus=default_bonus,
        l2=l2,
        epochs=epochs,
        top_k=top_k,
        device=device,
        progress_every=progress_every,
        workflow_name="run_v6_plan_a",
        extra_summary={
            "input_data_dir": str(input_path),
            "split_policy": "reuse_existing_raw_split",
            "reward_config": {
                "most_relevant_reward": float(most_relevant_reward),
                "plausible_reward": float(plausible_reward),
                "irrelevant_reward": float(irrelevant_reward),
            },
            "repeat_config": {
                "most_relevant_repeat": int(most_relevant_repeat),
                "plausible_repeat": int(plausible_repeat),
                "irrelevant_repeat": int(irrelevant_repeat),
            },
            "epochs": int(epochs),
        },
        track_train_hit_rate=track_train_hit_rate,
    )


def run_v6_plan_b(
    input_path: str | Path,
    catalog_markdown: str | Path,
    output_dir: str | Path,
    relevance_markdown: str | Path = "docs/scenario_recommendation_actions_v6.md",
    test_ratio: float = 0.2,
    most_relevant_reward: float = 1.0,
    plausible_reward: float = 0.6,
    irrelevant_reward: float = 0.0,
    most_relevant_repeat: int = 1,
    plausible_repeat: int = 1,
    irrelevant_repeat: int = 1,
    alpha: float = 0.05,
    default_bonus: float = 0.0,
    l2: float = 1.0,
    epochs: int = 1,
    top_k: int = 3,
    device: str = "auto",
    progress_every: int = 1000,
    track_train_hit_rate: bool = False,
) -> V6WorkflowSummary:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ro_catalog = build_ro_metadata_from_catalog_markdown(catalog_markdown, output_path / "ro_metadata.json")
    app_catalog = build_app_metadata_from_catalog_markdown(catalog_markdown, output_path / "app_metadata.json")
    split_summary = split_raw_by_scenario_episode(
        input_path=input_path,
        train_output_path=output_path / "train.raw.jsonl",
        test_output_path=output_path / "test.raw.jsonl",
        test_ratio=test_ratio,
    )
    convert_raw_sequence_to_v0(
        input_path=output_path / "test.raw.jsonl",
        output_samples_path=output_path / "ro_test_samples.jsonl",
        output_metadata_path=output_path / "ro_metadata_inferred_test.json",
        reward=1.0,
    )
    convert_raw_sequence_to_v0_app(
        input_path=output_path / "test.raw.jsonl",
        output_samples_path=output_path / "app_test_samples.jsonl",
        output_metadata_path=output_path / "app_metadata_inferred_test.json",
        reward=1.0,
    )

    return _run_v6_training_bundle(
        output_dir=output_path,
        relevance_markdown=relevance_markdown,
        most_relevant_reward=most_relevant_reward,
        plausible_reward=plausible_reward,
        irrelevant_reward=irrelevant_reward,
        most_relevant_repeat=most_relevant_repeat,
        plausible_repeat=plausible_repeat,
        irrelevant_repeat=irrelevant_repeat,
        alpha=alpha,
        default_bonus=default_bonus,
        l2=l2,
        epochs=epochs,
        top_k=top_k,
        device=device,
        progress_every=progress_every,
        workflow_name="run_v6_plan_b",
        extra_summary={
            "input_path": str(Path(input_path)),
            "catalog_markdown": str(Path(catalog_markdown)),
            "test_ratio": float(test_ratio),
            "reward_config": {
                "most_relevant_reward": float(most_relevant_reward),
                "plausible_reward": float(plausible_reward),
                "irrelevant_reward": float(irrelevant_reward),
            },
            "repeat_config": {
                "most_relevant_repeat": int(most_relevant_repeat),
                "plausible_repeat": int(plausible_repeat),
                "irrelevant_repeat": int(irrelevant_repeat),
            },
            "epochs": int(epochs),
            "split_summary": asdict(split_summary),
            "ro_metadata": asdict(ro_catalog),
            "app_metadata": asdict(app_catalog),
        },
        track_train_hit_rate=track_train_hit_rate,
    )


def train_v0_both(
    data_dir: str | Path,
    alpha: float = 0.05,
    default_bonus: float = 0.75,
    l2: float = 1.0,
    epochs: int = 1,
    device: str = "auto",
    progress_every: int = 1000,
    track_train_hit_rate: bool = False,
) -> TrainBothSummary:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data_path = Path(data_dir)
    ro_model_dir = data_path / "ro_model"
    app_model_dir = data_path / "app_model"
    ro_train_path = data_path / "ro_train_samples_expanded.jsonl"
    if not ro_train_path.exists():
        ro_train_path = data_path / "ro_train_samples.jsonl"
    app_train_path = data_path / "app_train_samples_expanded.jsonl"
    if not app_train_path.exists():
        app_train_path = data_path / "app_train_samples.jsonl"
    with ro_train_path.open() as handle:
        ro_rows = sum(1 for line in handle if line.strip())
    with app_train_path.open() as handle:
        app_rows = sum(1 for line in handle if line.strip())
    total_start = time.time()
    logger.info(
        "[WORKFLOW] Starting joint training | data_dir=%s | device=%s | epochs=%d | track_train_hit_rate=%s",
        data_path,
        device,
        epochs,
        track_train_hit_rate,
    )
    logger.info("[WORKFLOW] R/O train rows=%d | App train rows=%d", ro_rows, app_rows)
    ro_start = time.time()
    logger.info("[WORKFLOW] Starting R/O model training...")
    ro_metrics = train_v0(
        metadata_path=data_path / "ro_metadata.json",
        samples_path=ro_train_path,
        output_dir=ro_model_dir,
        alpha=alpha,
        default_bonus=default_bonus,
        l2=l2,
        epochs=epochs,
        device=device,
        progress_every=progress_every,
        progress_label="RO",
        track_train_hit_rate=track_train_hit_rate,
    )
    logger.info("[WORKFLOW] Finished R/O model training in %.1fs", time.time() - ro_start)
    app_start = time.time()
    logger.info("[WORKFLOW] Starting App model training...")
    app_metrics = train_v0(
        metadata_path=data_path / "app_metadata.json",
        samples_path=app_train_path,
        output_dir=app_model_dir,
        alpha=alpha,
        default_bonus=default_bonus,
        l2=l2,
        epochs=epochs,
        device=device,
        progress_every=progress_every,
        progress_label="APP",
        track_train_hit_rate=track_train_hit_rate,
    )
    logger.info("[WORKFLOW] Finished App model training in %.1fs", time.time() - app_start)
    logger.info("[WORKFLOW] Finished joint training in %.1fs", time.time() - total_start)
    summary = TrainBothSummary(
        data_dir=str(data_path),
        ro_model_dir=str(ro_model_dir),
        app_model_dir=str(app_model_dir),
        alpha=float(alpha),
        default_bonus=float(default_bonus),
        l2=float(l2),
        epochs=int(epochs),
        device=device,
        ro=ro_metrics,
        app=app_metrics,
    )
    _write_json(data_path / "train_both_summary.json", {
        "data_dir": summary.data_dir,
        "ro_model_dir": summary.ro_model_dir,
        "app_model_dir": summary.app_model_dir,
        "alpha": summary.alpha,
        "default_bonus": summary.default_bonus,
        "l2": summary.l2,
        "epochs": summary.epochs,
        "device": summary.device,
        "ro": asdict(summary.ro),
        "app": asdict(summary.app),
    })
    return summary


def _top_action_distribution(
    action_counts: Counter[str],
    sample_count: int,
    top_k: int,
    limit: int = 6,
) -> list[dict[str, Any]]:
    total_slots = sample_count * top_k
    distribution: list[dict[str, Any]] = []
    for action_id, count in action_counts.most_common(limit):
        distribution.append(
            {
                "action_id": action_id,
                "count": count,
                "topk_appearance_rate": (count / sample_count) if sample_count > 0 else 0.0,
                "slot_share": (count / total_slots) if total_slots > 0 else 0.0,
            }
        )
    return distribution


def evaluate_v0_topk(
    artifact_dir: str | Path,
    metadata_path: str | Path,
    test_samples_path: str | Path,
    catalog_markdown: str | Path,
    label_namespace: str,
    top_k: int = 3,
    device: str = "auto",
    progress_every: int = 1000,
    progress_label: str | None = None,
) -> EvalTopKSummary:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    label_prefix = f"[{progress_label}] " if progress_label else ""
    if label_namespace not in {"ro", "app"}:
        raise ValueError("label_namespace must be 'ro' or 'app'")
    metadata = BanditMetadata.load(metadata_path)
    relevance_catalog = parse_v6_relevance_markdown(catalog_markdown)[label_namespace]
    feature_space = V0FeatureSpace()
    model = MaskedDisjointLinUCB.load(artifact_dir, device=device)
    rows = [
        TrainingEvent.from_dict(json.loads(line))
        for line in Path(test_samples_path).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No test rows found in {test_samples_path}")

    total_rows = len(rows)
    start_time = time.time()
    logger.info(
        "%sStarting evaluation | rows=%d | top_k=%d | device=%s",
        label_prefix,
        total_rows,
        top_k,
        device,
    )

    pred_topk_counts: Counter[str] = Counter()
    scenario_row_counts: Counter[str] = Counter()
    scenario_pred_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scenario_most_sum: Counter[str] = Counter()
    scenario_acceptable_sum: Counter[str] = Counter()
    scenario_irrelevant_sum: Counter[str] = Counter()
    total_most_sum = 0.0
    total_acceptable_sum = 0.0
    total_irrelevant_sum = 0.0
    interval_most_sum = 0.0
    interval_acceptable_sum = 0.0
    interval_irrelevant_sum = 0.0
    skipped_rows_missing_scenario = 0

    for index, event in enumerate(rows, start=1):
        scenario_id = event.scenario_id
        if scenario_id is None or scenario_id not in relevance_catalog:
            skipped_rows_missing_scenario += 1
            continue
        x = feature_space.encode(event.context)
        ranked = model.rank(
            x,
            metadata.candidate_action_ids(scenario_id),
            metadata.default_action_id(scenario_id),
            top_k=top_k,
        )
        pred = tuple(item.action_id for item in ranked)
        pred_set = set(pred)
        scenario_spec = relevance_catalog[scenario_id]
        most_relevant = tuple(metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["most_relevant_3"])
        plausible = tuple(metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["other_plausible_3"])
        irrelevant = tuple(metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["irrelevant_2"])
        acceptable = set(most_relevant) | set(plausible)

        most_count = sum(1 for action_id in most_relevant if action_id in pred_set)
        acceptable_count = sum(1 for action_id in pred if action_id in acceptable)
        irrelevant_count = sum(1 for action_id in pred if action_id in irrelevant)

        total_most_sum += most_count
        total_acceptable_sum += acceptable_count
        total_irrelevant_sum += irrelevant_count
        interval_most_sum += most_count
        interval_acceptable_sum += acceptable_count
        interval_irrelevant_sum += irrelevant_count

        scenario_row_counts[scenario_id] += 1
        scenario_most_sum[scenario_id] += most_count
        scenario_acceptable_sum[scenario_id] += acceptable_count
        scenario_irrelevant_sum[scenario_id] += irrelevant_count
        for action_id in pred:
            pred_topk_counts[action_id] += 1
            scenario_pred_counts[scenario_id][action_id] += 1

        if index % progress_every == 0:
            elapsed = time.time() - start_time
            rows_per_sec = index / elapsed if elapsed > 0 else 0.0
            remaining = max(total_rows - index, 0)
            eta_seconds = remaining / rows_per_sec if rows_per_sec > 0 else 0.0
            interval_denominator = progress_every
            logger.info(
                "%sProcessed %d/%d rows | interval most=%.3f/3 | cumulative most=%.3f/3 | interval acceptable=%.3f/%d | cumulative acceptable=%.3f/%d | interval irrelevant=%.3f/%d | cumulative irrelevant=%.3f/%d | elapsed=%s | eta=%s | %.1f rows/s",
                label_prefix,
                index,
                total_rows,
                interval_most_sum / interval_denominator,
                total_most_sum / max(index - skipped_rows_missing_scenario, 1),
                interval_acceptable_sum / interval_denominator,
                top_k,
                total_acceptable_sum / max(index - skipped_rows_missing_scenario, 1),
                top_k,
                interval_irrelevant_sum / interval_denominator,
                top_k,
                total_irrelevant_sum / max(index - skipped_rows_missing_scenario, 1),
                top_k,
                _format_duration(elapsed),
                _format_duration(eta_seconds),
                rows_per_sec,
            )
            interval_most_sum = 0.0
            interval_acceptable_sum = 0.0
            interval_irrelevant_sum = 0.0

    remainder = total_rows % progress_every
    if remainder:
        elapsed = time.time() - start_time
        rows_per_sec = total_rows / elapsed if elapsed > 0 else 0.0
        processed_rows = total_rows - skipped_rows_missing_scenario
        logger.info(
            "%sProcessed %d/%d rows | interval most=%.3f/3 | cumulative most=%.3f/3 | interval acceptable=%.3f/%d | cumulative acceptable=%.3f/%d | interval irrelevant=%.3f/%d | cumulative irrelevant=%.3f/%d | elapsed=%s | eta=0s | %.1f rows/s",
            label_prefix,
            total_rows,
            total_rows,
            interval_most_sum / remainder,
            total_most_sum / max(processed_rows, 1),
            interval_acceptable_sum / remainder,
            top_k,
            total_acceptable_sum / max(processed_rows, 1),
            top_k,
            interval_irrelevant_sum / remainder,
            top_k,
            total_irrelevant_sum / max(processed_rows, 1),
            top_k,
            _format_duration(elapsed),
            rows_per_sec,
        )

    scenario_eval: dict[str, dict[str, Any]] = {}
    for scenario_id in sorted(relevance_catalog):
        scenario_spec = relevance_catalog[scenario_id]
        row_count = scenario_row_counts[scenario_id]
        if row_count == 0:
            continue
        most_relevant = [metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["most_relevant_3"]]
        plausible = [metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["other_plausible_3"]]
        irrelevant = [metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["irrelevant_2"]]
        scenario_eval[scenario_id] = {
            "sample_count": row_count,
            "most_relevant_3": most_relevant,
            "other_plausible_3": plausible,
            "acceptable_6": most_relevant + plausible,
            "irrelevant_2": irrelevant,
            "avg_most_relevant_covered_in_topk": scenario_most_sum[scenario_id] / row_count,
            "avg_most_relevant_covered_in_topk_ratio": scenario_most_sum[scenario_id] / (row_count * 3.0),
            "avg_acceptable_covered_in_topk": scenario_acceptable_sum[scenario_id] / row_count,
            "avg_acceptable_covered_in_topk_ratio": scenario_acceptable_sum[scenario_id] / (row_count * top_k),
            "avg_irrelevant_in_topk": scenario_irrelevant_sum[scenario_id] / row_count,
            "avg_irrelevant_in_topk_ratio": scenario_irrelevant_sum[scenario_id] / (row_count * top_k),
            "predicted_top6_action_distribution": _top_action_distribution(
                scenario_pred_counts[scenario_id],
                sample_count=row_count,
                top_k=top_k,
                limit=6,
            ),
        }

    elapsed_total = time.time() - start_time
    evaluated_rows = total_rows - skipped_rows_missing_scenario
    logger.info(
        "%sFinished evaluation | rows=%d | evaluated=%d | avg most=%.3f/3 | avg acceptable=%.3f/%d | avg irrelevant=%.3f/%d | elapsed=%s | avg %.1f rows/s",
        label_prefix,
        total_rows,
        evaluated_rows,
        total_most_sum / max(evaluated_rows, 1),
        total_acceptable_sum / max(evaluated_rows, 1),
        top_k,
        total_irrelevant_sum / max(evaluated_rows, 1),
        top_k,
        _format_duration(elapsed_total),
        total_rows / elapsed_total if elapsed_total > 0 else 0.0,
    )

    return EvalTopKSummary(
        sample_count=evaluated_rows,
        top_k=top_k,
        avg_most_relevant_covered_in_topk=total_most_sum / max(evaluated_rows, 1),
        avg_most_relevant_covered_in_topk_ratio=total_most_sum / max(evaluated_rows * 3.0, 1.0),
        avg_acceptable_covered_in_topk=total_acceptable_sum / max(evaluated_rows, 1),
        avg_acceptable_covered_in_topk_ratio=total_acceptable_sum / max(evaluated_rows * top_k, 1),
        avg_irrelevant_in_topk=total_irrelevant_sum / max(evaluated_rows, 1),
        avg_irrelevant_in_topk_ratio=total_irrelevant_sum / max(evaluated_rows * top_k, 1),
        top6_predicted_action_distribution=_top_action_distribution(
            pred_topk_counts,
            sample_count=max(evaluated_rows, 1),
            top_k=top_k,
            limit=6,
        ),
        scenarios_with_test_samples=len(scenario_eval),
        scenarios_without_test_samples=[
            scenario_id for scenario_id in sorted(relevance_catalog) if scenario_row_counts[scenario_id] == 0
        ],
        per_scenario=scenario_eval,
    )


def eval_v0_both(
    data_dir: str | Path,
    catalog_markdown: str | Path = "docs/scenario_recommendation_actions_v6.md",
    top_k: int = 3,
    device: str = "auto",
    progress_every: int = 1000,
) -> EvalBothSummary:
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data_path = Path(data_dir)
    logger.info("[WORKFLOW] Starting joint evaluation | data_dir=%s | top_k=%d | device=%s", data_path, top_k, device)
    ro = evaluate_v0_topk(
        artifact_dir=data_path / "ro_model",
        metadata_path=data_path / "ro_metadata.json",
        test_samples_path=data_path / "ro_test_samples.jsonl",
        catalog_markdown=catalog_markdown,
        label_namespace="ro",
        top_k=top_k,
        device=device,
        progress_every=progress_every,
        progress_label="RO-EVAL",
    )
    app = evaluate_v0_topk(
        artifact_dir=data_path / "app_model",
        metadata_path=data_path / "app_metadata.json",
        test_samples_path=data_path / "app_test_samples.jsonl",
        catalog_markdown=catalog_markdown,
        label_namespace="app",
        top_k=top_k,
        device=device,
        progress_every=progress_every,
        progress_label="APP-EVAL",
    )
    logger.info("[WORKFLOW] Finished joint evaluation")
    summary = EvalBothSummary(
        data_dir=str(data_path),
        catalog_markdown=str(Path(catalog_markdown)),
        top_k=top_k,
        device=device,
        ro=ro,
        app=app,
    )
    _write_json_unsorted(data_path / f"eval_both_top{top_k}.json", {
        "data_dir": summary.data_dir,
        "catalog_markdown": summary.catalog_markdown,
        "top_k": summary.top_k,
        "device": summary.device,
        "ro": asdict(summary.ro),
        "app": asdict(summary.app),
    })
    return summary
