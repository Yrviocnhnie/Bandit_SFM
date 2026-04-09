"""High-level one-line workflows for V0 data prep, training, and evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
import numpy as np
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
from recommendation_agents.in_between_relevance import parse_in_between_scenarios_markdown
from recommendation_agents.linucb import load_bandit_model
from recommendation_agents.metadata import BanditMetadata
from recommendation_agents.raw_synthetic import (
    ExpandedRawConversionSummary,
    _build_context,
    _load_jsonl,
    _scenario_id,
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
class SoftScenarioEvalSummary:
    sample_count: int
    reference_size: int
    avg_hits_in_top3: float
    avg_hits_in_top3_ratio: float
    avg_hits_in_top5: float
    avg_hits_in_top5_ratio: float
    top10_predicted_action_distribution_top5: list[dict[str, Any]]
    scenarios_with_samples: int
    scenarios_without_samples: list[str]
    per_scenario: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SoftScenarioEvalBothSummary:
    data_dir: str
    raw_input_path: str
    spec_markdown: str
    device: str
    ro: SoftScenarioEvalSummary
    app: SoftScenarioEvalSummary


@dataclass(frozen=True)
class FeedbackPropagationExperimentSummary:
    artifact_dir: str
    output_dir: str
    relevance_markdown: str
    device: str
    n_values: list[int]
    feedback_items: list[dict[str, Any]]
    conditions: list[dict[str, Any]]


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


@dataclass(frozen=True)
class HardNegativeMiningSummary:
    raw_input_path: str
    artifact_dir: str
    metadata_path: str
    catalog_markdown: str
    label_namespace: str
    top_k: int
    sample_count: int
    scenarios_with_samples: int
    max_candidates_per_scenario: int
    per_scenario: dict[str, dict[str, Any]]


def _write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_json_unsorted(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def _display_action_id(model_action_id: str) -> str:
    if "::" in model_action_id:
        return model_action_id.split("::", 1)[1]
    return model_action_id


def _copy_file(src: str | Path, dst: str | Path) -> None:
    destination = Path(dst)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


_FEEDBACK_PROPAGATION_SCENARIOS = (
    ("ARRIVE_OFFICE", "like", 1),
    ("OFFICE_LUNCH_OUT", "dislike", 1),
    ("OFFICE_WORKING", "like", 2),
    ("LEAVE_OFFICE", "dislike", 2),
)


def _top_action_distribution_topk(predicted_counts: Counter[str], sample_count: int, top_k: int, limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action_id, count in predicted_counts.most_common(limit):
        rows.append(
            {
                "action_id": action_id,
                "count": count,
                "topk_appearance_rate": count / sample_count if sample_count else 0.0,
                "slot_share": count / (sample_count * top_k) if sample_count and top_k else 0.0,
            }
        )
    return rows


def _cosine_similarity(anchor: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    anchor_norm = float(np.linalg.norm(anchor))
    if anchor_norm == 0.0:
        return np.zeros((candidates.shape[0],), dtype=np.float32)
    candidate_norms = np.linalg.norm(candidates, axis=1)
    safe_norms = np.where(candidate_norms > 0.0, candidate_norms, 1.0)
    scores = np.matmul(candidates, anchor) / (safe_norms * anchor_norm)
    scores = np.where(candidate_norms > 0.0, scores, 0.0)
    return scores.astype(np.float32, copy=False)


def _sorted_neighbor_indices(
    anchor_latent: np.ndarray,
    candidate_latents: np.ndarray,
    candidate_episode_ids: list[str],
    top_n: int,
) -> list[int]:
    if top_n <= 0 or len(candidate_episode_ids) == 0:
        return []
    similarities = _cosine_similarity(anchor_latent, candidate_latents)
    order = sorted(
        range(len(candidate_episode_ids)),
        key=lambda index: (-float(similarities[index]), candidate_episode_ids[index]),
    )
    return order[:top_n]


def _rank_action_details(
    model: Any,
    x: np.ndarray,
    metadata: BanditMetadata,
    scenario_id: str,
    action_id: str,
) -> dict[str, Any]:
    ranked = model.rank(
        x,
        metadata.candidate_action_ids(scenario_id),
        metadata.default_action_id(scenario_id),
        top_k=None,
    )
    for position, item in enumerate(ranked, start=1):
        if item.action_id == action_id:
            return {
                "rank": position,
                "score": item.score,
                "mean_reward": item.mean_reward,
                "uncertainty": item.uncertainty,
                "default_bonus": item.default_bonus,
                "in_top3": position <= 3,
                "top3_action_ids": [ranked_item.action_id for ranked_item in ranked[:3]],
                "top5_action_ids": [ranked_item.action_id for ranked_item in ranked[:5]],
            }
    raise KeyError(f"Action {action_id!r} was not returned by model.rank for scenario {scenario_id!r}")


def _evaluate_target_action_on_rows(
    model: Any,
    rows: list[dict[str, Any]],
    metadata: BanditMetadata,
    target_action_id: str,
) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "avg_rank": math.nan,
            "top3_rate": math.nan,
            "avg_score": math.nan,
            "avg_mean_reward": math.nan,
            "avg_uncertainty": math.nan,
        }

    rank_sum = 0.0
    top3_hits = 0
    score_sum = 0.0
    mean_reward_sum = 0.0
    uncertainty_sum = 0.0

    for row in rows:
        details = _rank_action_details(model, row["x"], metadata, row["scenario_id"], target_action_id)
        rank_sum += details["rank"]
        top3_hits += 1 if details["in_top3"] else 0
        score_sum += details["score"]
        mean_reward_sum += details["mean_reward"]
        uncertainty_sum += details["uncertainty"]

    sample_count = len(rows)
    return {
        "sample_count": sample_count,
        "avg_rank": rank_sum / sample_count,
        "top3_rate": top3_hits / sample_count,
        "avg_score": score_sum / sample_count,
        "avg_mean_reward": mean_reward_sum / sample_count,
        "avg_uncertainty": uncertainty_sum / sample_count,
    }


def _evaluate_ro_quality_on_rows(
    model: Any,
    rows: list[dict[str, Any]],
    metadata: BanditMetadata,
    relevance_catalog: dict[str, Any],
    top_k: int = 3,
) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "avg_most_relevant_covered_in_topk": math.nan,
            "avg_acceptable_covered_in_topk": math.nan,
            "avg_irrelevant_in_topk": math.nan,
        }

    total_most = 0.0
    total_acceptable = 0.0
    total_irrelevant = 0.0
    predicted_counts: Counter[str] = Counter()

    for row in rows:
        scenario_id = row["scenario_id"]
        rel = relevance_catalog[scenario_id]
        if isinstance(rel, dict):
            most_relevant_ids = tuple(rel.get("most_relevant_3", ()))
            plausible_ids = tuple(rel.get("other_plausible_3", ()))
            irrelevant_ids = tuple(rel.get("irrelevant_2", ())) + tuple(rel.get("extra_hard_negative_ro", ()))
        else:
            most_relevant_ids = tuple(getattr(rel, "most_relevant", ()))
            plausible_ids = tuple(getattr(rel, "plausible", ()))
            irrelevant_ids = tuple(getattr(rel, "irrelevant", ())) + tuple(getattr(rel, "extra_hard_negative", ()))
        ranked = model.rank(
            row["x"],
            metadata.candidate_action_ids(scenario_id),
            metadata.default_action_id(scenario_id),
            top_k=top_k,
        )
        predicted = tuple(item.action_id for item in ranked)
        predicted_counts.update(predicted)
        most_relevant = {metadata.resolve_action_token(action_id, scenario_id) for action_id in most_relevant_ids}
        plausible = {metadata.resolve_action_token(action_id, scenario_id) for action_id in plausible_ids}
        irrelevant = {metadata.resolve_action_token(action_id, scenario_id) for action_id in irrelevant_ids}
        total_most += sum(1 for action_id in predicted if action_id in most_relevant)
        total_acceptable += sum(1 for action_id in predicted if action_id in most_relevant or action_id in plausible)
        total_irrelevant += sum(1 for action_id in predicted if action_id in irrelevant)

    sample_count = len(rows)
    return {
        "sample_count": sample_count,
        "avg_most_relevant_covered_in_topk": total_most / sample_count,
        "avg_acceptable_covered_in_topk": total_acceptable / sample_count,
        "avg_irrelevant_in_topk": total_irrelevant / sample_count,
        "top10_predicted_action_distribution_top3": _top_action_distribution_topk(predicted_counts, sample_count, top_k, limit=10),
    }


def _render_feedback_propagation_markdown(summary: FeedbackPropagationExperimentSummary) -> str:
    lines = [
        "# Feedback Propagation Experiment",
        "",
        f"- artifact_dir: `{summary.artifact_dir}`",
        f"- device: `{summary.device}`",
        f"- n_values: `{','.join(str(value) for value in summary.n_values)}`",
        "",
        "## Locked Feedback Items",
        "",
        "| scenario | feedback | target_position | target_action_id | anchor_episode_id | baseline_top3 |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for item in summary.feedback_items:
        lines.append(
            f"| `{item['scenario_id']}` | `{item['feedback_type']}` | {item['target_position']} | "
            f"`{item['target_action_id']}` | `{item['anchor_episode_id']}` | "
            f"`{' , '.join(item['baseline_top3_action_ids'])}` |"
        )

    lines.extend(
        [
            "",
            "## Condition Summary",
            "",
            "| mode | N | avg anchor rank delta | avg same-scenario rank delta | avg cross-scenario rank delta | selected-scenario most_rel before | selected-scenario most_rel after |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in summary.conditions:
        lines.append(
            f"| `{condition['mode']}` | `{condition['requested_n']}` | "
            f"{condition['aggregate']['avg_anchor_rank_delta']:.3f} | "
            f"{condition['aggregate']['avg_same_scenario_rank_delta']:.3f} | "
            f"{condition['aggregate']['avg_cross_scenario_rank_delta']:.3f} | "
            f"{condition['selected_scenarios_quality_before']['avg_most_relevant_covered_in_topk']:.3f} | "
            f"{condition['selected_scenarios_quality_after']['avg_most_relevant_covered_in_topk']:.3f} |"
        )
    return "\n".join(lines) + "\n"


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
    other_zero_mode: str,
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
    model_type: str,
) -> V6WorkflowSummary:
    output_path = Path(output_dir)
    _write_json(
        output_path / "v6_training_config.json",
        {
            "relevance_markdown": str(Path(relevance_markdown)),
            "most_relevant_reward": float(most_relevant_reward),
            "plausible_reward": float(plausible_reward),
            "irrelevant_reward": float(irrelevant_reward),
            "most_relevant_repeat": int(most_relevant_repeat),
            "plausible_repeat": int(plausible_repeat),
            "irrelevant_repeat": int(irrelevant_repeat),
            "other_zero_mode": other_zero_mode,
        },
    )
    ro_metadata = BanditMetadata.load(output_path / "ro_metadata.json")
    app_metadata = BanditMetadata.load(output_path / "app_metadata.json")
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
        all_action_ids=list(ro_metadata.global_action_ids),
        other_zero_mode=other_zero_mode,
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
        all_action_ids=list(app_metadata.global_action_ids),
        other_zero_mode=other_zero_mode,
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
        model_type=model_type,
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
    other_zero_mode: str = "none",
    alpha: float = 0.05,
    default_bonus: float = 0.0,
    l2: float = 1.0,
    epochs: int = 1,
    top_k: int = 3,
    device: str = "auto",
    progress_every: int = 1000,
    track_train_hit_rate: bool = False,
    model_type: str = "disjoint",
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
        other_zero_mode=other_zero_mode,
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
            "other_zero_mode": other_zero_mode,
            "epochs": int(epochs),
            "model_type": model_type,
        },
        track_train_hit_rate=track_train_hit_rate,
        model_type=model_type,
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
    other_zero_mode: str = "none",
    alpha: float = 0.05,
    default_bonus: float = 0.0,
    l2: float = 1.0,
    epochs: int = 1,
    top_k: int = 3,
    device: str = "auto",
    progress_every: int = 1000,
    track_train_hit_rate: bool = False,
    model_type: str = "disjoint",
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
        other_zero_mode=other_zero_mode,
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
            "other_zero_mode": other_zero_mode,
            "epochs": int(epochs),
            "model_type": model_type,
            "split_summary": asdict(split_summary),
            "ro_metadata": asdict(ro_catalog),
            "app_metadata": asdict(app_catalog),
        },
        track_train_hit_rate=track_train_hit_rate,
        model_type=model_type,
    )


def run_v6_plan_all_data(
    input_path: str | Path,
    catalog_markdown: str | Path,
    output_dir: str | Path,
    relevance_markdown: str | Path = "docs/scenario_recommendation_actions_v6.md",
    most_relevant_reward: float = 1.0,
    plausible_reward: float = 0.6,
    irrelevant_reward: float = 0.0,
    most_relevant_repeat: int = 1,
    plausible_repeat: int = 1,
    irrelevant_repeat: int = 1,
    other_zero_mode: str = "none",
    alpha: float = 0.05,
    default_bonus: float = 0.0,
    l2: float = 1.0,
    epochs: int = 1,
    top_k: int = 3,
    device: str = "auto",
    progress_every: int = 1000,
    track_train_hit_rate: bool = False,
    model_type: str = "disjoint",
) -> V6WorkflowSummary:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ro_catalog = build_ro_metadata_from_catalog_markdown(catalog_markdown, output_path / "ro_metadata.json")
    app_catalog = build_app_metadata_from_catalog_markdown(catalog_markdown, output_path / "app_metadata.json")

    train_raw_path = output_path / "train.raw.jsonl"
    test_raw_path = output_path / "test.raw.jsonl"
    _copy_file(input_path, train_raw_path)
    _copy_file(input_path, test_raw_path)

    convert_raw_sequence_to_v0(
        input_path=test_raw_path,
        output_samples_path=output_path / "ro_test_samples.jsonl",
        output_metadata_path=output_path / "ro_metadata_inferred_test.json",
        reward=1.0,
    )
    convert_raw_sequence_to_v0_app(
        input_path=test_raw_path,
        output_samples_path=output_path / "app_test_samples.jsonl",
        output_metadata_path=output_path / "app_metadata_inferred_test.json",
        reward=1.0,
    )

    raw_rows = _load_jsonl_for_split(input_path)
    scenario_ids = sorted(
        {
            str(row.get("scenario_id") or row.get("scenarioId"))
            for row in raw_rows
            if row.get("scenario_id") or row.get("scenarioId")
        }
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
        other_zero_mode=other_zero_mode,
        alpha=alpha,
        default_bonus=default_bonus,
        l2=l2,
        epochs=epochs,
        top_k=top_k,
        device=device,
        progress_every=progress_every,
        workflow_name="run_v6_plan_all_data",
        extra_summary={
            "input_path": str(Path(input_path)),
            "catalog_markdown": str(Path(catalog_markdown)),
            "split_policy": "all_rows_for_train_and_test",
            "raw_row_count": len(raw_rows),
            "scenario_count": len(scenario_ids),
            "scenario_ids": scenario_ids,
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
            "other_zero_mode": other_zero_mode,
            "epochs": int(epochs),
            "model_type": model_type,
            "ro_metadata": asdict(ro_catalog),
            "app_metadata": asdict(app_catalog),
        },
        track_train_hit_rate=track_train_hit_rate,
        model_type=model_type,
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
    model_type: str = "disjoint",
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
        "[WORKFLOW] Starting joint training | data_dir=%s | model_type=%s | device=%s | epochs=%d | track_train_hit_rate=%s",
        data_path,
        model_type,
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
        model_type=model_type,
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
        model_type=model_type,
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
        "model_type": model_type,
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


def _select_hard_negative_candidates(
    counts: Counter[str],
    *,
    sample_count: int,
    exclude_action_ids: set[str],
    max_candidates: int,
    min_count: int = 1,
    min_rate: float = 0.0,
) -> list[dict[str, Any]]:
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    if not 0.0 <= min_rate <= 1.0:
        raise ValueError("min_rate must be in [0, 1]")
    if sample_count <= 0 or max_candidates == 0:
        return []

    rows: list[dict[str, Any]] = []
    for action_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if action_id in exclude_action_ids:
            continue
        rate = count / sample_count
        if count < min_count or rate < min_rate:
            continue
        rows.append(
            {
                "action_id": action_id,
                "display_action_id": _display_action_id(action_id),
                "count": count,
                "rate": rate,
            }
        )
        if len(rows) >= max_candidates:
            break
    return rows


def evaluate_soft_scenarios(
    artifact_dir: str | Path,
    metadata_path: str | Path,
    raw_input_path: str | Path,
    spec_markdown: str | Path,
    label_namespace: str,
    device: str = "auto",
    progress_every: int = 1000,
    progress_label: str | None = None,
) -> SoftScenarioEvalSummary:
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if label_namespace not in {"ro", "app"}:
        raise ValueError("label_namespace must be 'ro' or 'app'")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    label_prefix = f"[{progress_label}] " if progress_label else ""

    metadata = BanditMetadata.load(metadata_path)
    soft_catalog = parse_in_between_scenarios_markdown(spec_markdown)
    feature_space = V0FeatureSpace()
    model = load_bandit_model(artifact_dir, device=device)
    rows = _load_jsonl(raw_input_path)
    if not rows:
        raise ValueError(f"No raw rows found in {raw_input_path}")

    total_rows = len(rows)
    start_time = time.time()
    logger.info(
        "%sStarting soft-scenario evaluation | rows=%d | label=%s | device=%s",
        label_prefix,
        total_rows,
        label_namespace,
        device,
    )

    pred_top5_counts: Counter[str] = Counter()
    scenario_row_counts: Counter[str] = Counter()
    scenario_pred_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scenario_hits_top3: Counter[str] = Counter()
    scenario_hits_top5: Counter[str] = Counter()
    total_hits_top3 = 0.0
    total_hits_top5 = 0.0
    interval_hits_top3 = 0.0
    interval_hits_top5 = 0.0
    skipped_rows_missing_scenario = 0

    for index, row in enumerate(rows, start=1):
        scenario_id = _scenario_id(row)
        if scenario_id is None or scenario_id not in soft_catalog:
            skipped_rows_missing_scenario += 1
            continue

        event = TrainingEvent(
            context=_build_context(row),
            selected_action="NONE",
            reward=0.0,
            scenario_id=scenario_id,
            event_id=str(row.get("sample_id") or row.get("episode_id") or index),
        )
        x = feature_space.encode(event.context)
        ranked = model.rank(
            x,
            metadata.candidate_action_ids(scenario_id),
            metadata.default_action_id(scenario_id),
            top_k=5,
        )
        pred_top5 = tuple(item.action_id for item in ranked)
        pred_top3 = pred_top5[:3]

        if label_namespace == "ro":
            reference_actions = tuple(
                metadata.resolve_action_token(action_id, None)
                for action_id in soft_catalog[scenario_id]["ro_top10"]
            )
            reference_size = 10
        else:
            reference_actions = tuple(
                metadata.resolve_action_token(action_id, None)
                for action_id in soft_catalog[scenario_id]["app_top5"]
            )
            reference_size = 5
        reference_set = set(reference_actions)

        hits_top3 = sum(1 for action_id in pred_top3 if action_id in reference_set)
        hits_top5 = sum(1 for action_id in pred_top5 if action_id in reference_set)

        total_hits_top3 += hits_top3
        total_hits_top5 += hits_top5
        interval_hits_top3 += hits_top3
        interval_hits_top5 += hits_top5

        scenario_row_counts[scenario_id] += 1
        scenario_hits_top3[scenario_id] += hits_top3
        scenario_hits_top5[scenario_id] += hits_top5
        for action_id in pred_top5:
            pred_top5_counts[action_id] += 1
            scenario_pred_counts[scenario_id][action_id] += 1

        if index % progress_every == 0:
            elapsed = time.time() - start_time
            rows_per_sec = index / elapsed if elapsed > 0 else 0.0
            remaining = max(total_rows - index, 0)
            eta_seconds = remaining / rows_per_sec if rows_per_sec > 0 else 0.0
            logger.info(
                "%sProcessed %d/%d rows | interval hits@3=%.3f/3 | cumulative hits@3=%.3f/3 | interval hits@5=%.3f/5 | cumulative hits@5=%.3f/5 | elapsed=%s | eta=%s | %.1f rows/s",
                label_prefix,
                index,
                total_rows,
                interval_hits_top3 / progress_every,
                total_hits_top3 / max(index - skipped_rows_missing_scenario, 1),
                interval_hits_top5 / progress_every,
                total_hits_top5 / max(index - skipped_rows_missing_scenario, 1),
                _format_duration(elapsed),
                _format_duration(eta_seconds),
                rows_per_sec,
            )
            interval_hits_top3 = 0.0
            interval_hits_top5 = 0.0

    remainder = total_rows % progress_every
    evaluated_rows = total_rows - skipped_rows_missing_scenario
    if remainder:
        elapsed = time.time() - start_time
        rows_per_sec = total_rows / elapsed if elapsed > 0 else 0.0
        logger.info(
            "%sProcessed %d/%d rows | interval hits@3=%.3f/3 | cumulative hits@3=%.3f/3 | interval hits@5=%.3f/5 | cumulative hits@5=%.3f/5 | elapsed=%s | eta=0s | %.1f rows/s",
            label_prefix,
            total_rows,
            total_rows,
            interval_hits_top3 / remainder,
            total_hits_top3 / max(evaluated_rows, 1),
            interval_hits_top5 / remainder,
            total_hits_top5 / max(evaluated_rows, 1),
            _format_duration(elapsed),
            rows_per_sec,
        )

    per_scenario: dict[str, dict[str, Any]] = {}
    for scenario_id in sorted(soft_catalog):
        row_count = scenario_row_counts[scenario_id]
        if row_count == 0:
            continue
        if label_namespace == "ro":
            reference_actions = [metadata.resolve_action_token(action_id, None) for action_id in soft_catalog[scenario_id]["ro_top10"]]
        else:
            reference_actions = [metadata.resolve_action_token(action_id, None) for action_id in soft_catalog[scenario_id]["app_top5"]]
        per_scenario[scenario_id] = {
            "sample_count": row_count,
            "reference_actions": reference_actions,
            "avg_hits_in_top3": scenario_hits_top3[scenario_id] / row_count,
            "avg_hits_in_top3_ratio": scenario_hits_top3[scenario_id] / (row_count * 3.0),
            "avg_hits_in_top5": scenario_hits_top5[scenario_id] / row_count,
            "avg_hits_in_top5_ratio": scenario_hits_top5[scenario_id] / (row_count * 5.0),
            "predicted_top10_action_distribution_top5": _top_action_distribution(
                scenario_pred_counts[scenario_id],
                sample_count=row_count,
                top_k=5,
                limit=10,
            ),
        }

    elapsed_total = time.time() - start_time
    logger.info(
        "%sFinished soft-scenario evaluation | rows=%d | evaluated=%d | avg hits@3=%.3f/3 | avg hits@5=%.3f/5 | elapsed=%s | avg %.1f rows/s",
        label_prefix,
        total_rows,
        evaluated_rows,
        total_hits_top3 / max(evaluated_rows, 1),
        total_hits_top5 / max(evaluated_rows, 1),
        _format_duration(elapsed_total),
        total_rows / elapsed_total if elapsed_total > 0 else 0.0,
    )

    return SoftScenarioEvalSummary(
        sample_count=evaluated_rows,
        reference_size=10 if label_namespace == "ro" else 5,
        avg_hits_in_top3=total_hits_top3 / max(evaluated_rows, 1),
        avg_hits_in_top3_ratio=total_hits_top3 / max(evaluated_rows * 3.0, 1.0),
        avg_hits_in_top5=total_hits_top5 / max(evaluated_rows, 1),
        avg_hits_in_top5_ratio=total_hits_top5 / max(evaluated_rows * 5.0, 1.0),
        top10_predicted_action_distribution_top5=_top_action_distribution(
            pred_top5_counts,
            sample_count=max(evaluated_rows, 1),
            top_k=5,
            limit=10,
        ),
        scenarios_with_samples=len(per_scenario),
        scenarios_without_samples=[scenario_id for scenario_id in sorted(soft_catalog) if scenario_row_counts[scenario_id] == 0],
        per_scenario=per_scenario,
    )


def eval_soft_scenarios_both(
    data_dir: str | Path,
    raw_input_path: str | Path,
    spec_markdown: str | Path,
    device: str = "auto",
    progress_every: int = 1000,
) -> SoftScenarioEvalBothSummary:
    data_path = Path(data_dir)
    logger.info(
        "[WORKFLOW] Starting soft-scenario evaluation | data_dir=%s | raw_input=%s | device=%s",
        data_path,
        raw_input_path,
        device,
    )
    ro = evaluate_soft_scenarios(
        artifact_dir=data_path / "ro_model",
        metadata_path=data_path / "ro_metadata.json",
        raw_input_path=raw_input_path,
        spec_markdown=spec_markdown,
        label_namespace="ro",
        device=device,
        progress_every=progress_every,
        progress_label="RO-SOFT-EVAL",
    )
    app = evaluate_soft_scenarios(
        artifact_dir=data_path / "app_model",
        metadata_path=data_path / "app_metadata.json",
        raw_input_path=raw_input_path,
        spec_markdown=spec_markdown,
        label_namespace="app",
        device=device,
        progress_every=progress_every,
        progress_label="APP-SOFT-EVAL",
    )
    summary = SoftScenarioEvalBothSummary(
        data_dir=str(data_path),
        raw_input_path=str(Path(raw_input_path)),
        spec_markdown=str(Path(spec_markdown)),
        device=device,
        ro=ro,
        app=app,
    )
    _write_json_unsorted(
        data_path / "eval_soft_scenarios.json",
        {
            "data_dir": summary.data_dir,
            "raw_input_path": summary.raw_input_path,
            "spec_markdown": summary.spec_markdown,
            "device": summary.device,
            "ro": asdict(summary.ro),
            "app": asdict(summary.app),
        },
    )
    logger.info("[WORKFLOW] Finished soft-scenario evaluation")
    return summary


def simulate_feedback_propagation_on_frozen_neural_linear(
    artifact_dir: str | Path,
    relevance_markdown: str | Path = "docs/scenario_recommendation_actions_v6.md",
    n_values: list[int] | None = None,
    output_dir: str | Path | None = None,
    device: str = "auto",
    progress_every: int = 25000,
) -> FeedbackPropagationExperimentSummary:
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    n_values = [1, 5, 10, 20, 50, 100] if n_values is None else sorted({int(value) for value in n_values if int(value) > 0})
    if not n_values:
        raise ValueError("n_values must contain at least one positive integer")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_path = Path(artifact_dir)
    output_path = Path(output_dir) if output_dir is not None else data_path / "feedback_propagation_ro_v1"
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = BanditMetadata.load(data_path / "ro_metadata.json")
    feature_space = V0FeatureSpace()
    base_model = load_bandit_model(data_path / "ro_model", device=device)
    if not hasattr(base_model, "encode_context") or not hasattr(base_model, "partial_fit"):
        raise ValueError("This workflow requires a frozen neural-linear style model with encode_context and partial_fit support")

    relevance_catalog = parse_v6_relevance_markdown(relevance_markdown)["ro"]
    train_rows_raw = _load_jsonl(data_path / "train.raw.jsonl")
    test_rows_raw = _load_jsonl(data_path / "test.raw.jsonl")
    if not train_rows_raw or not test_rows_raw:
        raise ValueError("artifact_dir must contain non-empty train.raw.jsonl and test.raw.jsonl")

    def _encode_raw_rows(raw_rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        encoded_rows: list[dict[str, Any]] = []
        start = time.time()
        for index, row in enumerate(raw_rows, start=1):
            scenario_id = _scenario_id(row)
            if scenario_id is None:
                continue
            episode_id = str(row.get("episode_id") or row.get("sample_id") or index)
            context = _build_context(row)
            x = feature_space.encode(context)
            latent = base_model.encode_context(x)
            encoded_rows.append(
                {
                    "scenario_id": scenario_id,
                    "episode_id": episode_id,
                    "context": context,
                    "x": x,
                    "latent": latent,
                }
            )
            if index % progress_every == 0:
                elapsed = time.time() - start
                rows_per_sec = index / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[FEEDBACK-%s] Encoded %d/%d rows | elapsed=%s | %.1f rows/s",
                    label,
                    index,
                    len(raw_rows),
                    _format_duration(elapsed),
                    rows_per_sec,
                )
        return encoded_rows

    train_rows = _encode_raw_rows(train_rows_raw, "TRAIN")
    test_rows = _encode_raw_rows(test_rows_raw, "TEST")

    train_latents = np.stack([row["latent"] for row in train_rows]).astype(np.float32, copy=False)
    train_episode_ids = [str(row["episode_id"]) for row in train_rows]
    scenario_to_train_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        scenario_to_train_indices[row["scenario_id"]].append(index)

    anchor_rows: dict[str, dict[str, Any]] = {}
    for scenario_id, _, _ in _FEEDBACK_PROPAGATION_SCENARIOS:
        matching = [row for row in test_rows if row["scenario_id"] == scenario_id]
        if not matching:
            raise ValueError(f"No test.raw rows found for required feedback scenario {scenario_id!r}")
        anchor_rows[scenario_id] = min(matching, key=lambda row: row["episode_id"])

    locked_feedback_items: list[dict[str, Any]] = []
    for scenario_id, feedback_type, target_position in _FEEDBACK_PROPAGATION_SCENARIOS:
        anchor = anchor_rows[scenario_id]
        anchor_ranked = base_model.rank(
            anchor["x"],
            metadata.candidate_action_ids(scenario_id),
            metadata.default_action_id(scenario_id),
            top_k=5,
        )
        if len(anchor_ranked) < target_position:
            raise ValueError(f"Scenario {scenario_id!r} returned only {len(anchor_ranked)} candidates; cannot use target position {target_position}")
        reward = 1.0 if feedback_type == "like" else -1.0
        locked_feedback_items.append(
            {
                "scenario_id": scenario_id,
                "feedback_type": feedback_type,
                "target_position": target_position,
                "target_action_id": anchor_ranked[target_position - 1].action_id,
                "reward": reward,
                "anchor_episode_id": anchor["episode_id"],
                "anchor_context": anchor["context"],
                "anchor_top5": [
                    {
                        "action_id": item.action_id,
                        "score": item.score,
                        "mean_reward": item.mean_reward,
                        "uncertainty": item.uncertainty,
                    }
                    for item in anchor_ranked
                ],
                "baseline_top3_action_ids": [item.action_id for item in anchor_ranked[:3]],
            }
        )

    _write_json_unsorted(output_path / "locked_feedback_spec.json", {"feedback_items": locked_feedback_items})
    _write_json_unsorted(
        output_path / "baseline_anchor_predictions.json",
        {
            "anchors": [
                {
                    "scenario_id": item["scenario_id"],
                    "anchor_episode_id": item["anchor_episode_id"],
                    "top5": item["anchor_top5"],
                }
                for item in locked_feedback_items
            ]
        },
    )

    def _baseline_target_stats_by_feedback() -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for item in locked_feedback_items:
            scenario_id = item["scenario_id"]
            target_action_id = item["target_action_id"]
            anchor = anchor_rows[scenario_id]
            same_rows = [row for row in test_rows if row["scenario_id"] == scenario_id and row["episode_id"] != anchor["episode_id"]]
            cross_rows = [row for row in test_rows if row["scenario_id"] != scenario_id]
            stats[scenario_id] = {
                "anchor": _rank_action_details(base_model, anchor["x"], metadata, scenario_id, target_action_id),
                "same_scenario": _evaluate_target_action_on_rows(base_model, same_rows, metadata, target_action_id),
                "cross_scenario": _evaluate_target_action_on_rows(base_model, cross_rows, metadata, target_action_id),
            }
        return stats

    baseline_target_stats = _baseline_target_stats_by_feedback()
    selected_test_rows = [row for row in test_rows if row["scenario_id"] in {scenario_id for scenario_id, _, _ in _FEEDBACK_PROPAGATION_SCENARIOS}]
    selected_scenarios_quality_before = _evaluate_ro_quality_on_rows(
        base_model,
        selected_test_rows,
        metadata,
        relevance_catalog,
        top_k=3,
    )

    def _propagation_rows_for_item(item: dict[str, Any], mode: str, requested_n: int | str) -> list[dict[str, Any]]:
        if mode == "single":
            return [anchor_rows[item["scenario_id"]]]
        if mode == "global-latent-nearest":
            indices = _sorted_neighbor_indices(item_anchor_latents[item["scenario_id"]], train_latents, train_episode_ids, int(requested_n))
            return [train_rows[index] for index in indices]
        if mode == "same-scenario-nearest":
            same_indices = scenario_to_train_indices[item["scenario_id"]]
            same_latents = train_latents[same_indices]
            same_episode_ids = [train_episode_ids[index] for index in same_indices]
            selected_local = _sorted_neighbor_indices(item_anchor_latents[item["scenario_id"]], same_latents, same_episode_ids, int(requested_n))
            return [train_rows[same_indices[index]] for index in selected_local]
        if mode == "entire-scenario-all":
            return [train_rows[index] for index in scenario_to_train_indices[item["scenario_id"]]]
        raise ValueError(f"Unknown propagation mode {mode!r}")

    item_anchor_latents = {item["scenario_id"]: anchor_rows[item["scenario_id"]]["latent"] for item in locked_feedback_items}

    conditions: list[dict[str, Any]] = []
    condition_specs: list[tuple[str, int | str]] = [("single", 1)]
    condition_specs.extend(("global-latent-nearest", value) for value in n_values)
    condition_specs.extend(("same-scenario-nearest", value) for value in n_values)
    condition_specs.append(("entire-scenario-all", "all"))

    for mode, requested_n in condition_specs:
        logger.info("[FEEDBACK] Running condition | mode=%s | requested_n=%s", mode, requested_n)
        model = load_bandit_model(data_path / "ro_model", device=device)
        feedback_results: list[dict[str, Any]] = []

        for item in locked_feedback_items:
            propagation_rows = _propagation_rows_for_item(item, mode, requested_n)
            for row in propagation_rows:
                model.partial_fit(row["x"], item["target_action_id"], item["reward"])

            scenario_id = item["scenario_id"]
            target_action_id = item["target_action_id"]
            anchor = anchor_rows[scenario_id]
            same_rows = [row for row in test_rows if row["scenario_id"] == scenario_id and row["episode_id"] != anchor["episode_id"]]
            cross_rows = [row for row in test_rows if row["scenario_id"] != scenario_id]

            anchor_before = baseline_target_stats[scenario_id]["anchor"]
            anchor_after = _rank_action_details(model, anchor["x"], metadata, scenario_id, target_action_id)
            same_before = baseline_target_stats[scenario_id]["same_scenario"]
            same_after = _evaluate_target_action_on_rows(model, same_rows, metadata, target_action_id)
            cross_before = baseline_target_stats[scenario_id]["cross_scenario"]
            cross_after = _evaluate_target_action_on_rows(model, cross_rows, metadata, target_action_id)

            feedback_results.append(
                {
                    "scenario_id": scenario_id,
                    "feedback_type": item["feedback_type"],
                    "target_action_id": target_action_id,
                    "reward": item["reward"],
                    "requested_n": requested_n,
                    "effective_n": len(propagation_rows),
                    "anchor": {
                        "before": anchor_before,
                        "after": anchor_after,
                        "rank_delta": anchor_before["rank"] - anchor_after["rank"],
                        "score_delta": anchor_after["score"] - anchor_before["score"],
                    },
                    "same_scenario": {
                        "before": same_before,
                        "after": same_after,
                        "avg_rank_delta": same_before["avg_rank"] - same_after["avg_rank"] if not math.isnan(same_before["avg_rank"]) else math.nan,
                        "avg_score_delta": same_after["avg_score"] - same_before["avg_score"] if not math.isnan(same_before["avg_score"]) else math.nan,
                    },
                    "cross_scenario": {
                        "before": cross_before,
                        "after": cross_after,
                        "avg_rank_delta": cross_before["avg_rank"] - cross_after["avg_rank"] if not math.isnan(cross_before["avg_rank"]) else math.nan,
                        "avg_score_delta": cross_after["avg_score"] - cross_before["avg_score"] if not math.isnan(cross_before["avg_score"]) else math.nan,
                    },
                }
            )

        selected_scenarios_quality_after = _evaluate_ro_quality_on_rows(
            model,
            selected_test_rows,
            metadata,
            relevance_catalog,
            top_k=3,
        )

        valid_anchor_deltas = [row["anchor"]["rank_delta"] for row in feedback_results]
        valid_same_deltas = [row["same_scenario"]["avg_rank_delta"] for row in feedback_results if not math.isnan(row["same_scenario"]["avg_rank_delta"])]
        valid_cross_deltas = [row["cross_scenario"]["avg_rank_delta"] for row in feedback_results if not math.isnan(row["cross_scenario"]["avg_rank_delta"])]
        condition_summary = {
            "mode": mode,
            "requested_n": requested_n,
            "feedback_results": feedback_results,
            "aggregate": {
                "avg_anchor_rank_delta": sum(valid_anchor_deltas) / len(valid_anchor_deltas) if valid_anchor_deltas else math.nan,
                "avg_same_scenario_rank_delta": sum(valid_same_deltas) / len(valid_same_deltas) if valid_same_deltas else math.nan,
                "avg_cross_scenario_rank_delta": sum(valid_cross_deltas) / len(valid_cross_deltas) if valid_cross_deltas else math.nan,
            },
            "selected_scenarios_quality_before": selected_scenarios_quality_before,
            "selected_scenarios_quality_after": selected_scenarios_quality_after,
        }
        conditions.append(condition_summary)

    summary = FeedbackPropagationExperimentSummary(
        artifact_dir=str(data_path),
        output_dir=str(output_path),
        relevance_markdown=str(Path(relevance_markdown)),
        device=device,
        n_values=n_values,
        feedback_items=locked_feedback_items,
        conditions=conditions,
    )
    _write_json_unsorted(
        output_path / "results.json",
        {
            "artifact_dir": summary.artifact_dir,
            "output_dir": summary.output_dir,
            "relevance_markdown": summary.relevance_markdown,
            "device": summary.device,
            "n_values": summary.n_values,
            "feedback_items": summary.feedback_items,
            "conditions": summary.conditions,
        },
    )
    (output_path / "summary_table.md").write_text(_render_feedback_propagation_markdown(summary))
    return summary


def render_hard_negative_candidates_markdown(summary: HardNegativeMiningSummary) -> str:
    lines = [
        "# Hard-Negative Candidate Mining",
        "",
        f"- label_namespace: `{summary.label_namespace}`",
        f"- top_k: `{summary.top_k}`",
        f"- sample_count: `{summary.sample_count}`",
        f"- scenarios_with_samples: `{summary.scenarios_with_samples}`",
        f"- max_candidates_per_scenario: `{summary.max_candidates_per_scenario}`",
        "",
        "| scenario_id | sample_count | most_relevant_3 | plausible_3 | current_irrelevant_2 | candidate_hard_negatives |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for scenario_id in sorted(summary.per_scenario):
        row = summary.per_scenario[scenario_id]
        formatted_candidates = "<br>".join(
            f"`{candidate['display_action_id']}` ({candidate['count']}, {candidate['rate']:.1%})"
            for candidate in row["candidate_hard_negatives"]
        ) or "-"
        lines.append(
            "| {scenario_id} | {sample_count} | {most_relevant} | {plausible} | {irrelevant} | {candidates} |".format(
                scenario_id=scenario_id,
                sample_count=row["sample_count"],
                most_relevant="<br>".join(f"`{_display_action_id(action_id)}`" for action_id in row["most_relevant_3"]),
                plausible="<br>".join(f"`{_display_action_id(action_id)}`" for action_id in row["other_plausible_3"]),
                irrelevant="<br>".join(f"`{_display_action_id(action_id)}`" for action_id in row["irrelevant_2"]) or "-",
                candidates=formatted_candidates,
            )
        )
    return "\n".join(lines) + "\n"


def mine_v6_hard_negative_candidates(
    artifact_dir: str | Path,
    metadata_path: str | Path,
    raw_input_path: str | Path,
    catalog_markdown: str | Path,
    label_namespace: str,
    top_k: int = 6,
    max_candidates_per_scenario: int = 5,
    device: str = "auto",
    progress_every: int = 1000,
    min_count: int = 1,
    min_rate: float = 0.0,
) -> HardNegativeMiningSummary:
    if label_namespace not in {"ro", "app"}:
        raise ValueError("label_namespace must be 'ro' or 'app'")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    metadata = BanditMetadata.load(metadata_path)
    relevance_catalog = parse_v6_relevance_markdown(catalog_markdown)[label_namespace]
    feature_space = V0FeatureSpace()
    model = load_bandit_model(artifact_dir, device=device)
    raw_rows = _load_jsonl(raw_input_path)
    if not raw_rows:
        raise ValueError(f"No raw rows found in {raw_input_path}")

    logger.info(
        "Starting hard-negative mining | rows=%d | top_k=%d | label=%s | device=%s",
        len(raw_rows),
        top_k,
        label_namespace,
        device,
    )
    start_time = time.time()
    per_scenario_counts: dict[str, Counter[str]] = defaultdict(Counter)
    per_scenario_samples: Counter[str] = Counter()
    skipped_rows_missing_scenario = 0

    for index, row in enumerate(raw_rows, start=1):
        scenario_id = _scenario_id(row)
        if scenario_id is None or scenario_id not in relevance_catalog:
            skipped_rows_missing_scenario += 1
            continue

        context = _build_context(row)
        x = feature_space.encode(context)
        ranked = model.rank(
            x,
            metadata.candidate_action_ids(scenario_id),
            metadata.default_action_id(scenario_id),
            top_k=top_k,
        )
        scenario_spec = relevance_catalog[scenario_id]
        most_relevant = {
            metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["most_relevant_3"]
        }
        plausible = {
            metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["other_plausible_3"]
        }
        irrelevant = {
            metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["irrelevant_2"]
        }
        exclude = most_relevant | plausible | irrelevant
        per_scenario_samples[scenario_id] += 1
        for item in ranked:
            if item.action_id not in exclude:
                per_scenario_counts[scenario_id][item.action_id] += 1

        if index % progress_every == 0:
            elapsed = time.time() - start_time
            rows_per_sec = index / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Processed %d/%d rows for mining | elapsed=%s | %.1f rows/s",
                index,
                len(raw_rows),
                _format_duration(elapsed),
                rows_per_sec,
            )

    per_scenario: dict[str, dict[str, Any]] = {}
    for scenario_id in sorted(relevance_catalog):
        row_count = per_scenario_samples[scenario_id]
        if row_count == 0:
            continue
        scenario_spec = relevance_catalog[scenario_id]
        most_relevant = [metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["most_relevant_3"]]
        plausible = [metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["other_plausible_3"]]
        irrelevant = [metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_spec["irrelevant_2"]]
        candidates = _select_hard_negative_candidates(
            per_scenario_counts[scenario_id],
            sample_count=row_count,
            exclude_action_ids=set(),
            max_candidates=max_candidates_per_scenario,
            min_count=min_count,
            min_rate=min_rate,
        )
        per_scenario[scenario_id] = {
            "sample_count": row_count,
            "most_relevant_3": most_relevant,
            "other_plausible_3": plausible,
            "irrelevant_2": irrelevant,
            "candidate_hard_negatives": candidates,
            "top10_non_acceptable_predicted_actions": _top_action_distribution(
                per_scenario_counts[scenario_id],
                sample_count=row_count,
                top_k=top_k,
                limit=10,
            ),
        }

    elapsed_total = time.time() - start_time
    evaluated_rows = len(raw_rows) - skipped_rows_missing_scenario
    logger.info(
        "Finished hard-negative mining | rows=%d | evaluated=%d | scenarios=%d | elapsed=%s | avg %.1f rows/s",
        len(raw_rows),
        evaluated_rows,
        len(per_scenario),
        _format_duration(elapsed_total),
        len(raw_rows) / elapsed_total if elapsed_total > 0 else 0.0,
    )
    return HardNegativeMiningSummary(
        raw_input_path=str(Path(raw_input_path)),
        artifact_dir=str(Path(artifact_dir)),
        metadata_path=str(Path(metadata_path)),
        catalog_markdown=str(Path(catalog_markdown)),
        label_namespace=label_namespace,
        top_k=top_k,
        sample_count=evaluated_rows,
        scenarios_with_samples=len(per_scenario),
        max_candidates_per_scenario=max_candidates_per_scenario,
        per_scenario=per_scenario,
    )


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
    model = load_bandit_model(artifact_dir, device=device)
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
        if label_namespace == "ro":
            irrelevant = irrelevant + tuple(
                metadata.resolve_action_token(action_id, scenario_id)
                for action_id in scenario_spec.get("extra_hard_negative_ro", ())
            )
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
        extra_hard_negative_ro = []
        if label_namespace == "ro":
            extra_hard_negative_ro = [
                metadata.resolve_action_token(action_id, scenario_id)
                for action_id in scenario_spec.get("extra_hard_negative_ro", ())
            ]
        scenario_eval[scenario_id] = {
            "sample_count": row_count,
            "most_relevant_3": most_relevant,
            "other_plausible_3": plausible,
            "acceptable_6": most_relevant + plausible,
            "irrelevant_2": irrelevant,
            "extra_hard_negative_ro": extra_hard_negative_ro,
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
