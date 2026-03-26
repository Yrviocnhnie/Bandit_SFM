"""High-level one-line workflows for V0 data prep, training, and evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
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
from recommendation_agents.raw_synthetic import convert_raw_sequence_to_v0, convert_raw_sequence_to_v0_app
from recommendation_agents.schemas import TrainingEvent
from recommendation_agents.trainer import TrainingMetrics, _format_duration, train_v0


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


def _write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_json_unsorted(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


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


def train_v0_both(
    data_dir: str | Path,
    alpha: float = 0.05,
    default_bonus: float = 0.75,
    l2: float = 1.0,
    device: str = "auto",
    progress_every: int = 1000,
) -> TrainBothSummary:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data_path = Path(data_dir)
    ro_model_dir = data_path / "ro_model"
    app_model_dir = data_path / "app_model"
    ro_train_path = data_path / "ro_train_samples.jsonl"
    app_train_path = data_path / "app_train_samples.jsonl"
    ro_rows = sum(1 for line in ro_train_path.open() if line.strip())
    app_rows = sum(1 for line in app_train_path.open() if line.strip())
    total_start = time.time()
    logger.info("[WORKFLOW] Starting joint training | data_dir=%s | device=%s", data_path, device)
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
        device=device,
        progress_every=progress_every,
        progress_label="RO",
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
        device=device,
        progress_every=progress_every,
        progress_label="APP",
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
        "device": summary.device,
        "ro": asdict(summary.ro),
        "app": asdict(summary.app),
    })
    return summary


def _parse_v6_relevance_catalog(markdown_path: str | Path) -> dict[str, dict[str, dict[str, tuple[str, ...]]]]:
    text = Path(markdown_path).read_text()
    if "## Scenario Defaults" not in text:
        raise ValueError(f"Could not find '## Scenario Defaults' in {markdown_path}")
    section = text.split("## Scenario Defaults", 1)[1]
    parsed: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {"ro": {}, "app": {}}

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line.startswith("| `"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 8:
            continue
        scenario_id = parts[0].strip("`")
        ro_most = tuple(re.findall(r"`([^`]+)`", parts[2]))
        ro_plausible = tuple(re.findall(r"`([^`]+)`", parts[3]))
        ro_irrelevant = tuple(re.findall(r"`([^`]+)`", parts[4]))
        app_most = tuple(re.findall(r"`([^`]+)`", parts[5]))
        app_plausible = tuple(re.findall(r"`([^`]+)`", parts[6]))
        app_irrelevant = tuple(re.findall(r"`([^`]+)`", parts[7]))
        if len(ro_most) < 3 or len(ro_plausible) < 3 or len(ro_irrelevant) < 2:
            raise ValueError(f"Scenario {scenario_id!r} does not satisfy the required R/O 3+3+2 structure")
        if len(app_most) < 3 or len(app_plausible) < 3 or len(app_irrelevant) < 2:
            raise ValueError(f"Scenario {scenario_id!r} does not satisfy the required App 3+3+2 structure")
        parsed["ro"][scenario_id] = {
            "most_relevant_3": ro_most[:3],
            "other_plausible_3": ro_plausible[:3],
            "irrelevant_2": ro_irrelevant[:2],
        }
        parsed["app"][scenario_id] = {
            "most_relevant_3": app_most[:3],
            "other_plausible_3": app_plausible[:3],
            "irrelevant_2": app_irrelevant[:2],
        }

    if not parsed["ro"] or not parsed["app"]:
        raise ValueError(f"No scenario relevance rows were parsed from {markdown_path}")
    return parsed


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
    relevance_catalog = _parse_v6_relevance_catalog(catalog_markdown)[label_namespace]
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
