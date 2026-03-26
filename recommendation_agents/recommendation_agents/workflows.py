"""High-level one-line workflows for V0 data prep, training, and evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any

from recommendation_agents.catalog import (
    AppCatalogSummary,
    CatalogSummary,
    build_app_metadata_from_catalog_markdown,
    build_ro_metadata_from_catalog_markdown,
)
from recommendation_agents.metadata import BanditMetadata
from recommendation_agents.raw_synthetic import convert_raw_sequence_to_v0, convert_raw_sequence_to_v0_app
from recommendation_agents.schemas import ScoreRequest, TrainingEvent
from recommendation_agents.trainer import TrainingMetrics, _format_duration, score_v0, train_v0


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
    row_hit_at_k: float
    pred_topk_rate_by_action: dict[str, float]
    label_rate_by_action: dict[str, float]
    scenario_topk_majority_set_match_rate: float
    scenario_topk_examples: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class EvalBothSummary:
    data_dir: str
    top_k: int
    device: str
    ro: EvalTopKSummary
    app: EvalTopKSummary


def _write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


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


def evaluate_v0_topk(
    artifact_dir: str | Path,
    metadata_path: str | Path,
    test_samples_path: str | Path,
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
    metadata = BanditMetadata.load(metadata_path)
    metadata_payload = json.loads(Path(metadata_path).read_text())
    scenario_gt = {
        str(row["scenario_id"]): tuple(str(value) for value in row.get("default_action_ids", [])[:top_k])
        for row in metadata_payload.get("scenario_default_rankings", [])
    }
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

    row_hit = 0
    pred_topk_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    scenario_preds: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    interval_hit = 0

    for index, event in enumerate(rows, start=1):
        ranked = score_v0(
            artifact_dir=artifact_dir,
            metadata_path=metadata_path,
            request=ScoreRequest(context=event.context),
            top_k=top_k,
            device=device,
        )
        pred = tuple(item.action_id for item in ranked)
        label = metadata.resolve_action_token(event.selected_action, event.scenario_id)
        if label in pred:
            row_hit += 1
            interval_hit += 1
        for action_id in pred:
            pred_topk_counts[action_id] += 1
        label_counts[label] += 1
        if event.scenario_id is not None and event.scenario_id in scenario_gt:
            scenario_preds[event.scenario_id].append(pred)

        if index % progress_every == 0:
            elapsed = time.time() - start_time
            rows_per_sec = index / elapsed if elapsed > 0 else 0.0
            remaining = max(total_rows - index, 0)
            eta_seconds = remaining / rows_per_sec if rows_per_sec > 0 else 0.0
            logger.info(
                "%sProcessed %d/%d rows | interval hit@%d=%.4f | cumulative hit@%d=%.4f | elapsed=%s | eta=%s | %.1f rows/s",
                label_prefix,
                index,
                total_rows,
                top_k,
                interval_hit / progress_every,
                top_k,
                row_hit / index,
                _format_duration(elapsed),
                _format_duration(eta_seconds),
                rows_per_sec,
            )
            interval_hit = 0

    remainder = total_rows % progress_every
    if remainder:
        elapsed = time.time() - start_time
        rows_per_sec = total_rows / elapsed if elapsed > 0 else 0.0
        logger.info(
            "%sProcessed %d/%d rows | interval hit@%d=%.4f | cumulative hit@%d=%.4f | elapsed=%s | eta=0s | %.1f rows/s",
            label_prefix,
            total_rows,
            total_rows,
            top_k,
            interval_hit / remainder,
            top_k,
            row_hit / total_rows,
            _format_duration(elapsed),
            rows_per_sec,
        )

    scenario_eval: dict[str, dict[str, Any]] = {}
    for scenario_id, pred_lists in sorted(scenario_preds.items()):
        pred_majority = Counter(pred_lists).most_common(1)[0][0]
        gt_topk = tuple(metadata.resolve_action_token(action_id, scenario_id) for action_id in scenario_gt[scenario_id])
        scenario_eval[scenario_id] = {
            "pred_topk": list(pred_majority),
            "gt_topk": list(gt_topk),
            "set_match": set(pred_majority) == set(gt_topk),
        }

    elapsed_total = time.time() - start_time
    logger.info(
        "%sFinished evaluation | rows=%d | hit@%d=%.4f | elapsed=%s | avg %.1f rows/s",
        label_prefix,
        total_rows,
        top_k,
        row_hit / total_rows,
        _format_duration(elapsed_total),
        total_rows / elapsed_total if elapsed_total > 0 else 0.0,
    )

    return EvalTopKSummary(
        sample_count=total_rows,
        top_k=top_k,
        row_hit_at_k=row_hit / total_rows,
        pred_topk_rate_by_action={action_id: count / total_rows for action_id, count in pred_topk_counts.most_common()},
        label_rate_by_action={action_id: count / total_rows for action_id, count in label_counts.most_common()},
        scenario_topk_majority_set_match_rate=(
            sum(1 for item in scenario_eval.values() if item["set_match"]) / len(scenario_eval)
            if scenario_eval else 0.0
        ),
        scenario_topk_examples=dict(list(scenario_eval.items())[:10]),
    )


def eval_v0_both(
    data_dir: str | Path,
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
        top_k=top_k,
        device=device,
        progress_every=progress_every,
        progress_label="RO-EVAL",
    )
    app = evaluate_v0_topk(
        artifact_dir=data_path / "app_model",
        metadata_path=data_path / "app_metadata.json",
        test_samples_path=data_path / "app_test_samples.jsonl",
        top_k=top_k,
        device=device,
        progress_every=progress_every,
        progress_label="APP-EVAL",
    )
    logger.info("[WORKFLOW] Finished joint evaluation")
    summary = EvalBothSummary(
        data_dir=str(data_path),
        top_k=top_k,
        device=device,
        ro=ro,
        app=app,
    )
    _write_json(data_path / f"eval_both_top{top_k}.json", {
        "data_dir": summary.data_dir,
        "top_k": summary.top_k,
        "device": summary.device,
        "ro": asdict(summary.ro),
        "app": asdict(summary.app),
    })
    return summary
