"""Offline replay training for the V0 shared-action LinUCB model."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np

from recommendation_agents.feature_space import V0FeatureSpace
from recommendation_agents.linucb import MaskedDisjointLinUCB, RankedAction
from recommendation_agents.metadata import BanditMetadata
from recommendation_agents.raw_synthetic import (
    _build_context,
    _event_id,
    _load_jsonl,
    _raw_label,
    _scenario_id,
    convert_raw_sequence_to_v0,
    convert_raw_sequence_to_v0_app,
)
from recommendation_agents.schemas import ScoreRequest, TrainingEvent


logger = logging.getLogger(__name__)


def _count_nonempty_lines(path: str | Path) -> int:
    with Path(path).open() as handle:
        return sum(1 for line in handle if line.strip())


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


@dataclass(frozen=True)
class TrainingMetrics:
    sample_count: int
    mean_reward: float
    pre_update_policy_hit_rate: float
    default_logged_action_rate: float
    unique_scenarios_seen: int


@dataclass(frozen=True)
class RawTrainingSummary:
    input_path: str
    metadata_path: str
    samples_path: str
    label_type: str
    reward: float
    conversion: dict[str, int]
    training: TrainingMetrics


@dataclass(frozen=True)
class DualRawTrainingSummary:
    input_path: str
    output_dir: str
    tensorboard_logdir: str
    reward: float
    alpha_start: float
    alpha_end: float
    train_window: int
    eval_window: int
    shuffle_seed: int
    ro: "DualHeadArtifactSummary"
    app: "DualHeadArtifactSummary"


@dataclass(frozen=True)
class DualPreparedSample:
    event_id: str
    scenario_id: str | None
    context: dict[str, Any]
    x: Any
    reward: float
    ro_action: str
    app_action: str


@dataclass(frozen=True)
class InterleavedTrainingMetrics:
    train_count: int
    eval_count: int
    mean_train_reward: float
    train_pre_update_hit_rate: float
    train_default_action_rate: float
    eval_top1_rate: float
    unique_scenarios_seen: int


@dataclass(frozen=True)
class DualHeadArtifactSummary:
    artifact_dir: str
    metadata_path: str
    metrics: InterleavedTrainingMetrics


def load_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc


def resolve_candidate_actions(event: TrainingEvent, metadata: BanditMetadata) -> tuple[str, ...]:
    if event.shown_actions is None:
        return metadata.candidate_action_ids(event.scenario_id)
    try:
        return tuple(metadata.resolve_action_token(action_id, event.scenario_id) for action_id in event.shown_actions)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Event {event.event_id or '<unknown>'} includes invalid shown_actions: {exc}"
        ) from exc


def _configure_progress_logger() -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _resolve_alpha(alpha_start: float, alpha_end: float, train_step: int, total_train_steps: int) -> float:
    if total_train_steps <= 1:
        return float(alpha_end)
    clamped_step = min(max(train_step, 1), total_train_steps)
    progress = (clamped_step - 1) / (total_train_steps - 1)
    return float(alpha_start + (alpha_end - alpha_start) * progress)


def _log_train_progress(
    *,
    label_prefix: str,
    processed_samples: int,
    total_samples_expected: int,
    interval_reward: float,
    interval_hits: int,
    interval_size: int,
    total_reward: float,
    pre_update_hits: int,
    start_time: float,
    track_train_hit_rate: bool,
) -> None:
    elapsed = time.time() - start_time
    samples_per_sec = processed_samples / elapsed if elapsed > 0 else 0.0
    remaining = max(total_samples_expected - processed_samples, 0)
    eta_seconds = remaining / samples_per_sec if samples_per_sec > 0 else 0.0
    if track_train_hit_rate:
        logger.info(
            "%sProcessed %d/%d samples | interval avg reward=%.4f | interval top1=%.4f | "
            "cumulative avg reward=%.4f | cumulative top1=%.4f | elapsed=%s | eta=%s | %.1f samples/s",
            label_prefix,
            processed_samples,
            total_samples_expected,
            interval_reward / interval_size,
            interval_hits / interval_size,
            total_reward / processed_samples,
            pre_update_hits / processed_samples,
            _format_duration(elapsed),
            _format_duration(eta_seconds),
            samples_per_sec,
        )
        return
    logger.info(
        "%sProcessed %d/%d samples | interval avg reward=%.4f | cumulative avg reward=%.4f | "
        "elapsed=%s | eta=%s | %.1f samples/s",
        label_prefix,
        processed_samples,
        total_samples_expected,
        interval_reward / interval_size,
        total_reward / processed_samples,
        _format_duration(elapsed),
        _format_duration(eta_seconds),
        samples_per_sec,
    )


def _prepare_dual_samples(
    input_path: str | Path,
    reward: float,
    feature_space: V0FeatureSpace,
) -> list[DualPreparedSample]:
    prepared: list[DualPreparedSample] = []
    for row in _load_jsonl(input_path):
        ro_action = _raw_label(row, "gt_ro_action_id", "gt_ro")
        app_action = _raw_label(row, "gt_app_category", "gt_app")
        if ro_action == "NONE" or app_action == "NONE":
            continue
        scenario_id = _scenario_id(row)
        context = _build_context(row)
        prepared.append(
            DualPreparedSample(
                event_id=_event_id(row, scenario_id),
                scenario_id=scenario_id,
                context=context,
                x=feature_space.encode(context),
                reward=float(reward),
                ro_action=ro_action,
                app_action=app_action,
            )
        )
    if not prepared:
        raise ValueError("No trainable rows found with both gt_ro and gt_app labels")
    return prepared


def _count_interleaved_splits(total_rows: int, train_window: int, eval_window: int) -> tuple[int, int]:
    cycle = train_window + eval_window
    train_rows = 0
    eval_rows = 0
    for index in range(total_rows):
        if index % cycle < train_window:
            train_rows += 1
        else:
            eval_rows += 1
    return train_rows, eval_rows


def _resolve_rank_result(
    model: MaskedDisjointLinUCB,
    metadata: BanditMetadata,
    scenario_id: str | None,
    x: Any,
    selected_token: str,
) -> tuple[str, bool, bool]:
    selected_action = metadata.resolve_action_token(selected_token, scenario_id)
    candidate_actions = metadata.candidate_action_ids(scenario_id)
    default_action_id = metadata.default_action_id(scenario_id)
    ranking = model.rank(x, candidate_actions, default_action_id, top_k=1)
    return selected_action, ranking[0].action_id == selected_action, selected_action == default_action_id


def _write_interleaved_artifact(
    output_dir: str | Path,
    metadata_path: str | Path,
    model: MaskedDisjointLinUCB,
    metrics: InterleavedTrainingMetrics,
    scenario_counts: Counter[str],
    hyperparameters: dict[str, Any],
    feature_dim: int,
) -> None:
    model.save(output_dir)
    report = {
        "metrics": asdict(metrics),
        "feature_dim": feature_dim,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "hyperparameters": hyperparameters,
    }
    Path(output_dir, "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    Path(output_dir, "metadata.snapshot.json").write_text(Path(metadata_path).read_text())


def _create_tensorboard_writer(log_dir: str | Path):
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[reportMissingImports]

    return SummaryWriter(log_dir=str(log_dir))


def train_v0(
    metadata_path: str | Path,
    samples_path: str | Path,
    output_dir: str | Path,
    alpha: float = 0.15,
    default_bonus: float = 0.75,
    l2: float = 1.0,
    device: str = "auto",
    progress_every: int = 1000,
    progress_label: str | None = None,
    track_train_hit_rate: bool = False,
    epochs: int = 1,
) -> TrainingMetrics:
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    _configure_progress_logger()
    label_prefix = f"[{progress_label}] " if progress_label else ""
    metadata = BanditMetadata.load(metadata_path)
    samples_per_epoch = _count_nonempty_lines(samples_path)
    total_samples_expected = samples_per_epoch * epochs
    start_time = time.time()
    logger.info(
        "%sStarting training | samples_per_epoch=%d | epochs=%d | total_samples=%d | actions=%d | feature_dim=%d | device=%s",
        label_prefix,
        samples_per_epoch,
        epochs,
        total_samples_expected,
        len(metadata.all_action_ids()),
        V0FeatureSpace().dimension,
        device,
    )
    feature_space = V0FeatureSpace()
    model = MaskedDisjointLinUCB(
        action_ids=metadata.all_action_ids(),
        feature_dim=feature_space.dimension,
        alpha=alpha,
        default_bonus=default_bonus,
        l2=l2,
        device=device,
    )

    total_reward = 0.0
    sample_count = 0
    pre_update_hits = 0
    logged_defaults = 0
    scenarios_seen: Counter[str] = Counter()
    interval_reward = 0.0
    interval_hits = 0
    interval_defaults = 0

    for epoch_index in range(epochs):
        logger.info("%sStarting epoch %d/%d", label_prefix, epoch_index + 1, epochs)
        for payload in load_jsonl(samples_path):
            sample_count += 1
            event = TrainingEvent.from_dict(payload)
            candidate_actions = resolve_candidate_actions(event, metadata)
            selected_action = metadata.resolve_action_token(event.selected_action, event.scenario_id)
            if selected_action not in candidate_actions:
                raise ValueError(
                    f"selected_action {event.selected_action!r} is not in shown_actions "
                    f"for event {event.event_id or '<unknown>'}"
                )

            default_action_id = metadata.default_action_id(event.scenario_id)
            if event.context_vector is not None:
                x = np.asarray(event.context_vector, dtype=np.float32)
                if x.shape != (feature_space.dimension,):
                    raise ValueError(
                        f"context_vector for event {event.event_id or '<unknown>'} has shape {x.shape}; "
                        f"expected ({feature_space.dimension},)"
                    )
            else:
                x = feature_space.encode(event.context)
            if track_train_hit_rate:
                ranking = model.rank(x, candidate_actions, default_action_id, top_k=1)
                if ranking[0].action_id == selected_action:
                    pre_update_hits += 1
                    interval_hits += 1

            if default_action_id is not None and selected_action == default_action_id:
                logged_defaults += 1
                interval_defaults += 1

            model.partial_fit(x, selected_action, event.reward)
            total_reward += event.reward
            interval_reward += event.reward
            if event.scenario_id is not None:
                scenarios_seen[event.scenario_id] += 1

            if sample_count % progress_every == 0:
                _log_train_progress(
                    label_prefix=label_prefix,
                    processed_samples=sample_count,
                    total_samples_expected=total_samples_expected,
                    interval_reward=interval_reward,
                    interval_hits=interval_hits,
                    interval_size=progress_every,
                    total_reward=total_reward,
                    pre_update_hits=pre_update_hits,
                    start_time=start_time,
                    track_train_hit_rate=track_train_hit_rate,
                )
                interval_reward = 0.0
                interval_hits = 0
                interval_defaults = 0

    if sample_count == 0:
        raise ValueError("No training samples were loaded")

    remainder = sample_count % progress_every
    if remainder:
        _log_train_progress(
            label_prefix=label_prefix,
            processed_samples=sample_count,
            total_samples_expected=total_samples_expected,
            interval_reward=interval_reward,
            interval_hits=interval_hits,
            interval_size=remainder,
            total_reward=total_reward,
            pre_update_hits=pre_update_hits,
            start_time=start_time,
            track_train_hit_rate=track_train_hit_rate,
        )

    model.save(output_dir)
    report = {
        "metrics": asdict(
            TrainingMetrics(
                sample_count=sample_count,
                mean_reward=total_reward / sample_count,
                pre_update_policy_hit_rate=(pre_update_hits / sample_count) if track_train_hit_rate else 0.0,
                default_logged_action_rate=logged_defaults / sample_count,
                unique_scenarios_seen=len(scenarios_seen),
            )
        ),
        "feature_dim": feature_space.dimension,
        "scenario_counts": dict(sorted(scenarios_seen.items())),
        "hyperparameters": {
            "alpha": alpha,
            "default_bonus": default_bonus,
            "l2": l2,
            "device": model.device,
            "track_train_hit_rate": track_train_hit_rate,
            "epochs": epochs,
        },
    }
    elapsed_total = time.time() - start_time
    logger.info(
        "%sFinished training | samples=%d | elapsed=%s | avg %.1f samples/s",
        label_prefix,
        sample_count,
        _format_duration(elapsed_total),
        sample_count / elapsed_total if elapsed_total > 0 else 0.0,
    )
    Path(output_dir, "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    Path(output_dir, "metadata.snapshot.json").write_text(Path(metadata_path).read_text())
    return TrainingMetrics(**report["metrics"])


def train_v0_from_raw(
    input_path: str | Path,
    output_dir: str | Path,
    metadata_path: str | Path | None = None,
    reward: float = 1.0,
    label_type: str = "ro",
    alpha: float = 0.15,
    default_bonus: float = 0.75,
    l2: float = 1.0,
    device: str = "auto",
    progress_every: int = 1000,
    progress_label: str | None = None,
    track_train_hit_rate: bool = False,
    epochs: int = 1,
) -> RawTrainingSummary:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples_path = output_path / f"train_{label_type}_samples.jsonl"
    resolved_metadata_path = Path(metadata_path) if metadata_path is not None else output_path / f"train_{label_type}_metadata.inferred.json"

    if label_type == "ro":
        conversion = convert_raw_sequence_to_v0(
            input_path=input_path,
            output_samples_path=samples_path,
            output_metadata_path=None if metadata_path is not None else resolved_metadata_path,
            reward=reward,
        )
    elif label_type == "app":
        conversion = convert_raw_sequence_to_v0_app(
            input_path=input_path,
            output_samples_path=samples_path,
            output_metadata_path=None if metadata_path is not None else resolved_metadata_path,
            reward=reward,
        )
    else:
        raise ValueError("label_type must be 'ro' or 'app'")

    metrics = train_v0(
        metadata_path=resolved_metadata_path,
        samples_path=samples_path,
        output_dir=output_path,
        alpha=alpha,
        default_bonus=default_bonus,
        l2=l2,
        device=device,
        progress_every=progress_every,
        progress_label=progress_label or label_type.upper(),
        track_train_hit_rate=track_train_hit_rate,
        epochs=epochs,
    )
    summary = RawTrainingSummary(
        input_path=str(Path(input_path)),
        metadata_path=str(resolved_metadata_path),
        samples_path=str(samples_path),
        label_type=label_type,
        reward=float(reward),
        conversion=asdict(conversion),
        training=metrics,
    )
    (output_path / "raw_training_summary.json").write_text(
        json.dumps(
            {
                "input_path": summary.input_path,
                "metadata_path": summary.metadata_path,
                "samples_path": summary.samples_path,
                "label_type": summary.label_type,
                "reward": summary.reward,
                "conversion": summary.conversion,
                "training": asdict(summary.training),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def train_v0_dual_from_raw(
    input_path: str | Path,
    output_dir: str | Path,
    ro_metadata_path: str | Path | None = None,
    app_metadata_path: str | Path | None = None,
    reward: float = 1.0,
    alpha: float = 0.15,
    alpha_end: float = 0.01,
    default_bonus: float = 0.75,
    l2: float = 1.0,
    device: str = "auto",
    progress_every: int = 1000,
    train_window: int = 100,
    eval_window: int = 10,
    shuffle_seed: int = 0,
    tensorboard_logdir: str | Path | None = None,
) -> DualRawTrainingSummary:
    if ro_metadata_path is None or app_metadata_path is None:
        raise ValueError("train_v0_dual_from_raw requires both ro_metadata_path and app_metadata_path")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if train_window <= 0 or eval_window <= 0:
        raise ValueError("train_window and eval_window must be positive")
    if alpha_end < 0.0:
        raise ValueError("alpha_end must be non-negative")

    _configure_progress_logger()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ro_output_dir = output_path / "ro_model"
    app_output_dir = output_path / "app_model"
    resolved_tensorboard_logdir = Path(tensorboard_logdir) if tensorboard_logdir is not None else output_path / "tensorboard"
    writer = _create_tensorboard_writer(resolved_tensorboard_logdir)

    feature_space = V0FeatureSpace()
    samples = _prepare_dual_samples(input_path, reward, feature_space)
    rng = random.Random(shuffle_seed)
    rng.shuffle(samples)

    train_rows, eval_rows = _count_interleaved_splits(len(samples), train_window, eval_window)
    ro_metadata = BanditMetadata.load(ro_metadata_path)
    app_metadata = BanditMetadata.load(app_metadata_path)
    ro_model = MaskedDisjointLinUCB(
        action_ids=ro_metadata.all_action_ids(),
        feature_dim=feature_space.dimension,
        alpha=alpha,
        default_bonus=default_bonus,
        l2=l2,
        device=device,
    )
    app_model = MaskedDisjointLinUCB(
        action_ids=app_metadata.all_action_ids(),
        feature_dim=feature_space.dimension,
        alpha=alpha,
        default_bonus=default_bonus,
        l2=l2,
        device=device,
    )

    ro_scenarios_seen: Counter[str] = Counter()
    app_scenarios_seen: Counter[str] = Counter()
    train_processed = 0
    eval_processed = 0
    ro_train_hits = 0
    app_train_hits = 0
    ro_train_defaults = 0
    app_train_defaults = 0
    ro_eval_hits = 0
    app_eval_hits = 0
    ro_reward_total = 0.0
    app_reward_total = 0.0
    ro_block_eval_hits = 0
    app_block_eval_hits = 0
    ro_block_eval_reward = 0.0
    app_block_eval_reward = 0.0
    ro_eval_reward_total = 0.0
    app_eval_reward_total = 0.0
    ro_block_train_hits = 0
    app_block_train_hits = 0
    ro_block_train_reward = 0.0
    app_block_train_reward = 0.0
    train_in_block = 0
    eval_in_block = 0
    cycle = train_window + eval_window

    try:
        for index, sample in enumerate(samples):
            in_train = index % cycle < train_window
            current_alpha = _resolve_alpha(
                alpha,
                alpha_end,
                train_processed + 1 if in_train else max(train_processed, 1),
                max(train_rows, 1),
            )
            ro_model.alpha = current_alpha
            app_model.alpha = current_alpha

            ro_selected, ro_hit, ro_is_default = _resolve_rank_result(
                ro_model, ro_metadata, sample.scenario_id, sample.x, sample.ro_action
            )
            app_selected, app_hit, app_is_default = _resolve_rank_result(
                app_model, app_metadata, sample.scenario_id, sample.x, sample.app_action
            )

            if sample.scenario_id is not None:
                ro_scenarios_seen[sample.scenario_id] += 1
                app_scenarios_seen[sample.scenario_id] += 1

            if in_train:
                train_processed += 1
                train_in_block += 1
                ro_train_hits += int(ro_hit)
                app_train_hits += int(app_hit)
                ro_block_train_hits += int(ro_hit)
                app_block_train_hits += int(app_hit)
                ro_train_defaults += int(ro_is_default)
                app_train_defaults += int(app_is_default)
                ro_reward_total += sample.reward
                app_reward_total += sample.reward
                ro_block_train_reward += sample.reward
                app_block_train_reward += sample.reward
                ro_model.partial_fit(sample.x, ro_selected, sample.reward)
                app_model.partial_fit(sample.x, app_selected, sample.reward)
                continue

            eval_processed += 1
            eval_in_block += 1
            ro_eval_hits += int(ro_hit)
            app_eval_hits += int(app_hit)
            ro_block_eval_hits += int(ro_hit)
            app_block_eval_hits += int(app_hit)
            ro_eval_reward_total += sample.reward if ro_hit else 0.0
            app_eval_reward_total += sample.reward if app_hit else 0.0
            ro_block_eval_reward += sample.reward if ro_hit else 0.0
            app_block_eval_reward += sample.reward if app_hit else 0.0

            block_complete = eval_in_block == eval_window or index == len(samples) - 1
            should_log = block_complete and (train_processed % progress_every == 0 or index == len(samples) - 1)
            if should_log:
                logger.info(
                    "[RO] train=%d eval=%d block_eval_top1=%.4f cumulative_eval_top1=%.4f "
                    "train_hit=%.4f alpha=%.4f",
                    train_processed,
                    eval_processed,
                    ro_block_eval_hits / eval_in_block,
                    ro_eval_hits / eval_processed,
                    ro_train_hits / train_processed if train_processed else 0.0,
                    current_alpha,
                )
                logger.info(
                    "[APP] train=%d eval=%d block_eval_top1=%.4f cumulative_eval_top1=%.4f "
                    "train_hit=%.4f alpha=%.4f",
                    train_processed,
                    eval_processed,
                    app_block_eval_hits / eval_in_block,
                    app_eval_hits / eval_processed,
                    app_train_hits / train_processed if train_processed else 0.0,
                    current_alpha,
                )
                writer.add_scalar("shared/alpha", current_alpha, train_processed)
                writer.add_scalar("ro/train/block_accuracy", ro_block_train_hits / train_in_block, train_processed)
                writer.add_scalar("ro/train/block_reward", ro_block_train_reward / train_in_block, train_processed)
                writer.add_scalar("ro/train/cumulative_accuracy", ro_train_hits / train_processed, train_processed)
                writer.add_scalar("ro/train/cumulative_reward", ro_reward_total / train_processed, train_processed)
                writer.add_scalar("ro/eval/block_accuracy", ro_block_eval_hits / eval_in_block, train_processed)
                writer.add_scalar("ro/eval/block_reward", ro_block_eval_reward / eval_in_block, train_processed)
                writer.add_scalar("ro/eval/cumulative_accuracy", ro_eval_hits / eval_processed, train_processed)
                writer.add_scalar("ro/eval/cumulative_reward", ro_eval_reward_total / eval_processed, train_processed)
                writer.add_scalar("app/train/block_accuracy", app_block_train_hits / train_in_block, train_processed)
                writer.add_scalar("app/train/block_reward", app_block_train_reward / train_in_block, train_processed)
                writer.add_scalar("app/train/cumulative_accuracy", app_train_hits / train_processed, train_processed)
                writer.add_scalar("app/train/cumulative_reward", app_reward_total / train_processed, train_processed)
                writer.add_scalar("app/eval/block_accuracy", app_block_eval_hits / eval_in_block, train_processed)
                writer.add_scalar("app/eval/block_reward", app_block_eval_reward / eval_in_block, train_processed)
                writer.add_scalar("app/eval/cumulative_accuracy", app_eval_hits / eval_processed, train_processed)
                writer.add_scalar("app/eval/cumulative_reward", app_eval_reward_total / eval_processed, train_processed)
            if block_complete:
                ro_block_eval_hits = 0
                app_block_eval_hits = 0
                ro_block_eval_reward = 0.0
                app_block_eval_reward = 0.0
                ro_block_train_hits = 0
                app_block_train_hits = 0
                ro_block_train_reward = 0.0
                app_block_train_reward = 0.0
                train_in_block = 0
                eval_in_block = 0
    finally:
        writer.flush()
        writer.close()

    ro_model.alpha = alpha_end
    app_model.alpha = alpha_end
    ro_metrics = InterleavedTrainingMetrics(
        train_count=train_processed,
        eval_count=eval_processed,
        mean_train_reward=ro_reward_total / train_processed if train_processed else 0.0,
        train_pre_update_hit_rate=ro_train_hits / train_processed if train_processed else 0.0,
        train_default_action_rate=ro_train_defaults / train_processed if train_processed else 0.0,
        eval_top1_rate=ro_eval_hits / eval_processed if eval_processed else 0.0,
        unique_scenarios_seen=len(ro_scenarios_seen),
    )
    app_metrics = InterleavedTrainingMetrics(
        train_count=train_processed,
        eval_count=eval_processed,
        mean_train_reward=app_reward_total / train_processed if train_processed else 0.0,
        train_pre_update_hit_rate=app_train_hits / train_processed if train_processed else 0.0,
        train_default_action_rate=app_train_defaults / train_processed if train_processed else 0.0,
        eval_top1_rate=app_eval_hits / eval_processed if eval_processed else 0.0,
        unique_scenarios_seen=len(app_scenarios_seen),
    )

    hyperparameters = {
        "alpha_start": alpha,
        "alpha_end": alpha_end,
        "default_bonus": default_bonus,
        "l2": l2,
        "device": ro_model.device,
        "reward": reward,
        "shuffle_seed": shuffle_seed,
        "train_window": train_window,
        "eval_window": eval_window,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
    }
    _write_interleaved_artifact(
        output_dir=ro_output_dir,
        metadata_path=ro_metadata_path,
        model=ro_model,
        metrics=ro_metrics,
        scenario_counts=ro_scenarios_seen,
        hyperparameters=hyperparameters,
        feature_dim=feature_space.dimension,
    )
    _write_interleaved_artifact(
        output_dir=app_output_dir,
        metadata_path=app_metadata_path,
        model=app_model,
        metrics=app_metrics,
        scenario_counts=app_scenarios_seen,
        hyperparameters=hyperparameters,
        feature_dim=feature_space.dimension,
    )

    ro_summary = DualHeadArtifactSummary(
        artifact_dir=str(ro_output_dir),
        metadata_path=str(Path(ro_metadata_path)),
        metrics=ro_metrics,
    )
    app_summary = DualHeadArtifactSummary(
        artifact_dir=str(app_output_dir),
        metadata_path=str(Path(app_metadata_path)),
        metrics=app_metrics,
    )

    summary = DualRawTrainingSummary(
        input_path=str(Path(input_path)),
        output_dir=str(output_path),
        tensorboard_logdir=str(resolved_tensorboard_logdir),
        reward=float(reward),
        alpha_start=float(alpha),
        alpha_end=float(alpha_end),
        train_window=train_window,
        eval_window=eval_window,
        shuffle_seed=shuffle_seed,
        ro=ro_summary,
        app=app_summary,
    )
    (output_path / "dual_training_summary.json").write_text(
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
                    "metrics": asdict(summary.ro.metrics),
                },
                "app": {
                    "metadata_path": summary.app.metadata_path,
                    "artifact_dir": summary.app.artifact_dir,
                    "metrics": asdict(summary.app.metrics),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def score_v0(
    artifact_dir: str | Path,
    metadata_path: str | Path | None,
    request: ScoreRequest,
    top_k: int = 5,
    device: str = "auto",
) -> list[RankedAction]:
    resolved_metadata_path = Path(metadata_path) if metadata_path is not None else Path(artifact_dir) / "metadata.snapshot.json"
    metadata = BanditMetadata.load(resolved_metadata_path)
    feature_space = V0FeatureSpace()
    model = MaskedDisjointLinUCB.load(artifact_dir, device=device)
    candidate_actions = (
        tuple(metadata.resolve_action_token(action_id, request.scenario_id) for action_id in request.shown_actions)
        if request.shown_actions
        else metadata.candidate_action_ids(request.scenario_id)
    )
    x = feature_space.encode(request.context)
    return model.rank(x, candidate_actions, metadata.default_action_id(request.scenario_id), top_k=top_k)


def choose_v0(
    artifact_dir: str | Path,
    metadata_path: str | Path | None,
    request: ScoreRequest,
    epsilon: float = 0.0,
    seed: int | None = None,
    device: str = "auto",
) -> RankedAction:
    resolved_metadata_path = Path(metadata_path) if metadata_path is not None else Path(artifact_dir) / "metadata.snapshot.json"
    metadata = BanditMetadata.load(resolved_metadata_path)
    feature_space = V0FeatureSpace()
    model = MaskedDisjointLinUCB.load(artifact_dir, device=device)
    candidate_actions = (
        tuple(metadata.resolve_action_token(action_id, request.scenario_id) for action_id in request.shown_actions)
        if request.shown_actions
        else metadata.candidate_action_ids(request.scenario_id)
    )
    x = feature_space.encode(request.context)
    return model.choose(x, candidate_actions, metadata.default_action_id(request.scenario_id), epsilon=epsilon, seed=seed)
