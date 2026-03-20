"""Offline replay training for the V0 masked LinUCB model."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from recommendation_agents.feature_space import V0FeatureSpace
from recommendation_agents.linucb import MaskedDisjointLinUCB, RankedAction
from recommendation_agents.metadata import BanditMetadata
from recommendation_agents.schemas import ScoreRequest, TrainingEvent


@dataclass(frozen=True)
class TrainingMetrics:
    sample_count: int
    mean_reward: float
    pre_update_policy_hit_rate: float
    default_logged_action_rate: float
    unique_scenarios_seen: int


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
    scenario = metadata.scenario(event.scenario_id)
    if event.shown_actions is None:
        return scenario.action_ids
    try:
        return tuple(scenario.resolve_action_token(action_id) for action_id in event.shown_actions)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Event {event.event_id or '<unknown>'} includes invalid shown_actions "
            f"for scenario {event.scenario_id!r}: {exc}"
        ) from exc


def train_v0(
    metadata_path: str | Path,
    samples_path: str | Path,
    output_dir: str | Path,
    alpha: float = 0.15,
    default_bonus: float = 0.75,
    l2: float = 1.0,
    device: str = "auto",
) -> TrainingMetrics:
    metadata = BanditMetadata.load(metadata_path)
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

    for payload in load_jsonl(samples_path):
        event = TrainingEvent.from_dict(payload)
        scenario = metadata.scenario(event.scenario_id)
        candidate_actions = resolve_candidate_actions(event, metadata)
        selected_action = scenario.resolve_action_token(event.selected_action)
        if selected_action not in candidate_actions:
            raise ValueError(
                f"selected_action {event.selected_action!r} is not in shown_actions "
                f"for event {event.event_id or '<unknown>'}"
            )

        x = feature_space.encode(event.scenario_id, event.context)
        ranking = model.rank(x, candidate_actions, scenario.default_arm_id, top_k=1)
        if ranking[0].action_id == selected_action:
            pre_update_hits += 1

        if selected_action == scenario.default_arm_id:
            logged_defaults += 1

        model.partial_fit(x, selected_action, event.reward)
        total_reward += event.reward
        sample_count += 1
        scenarios_seen[event.scenario_id] += 1

    if sample_count == 0:
        raise ValueError("No training samples were loaded")

    model.save(output_dir)
    report = {
        "metrics": asdict(
            TrainingMetrics(
                sample_count=sample_count,
                mean_reward=total_reward / sample_count,
                pre_update_policy_hit_rate=pre_update_hits / sample_count,
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
        },
    }
    Path(output_dir, "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    Path(output_dir, "metadata.snapshot.json").write_text(Path(metadata_path).read_text())
    return TrainingMetrics(**report["metrics"])


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
    scenario = metadata.scenario(request.scenario_id)
    candidate_actions = (
        tuple(scenario.resolve_action_token(action_id) for action_id in request.shown_actions)
        if request.shown_actions
        else scenario.action_ids
    )
    x = feature_space.encode(request.scenario_id, request.context)
    return model.rank(x, candidate_actions, scenario.default_arm_id, top_k=top_k)


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
    scenario = metadata.scenario(request.scenario_id)
    candidate_actions = (
        tuple(scenario.resolve_action_token(action_id) for action_id in request.shown_actions)
        if request.shown_actions
        else scenario.action_ids
    )
    x = feature_space.encode(request.scenario_id, request.context)
    return model.choose(x, candidate_actions, scenario.default_arm_id, epsilon=epsilon, seed=seed)
