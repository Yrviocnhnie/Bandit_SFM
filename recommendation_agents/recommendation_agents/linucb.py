"""Bandit model implementations for shared-action V0 ranking."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RankedAction:
    action_id: str
    score: float
    mean_reward: float
    uncertainty: float
    default_bonus: float


def _torch_is_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def _require_torch():
    if not _torch_is_installed():
        raise RuntimeError(
            "PyTorch is not installed. Install torch first to run training or scoring."
        )
    import torch

    return torch


def _resolve_device(device: str) -> str:
    resolved_device = device.lower()
    torch = _require_torch()
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    if resolved_device not in {"cpu"} and not resolved_device.startswith("cuda"):
        raise ValueError("device must be cpu, cuda, cuda:0, or auto")

    return resolved_device


def _tokenize_action_id(action_id: str) -> list[str]:
    return [piece for piece in action_id.lower().replace("::", "_").split("_") if piece]


def _build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
):
    torch = _require_torch()
    layers: list[object] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(torch.nn.Linear(prev_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        prev_dim = hidden_dim
    layers.append(torch.nn.Linear(prev_dim, output_dim))
    return torch.nn.Sequential(*layers)


def _augment_with_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.astype(np.float32, copy=False), np.ones((1,), dtype=np.float32)], axis=0)


def _flatten_grad_tuple(gradients) -> np.ndarray:
    flattened = [gradient.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False) for gradient in gradients]
    if not flattened:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(flattened, axis=0)


SHARED_ACTION_TYPES = (
    "show",
    "recommend",
    "start",
    "toggle",
    "open",
    "prompt",
    "app",
    "other",
)

SHARED_SEMANTIC_GROUPS = (
    "schedule",
    "todo_notes",
    "weather",
    "payment",
    "commute_travel",
    "booking_meeting",
    "nearby_places",
    "activity_fitness",
    "timer",
    "device_mode",
    "rest_recovery",
    "safety",
    "pickup_places",
    "app_category",
    "general",
)

SHARED_SCENE_AFFINITIES = (
    "work",
    "home",
    "commute",
    "travel",
    "meeting",
    "outdoor",
    "exercise",
    "meal_break",
    "night_quiet",
    "social_errand",
)

SHARED_CONTEXT_DERIVED_FEATURES = (
    "ctx_work",
    "ctx_home",
    "ctx_commute_travel",
    "ctx_meeting",
    "ctx_outdoor",
    "ctx_exercise",
    "ctx_meal_errand",
    "ctx_night_rest",
    "ctx_quiet_focus",
    "ctx_noisy_safety",
)

_SHARED_CONTEXT_INDEX_CACHE: dict[int, dict[str, object] | None] = {}


def _build_shared_context_index_cache(feature_dim: int) -> dict[str, object] | None:
    cached = _SHARED_CONTEXT_INDEX_CACHE.get(feature_dim)
    if cached is not None or feature_dim in _SHARED_CONTEXT_INDEX_CACHE:
        return cached

    from recommendation_agents.feature_space import V0FeatureSpace

    feature_space = V0FeatureSpace()
    if feature_space.dimension != feature_dim:
        _SHARED_CONTEXT_INDEX_CACHE[feature_dim] = None
        return None

    names = feature_space.feature_names()
    name_to_index = {name: index for index, name in enumerate(names)}

    def index(name: str) -> int:
        return name_to_index[name]

    def indices_with_prefix(prefix: str) -> tuple[int, ...]:
        return tuple(i for i, name in enumerate(names) if name.startswith(prefix))

    cache = {
        "work": (
            index("ps_location=work"),
            index("wifiLostCategory=work"),
            index("cal_nextLocation=work"),
            *indices_with_prefix("state_current=office_"),
            *indices_with_prefix("precondition=office_"),
        ),
        "home": (
            index("ps_location=home"),
            index("wifiLostCategory=home"),
            index("cal_nextLocation=home"),
            *indices_with_prefix("state_current=home_"),
            *indices_with_prefix("precondition=home_"),
        ),
        "commute_travel": (
            index("ps_motion=driving"),
            index("ps_motion=transit"),
            index("ps_location=metro"),
            index("ps_location=rail_station"),
            index("ps_location=airport"),
            index("ps_location=transit"),
            index("ps_location=en_route"),
            index("wifiLostCategory=metro"),
            index("wifiLostCategory=rail_station"),
            index("wifiLostCategory=airport"),
            index("wifiLostCategory=transit"),
            index("wifiLostCategory=en_route"),
            index("cal_nextLocation=metro"),
            index("cal_nextLocation=rail_station"),
            index("cal_nextLocation=airport"),
            index("cal_nextLocation=transit"),
            index("cal_nextLocation=en_route"),
            index("sms_train_pending"),
            index("sms_flight_pending"),
            index("sms_hotel_pending"),
            index("sms_ride_pending"),
            *indices_with_prefix("state_current=commuting_"),
            index("state_current=driving"),
            index("state_current=in_transit"),
            index("state_current=at_metro"),
            index("state_current=at_rail_station"),
            index("state_current=at_airport"),
            index("state_current=at_transit_hub"),
            *indices_with_prefix("precondition=commuting_"),
            index("precondition=driving"),
            index("precondition=in_transit"),
            index("precondition=at_metro"),
            index("precondition=at_rail_station"),
            index("precondition=at_airport"),
            index("precondition=at_transit_hub"),
        ),
        "meeting": (
            index("cal_hasUpcoming"),
            index("cal_inMeeting"),
            *indices_with_prefix("state_current=office_"),
        ),
        "outdoor": (
            index("ps_location=outdoor"),
            index("wifiLostCategory=outdoor"),
            index("cal_nextLocation=outdoor"),
            index("state_current=outdoor_walking"),
            index("state_current=outdoor_running"),
            index("state_current=outdoor_cycling"),
            index("state_current=outdoor_resting"),
            index("precondition=outdoor_walking"),
            index("precondition=outdoor_running"),
            index("precondition=outdoor_cycling"),
            index("precondition=outdoor_resting"),
        ),
        "exercise": (
            index("ps_location=gym"),
            index("wifiLostCategory=gym"),
            index("cal_nextLocation=gym"),
            index("activityState=active"),
            index("ps_motion=running"),
            index("ps_motion=cycling"),
            index("state_current=at_gym_exercising"),
            index("state_current=at_gym"),
            index("state_current=outdoor_running"),
            index("state_current=outdoor_cycling"),
            index("precondition=at_gym_exercising"),
            index("precondition=at_gym"),
            index("precondition=outdoor_running"),
            index("precondition=outdoor_cycling"),
        ),
        "meal_errand": (
            index("ps_location=restaurant"),
            index("ps_location=cafe"),
            index("ps_location=shopping"),
            index("wifiLostCategory=restaurant"),
            index("wifiLostCategory=cafe"),
            index("wifiLostCategory=shopping"),
            index("cal_nextLocation=restaurant"),
            index("cal_nextLocation=cafe"),
            index("cal_nextLocation=shopping"),
            index("state_current=at_restaurant_lunch"),
            index("state_current=at_restaurant_dinner"),
            index("state_current=at_restaurant_other"),
            index("state_current=at_cafe"),
            index("state_current=at_shopping"),
            index("precondition=at_restaurant_lunch"),
            index("precondition=at_restaurant_dinner"),
            index("precondition=at_restaurant_other"),
            index("precondition=at_cafe"),
            index("precondition=at_shopping"),
            index("ps_time=lunch"),
        ),
        "night_rest": (
            index("ps_time=sleeping"),
            index("ps_time=night"),
            index("ps_time=late_night"),
            index("activityState=sleeping"),
            *indices_with_prefix("state_current=home_sleeping"),
            index("state_current=home_evening"),
            index("state_current=home_evening_dark"),
            index("state_current=home_evening_lying"),
            index("state_current=unknown_lying"),
            *indices_with_prefix("precondition=home_sleeping"),
            index("precondition=home_evening"),
            index("precondition=home_evening_dark"),
            index("precondition=home_evening_lying"),
            index("precondition=unknown_lying"),
        ),
        "quiet_focus": (
            index("ps_sound=silent"),
            index("ps_sound=quiet"),
            index("state_current=office_working_focused"),
            index("state_current=at_cafe_quiet"),
            index("precondition=office_working_focused"),
            index("precondition=at_cafe_quiet"),
        ),
        "noisy_safety": (
            index("ps_sound=noisy"),
            index("state_current=home_evening_noisy"),
            index("state_current=office_working_noisy"),
            index("state_current=unknown_noisy"),
            index("precondition=home_evening_noisy"),
            index("precondition=office_working_noisy"),
            index("precondition=unknown_noisy"),
        ),
    }
    _SHARED_CONTEXT_INDEX_CACHE[feature_dim] = cache
    return cache


def _augment_shared_context_vector(x: np.ndarray, enable_context_interactions: bool) -> np.ndarray:
    if not enable_context_interactions:
        return x
    cache = _build_shared_context_index_cache(int(x.shape[0]))
    if cache is None:
        return x

    def pooled(name: str) -> float:
        indices = cache[name]
        return max(float(x[index]) for index in indices) if indices else 0.0

    derived = np.array(
        [
            pooled("work"),
            pooled("home"),
            pooled("commute_travel"),
            pooled("meeting"),
            pooled("outdoor"),
            pooled("exercise"),
            pooled("meal_errand"),
            pooled("night_rest"),
            pooled("quiet_focus"),
            pooled("noisy_safety"),
        ],
        dtype=np.float32,
    )
    return np.concatenate([x, derived], axis=0)


def _infer_action_type(action_id: str, tokens: list[str]) -> str:
    if action_id.startswith("R_"):
        return "recommend"
    if action_id.startswith("O_SHOW_"):
        return "show"
    if action_id.startswith("O_START_"):
        return "start"
    if action_id.startswith("O_TURN_ON_") or action_id.startswith("O_ENABLE_") or action_id.startswith("O_INCREASE_"):
        return "toggle"
    if action_id.startswith("O_OPEN_"):
        return "open"
    if action_id.startswith("O_PROMPT_"):
        return "prompt"
    if len(tokens) == 1 and action_id.islower():
        return "app"
    return "other"


def _infer_semantic_group(action_id: str, tokens: list[str]) -> str:
    token_set = set(tokens)
    if len(tokens) == 1 and action_id.islower():
        return "app_category"
    if {"schedule", "calendar", "day"} & token_set:
        return "schedule"
    if {"todo", "notes", "note", "follow", "quick", "entry"} & token_set:
        return "todo_notes"
    if "weather" in token_set:
        return "weather"
    if {"payment", "qr"} & token_set:
        return "payment"
    if {"commute", "traffic", "travel", "trip", "drive", "depart"} & token_set:
        return "commute_travel"
    if {"booking", "meeting", "offsite"} & token_set:
        return "booking_meeting"
    if {"nearby", "walking", "route", "place", "places"} & token_set:
        return "nearby_places"
    if {"activity", "tracking", "workout", "walk"} & token_set:
        return "activity_fitness"
    if "timer" in token_set:
        return "timer"
    if {"dnd", "silent", "vibration", "volume", "comfort", "mode", "light", "curtains", "charging", "charge"} & token_set:
        return "device_mode"
    if {"rest", "relax", "sleep", "hydrate", "stretch", "recovery", "meal", "coffee", "break"} & token_set:
        return "rest_recovery"
    if {"safe", "safety", "noisy"} & token_set:
        return "safety"
    if {"pickup", "delivery", "frequent"} & token_set:
        return "pickup_places"
    return "general"


def _infer_scene_affinities(action_id: str, tokens: list[str]) -> tuple[str, ...]:
    token_set = set(tokens)
    affinities: list[str] = []

    def add(name: str) -> None:
        if name not in affinities:
            affinities.append(name)

    if {"work", "office", "calendar", "schedule", "todo", "meeting", "offsite"} & token_set:
        add("work")
    if {"home", "sleep", "relax", "curtains", "light", "charge"} & token_set:
        add("home")
    if {"commute", "traffic"} & token_set:
        add("commute")
    if {"travel", "trip", "booking", "drive", "depart", "pickup"} & token_set:
        add("travel")
    if {"meeting", "calendar"} & token_set:
        add("meeting")
    if {"outdoor", "weather", "walking", "nearby", "route", "hydrate"} & token_set:
        add("outdoor")
    if {"activity", "tracking", "workout", "recovery", "walk"} & token_set:
        add("exercise")
    if {"meal", "coffee", "payment", "qr"} & token_set:
        add("meal_break")
    if {"sleep", "dnd", "silent", "comfort", "light"} & token_set:
        add("night_quiet")
    if {"payment", "nearby", "booking", "pickup", "frequent", "social"} & token_set:
        add("social_errand")

    if len(tokens) == 1 and action_id.islower():
        app_affinity_map = {
            "productivity": ("work",),
            "navigation": ("commute", "travel"),
            "health": ("exercise", "outdoor"),
            "music": ("commute", "exercise", "home"),
            "shopping": ("social_errand",),
            "social": ("social_errand",),
            "reading": ("home", "night_quiet", "work"),
            "news": ("commute", "work"),
            "entertainment": ("home", "social_errand"),
            "game": ("home",),
        }
        for affinity in app_affinity_map.get(action_id, ()):
            add(affinity)

    if not affinities:
        add("work" if action_id.startswith("O_SHOW_") else "home")
    return tuple(affinities)


def build_shared_action_feature_matrix(
    action_ids: list[str],
) -> np.ndarray:
    """Shared action features: bias + identity + action type + semantics + scene affinity."""
    identity_dim = len(action_ids)
    type_dim = len(SHARED_ACTION_TYPES)
    semantic_dim = len(SHARED_SEMANTIC_GROUPS)
    affinity_dim = len(SHARED_SCENE_AFFINITIES)
    feature_dim = 1 + identity_dim + type_dim + semantic_dim + affinity_dim
    matrix = np.zeros((len(action_ids), feature_dim), dtype=np.float32)
    type_offset = 1 + identity_dim
    semantic_offset = type_offset + type_dim
    affinity_offset = semantic_offset + semantic_dim

    for index, action_id in enumerate(action_ids):
        row = matrix[index]
        row[0] = 1.0
        row[1 + index] = 1.0
        tokens = _tokenize_action_id(action_id)
        action_type = _infer_action_type(action_id, tokens)
        semantic_group = _infer_semantic_group(action_id, tokens)
        row[type_offset + SHARED_ACTION_TYPES.index(action_type)] = 1.0
        row[semantic_offset + SHARED_SEMANTIC_GROUPS.index(semantic_group)] = 1.0
        for affinity in _infer_scene_affinities(action_id, tokens):
            row[affinity_offset + SHARED_SCENE_AFFINITIES.index(affinity)] = 1.0
    return matrix


class MaskedDisjointLinUCB:
    """One LinUCB state per action, ranked over a provided candidate set."""

    def __init__(
        self,
        action_ids: list[str],
        feature_dim: int,
        alpha: float = 0.15,
        default_bonus: float = 0.75,
        l2: float = 1.0,
        device: str = "auto",
    ) -> None:
        if not action_ids:
            raise ValueError("action_ids must not be empty")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if l2 <= 0:
            raise ValueError("l2 must be positive")

        self.device = _resolve_device(device)
        self.action_ids = list(action_ids)
        self.feature_dim = feature_dim
        self.alpha = float(alpha)
        self.default_bonus = float(default_bonus)
        self.l2 = float(l2)
        self.action_to_index = {action_id: index for index, action_id in enumerate(self.action_ids)}

        torch = _require_torch()
        identity = torch.eye(feature_dim, dtype=torch.float32, device=self.device) / self.l2
        self.a_inv = identity.unsqueeze(0).repeat(len(self.action_ids), 1, 1)
        self.b = torch.zeros((len(self.action_ids), feature_dim), dtype=torch.float32, device=self.device)

    def rank(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RankedAction]:
        if x.shape != (self.feature_dim,):
            raise ValueError(f"Expected x with shape ({self.feature_dim},), got {x.shape}")
        candidates = list(dict.fromkeys(candidate_action_ids))
        if not candidates:
            raise ValueError("candidate_action_ids must not be empty")
        torch = _require_torch()
        candidate_indices = []
        for action_id in candidates:
            if action_id not in self.action_to_index:
                raise KeyError(f"Unknown action_id {action_id!r}")
            candidate_indices.append(self.action_to_index[action_id])

        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        index_tensor = torch.as_tensor(candidate_indices, dtype=torch.long, device=self.device)
        a_inv = self.a_inv.index_select(0, index_tensor)
        b = self.b.index_select(0, index_tensor)
        theta = torch.matmul(a_inv, b.unsqueeze(-1)).squeeze(-1)
        mean_rewards = torch.sum(theta * x_tensor.unsqueeze(0), dim=1)
        a_inv_x = torch.matmul(a_inv, x_tensor)
        uncertainties = torch.sqrt(torch.clamp(torch.sum(a_inv_x * x_tensor.unsqueeze(0), dim=1), min=0.0))
        default_bonus_tensor = torch.tensor(
            [self.default_bonus if default_action_id is not None and action_id == default_action_id else 0.0 for action_id in candidates],
            dtype=torch.float32,
            device=self.device,
        )
        scores = mean_rewards + self.alpha * uncertainties + default_bonus_tensor

        ranked = [
            RankedAction(
                action_id=action_id,
                score=float(scores[position].item()),
                mean_reward=float(mean_rewards[position].item()),
                uncertainty=float(uncertainties[position].item()),
                default_bonus=float(default_bonus_tensor[position].item()),
            )
            for position, action_id in enumerate(candidates)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        if top_k is None:
            return ranked
        return ranked[:top_k]

    def partial_fit(self, x: np.ndarray, action_id: str, reward: float) -> None:
        torch = _require_torch()
        index = self.action_to_index[action_id]
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        a_inv = self.a_inv[index]
        a_inv_x = torch.matmul(a_inv, x_tensor)
        denominator = 1.0 + torch.dot(x_tensor, a_inv_x)
        self.a_inv[index] = a_inv - torch.outer(a_inv_x, a_inv_x) / denominator
        self.b[index] = self.b[index] + float(reward) * x_tensor

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_type": "masked_disjoint_linucb_v0",
            "action_ids": self.action_ids,
            "feature_dim": self.feature_dim,
            "alpha": self.alpha,
            "default_bonus": self.default_bonus,
            "l2": self.l2,
            "device": self.device,
        }
        (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        np.savez_compressed(
            output_path / "weights.npz",
            a_inv=self.a_inv.detach().cpu().numpy(),
            b=self.b.detach().cpu().numpy(),
        )

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        device: str = "auto",
    ) -> "MaskedDisjointLinUCB":
        artifact_path = Path(artifact_dir)
        manifest = json.loads((artifact_path / "manifest.json").read_text())
        weights = np.load(artifact_path / "weights.npz")
        model = cls(
            action_ids=list(manifest["action_ids"]),
            feature_dim=int(manifest["feature_dim"]),
            alpha=float(manifest["alpha"]),
            default_bonus=float(manifest["default_bonus"]),
            l2=float(manifest["l2"]),
            device=device,
        )
        torch = _require_torch()
        model.a_inv = torch.as_tensor(weights["a_inv"], dtype=torch.float32, device=model.device)
        model.b = torch.as_tensor(weights["b"], dtype=torch.float32, device=model.device)
        return model

    def choose(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> RankedAction:
        ranked = self.rank(x, candidate_action_ids, default_action_id, top_k=None)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon == 0.0:
            return ranked[0]
        rng = random.Random(seed)
        if rng.random() < epsilon:
            return ranked[rng.randrange(len(ranked))]
        return ranked[0]


class MaskedSharedLinearUCB:
    """Shared-parameter diagonal linear UCB over compact action features."""

    def __init__(
        self,
        action_ids: list[str],
        feature_dim: int,
        alpha: float = 0.15,
        default_bonus: float = 0.75,
        l2: float = 1.0,
        device: str = "auto",
        enable_context_interactions: bool = True,
    ) -> None:
        if not action_ids:
            raise ValueError("action_ids must not be empty")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if l2 <= 0:
            raise ValueError("l2 must be positive")

        self.device = _resolve_device(device)
        self.action_ids = list(action_ids)
        self.feature_dim = feature_dim
        self.enable_context_interactions = bool(enable_context_interactions)
        self.alpha = float(alpha)
        self.default_bonus = float(default_bonus)
        self.l2 = float(l2)
        self.action_to_index = {action_id: index for index, action_id in enumerate(self.action_ids)}

        context_probe = _augment_shared_context_vector(
            np.zeros((feature_dim,), dtype=np.float32),
            enable_context_interactions=self.enable_context_interactions,
        )
        self.context_feature_dim = int(context_probe.shape[0])
        self.context_interaction_dim = int(self.context_feature_dim - self.feature_dim)

        action_features = build_shared_action_feature_matrix(self.action_ids)
        self.action_feature_dim = int(action_features.shape[1])
        self.parameter_dim = self.action_feature_dim * (self.context_feature_dim + 1)

        torch = _require_torch()
        self.action_features = torch.as_tensor(action_features, dtype=torch.float32, device=self.device)
        self.a_diag_inter = torch.full(
            (self.action_feature_dim, self.context_feature_dim),
            fill_value=self.l2,
            dtype=torch.float32,
            device=self.device,
        )
        self.a_diag_bias = torch.full(
            (self.action_feature_dim,),
            fill_value=self.l2,
            dtype=torch.float32,
            device=self.device,
        )
        self.b_inter = torch.zeros(
            (self.action_feature_dim, self.context_feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.b_bias = torch.zeros((self.action_feature_dim,), dtype=torch.float32, device=self.device)

    def rank(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RankedAction]:
        if x.shape != (self.feature_dim,):
            raise ValueError(f"Expected x with shape ({self.feature_dim},), got {x.shape}")
        candidates = list(dict.fromkeys(candidate_action_ids))
        if not candidates:
            raise ValueError("candidate_action_ids must not be empty")
        torch = _require_torch()
        candidate_indices = []
        for action_id in candidates:
            if action_id not in self.action_to_index:
                raise KeyError(f"Unknown action_id {action_id!r}")
            candidate_indices.append(self.action_to_index[action_id])

        x_aug = _augment_shared_context_vector(x, enable_context_interactions=self.enable_context_interactions)
        x_tensor = torch.as_tensor(x_aug, dtype=torch.float32, device=self.device)
        x_squared = x_tensor.square()
        z = self.action_features.index_select(
            0, torch.as_tensor(candidate_indices, dtype=torch.long, device=self.device)
        )
        theta_inter = self.b_inter / self.a_diag_inter
        theta_bias = self.b_bias / self.a_diag_bias
        shared_projection = torch.matmul(theta_inter, x_tensor) + theta_bias
        mean_rewards = torch.matmul(z, shared_projection)

        inter_uncertainty_base = torch.sum((x_squared.unsqueeze(0) / self.a_diag_inter), dim=1)
        uncertainty_base = inter_uncertainty_base + (1.0 / self.a_diag_bias)
        uncertainties = torch.sqrt(torch.clamp(torch.matmul(z.square(), uncertainty_base), min=0.0))
        default_bonus_tensor = torch.tensor(
            [self.default_bonus if default_action_id is not None and action_id == default_action_id else 0.0 for action_id in candidates],
            dtype=torch.float32,
            device=self.device,
        )
        scores = mean_rewards + self.alpha * uncertainties + default_bonus_tensor

        ranked = [
            RankedAction(
                action_id=action_id,
                score=float(scores[position].item()),
                mean_reward=float(mean_rewards[position].item()),
                uncertainty=float(uncertainties[position].item()),
                default_bonus=float(default_bonus_tensor[position].item()),
            )
            for position, action_id in enumerate(candidates)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        if top_k is None:
            return ranked
        return ranked[:top_k]

    def partial_fit(self, x: np.ndarray, action_id: str, reward: float) -> None:
        if x.shape != (self.feature_dim,):
            raise ValueError(f"Expected x with shape ({self.feature_dim},), got {x.shape}")
        torch = _require_torch()
        index = self.action_to_index[action_id]
        x_aug = _augment_shared_context_vector(x, enable_context_interactions=self.enable_context_interactions)
        x_tensor = torch.as_tensor(x_aug, dtype=torch.float32, device=self.device)
        z = self.action_features[index]
        z_squared = z.square().unsqueeze(1)
        x_squared = x_tensor.square().unsqueeze(0)
        self.a_diag_inter = self.a_diag_inter + z_squared * x_squared
        self.a_diag_bias = self.a_diag_bias + z.square()
        self.b_inter = self.b_inter + float(reward) * torch.outer(z, x_tensor)
        self.b_bias = self.b_bias + float(reward) * z

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_type": "masked_shared_linear_ucb_v0",
            "action_ids": self.action_ids,
            "feature_dim": self.feature_dim,
            "context_feature_dim": self.context_feature_dim,
            "context_interaction_dim": self.context_interaction_dim,
            "action_feature_dim": self.action_feature_dim,
            "parameter_dim": self.parameter_dim,
            "alpha": self.alpha,
            "default_bonus": self.default_bonus,
            "l2": self.l2,
            "device": self.device,
            "action_feature_schema": "identity_type_semantic_affinity_v1",
            "context_interaction_schema": "derived_context_block_v1" if self.enable_context_interactions else "none",
            "enable_context_interactions": self.enable_context_interactions,
        }
        (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        np.savez_compressed(
            output_path / "weights.npz",
            action_features=self.action_features.detach().cpu().numpy(),
            a_diag_inter=self.a_diag_inter.detach().cpu().numpy(),
            a_diag_bias=self.a_diag_bias.detach().cpu().numpy(),
            b_inter=self.b_inter.detach().cpu().numpy(),
            b_bias=self.b_bias.detach().cpu().numpy(),
        )

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        device: str = "auto",
    ) -> "MaskedSharedLinearUCB":
        artifact_path = Path(artifact_dir)
        manifest = json.loads((artifact_path / "manifest.json").read_text())
        weights = np.load(artifact_path / "weights.npz")
        model = cls(
            action_ids=list(manifest["action_ids"]),
            feature_dim=int(manifest.get("feature_dim", manifest.get("raw_feature_dim"))),
            alpha=float(manifest["alpha"]),
            default_bonus=float(manifest["default_bonus"]),
            l2=float(manifest["l2"]),
            device=device,
            enable_context_interactions=bool(manifest.get("enable_context_interactions", False)),
        )
        torch = _require_torch()
        model.action_features = torch.as_tensor(weights["action_features"], dtype=torch.float32, device=model.device)
        model.a_diag_inter = torch.as_tensor(weights["a_diag_inter"], dtype=torch.float32, device=model.device)
        model.a_diag_bias = torch.as_tensor(weights["a_diag_bias"], dtype=torch.float32, device=model.device)
        model.b_inter = torch.as_tensor(weights["b_inter"], dtype=torch.float32, device=model.device)
        model.b_bias = torch.as_tensor(weights["b_bias"], dtype=torch.float32, device=model.device)
        model.action_feature_dim = int(model.action_features.shape[1])
        model.context_feature_dim = int(model.a_diag_inter.shape[1])
        model.context_interaction_dim = int(model.context_feature_dim - model.feature_dim)
        model.parameter_dim = model.action_feature_dim * (model.context_feature_dim + 1)
        return model

    def choose(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> RankedAction:
        ranked = self.rank(x, candidate_action_ids, default_action_id, top_k=None)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon == 0.0:
            return ranked[0]
        rng = random.Random(seed)
        if rng.random() < epsilon:
            return ranked[rng.randrange(len(ranked))]
        return ranked[0]


class MaskedNeuralLinearUCB:
    """NeuralLinear UCB with a small MLP encoder and latent disjoint UCB heads."""

    def __init__(
        self,
        action_ids: list[str],
        feature_dim: int,
        alpha: float = 0.15,
        default_bonus: float = 0.75,
        l2: float = 1.0,
        device: str = "auto",
        hidden_dim: int = 128,
        latent_dim: int = 64,
    ) -> None:
        if not action_ids:
            raise ValueError("action_ids must not be empty")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if hidden_dim <= 0 or latent_dim <= 0:
            raise ValueError("hidden_dim and latent_dim must be positive")
        if l2 <= 0:
            raise ValueError("l2 must be positive")

        self.device = _resolve_device(device)
        self.action_ids = list(action_ids)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.alpha = float(alpha)
        self.default_bonus = float(default_bonus)
        self.l2 = float(l2)
        self.action_to_index = {action_id: index for index, action_id in enumerate(self.action_ids)}

        torch = _require_torch()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.latent_dim),
            torch.nn.ReLU(),
        ).to(self.device)
        self.encoder.eval()
        identity = torch.eye(self.latent_dim, dtype=torch.float32, device=self.device) / self.l2
        self.a_inv = identity.unsqueeze(0).repeat(len(self.action_ids), 1, 1)
        self.b = torch.zeros((len(self.action_ids), self.latent_dim), dtype=torch.float32, device=self.device)

    def encode_context(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (self.feature_dim,):
            raise ValueError(f"Expected x with shape ({self.feature_dim},), got {x.shape}")
        torch = _require_torch()
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.encoder.eval()
        with torch.no_grad():
            latent = self.encoder(x_tensor).squeeze(0)
        return latent.detach().cpu().numpy().astype(np.float32, copy=False)

    def rank(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RankedAction]:
        h = self.encode_context(x)
        candidates = list(dict.fromkeys(candidate_action_ids))
        if not candidates:
            raise ValueError("candidate_action_ids must not be empty")
        torch = _require_torch()
        candidate_indices = []
        for action_id in candidates:
            if action_id not in self.action_to_index:
                raise KeyError(f"Unknown action_id {action_id!r}")
            candidate_indices.append(self.action_to_index[action_id])

        h_tensor = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        index_tensor = torch.as_tensor(candidate_indices, dtype=torch.long, device=self.device)
        a_inv = self.a_inv.index_select(0, index_tensor)
        b = self.b.index_select(0, index_tensor)
        theta = torch.matmul(a_inv, b.unsqueeze(-1)).squeeze(-1)
        mean_rewards = torch.sum(theta * h_tensor.unsqueeze(0), dim=1)
        a_inv_h = torch.matmul(a_inv, h_tensor)
        uncertainties = torch.sqrt(torch.clamp(torch.sum(a_inv_h * h_tensor.unsqueeze(0), dim=1), min=0.0))
        default_bonus_tensor = torch.tensor(
            [self.default_bonus if default_action_id is not None and action_id == default_action_id else 0.0 for action_id in candidates],
            dtype=torch.float32,
            device=self.device,
        )
        scores = mean_rewards + self.alpha * uncertainties + default_bonus_tensor

        ranked = [
            RankedAction(
                action_id=action_id,
                score=float(scores[position].item()),
                mean_reward=float(mean_rewards[position].item()),
                uncertainty=float(uncertainties[position].item()),
                default_bonus=float(default_bonus_tensor[position].item()),
            )
            for position, action_id in enumerate(candidates)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        if top_k is None:
            return ranked
        return ranked[:top_k]

    def partial_fit(self, x: np.ndarray, action_id: str, reward: float) -> None:
        torch = _require_torch()
        index = self.action_to_index[action_id]
        h = self.encode_context(x)
        h_tensor = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        a_inv = self.a_inv[index]
        a_inv_h = torch.matmul(a_inv, h_tensor)
        denominator = 1.0 + torch.dot(h_tensor, a_inv_h)
        self.a_inv[index] = a_inv - torch.outer(a_inv_h, a_inv_h) / denominator
        self.b[index] = self.b[index] + float(reward) * h_tensor

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_type": "masked_neural_linear_ucb_v0",
            "action_ids": self.action_ids,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "alpha": self.alpha,
            "default_bonus": self.default_bonus,
            "l2": self.l2,
            "device": self.device,
            "encoder_architecture": [self.feature_dim, self.hidden_dim, self.latent_dim],
        }
        (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        torch = _require_torch()
        torch.save(self.encoder.state_dict(), output_path / "encoder_state.pt")
        np.savez_compressed(
            output_path / "weights.npz",
            a_inv=self.a_inv.detach().cpu().numpy(),
            b=self.b.detach().cpu().numpy(),
        )

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        device: str = "auto",
    ) -> "MaskedNeuralLinearUCB":
        artifact_path = Path(artifact_dir)
        manifest = json.loads((artifact_path / "manifest.json").read_text())
        weights = np.load(artifact_path / "weights.npz")
        model = cls(
            action_ids=list(manifest["action_ids"]),
            feature_dim=int(manifest["feature_dim"]),
            hidden_dim=int(manifest.get("hidden_dim", 128)),
            latent_dim=int(manifest.get("latent_dim", 64)),
            alpha=float(manifest["alpha"]),
            default_bonus=float(manifest["default_bonus"]),
            l2=float(manifest["l2"]),
            device=device,
        )
        torch = _require_torch()
        state_dict = torch.load(artifact_path / "encoder_state.pt", map_location=model.device)
        model.encoder.load_state_dict(state_dict)
        model.encoder.eval()
        model.a_inv = torch.as_tensor(weights["a_inv"], dtype=torch.float32, device=model.device)
        model.b = torch.as_tensor(weights["b"], dtype=torch.float32, device=model.device)
        return model

    def choose(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> RankedAction:
        ranked = self.rank(x, candidate_action_ids, default_action_id, top_k=None)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon == 0.0:
            return ranked[0]
        rng = random.Random(seed)
        if rng.random() < epsilon:
            return ranked[rng.randrange(len(ranked))]
        return ranked[0]


class MaskedNeuralScorer:
    """Pure neural scorer with no uncertainty bonus."""

    def __init__(
        self,
        action_ids: list[str],
        feature_dim: int,
        alpha: float = 0.0,
        default_bonus: float = 0.0,
        l2: float = 1.0,
        device: str = "auto",
        hidden_dims: tuple[int, int] = (128, 64),
    ) -> None:
        if not action_ids:
            raise ValueError("action_ids must not be empty")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")

        self.device = _resolve_device(device)
        self.action_ids = list(action_ids)
        self.feature_dim = int(feature_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.alpha = float(alpha)
        self.default_bonus = float(default_bonus)
        self.l2 = float(l2)
        self.action_to_index = {action_id: index for index, action_id in enumerate(self.action_ids)}

        self.network = _build_mlp(self.feature_dim, self.hidden_dims, len(self.action_ids)).to(self.device)
        self.network.eval()

    def score_all(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (self.feature_dim,):
            raise ValueError(f"Expected x with shape ({self.feature_dim},), got {x.shape}")
        torch = _require_torch()
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.network.eval()
        with torch.no_grad():
            scores = self.network(x_tensor).squeeze(0)
        return scores.detach().cpu().numpy().astype(np.float32, copy=False)

    def rank(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RankedAction]:
        candidates = list(dict.fromkeys(candidate_action_ids))
        if not candidates:
            raise ValueError("candidate_action_ids must not be empty")
        scores = self.score_all(x)
        ranked: list[RankedAction] = []
        for action_id in candidates:
            if action_id not in self.action_to_index:
                raise KeyError(f"Unknown action_id {action_id!r}")
            mean_reward = float(scores[self.action_to_index[action_id]])
            default_bonus_value = self.default_bonus if default_action_id is not None and action_id == default_action_id else 0.0
            ranked.append(
                RankedAction(
                    action_id=action_id,
                    score=mean_reward + default_bonus_value,
                    mean_reward=mean_reward,
                    uncertainty=0.0,
                    default_bonus=float(default_bonus_value),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        if top_k is None:
            return ranked
        return ranked[:top_k]

    def partial_fit(self, x: np.ndarray, action_id: str, reward: float) -> None:  # noqa: ARG002
        return None

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_type": "masked_neural_scorer_v0",
            "action_ids": self.action_ids,
            "feature_dim": self.feature_dim,
            "hidden_dims": list(self.hidden_dims),
            "alpha": self.alpha,
            "default_bonus": self.default_bonus,
            "l2": self.l2,
            "device": self.device,
        }
        (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        torch = _require_torch()
        torch.save(self.network.state_dict(), output_path / "network_state.pt")

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        device: str = "auto",
    ) -> "MaskedNeuralScorer":
        artifact_path = Path(artifact_dir)
        manifest = json.loads((artifact_path / "manifest.json").read_text())
        model = cls(
            action_ids=list(manifest["action_ids"]),
            feature_dim=int(manifest["feature_dim"]),
            alpha=float(manifest.get("alpha", 0.0)),
            default_bonus=float(manifest.get("default_bonus", 0.0)),
            l2=float(manifest.get("l2", 1.0)),
            device=device,
            hidden_dims=tuple(int(value) for value in manifest.get("hidden_dims", [128, 64])),
        )
        torch = _require_torch()
        model.network.load_state_dict(torch.load(artifact_path / "network_state.pt", map_location=model.device))
        model.network.eval()
        return model

    def choose(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> RankedAction:
        ranked = self.rank(x, candidate_action_ids, default_action_id, top_k=None)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon == 0.0:
            return ranked[0]
        rng = random.Random(seed)
        if rng.random() < epsilon:
            return ranked[rng.randrange(len(ranked))]
        return ranked[0]


class MaskedNeuralUCBLite:
    """Neural trunk with last-layer linear UCB heads."""

    def __init__(
        self,
        action_ids: list[str],
        feature_dim: int,
        alpha: float = 0.15,
        default_bonus: float = 0.75,
        l2: float = 1.0,
        device: str = "auto",
        hidden_dims: tuple[int, int] = (128, 64),
    ) -> None:
        if not action_ids:
            raise ValueError("action_ids must not be empty")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if l2 <= 0:
            raise ValueError("l2 must be positive")

        self.device = _resolve_device(device)
        self.action_ids = list(action_ids)
        self.feature_dim = int(feature_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.latent_dim = int(self.hidden_dims[-1])
        self.alpha = float(alpha)
        self.default_bonus = float(default_bonus)
        self.l2 = float(l2)
        self.action_to_index = {action_id: index for index, action_id in enumerate(self.action_ids)}

        self.trunk = _build_mlp(self.feature_dim, self.hidden_dims[:-1], self.latent_dim).to(self.device)
        self.trunk.eval()
        self.ucb_feature_dim = self.latent_dim + 1
        torch = _require_torch()
        identity = torch.eye(self.ucb_feature_dim, dtype=torch.float32, device=self.device) / self.l2
        self.a_inv = identity.unsqueeze(0).repeat(len(self.action_ids), 1, 1)
        self.b = torch.zeros((len(self.action_ids), self.ucb_feature_dim), dtype=torch.float32, device=self.device)

    def encode_context(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (self.feature_dim,):
            raise ValueError(f"Expected x with shape ({self.feature_dim},), got {x.shape}")
        torch = _require_torch()
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.trunk.eval()
        with torch.no_grad():
            latent = self.trunk(x_tensor).squeeze(0)
        return latent.detach().cpu().numpy().astype(np.float32, copy=False)

    def initialize_from_pretrained_head(self, weight: np.ndarray, bias: np.ndarray) -> None:
        if weight.shape != (len(self.action_ids), self.latent_dim):
            raise ValueError(f"Expected weight shape ({len(self.action_ids)}, {self.latent_dim}), got {weight.shape}")
        if bias.shape != (len(self.action_ids),):
            raise ValueError(f"Expected bias shape ({len(self.action_ids)},), got {bias.shape}")
        theta = np.concatenate([weight.astype(np.float32, copy=False), bias[:, None].astype(np.float32, copy=False)], axis=1)
        torch = _require_torch()
        theta_tensor = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        self.b = theta_tensor * self.l2

    def rank(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RankedAction]:
        candidates = list(dict.fromkeys(candidate_action_ids))
        if not candidates:
            raise ValueError("candidate_action_ids must not be empty")
        h = _augment_with_bias(self.encode_context(x))
        torch = _require_torch()
        h_tensor = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        ranked: list[RankedAction] = []
        for action_id in candidates:
            if action_id not in self.action_to_index:
                raise KeyError(f"Unknown action_id {action_id!r}")
            index = self.action_to_index[action_id]
            a_inv = self.a_inv[index]
            b = self.b[index]
            theta = torch.matmul(a_inv, b.unsqueeze(-1)).squeeze(-1)
            mean_reward = float(torch.dot(theta, h_tensor).item())
            a_inv_h = torch.matmul(a_inv, h_tensor)
            uncertainty = float(torch.sqrt(torch.clamp(torch.dot(h_tensor, a_inv_h), min=0.0)).item())
            default_bonus_value = self.default_bonus if default_action_id is not None and action_id == default_action_id else 0.0
            ranked.append(
                RankedAction(
                    action_id=action_id,
                    score=mean_reward + self.alpha * uncertainty + default_bonus_value,
                    mean_reward=mean_reward,
                    uncertainty=uncertainty,
                    default_bonus=float(default_bonus_value),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        if top_k is None:
            return ranked
        return ranked[:top_k]

    def partial_fit(self, x: np.ndarray, action_id: str, reward: float) -> None:
        if action_id not in self.action_to_index:
            raise KeyError(f"Unknown action_id {action_id!r}")
        torch = _require_torch()
        index = self.action_to_index[action_id]
        h = _augment_with_bias(self.encode_context(x))
        h_tensor = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        a_inv = self.a_inv[index]
        a_inv_h = torch.matmul(a_inv, h_tensor)
        denominator = 1.0 + torch.dot(h_tensor, a_inv_h)
        self.a_inv[index] = a_inv - torch.outer(a_inv_h, a_inv_h) / denominator
        self.b[index] = self.b[index] + float(reward) * h_tensor

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_type": "masked_neural_ucb_lite_v0",
            "action_ids": self.action_ids,
            "feature_dim": self.feature_dim,
            "hidden_dims": list(self.hidden_dims),
            "latent_dim": self.latent_dim,
            "ucb_feature_dim": self.ucb_feature_dim,
            "alpha": self.alpha,
            "default_bonus": self.default_bonus,
            "l2": self.l2,
            "device": self.device,
        }
        (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        torch = _require_torch()
        torch.save(self.trunk.state_dict(), output_path / "trunk_state.pt")
        np.savez_compressed(
            output_path / "weights.npz",
            a_inv=self.a_inv.detach().cpu().numpy(),
            b=self.b.detach().cpu().numpy(),
        )

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        device: str = "auto",
    ) -> "MaskedNeuralUCBLite":
        artifact_path = Path(artifact_dir)
        manifest = json.loads((artifact_path / "manifest.json").read_text())
        model = cls(
            action_ids=list(manifest["action_ids"]),
            feature_dim=int(manifest["feature_dim"]),
            alpha=float(manifest["alpha"]),
            default_bonus=float(manifest["default_bonus"]),
            l2=float(manifest["l2"]),
            device=device,
            hidden_dims=tuple(int(value) for value in manifest.get("hidden_dims", [128, 64])),
        )
        torch = _require_torch()
        model.trunk.load_state_dict(torch.load(artifact_path / "trunk_state.pt", map_location=model.device))
        model.trunk.eval()
        weights = np.load(artifact_path / "weights.npz")
        model.a_inv = torch.as_tensor(weights["a_inv"], dtype=torch.float32, device=model.device)
        model.b = torch.as_tensor(weights["b"], dtype=torch.float32, device=model.device)
        return model

    def choose(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> RankedAction:
        ranked = self.rank(x, candidate_action_ids, default_action_id, top_k=None)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon == 0.0:
            return ranked[0]
        rng = random.Random(seed)
        if rng.random() < epsilon:
            return ranked[rng.randrange(len(ranked))]
        return ranked[0]


class MaskedNeuralUCB:
    """Exact full-network NeuralUCB over a small projected input space."""

    def __init__(
        self,
        action_ids: list[str],
        feature_dim: int,
        alpha: float = 0.15,
        default_bonus: float = 0.75,
        l2: float = 1.0,
        device: str = "auto",
        projection_dim: int = 2,
        hidden_dim: int = 2,
        projection_seed: int = 0,
        learning_rate: float = 1e-3,
    ) -> None:
        if not action_ids:
            raise ValueError("action_ids must not be empty")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if l2 <= 0:
            raise ValueError("l2 must be positive")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.device = _resolve_device(device)
        self.action_ids = list(action_ids)
        self.feature_dim = int(feature_dim)
        self.alpha = float(alpha)
        self.default_bonus = float(default_bonus)
        self.l2 = float(l2)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection_seed = int(projection_seed)
        self.learning_rate = float(learning_rate)
        self.action_to_index = {action_id: index for index, action_id in enumerate(self.action_ids)}

        rng = np.random.default_rng(self.projection_seed)
        self.projection_matrix = rng.normal(
            loc=0.0,
            scale=1.0 / max(self.feature_dim, 1) ** 0.5,
            size=(self.feature_dim, self.projection_dim),
        ).astype(np.float32)

        torch = _require_torch()
        self.network = _build_mlp(self.projection_dim, (self.hidden_dim,), len(self.action_ids)).to(self.device)
        self.network.eval()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.parameter_dim = int(sum(parameter.numel() for parameter in self.network.parameters()))
        identity = torch.eye(self.parameter_dim, dtype=torch.float64, device=self.device) / self.l2
        self.a_inv = identity

    def _project_context(self, x: np.ndarray):
        if x.shape != (self.feature_dim,):
            raise ValueError(f"Expected x with shape ({self.feature_dim},), got {x.shape}")
        projected = np.matmul(x.astype(np.float32, copy=False), self.projection_matrix)
        torch = _require_torch()
        return torch.as_tensor(projected, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _score_tensor(self, x: np.ndarray):
        x_tensor = self._project_context(x)
        return self.network(x_tensor).squeeze(0)

    def _gradient_feature(self, x: np.ndarray, action_index: int) -> np.ndarray:
        torch = _require_torch()
        self.network.zero_grad(set_to_none=True)
        scores = self._score_tensor(x)
        scores[action_index].backward()
        gradients = [parameter.grad for parameter in self.network.parameters() if parameter.grad is not None]
        return _flatten_grad_tuple(gradients)

    def rank(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RankedAction]:
        candidates = list(dict.fromkeys(candidate_action_ids))
        if not candidates:
            raise ValueError("candidate_action_ids must not be empty")
        torch = _require_torch()
        self.network.eval()
        with torch.no_grad():
            scores = self._score_tensor(x).detach().cpu().numpy().astype(np.float32, copy=False)
        ranked: list[RankedAction] = []
        for action_id in candidates:
            if action_id not in self.action_to_index:
                raise KeyError(f"Unknown action_id {action_id!r}")
            index = self.action_to_index[action_id]
            gradient = self._gradient_feature(x, index)
            gradient_tensor = torch.as_tensor(gradient, dtype=torch.float64, device=self.device)
            a_inv_g = torch.matmul(self.a_inv, gradient_tensor)
            uncertainty = float(torch.sqrt(torch.clamp(torch.dot(gradient_tensor, a_inv_g), min=0.0)).item())
            mean_reward = float(scores[index])
            default_bonus_value = self.default_bonus if default_action_id is not None and action_id == default_action_id else 0.0
            ranked.append(
                RankedAction(
                    action_id=action_id,
                    score=mean_reward + self.alpha * uncertainty + default_bonus_value,
                    mean_reward=mean_reward,
                    uncertainty=uncertainty,
                    default_bonus=float(default_bonus_value),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        if top_k is None:
            return ranked
        return ranked[:top_k]

    def partial_fit(self, x: np.ndarray, action_id: str, reward: float) -> None:
        if action_id not in self.action_to_index:
            raise KeyError(f"Unknown action_id {action_id!r}")
        index = self.action_to_index[action_id]
        torch = _require_torch()
        gradient = self._gradient_feature(x, index)
        gradient_tensor = torch.as_tensor(gradient, dtype=torch.float64, device=self.device)
        a_inv_g = torch.matmul(self.a_inv, gradient_tensor)
        denominator = 1.0 + torch.dot(gradient_tensor, a_inv_g)
        self.a_inv = self.a_inv - torch.outer(a_inv_g, a_inv_g) / denominator

        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        scores = self._score_tensor(x)
        loss = 0.5 * (scores[index] - float(reward)) ** 2
        loss.backward()
        self.optimizer.step()
        self.network.eval()

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_type": "masked_neural_ucb_v0",
            "action_ids": self.action_ids,
            "feature_dim": self.feature_dim,
            "alpha": self.alpha,
            "default_bonus": self.default_bonus,
            "l2": self.l2,
            "device": self.device,
            "projection_dim": self.projection_dim,
            "hidden_dim": self.hidden_dim,
            "projection_seed": self.projection_seed,
            "learning_rate": self.learning_rate,
            "parameter_dim": self.parameter_dim,
        }
        (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        torch = _require_torch()
        torch.save(self.network.state_dict(), output_path / "network_state.pt")
        np.savez_compressed(
            output_path / "weights.npz",
            a_inv=self.a_inv.detach().cpu().numpy(),
            projection_matrix=self.projection_matrix,
        )

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        device: str = "auto",
    ) -> "MaskedNeuralUCB":
        artifact_path = Path(artifact_dir)
        manifest = json.loads((artifact_path / "manifest.json").read_text())
        model = cls(
            action_ids=list(manifest["action_ids"]),
            feature_dim=int(manifest["feature_dim"]),
            alpha=float(manifest["alpha"]),
            default_bonus=float(manifest["default_bonus"]),
            l2=float(manifest["l2"]),
            device=device,
            projection_dim=int(manifest.get("projection_dim", 2)),
            hidden_dim=int(manifest.get("hidden_dim", 2)),
            projection_seed=int(manifest.get("projection_seed", 0)),
            learning_rate=float(manifest.get("learning_rate", 1e-3)),
        )
        torch = _require_torch()
        model.network.load_state_dict(torch.load(artifact_path / "network_state.pt", map_location=model.device))
        model.network.eval()
        weights = np.load(artifact_path / "weights.npz")
        if "projection_matrix" in weights:
            model.projection_matrix = weights["projection_matrix"].astype(np.float32, copy=False)
        model.a_inv = torch.as_tensor(weights["a_inv"], dtype=torch.float64, device=model.device)
        model.parameter_dim = int(manifest.get("parameter_dim", model.parameter_dim))
        return model

    def choose(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> RankedAction:
        ranked = self.rank(x, candidate_action_ids, default_action_id, top_k=None)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon == 0.0:
            return ranked[0]
        rng = random.Random(seed)
        if rng.random() < epsilon:
            return ranked[rng.randrange(len(ranked))]
        return ranked[0]


class MaskedNeuralUCBDirect:
    """Shared neural scorer trained directly on bandit replay with hidden-feature UCB."""

    def __init__(
        self,
        action_ids: list[str],
        feature_dim: int,
        alpha: float = 0.15,
        default_bonus: float = 0.75,
        l2: float = 1.0,
        device: str = "auto",
        hidden_dims: tuple[int, int] = (128, 64),
        learning_rate: float = 1e-3,
    ) -> None:
        if not action_ids:
            raise ValueError("action_ids must not be empty")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if l2 <= 0:
            raise ValueError("l2 must be positive")

        self.device = _resolve_device(device)
        self.action_ids = list(action_ids)
        self.feature_dim = int(feature_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        if len(self.hidden_dims) != 2:
            raise ValueError("hidden_dims for neural-ucb-direct must have length 2")
        self.latent_dim = int(self.hidden_dims[-1])
        self.alpha = float(alpha)
        self.default_bonus = float(default_bonus)
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.action_to_index = {action_id: index for index, action_id in enumerate(self.action_ids)}

        torch = _require_torch()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[0], self.latent_dim),
            torch.nn.ReLU(),
        ).to(self.device)
        self.head = torch.nn.Linear(self.latent_dim, len(self.action_ids)).to(self.device)
        self.trunk.eval()
        self.head.eval()
        self.optimizer = torch.optim.Adam(
            list(self.trunk.parameters()) + list(self.head.parameters()),
            lr=self.learning_rate,
        )
        self.ucb_feature_dim = self.latent_dim + 1
        identity = torch.eye(self.ucb_feature_dim, dtype=torch.float32, device=self.device) / self.l2
        self.a_inv = identity.unsqueeze(0).repeat(len(self.action_ids), 1, 1)

    def _forward_with_hidden(self, x: np.ndarray):
        if x.shape != (self.feature_dim,):
            raise ValueError(f"Expected x with shape ({self.feature_dim},), got {x.shape}")
        torch = _require_torch()
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        hidden = self.trunk(x_tensor).squeeze(0)
        scores = self.head(hidden.unsqueeze(0)).squeeze(0)
        return hidden, scores

    def encode_context(self, x: np.ndarray) -> np.ndarray:
        self.trunk.eval()
        self.head.eval()
        with _require_torch().no_grad():
            hidden, _scores = self._forward_with_hidden(x)
        return hidden.detach().cpu().numpy().astype(np.float32, copy=False)

    def score_all(self, x: np.ndarray) -> np.ndarray:
        self.trunk.eval()
        self.head.eval()
        with _require_torch().no_grad():
            _hidden, scores = self._forward_with_hidden(x)
        return scores.detach().cpu().numpy().astype(np.float32, copy=False)

    def rank(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RankedAction]:
        candidates = list(dict.fromkeys(candidate_action_ids))
        if not candidates:
            raise ValueError("candidate_action_ids must not be empty")
        h = _augment_with_bias(self.encode_context(x))
        scores = self.score_all(x)
        torch = _require_torch()
        h_tensor = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        ranked: list[RankedAction] = []
        for action_id in candidates:
            if action_id not in self.action_to_index:
                raise KeyError(f"Unknown action_id {action_id!r}")
            index = self.action_to_index[action_id]
            a_inv = self.a_inv[index]
            a_inv_h = torch.matmul(a_inv, h_tensor)
            uncertainty = float(torch.sqrt(torch.clamp(torch.dot(h_tensor, a_inv_h), min=0.0)).item())
            mean_reward = float(scores[index])
            default_bonus_value = self.default_bonus if default_action_id is not None and action_id == default_action_id else 0.0
            ranked.append(
                RankedAction(
                    action_id=action_id,
                    score=mean_reward + self.alpha * uncertainty + default_bonus_value,
                    mean_reward=mean_reward,
                    uncertainty=uncertainty,
                    default_bonus=float(default_bonus_value),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        if top_k is None:
            return ranked
        return ranked[:top_k]

    def partial_fit(self, x: np.ndarray, action_id: str, reward: float) -> None:
        if action_id not in self.action_to_index:
            raise KeyError(f"Unknown action_id {action_id!r}")
        index = self.action_to_index[action_id]
        torch = _require_torch()
        self.trunk.train()
        self.head.train()
        hidden, scores = self._forward_with_hidden(x)
        h = torch.cat([hidden.detach(), torch.ones((1,), dtype=torch.float32, device=self.device)], dim=0)
        a_inv = self.a_inv[index]
        a_inv_h = torch.matmul(a_inv, h)
        denominator = 1.0 + torch.dot(h, a_inv_h)
        self.a_inv[index] = a_inv - torch.outer(a_inv_h, a_inv_h) / denominator

        self.optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * (scores[index] - float(reward)) ** 2
        loss.backward()
        self.optimizer.step()
        self.trunk.eval()
        self.head.eval()

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_type": "masked_neural_ucb_direct_v0",
            "action_ids": self.action_ids,
            "feature_dim": self.feature_dim,
            "hidden_dims": list(self.hidden_dims),
            "latent_dim": self.latent_dim,
            "ucb_feature_dim": self.ucb_feature_dim,
            "alpha": self.alpha,
            "default_bonus": self.default_bonus,
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "device": self.device,
        }
        (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        torch = _require_torch()
        torch.save(self.trunk.state_dict(), output_path / "trunk_state.pt")
        torch.save(self.head.state_dict(), output_path / "head_state.pt")
        np.savez_compressed(
            output_path / "weights.npz",
            a_inv=self.a_inv.detach().cpu().numpy(),
        )

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        device: str = "auto",
    ) -> "MaskedNeuralUCBDirect":
        artifact_path = Path(artifact_dir)
        manifest = json.loads((artifact_path / "manifest.json").read_text())
        model = cls(
            action_ids=list(manifest["action_ids"]),
            feature_dim=int(manifest["feature_dim"]),
            alpha=float(manifest["alpha"]),
            default_bonus=float(manifest["default_bonus"]),
            l2=float(manifest["l2"]),
            device=device,
            hidden_dims=tuple(int(value) for value in manifest.get("hidden_dims", [128, 64])),
            learning_rate=float(manifest.get("learning_rate", 1e-3)),
        )
        torch = _require_torch()
        model.trunk.load_state_dict(torch.load(artifact_path / "trunk_state.pt", map_location=model.device))
        model.head.load_state_dict(torch.load(artifact_path / "head_state.pt", map_location=model.device))
        model.trunk.eval()
        model.head.eval()
        weights = np.load(artifact_path / "weights.npz")
        model.a_inv = torch.as_tensor(weights["a_inv"], dtype=torch.float32, device=model.device)
        return model

    def choose(
        self,
        x: np.ndarray,
        candidate_action_ids: Iterable[str],
        default_action_id: str | None = None,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> RankedAction:
        ranked = self.rank(x, candidate_action_ids, default_action_id, top_k=None)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon == 0.0:
            return ranked[0]
        rng = random.Random(seed)
        if rng.random() < epsilon:
            return ranked[rng.randrange(len(ranked))]
        return ranked[0]


def create_bandit_model(
    model_type: str,
    action_ids: list[str],
    feature_dim: int,
    alpha: float = 0.15,
    default_bonus: float = 0.75,
    l2: float = 1.0,
    device: str = "auto",
):
    normalized_model_type = model_type.replace("-", "_").lower()
    if normalized_model_type == "disjoint":
        return MaskedDisjointLinUCB(
            action_ids=action_ids,
            feature_dim=feature_dim,
            alpha=alpha,
            default_bonus=default_bonus,
            l2=l2,
            device=device,
        )
    if normalized_model_type == "shared_linear":
        return MaskedSharedLinearUCB(
            action_ids=action_ids,
            feature_dim=feature_dim,
            alpha=alpha,
            default_bonus=default_bonus,
            l2=l2,
            device=device,
        )
    if normalized_model_type == "neural_linear":
        return MaskedNeuralLinearUCB(
            action_ids=action_ids,
            feature_dim=feature_dim,
            alpha=alpha,
            default_bonus=default_bonus,
            l2=l2,
            device=device,
        )
    if normalized_model_type == "neural_scorer":
        return MaskedNeuralScorer(
            action_ids=action_ids,
            feature_dim=feature_dim,
            alpha=alpha,
            default_bonus=default_bonus,
            l2=l2,
            device=device,
        )
    if normalized_model_type == "neural_ucb_lite":
        return MaskedNeuralUCBLite(
            action_ids=action_ids,
            feature_dim=feature_dim,
            alpha=alpha,
            default_bonus=default_bonus,
            l2=l2,
            device=device,
        )
    if normalized_model_type == "neural_ucb":
        return MaskedNeuralUCB(
            action_ids=action_ids,
            feature_dim=feature_dim,
            alpha=alpha,
            default_bonus=default_bonus,
            l2=l2,
            device=device,
        )
    if normalized_model_type == "neural_ucb_direct":
        return MaskedNeuralUCBDirect(
            action_ids=action_ids,
            feature_dim=feature_dim,
            alpha=alpha,
            default_bonus=default_bonus,
            l2=l2,
            device=device,
        )
    raise ValueError(
        f"Unknown model_type {model_type!r}. Expected one of disjoint, shared-linear, neural-linear, "
        "neural-scorer, neural-ucb-lite, neural-ucb, neural-ucb-direct."
    )


def load_bandit_model(artifact_dir: str | Path, device: str = "auto"):
    artifact_path = Path(artifact_dir)
    manifest = json.loads((artifact_path / "manifest.json").read_text())
    model_type = manifest.get("model_type")
    if model_type == "masked_disjoint_linucb_v0":
        return MaskedDisjointLinUCB.load(artifact_path, device=device)
    if model_type == "masked_shared_linear_ucb_v0":
        return MaskedSharedLinearUCB.load(artifact_path, device=device)
    if model_type == "masked_neural_linear_ucb_v0":
        return MaskedNeuralLinearUCB.load(artifact_path, device=device)
    if model_type == "masked_neural_scorer_v0":
        return MaskedNeuralScorer.load(artifact_path, device=device)
    if model_type == "masked_neural_ucb_lite_v0":
        return MaskedNeuralUCBLite.load(artifact_path, device=device)
    if model_type == "masked_neural_ucb_v0":
        return MaskedNeuralUCB.load(artifact_path, device=device)
    if model_type == "masked_neural_ucb_direct_v0":
        return MaskedNeuralUCBDirect.load(artifact_path, device=device)
    raise ValueError(f"Unknown manifest model_type {model_type!r}")
