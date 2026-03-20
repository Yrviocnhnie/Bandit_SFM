"""Dual UCB model for RO and APP recommendations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .feature_space import APP_ACTIONS, BANDIT_FEATURE_DIM, RO_ACTIONS
except ImportError:
    from feature_space import APP_ACTIONS, BANDIT_FEATURE_DIM, RO_ACTIONS


CPP_HEADER_DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "entry"
    / "src"
    / "main"
    / "cpp"
    / "context_engine"
    / "pretrained_weights_v2.h"
)


@dataclass(frozen=True)
class HeadPrediction:
    action: str
    score: float
    prior_prob: float
    ucb_score: float


def _softmax(logits: Sequence[float]) -> List[float]:
    max_logit = max(logits)
    exp_values = [math.exp(v - max_logit) for v in logits]
    total = sum(exp_values)
    return [v / total for v in exp_values]


def _relu(values: Sequence[float]) -> List[float]:
    return [v if v > 0.0 else 0.0 for v in values]


def _random_matrix(rows: int, cols: int, rng: random.Random, scale: float) -> List[List[float]]:
    return [[scale * (rng.random() - 0.5) for _ in range(cols)] for _ in range(rows)]


class MLPPrior:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.layer_sizes = [input_dim] + self.hidden_dims + [output_dim]
        rng = random.Random(seed)
        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = []
        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights.append(_random_matrix(out_dim, in_dim, rng, 1.0 / max(in_dim, 1)))
            self.biases.append([0.0] * out_dim)

    def forward(self, feature: Sequence[float]) -> List[float]:
        activations = list(feature)
        for layer_idx, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            next_values: List[float] = []
            for row, bias in zip(weights, biases):
                z = bias
                for weight, value in zip(row, activations):
                    z += weight * value
                next_values.append(z)
            if layer_idx < len(self.weights) - 1:
                activations = _relu(next_values)
            else:
                activations = _softmax(next_values)
        return activations

    def train_step(
        self,
        feature: Sequence[float],
        target: Sequence[float],
        learning_rate: float = 0.01,
    ) -> List[float]:
        activations: List[List[float]] = [list(feature)]
        pre_activations: List[List[float]] = []

        for layer_idx, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            z_values: List[float] = []
            for row, bias in zip(weights, biases):
                z = bias
                for weight, value in zip(row, activations[-1]):
                    z += weight * value
                z_values.append(z)
            pre_activations.append(z_values)
            if layer_idx < len(self.weights) - 1:
                activations.append(_relu(z_values))
            else:
                activations.append(_softmax(z_values))

        delta = [pred - truth for pred, truth in zip(activations[-1], target)]
        deltas: List[List[float]] = [delta]

        for layer_idx in range(len(self.weights) - 2, -1, -1):
            next_delta = deltas[0]
            next_weights = self.weights[layer_idx + 1]
            curr_pre = pre_activations[layer_idx]
            curr_delta = [0.0] * len(curr_pre)
            for j in range(len(curr_pre)):
                grad = 0.0
                for i in range(len(next_delta)):
                    grad += next_weights[i][j] * next_delta[i]
                curr_delta[j] = grad if curr_pre[j] > 0.0 else 0.0
            deltas.insert(0, curr_delta)

        for layer_idx, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            layer_input = activations[layer_idx]
            layer_delta = deltas[layer_idx]
            for out_idx in range(len(weights)):
                for in_idx in range(len(weights[out_idx])):
                    weights[out_idx][in_idx] -= learning_rate * layer_delta[out_idx] * layer_input[in_idx]
                biases[out_idx] -= learning_rate * layer_delta[out_idx]

        return activations[-1]

    def to_dict(self) -> Dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "output_dim": self.output_dim,
            "weights": self.weights,
            "biases": self.biases,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "MLPPrior":
        model = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dims=list(payload["hidden_dims"]),
            output_dim=int(payload["output_dim"]),
        )
        model.weights = payload["weights"]  # type: ignore[assignment]
        model.biases = payload["biases"]  # type: ignore[assignment]
        return model


class UCBArm:
    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        self.diag_a = [1.0] * feature_dim
        self.b = [0.0] * feature_dim
        self.pulls = 0

    def score(self, feature: Sequence[float], alpha: float) -> float:
        exploit = 0.0
        explore = 0.0
        for i in range(self.feature_dim):
            inv = 1.0 / (self.diag_a[i] + 1e-6)
            exploit += feature[i] * self.b[i] * inv
            explore += feature[i] * feature[i] * inv
        return exploit + alpha * math.sqrt(explore + 1e-6)

    def update(self, feature: Sequence[float], reward: float) -> None:
        for i in range(self.feature_dim):
            self.diag_a[i] += feature[i] * feature[i]
            self.b[i] += reward * feature[i]
        self.pulls += 1

    def to_dict(self) -> Dict[str, object]:
        return {"diag_a": self.diag_a, "b": self.b, "pulls": self.pulls}

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "UCBArm":
        arm = cls(feature_dim=len(payload["diag_a"]))  # type: ignore[arg-type]
        arm.diag_a = list(payload["diag_a"])  # type: ignore[assignment]
        arm.b = list(payload["b"])  # type: ignore[assignment]
        arm.pulls = int(payload["pulls"])
        return arm


class UCBHead:
    def __init__(
        self,
        action_space: Sequence[str],
        feature_dim: int = BANDIT_FEATURE_DIM,
        hidden_dims: Sequence[int] = (256, 256),
        mlp_weight: float = 0.6,
        ucb_alpha: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        self.action_space = list(action_space)
        self.feature_dim = feature_dim
        self.hidden_dims = list(hidden_dims)
        self.mlp_weight = mlp_weight
        self.ucb_alpha = ucb_alpha
        self.prior = MLPPrior(feature_dim, hidden_dims, len(action_space), seed=seed)
        self.arms = [UCBArm(feature_dim) for _ in action_space]
        self.last_feature = [0.0] * feature_dim

    def predict(self, feature: Sequence[float], top_k: int = 3) -> List[HeadPrediction]:
        self.last_feature = list(feature)
        prior_probs = self.prior.forward(feature)
        scored: List[Tuple[float, int, float]] = []
        for idx, arm in enumerate(self.arms):
            ucb_score = arm.score(feature, self.ucb_alpha)
            combined = self.mlp_weight * prior_probs[idx] + (1.0 - self.mlp_weight) * math.tanh(ucb_score)
            scored.append((combined, idx, ucb_score))
        scored.sort(key=lambda item: item[0], reverse=True)
        output: List[HeadPrediction] = []
        for combined, idx, ucb_score in scored[: min(top_k, len(scored))]:
            output.append(
                HeadPrediction(
                    action=self.action_space[idx],
                    score=combined,
                    prior_prob=prior_probs[idx],
                    ucb_score=ucb_score,
                )
            )
        return output

    def train_prior(
        self,
        feature: Sequence[float],
        target_distribution: Sequence[float],
        learning_rate: float = 0.01,
    ) -> List[float]:
        self.last_feature = list(feature)
        return self.prior.train_step(feature, target_distribution, learning_rate=learning_rate)

    def reward(self, action: str, reward_value: float, feature: Optional[Sequence[float]] = None) -> None:
        if action not in self.action_space:
            return
        idx = self.action_space.index(action)
        self.arms[idx].update(feature or self.last_feature, reward_value)

    def to_dict(self) -> Dict[str, object]:
        return {
            "action_space": self.action_space,
            "feature_dim": self.feature_dim,
            "hidden_dims": self.hidden_dims,
            "mlp_weight": self.mlp_weight,
            "ucb_alpha": self.ucb_alpha,
            "prior": self.prior.to_dict(),
            "arms": [arm.to_dict() for arm in self.arms],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "UCBHead":
        head = cls(
            action_space=payload["action_space"],  # type: ignore[arg-type]
            feature_dim=int(payload["feature_dim"]),
            hidden_dims=payload["hidden_dims"],  # type: ignore[arg-type]
            mlp_weight=float(payload["mlp_weight"]),
            ucb_alpha=float(payload["ucb_alpha"]),
        )
        head.prior = MLPPrior.from_dict(payload["prior"])  # type: ignore[arg-type]
        head.arms = [UCBArm.from_dict(item) for item in payload["arms"]]  # type: ignore[arg-type]
        return head


class DualUCBModel:
    def __init__(
        self,
        feature_dim: int = BANDIT_FEATURE_DIM,
        hidden_dims: Sequence[int] = (256, 256),
        ro_actions: Sequence[str] = RO_ACTIONS,
        app_actions: Sequence[str] = APP_ACTIONS,
        seed: Optional[int] = None,
    ) -> None:
        self.feature_dim = feature_dim
        self.hidden_dims = list(hidden_dims)
        self.ro_head = UCBHead(ro_actions, feature_dim, hidden_dims, seed=seed)
        self.app_head = UCBHead(app_actions, feature_dim, hidden_dims, seed=None if seed is None else seed + 1)

    def predict(self, feature: Sequence[float], top_k: int = 3) -> Dict[str, List[HeadPrediction]]:
        return {
            "ro": self.ro_head.predict(feature, top_k=top_k),
            "app": self.app_head.predict(feature, top_k=top_k),
        }

    def train_step(
        self,
        feature: Sequence[float],
        ro_target: Sequence[float],
        app_target: Sequence[float],
        learning_rate: float = 0.01,
    ) -> None:
        self.ro_head.train_prior(feature, ro_target, learning_rate=learning_rate)
        self.app_head.train_prior(feature, app_target, learning_rate=learning_rate)

    def reward(
        self,
        ro_action: Optional[str] = None,
        app_action: Optional[str] = None,
        reward_value: float = 1.0,
        feature: Optional[Sequence[float]] = None,
    ) -> None:
        if ro_action is not None:
            self.ro_head.reward(ro_action, reward_value, feature=feature)
        if app_action is not None:
            self.app_head.reward(app_action, reward_value, feature=feature)

    def save_json(self, path: Path | str) -> None:
        payload = {
            "feature_dim": self.feature_dim,
            "hidden_dims": self.hidden_dims,
            "ro_head": self.ro_head.to_dict(),
            "app_head": self.app_head.to_dict(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path | str) -> "DualUCBModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            feature_dim=int(payload["feature_dim"]),
            hidden_dims=payload["hidden_dims"],
            ro_actions=payload["ro_head"]["action_space"],
            app_actions=payload["app_head"]["action_space"],
        )
        model.ro_head = UCBHead.from_dict(payload["ro_head"])
        model.app_head = UCBHead.from_dict(payload["app_head"])
        return model


def load_cpp_weight_arrays(header_path: Path | str = CPP_HEADER_DEFAULT) -> Dict[str, List[float]]:
    text = Path(header_path).read_text(encoding="utf-8")
    result: Dict[str, List[float]] = {}
    for name in ("DISTILLED_W1", "DISTILLED_B1", "DISTILLED_W2", "DISTILLED_B2"):
        pattern = re.compile(rf"static const float {name}\[\d+\] = \{{(.*?)\}};", re.S)
        match = pattern.search(text)
        if not match:
            raise ValueError(f"{name} not found in {header_path}")
        tokens = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?f?", match.group(1))
        values = [float(token[:-1] if token.endswith("f") else token) for token in tokens]
        result[name] = values
    return result

