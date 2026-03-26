"""Shared-action disjoint LinUCB with an optional default-action prior."""

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
