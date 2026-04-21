"""Triplet-VAE model and mixed-loss computation for the surprise-trigger demo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_encoder import ENCODING


@dataclass
class TripletVAEConfig:
    input_dim: int
    latent_dim: int = 32
    hidden1: int = 128
    hidden2: int = 64
    dropout: float = 0.1
    triplet_margin: float = 1.0


class TripletVAE(nn.Module):
    def __init__(self, cfg: TripletVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden1),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden1, cfg.hidden2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(cfg.hidden2, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.hidden2, cfg.latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden2),
            nn.ReLU(),
            nn.Linear(cfg.hidden2, cfg.hidden1),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden1, cfg.input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, z


def mixed_recon_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-group reconstruction loss summed over groups, averaged over batch.

    logits: (B, D)   -- raw outputs of the decoder
    target: (B, D)   -- the target encoded vector
    """
    losses = []
    for g in ENCODING.groups:
        logit_block = logits[:, g.offset : g.offset + g.size]
        target_block = target[:, g.offset : g.offset + g.size]
        if g.kind == "onehot":
            # target is one-hot; argmax gives the class index
            class_idx = target_block.argmax(dim=1)
            ce = F.cross_entropy(logit_block, class_idx, reduction="mean")
            losses.append(ce)
        elif g.kind == "binary":
            bce = F.binary_cross_entropy_with_logits(
                logit_block.squeeze(-1), target_block.squeeze(-1), reduction="mean"
            )
            losses.append(bce)
        elif g.kind == "scalar":
            pred = torch.sigmoid(logit_block)
            mse = F.mse_loss(pred, target_block, reduction="mean")
            losses_scalar_weight = 1.0
            losses.append(mse * losses_scalar_weight)
        else:
            raise ValueError(f"unknown group kind: {g.kind}")
    return torch.stack(losses).sum()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL( N(mu, sigma^2) || N(0, I) )  averaged over batch
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def triplet_loss(
    mu_anchor: torch.Tensor,
    mu_positive: torch.Tensor,
    mu_negative: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    d_ap = (mu_anchor - mu_positive).pow(2).sum(dim=1)
    d_an = (mu_anchor - mu_negative).pow(2).sum(dim=1)
    return F.relu(d_ap - d_an + margin).mean()
