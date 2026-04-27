"""Neural sequence models for next-app prediction.

Two architectures:
- GRUModel: embed -> GRU -> MLP -> dual heads (softmax / sigmoid+poisson)
- TGTLiteModel: embed -> Transformer with Fourier-hour gating -> dual heads
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Fnn


@dataclass
class ModelConfig:
    vocab_size: int
    numeric_dim: int          # NUMERIC_FEAT_DIM from features.py
    d_model: int = 64
    app_emb_dim: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    history_k: int = 16


def token_project_input_dim(cfg: ModelConfig) -> int:
    return cfg.app_emb_dim + cfg.numeric_dim


class _EmbedProject(nn.Module):
    """Combine app embedding + numeric features -> d_model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        in_dim = cfg.app_emb_dim + cfg.numeric_dim
        self.proj = nn.Linear(in_dim, cfg.d_model)

    def forward(self, app_idx: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        e = self.app_emb(app_idx)
        x = torch.cat([e, feat], dim=-1)
        return self.proj(x)


# Need to update ModelConfig to carry app_emb_dim
@dataclass
class ModelConfig:
    vocab_size: int
    numeric_dim: int
    d_model: int = 64
    app_emb_dim: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    history_k: int = 16


class TokenEncoder(nn.Module):
    """app embedding + numeric features -> d_model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        self.proj = nn.Linear(cfg.app_emb_dim + cfg.numeric_dim, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, app_idx: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        e = self.embedding_then_concat(app_idx, feat)
        return self.ln(self.proj(e))

    def embedding_then_concat(self, app_idx, feat):
        emb = self.app_emb(app_idx)
        return torch.cat([emb, feat], dim=-1)


class DualHeads(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.head_a = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.head_b_sigmoid = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.head_b_poisson = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, h_final: torch.Tensor):
        # h_final (B, d_model)
        return {
            "logits_a": self.head_a(h_final),
            "logits_b_sig": self.head_b_sigmoid(h_final),
            "log_rate_b": self.head_b_poisson(h_final),  # interpret as log-rate
        }


class GRUModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = TokenEncoder(cfg)
        self.gru = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.heads = nn.ModuleDict({
            "a": nn.Linear(cfg.d_model, cfg.vocab_size),
            "b_sig": nn.Linear(cfg.d_model, cfg.vocab_size),
            "b_poisson": nn.Linear(cfg.d_model, cfg.vocab_size),
        })

    def forward(self, app_idx, feat, mask, side_hour_fourier=None):
        """app_idx (B, K) int64; feat (B, K, numeric_dim); mask (B, K) bool.
        side_hour_fourier (B, 4) at target/anchor time (unused by GRU, kept for API parity).
        """
        x = self.proj_input(app_idx, feat)
        out, h = self.gru(x)
        # Use last token representation where mask is valid, else last position
        last = self.pick_last_valid(out, mask)
        last = self.dropout(last)
        return {
            "logits_a": self.heads["a"](last),
            "logits_b_sig": self.heads["b_sig"](last),
            "log_rate_b": self.heads["b_poisson"](last),
            "hidden": last,
        }

    def proj_input(self, app_idx, feat):
        return self.enc(app_idx, feat)

    def pick_last_valid(self, out, mask):
        # mask (B, K) where True = valid
        # Take the last True index per row; if none, take position -1
        B, K, D = out.shape
        any_valid = mask.any(dim=1)
        # last True index
        idx_last = mask.float().cumsum(dim=1).argmax(dim=1)  # pos of last True
        # but cumsum argmax of all-zeros returns 0, which we interpret as last pos
        idx_last = torch.where(any_valid, idx_last, torch.full_like(idx_last, K - 1))
        return out[torch.arange(B, device=out.device), idx_last]


class TGTLite(nn.Module):
    """Transformer with Fourier-hour temporal gating applied to the pooled session rep."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = TokenEncoder(cfg)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        # Temporal gate: FourierHour (4 dims) -> d_model -> sigmoid gate
        self.gate = nn.Sequential(
            nn.Linear(4, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.heads = _Heads(cfg)

    def forward(self, app_idx, feat, mask, side_hour_fourier):
        x = self.enc(app_idx, feat)
        # src_key_padding_mask: True where PAD (invalid). Our mask is True where valid.
        kpad = ~mask
        h = self.trf(x, src_key_padding_mask=kpad)
        # Pool: take last valid token; if none, zeros
        pooled = self._pool_last(h, mask)
        gate = self.gate(side_hour_fourier)
        pooled = pooled * gate
        pooled = self.dropout(pooled)
        return self.heads(pooled)

    @property
    def trf(self):
        return self.transformer_encoder if hasattr(self, "transformer_encoder") else self.transformer

    def _pool_last(self, h, mask):
        B, K, D = h.shape
        any_valid = mask.any(dim=1)
        idx_last = mask.float().cumsum(dim=1).argmax(dim=1)
        idx_last = torch.where(any_valid, idx_last, torch.full_like(idx_last, K - 1))
        out = h[torch.arange(B, device=h.device), idx_last]
        # Zero out rows that have no valid tokens at all
        out = out * any_valid.unsqueeze(-1).float()
        return out


class _Heads(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.a = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.b_sig = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.b_poisson = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, h):
        return {
            "logits_a": self.a(h),
            "logits_b_sig": self.b_sig(h),
            "log_rate_b": self.b_poisson(h),
            "hidden": h,
        }


# =================== Losses ===================
def task_a_loss(logits_a: torch.Tensor, target: torch.Tensor, class_weight=None, label_smoothing: float = 0.05) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits_a, target, weight=class_weight, label_smoothing=label_smoothing)


def task_b_loss(logits_sig: torch.Tensor, log_rate: torch.Tensor, target_counts: torch.Tensor, class_weight=None) -> torch.Tensor:
    """target_counts (B, V) non-negative integer counts of apps in window."""
    # BCE on binary indicator
    y_bin = (target_counts > 0).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits_sig, y_bin, weight=class_weight)
    # Poisson NLL: y * log_rate - rate (plus const)
    rate = torch.exp(torch.clamp(log_rate, max=8.0))
    pois = (rate - target_counts * log_rate).mean()
    return bce * 0.5 + pois * 0.2


# Default config factory
def make_config(vocab_size: int, numeric_dim: int, history_k: int = 16) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size, numeric_dim=numeric_dim, history_k=history_k)
