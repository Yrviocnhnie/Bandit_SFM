"""v3 models: v2 encoders + per-token category/location embeddings + Markov prior."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


LONG_DT_BUCKETS = 6


@dataclass
class ConfigV3:
    vocab_size: int = 50
    num_categories: int = 11
    num_locations: int = 18
    numeric_dim: int = 28
    profile_dim: int = 38
    app_emb_dim: int = 32
    cat_emb_dim: int = 8
    loc_emb_dim: int = 8
    dt_emb_dim: int = 8
    d_local: int = 64
    d_global_inner: int = 48
    d_global: int = 32
    d_profile: int = 16
    d_fused: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    use_category: bool = True
    use_loc: bool = True
    use_local: bool = True
    use_global: bool = True
    use_profile: bool = True
    use_markov_prior: bool = False


def _pool_last_valid(seq: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    B, K, _ = seq.shape
    any_valid = mask.any(dim=1)
    idx = torch.clamp(mask.float().cumsum(dim=1).argmax(dim=1), 0, K - 1)
    idx = torch.where(any_valid, idx, torch.zeros_like(idx))
    rng = torch.arange(B, device=seq.device)
    out = seq[rng, idx]
    return out * any_valid_to_float(any_valid := any_valid)


def any_valid_to_float(any_valid: torch.Tensor) -> torch.Tensor:
    return any_valid.unsqueeze(-1).float()


class LocalEncoderV3(nn.Module):
    def __init__(self, cfg: ConfigV3, app_emb: nn.Embedding,
                 cat_emb: Optional[nn.Embedding], loc_emb: Optional[nn.Embedding]):
        super().__init__()
        self.cfg = cfg
        self.app_emb = app_emb
        self.cat_emb = cat_emb
        self.loc_emb = loc_emb
        in_dim = app_emb.embedding_dim + cfg.numeric_dim
        if cfg.use_category and cat_emb is not None:
            in_dim += cat_emb.embedding_dim
        if cfg.use_loc and loc_emb is not None:
            in_dim += loc_emb.embedding_dim
        self.proj = nn.Linear(in_dim, cfg.d_local)
        self.ln = nn.LayerNorm(cfg.d_local)
        self.gru = nn.GRU(cfg.d_local, cfg.d_local, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, history_app, history_feat, history_mask,
                history_category=None, history_loc=None):
        parts = [self.app_emb(history_app), history_feat]
        if self.cfg.use_category and self.cat_emb is not None and history_category is not None:
            parts.append(self.cat_emb(history_category))
        if self.cfg.use_loc and self.loc_emb is not None and history_loc is not None:
            parts.append(self.loc_emb(history_loc))
        x = torch.cat(parts, dim=-1)
        x = self.ln(self.proj(x))
        out, _ = self.gru(x)
        return self.dropout(_pool_last_valid(out, history_mask))


class GlobalEncoderV3(nn.Module):
    def __init__(self, cfg: ConfigV3, app_emb, cat_emb, loc_emb):
        super().__init__()
        self.cfg = cfg
        self.app_emb = app_emb
        self.cat_emb = cat_emb
        self.loc_emb = loc_emb
        self.dt_emb = nn.Embedding(LONG_DT_BUCKETS, cfg.dt_emb_dim)
        in_dim = app_emb.embedding_dim + cfg.numeric_dim + cfg.dt_emb_dim
        if cfg.use_category and cat_emb is not None:
            in_dim += cat_emb.embedding_dim
        if cfg.use_loc and loc_emb is not None:
            in_dim += loc_emb.embedding_dim
        self.proj = nn.Linear(in_dim, cfg.d_global_inner)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_global_inner,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_global_inner * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.q = nn.Parameter(torch.randn(cfg.d_global_inner) * 0.01)
        self.out = nn.Linear(cfg.d_global_inner, cfg.d_global)
        self.ln = nn.LayerNorm(cfg.d_global)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, long_app, long_feat, long_mask, long_dt_bin,
                long_category=None, long_loc=None):
        parts = [self.app_emb(long_app), long_feat, self.dt_emb(long_dt_bin)]
        if self.cfg.use_category and self.cat_emb is not None and long_category is not None:
            parts.append(self.cat_emb(long_category))
        if self.cfg.use_loc and self.loc_emb is not None and long_loc is not None:
            parts.append(self.loc_emb(long_loc))
        x = torch.cat(parts, dim=-1)
        x = self.proj(x)
        kpad = ~long_mask
        h = self.transformer(x, src_key_padding_mask=kpad)
        q = self.q.unsqueeze(0).unsqueeze(0)
        scores = (h * q).sum(dim=-1) / (self.cfg.d_global_inner ** 0.5)
        scores = scores.masked_fill(kpad, float("-inf"))
        any_valid = long_mask.any(dim=1)
        scores = torch.where(
            any_valid.unsqueeze(-1).expand_as(scores),
            scores,
            torch.zeros_like(scores),
        )
        weights = torch.softmax(scores, dim=1)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled * any_valid.unsqueeze(-1).float()
        return self.dropout(self.ln(self.out(pooled)))


class ProfileEncoderV3(nn.Module):
    def __init__(self, profile_dim: int, d_profile: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(profile_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, d_profile),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusionV3(nn.Module):
    def __init__(self, cfg: ConfigV3):
        super().__init__()
        self.cfg = cfg
        n_branches = int(cfg.use_local) + int(cfg.use_global) + int(cfg.use_profile)
        assert n_branches > 0, "at least one encoder must be enabled"
        self.n_branches = n_branches
        if cfg.use_local:
            self.proj_local = nn.Linear(cfg.d_local, cfg.d_fused)
        if cfg.use_global:
            self.proj_global = nn.Linear(cfg.d_global, cfg.d_fused)
        if cfg.use_profile:
            self.proj_profile = nn.Linear(cfg.d_profile, cfg.d_fused)
        gate_in = 0
        if cfg.use_local:
            gate_in += cfg.d_local
        if cfg.use_global:
            gate_in += cfg.d_global
        if cfg.use_profile:
            gate_in += cfg.d_profile
        self.gate = nn.Linear(gate_in, n_branches)
        self.ln = nn.LayerNorm(cfg.d_fused)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, h_local=None, h_global=None, h_profile=None):
        raws, projs = [], []
        if h_local is not None:
            raws.append(h_local)
            projs.append(self.proj_local(h_local))
        if h_global is not None:
            raws.append(h_global)
            projs.append(self.proj_global(h_global))
        if h_profile is not None:
            raws.append(h_profile)
            projs.append(self.proj_profile(h_profile))
        raw_cat = torch.cat(raws := raws, dim=-1) if False else torch.cat(raws, dim=-1)
        gate = torch.softmax(self.gate(raw_cat), dim=-1)
        stacked = torch.stack(projs, dim=1)
        fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)
        return self.dropout(self.ln(fused))


class TaskAModelV3(nn.Module):
    def __init__(self, cfg: ConfigV3):
        super().__init__()
        self.cfg = cfg
        self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(cfg.num_categories, cfg.cat_emb_dim, padding_idx=0) if cfg.use_category else None
        self.loc_emb = nn.Embedding(cfg.num_locations, cfg.loc_emb_dim, padding_idx=0) if cfg.use_loc else None
        if cfg.use_local:
            self.local_enc = LocalEncoderV3(cfg, self.app_emb, self.cat_emb, self.loc_emb)
        if cfg.use_global:
            self.global_enc = GlobalEncoderV3(cfg, self.app_emb, self.cat_emb, self.loc_emb)
        if cfg.use_profile:
            self.profile_enc = ProfileEncoderV3(cfg.profile_dim, cfg.d_profile, cfg.dropout)
        self.fusion = GatedFusionV3(cfg)
        self.head = nn.Linear(cfg.d_fused, cfg.vocab_size)

    def forward(self, batch):
        h_local = h_global = h_profile = None
        if self.cfg.use_local:
            h_local = self.local_enc(
                batch["history_app"], batch["history_feat"], batch["history_mask"],
                history_category=batch.get("history_category") if self.cfg.use_category else None,
                history_loc=batch.get("history_loc") if self.cfg.use_loc else None,
            )
        if self.cfg.use_global:
            h_global = self.global_enc(
                batch["long_app"], batch["long_feat"], batch["long_mask"], batch["long_dt_bin"],
                long_category=batch.get("long_category") if self.cfg.use_category else None,
                long_loc=batch.get("long_loc") if self.cfg.use_loc else None,
            )
        if self.cfg.use_profile:
            h_profile = self.profile_enc(batch["profile"])
        h = self.fusion(h_local=h_local, h_global=h_global, h_profile=h_profile)
        return self.head(h)


class TaskBModelV3(nn.Module):
    def __init__(self, cfg: ConfigV3, markov_log_prior: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(cfg.num_categories, cfg.cat_emb_dim, padding_idx=0) if cfg.use_category else None
        self.loc_emb = nn.Embedding(cfg.num_locations, cfg.loc_emb_dim, padding_idx=0) if cfg.use_loc else None
        if cfg.use_local:
            self.local_enc = LocalEncoderV3(cfg, self.app_emb, self.cat_emb, self.loc_emb)
        if cfg.use_global:
            self.global_enc = GlobalEncoderV3(cfg, self.app_emb, self.cat_emb, self.loc_emb)
        if cfg.use_profile:
            self.profile_enc = ProfileEncoderV3(cfg.profile_dim, cfg.d_profile, cfg.dropout)
        self.fusion = GatedFusionV3(cfg)
        self.head_sig = nn.Linear(cfg.d_fused, cfg.vocab_size)
        self.head_rate = nn.Linear(cfg.d_fused, cfg.vocab_size)
        self.use_markov_prior = bool(cfg.use_markov_prior)
        if self.use_markov_prior and markov_log_prior is not None:
            self.register_buffer("markov_log_prior", markov_log_prior.float())
            self.alpha_markov = nn.Parameter(torch.tensor(0.5))
        else:
            self.markov_log_prior = None
            self.alpha_markov = None

    def forward(self, batch):
        h_local = h_global = h_profile = None
        if self.cfg.use_local:
            h_local = self.local_enc(
                batch["history_app"], batch["history_feat"], batch["history_mask"],
                history_category=batch.get("history_category") if self.cfg.use_category else None,
                history_loc=batch.get("history_loc") if self.cfg.use_loc else None,
            )
        if self.cfg.use_global:
            h_global = self.global_enc(
                batch["long_app"], batch["long_feat"], batch["long_mask"], batch["long_dt_bin"],
                long_category=batch.get("long_category") if self.cfg.use_category else None,
                long_loc=batch.get("long_loc") if self.cfg.use_loc else None,
            )
        if self.cfg.use_profile:
            h_profile = self.profile_enc(batch["profile"])
        fused = self.fusion(h_local=h_local, h_global=h_global, h_profile=h_profile)
        sig = self.head_sig(fused)
        rate = self.head_rate(fused)
        if self.use_markov_prior and self.markov_log_prior is not None:
            last_app = batch["last_app_idx"].long()
            prior = self.markov_log_prior[last_app]
            alpha = torch.clamp(self.alpha_markov, 0.0, 2.0)
            sig_logits = sig + alpha * prior
        else:
            sig_logits = sig
        return {"logits_b_sig": sig_logits, "log_rate_b": rate}


# rebuild TaskAModelV3 cleanly — previous instance was OK but we lost local_enc assignment
class TaskAModelV3(nn.Module):  # noqa: F811
    def __init__(self, cfg: ConfigV3):
        super().__init__()
        self.cfg = cfg
        self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(cfg.num_categories, cfg.cat_emb_dim, padding_idx=0) if cfg.use_category else None
        self.loc_emb = nn.Embedding(cfg.num_locations, cfg.loc_emb_dim, padding_idx=0) if cfg.use_loc else None
        if cfg.use_local:
            self.local_enc = LocalEncoderV3(cfg, self.app_emb, self.cat_emb, self.loc_emb)
        if cfg.use_global:
            self.global_enc = GlobalEncoderV3(cfg, self.app_emb, self.cat_emb, self.loc_emb)
        if cfg.use_profile:
            self.profile_enc = ProfileEncoderV3(cfg.profile_dim, cfg.d_profile, cfg.dropout)
        self.fusion = GatedFusionV3(cfg)
        self.head = nn.Linear(cfg.d_fused, cfg.vocab_size)

    def forward(self, batch):
        h_local = h_global = h_profile = None
        if self.cfg.use_local:
            h_local = self.local_enc(
                batch["history_app"],
                batch["history_feat"],
                batch["history_mask"],
                history_category=batch.get("history_category") if self.cfg.use_category else None,
                history_loc=batch.get("history_loc") if self.cfg.use_loc else None,
            )
        if self.cfg.use_global:
            h_global = self.global_enc(
                batch["long_app"], batch["long_feat"], batch["long_mask"], batch["long_dt_bin"],
                long_category=batch.get("long_category") if self.cfg.use_category else None,
                long_loc=batch.get("long_loc") if self.cfg.use_loc else None,
            )
        if self.cfg.use_profile:
            h_profile = self.profile_enc(batch["profile"])
        fused = self.fusion(h_local=h_local, h_global=h_global, h_profile=h_profile)
        return self.head(fused)


def make_task_a(cfg: ConfigV3) -> TaskAModelV3:
    return TaskAModelV3(cfg)


def make_task_b(cfg: ConfigV3, markov_log_prior: Optional[torch.Tensor] = None) -> TaskBModelV3:
    return TaskBModelV3(cfg, markov_log_prior=markov_log_prior)
