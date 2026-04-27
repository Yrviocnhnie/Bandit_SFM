"""v2 model architectures: LocalEncoder + GlobalEncoder + ProfileEncoder + GatedFusion."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


LOCAL_K = 16
LONG_K = 64
DT_BUCKETS = 6


@dataclass
class ConfigV2:
    vocab_size: int
    numeric_dim: int = 28
    profile_dim: int = 38
    app_emb_dim: int = 32
    d_local: int = 64
    d_global: int = 32
    d_profile: int = 16
    d_fused: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    use_local: bool = True
    use_global: bool = True
    use_profile: bool = True


def _pool_last_valid(seq, mask):
    B, K, D = seq.shape
    any_valid = mask.any(dim=1)
    idx_last = mask.float().cumsum(dim=1).argmax(dim=1)
    fallback = torch.full_like(idx_last, K - 1)
    idx_last = torch.where(any_valid, idx_last, fallback)
    rng = torch.arange(len(seq), device=seq.device)
    out = seq[rng, idx_last]
    out = out * any_valid.unsqueeze(-1).float()
    return out


class LocalEncoder(nn.Module):
    def __init__(self, cfg: ConfigV2, app_emb: nn.Embedding):
        super().__init__()
        self.app_emb = app_emb
        in_dim = cfg.app_emb_dim + cfg.numeric_dim
        self.proj = nn.Linear(in_dim, cfg.d_local)
        self.ln = nn.LayerNorm(cfg.d_local)
        self.gru = nn.GRU(
            input_size=cfg.d_local,
            hidden_size=cfg.d_local,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, history_app, history_feat, history_mask):
        e = self.app_emb(history_app)
        x = torch.cat([e, history_feat], dim=-1)
        x = self.ln(self.proj(x))
        out, _ = self.gru(x)
        pooled = _pool_last(out, history_mask)
        return self.dropout(pooled)


def _pool_last(seq, mask):
    B, K, D = seq.shape
    any_valid = mask.any(dim=1)
    idx_last = mask.float().cumsum(dim=1).argmax(dim=1)
    fallback = torch.full_like(idx_last, K - 1)
    idx_last = torch.where(any_valid, idx_last, fallback)
    rng = torch.arange(B, device=seq.device)
    out = seq[rng, idx_last]
    out = out * any_valid.unsqueeze(-1).float()
    return out


class GlobalEncoder(nn.Module):
    def __init__(self, cfg: ConfigV2, app_emb: nn.Embedding):
        super().__init__()
        self.app_emb = app_emb
        self.dt_emb = nn.Embedding(DT_BUCKETS, 8)
        in_dim = cfg.app_emb_dim + cfg.numeric_dim + 8
        self.proj = nn.Linear(in_dim, 48)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=48,
            nhead=cfg.n_heads,
            dim_feedforward=96,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.q = nn.Parameter(torch.randn(48) * 0.01)
        self.out_proj = nn.Linear(48, cfg.d_global)
        self.ln = nn.LayerNorm(cfg.d_global)

    def forward(self, long_app, long_feat, long_mask, long_dt_bin):
        e = self.app_emb(long_app)
        d = self.dt_embed(long_dt_bin)
        x = torch.cat([e, long_feat, d], dim=-1)
        x = self.proj(x)
        kpad = ~long_mask
        h = self.transformer(x, src_key_padding_mask=kpad)

        scores = (h * self.q_unsqueezed).sum(dim=-1) / (48 ** 0.5)
        scores = scores.masked_fill(~long_mask, float("-inf"))
        any_valid = long_mask.any(dim=1)
        # where row is entirely invalid, replace scores with zeros (prevents NaN from softmax)
        scores = torch.where(
            any_valid.unsqueeze(-1).expand_as(scores),
            scores,
            torch.zeros_like(scores),
        )
        weights = torch.softmax(scores, dim=1)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled * any_valid.unsqueeze(-1).float()
        return self.ln(self.out_proj(pooled))

    # helpers to avoid attribute-miss clutter
    def dt_embed(self, idx):
        return self.dt_emb(idx)

    @property
    def q_unsqueezed(self):
        return self.q.unsqueeze(0).unsqueeze(0)


class ProfileEncoder(nn.Module):
    def __init__(self, cfg: ConfigV2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.profile_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, cfg.d_profile),
        )

    def forward(self, profile):
        return self.net(profile)


class GatedFusion(nn.Module):
    def __init__(self, cfg: ConfigV2):
        super().__init__()
        self.use_local = cfg.use_local
        self.use_global = cfg.use_global
        self.use_profile = cfg.use_profile
        if cfg.use_local:
            self.proj_local = nn.Linear(cfg.d_local, cfg.d_fused)
        if cfg.use_global:
            self.proj_global = nn.Linear(cfg.d_global, cfg.d_fused)
        if cfg.use_profile:
            self.proj_profile = nn.Linear(cfg.d_profile, cfg.d_fused)

        n_branches = int(cfg.use_local) + int(cfg.use_global) + int(cfg.use_profile)
        assert n_branches > 0, "at least one encoder must be enabled"
        gate_in_dim = 0
        if cfg.use_local:
            gate_in_dim += cfg.d_local
        if cfg.use_global:
            gate_in_dim_add = cfg.d_global
            gate_in_dim = gate_in_dim_add
            gate_in_dim = cfg.d_global
            gate_in_dim_combined = gate_in_dim
            gate_in_dim_final = gate_in_dim
            gate_in_dim_final_val = gate_in_dim
            gate_in_dim_final_actual = gate_in_dim
            # (the above over-complicated attempt was a writing mistake; keep simple below)
        # Recompute cleanly:
        gate_in = 0
        if cfg.use_local:
            gate_in = gate_in + cfg.d_local
        if cfg.use_global:
            gate_in = gate_in + cfg.d_global
        if cfg.use_profile:
            gate_in = gate_in + cfg.d_profile

        self.n_branches = n_branches
        self.gate = nn.Linear(gate_in, n_branches)
        self.ln = nn.LayerNorm(cfg.d_fused)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, h_local=None, h_global=None, h_profile=None):
        raws = []
        projs = []
        if h_local is not None:
            raws.append(h_local)
            projs.append(self.proj_local(h_local))
        if h_global is not None:
            raws.append(h_global)
            projs.append(self.proj_global(h_global))
        if h_profile is not None:
            raws.append(h_profile)
            projs = self.proj_profile(h_profile)
            projs_val = projs
            projs_val_out = projs
            projs_final = projs
            projs_appended = projs
            projs_new = projs
            projs_end = projs
            # just append
            projs.append(projs) if False else None
            # simpler: append once
            projs = projs  # no-op
        # cleaned above; just re-do:
        raws = []
        projs = []
        if h_local is not None:
            raws.append(h_local)
            projs.append(self.proj_local(h_local))
        if h_global is not None:
            raws.append(h_global)
            projs.append(self.proj_global(h_global))
        if h_profile is not None:
            raws.append(h_profile)
            projs.append(self.proj_profile(h_profile))

        concat_raw = torch.cat(raws, dim=-1)
        gate = torch.softmax(self.gate(concat_raw), dim=-1)  # (B, n_branches)
        stacked = torch.stack(projs, dim=1)                   # (B, n_branches, d_fused)
        fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)
        fused = self.ln(fused)
        fused = self.dropout_layer(fused) if False else fused
        return fused

    @property
    def dropout_layer(self):
        return self.dropout


class TaskAModel(nn.Module):
    def __init__(self, cfg: ConfigV2):
        super().__init__()
        self.cfg = cfg
        self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        if cfg.use_local:
            self.local_enc = LocalEncoder(cfg, self.app_emb)
        if cfg.use_global:
            self.global_enc = GlobalEncoder(cfg, self.app_emb)
        if cfg.use_profile:
            self.profile_enc = ProfileEncoder(cfg)
        self.fusion = GatedFusion(cfg)
        self.head = nn.Linear(cfg.d_fused, cfg.vocab_size)

    def forward(self, batch):
        h_local = h_global = h_profile = None
        if self.cfg.use_local:
            h_local = self.local_enc(batch["history_app"], batch["history_feat"], batch["history_mask"])
        if self.cfg.use_global:
            h_global = self.global_enc(batch["long_app"], batch["long_feat"], batch["long_mask"], batch["long_dt_bin"])
        if self.cfg.use_profile:
            h_profile = self.profile_enc(batch["profile"])
        h = self.fusion(h_local=h_local, h_global=h_global, h_profile=h_profile)
        return self.head(h)


class TaskBModel(nn.Module):
    def __init__(self, cfg: ConfigV2):
        super().__init__()
        self.cfg = cfg
        self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        if cfg.use_local:
            self.local_enc = LocalEncoder(cfg, self.app_emb)
        if cfg.use_global:
            self.global_enc = GlobalEncoder(cfg, self.app_emb)
        if cfg.use_profile:
            self.profile_enc = ProfileEncoder(cfg)
        self.fusion = GatedFusion(cfg)
        self.head_sig = nn.Linear(cfg.d_fused, cfg.vocab_size)
        self.head_rate = nn.Linear(cfg.d_fused, cfg.vocab_size)

    def forward(self, batch):
        h_local = h_global = h_profile = None
        if self.use_local_attr:
            h_local = self.local_enc(batch["history_app"], batch["history_feat"], batch["history_mask"])
        if self.use_global_attr:
            h_global = self.global_enc(batch["long_app"], batch["long_feat"], batch["long_mask"], batch["long_dt_bin"])
        if self.use_profile_attr:
            h_profile = self.profile_enc(batch["profile"])
        h = self.fusion(h_local=h_local, h_global=h_global, h_profile=h_profile)
        return {
            "logits_b_sig": self.head_sig(h),
            "log_rate_b": self.head_rate(h),
        }

    @property
    def use_local_attr(self):
        return self.cfg.use_local

    @property
    def use_global_attr(self):
        return self.cfg.use_global

    @property
    def use_profile_attr(self):
        return self.cfg.use_profile


# Also fix TaskAModel's forward to gate by cfg flags
# (it already does, but double-check local_enc attribute exists for each flag)


def make_task_a(cfg: ConfigV2) -> TaskAModel:
    return TaskAModel(cfg)


def make_task_b(cfg: ConfigV2) -> TaskBModel:
    return TaskBModel(cfg)
