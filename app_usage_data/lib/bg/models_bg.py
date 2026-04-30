"""Task-C learned models — per-(anchor, app) MLP with dual sigmoid heads.

Two model variants are exposed:

  * ``BgPairMLP`` (legacy, used by C1)
        16-d app embedding + 4-d cat embedding + 14 numeric features
        → MLP → two sigmoid heads (H=5, H=10).

  * ``BgPairMLPv2`` (new, used by C2)
        Drops the cat embedding (redundant with markov_prob + hour_cond_prob).
        Uses a 15-feature schema chosen to be Task-C-relevant only — see
        ``FEATURE_NAMES_V2``. App embedding can be toggled via cfg.use_app_emb.

Both are trained with BCE + pos_weight on the binary "app foregrounded in
(t, t+H]" labels, restricted to apps in B(t).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== C1 (legacy) =====================
FEATURE_NAMES = (
    "log_time_in_bg",
    "log_time_since_fg",
    "log_fg_count_today",
    "age_bg_over_6h",
    "hour_sin_24",
    "hour_cos_24",
    "wday_sin",
    "wday_cos",
    "log_bg_set_size",
    "is_weekend",
    "loc_match_flag",
    "daypart_match_flag",
    "markov_prob",
    "hour_cond_prob",
)
NUM_FEATURES = len(FEATURE_NAMES)


@dataclass
class BgMLPConfig:
    vocab_size: int = 50
    num_categories: int = 11
    app_emb_dim: int = 16
    cat_emb_dim: int = 4
    num_features: int = NUM_FEATURES
    d_hidden: int = 64
    dropout: float = 0.2


class BgPairMLP(nn.Module):
    """Legacy C1 model. Kept for reproducibility of the published numbers."""

    def __init__(self, cfg: BgMLPConfig):
        super().__init__()
        self.cfg = cfg
        self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(cfg.num_categories, cfg.cat_emb_dim, padding_idx=0)
        in_dim = cfg.app_emb_dim + cfg.cat_emb_dim + cfg.num_features
        d_h = cfg.d_hidden
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, d_h),
            nn.ReLU(),
            nn.LayerNorm(d_h),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.LayerNorm(d_h // 2),
            nn.Dropout(cfg.dropout),
        )
        self.head_5 = nn.Linear(d_h // 2, 1)
        self.head_10 = nn.Linear(d_h // 2, 1)

    def forward(self, app_idx, cat_idx, features):
        ae = self.app_emb(app_idx)
        ce = self.cat_emb(cat_idx)
        x = torch.cat([ae, ce, features], dim=-1)
        h = self.trunk(x)
        return {
            "logit_5": self.head_5(h).squeeze(-1),
            "logit_10": self.head_10(h).squeeze(-1),
        }


# ===================== C2 (new) =====================
# Feature selection rationale (see REPORT_bgkill_v2.md §4):
#   Per-app recency        : log_time_in_bg, log_time_since_fg, recency_rank_in_bg
#   Per-app rhythm         : log_fg_count_today, log_fg_count_last_1h, log_fg_count_last_6h
#   Per-app priors         : markov_prob, hour_cond_prob
#   Anchor context         : bg_recency_min_norm, time_since_screen_on_norm,
#                            prev_killed_app_match
#   Time-of-day rhythm     : hour_sin_24, hour_cos_24, daypart_match_flag, is_weekend

FEATURE_NAMES_V2 = (
    "log_time_in_bg",
    "log_time_since_fg",
    "recency_rank_in_bg",
    "log_fg_count_today",
    "log_fg_count_last_1h",
    "log_fg_count_last_6h",
    "markov_prob",
    "hour_cond_prob",
    "bg_recency_min_norm",
    "time_since_screen_on_norm",
    "prev_killed_app_match",
    "hour_sin_24",
    "hour_cos_24",
    "daypart_match_flag",
    "is_weekend",
)
NUM_FEATURES_V2 = len(FEATURE_NAMES_V2)


@dataclass
class BgMLPv2Config:
    vocab_size: int = 50
    app_emb_dim: int = 16
    use_app_emb: bool = True
    num_features: int = NUM_FEATURES_V2
    d_hidden: int = 64
    dropout: float = 0.2


class BgPairMLPv2(nn.Module):
    """C2 model. Tabular MLP on the v2 feature schema; optional app embedding."""

    def __init__(self, cfg: BgMLPv2Config):
        super().__init__()
        self.cfg = cfg
        if cfg.use_app_emb:
            self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        else:
            self.app_emb = None
        in_dim = cfg.num_features + (cfg.app_emb_dim if cfg.use_app_emb else 0)
        d_h = cfg.d_hidden
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, d_h),
            nn.ReLU(),
            nn.LayerNorm(d_h),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.LayerNorm(d_h // 2),
            nn.Dropout(cfg.dropout),
        )
        self.head_5 = nn.Linear(d_h // 2, 1)
        self.head_10 = nn.Linear(d_h // 2, 1)

    def forward(self, app_idx, features):
        parts = [features]
        if self.app_emb is not None:
            parts = [self.app_emb(app_idx), features]
        x = torch.cat(parts, dim=-1)
        h = self.trunk(x)
        return {
            "logit_5": self.head_5(h).squeeze(-1),
            "logit_10": self.head_10(h).squeeze(-1),
        }


def bce_dual(out, y_5, y_10, pos_weight_5=None, pos_weight_10=None):
    b5 = F.binary_cross_entropy_with_logits(out["logit_5"], y_5.float(),
                                             pos_weight=pos_weight_5)
    b10 = F.binary_cross_entropy_with_logits(out["logit_10"], y_10.float(),
                                              pos_weight=pos_weight_10)
    return b5 + b10, b5.detach(), b10.detach()
