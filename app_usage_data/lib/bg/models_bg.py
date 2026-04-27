"""Task-C learned model: per-(anchor, app) MLP with dual sigmoid heads.

The data layout for this model is naturally per-pair (one row per (anchor,
app ∈ B(t)) in the parquet), so the model is a straightforward tabular
classifier consuming:
  - `app_idx`  -> 16-d learnable embedding
  - `cat_idx`  -> 4-d learnable embedding
  - 14 continuous features (see FEATURE_NAMES below)
  -> MLP -> two sigmoid heads (H=5, H=10).

Loss: BCE with `pos_weight` for the ~5% positive rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_NAMES = (
    "log_time_in_bg",         # log1p(time_in_bg_sec)
    "log_time_since_fg",      # log1p(max(0, time_since_fg_sec))
    "log_fg_count_today",     # log1p(fg_count_today)
    "age_bg_over_6h",         # time_in_bg_sec / (6*3600), clipped to [0, 2]
    "hour_sin_24",            # sin(2π hour / 24)
    "hour_cos_24",            # cos(2π hour / 24)
    "wday_sin",               # sin(2π weekday / 7)
    "wday_cos",               # cos(2π weekday / 7)
    "log_bg_set_size",        # log1p(bg_set_size)
    "is_weekend",             # 1 if weekday >= 5
    "loc_match_flag",         # last_fg_loc_id == current loc_id
    "daypart_match_flag",     # last_fg_daypart == current daypart
    "markov_prob",            # P(app | last_fg_app) from v3 Markov prior
    "hour_cond_prob",         # P(app | anchor hour) from HourMFU train fit
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


def bce_dual(out, y_5, y_10, pos_weight_5=None, pos_weight_10=None):
    b5 = F.binary_cross_entropy_with_logits(out["logit_5"], y_5.float(), pos_weight=pos_weight_5)
    b10 = F.binary_cross_entropy_with_logits(out["logit_10"], y_10.float(), pos_weight=pos_weight_10)
    return b5 + b10, b5.detach(), b10.detach()


FEATURE_NAMES = FEATURE_NAMES if 'FEATURE_NAMES' in dir() else FEATURE_NAMES  # export


FEATURE_NAMES = FEATURE_NAMES
