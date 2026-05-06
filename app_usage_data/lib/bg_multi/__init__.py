"""Multi-user extension of Task C (background-app suspension prediction).

The single-user pipeline lives in `lib/bg/`; this package adds:
  - pooled-vocab construction across users
  - per-user feature-stat fitting (fed into the model as features at scoring time)
  - global-anchor-id helpers so listwise loss / metric groupbys don't conflate users

The model is *globally trained on pooled bg rows* but *personalized via features*
sourced from per-user stats — see `REPORT_bgkill_multiuser.md`.
"""
from __future__ import annotations

from .helpers import (
    GLOBAL_ANCHOR_OFFSET,
    attach_user_columns,
    build_pooled_vocab,
    fit_per_user_stats,
    list_cohort_users,
    load_per_user_stats,
    make_global_anchor_id,
    save_per_user_stats,
    stack_per_user,
)

__all__ = [
    "GLOBAL_ANCHOR_OFFSET",
    "attach_user_columns",
    "build_pooled_vocab",
    "fit_per_user_stats",
    "list_cohort_users",
    "load_per_user_stats",
    "make_global_anchor_id",
    "save_per_user_stats",
    "stack_per_user",
]
