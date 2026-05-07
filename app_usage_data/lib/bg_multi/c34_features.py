"""c3.4 feature fitters (per-user and global).

All per-user fits read TRAIN events only and shape outputs against the
pooled vocab. Global fits read the pooled bg_train (or pooled train events)
and return global priors usable across users.

Functions exposed:
  - fit_dow_hour_freq            (F3) per-user (7, 24, V) DOW × hour table
  - fit_app_popularity_bucket    (F5) global (V,) bucket {0=rare, 1=medium, 2=common}
  - fit_cross_user_app_prior     (F6) leave-one-out per-user (V,) priors
  - compute_last2_fg_for_anchors (F1 prep) anchor → 2nd-most-recent FG app idx
  - fit_two_step_markov_per_user (F1) sparse {(a_{t-2}, a_{t-1}) → (V,) probs}
  - fit_co_fg_matrix_per_user    (F2) per-user (V, V) co-FG matrix
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# F3 — per-user day-of-week × hour conditional probability
# ============================================================================
def fit_dow_hour_freq(train_events: pd.DataFrame,
                       vocab: Dict[str, int]) -> np.ndarray:
    """P(app | day_of_week, hour) per-user table.

    Returns: (7, 24, V) float32 array. Reserved columns (<PAD>/<UNK>/<RARE>)
    are zeroed; every (dow, hour) row over real apps sums to 1.0 (Dirichlet-
    smoothed with alpha=0.5).
    """
    V = len(vocab)
    rare = vocab.get("<RARE>", 2)
    table = np.zeros((7, 24, V), dtype=np.float64)
    mask = train_events["is_target_event"].astype(bool).to_numpy()
    ts = pd.to_datetime(train_events.loc[mask, "event_ts"])
    dows = ts.dt.weekday.to_numpy()
    hrs = ts.dt.hour.to_numpy()
    apps = train_events.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    for d, h, a in zip(dows, hrs, idx):
        if 0 <= d < 7 and 0 <= h < 24 and 3 <= a < V:
            table[int(d), int(h), int(a)] += 1.0
    table += 0.5
    table[:, :, :3] = 0.0
    s = table.sum(axis=2, keepdims=True)
    s[s == 0] = 1.0
    return (table / s).astype(np.float32)


# ============================================================================
# F5 — global app popularity bucket (rare / medium / common)
# ============================================================================
def fit_app_popularity_bucket(per_user_train_dfs: Dict[str, pd.DataFrame],
                               vocab: Dict[str, int],
                               common_min: int = 15,
                               medium_min: int = 5) -> np.ndarray:
    """Bucket apps into rare(0)/medium(1)/common(2) by # users they appear in.

    common: appears (as a target event) in >= common_min users
    medium: appears in [medium_min, common_min) users
    rare:   appears in < medium_min users (incl. all reserved tokens)
    """
    V = len(vocab)
    out = np.zeros(V, dtype=np.int64)  # default rare = 0
    n_users_per_app: Dict[int, int] = {}
    for uid, df in per_user_train_dfs.items():
        if "is_target_event" not in df.columns or "app_label_clean" not in df.columns:
            continue
        mask = df["is_target_event"].astype(bool).to_numpy()
        if not mask.any():
            continue
        apps = df.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).unique()
        for a in apps:
            i = vocab.get(a, vocab.get("<RARE>", 2))
            if i >= 3:
                n_users_per_app[i] = n_users_per_app.get(i, 0) + 1
    for i, n in n_users_per_app.items():
        if n >= common_min:
            out[i] = 2
        elif n >= medium_min:
            out[i] = 1
    return out


# ============================================================================
# F6 — leave-one-out cross-user same-app prior
# ============================================================================
def fit_cross_user_app_prior(bg_train: pd.DataFrame,
                              vocab: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Leave-one-out cross-user mean pos-rate per (user, app).

    For each (user u, app a):
        prior[u][a] = mean over OTHER users v of P(y=1 | app=a, user=v)

    (Per-user means averaged across users — NOT row-weighted.)

    Apps absent from all other users' bg_train get the global mean. Reserved
    tokens (idx < 3) are returned as the global mean.
    """
    V = len(vocab)
    if "user_uid" not in bg_train.columns or "app_idx" not in bg_train.columns:
        raise KeyError("bg_train must have user_uid + app_idx + y_3600 columns")
    if len(bg_train) == 0:
        return {}
    grp = bg_train.groupby(["user_uid", "app_idx"])["y_3600"].agg(["sum", "count"]).reset_index()
    grp = grp.rename(columns={"sum": "n_pos", "count": "n_rows"})
    global_mean = float(bg_train["y_3600"].mean())
    out: Dict[str, np.ndarray] = {}
    users = sorted(bg_train["user_uid"].unique().tolist())
    for u in users:
        prior = np.full(V, global_mean, dtype=np.float32)
        others = grp[grp["user_uid"] != u]
        if len(others) == 0:
            out[u] = prior
            continue
        # Per-app mean of per-user pos-rates among the other users
        for app_idx, sub in others.groupby("app_idx"):
            rates = (sub["n_pos"] / sub["n_rows"]).to_numpy()
            i = int(app_idx)
            if 0 <= i < V:
                prior[i] = float(rates.mean())
        out[u] = prior
    return out


# ============================================================================
# F1 — 2-step Markov per user (requires last2_fg_app_idx, see compute_last2_*)
# ============================================================================
def compute_last2_fg_for_anchors(anchor_ts_ns: np.ndarray,
                                   fg_ts_ns: np.ndarray,
                                   fg_app_idx: np.ndarray) -> np.ndarray:
    """For each anchor, return the second-most-recent FG app idx (or 0 if none).

    fg_ts_ns and fg_app_idx must be aligned; sorted-by-ts will be enforced if needed.
    """
    if len(fg_ts_ns) == 0:
        return np.zeros(len(anchor_ts_ns), dtype=np.int64)
    if not (np.diff(fg_ts_ns) >= 0).all():
        order = np.argsort(fg_ts_ns, kind="mergesort")
        fg_ts_ns = fg_ts_ns[order]
        fg_app_idx = fg_app_idx[order]
    out = np.zeros(len(anchor_ts_ns), dtype=np.int64)
    for i, t in enumerate(anchor_ts_ns):
        pos = int(np.searchsorted(fg_ts_ns, int(t), side="right"))
        if pos >= 2:
            out[i] = int(fg_app_idx[pos - 2])
    return out


def fit_two_step_markov_per_user(train_events: pd.DataFrame,
                                   vocab: Dict[str, int],
                                   alpha: float = 0.5) -> Dict[Tuple[int, int], np.ndarray]:
    """Sparse map (a_{t-2}, a_{t-1}) -> Dirichlet-smoothed (V,) probability vector
    over a_t, fit on train target events of one user.
    Reserved tokens are zeroed; row sum over non-reserved is 1.0."""
    V = len(vocab)
    rare = vocab.get("<RARE>", 2)
    df = train_events[train_events["is_target_event"].astype(bool)].sort_values("event_ts")
    apps = df["app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    counts: Dict[Tuple[int, int], np.ndarray] = {}
    for i in range(2, len(idx)):
        key = (int(idx[i - 2]), int(idx[i - 1]))
        if key not in counts:
            counts[key] = np.zeros(V, dtype=np.float64)
        a = int(idx[i])
        if 3 <= a < V:
            counts[key][a] += 1.0
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for k, v in counts.items():
        v = v + alpha
        v[:3] = 0.0
        s = v.sum()
        if s == 0:
            continue
        out[k] = (v / s).astype(np.float32)
    return out


# ============================================================================
# F2 — per-user co-FG matrix
# ============================================================================
def fit_co_fg_matrix_per_user(bg_train: pd.DataFrame,
                                vocab: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Per-user (V, V) co-fg matrix: co_fg[u][a, b] = P(y=1 | app=a, last_fg=b).

    Cells with no training rows fall back to the user's overall pos rate.
    """
    V = len(vocab)
    out: Dict[str, np.ndarray] = {}
    if len(bg_train) == 0:
        return out
    for uid, sub in bg_train.groupby("user_uid", sort=False):
        overall = float(sub["y_3600"].mean())
        mat = np.full((V, V), overall, dtype=np.float32)
        if len(sub) == 0:
            out[uid] = mat
            continue
        grp = sub.groupby(["app_idx", "last_fg_app_idx"])["y_3600"].agg(["sum", "count"]).reset_index()
        for _, r in grp.iterrows():
            a = int(r["app_idx"]); b = int(r["last_fg_app_idx"])
            if 0 <= a < V and 0 <= b < V and int(r["count"]) >= 1:
                mat[a, b] = float(r["sum"]) / float(r["count"])
        out[uid] = mat
    return out
