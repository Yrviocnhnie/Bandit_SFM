# c3.4 Multi-User Feature Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate whether 7 new features (F1–F7 from `REPORT_bgkill_multiuser_eda.md` §13.2) and an explicit `cat_emb` improve the multi-user Task C model's test PR-AUC over the c3.3 baseline (current best: Pro-List/c3.3 = 0.908). Run staged ablations so we can attribute lift to specific feature groups.

**Architecture:** Same single-head MLP family as c3.3 with two changes possible per stage: (a) feature_dim grows from 33 → 41 → 44 across stages, (b) optional architecture change to concat `cat_emb (4d) + bg_comp_emb (4d)` alongside the existing `app_emb (16d) + numeric` input.

**Tech Stack:** PyTorch, pandas, numpy, pytest. CPU-friendly. New code lives under `app_usage_data/lib/bg_multi/` and `app_usage_data/scripts/`.

---

## Spec — what we're testing, in 3 stages

| Stage | Schema name | New features added (cumulative) | Numeric dim | Arch change | Gate |
|-------|-------------|---------------------------------|------------:|------------:|------|
| 1 | `c3.4n_cheap` | F3 (DOW × hour), F5 (popularity bucket), F6 (cross-user prior), F7 (hour-segment one-hot) | 33 → 41 | none | always run |
| 2 | `c3.4n_full` | + F1 (2-step Markov), F2 (co-FG matrix max + mean) | 41 → 44 | none | run if Stage-1 lift ≥ +0.3 pp PR-AUC |
| 3 | `c3.4_full` | + F4 (B(t) composition embedding) — `cat_emb` mean-pool | 44 (numeric) | **+ `cat_emb` (4d) + `bg_comp_emb` (4d)** | run if Stage-2 lift ≥ +0.3 pp PR-AUC |

**Comparison protocol:** every stage trains the same recipe family (`c3p4_X` ≡ baseline MLP, dropout 0.2, no listwise, no SWA) and is compared to the **c3p3 multi-user baseline** (PR-AUC = 0.913 anchor-mean / 0.904 user-mean, recorded in `artifacts/bg_multi/results/task_c_multi_summary.json`). Lift threshold = +0.3 pp on **per-user mean PR-AUC** to advance to the next stage.

**Reference baselines** (from `artifacts/bg_multi/results/h60_leaderboard.json`, test, per-user mean):
- random: 0.7350
- Markov-inv: 0.8903
- C3.3: 0.9042
- Pro-List/c3.3: **0.9076** ← best trained
- Pro-Wide/c3.3: 0.9049

**Out of scope (deferred):**
- Re-training the wide / listwise / regularised recipes on c3.4 (only baseline arch this run).
- Per-user fine-tune heads.
- Schema c3.4 single-user counterpart (this is multi-user only).

---

## File Structure

**New files (additive only):**

| Path | Purpose |
|------|---------|
| `app_usage_data/lib/bg_multi/c34_features.py` | New per-user / global stat fitters: `fit_dow_hour_freq`, `fit_app_popularity_bucket`, `fit_cross_user_app_prior`, `fit_two_step_markov_per_user`, `fit_co_fg_matrix_per_user` + helpers. |
| `app_usage_data/lib/bg_multi/tests/test_c34_features.py` | Layer-1/2 tests for the new fitters. |

**Files to modify (additive within file):**

| Path | Change |
|------|--------|
| `app_usage_data/lib/bg_multi/helpers.py` | Extend `fit_per_user_stats` to include the new per-user stats. Add `fit_global_c34_stats` that returns popularity bucket + cross-user prior. |
| `app_usage_data/scripts/50_build_bg_data_multiuser.py` | Save the new global stats (popularity bucket + cross-user prior) to `artifacts/bg_multi/global_stats.pkl`. Save the new per-user stats inside the existing per-user `stats.pkl`. Add `last2_fg_app_idx` column to bg parquets via post-hoc lookup. |
| `app_usage_data/scripts/52_train_task_c_multiuser.py` | Add `build_c34n_cheap_multi`, `build_c34n_full_multi`, `build_c34_full_multi`. Add 3 recipes: `c3p4_cheap`, `c3p4_full`, `c3p4_full_cat`. Add `make_baseline_model_with_cat_emb` for the architecture change. |
| `app_usage_data/scripts/53_eval_h60_multiuser.py` | Accept `--schema` flag (default `c3.3`); look up the right `build_*_multi` based on the recipe metadata in the saved checkpoint. |
| `app_usage_data/scripts/54_threshold_multiuser.py` | Same `--schema` plumbing. |
| `app_usage_data/scripts/55_positive_metrics_multiuser.py` | Same `--schema` plumbing. |
| `app_usage_data/REPORT_bgkill_multiuser.md` | Append §13 ("c3.4 ablation") with stage-by-stage results. |

**No edits to `lib/bg/*.py`** — staying additive in the multi-user namespace.

---

## STAGE 1 — `c3.4n_cheap` (4 cheap features, no arch change)

### Task 1: Fit per-user `dow_hour_freq` (F3)

**Files:**
- Create: `app_usage_data/lib/bg_multi/c34_features.py`
- Test: `app_usage_data/lib/bg_multi/tests/test_c34_features.py`

- [ ] **Step 1.1: Write the failing test for `fit_dow_hour_freq`**

```python
# app_usage_data/lib/bg_multi/tests/test_c34_features.py
"""Tests for c3.4 feature fitters."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg_multi.c34_features import fit_dow_hour_freq


def _train_events_for_user():
    """Tiny synthetic train_events for one user: 5 target events spread over 2 days."""
    return pd.DataFrame({
        "event_ts": pd.to_datetime([
            "2026-04-01 09:00:00",  # Wed (weekday=2), hour=9, app=A
            "2026-04-01 09:30:00",  # Wed,  hour=9, app=A
            "2026-04-01 14:00:00",  # Wed,  hour=14, app=B
            "2026-04-02 09:00:00",  # Thu (weekday=3), hour=9, app=A
            "2026-04-02 14:00:00",  # Thu,  hour=14, app=A
        ]),
        "is_target_event": [True, True, True, True, True],
        "app_label_clean": ["A", "A", "B", "A", "A"],
    })


def test_fit_dow_hour_freq_shape_and_normalisation():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2, "A": 3, "B": 4}
    train = _train_events_for_user()
    table = fit_dow_hour_freq(train, vocab)
    # Shape: (7 days, 24 hours, V=5)
    assert table.shape == (7, 24, 5)
    assert table.dtype == np.float32
    # Reserved cols (<PAD>, <UNK>, <RARE>) zeroed
    assert (table[:, :, :3] == 0.0).all()
    # Each (dow, hour) row over the non-reserved app columns sums to 1.0 (Dirichlet-smoothed)
    sums = table[:, :, 3:].sum(axis=2)
    np.testing.assert_allclose(sums, 1.0, atol=1e-4)


def test_fit_dow_hour_freq_picks_up_observed_app():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2, "A": 3, "B": 4}
    train = _train_events_for_user()
    table = fit_dow_hour_freq(train, vocab)
    # On Wed (weekday=2) at hour=9, only A appears -> A's smoothed prob > B's
    assert table[2, 9, 3] > table[2, 9, 4]
    # On Wed at hour=14, only B appears -> B > A
    assert table[2, 14, 4] > table[2, 14, 3]
```

- [ ] **Step 1.2: Run the test to verify it fails**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -m pytest lib/bg_multi/tests/test_c34_features.py::test_fit_dow_hour_freq_shape_and_normalisation -v
```

Expected: `ImportError: cannot import name 'fit_dow_hour_freq'` — the module doesn't exist yet.

- [ ] **Step 1.3: Implement `fit_dow_hour_freq`**

```python
# app_usage_data/lib/bg_multi/c34_features.py
"""c3.4 feature fitters (per-user and global).

All per-user fits read TRAIN events only and shape outputs against the
pooled vocab. Global fits read the pooled bg_train (or pooled train events)
and return global priors usable across users.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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
```

- [ ] **Step 1.4: Run the test to verify it passes**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -m pytest lib/bg_multi/tests/test_c34_features.py::test_fit_dow_hour_freq_shape_and_normalisation lib/bg_multi/tests/test_c34_features.py::test_fit_dow_hour_freq_picks_up_observed_app -v
```

Expected: 2 passed.

- [ ] **Step 1.5: Commit**

```bash
cd /home/mohan/Bandit_SFM
git add app_usage_data/lib/bg_multi/c34_features.py app_usage_data/lib/bg_multi/tests/test_c34_features.py
git commit -m "c3.4 features: add fit_dow_hour_freq (F3) per-user fitter + tests"
```

### Task 2: Fit global `app_popularity_bucket` (F5)

**Files:**
- Modify: `app_usage_data/lib/bg_multi/c34_features.py`
- Modify: `app_usage_data/lib/bg_multi/tests/test_c34_features.py`

- [ ] **Step 2.1: Add the failing test for `fit_app_popularity_bucket`**

Append to `test_c34_features.py`:

```python
from lib.bg_multi.c34_features import fit_app_popularity_bucket


def test_fit_app_popularity_bucket_thresholds():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2,
             "WIDE_APP": 3, "MED_APP": 4, "RARE_APP": 5, "UNUSED_APP": 6}
    # Synthetic per-user train target sets:
    #   WIDE_APP appears in all 22 users -> "common"
    #   MED_APP  appears in 10 users     -> "medium"
    #   RARE_APP appears in 2 users      -> "rare"
    #   UNUSED_APP never appears         -> "rare"
    per_user_train_dfs = {}
    for i in range(22):
        apps = ["WIDE_APP"]
        if i < 10:
            apps.append("MED_APP")
        if i < 2:
            apps.append("RARE_APP")
        df = pd.DataFrame({
            "is_target_event": [True] * len(apps),
            "app_label_clean": apps,
        })
        per_user_train_dfs[f"u{i}"] = df

    bucket = fit_app_popularity_bucket(
        per_user_train_dfs, vocab,
        common_min=15, medium_min=5,
    )
    assert bucket.shape == (len(vocab),)
    assert bucket.dtype == np.int64
    # Reserved tokens get the "rare" bucket (= 0)
    assert bucket[0] == 0  # <PAD>
    assert bucket[1] == 0  # <UNK>
    assert bucket[2] == 0  # <RARE>
    # WIDE_APP in all 22 -> common (= 2)
    assert bucket[vocab["WIDE_APP"]] == 2
    # MED_APP in 10 -> medium (= 1)
    assert bucket[vocab["MED_APP"]] == 1
    # RARE_APP in 2 -> rare (= 0)
    assert bucket[vocab["RARE_APP"]] == 0
    # UNUSED_APP -> rare (= 0)
    assert bucket[vocab["UNUSED_APP"]] == 0
```

- [ ] **Step 2.2: Run test, verify it fails**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -m pytest lib/bg_multi/tests/test_c34_features.py::test_fit_app_popularity_bucket_thresholds -v
```

Expected: ImportError on `fit_app_popularity_bucket`.

- [ ] **Step 2.3: Implement `fit_app_popularity_bucket`**

Append to `app_usage_data/lib/bg_multi/c34_features.py`:

```python
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
        if "is_target_event" not in df.columns:
            continue
        mask = df["is_target_event"].astype(bool).to_numpy()
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
        # else: keep 0 (rare)
    return out
```

- [ ] **Step 2.4: Run test, verify it passes**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -m pytest lib/bg_multi/tests/test_c34_features.py::test_fit_app_popularity_bucket_thresholds -v
```

Expected: 1 passed.

- [ ] **Step 2.5: Commit**

```bash
cd /home/mohan/Bandit_SFM
git add app_usage_data/lib/bg_multi/c34_features.py app_usage_data/lib/bg_multi/tests/test_c34_features.py
git commit -m "c3.4 features: add fit_app_popularity_bucket (F5) global fitter + test"
```

### Task 3: Fit global `cross_user_app_prior` (F6)

**Files:**
- Modify: `app_usage_data/lib/bg_multi/c34_features.py`
- Modify: `app_usage_data/lib/bg_multi/tests/test_c34_features.py`

- [ ] **Step 3.1: Add failing test for `fit_cross_user_app_prior`**

Append to `test_c34_features.py`:

```python
from lib.bg_multi.c34_features import fit_cross_user_app_prior


def test_fit_cross_user_app_prior_excludes_self():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2, "X": 3, "Y": 4}
    # Three users, each with their own bg_train slice on X and Y
    bg_train = pd.DataFrame({
        "user_uid": ["u0"]*4 + ["u1"]*4 + ["u2"]*4,
        "app_idx":  [3, 3, 4, 4]*3,
        "y_3600":   [1, 1, 0, 0,    # u0: P(used | X) = 1.0, P(used | Y) = 0.0
                     1, 0, 0, 0,    # u1: P(used | X) = 0.5, P(used | Y) = 0.0
                     0, 0, 1, 1],   # u2: P(used | X) = 0.0, P(used | Y) = 1.0
    })
    table = fit_cross_user_app_prior(bg_train, vocab)
    # Shape: dict {uid: (V,)} of leave-one-out cross-user mean pos rates
    assert isinstance(table, dict)
    for uid in ("u0", "u1", "u2"):
        assert table[uid].shape == (5,)
    # u0's prior for app X is mean of u1 and u2: (0.5 + 0.0) / 2 = 0.25
    np.testing.assert_allclose(table["u0"][3], 0.25, atol=1e-6)
    # u0's prior for app Y is mean of u1 and u2: (0.0 + 1.0) / 2 = 0.5
    np.testing.assert_allclose(table["u0"][4], 0.5, atol=1e-6)
    # u1's prior for X excludes u1: (1.0 + 0.0) / 2 = 0.5
    np.testing.assert_allclose(table["u1"][3], 0.5, atol=1e-6)
```

- [ ] **Step 3.2: Run, verify fails**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -m pytest lib/bg_multi/tests/test_c34_features.py::test_fit_cross_user_app_prior_excludes_self -v
```

- [ ] **Step 3.3: Implement `fit_cross_user_app_prior`**

Append to `c34_features.py`:

```python
def fit_cross_user_app_prior(bg_train: pd.DataFrame,
                              vocab: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Leave-one-out cross-user mean pos-rate per (user, app).

    For each (user u, app a):
        prior[u][a] = mean over OTHER users v of P(y=1 | app=a, user=v)

    Apps absent from all other users' bg_train get the global mean. Reserved
    tokens (idx < 3) are returned as the global mean as well (we never look
    them up in practice).
    """
    V = len(vocab)
    if "user_uid" not in bg_train.columns or "app_idx" not in bg_train.columns:
        raise KeyError("bg_train must have user_uid + app_idx + y_3600 columns")
    # Compute per-(user, app) sum + count first
    grp = bg_train.groupby(["user_uid", "app_idx"])["y_3600"].agg(["sum", "count"])
    grp = grp.rename(columns={"sum": "n_pos", "count": "n_rows"}).reset_index()
    global_mean = float(bg_train["y_3600"].mean()) if len(bg_train) else 0.0
    out: Dict[str, np.ndarray] = {}
    users = sorted(bg_train["user_uid"].unique().tolist())
    # Precompute per-app totals
    per_app = grp.groupby("app_idx")[["n_pos", "n_rows"]].sum().reset_index()
    per_app_dict = {int(r["app_idx"]): (float(r["n_pos"]), float(r["n_rows"]))
                    for _, r in per_app.iterrows()}
    for u in users:
        prior = np.full(V, global_mean, dtype=np.float32)
        # u's own contribution
        u_grp = grp[grp["user_uid"] == u].set_index("app_idx")
        u_pos_count = {int(idx): (float(r["n_pos"]), float(r["n_rows"]))
                       for idx, r in u_grp.iterrows()}
        for app_idx, (total_pos, total_rows) in per_app_dict.items():
            u_pos, u_rows = u_pos_count.get(app_idx, (0.0, 0.0))
            other_pos = total_pos - u_pos
            other_rows = total_rows - u_rows
            if other_rows > 0:
                # Per-user mean (not row-weighted): average across the (n-1)
                # other users that have this app.
                # Compute that explicitly:
                others = grp[(grp["user_uid"] != u) & (grp["app_idx"] == app_idx)]
                if len(others) > 0:
                    rates = (others["n_pos"] / others["n_rows"]).to_numpy()
                    prior[app_idx] = float(rates.mean())
                else:
                    prior[app_idx] = global_mean
            # else: leave default = global_mean
        out[u] = prior
    return out
```

- [ ] **Step 3.4: Run, verify passes**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -m pytest lib/bg_multi/tests/test_c34_features.py::test_fit_cross_user_app_prior_excludes_self -v
```

Expected: 1 passed.

- [ ] **Step 3.5: Commit**

```bash
cd /home/mohan/Bandit_SFM
git add app_usage_data/lib/bg_multi/c34_features.py app_usage_data/lib/bg_multi/tests/test_c34_features.py
git commit -m "c3.4 features: add fit_cross_user_app_prior (F6) leave-one-out fitter + test"
```

### Task 4: Build `c3.4n_cheap` schema in `52_train_task_c_multiuser.py`

**Files:**
- Modify: `app_usage_data/scripts/52_train_task_c_multiuser.py`

This task wires F3, F5, F6, F7 into `build_schema_data_multi`. F7 (hour-segment one-hot) is a pure derivation from `anchor_hour` — no fit needed.

- [ ] **Step 4.1: Add the new schema function `build_c34n_cheap_multi`**

Append to `52_train_task_c_multiuser.py` after `build_c2_multi`:

```python
HOUR_SEGMENT_BINS = [(6, 11, 0),    # morning  (6 - 10)
                     (11, 14, 1),   # lunch    (11 - 13)
                     (14, 18, 2),   # afternoon(14 - 17)
                     (18, 22, 3),   # evening  (18 - 21)
                     (22, 6, 4)]    # night    (22 - 5)


def hour_segment_onehot(hours: np.ndarray) -> np.ndarray:
    """Convert hours-of-day → 5-class one-hot (morning/lunch/afternoon/evening/night).

    Returns: (N, 5) float32.
    """
    n = len(hours)
    out = np.zeros((n, 5), dtype=np.float32)
    h = hours.astype(np.int64) % 24
    for i, (a, b, k) in enumerate(HOUR_SEGMENT_BINS):
        if a <= b:
            mask = (h >= a) & (h < b)
        else:  # wrap-around (night)
            mask = (h >= a) | (h < b)
        out[mask, k] = 1.0
    return out


def add_c34_cheap_features_multi(bg_df, ctx) -> np.ndarray:
    """8 columns: F3 (1) + F5 (1) + F6 (1) + F7 (5).

    Requires ctx to contain:
      ctx['dow_hour_freq_stack']: (U, 7, 24, V)
      ctx['app_popularity_bucket']: (V,) int64 in {0,1,2}
      ctx['cross_user_app_prior_per_user']: dict[uid -> (V,) float32]
      ctx['uid_to_idx']: dict[uid -> int]
    """
    uid_idx = bg_df["user_id_idx"].to_numpy(dtype=np.int64)
    hr = bg_df["anchor_hour"].to_numpy(dtype=np.int64) % 24
    dow = bg_df["anchor_weekday"].to_numpy(dtype=np.int64) % 7
    app = bg_df["app_idx"].to_numpy(dtype=np.int64)
    V = ctx["dow_hour_freq_stack"].shape[3]
    app_clip = np.clip(app, 0, V - 1)

    # F3: per-user P(app | dow, hour)
    f3_dow_hour_prob = ctx["dow_hour_freq_stack"][uid_idx, dow, hr, app_clip].astype(np.float32)

    # F5: app popularity bucket as a scalar in {0, 1, 2}
    pop_int = ctx["app_popularity_bucket"][app_clip].astype(np.float32)
    f5_pop_bucket = pop_int  # leave as scalar; tree/MLP handles it as ordinal

    # F6: cross-user same-app prior (per-user; built per-user-uid lookup)
    f6_cross_user_prior = np.empty(len(bg_df), dtype=np.float32)
    cu_per_user = ctx["cross_user_app_prior_per_user"]
    uid_str = bg_df["user_uid"].to_numpy()
    # Vectorize via group-by: for each unique uid, slice and assign
    for u in np.unique(uid_str):
        mask = uid_str == u
        prior = cu_per_user.get(str(u))
        if prior is None:
            f6_cross_user_prior[mask] = 0.5
        else:
            f6_cross_user_prior[mask] = prior[app_clip[mask]]

    # F7: hour-segment one-hot (5 columns)
    f7_seg = hour_segment_onehot(hr)

    return np.concatenate(
        [f3_dow_hour_prob[:, None], f5_pop_bucket[:, None],
         f6_cross_user_prior[:, None], f7_seg],
        axis=1,
    ).astype(np.float32)


def build_c34n_cheap_multi(schema: str, bg_df: pd.DataFrame, ctx: dict) -> dict:
    """c3.3 schema + 8 cheap features (F3, F5, F6, F7). 33 + 8 = 41 numeric features."""
    base = build_schema_data_multi("c3.3", bg_df, ctx)
    extra = add_c34_cheap_features_multi(bg_df, ctx)
    base["features"] = np.concatenate([base["features"], extra], axis=1)
    return base
```

- [ ] **Step 4.2: Add a smoke test for the schema build**

Append to `app_usage_data/lib/bg_multi/tests/test_c34_features.py`:

```python
import importlib.util


def _load_train_multi():
    spec = importlib.util.spec_from_file_location(
        "train_multi_c34_smoke",
        ROOT / "scripts" / "52_train_task_c_multiuser.py",
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["train_multi_c34_smoke"] = m
    spec.loader.exec_module(m)
    return m


def test_hour_segment_onehot_partitions_correctly():
    train_multi = _load_train_multi()
    hours = np.arange(24)
    out = train_multi.hour_segment_onehot(hours)
    assert out.shape == (24, 5)
    # Every hour belongs to exactly one segment
    np.testing.assert_array_equal(out.sum(axis=1), np.ones(24))
    # Spot checks
    assert out[7, 0] == 1.0   # morning bucket
    assert out[12, 1] == 1.0  # lunch
    assert out[15, 2] == 1.0  # afternoon
    assert out[20, 3] == 1.0  # evening
    assert out[23, 4] == 1.0  # night
    assert out[3, 4] == 1.0   # night wrap
```

- [ ] **Step 4.3: Run, verify pass**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -m pytest lib/bg_multi/tests/test_c34_features.py::test_hour_segment_onehot_partitions_correctly -v
```

Expected: 1 passed.

- [ ] **Step 4.4: Commit**

```bash
cd /home/mohan/Bandit_SFM
git add app_usage_data/scripts/52_train_task_c_multiuser.py app_usage_data/lib/bg_multi/tests/test_c34_features.py
git commit -m "c3.4 schema: add build_c34n_cheap_multi (c3.3 + F3/F5/F6/F7) + test"
```

### Task 5: Wire new stats into `build_ctx_multi`

**Files:**
- Modify: `app_usage_data/scripts/52_train_task_c_multiuser.py`
- Modify: `app_usage_data/scripts/50_build_bg_data_multiuser.py`

The new ctx requires `dow_hour_freq_stack`, `app_popularity_bucket`, and `cross_user_app_prior_per_user`. We compute them on-the-fly from the existing per-user stats and the bg_train, since none of the current stats.pkl files contain them.

- [ ] **Step 5.1: Update `build_ctx_multi` in `52_train_task_c_multiuser.py`**

Find `build_ctx_multi` and add these computations near the end (before the `return` statement):

```python
    # === c3.4 cheap features context ===
    # F3: per-user dow_hour_freq stack (compute on-the-fly from train_events
    #     loaded per-user). Re-uses the per-user enriched parquets.
    from lib.bg_multi.c34_features import (
        fit_dow_hour_freq, fit_app_popularity_bucket, fit_cross_user_app_prior,
    )
    art_root = art_dir
    dow_hour_freq_per_user: dict[str, np.ndarray] = {}
    train_dfs_per_user: dict[str, pd.DataFrame] = {}
    for uid in uid_to_idx:
        # The per-user enriched parquet was saved by 50_build_bg_data_multiuser.py.
        # If unavailable, fall back to the multiuser/<set>/<uid>/splits/train.parquet.
        candidate_paths = list((art_root / "multiuser").glob(f"*/{uid}/splits/train.parquet"))
        if not candidate_paths:
            raise FileNotFoundError(
                f"per-user train.parquet not found for uid={uid}. Run 40_prep_multiuser.py first."
            )
        tr = pd.read_parquet(candidate_paths[0])
        train_dfs_per_user[uid] = tr
        dow_hour_freq_per_user[uid] = fit_dow_hour_freq(tr, vocab)
    # Stack
    dow_hour_freq_stack = stack_per_user(
        {u: {"dow_hour_freq": v} for u, v in dow_hour_freq_per_user.items()},
        "dow_hour_freq", uid_to_idx,
    )

    # F5: global app popularity bucket
    app_popularity_bucket = fit_app_popularity_bucket(
        train_dfs_per_user, vocab,
        common_min=15, medium_min=5,
    )

    # F6: cross-user app prior (uses bg_train pooled — load it)
    bg_train = pd.read_parquet(bg_multi / "splits" / "bg_train.parquet")
    cross_user_app_prior_per_user = fit_cross_user_app_prior(bg_train, vocab)
```

In the same function, append to the returned context dict:

```python
        "dow_hour_freq_stack": dow_hour_freq_stack,
        "app_popularity_bucket": app_popularity_bucket,
        "cross_user_app_prior_per_user": cross_user_app_prior_per_user,
```

- [ ] **Step 5.2: Add a quick smoke check via Python REPL**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -c "
import importlib.util, sys
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('train_multi', 'scripts/52_train_task_c_multiuser.py')
m = importlib.util.module_from_spec(spec)
sys.modules['train_multi'] = m
spec.loader.exec_module(m)
from pathlib import Path
ctx = m.build_ctx_multi(Path('artifacts'))
print('dow_hour_freq_stack', ctx['dow_hour_freq_stack'].shape)
print('app_popularity_bucket', ctx['app_popularity_bucket'].shape)
print('cross_user_app_prior n_users', len(ctx['cross_user_app_prior_per_user']))
"
```

Expected output (approx):
```
dow_hour_freq_stack (22, 7, 24, 243)
app_popularity_bucket (243,)
cross_user_app_prior n_users 22
```

- [ ] **Step 5.3: Commit**

```bash
cd /home/mohan/Bandit_SFM
git add app_usage_data/scripts/52_train_task_c_multiuser.py
git commit -m "c3.4: extend build_ctx_multi with dow_hour_freq + popularity + cross-user prior"
```

### Task 6: Add the `c3p4_cheap` recipe and train it

**Files:**
- Modify: `app_usage_data/scripts/52_train_task_c_multiuser.py`

- [ ] **Step 6.1: Add the recipe + dispatch**

In `52_train_task_c_multiuser.py:RECIPES`, add:

```python
RECIPES["c3p4_cheap"] = {
    "schema": "c3.4n_cheap", "wide": False, "dropout": 0.2,
    "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.0,
}
```

And in `train_recipe`, replace the line `d_tr = build_schema_data_multi(schema, ...)` with a dispatch:

```python
def _build_data_dispatch(schema: str, bg_df: pd.DataFrame, ctx: dict) -> dict:
    if schema == "c3.4n_cheap":
        return build_c34n_cheap_multi(schema, bg_df, ctx)
    return build_schema_data_multi(schema, bg_df, ctx)
```

Then use `_build_data_dispatch` everywhere `build_schema_data_multi` is currently called inside `train_recipe`.

- [ ] **Step 6.2: Train one recipe**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python scripts/52_train_task_c_multiuser.py --recipes c3p4_cheap --epochs 30 --patience 4 2>&1 | tail -20
```

Expected: training runs ~10 min, writes `artifacts/bg_multi/checkpoints/task_c_multi_c3p4_cheap.pt` and `results/task_c_multi_c3p4_cheap.json`.

- [ ] **Step 6.3: Inspect the result and compare to c3p3**

```bash
python -c "
import json
c33 = json.load(open('artifacts/bg_multi/results/task_c_multi_c3p3.json'))
c34 = json.load(open('artifacts/bg_multi/results/task_c_multi_c3p4_cheap.json'))
print(f'c3p3       feature_dim={c33[\"feature_dim\"]} test_PR={c33[\"test\"][\"pr_auc_mean\"]:.4f} test_FK@.5={c33[\"test\"][\"false_kill_rate\"][\"0.5\"]:.4f}')
print(f'c3p4_cheap feature_dim={c34[\"feature_dim\"]} test_PR={c34[\"test\"][\"pr_auc_mean\"]:.4f} test_FK@.5={c34[\"test\"][\"false_kill_rate\"][\"0.5\"]:.4f}')
delta = c34['test']['pr_auc_mean'] - c33['test']['pr_auc_mean']
print(f'Δ (anchor-mean) PR-AUC = {delta:+.4f}')
"
```

- [ ] **Step 6.4: Run full eval (per-user mean + Pareto)**

The 53/54/55 eval scripts will need to know how to score the new recipe. Add it to `TRAINED_PICKS` in each:

For `53_eval_h60_multiuser.py`, change:
```python
TRAINED_PICKS = ("c3p3", "c3pro_reg", "c3pro_listwise", "c3pro_wide")
RECIPE_LABELS = {
    "c3p3":           "C3.3",
    "c3pro_reg":      "Pro-Reg/c3.3",
    "c3pro_listwise": "Pro-List/c3.3",
    "c3pro_wide":     "Pro-Wide/c3.3",
}
```
to:
```python
TRAINED_PICKS = ("c3p3", "c3pro_reg", "c3pro_listwise", "c3pro_wide", "c3p4_cheap")
RECIPE_LABELS = {
    "c3p3":           "C3.3",
    "c3pro_reg":      "Pro-Reg/c3.3",
    "c3pro_listwise": "Pro-List/c3.3",
    "c3pro_wide":     "Pro-Wide/c3.3",
    "c3p4_cheap":     "C3.4n-cheap",
}
```

The script also calls `score_trained()` which builds features via `_TRAIN_MULTI.build_schema_data_multi("c3.3", ...)`. Replace that with dispatch:

```python
def _build_data_dispatch(schema, df, ctx):
    if schema == "c3.4n_cheap":
        return _TRAIN_MULTI.build_c34n_cheap_multi(schema, df, ctx)
    return _TRAIN_MULTI.build_schema_data_multi(schema, df, ctx)
```

And inside `score_trained`, look up the recipe's schema:
```python
rec = _TRAIN_MULTI.RECIPES[model_recipe]
data_d = _build_data_dispatch(rec["schema"], df, ctx_multi)
```

Apply the same `_build_data_dispatch` change in `54_threshold_multiuser.py` and `55_positive_metrics_multiuser.py`.

- [ ] **Step 6.5: Run full eval**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python scripts/53_eval_h60_multiuser.py --include-slice-b 2>&1 | grep -E 'C3\.4n-cheap|C3\.3 ' | tail -6
```

- [ ] **Step 6.6: Commit**

```bash
cd /home/mohan/Bandit_SFM
git add app_usage_data/scripts/52_train_task_c_multiuser.py \
        app_usage_data/scripts/53_eval_h60_multiuser.py \
        app_usage_data/scripts/54_threshold_multiuser.py \
        app_usage_data/scripts/55_positive_metrics_multiuser.py \
        app_usage_data/artifacts/bg_multi/checkpoints/task_c_multi_c3p4_cheap.pt \
        app_usage_data/artifacts/bg_multi/results/task_c_multi_c3p4_cheap.json
git commit -m "c3.4 stage 1: train + eval c3p4_cheap (33+8 features, no arch change)"
```

### Task 7: Stage-1 GATE — decide whether to proceed to Stage 2

- [ ] **Step 7.1: Read the Stage-1 result and decide**

Open `artifacts/bg_multi/results/h60_leaderboard.json` and compare `slice_a.test["C3.4n-cheap"].agg.pr_auc_mean.mean` vs `slice_a.test["C3.3"].agg.pr_auc_mean.mean`.

Gate: if **`(c3.4n_cheap_PR − c3.3_PR) ≥ 0.003`** (≥ +0.3 pp), continue to Stage 2.
Else: stop here. Document the negative result in the report and consider scope-down or a different feature set.

---

## STAGE 2 — `c3.4n_full` (add F1, F2 — heavier numeric features)

(Only execute these tasks if Stage-1 gate passed.)

### Task 8: Compute `last2_fg_app_idx` for every bg row

**Background:** F1 (2-step Markov) needs the *second-to-last* foreground app at each anchor. The bg parquets currently only store `last_fg_app_idx`. We compute `last2` post-hoc using the per-user FG timeline and add it as a column.

**Files:**
- Modify: `app_usage_data/lib/bg_multi/c34_features.py`
- Modify: `app_usage_data/lib/bg_multi/tests/test_c34_features.py`
- Modify: `app_usage_data/scripts/52_train_task_c_multiuser.py`

- [ ] **Step 8.1: Add the failing test**

Append to `test_c34_features.py`:

```python
from lib.bg_multi.c34_features import compute_last2_fg_for_anchors


def test_compute_last2_fg_for_anchors():
    # FG events at t=10s app=A, t=20s app=B, t=30s app=A, t=40s app=C
    fg_ts_ns = np.array([10, 20, 30, 40], dtype=np.int64) * 1_000_000_000
    fg_app_idx = np.array([3, 4, 3, 5], dtype=np.int64)
    # Anchors at t=15s, 25s, 35s, 45s, 50s
    anchor_ts_ns = np.array([15, 25, 35, 45, 50], dtype=np.int64) * 1_000_000_000
    last2 = compute_last2_fg_for_anchors(anchor_ts_ns, fg_ts_ns, fg_app_idx)
    # At t=15: only 1 prior FG (A) → last2 = 0 (no second-prior)
    assert last2[0] == 0
    # At t=25: 2 prior FGs A, B → last2 = A = 3
    assert last2[1] == 3
    # At t=35: 3 prior FGs A, B, A → last2 = B = 4
    assert last2[2] == 4
    # At t=45: 4 prior FGs A, B, A, C → last2 = A = 3
    assert last2[3] == 3
    # At t=50: 4 prior FGs (same as t=45) → last2 = A = 3
    assert last2[4] == 3
```

- [ ] **Step 8.2: Run, verify fails**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -m pytest lib/bg_multi/tests/test_c34_features.py::test_compute_last2_fg_for_anchors -v
```

- [ ] **Step 8.3: Implement `compute_last2_fg_for_anchors`**

Append to `c34_features.py`:

```python
def compute_last2_fg_for_anchors(anchor_ts_ns: np.ndarray,
                                   fg_ts_ns: np.ndarray,
                                   fg_app_idx: np.ndarray) -> np.ndarray:
    """For each anchor, return the second-most-recent FG app idx (or 0 if none).

    All inputs must be int64. fg_ts_ns and fg_app_idx must be aligned and
    sorted by ts ascending.
    """
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
```

- [ ] **Step 8.4: Run, verify passes**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -m pytest lib/bg_multi/tests/test_c34_features.py::test_compute_last2_fg_for_anchors -v
```

- [ ] **Step 8.5: Wire into `build_ctx_multi` to attach `last2_fg_app_idx` to each bg DataFrame at scoring time**

Add to `build_ctx_multi`:

```python
    # F1 prep — flat per-user FG timeline (sorted by ts, with app_idx)
    flat_fg_per_user: dict[str, dict[str, np.ndarray]] = {}
    for uid, ftl_dict in fg_timeline_per_user.items():
        # ftl_dict: {app_idx -> sorted ts_ns}
        all_ts: list[np.ndarray] = []
        all_app: list[np.ndarray] = []
        for ai, ts_arr in ftl_dict.items():
            if len(ts_arr) == 0:
                continue
            all_ts.append(ts_arr)
            all_app.append(np.full(len(ts_arr), int(ai), dtype=np.int64))
        if all_ts:
            ts = np.concatenate(all_ts)
            app = np.concatenate(all_app)
            order = np.argsort(ts, kind="mergesort")
            flat_fg_per_user[uid] = {"ts": ts[order], "app": app[order]}
        else:
            flat_fg_per_user[uid] = {"ts": np.zeros(0, dtype=np.int64),
                                      "app": np.zeros(0, dtype=np.int64)}
```

Note: `fg_timeline_per_user` keys are integer uid_idx in the existing code, so this loop needs to iterate `uid_to_idx` and map to/from the right key:

```python
    flat_fg_per_user = {}
    for uid, idx in uid_to_idx.items():
        ftl = fg_timeline_per_user[idx]
        all_ts, all_app = [], []
        for ai, ts_arr in ftl.items():
            if len(ts_arr) == 0:
                continue
            all_ts.append(ts_arr)
            all_app.append(np.full(len(ts_arr), int(ai), dtype=np.int64))
        if all_ts:
            ts = np.concatenate(all_ts)
            app = np.concatenate(all_app)
            order = np.argsort(ts, kind="mergesort")
            flat_fg_per_user[uid] = {"ts": ts[order], "app": app[order]}
        else:
            flat_fg_per_user[uid] = {"ts": np.zeros(0, dtype=np.int64),
                                      "app": np.zeros(0, dtype=np.int64)}
```

Add to the returned context dict:
```python
        "flat_fg_per_user": flat_fg_per_user,
```

- [ ] **Step 8.6: Add the helper that attaches `last2_fg_app_idx` to a bg_df**

Append to `52_train_task_c_multiuser.py`:

```python
def attach_last2_fg(bg_df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """Add a `last2_fg_app_idx` int64 column to bg_df via per-user lookup.
    Idempotent — overwrites if present."""
    from lib.bg_multi.c34_features import compute_last2_fg_for_anchors
    out = np.zeros(len(bg_df), dtype=np.int64)
    uid_str = bg_df["user_uid"].to_numpy()
    anchor_ts_ns = bg_df["anchor_ts_ns"].to_numpy(dtype=np.int64)
    for u in np.unique(uid_str):
        mask = uid_str == u
        flat = ctx["flat_fg_per_user"].get(str(u))
        if flat is None or len(flat["ts"]) == 0:
            continue
        out[mask] = compute_last2_fg_for_anchors(
            anchor_ts_ns[mask], flat["ts"], flat["app"],
        )
    df = bg_df.copy()
    df["last2_fg_app_idx"] = out
    return df
```

- [ ] **Step 8.7: Commit**

```bash
cd /home/mohan/Bandit_SFM
git add app_usage_data/lib/bg_multi/c34_features.py \
        app_usage_data/lib/bg_multi/tests/test_c34_features.py \
        app_usage_data/scripts/52_train_task_c_multiuser.py
git commit -m "c3.4 stage 2: add compute_last2_fg_for_anchors + attach_last2_fg helper"
```

### Task 9: Fit per-user 2-step Markov + co-FG matrix (F1, F2)

**Files:**
- Modify: `app_usage_data/lib/bg_multi/c34_features.py`
- Modify: `app_usage_data/lib/bg_multi/tests/test_c34_features.py`

- [ ] **Step 9.1: Tests for F1 + F2 fits**

Append to `test_c34_features.py`:

```python
from lib.bg_multi.c34_features import (
    fit_two_step_markov_per_user, fit_co_fg_matrix_per_user,
)


def test_fit_two_step_markov_per_user_normalisation():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2, "A": 3, "B": 4, "C": 5}
    train = pd.DataFrame({
        "event_ts": pd.to_datetime([
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
            "2026-04-01 10:00:00",
            "2026-04-01 11:00:00",
            "2026-04-01 12:00:00",
        ]),
        "is_target_event": [True] * 5,
        "app_label_clean": ["A", "B", "C", "A", "B"],   # 3 triples: ABC, BCA, CAB
    })
    table = fit_two_step_markov_per_user(train, vocab)
    assert isinstance(table, dict)
    # Triple (A, B, C) was observed once -> P(C | A, B) > P(other | A, B)
    if (vocab["A"], vocab["B"]) in table:
        probs = table[(vocab["A"], vocab["B"])]
        assert probs[vocab["C"]] > probs[vocab["A"]]
        # Smoothed sum to 1 over non-reserved
        np.testing.assert_allclose(probs[3:].sum(), 1.0, atol=1e-4)


def test_fit_co_fg_matrix_per_user():
    vocab = {"<PAD>": 0, "<UNK>": 1, "<RARE>": 2, "A": 3, "B": 4}
    # bg_train rows for one user u0:
    #   anchor 1: last_fg=A, app=B, y=1 (B used)
    #   anchor 2: last_fg=A, app=B, y=0 (B not used)
    #   anchor 3: last_fg=B, app=A, y=1 (A used)
    bg_train = pd.DataFrame({
        "user_uid": ["u0"] * 3,
        "anchor_id": [1, 2, 3],
        "app_idx":   [4, 4, 3],
        "last_fg_app_idx": [3, 3, 4],
        "y_3600": [1, 0, 1],
    })
    co_fg = fit_co_fg_matrix_per_user(bg_train, vocab)
    # co_fg["u0"][app=B][last_fg=A] = mean(y_3600 over (last=A, app=B)) = 0.5
    np.testing.assert_allclose(co_fg["u0"][vocab["B"], vocab["A"]], 0.5, atol=1e-4)
    np.testing.assert_allclose(co_fg["u0"][vocab["A"], vocab["B"]], 1.0, atol=1e-4)
```

- [ ] **Step 9.2: Run, verify fails**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -m pytest lib/bg_multi/tests/test_c34_features.py::test_fit_two_step_markov_per_user_normalisation lib/bg_multi/tests/test_c34_features.py::test_fit_co_fg_matrix_per_user -v
```

- [ ] **Step 9.3: Implement F1 and F2 fits**

Append to `c34_features.py`:

```python
def fit_two_step_markov_per_user(train_events: pd.DataFrame,
                                   vocab: Dict[str, int],
                                   alpha: float = 0.5) -> Dict[Tuple[int, int], np.ndarray]:
    """Sparse map (a_{t-2}, a_{t-1}) -> Dirichlet-smoothed (V,) probability vector
    over a_t, fit on train target events of one user."""
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
        out[k] = (v / v.sum()).astype(np.float32)
    return out


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
        grp = sub.groupby(["app_idx", "last_fg_app_idx"])["y_3600"].agg(["sum", "count"])
        for (a, b), row in grp.iterrows():
            if int(row["count"]) >= 1:
                mat[int(a), int(b)] = float(row["sum"]) / float(row["count"])
        out[uid] = mat
    return out
```

- [ ] **Step 9.4: Run, verify passes**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python -m pytest lib/bg_multi/tests/test_c34_features.py::test_fit_two_step_markov_per_user_normalisation lib/bg_multi/tests/test_c34_features.py::test_fit_co_fg_matrix_per_user -v
```

- [ ] **Step 9.5: Commit**

```bash
cd /home/mohan/Bandit_SFM
git add app_usage_data/lib/bg_multi/c34_features.py app_usage_data/lib/bg_multi/tests/test_c34_features.py
git commit -m "c3.4 stage 2: add F1 (2-step Markov) + F2 (co-FG) per-user fitters + tests"
```

### Task 10: Wire F1 + F2 into ctx + new schema `c3.4n_full`

**Files:**
- Modify: `app_usage_data/scripts/52_train_task_c_multiuser.py`

- [ ] **Step 10.1: Extend `build_ctx_multi` with F1 + F2 stats**

In `build_ctx_multi`, after computing `flat_fg_per_user`, add:

```python
    from lib.bg_multi.c34_features import (
        fit_two_step_markov_per_user, fit_co_fg_matrix_per_user,
    )
    two_step_per_user: dict[str, dict] = {}
    for uid, tr in train_dfs_per_user.items():
        two_step_per_user[uid] = fit_two_step_markov_per_user(tr, vocab)
    co_fg_per_user = fit_co_fg_matrix_per_user(bg_train, vocab)
```

Add to context dict:
```python
        "two_step_per_user": two_step_per_user,
        "co_fg_per_user": co_fg_per_user,
```

- [ ] **Step 10.2: Add `add_c34_full_features_multi` (F1 + F2 = 3 columns)**

Append:

```python
def add_c34_full_features_multi(bg_df, ctx) -> np.ndarray:
    """F1 (1) + F2 (2) = 3 columns. Requires `last2_fg_app_idx` in bg_df."""
    if "last2_fg_app_idx" not in bg_df.columns:
        raise KeyError("bg_df must have last2_fg_app_idx; call attach_last2_fg first")
    n = len(bg_df)
    uid_str = bg_df["user_uid"].to_numpy()
    last2 = bg_df["last2_fg_app_idx"].to_numpy(dtype=np.int64)
    last  = bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64)
    app   = bg_df["app_idx"].to_numpy(dtype=np.int64)

    # F1: 2-step Markov P(app | last2, last) — fall back to 1-step Markov on miss
    f1 = np.zeros(n, dtype=np.float32)
    markov_stack = ctx["markov_probs_stack"]   # (U, V, V)
    uid_idx = bg_df["user_id_idx"].to_numpy(dtype=np.int64)
    V = markov_stack.shape[1]
    for u in np.unique(uid_str):
        m_u = ctx["two_step_per_user"].get(str(u), {})
        m_idx = mask = (uid_str == u)
        for i in np.where(mask)[0]:
            key = (int(last2[i]), int(last[i]))
            probs = m_u.get(key)
            a = max(0, min(int(app[i]), V - 1))
            if probs is not None and a < probs.shape[0]:
                f1[i] = float(probs[a])
            else:
                # Fallback to per-user 1-step
                f1[i] = float(markov_stack[uid_idx[i], min(max(int(last[i]), 0), V - 1), a])

    # F2: co-FG matrix lookup
    co_fg = ctx["co_fg_per_user"]
    f2_self = np.zeros(n, dtype=np.float32)  # co_fg[uid][app, last]
    f2_mean = np.zeros(n, dtype=np.float32)  # mean over apps in B(t) of co_fg[uid][app, b]
    # Precompute per-anchor B(t) cat-of-apps lookup is expensive; we compute
    # mean co-FG over peers within the same anchor:
    for u in np.unique(uid_str):
        mat = co_fg.get(str(u))
        if mat is None:
            continue
        mask = uid_str == u
        a_idx = np.clip(app[mask], 0, V - 1)
        b_idx = np.clip(last[mask], 0, V - 1)
        f2_self[mask] = mat[a_idx, b_idx]
    # f2_mean: for each (anchor, app), mean over OTHER apps in B(t) of co_fg[app, b_other]
    # where b_other is the LAST_FG seen by that other peer at the same anchor (which is the same `last_fg_app_idx`!).
    # Simplified: for the same anchor and the same last_fg, all peers share `b`. So co_fg[app, b]
    # for each peer just becomes co_fg lookup of that peer's app idx with the shared b.
    df = bg_df.assign(_row=np.arange(n))
    for (uid, anchor_id), grp in df.groupby(["user_uid", "anchor_id"], sort=False):
        mat = co_fg.get(str(uid))
        if mat is None:
            continue
        rows = grp["_row"].to_numpy()
        b = int(np.clip(grp["last_fg_app_idx"].iloc[0], 0, V - 1))
        peers = np.clip(grp["app_idx"].to_numpy(dtype=np.int64), 0, V - 1)
        peer_co = mat[peers, b]
        for j, r in enumerate(rows):
            others = np.delete(peer_co, j)
            f2_mean[r] = float(others.mean()) if others.size > 0 else float(peer_co[j])

    return np.stack([f1, f2_self, f2_mean], axis=1).astype(np.float32)


def build_c34n_full_multi(schema: str, bg_df: pd.DataFrame, ctx: dict) -> dict:
    """c3.4n_cheap + 3 features (F1 + F2). 41 + 3 = 44 numeric features."""
    bg_df = attach_last2_fg(bg_df, ctx)
    base = build_c34n_cheap_multi("c3.4n_cheap", bg_df, ctx)
    extra = add_c34_full_features_multi(bg_df, ctx)
    base["features"] = np.concatenate([base["features"], extra], axis=1)
    return base
```

- [ ] **Step 10.3: Add the recipe**

```python
RECIPES["c3p4_full"] = {
    "schema": "c3.4n_full", "wide": False, "dropout": 0.2,
    "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.0,
}
```

And in `_build_data_dispatch`:

```python
def _build_data_dispatch(schema: str, bg_df: pd.DataFrame, ctx: dict) -> dict:
    if schema == "c3.4n_cheap":
        return build_c34n_cheap_multi(schema, bg_df, ctx)
    if schema == "c3.4n_full":
        return build_c34n_full_multi(schema, bg_df, ctx)
    return build_schema_data_multi(schema, bg_df, ctx)
```

- [ ] **Step 10.4: Train + eval**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python scripts/52_train_task_c_multiuser.py --recipes c3p4_full --epochs 30 --patience 4 2>&1 | tail -20
```

Then add `c3p4_full` to `TRAINED_PICKS` in 53/54/55, run eval:

```bash
python scripts/53_eval_h60_multiuser.py --include-slice-b 2>&1 | grep "C3.4" | tail -3
```

- [ ] **Step 10.5: Commit**

```bash
cd /home/mohan/Bandit_SFM
git add -A app_usage_data/scripts/ app_usage_data/artifacts/bg_multi/checkpoints/task_c_multi_c3p4_full.pt app_usage_data/artifacts/bg_multi/results/
git commit -m "c3.4 stage 2: add c3p4_full (44 numeric features) + train + eval"
```

### Task 11: Stage-2 GATE — proceed to Stage 3?

- [ ] **Step 11.1: Compare c3p4_full to c3p4_cheap (and to c3p3)**

```bash
python -c "
import json
for r in ('c3p3', 'c3p4_cheap', 'c3p4_full'):
    d = json.load(open(f'artifacts/bg_multi/results/task_c_multi_{r}.json'))
    print(f'{r}: feature_dim={d[\"feature_dim\"]} test_PR={d[\"test\"][\"pr_auc_mean\"]:.4f}')
"
```

Gate: if `(c3p4_full_PR − c3p4_cheap_PR) ≥ 0.003`, continue to Stage 3.
Else: stop here. Document.

---

## STAGE 3 — `c3.4_full` (architecture change: + cat_emb + B(t) composition embedding)

(Only execute these tasks if Stage-2 gate passed.)

### Task 12: Add `cat_emb` and `bg_comp_emb` to the model

**Files:**
- Modify: `app_usage_data/scripts/52_train_task_c_multiuser.py`

- [ ] **Step 12.1: Define a new model class `BgPairMLPCat`**

Append to `52_train_task_c_multiuser.py`:

```python
import torch
import torch.nn as nn


class BgPairMLPCat(nn.Module):
    """Single-head MLP with explicit cat_emb + bg_comp_emb (mean-pool of cat_emb over B(t))."""

    def __init__(self, vocab_size: int, num_categories: int, num_features: int,
                 app_emb_dim: int = 16, cat_emb_dim: int = 4, dropout: float = 0.2):
        super().__init__()
        self.app_emb = nn.Embedding(vocab_size, app_emb_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(num_categories, cat_emb_dim)
        in_dim = app_emb_dim + cat_emb_dim + cat_emb_dim + num_features
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, app_idx, features, cat_idx=None, bg_cat_indices=None, bg_mask=None):
        ae = self.app_emb(app_idx)                         # (B, app_emb_dim)
        ce = self.cat_emb(cat_idx)                         # (B, cat_emb_dim)
        # bg_cat_indices: (B, max_bg) long; bg_mask: (B, max_bg) bool
        bg_emb = self.cat_emb(bg_cat_indices)              # (B, max_bg, cat_emb_dim)
        m = bg_mask.float().unsqueeze(-1)                  # (B, max_bg, 1)
        bg_emb = (bg_emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        x = torch.cat([ae, ce, bg_emb, features], dim=-1)  # (B, in_dim)
        h = self.trunk(x)
        return self.head(h).squeeze(-1)


def make_baseline_with_cat_emb(num_features: int, vocab_size: int,
                                 num_categories: int = 11, dropout: float = 0.2):
    return BgPairMLPCat(
        vocab_size=vocab_size, num_categories=num_categories,
        num_features=num_features, app_emb_dim=16, cat_emb_dim=4,
        dropout=dropout,
    )
```

- [ ] **Step 12.2: Build B(t) composition tensors**

Append:

```python
def build_bg_cat_tensors(bg_df: pd.DataFrame, app_to_cat: np.ndarray,
                          max_bg: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """For each row, return:
        bg_cat_indices: (N, max_bg) int64 — categories of each app in this anchor's B(t)
        bg_mask:        (N, max_bg) bool — True where the cell is a real BG app
    Pad with zeros (cat 0 is "unknown" / harmless given mask).
    """
    n = len(bg_df)
    out_cat = np.zeros((n, max_bg), dtype=np.int64)
    out_mask = np.zeros((n, max_bg), dtype=bool)
    df = bg_df.assign(_row=np.arange(n))
    for _, grp in df.groupby(["user_uid", "anchor_id"], sort=False):
        rows = grp["_row"].to_numpy()
        apps = grp["app_idx"].to_numpy(dtype=np.int64)
        cats = app_to_cat[np.clip(apps, 0, len(app_to_cat) - 1)].astype(np.int64)
        k = min(len(cats), max_bg)
        for r in rows:
            out_cat[r, :k] = cats[:k]
            out_mask[r, :k] = True
    return out_cat, out_mask
```

- [ ] **Step 12.3: Build `build_c34_full_multi` schema**

Append:

```python
def build_c34_full_multi(schema: str, bg_df: pd.DataFrame, ctx: dict) -> dict:
    """c3.4n_full + cat_idx + bg_cat_indices + bg_mask for the cat-emb model.
    Numeric feature_dim stays at 44; we add embedding-side inputs."""
    base = build_c34n_full_multi("c3.4n_full", bg_df, ctx)
    a_idx = base["app_idx"]
    cat_idx = ctx["app_to_cat"][np.clip(a_idx, 0, len(ctx["app_to_cat"]) - 1)].astype(np.int64)
    base["cat_idx"] = cat_idx
    bg_cat_indices, bg_mask = build_bg_cat_tensors(bg_df, ctx["app_to_cat"], max_bg=30)
    base["bg_cat_indices"] = bg_cat_indices
    base["bg_mask"] = bg_mask
    return base
```

- [ ] **Step 12.4: Update `_build_data_dispatch`**

```python
    if schema == "c3.4_full":
        return build_c34_full_multi(schema, bg_df, ctx)
```

- [ ] **Step 12.5: Add the recipe**

```python
RECIPES["c3p4_full_cat"] = {
    "schema": "c3.4_full", "wide": False, "dropout": 0.2,
    "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.0,
    "use_cat_emb": True,
}
```

- [ ] **Step 12.6: Update `make_model` to dispatch on `use_cat_emb`**

```python
def make_model(recipe: dict, num_features: int, vocab_size: int) -> torch.nn.Module:
    if recipe.get("use_cat_emb"):
        return make_baseline_with_cat_emb(num_features=num_features,
                                            vocab_size=vocab_size,
                                            dropout=recipe["dropout"])
    if recipe["wide"]:
        return _GRID.make_wide_model(num_features=num_features, vocab_size=vocab_size,
                                      dropout=recipe["dropout"])
    return _GRID.make_baseline_model(num_features=num_features, vocab_size=vocab_size,
                                      dropout=recipe["dropout"])
```

- [ ] **Step 12.7: Update training loop to forward extra kwargs for cat-emb model**

In `train_recipe` and `evaluate_multi`, when calling the model, also pass `cat_idx`, `bg_cat_indices`, `bg_mask` if present in `data_d`. Modify the forward call site, e.g.:

```python
def _model_forward(model, b):
    kwargs = {}
    if "cat_idx" in b:
        kwargs["cat_idx"] = b["cat_idx"]
    if "bg_cat_indices" in b:
        kwargs["bg_cat_indices"] = b["bg_cat_indices"]
    if "bg_mask" in b:
        kwargs["bg_mask"] = b["bg_mask"]
    return model(b["app_idx"], b["features"], **kwargs)
```

And in `PairDS` (in `36_train_c3_grid.py:PairDS`), this won't have those fields — for c3.4 we need a different DS class. Add to `52_train_task_c_multiuser.py`:

```python
class PairDSCat(torch.utils.data.Dataset):
    def __init__(self, d):
        self.f = torch.as_tensor(d["features"].copy(), dtype=torch.float32)
        self.ai = torch.as_tensor(d["app_idx"].copy(), dtype=torch.long)
        self.ci = torch.as_tensor(d["cat_idx"].copy(), dtype=torch.long)
        self.bgci = torch.as_tensor(d["bg_cat_indices"].copy(), dtype=torch.long)
        self.bgm = torch.as_tensor(d["bg_mask"].copy(), dtype=torch.bool)
        self.y = torch.as_tensor(d["y"].copy(), dtype=torch.float32)
        self.aid = torch.as_tensor(d["anchor_id"].copy(), dtype=torch.long)

    def __len__(self):
        return self.f.shape[0]

    def __getitem__(self, i):
        return {"features": self.f[i], "app_idx": self.ai[i],
                "cat_idx": self.ci[i], "bg_cat_indices": self.bgci[i],
                "bg_mask": self.bgm[i], "y": self.y[i],
                "anchor_id": self.aid[i]}
```

Add a wrapper `train_one_cat` that mirrors `_GRID.train_one` but uses `PairDSCat` and `_model_forward`. (Easiest: copy `_GRID.train_one` into `52_train_task_c_multiuser.py` and modify in-place.)

- [ ] **Step 12.8: Train**

```bash
cd /home/mohan/Bandit_SFM/app_usage_data && python scripts/52_train_task_c_multiuser.py --recipes c3p4_full_cat --epochs 30 --patience 4 2>&1 | tail -20
```

- [ ] **Step 12.9: Eval**

Add `c3p4_full_cat` to `TRAINED_PICKS` in 53/54/55. Run:

```bash
python scripts/53_eval_h60_multiuser.py --include-slice-b
```

- [ ] **Step 12.10: Commit**

```bash
cd /home/mohan/Bandit_SFM
git add -A app_usage_data/scripts/ app_usage_data/artifacts/bg_multi/checkpoints/task_c_multi_c3p4_full_cat.pt app_usage_data/artifacts/bg_multi/results/
git commit -m "c3.4 stage 3: add c3p4_full_cat with cat_emb + B(t) composition embedding"
```

---

## Final task: update the multi-user report

### Task 13: Document the c3.4 ablation results

**Files:**
- Modify: `app_usage_data/REPORT_bgkill_multiuser.md`

- [ ] **Step 13.1: Append a new section §13 "c3.4 feature-expansion ablation"**

Add a table comparing c3p3 / c3p4_cheap / c3p4_full / c3p4_full_cat on test PR-AUC, ROC-AUC, FK@.5, MCC. Note which gate(s) passed and which features turned out to matter.

Template:

```markdown
## 13. c3.4 ablation — what helped, what didn't

| Recipe          | Schema (numeric dim) | Test PR-AUC | Δ vs c3p3 | Test FK@.5 | Test MCC@τ_keep | Outcome |
|-----------------|----------------------|------------:|----------:|-----------:|----------------:|---------|
| c3p3            | c3.3 (33)            | TBD         | (ref)     | TBD        | TBD             |         |
| c3p4_cheap      | c3.4n_cheap (41)     | TBD         | TBD       | TBD        | TBD             | gate?   |
| c3p4_full       | c3.4n_full (44)      | TBD         | TBD       | TBD        | TBD             | gate?   |
| c3p4_full_cat   | c3.4_full (44 + cat) | TBD         | TBD       | TBD        | TBD             | gate?   |

**Reads:**
- F3 / F5 / F6 / F7 (cheap features) lift PR-AUC by [TBD pp] over c3p3.
- F1 / F2 (heavier features) add [TBD pp] over c3p4_cheap.
- cat_emb + F4 (architecture change) add [TBD pp] over c3p4_full.
```

- [ ] **Step 13.2: Commit + push**

```bash
cd /home/mohan/Bandit_SFM
git add app_usage_data/REPORT_bgkill_multiuser.md
git commit -m "REPORT_bgkill_multiuser: add §13 c3.4 ablation results"
git push origin mohan
```

---

## Self-review checklist

**Spec coverage:** every feature F1–F7 + cat_emb is implemented and exercised in at least one trained recipe. Gate criteria are documented at every stage. ✓

**Placeholder scan:** every code block in this plan contains real Python; no "TBD" or "implement later" except in the final-report template (filled in after running). ✓

**Type consistency:** `compute_last2_fg_for_anchors` returns `np.ndarray (int64)`; consumed by `attach_last2_fg` which writes to a column also int64. `fit_dow_hour_freq` returns `(7, 24, V) float32`; `stack_per_user` returns `(U, 7, 24, V) float32`; consumed by `add_c34_cheap_features_multi` via `[uid_idx, dow, hr, app]` indexing. `BgPairMLPCat.forward` accepts the kwargs `cat_idx`, `bg_cat_indices`, `bg_mask` that match the dict keys produced by `build_c34_full_multi`. ✓

**Risks / mitigations:**
- *F1 fallback.* If `(last2, last)` triple is unseen at training time → fall back to per-user 1-step Markov (same as single-user `add_c34_col`). Already in implementation.
- *F4 padding.* `bg_cat_indices` padded with zeros + `bg_mask` masked-mean — guarded against div-by-zero with `clamp(min=1.0)`. Already in implementation.
- *Eval-script schema dispatch.* Each of `53/54/55` calls `build_schema_data_multi("c3.3", ...)` directly today. The plan replaces that with `_build_data_dispatch(rec["schema"], ...)`. Sanity-check after Task 6 by re-running 53 on the existing c3p3 ckpt and confirming PR-AUC unchanged.

---

## Compute estimate

| Stage | Time |
|------:|-----:|
| 1 | ~10 min train + ~2 min eval = ~12 min |
| 2 | ~12 min train + ~2 min eval = ~14 min |
| 3 | ~15 min train + ~2 min eval = ~17 min |
| **Total** | **~45 min** if all 3 stages run |

Plus dev time: ~3 hours implementation + tests.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-07-c34-multiuser-feature-expansion.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
