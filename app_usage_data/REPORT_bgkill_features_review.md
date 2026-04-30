# Task C — Feature-Space Review

A sequential walk-through of the inputs to the background-suspension model:

1. **C1** — the original schema (14 numeric features + 2 embeddings).
2. **C1 → C2** — what was dropped, what was added, and why.
3. **C2** — the current production schema (15 numeric features + 1 embedding).
4. **Proposed next features** — what's still missing and what I'd add next.

The question every feature has to help answer:

> *App `a` is currently in `B(t)`. Will the user foreground it in the next 60 min?*

> **Update (current experimental plan):** the focus has shifted from short-horizon (H=5/10) to **single-horizon H=60 min**. Class imbalance is much milder at H=1h (~18.7% positive at 6h-window train, ~25% at 2h-window train) and the deployment relevance is closer to actual OS eviction policies. **Sections below are pinned to H=1h** unless explicitly noted; the priority ordering changes accordingly. Multi-horizon framing remains useful as an *auxiliary* loss signal at training time, but it is no longer the primary head.

---

## 1. C1 — the original 14-feature model

**Architecture:** `app_idx → 16-d emb` ⊕ `cat_idx → 4-d emb` ⊕ 14 numeric features → MLP → 2 sigmoid heads (H=5, H=10). 5,422 params total.

### 1.1 Embeddings

| Embedding | Shape | What it captures |
|---|---|---|
| `app_emb` | (50, 16) | Learnable per-app identity. Lets the model store nuanced per-app behavior the numeric features can't. |
| `cat_emb` | (11, 4) | Learnable per-category vector across the v3 taxonomy (messaging / productivity / system / etc.). |

### 1.2 The 14 numeric features

| # | Feature | Plain English |
|---|---|---|
| 1 | `log_time_in_bg` | How long since this app went to background. **Core LRU signal.** |
| 2 | `log_time_since_fg` | How long since this app was last foregrounded. (Robust LRU under BG→FG→BG cycles.) |
| 3 | `log_fg_count_today` | How many times the user has opened this app since local midnight. |
| 4 | `age_bg_over_6h` | `time_in_bg / (6h)`, clipped — normalized BG age relative to the staleness cutoff. |
| 5 | `hour_sin_24` | Smooth hour-of-day encoding. |
| 6 | `hour_cos_24` | Smooth hour-of-day encoding. |
| 7 | `wday_sin` | Smooth weekday encoding. |
| 8 | `wday_cos` | Smooth weekday encoding. |
| 9 | `log_bg_set_size` | How crowded RAM is right now (`|B(t)|`). |
| 10 | `is_weekend` | 1 if weekday ∈ {Sat, Sun}. |
| 11 | `loc_match_flag` | 1 if last-FG location is known for this app. (A weak proxy.) |
| 12 | `daypart_match_flag` | 1 if app's last-FG daypart bin == anchor's daypart. |
| 13 | `markov_prob` | `P(this app | last-FG app)` from the train Markov-1 table. |
| 14 | `hour_cond_prob` | `P(this app | anchor hour)` from the train HourMFU table (Dirichlet-smoothed). |

**C1 result (test, H=5):** ROC-AUC **0.819**, FK@0.5 **0.0162**.
The strongest closed-form baseline (Markov-inverse) lands at ROC-AUC 0.806 / FK@0.5 0.0151 — so C1 only marginally beats it.

---

## 2. C1 → C2: what changed and why

After auditing each C1 input against *Task-C-specific* relevance, five numeric features and one embedding were dropped, and six new features were added.

### 2.1 Dropped from C1

| Dropped | Reason |
|---|---|
| `cat_emb` (4-d) | Redundant — `markov_prob` and `hour_cond_prob` already condition on app identity. The 44 params would re-learn what's already encoded. |
| `wday_sin`, `wday_cos` | Redundant with `is_weekend` + `daypart_match_flag` for a 5-min decision; daypart bins already split weekday vs weekend. |
| `age_bg_over_6h` | Monotonic transformation of `log_time_in_bg`; the model can derive it. |
| `log_bg_set_size` | Replaced by `bg_recency_min_norm`, which tells you *how stale* the BG set is, not just how *big*. |
| `loc_match_flag` | Weak proxy ("location known", not "location matches"). Dropped pending a real anchor-time `loc_id`. |

### 2.2 Added in C2

| New feature | What it captures (gap it closes) |
|---|---|
| `recency_rank_in_bg` | This app's rank by recency *within `B(t)`* — captures within-anchor competition. |
| `log_fg_count_last_1h` | Per-app rhythm at finer time scale than `today`. |
| `log_fg_count_last_6h` | Per-app rhythm at mid time scale. |
| `bg_recency_min_norm` | Anchor-level "freshness" — is the entire BG set fresh or stale? |
| `time_since_screen_on_norm` | Phone-activity context (active session vs idle). |
| `prev_killed_app_match` | 1 if this app was the OS's most recent kill (within last 10 min). |

---

## 3. C2 — the current 15-feature schema

**Architecture:** `app_idx → 16-d emb` ⊕ 15 numeric features → MLP → 2 sigmoid heads. **No `cat_emb`.** 5,186 params total.

### 3.1 Final feature list (15)

| # | Feature | Meaning | Status vs C1 |
|---|---|---|---|
| 1 | `log_time_in_bg` | LRU core: time since last `APP_BACKGROUND` for this app. | kept |
| 2 | `log_time_since_fg` | LRU robust: time since last FG for this app. | kept |
| 3 | `recency_rank_in_bg` | This app's rank in `B(t)` by recency, normalized to [0,1]. | **new** |
| 4 | `log_fg_count_today` | Times opened today. | kept |
| 5 | `log_fg_count_last_1h` | Times opened in the last hour. | **new** |
| 6 | `log_fg_count_last_6h` | Times opened in the last 6h. | **new** |
| 7 | `markov_prob` | `P(app | last_fg_app)` from train Markov-1. | kept |
| 8 | `hour_cond_prob` | `P(app | anchor_hour)` from train HourMFU. | kept |
| 9 | `bg_recency_min_norm` | log1p(min `time_in_bg_sec` across **all apps in B(t)**, i.e. the *freshest* BG app's age) / log1p(6h). | **new** |
| 10 | `time_since_screen_on_norm` | log1p(seconds since SCREENON, capped at 1h) / log1p(1h). | **new** |
| 11 | `prev_killed_app_match` | 1 if app == most-recently OS-killed (within 10 min). | **new** |
| 12 | `hour_sin_24` | Smooth hour. | kept |
| 13 | `hour_cos_24` | Smooth hour. | kept |
| 14 | `daypart_match_flag` | 1 if last-FG daypart == anchor daypart. | kept |
| 15 | `is_weekend` | 1 if Sat/Sun. | kept |

**C2 result (test, H=5):** ROC-AUC **0.829** (+1.0 pp vs C1), FK@0.5 **0.0157**.

---

## 4. What's still missing in C2

C2 only beats Markov-inverse by +2.3 pp ROC-AUC. Auditing each *signal class* against C2's coverage exposes four real gaps:

| Signal class | Covered in C2? | Gap |
|---|---|---|
| Self-recency | ✅ (#1, #2, #3) | strong |
| Per-app rhythm | ✅ (#4, #5, #6) | strong |
| 1-step transition | ✅ (#7) | only 1-hop |
| Time-of-day priors | ✅ (#8, #12-#15) | adequate |
| **Higher-order Markov** (P(app | last-2 FG)) | ❌ | round-trip transitions look identical to 1-hop |
| **Periodicity** (clock-time yesterday / week-ago) | ❌ | no signal that "this is when this app is usually due" |
| **Active-session intensity** (FG events in last 1-5 min) | ⚠️ only `time_since_screen_on` (coarse) | can't tell "tapping right now" from "screen on but idle" |
| **Per-app overdue normalization** (vs the app's own typical interval) | ⚠️ implicit via #2 | `time_since_fg` is on absolute scale, doesn't bake in per-app rhythm |
| OS-eviction echo | ⚠️ only #11 (single bit) | no kill-history beyond the most recent event |

---

## 5. Proposed new features — comprehensive review

This iteration cross-checks every entry in `feature_engineering_readme.md` (the general phone-usage feature catalogue) against C2's current schema, then organizes the missing signals by **time scale** (lifetime → long-term → medium → short-term → instant) so we can reason about coverage instead of just adding ad-hoc features.

### 5.0 The readme → C2 cross-walk

Each row of the readme, mapped to C2's coverage and Task-C relevance:

| Readme group | What it lists | Already in C2? | New for Task C? |
|---|---|---|---|
| §1 Identity & metadata | app_id, category, **is_system_app, launch_cost**, region, interaction_intensity | only `app_emb` | **`is_system_app`, `launch_cost` — yes** |
| §2 Time | hour, weekday, week_of_year, day_of_month, is_weekend, time_segment | hour, is_weekend, daypart | nothing high-value at 5-min decision cadence |
| §3 Cyclical encoding | hour/dow/minute sin/cos | hour sin/cos only | sub-hour useless on 5-min grid |
| §4 Sequential | prev_app, prev_category, time_since_prev, **last_K_app_id** | `markov_prob` (1-step only) | **last-2 / last-3 FG apps**, **prev_cat** |
| §5 BG state | bg_count, bg_min/mean/max recency, **bg_category_list**, **num_kills_in_window**, contains_prev | `bg_recency_min`, `bg_set_size` (dropped) | **bg_recency_mean / max, bg_unique_cat_count, num_recent_kills** |
| §6 FG entry type | is_start_app, **is_reuse_from_background** | none | **`prev_fg_was_reuse_from_bg`** — could matter |
| §7 Kill events | is_explicit_kill, killed_app_id | `prev_killed_app_match` (10-min) | **kills in last 30 min, lifetime kill rate** |
| §8 Session features | **session_id, position_in_session, session_elapsed, event_count, unique_apps, dom_cat** | `time_since_screen_on` only | **all 6 session features missing** — biggest gap |
| §9 Historical | **hist_app_count, hist_app_prob, hist_cat_count, hist_cat_prob** | none directly | **lifetime per-app share, lifetime cat share** |
| §10 Transition prob | app→app, **cat→cat** | only app→app (`markov_prob`) | **`cat_markov_prob`** — denser |
| §11 Rolling windows | count(N min), unique_apps, total_events | per-app 1h, 6h | **5-min, 30-min** windows; **anchor-level events_last_5min, unique_apps_last_5min** |
| §12 Recency | **time_since_cat_last_used** | `time_since_fg` | **per-category recency** |
| §13 Network | wifi/cellular | none | skip (single user) |

**Net-new directions surfaced:**

1. **Session-state features** (§8) — totally missing in C2.
2. **Category-level analogues** of every per-app signal (cat-Markov, cat-recency, cat-co-occurrence in B(t)) — denser estimates.
3. **Multi-scale rolling counts** — fill the 5-min / 30-min gaps.
4. **Lifetime priors** — `is_system_app`, popularity share, kill rate, mean inter-FG.
5. **BG-set composition** — currently only `min` recency is captured; add mean / max / unique cat count.
6. **Higher-order transitions** (already F4) and **periodicity** (already F5).

---

### 5.1 Proposed feature set, organized by time scale

Each feature has a **scope** (anchor-level shared by all rows, vs per-(anchor, app) row-specific), a **time-scale** (instant / short / medium / long / lifetime), and an indication of whether it was already in C1 / C2 / new.

#### A. Lifetime priors (anchor-independent; fit on train, frozen at inference)

These are scalar-per-app constants. One-time fit on train, lookup at inference.

| Feature | Scope | Source | Status |
|---|---|---|---|
| `is_system_app` | per-app | hand-labelled | NEW |
| ~~`launch_cost_class`~~ | per-app | — | **DROPPED** — describes UX cost of relaunch, not P(re-use). Belongs in post-model deployment policy, not as a model input. |
| `app_lifetime_pct_share` | per-app scalar | `count(app, train) / total_count(train)` | NEW |
| `app_lifetime_kill_rate` | per-app scalar | `#PROCESS_EXIT(a, ≤10min after BG) / #BG(a)` on train | NEW (was F6) |
| `app_lifetime_mean_inter_fg_sec` | per-app scalar | mean of inter-FG intervals on train | NEW (denominator of F2) |
| `cat_lifetime_share` | per-cat scalar | sum of app shares in same category | NEW |

#### B. Long-term per-app rhythm (≥ 24 h)

| Feature | Scope | Status |
|---|---|---|
| `log_fg_count_today` | per-pair | ✅ in C2 |
| `log_fg_count_last_24h` | per-pair | NEW (rolling 24h, no midnight artefact) |
| `log_fg_count_last_7d` | per-pair | NEW (long-term habit signal) |
| `was_fg_24h_ago_in_30min` | per-pair binary | NEW (was F5 — periodicity) |
| `was_fg_7d_ago_in_30min` | per-pair binary | NEW (was F5 — weekly periodicity) |

#### C. Medium-term per-app rhythm (30 min – 6 h)

| Feature | Scope | Status |
|---|---|---|
| `log_fg_count_last_6h` | per-pair | ✅ in C2 |
| `log_fg_count_last_1h` | per-pair | ✅ in C2 |
| `log_fg_count_last_30min` | per-pair | **new** — fills the gap below 1h |
| `time_since_category_last_used_sec` | per-pair (log-norm) | **new** — when did the user last open *any* app of this category? |

#### D. Short-term per-app and active-session intensity (< 5 min)

| Feature | Scope | Status |
|---|---|---|
| `log_fg_count_last_5min` | per-pair | NEW (cf. category-level below) |
| `was_fg_in_last_60s` | per-pair binary | NEW (very strong negative signal — if app was just FG, it's not getting killed) |
| `is_in_active_session` | anchor-level | NEW (was F1) — `time_since_screen_on < 60s` |
| `log_fg_events_last_60s` | anchor-level | NEW (was F1) |
| `log_fg_events_last_5min` | anchor-level | NEW (was F1) |
| `unique_apps_last_5min` | anchor-level | NEW |

#### E. Instant / current-anchor signals (< 30 s)

| Feature | Scope | Status |
|---|---|---|
| `log_time_in_bg` | per-pair | ✅ in C2 |
| `log_time_since_fg` | per-pair | ✅ in C2 |
| `recency_rank_in_bg` | per-pair | ✅ in C2 |
| `time_since_screen_on_norm` | anchor | ✅ in C2 |
| `time_since_screen_off_norm` | anchor | NEW |

#### F. Anchor-level: BG-set composition

These are **summary statistics over all apps in `B(t)`** at the anchor — anchor-level scalars broadcast to every row of that anchor.

`bg_recency_min/mean/max` are computed across the *entire* `B(t)` set: take each app's `time_in_bg_sec` (= `t − last_bg_ts(app)`), then aggregate across all apps in B(t) at this anchor — *not* just one specific app. The "min" tells the model "the freshest BG app is X seconds old"; "max" tells it "the oldest BG app is Y seconds old"; "mean" gives an average freshness signal.

Formally, with apps `a_1, …, a_K ∈ B(t)`:

```
ages = [t − last_bg_ts(a_i)  for a_i ∈ B(t)]
bg_recency_min  = min(ages)
bg_recency_mean = mean(ages)
bg_recency_max  = max(ages)
```

Each is then `log1p(...) / log1p(2*3600)` to fit in [0, 1] for the new 2 h staleness window.

Adding the rest of the BG-set summaries:

| Feature | Scope | Status |
|---|---|---|
| `log_bg_set_size` | anchor | re-add (was in C1, dropped in C2) |
| `bg_recency_min_norm` | anchor | ✅ in C2 |
| `bg_recency_mean_norm` | anchor | NEW |
| `bg_recency_max_norm` | anchor | NEW |
| `bg_unique_category_count` | anchor | NEW |
| `bg_apps_in_same_category` | per-pair | NEW (count of *other* apps in B(t) sharing this app's category) |
| `bg_contains_last_fg_app` | anchor binary | NEW |

#### G. Anchor-level: session state (300-s idle gap = new session)

This is the biggest cluster of missing signals from the readme.

| Feature | Scope | Status |
|---|---|---|
| `log_session_elapsed_sec` | anchor | NEW |
| `log_session_event_count_so_far` | anchor | NEW |
| `log_session_unique_apps_so_far` | anchor | NEW |
| `session_dominant_category` | anchor (one-hot 11-d) | NEW |
| `prev_fg_was_reuse_from_bg` | anchor binary | NEW (was the last FG event a reuse of an app in B(t) at the time?) |

#### H. Higher-order transitions

| Feature | Scope | Status |
|---|---|---|
| `markov_prob` (1-step) | per-pair | ✅ in C2 |
| `markov_prob_2step` | per-pair | NEW (was F4) |
| `cat_markov_prob` | per-pair | NEW (P(cat | cat(last_fg)); denser than app-level) |
| `category_match_to_last_fg` | per-pair binary | NEW (was F3) |

#### I. Lifetime / device-pressure signals

| Feature | Scope | Status |
|---|---|---|
| `prev_killed_app_match` | per-pair binary | ✅ in C2 |
| `num_kills_last_30min` | anchor scalar | NEW |
| `app_lifetime_kill_rate` | per-pair (lifetime prior) | NEW (was F6) |
| `time_since_screen_on_norm` | anchor | ✅ in C2 |
| `time_since_screen_off_norm` | anchor | NEW (captures session length, complements screen_on) |

---

### 5.2 Final consolidated feature list (proposed `C3`)

If we ship every feature above, the schema becomes:

| Group | Features | Count |
|---|---|---|
| Identity (lifetime priors) | is_system_app, lifetime_share, lifetime_kill_rate, lifetime_mean_inter_fg, cat_lifetime_share | 5 |
| Per-app recency (instant) | time_in_bg, time_since_fg, recency_rank_in_bg | 3 (kept) |
| Per-app rhythm (multi-scale) | fg_count_last_5min / 30min / 1h / 6h / today / 24h / 7d | 7 |
| Per-app priors | markov_prob, markov_prob_2step, cat_markov_prob, hour_cond_prob, overdue_ratio, was_fg_24h_ago, was_fg_7d_ago | 7 |
| Per-app context | category_match_to_last_fg, time_since_category_last_used, was_app_fg_in_last_60s | 3 |
| Anchor: session intensity | is_active_session, fg_events_last_60s, fg_events_last_5min, unique_apps_last_5min | 4 |
| Anchor: session state | session_elapsed, session_event_count, session_unique_apps, session_dom_cat (11-d), prev_fg_was_reuse | 4 + 11 |
| Anchor: BG composition | bg_set_size, bg_rec_min/mean/max, bg_unique_cats, bg_contains_last_fg, bg_apps_same_cat | 7 |
| Anchor: device pressure | time_since_screen_on, time_since_screen_off, num_kills_30min, prev_killed_app_match | 4 |
| Anchor: time-of-day | hour_sin / cos, daypart_match, is_weekend | 4 |

**Total numeric: ~50 features.** Plus the 16-d `app_emb`. Total params ≈ 8 k–10 k for the same 64-32 trunk (vs C2's 5.2 k).

---

### 5.3 Priority order — revised for H=1h focus

The previous priority list (below) was tuned for short-horizon (H=5/10) prediction, where active-session intensity was the strongest signal. With the experimental focus shifting to **single-horizon H=60 min**, priorities flip: hourly-cadence features (periodicity, lifetime rhythm, daily/weekly counts) now lead, and 60-second activity bursts are demoted.

#### New priority order (six stages, ordered for H=60)

| Stage | Bundle | What it adds | Expected lift @ H=60 |
|---|---|---|---|
| **C3.1 — Hourly habit** | F2 `overdue_ratio` + F5 periodicity (`was_fg_24h_ago`, `was_fg_7d_ago`) + `log_fg_count_last_24h` + `log_fg_count_last_7d` | +1.5 to +2.5 pp |
| **C3.2 — Identity & BG composition** | F3 `category_match`, `is_system_app`, `app_lifetime_pct_share`, `app_lifetime_kill_rate`, `cat_lifetime_share`, `bg_recency_mean/max`, `bg_unique_category_count`, `log_bg_set_size` | +0.5 to +1 pp |
| **C3.3 — Category-aware** | `cat_markov_prob`, `time_since_category_last_used`, `bg_apps_in_same_category` | +0.3 to +0.8 pp |
| **C3.4 — Higher-order Markov** | F4 `markov_prob_2step` (sparse table, 1-step fallback) | +0 to +1 pp |
| **C3.5 — Session state** | `session_elapsed`, `session_event_count`, `session_unique_apps`, `session_dom_cat`, `prev_fg_was_reuse_from_bg`, `num_kills_last_30min`, `time_since_screen_off_norm` | +0.3 to +0.8 pp (was top priority for H=5; demoted for H=60) |
| **C3.6 — Active-session intensity** | F1 (`is_in_active_session`, `log_fg_events_last_60s/5min`, `unique_apps_last_5min`, `was_app_fg_in_last_60s`, `log_fg_count_last_5min/30min`) | +0.0 to +0.3 pp (lowest priority for H=60; would be C3.1 if H=5) |

#### Why the order changed

Going from H=5 → H=60, the dominant signals shift from **"is the user actively tapping right now"** (60-second/5-minute bursts) to **"does this app have a daily/hourly habit at this exact clock-time"** (24h/7d periodicity, hourly-conditional probability). Active-session intensity has roughly 60-second decay relevance — it tells you what happens in the *next minute*, not the *next hour*. Conversely, periodicity bits (was_fg_24h_ago) directly encode "the user does this every day at 2 PM" which is exactly the H=60 question.

#### Old order (kept for reference, was tuned for H=5)

The earlier ordering, kept here in case the H-focus shifts back. **DO NOT use this for the H=60 plan.**

#### Stage C2.1 — *Cheap & high-leverage* (~1.5 h)
- F3: `category_match_to_last_fg`
- F1: `is_in_active_session`, `log_fg_events_last_5min`, `log_fg_events_last_60s`
- A-extras: `is_system_app`, `app_lifetime_pct_share`
- BG-comp easy: `bg_recency_mean`, `bg_recency_max`, `bg_unique_category_count`

**Expected:** +1 to +2 pp ROC-AUC. Fills the biggest gaps with zero pipeline rework.

#### Stage C2.2 — *Per-app rhythm extension* (~1 h)
- F2: `overdue_ratio` + `app_lifetime_mean_inter_fg`
- `log_fg_count_last_5min`, `log_fg_count_last_30min`
- `log_fg_count_last_24h`, `log_fg_count_last_7d`
- `was_app_fg_in_last_60s` (binary)

**Expected:** +0.5 to +1 pp. Fills the rhythm gap top-to-bottom.

#### Stage C2.3 — *Session-state features* (~3 h)

Requires plumbing session_id forward through the state machine.

- `session_elapsed_sec`, `session_event_count_so_far`, `session_unique_apps_so_far`
- `session_dom_category` (11-d one-hot)
- `prev_fg_was_reuse_from_bg`
- `num_kills_last_30min`
- `time_since_screen_off_norm`

**Expected:** +0.5 to +2 pp. Highest engineering cost; may also be the single biggest single-stage lift since session-state is the largest gap from the readme audit.

#### Stage C2.4 — *Category-aware features* (~2 h)
- `cat_markov_prob` (11×11 dense table)
- `time_since_category_last_used_sec`
- `bg_apps_in_same_category` (per-pair)

**Expected:** +0.3 to +0.8 pp. Compensates for app-Markov sparsity, especially for tail apps.

#### Stage C2.5 — *Higher-order transitions* (~2 h)
- F4: `markov_prob_2step`

May land flat due to `(V, V, V)` sparsity — gracefully falls back to 1-step.

#### Stage C2.6 — *Periodicity* (~1 h)
- F5: `was_fg_24h_ago`, `was_fg_7d_ago`
- `cat_was_fg_24h_ago` (denser version)

**Expected:** unknown — v4 falsified this for Task A; Task C may differ.

#### Stage C2.7 — *Long-term per-app priors* (~30 min)
- F6: `app_lifetime_kill_rate`
- ~~`launch_cost_class`~~ — **DROPPED** (not predictive of re-use; belongs in deployment-time policy)

Nearly free. Add after C2.1–C2.4 settle.

---

### 5.4 What I deliberately won't add

| Skipped | Reason |
|---|---|
| `subcategory`, `app_region`, `interaction_intensity` from readme | No taxonomy; single-user single-region; intensity is implicit in fg_count features. |
| Network features (`networktype`, `wifiqoe`, `is_wifi`) | Single user; no heterogeneity to exploit. |
| Sub-minute cyclic features (`minute_sin/cos`) | 5-min anchor stride makes them too fine. |
| Anchor-time real `loc_id` | ~2 days of pipeline work for likely <1 pp lift; v3 found loc gains marginal. Defer until F1-F5 land. |
| Long-history token sequence (16-event window) | Architectural change, not a feature change. Out of scope. |
| Re-adding `cat_emb` (4-d) | F3 (1-bit category match) covers the useful slice. |
| `wday_sin/cos` | Captured by `is_weekend` + `daypart_match`. |
| Per-app interval distribution beyond mean (variance, percentiles) | Diminishing returns at 47k events; the mean is what's identifiable. |

---

## 6. Honest take

C2 → C3 with the proposed bundles plausibly buys **+2 to +4 pp ROC-AUC** at H=60 (cumulative across C3.1–C3.6). At H=60 the dominant gap is **hourly-habit signals** (periodicity, overdue_ratio, 24h/7d lifetime counts) — these are entirely missing from C2 and directly target the question being asked. **Session-state** (§5.G) and **active-session intensity** (§D) drop in priority at this horizon, because 60-min predictions don't depend strongly on 60-second activity bursts.

Beyond that, the model's ceiling is set by the **per-pair BCE loss** treating apps in the same anchor as independent — that's an architectural problem, not a feature problem. Once features plateau, the next move is listwise / softmax-over-`B(t)` ranking, which is a separate effort tracked outside this report.

> **Recommendation.** Ship Stages C2.1–C2.3 as a single `C3` rev (largest combined lift, ~5 h total work). Re-evaluate before doing C2.4–C2.7.

---


## Appendix A — Class enumerations

For every categorical / discrete input the model touches, here are the actual labels.

### A.1 App vocabulary — 50 tokens (`artifacts/vocab.json`)

3 reserved + 47 real apps. Apps appearing < 5 times in train were collapsed into `<RARE>` before the vocab was frozen.

| Idx | App label | Idx | App label | Idx | App label |
|---|---|---|---|---|---|
| 0 | `<PAD>` | 17 | `HOS` | 34 | `CONNECTMOBILE` |
| 1 | `<UNK>` | 18 | `SHELL_ASSISTANT` | 35 | `MEIJU` |
| 2 | `<RARE>` | 19 | `TAKEAWAY` | 36 | `HWSTARTUPGUIDE` |
| 3 | `LITE` (~37 % of all targets) | 20 | `XHS_HOS` | 37 | `6917584437461147371` (anonymized hash) |
| 4 | `WECHAT` (~17 %) | 21 | `NOTEPAD` | 38 | `ENTERPRISEAPP` |
| 5 | `XXXXXX` (anonymized; ~11 %) | 22 | `VISIONGLASS` | 39 | `THEMEMANAGER` |
| 6 | `CALLUI` | 23 | `VASSISTANT` | 40 | `IDLEFISH4OHOS` |
| 7 | `PHONE_CONTACTS` | 24 | `CAMERA` | 41 | `TOTEMWEATHER` |
| 8 | `CLOCK_CALENDAR` | 25 | `MOBILE` | 42 | `PMOBILE` |
| 9 | `MMS` | 26 | `TAOBAO4HMOS` | 43 | `HM` |
| 10 | `GALLERY` | 27 | `BETACLUB` | 44 | `6917591957328828429` (anon) |
| 11 | `HEALTH` | 28 | `FILES` | 45 | `WALLET` |
| 12 | `SAMPLEMANAGEMENT` | 29 | `NEXT` | 46 | `XTCWATCH` |
| 13 | `AWEME` | 30 | `BROWSER` | 47 | `PERSONAL` |
| 14 | `HMOS` | 31 | `HMEITUAN` | 48 | `VIDEO` |
| 15 | `SYSTEM_SETTINGS` | 32 | `INTELLIGENTUI` | 49 | `MEETIMESERVICE` |
| 16 | `MALL` | 33 | `INTELLIGENT` | | |

(The vocab is frozen at training time; index ↔ app is stable across all C1 / C2 / future C2.x runs. Top-3 apps cover ~65 % of targets.)

### A.2 Category taxonomy — 11 classes (`lib/v3/categories.py`)

Hand-built; used by both v3 (per-token `cat_emb`) and Task C v2 (proposed F3 `category_match_to_last_fg`).

| Idx | Category | Apps mapped here (examples) |
|---|---|---|
| 0 | `<PAD>` | (reserved for padding) |
| 1 | `social_messaging` | WECHAT |
| 2 | `news_feed_content` | LITE, BROWSER |
| 3 | `shopping_marketplace` | MALL, TAKEAWAY, TAOBAO4HMOS, HMEITUAN, IDLEFISH4OHOS, WALLET |
| 4 | `telephony_calling` | CALLUI, MMS, MEETIMESERVICE |
| 5 | `contacts_identity` | PHONE_CONTACTS |
| 6 | `media_camera` | CAMERA, GALLERY |
| 7 | `entertainment_video` | AWEME, MEIJU, VIDEO, XHS_HOS |
| 8 | `productivity` | CLOCK_CALENDAR, NOTEPAD, FILES, ENTERPRISEAPP |
| 9 | `system_utility` | SYSTEM_SETTINGS, HMOS, HOS, HEALTH, SHELL_ASSISTANT, VASSISTANT, INTELLIGENT, INTELLIGENTUI, BETACLUB, MOBILE, PMOBILE, HWSTARTUPGUIDE, THEMEMANAGER, TOTEMWEATHER, VISIONGLASS, XTCWATCH, SAMPLEMANAGEMENT, HM, PERSONAL, CONNECTMOBILE |
| 10 | `other_app` | Anything not explicitly mapped (e.g. the two `6917…` hashed app IDs) |

48 of 50 vocab apps are explicitly mapped; the rest fall back to `other_app`.

### A.2 Daypart bins (10 classes, `lib/v3/daypart.py`)

Used by `daypart_match_flag` in both C1 and C2.

| Idx | Daypart | Hour range | Notes |
|---|---|---|---|
| 0 | `early_morning` | 05–07 (any day) | |
| 1 | `morning_commute` | 07–09 weekday | weekend hours fall to bin 9 |
| 2 | `morning_work` | 09–12 weekday | same caveat |
| 3 | `noon` | 12–14 (any day) | |
| 4 | `afternoon_work` | 14–17 weekday | |
| 5 | `evening_commute` | 17–19 weekday | |
| 6 | `evening_home` | 19–22 (any day) | |
| 7 | `night` | 22–24 (any day) | |
| 8 | `late_night` | 00–05 (any day) | overrides everything else for `hour ∈ [0, 5)` |
| 9 | `weekend_daytime` | 07–22 weekend | overrides bins 1, 2, 4, 5, 6, 7 on Sat/Sun |

Logic: `lib/v3/daypart.py::daypart_bin(hour, weekday)`.

### A.3 Location vocabulary — 18 buckets (`artifacts/v3/loc_vocab.pkl`)

Built on **train only**: 3 reserved + 15 top-frequency parsed labels from `device_state_update_payload` (WiFi SSID or cell ID), forward-filled within each calendar day. The actual SSIDs / cell IDs are user-identifying and partially anonymized in the saved vocab; the table shows the *structure*, with examples masked.

| Idx | Bucket | Train count |
|---|---|---|
| 0 | `<PAD>` | – |
| 1 | `<NONE>` (no location resolved) | – |
| 2 | `<OTHER>` (parsed but not in top-15) | – |
| 3 | `wifi:Hua…yee` (likely home WiFi) | 7,174 |
| 4 | `cell:51…` (cell tower #1) | 5,511 |
| 5 | `wifi:Xia…B09` (likely office WiFi) | 3,874 |
| 6 | `cell:51…2` | 2,668 |
| 7 | `cell:51…3` | 2,614 |
| 8 | `cell:51…4` | 1,221 |
| 9 | `cell:51…5` | 912 |
| 10 | `cell:51…6` | 894 |
| 11 | `cell:51…7` | 793 |
| 12 | `wifi:public_2` | 758 |
| 13 | `cell:51…8` | 702 |
| 14 | `cell:51…9` | 569 |
| 15 | `wifi:public_3` | 467 |
| 16 | `cell:51…10` | 386 |
| 17 | `cell:51…11` | 320 |

(SSIDs / cell IDs masked in this report.) Coverage after forward-fill within day: train 97.7 % / val 98.8 % / test 99.0 % of rows resolve to a non-`<NONE>` bucket.

C1 used only the binary `loc_match_flag = 1 if last_fg_loc_id > 0` (i.e. *"location is known"*, not *"location matches"*). C2 dropped this; a real anchor-time `loc_id` is queued as future work.

### A.4 Event-type taxonomy (`evt_type`, 4 classes, `lib/data.py::EVENT_TYPES`)

Used by v1/v2/v3 token features but **not directly by Task C** (Task C is per-anchor, not per-token).

| Idx | Class | Source `name_norm` events |
|---|---|---|
| 0 | `TARGET` | `APP_FOREGROUND`, `APP_START` (the labelled events) |
| 1 | `BACKGROUND` | `APP_BACKGROUND` |
| 2 | `PAGE_SWITCH` | `ABILITY_OR_PAGE_SWITCH` |
| 3 | `OTHER` | `SCREENON_EVENT`, `SCREENOFF_EVENT`, `INTO_HOME_KEY`, `INTOKEYGUARD`, `DEVICE_STATE_UPDATE`, `PROCESS_START`, `PROCESS_EXIT`, `STARTUP_TIME`, `DH_APP_START_TIME` |

The Task C state machine in `lib/bg/background_state.py` consumes these classes directly to update the BG set: FG / START → status `FG`, BACKGROUND → `BG`, PROCESS_EXIT → `DEAD`, PROCESS_START → records last-start, SCREENON → updates anchor-level `time_since_screen_on`.

### A.5 Scene & network classes — dropped in C2

Both were per-token v1 features.

| Scene idx | Value | Network idx | Value |
|---|---|---|---|
| 0 | `-1` (unknown) | 0 | `-1` (unknown) |
| 1 | `1` | 1 | `0` |
| 2 | `2` | 2 | `1` |
| 3 | `3` | 3 | `2` |
| 4 | `4` | – | – |

`scene` was 86 % null on this user's data and dropped in v3 R6-arch trim → propagated into Task C v2.

### A.6 Horizons / labels

The current parquet (`bg_{train,val,test}.parquet`, 2 h staleness pipeline rev) carries four horizon labels per `(anchor, app)` row:

| Label column | Horizon | Train pos-rate | Val pos-rate | Test pos-rate | Use |
|---|---|---|---|---|---|
| `y_300` | 5 min | 4.25 % | 4.97 % | — | historical (C1/C2 H=5 head) |
| `y_600` | 10 min | — | — | — | historical (C1/C2 H=10 head) |
| `y_1800` | 30 min | — | — | — | secondary; usable as auxiliary head |
| `y_3600` | 60 min | **25.23 %** | **28.10 %** | **21.69 %** | **headline** — matches realistic OS-eviction cadence |

The H=60 column (`y_3600`) is the single training target across §5.4's grid (C3.1 → C3.3 + all C3-Pro variants). The shorter horizons remain in the parquet so a future multi-task model can use them as auxiliary heads (see `REPORT_bgkill_model_plan.md` §7).

> **Note on historical numbers.** §1, §3, and §A entries that quote C1 / C2 ROC-AUC at H=5 (e.g. 0.819, 0.829) come from the original H=5 pipeline. After the H=60 + 2 h-staleness pivot they are kept as history; the live numbers are in `REPORT_bgkill_v3.md` §5.

---

## Appendix B — Reproducing the feature pipeline end-to-end

This section shows the exact commands to regenerate every feature schema in this document from the raw event log. All scripts live under `app_usage_data/scripts/` and run on CPU in <10 minutes total (single user, ~35 k events).

### B.1 File flow

```
app_usage_cleaned_dictionary_mapped.xlsx           ← raw event log (input)
            │
            │   scripts/01_prep_data.py
            ▼
artifacts/splits/{train,val,test}.parquet          ← chronological event splits
artifacts/vocab.json                               ← 50-token app vocab (frozen)
            │
            │   scripts/20_build_v3_features.py    (v3 stat fits, used by C2/C3)
            ▼
artifacts/v3/markov_prior.pkl                      ← Markov-1 + cat priors
            │
            │   scripts/30_build_bg_data.py
            ▼
artifacts/bg/splits/bg_{train,val,test}.parquet    ← per-(anchor, app) rows + 4 horizons
artifacts/bg/stats/bg_data_stats.json              ← anchor-count + pos-rate sanity
            │
            │   scripts/31_run_baselines_bg.py
            ▼
artifacts/bg/results/baselines_bg.json             ← Random / LRU / Markov-inv / etc.
            │
            │   scripts/34_train_task_c_h60.py      (C1, C2, C3.1)
            │   scripts/36_train_c3_grid.py         (C3.2, C3.3, C3.4 + C3-Pro variants)
            │   scripts/37_train_c3pro_ensemble.py  (5-seed ensemble)
            ▼
artifacts/bg/checkpoints/task_c_*.pt
artifacts/bg/results/task_c_*.json
            │
            │   scripts/35_eval_h60.py              (bootstrap CIs + Pareto)
            ▼
artifacts/bg/results/h60_leaderboard_ci.json
figures/bg/pareto_h60_{val,test}.png
```

### B.2 One-shot reproduction commands

Working directory = `app_usage_data/`. Optional `OMP_NUM_THREADS=4` if the machine is shared (PyTorch greedily grabs every core otherwise).

```bash
# Stage 1 — raw events to chronological splits + vocab (~30 s)
python scripts/01_prep_data.py

# Stage 2 — Markov-1 + V×V transition prior fit on train (~5 s)
python scripts/20_build_v3_features.py

# Stage 3 — B(t) state machine, 2 h staleness, multi-horizon labels (~60 s)
python scripts/30_build_bg_data.py

# Stage 4 — closed-form baselines (~10 s)
python scripts/31_run_baselines_bg.py

# Stage 5 — train models
#   5a: C1 (legacy 14 features + cat_emb + app_emb)
python scripts/34_train_task_c_h60.py --schema c1   --tag C1_h60
#   5b: C2 (15 v2 features + app_emb)
python scripts/34_train_task_c_h60.py --schema c2   --tag C2_h60
#   5c: C3.1 (= C2 + 5 hourly-habit features)
python scripts/34_train_task_c_h60.py --schema c3.1 --tag C31_h60
#   5d: cumulative ablation C3.2 / C3.3 / C3.4
python scripts/36_train_c3_grid.py --variant c3.2
python scripts/36_train_c3_grid.py --variant c3.3
python scripts/36_train_c3_grid.py --variant c3.4
#   5e: C3-Pro architectural variants (on c3.3 = winning schema)
python scripts/36_train_c3_grid.py --variant c3pro_listwise --pro-schema c3.3
python scripts/36_train_c3_grid.py --variant c3pro_wide     --pro-schema c3.3
python scripts/36_train_c3_grid.py --variant c3pro_reg      --pro-schema c3.3
python scripts/36_train_c3_grid.py --variant c3pro_full     --pro-schema c3.3
python scripts/37_train_c3pro_ensemble.py c3.3   # 5-seed ensemble

# Stage 6 — bootstrap CIs + Pareto figures
python scripts/35_eval_h60.py

# Stage 7 — feature-pipeline validation (sanity checks)
python scripts/38_generate_features.py
python scripts/38_generate_features.py --schemas c3.3 --splits train val test
```

### B.3 Feature-build code pointer table

For each schema this table identifies the function that materialises the per-(anchor, app) feature matrix. All builders take a `bg_*.parquet` slice + train-only stats and return `{"features": np.ndarray, "app_idx": ..., "y": ..., "anchor_id": ..., "cat_idx"?: ...}`.

| Schema | Function | File:lineno | Inputs (train-only stats) | Output dim |
|---|---|---|---|---|
| C1 | `build_c1` | `scripts/34_train_task_c_h60.py:127` | `hour_freq`, `markov_probs`, `app_to_cat` | 14 |
| C2 | `build_c2` | `scripts/34_train_task_c_h60.py:172` | `hour_freq`, `markov_probs` | 15 |
| C3.1 | `build_c31` | `scripts/34_train_task_c_h60.py:285` | + `fg_timeline`, `per_app_inter_mean` | 20 |
| C3.2 | `add_c32_features` | `scripts/36_train_c3_grid.py:199` | + `app_lifetime_share`, `app_lifetime_kill_rate`, `cat_lifetime_share` | 29 |
| C3.3 | (cumulative via `build_schema_data`) + `add_c33_cols` | `scripts/36_train_c3_grid.py:237` | + `cat_markov_probs` | 32 |
| C3.4 | `add_c34_col` | `scripts/36_train_c3_grid.py:271` | + `last2_map`, `two_step_table` | 33 |
| C3-Pro variants | (same features, varied loss / arch) | `scripts/36_train_c3_grid.py:600+` | (same as C3.x) | (same as base) |

Train-only stat fitters (all live in `34_train_task_c_h60.py` or `36_train_c3_grid.py`):

| Stat | Function | Output shape |
|---|---|---|
| `hour_freq` | `fit_hour_freq` | `(24, V)` |
| `markov_probs` | `lib.v3.markov_prior.load_markov_prior` | `(V, V)` |
| `app_to_cat` | `lib.v3.categories.build_app_to_cat_idx` | `(V,)` int |
| `per_app_inter_mean` | `fit_per_app_inter_fg_mean` | `(V,)` float |
| `fg_timeline` | `collect_fg_timeline` | `dict[int → np.ndarray of int64 ts_ns]` |
| `app_lifetime_share` | `fit_app_lifetime_share` | `(V,)` float, sums to 1 |
| `app_lifetime_kill_rate` | `fit_app_lifetime_kill_rate` | `(V,)` float in [0, 1] |
| `cat_lifetime_share` | `fit_cat_lifetime_share` | `(11,)` float |
| `cat_markov_probs` | `fit_cat_markov` | `(11, 11)` float |
| `two_step_table` | `fit_two_step_markov` | sparse dict `(a_t-2, a_t-1) → (V,)` |

### B.4 Feature validator (`38_generate_features.py`)

Use this script to verify the pipeline end-to-end without retraining anything. It re-runs every train-only fit, materialises every feature schema, and checks shape / dtype / NaN-Inf / dimension count for each (schema × split) pair. Output is a JSON sanity report.

```bash
# Default: c1/c2/c3.1/c3.2/c3.3 on val + test (~6 s)
python scripts/38_generate_features.py

# All splits, single schema
python scripts/38_generate_features.py --schemas c3.3 --splits train val test

# Result lands at:
artifacts/bg/results/feature_validation_report.json
```

Exit code 0 = every (schema × split) passed:

* expected feature dim met (14 / 15 / 20 / 29 / 32 for C1 / C2 / C3.1 / C3.2 / C3.3),
* `n_nan == 0` and `n_inf == 0`,
* `y_3600` positive rate within ±0.1 % of `bg_data_stats.json`.

### B.5 Validation-report shape

The JSON has the following top-level keys:

| Key | Type | Meaning |
|---|---|---|
| `generated_at` | string | wall clock at script start |
| `schemas_run`, `splits_run` | list | cli echo |
| `expected_dim` | dict | mapping each schema to its expected #features |
| `row_counts_events` | dict | event-stream split sizes |
| `row_counts_bg` | dict | bg-parquet split sizes (after dropping `\|B(t)\|=0`) |
| `pos_rate_y_3600` | dict | per-split P(y_3600 = 1) |
| `stat_shapes` | dict | shape of each train-only fit table |
| `schema_reports` | dict[`<schema>/<split>` → report] | per-(schema, split) array summary + sample rows |
| `issues` | list | empty when everything passes |

Each `schema_reports[<schema>/<split>]` entry contains:

```jsonc
{
  "schema": "c3.3",
  "shape": [3121, 32], "dtype": "float32",
  "n_rows": 3121, "n_cols": 32, "expected_dim": 32, "dim_ok": true,
  "min": -1.0, "max": 9.301, "mean": 0.521, "std": 0.78,
  "n_nan": 0, "n_inf": 0,
  "y_pos_rate": 0.2169,
  "n_anchors": 870,
  "sample_rows": [
    {"anchor_id": 0, "app_idx": 3, "y": 1,
     "feat_first6": [3.5835, 3.989, 0.5, 1.0986, 1.0986, 1.0986],
     "feat_last6":  [5.6904, 3.0, 1.3863, 0.05, 0.5701, 0.0]}
  ]
}
```

### B.6 Validation results — current run

Latest end-to-end run (`scripts/38_generate_features.py --splits train val test`):

* events: train=34 691, val=5 521, test=3 752
* bg rows: train=25 013, val=4 061, test=3 121
* bg anchors: train=6 164, val=877, test=817 (post `\|B(t)\| ≥ 1` filter)
* pos-rate `y_3600`: train 0.2523, val 0.2810, test 0.2169
* hour_freq.shape = (24, 50); markov_probs.shape = (50, 50); cat_markov.shape = (11, 11); app_to_cat.shape = (50,)
* App lifetime-share top-5: 0.2741, 0.1680, 0.1100, 0.0540, 0.0420 (LITE, WECHAT, anonymous, GALLERY, BROWSER)
* All 5 schemas × 3 splits: **shape OK, NaN=0, Inf=0** for every row
* Total elapsed: ~3.8 s

The validation report itself is checked into `artifacts/bg/results/feature_validation_report.json`.

### B.7 Audit notes (issues found while writing this section)

| Finding | Action taken |
|---|---|
| §1 / §3 quoted C1/C2 metrics at **H=5** without flagging them as historical | Added a "Note (historical)" caveat in §A.6 above |
| §A.6 horizons table listed only `y_300` / `y_600` | Updated to all four horizons (`y_300`/`y_600`/`y_1800`/`y_3600`) with current train/val/test pos-rates |
| C3.x stage descriptions had no code-pointers | §B.3 above maps each schema to its builder function |
| §5.0 row 136 still showed `launch_cost` as "yes" but §5.A flags it dropped | Inconsistency flagged here; the canonical answer is dropped — verified absent in every current trainer (`scripts/`, `lib/bg/`) |
| Some §5.3 stages still labelled "C2.1 / C2.2 …" (legacy ordering) | Kept for reference; the *active* ordering is the H=60 table at the top of §5.3 |

If any of the validation invariants ever fail (e.g. a feature schema returns 31 cols instead of 32, or a NaN appears), `38_generate_features.py` exits with code 1 and writes a non-empty `issues` list to the JSON report — making it suitable as a CI smoke test.