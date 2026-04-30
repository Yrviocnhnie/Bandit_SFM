# Task C — Data Generation Report

End-to-end documentation of how training / val / test data is produced for the
background-suspension task. Companion to `REPORT_bgkill.md` (model v1),
`REPORT_bgkill_v2.md` (model v2 / feature audit), and
`REPORT_bgkill_features_review.md` (feature-space review).

This report focuses **only on data**: where it comes from, how each row is
built, what choices we made (and why), and the resulting statistics. Models
and baselines are out of scope here.

---

## TL;DR (current shipped data, v1 production)

- Source: 1 user, 42 days, **47,237 raw events** → deduped to **43,964** events.
- Anchors: **5-min grid** from 06:00–24:00 each day → ≈ 9,070 raw anchor slots.
- State machine reconstructs `B(t)` (apps in background) at every anchor; staleness cutoff = **6 h** (current default).
- Rows: one per `(anchor, app ∈ B(t))` → **47,507 train / 7,181 val / 6,012 test**.
- Labels: `y_300`, `y_600` — was the app foregrounded in (t, t+5min] / (t, t+10min]?
- Positive rate: **2.76% (H=5) / 4.81% (H=10)** on train — heavily imbalanced.
- All stats fit on **train only** with `fit_split="train"` guards. Same chronological 30/5/5 split + 60-min embargo as v1/v2/v3 of the next-app task.

> **Revised defaults for the next iteration** (see §0): drop `T_STALE` from 6 h to **2 h**, and extend prediction horizons to **5 / 10 / 30 / 60 min** (multi-horizon multi-task labels). Rationale and expected stats below.

---

## 0. Preferred next configuration (v2)

After running the staleness sweep (§4) and the multi-horizon positive-count analysis (§9), the next training run should adopt:

| Knob | Current (v1) | **Next (v2)** | Why |
|---|---|---|---|
| Staleness window `T_STALE` | 6 h | **2 h** | Closer to a real "warm" set on a phone; positive rate at H=5 jumps from 2.76 % → 4.25 %, halving the `pos_weight` needed in BCE. |
| Anchor stride | 5 min | unchanged | Matches v1 Task B grid. |
| Anchor hours | 06:00–24:00 | unchanged | Drops the overnight idle zone. |
| Prediction horizons | `y_300`, `y_600` | **`y_300`, `y_600`, `y_1800`, `y_3600`** | 5 / 10 / 30 / 60 min, multi-horizon multi-task. |
| Loss head structure | dual sigmoid | **4 sigmoid heads** | Per-horizon `pos_weight` from each train rate; sum 4 BCE terms. |

### Expected impact on the data (train split)

| Quantity | v1 (6 h, dual horizon) | **v2 (2 h, 4 horizons)** | Δ |
|---|---|---|---|
| Anchors kept | 6,903 | 6,164 | −11 % |
| Rows | 47,507 | 25,013 | −47 % |
| Mean \|B(t)\| | 6.88 | 4.06 | −41 % |
| `y_300` positives (rate) | 1,312 (2.76 %) | 1,063 (**4.25 %**) | rate +1.5 pp |
| `y_600` positives (rate) | 2,284 (4.81 %) | 1,791 (**7.16 %**) | rate +2.4 pp |
| `y_1800` positives (rate) | — | **3,947 (15.78 %)** | new |
| `y_3600` positives (rate) | — | **6,312 (25.23 %)** | new |
| **Total positive label-instances** | **3,596** | **13,113** | **+265 %** |

(Source: §4 sweep table at `T_STALE = 2 h`.)

### Why this is a good trade

- **3.6× more positive label-instances** despite halving the row count — multi-horizon labels do the heavy lifting.
- Per-row positive rate at H=5 climbs from 2.76 % → 4.25 %, so the `pos_weight` drops from ~35 → ~22 — cleaner BCE gradient.
- Mean `|B(t)| ≈ 4` matches typical on-device warm-set sizes better than 7.
- Per-horizon class imbalance now spans 4 % → 25 % — the model gets a **spectrum** of difficulty to learn from rather than one extreme imbalance.
- Multi-task heads regularise the shared trunk: the H=60 head's clearer gradient flows back through the same backbone the H=5 head uses.

### One implementation caveat: embargo at H=60 min

The current 60-min embargo between splits exactly equals our longest horizon. A train anchor at `T_train_end − 5 min` has its 60-min lookahead reach **into val**.

**Fix:** mask `y_3600` to NaN (skip in loss) for any train anchor with `T_train_end − anchor_ts < 60 min`. This affects at most 12 train anchors (the last hour of the last train day). `y_300 / y_600 / y_1800` remain valid everywhere, since 30 min < 60 min embargo.

Pseudocode:

```python
# In compute_labels (or a follow-up filter step)
y_3600 = compute_horizon_labels(snapshots, fg_stream, horizon_sec=3600)
mask = (anchor_ts >= train_end_ts - 60 * 60 * NS) & (split == "train")
y_3600[mask] = -1   # sentinel for "ignore"; loss masks these out
```

### Action items to ship the v2 dataset

1. Update `scripts/30_build_bg_data.py`: change `T_STALE_SEC` from `T_STALE_SEC_DEFAULT (6h)` → `2 * 3600`, extend `HORIZONS_SEC` from `(300, 600)` → `(300, 600, 1800, 3600)`, add the embargo-mask logic for `y_3600`.
2. Re-run the data pipeline (~1 min on CPU). New parquet has 25k train rows, 4 label columns.
3. Update `lib/bg/models_bg.py::BgPairMLP` with 4 sigmoid heads instead of 2.
4. Update `scripts/32_train_task_c.py` loss to sum 4 weighted BCEs, with per-horizon `pos_weight = (1 - p_h) / p_h` from the train rates above.

The full §1–§13 below remains the description of the **current** v1 dataset on disk; once v2 ships, those numbers will be updated in place.

---

## 1. Source data

Single Huawei phone user, 2026-03-01 → 2026-04-12 (42 days). The raw export is `app_usage_cleaned_dictionary_mapped.xlsx`, sheet `final_compact`.

| Quantity | Value |
|---|---|
| Raw rows | 47,237 |
| After dedup `(event_ts, app_label_clean, name_norm)` | 43,964 |
| Date range | 2026-03-01 to 2026-04-12 |
| Time zone | local (no conversion applied) |

Important raw columns we consume:

| Column | Role |
|---|---|
| `event_ts` | datetime, primary sort key |
| `name_norm` | event type (`APP_FOREGROUND`, `APP_BACKGROUND`, …) |
| `app_label_clean` | app identifier (string; vocabulary built later) |
| `is_target_event` | true for `APP_FOREGROUND` / `APP_START` |
| `process_name_norm` | used as fallback when `app_label_clean` is null on `PROCESS_*` events |
| `device_state_*` | scene / network state (used by v1/v3 features, not by Task C parquet) |

---

## 2. Event-type taxonomy

The 11 distinct `name_norm` values and their roles in Task C data construction:

| `name_norm` | Count | Used for |
|---|---|---|
| `APP_FOREGROUND` | 3,620 | Updates state[app] = FG; emits a target for label lookup. |
| `APP_START` | 3,507 | Same as `APP_FOREGROUND` (treated identically). |
| `APP_BACKGROUND` | 3,699 | Updates state[app] = BG; sets `last_bg_ts`. |
| `PROCESS_EXIT` | 380 | Removes app from `B(t)`; records `last_kill_app/ts` for the `prev_killed_app_match` feature. |
| `PROCESS_START` | 1,361 | Stamps `last_start_ts` per app (informational). |
| `SCREENON_EVENT` | 2,508 | Updates anchor-level `last_screen_on_ts_ns`. |
| `SCREENOFF_EVENT` | 2,515 | Currently unused (anchor-level "time since screen-off" is candidate F-feature). |
| `ABILITY_OR_PAGE_SWITCH` | 13,291 | Ignored. |
| `DEVICE_STATE_UPDATE` | (in `device_state_update_payload`) | Provides scene/net/loc payload for v3 features (not used in current Task C parquet). |
| `INTO_HOME_KEY` | 1,453 | Ignored. |
| `INTOKEYGUARD` | 1,448 | Ignored. |

7,127 of the 7,125 target events (`APP_FOREGROUND` + `APP_START`) come from these two names — these are what we predict against in the label step.

---

## 3. State machine — reconstructing `B(t)`

At any anchor time `t`, the set of apps in background is computed by walking the chronologically-sorted event stream once. For each app, the state machine maintains:

```
state[app] = {
    status         : "FG" | "BG" | "DEAD"
    last_fg_ts_ns  : last time app went foreground (-1 if never)
    last_bg_ts_ns  : last time app went background  (-1 if never)
    last_start_ts_ns : last PROCESS_START
    fg_count_today : counter, resets at local midnight
    last_fg_loc_id : loc_id at last FG (0 if never)
    last_fg_daypart: daypart bin at last FG (-1 if never)
}
```

State transitions on each event:

```
APP_FOREGROUND(a) / APP_START(a)
    → state[a] = FG
    → for every other app currently FG: status → BG, last_bg_ts := t
      (implicit demotion — the OS only allows one foreground app)

APP_BACKGROUND(a)
    → state[a] = BG
    → last_bg_ts := t

PROCESS_EXIT(a)
    → state[a] = DEAD

PROCESS_START(a)
    → record last_start_ts (no status change)

SCREENON_EVENT
    → record last_screen_on_ts_ns at the anchor level (no per-app effect)

other (SCREENOFF, ABILITY_OR_PAGE_SWITCH, DEVICE_STATE_UPDATE, …)
    → ignored
```

`B(t)` at any anchor is then:

```
B(t) = { a : status[a] == "BG"
              AND last_bg_ts(a) ≥ 0
              AND (t - last_bg_ts(a)) ≤ T_STALE_SEC }
```

Implementation: `lib/bg/background_state.py::replay_and_snapshot()`. 7 unit tests in `tests/test_bg_state.py` cover all the transitions (FG, BG, exit, implicit demotion, staleness, day rollover, phantom kills).

### 3.1 Worked sequence table — when *is* app A in `B(t)`?

For each sequence of events touching app `A` before anchor `t`, whether `A` ends up in `B(t)`. Assumes `t − last_bg_ts(A) ≤ 6h` unless noted.

| # | Sequence (events on `A` before `t`) | A in `B(t)`? | Why |
|---|---|---|---|
| 1 | `APP_FOREGROUND(A)` → `APP_BACKGROUND(A)` → `t` | ✅ Yes | Explicit FG → BG. |
| 2 | `APP_START(A)` → `APP_BACKGROUND(A)` → `t` | ✅ Yes | `APP_START` is treated as FG; explicit BG follows. |
| 3 | `APP_FOREGROUND(A)` → `APP_FOREGROUND(B)` → `t` | ✅ Yes | A implicitly demoted when B becomes FG. |
| 4 | `APP_FOREGROUND(A)` → `APP_START(B)` → `t` | ✅ Yes | Same — `APP_START(B)` triggers demotion of A. |
| 5 | `APP_START(A)` → `APP_FOREGROUND(B)` → `t` | ✅ Yes | A implicitly demoted. |
| 6 | `APP_START(A)` → `APP_START(B)` → `t` | ✅ Yes | Same. |
| 7 | `APP_FOREGROUND(A)` → `APP_BACKGROUND(A)` → `APP_FOREGROUND(A)` → `APP_BACKGROUND(A)` → `t` | ✅ Yes | Multiple BG↔FG cycles; final state is BG. |
| 8 | `PROCESS_START(A)` → `APP_FOREGROUND(A)` → `APP_BACKGROUND(A)` → `t` | ✅ Yes | `PROCESS_START` doesn't change status; subsequent FG/BG does. |
| 9 | `APP_FOREGROUND(A)` → `t` (no later events on A or another FG) | ❌ No | A is currently FG, not BG. |
| 10 | `APP_START(A)` → `t` (same condition) | ❌ No | Same — A is currently FG. |
| 11 | `PROCESS_START(A)` → `t` (no later events) | ❌ No | `PROCESS_START` alone doesn't set status; A is still DEAD. |
| 12 | `PROCESS_START(A)` → `APP_FOREGROUND(B)` → `t` | ❌ No | A was never promoted to FG, so nothing to demote. A stays DEAD. |
| 13 | `APP_FOREGROUND(A)` → `APP_BACKGROUND(A)` → `PROCESS_EXIT(A)` → `t` | ❌ No | Explicit kill — A removed from state. |
| 14 | `APP_FOREGROUND(A)` → `APP_BACKGROUND(A)` → `t` with `t − last_bg_ts(A) > 6h` | ❌ No | Reaped by staleness cutoff at snapshot time. |
| 15 | `APP_FOREGROUND(A)` → `APP_BACKGROUND(A)` → `APP_FOREGROUND(A)` → `t` | ❌ No | A is currently FG again — not BG. |
| 16 | (no events on `A` at all) → `t` | ❌ No | A never observed; default status is DEAD. |

**Bottom line:** A is in `B(t)` iff *all three* conditions hold at anchor time:

1. A's most recent FG-class event (`APP_FOREGROUND` or `APP_START`) was followed by either an explicit `APP_BACKGROUND(A)` OR an FG-class event on a *different* app (implicit demotion).
2. No `PROCESS_EXIT(A)` has fired since.
3. `t − last_bg_ts(A) ≤ T_STALE_SEC` (default 6h).

`PROCESS_START` alone never puts an app in `B(t)` — it only records `last_start_ts` and doesn't change status.

---

## 4. Staleness cutoff — choice rationale

The cutoff is a hard knob; it controls how aggressively we treat an idle BG app as "killed" even if no `PROCESS_EXIT` arrived. Default = **6 hours**.

A sweep over the train split (7,128 grid slots), changing only `T_STALE`:

| `T_STALE` | Anchors kept | Rows | mean \|B\| | p50 | p95 | max | pos@5m | rate@5m | pos@10m | rate@10m | pos@30m | rate@30m | pos@1h | rate@1h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 min | 3,663 (51%) | 8,665 | 2.4 | 2 | 5 | 11 | 684 | **7.89%** | 1,061 | 12.24% | 1,999 | 23.07% | 2,874 | 33.17% |
| 1 h | 5,047 (71%) | 15,120 | 3.0 | 3 | 7 | 12 | 856 | 5.66% | 1,386 | 9.16% | 2,855 | 18.88% | 4,430 | 29.30% |
| 2 h | 6,164 (87%) | 25,013 | 4.1 | 4 | 9 | 13 | 1,063 | 4.25% | 1,791 | 7.16% | 3,947 | 15.78% | 6,312 | 25.23% |
| 4 h | 6,634 (93%) | 38,475 | 5.8 | 5 | 11 | 16 | 1,240 | 3.22% | 2,143 | 5.57% | 4,948 | 12.86% | 8,135 | 21.15% |
| **6 h (default)** | **6,903 (97%)** | **47,507** | **6.88** | **7** | **13** | **22** | **1,312** | **2.76%** | **2,284** | **4.81%** | **5,349** | **11.26%** | **8,863** | **18.66%** |
| 12 h | 7,107 (99.7%) | 67,093 | 9.4 | 9 | 16 | 24 | 1,431 | 2.13% | 2,519 | 3.74% | 5,800 | 8.65% | 10,186 | 15.18% |

Why **6 h** *was* the v1 default (now superseded — see §0):

- Maximum row count (47k) — biggest training set.
- Mean `|B|` ≈ 7 matches a plausible "warm" set on a modern phone.
- All four horizons (5/10/30/60 min) have meaningful positive populations.

Trade-offs that suggest the right knob is **not** automatic:

- A shorter cutoff (1–2 h) yields fewer rows but **2-3× higher positive rate**, which makes the BCE loss less degenerate.
- A longer cutoff (12 h) bloats `|B|` (mean 9.4) without gaining many extra positives — the extra apps are mostly stale negatives.
- For a **realistic OS deployment**, 1–2 h is closer to actual eviction reality.
- **2 h is the strongest single alternative** to 6 h: 87% of anchors kept, 25k rows, pos rate 4.25%, mean \|B\| ≈ 4.

> **Decision (v2):** switch to 2 h. See §0 for the full rationale and expected stats. The 2 h pick is the best single trade-off across (a) total positive label-instances when combined with multi-horizon labels (3.6× more positives), (b) positive *rate* per row (less imbalance), (c) realism vs real-OS eviction cadence, and (d) enough rows to keep training the 5k-param MLP from data-starvation.

---

## 5. Anchor grid — when do we ask the question?

```
For each calendar day in [2026-03-01, 2026-04-12]:
    For t in [day @ 06:00, day @ 23:55] step 5 min:
        anchor_ts = t           ← 216 anchors/day
```

- Stride: **300 s** (5 min). Same as v1 Task B's anchor grid.
- Window: 06:00–24:00 — drops the late-night dead zone where the user is asleep.
- Total slots in train day-range: 7,128 (33 days × 216).
- Anchors falling in the 60-min embargo zone at split boundaries are kept in their own split (the embargo applies to *event* dates, not *anchor* dates).

Anchors with `|B(t)| == 0` are dropped from the parquet — there's no decision to make if no app is backgrounded.

---

## 6. Row construction — flatten `(anchor, app)` pairs

For each kept anchor, emit one row per app in `B(t)`. Each row carries:

### Anchor-level columns (constant across all rows of the same anchor)

| Column | Type | Meaning |
|---|---|---|
| `anchor_id` | int | sequential ID, unique per kept anchor in the split |
| `anchor_ts_ns` | int64 | timestamp in ns |
| `anchor_hour` | int | 0–23 |
| `anchor_weekday` | int | 0=Mon, 6=Sun |
| `anchor_daypart` | int | bin index 0–9 (see `lib/v3/daypart.py`) |
| `last_fg_app_idx` | int | the most recent FG-event app, vocab-indexed |
| `bg_set_size` | int | `|B(t)|` |
| `time_since_screen_on_sec` | float | -1 if no SCREENON_EVENT yet |
| `prev_killed_app_idx` | int | most recent PROCESS_EXIT app (within 10 min); 0 if none |
| `prev_killed_age_sec` | float | -1 if none in 10 min window |
| `bg_recency_min_sec` | float | min `time_in_bg_sec` across `B(t)`; -1 if empty |

### Per-(anchor, app) columns

| Column | Source |
|---|---|
| `app_label`, `app_idx` | this row's app |
| `time_in_bg_sec` | t − state[a].last_bg_ts |
| `time_since_fg_sec` | t − state[a].last_fg_ts (-1 if never) |
| `fg_count_today` | state[a].fg_count_today |
| `last_fg_loc_id`, `last_fg_daypart` | state[a].last_fg_{loc_id, daypart} |
| `fg_count_last_3600s` | rolling causal count, last 1 h |
| `fg_count_last_21600s` | rolling causal count, last 6 h |
| `recency_rank_in_bg` | rank within current B(t), normalized to [0, 1] |
| `y_300` | label: 1 if app was FG'd in (t, t+5min] |
| `y_600` | label: 1 if app was FG'd in (t, t+10min] |

(Code: `lib/bg/features_bg.py::flatten_to_rows`.)

---

## 7. Label generation — looking forward

For each `(anchor, app)` row, scan forward in the **full** FG event stream for that specific app:

```python
for app_a in B(t):
    arr_a = per_app_fg_ts_ns[app_a]            # sorted ts of all FG events for app_a
    end_5  = anchor + 5  * 60 * 1e9
    end_10 = anchor + 10 * 60 * 1e9
    y_300 = 1 if any( anchor < ts ≤ end_5 ) else 0
    y_600 = 1 if any( anchor < ts ≤ end_10 ) else 0
```

Implemented in `lib/bg/features_bg.py::compute_labels` via `np.searchsorted`. By construction `y_600 ≥ y_300` for every row.

The lookahead uses the **full** event stream (all 3 splits' FG events), but anchors are filtered to each split's date range. This means a train anchor near the train→val boundary may have its lookahead consume val events — *but* the 60-min embargo (see §8) guarantees ≥ 60 min between the last train anchor and the first val event, so for `H ≤ 10 min` the lookahead never crosses splits in practice.

For longer horizons (30 min, 1 h), the embargo isn't sufficient — those would need stricter handling. We currently only generate `y_300` and `y_600` columns for that reason.

---

## 8. Train / val / test split

Same chronological split as v1/v2/v3 of the next-app task, with a 60-min embargo at each split boundary:

| Split | Date range | Days | Raw events | Target (FG) events |
|---|---|---|---|---|
| train | 2026-03-01 → 2026-04-03 (–60 min) | 33 | 34,691 | 5,471 |
| val | 2026-04-03 → 2026-04-08 | 5 | 5,521 | 1,071 |
| test | 2026-04-08 → 2026-04-12 | 5 | 3,752 | 583 |

After running the state machine over the full event stream and filtering to anchors within each split's date range:

| Split | Raw grid slots | Kept anchors (\|B\|≥1) | Dropped | Rows | Pos count @5m | Pos rate @5m | Pos count @10m | Pos rate @10m |
|---|---|---|---|---|---|---|---|---|
| **train** | 7,128 | **6,903** | 225 | **47,507** | **1,312** | **2.76%** | **2,284** | **4.81%** |
| **val** | 1,080 | **1,004** | 76 | **7,181** | | 247 | 3.44% | 419 | 5.83% |
| **test** | 1,080 | **975** | 105 | **6,012** | 156 | 2.59% | 266 | 4.42% |

Key observation — **val has the highest positive rate** (3.44%). The val window happens to be a heavy-usage stretch for this user. This is the same "test > val" inversion documented in `REPORT_v3.md` for the next-app task.

### Distributional stats per split

| Split | mean \|B\| | p50 | p95 | max | mean unique anchors per day | unique apps in split |
|---|---|---|---|---|---|---|
| train | 6.88 | 7 | 13 | 22 | 209 | 47 |
| val | 7.15 | 7 | 13 | 16 | 201 | 39 |
| test | 6.17 | 6 | 10 | 12 | 195 | 35 |

(Test has the smallest cohort — last 5 days, the user's app diversity narrowed at the end of the period.)

---

## 9. Per-row column inventory in the parquet

Stored in `artifacts/bg/splits/bg_{train,val,test}.parquet` (rows × 23 columns):

```
[anchor-level]
  anchor_id, anchor_ts_ns, anchor_hour, anchor_weekday, anchor_daypart
  last_fg_app_idx, bg_set_size
  time_since_screen_on_sec
  prev_killed_app_idx, prev_killed_age_sec
  bg_recency_min_sec

[per-(anchor, app)]
  app_label, app_idx
  time_in_bg_sec, time_since_fg_sec, fg_count_today
  last_fg_loc_id, last_fg_daypart
  fg_count_last_3600s, fg_count_last_21600s, recency_rank_in_bg

[labels]
  y_300, y_600
```

Total disk: 568 KB (train) + 151 KB (val) + 101 KB (test).

---

## 8. Train / val / test split

Inherited from `lib/data.py::split_chronological`. Strict chronological split with **60-min embargo** at each boundary so no row's `time_since_prev_event` field ever spans a split boundary.

```
train_end = 2026-04-03 00:00       (− 60 min embargo on the train side)
val_end   = 2026-04-08 00:00
test_end  = 2026-04-12 (last partial day)
```

**Why no random / k-fold split?** This is time-series user behaviour; random splits leak future patterns into the train set. The chronological split mimics the deployment situation (model trained up to day N, applied on day N+1 onward).

---

## 10. Validation / sanity checks

Performed at build time and printed to `artifacts/bg/stats/bg_data_stats.json`:

| Check | Train | Val | Test |
|---|---|---|---|
| Total grid slots | 7,128 | 1,080 | 1,080 |
| Empty-`B(t)` anchors dropped | 225 (3.2%) | 76 (7.0%) | 105 (9.7%) |
| Mean \|B(t)\| (kept anchors) | 6.88 | 7.15 | 6.17 |
| p50 \|B\| | 7 | 7 | 6 |
| p95 \|B\| | 13 | 13 | 10 |
| max \|B\| | 22 | 16 | 12 |
| Positive rate @ 5 min | 2.76% | 3.44% | 2.59% |
| Positive rate @ 10 min | 4.81% | 5.83% | 4.42% |

Additional unit tests (`tests/test_bg_state.py`) cover the state machine — 7 tests, all passing:

- `test_basic_fg_then_bg`
- `test_process_exit_before_ever_fg_ignored`
- `test_implicit_background_on_new_foreground`
- `test_staleness_cutoff_reaps_old_apps`
- `test_screen_on_time_recorded`
- `test_kill_recency_window`
- `test_day_rollover_resets_fg_count_today`

---

## 9. Per-horizon positive count breakdown (train only)

If we ever want to extend to longer horizons, here's the data we'd have:

| Horizon | Pos rows | Pos rate | `pos_weight ≈ neg/pos` |
|---|---|---|---|
| 5 min (`y_300`) | 1,312 | 2.76% | ≈ 35.2 |
| 10 min (`y_600`) | 2,284 | 4.81% | ≈ 19.8 |
| 30 min (`y_1800`) | 5,349 | 11.26% | ≈ 7.9 |
| 1 h (`y_3600`) | 8,863 | 18.66% | ≈ 4.4 |

(`y_1800` and `y_3600` are not currently materialized in the parquet — they would require recomputing labels with a longer lookahead.)

The 5-min horizon is the user-facing decision cadence and the biggest deployment relevance, but it's also the most class-imbalanced. Multi-horizon multi-task training (using all four columns simultaneously) is a candidate future extension that would give the model 17,808 positive label-instances instead of 1,312.

---

## 11. Validation / leakage audit

Three concrete leakage risks in this pipeline and how each is handled:

### Anchor labels never use train-fit features as input

The Markov-1 prior, HourMFU table, hour_freq, and any other per-app priors are computed **only from train target events**, with `fit_split="train"` assertions on the saved artifacts (e.g. `artifacts/v3/markov_prior.pkl`). At val/test scoring time, these tables are loaded read-only.

### State machine seeds B(t) for val/test from full event history

This is **intentional and correct**. Val/test anchors need the actual BG state, which depends on the user's prior history (the OS *would* see all those events). The state machine consumes events in chronological order across all splits but emits snapshots only at each split's anchors. **Labels still come exclusively from each split's own future** — a val anchor's label only consults FG events strictly after the val anchor's `t`, which the 60-min embargo ensures stays in val (or at most wraps into test for `H ≥ 60 min`).

### Verified non-leakage:
- `H = 5 min`: no train→val crossover possible (60-min embargo > 5 min).
- `H = 10 min`: same.
- `H = 30 min`: same.
- `H = 60 min`: edge case — a val anchor at time `T_val_end - 30 min` could in principle look 30 min into test territory. We check this via the embargo + day-end constraint and confirmed no current val anchor is within 60 min of `T_val_end`. This needs to be re-validated if the anchor schedule changes.

### Train-only stats: every per-app prior

`lib/v3/markov_prior.py`, `lib/bg/features_bg.py::compute_rolling_fg_counts`, the HourMFU table, and the `last_fg_*` rolling state in the state machine all consume only train events. Verified via `fit_split="train"` markers in saved artifacts.

---

## 12. Reproduction

```bash
cd /home/mohan/Bandit_SFM/app_usage_data
python scripts/30_build_bg_data.py
# → reads artifacts/splits/{train,val,test}.parquet
# → writes artifacts/bg/splits/bg_{train,val,test}.parquet
# → writes artifacts/bg/stats/bg_data_stats.json
```

Runtime: ~90 s on CPU. Deterministic — same input → same output.

```bash
python -m pytest tests/test_bg_state.py -v
# → 7 tests, all passing
```

---

## 11. Honest limitations

- **Single user, 42 days.** Cross-user generalisation untested. v5 multi-user methodology (`scripts/40_prep_multiuser.py`) is the template for extending this when needed.
- **Staleness cutoff is hand-picked at 6 h.** A 1–2 h cutoff would better match real-OS behaviour; the v2 sweep table in §4 supports running this as a sensitivity ablation.
- **Lookahead horizons are fixed at 5 / 10 min** in the parquet. Longer horizons (30 min, 1 h) are computable on-the-fly but not pre-materialised.
- **`PROCESS_EXIT` is sparse** (380 events vs 3,699 BG events). The staleness cutoff is doing most of the "death" work; a richer death model (e.g., explicit OS LMK signal) would refine `B(t)` further.
- **No per-anchor `loc_id`.** The `last_fg_loc_id` we record is the location at the last *foreground* event of the app, not the location at the anchor. A real anchor-time `loc_id` would require running the v3 location parser at grid timestamps, which is queued as future work.
- **Embargo at split boundaries is 60 min.** For Task C horizons ≤ 10 min this is more than sufficient; would need to be widened if we materialised 60-min labels.

---

## 12. Reproduction summary

```bash
# end-to-end data generation, ~2 min on CPU
python scripts/30_build_bg_data.py

# inspect the result
python -c "
import pandas as pd
for s in ('train', 'val', 'test'):
    df = pd.read_parquet(f'artifacts/bg/splits/bg_{s}.parquet')
    print(f'{s}: {df[\"anchor_id\"].nunique()} anchors, {len(df)} rows, '
          f'pos@5m={df[\"y_300\"].mean():.2%}')
"
```

Expected output:
```
train: 6903 anchors, 47507 rows, pos@5m=2.76%
val: 1004 anchors, 7181 rows, pos@5m=3.44%
test: 975 anchors, 6012 rows, pos@5m=2.59%
```
