# Phase-2: Feedback Propagation for Context-Specific Recommendations

This document describes the **phase-2 online feedback propagation** experiment in full detail: motivation, data pipeline, algorithm, knobs, ablation matrix, metrics, and interpretation. It is meant as a standalone technical reference — readable without prior conversation context.

---

## 1. Motivation

### 1.1 What problem does phase-2 solve?

The frozen V6 neural-linear model (`artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/`) is trained from offline V6 relevance labels, where every row of a scenario gets the **same** `most_relevant_3` actions. That label scheme cannot express **context-sensitive recommendations within a single scenario** — e.g.:

| Scenario       | Context variation                     | Desired top-1          |
|----------------|----------------------------------------|------------------------|
| `ARRIVE_OFFICE`| `cal_eventCount ≥ 3`                   | `O_SHOW_SCHEDULE`      |
| `ARRIVE_OFFICE`| `cal_eventCount = 0`, office_working   | `O_SHOW_TODAY_TODO`    |
| `ARRIVE_OFFICE`| `cal_eventCount = 0`, dawn arrival     | `R_PLAN_DAY_OVER_COFFEE` |

Three different rows of the same scenario ought to yield three different top-1 predictions, but the frozen model trained on scenario-uniform labels naturally converges to one "averaged" ranking for the whole scenario.

**Phase-2 asks:** can an *online* feedback loop — applying `+like` updates to a handful of real context rows — teach the frozen model to distinguish these in-scenario contexts without retraining the trunk and without damaging the V6 most-relevant top-3 coverage?

### 1.2 Why "propagation"?

A single `like` update on a single row is weak evidence. We amplify it by also updating **N latent-neighbours** of that row (rows with similar neural-linear embeddings). The hypothesis is that neighbours in embedding space share whatever contextual feature is the real driver (`cal_eventCount`, `ps_time`, etc.), so the update should generalise to them without leaking into unrelated scenarios.

---

## 2. Inputs

### 2.1 The 9 context variations (spec)

Defined in `docs/phase2_multi_recommendation_scenarios.md`. The spec lists 9 contexts across 4 scenarios:

| phase2_context_id   | scenario           | label                    | target R/O action        |
|---------------------|--------------------|--------------------------|--------------------------|
| `ARRIVE_OFFICE_A`   | ARRIVE_OFFICE      | Meeting-heavy arrival    | `O_SHOW_SCHEDULE`        |
| `ARRIVE_OFFICE_B`   | ARRIVE_OFFICE      | Task-focused arrival     | `O_SHOW_TODAY_TODO`      |
| `ARRIVE_OFFICE_C`   | ARRIVE_OFFICE      | Soft / dawn arrival      | `R_PLAN_DAY_OVER_COFFEE` |
| `LEAVE_OFFICE_A`    | LEAVE_OFFICE       | Driving commute          | `O_SHOW_COMMUTE_TRAFFIC` |
| `LEAVE_OFFICE_B`    | LEAVE_OFFICE       | Walking commute          | `O_SHOW_WEATHER`         |
| `OFFICE_LUNCH_OUT_A`| `OFFICE_LUNCH_OUT` | Ready to pay             | `O_SHOW_PAYMENT_QR`      |
| `OFFICE_LUNCH_OUT_B`| `OFFICE_LUNCH_OUT` | Stepping out / break     | `R_ENJOY_LEISURE_BREAK`  |
| `CAFE_QUIET_A`      | `CAFE_QUIET`       | Upcoming meeting         | *(see §8 caveats)*       |
| `CAFE_QUIET_B`      | `CAFE_QUIET`       | Deep focus               | `O_TURN_ON_FOCUS_MODE`   |

Each context fixes a subset of feature values (state, precondition, ps_time, calendar, phone position, etc.). The intent: real rows in the training corpus already satisfy these feature constraints, so we can "pick" test rows from the existing data rather than synthesise.

### 2.2 Base artifact

All phase-2 experiments run against the frozen V6 model at:
```
artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/
```
This is a `MaskedNeuralLinearUCB` (`recommendation_agents/linucb.py:827-940`): a 306→128→64 MLP encoder (ReLU-gated) feeding per-action disjoint linear UCB heads over a global 46-action R/O space. The encoder weights are frozen throughout phase-2; only the `A_inv` / `b` parameters of the heads change.

### 2.3 Source data

Raw JSONL: `bandit_v0_65_scenarios_unique_support_to_2k_samples_office_ranked.jsonl` (referenced as the `DEFAULT_DATA` constant in `scripts/phase2_split.py:14`). Each row carries the standard V0 feature set plus `scenario_id`, `episode_id`, `gt_ro`, `gt_app`.

---

## 3. Splitter — `scripts/phase2_split.py`

### 3.1 Responsibility

Turn the raw JSONL into two files:

- `artifacts/phase2/train_phase2.raw.jsonl` — propagation pool (everything except the held-out test rows)
- `artifacts/phase2/test_phase2.raw.jsonl` — 27 rows (9 contexts × 3 samples) tagged with `phase2_context_id`, `phase2_gt_ro`, `phase2_label`, and optionally `phase2_patched_fields`

### 2.2 Context definitions

A `CONTEXTS` list (`scripts/phase2_split.py:148-231`) defines each of the 9 entries as:

```python
{
    "context_id": "ARRIVE_OFFICE_A",
    "label": "Meeting-heavy arrival",
    "scenario_id": "ARRIVE_OFFICE",
    "gt_ro": "O_SHOW_SCHEDULE",
    "exact_match": { <feature: required value>, ... },
    "patches":    { <feature: override value>, ... }  # empty for exact-match contexts
}
```

- **6 contexts** have non-empty `exact_match` and empty `patches` (`phase2_split.py:149-194`). Candidate rows are those where **every** `exact_match` feature matches the context spec.
- **3 contexts** (`ARRIVE_OFFICE_B`, `ARRIVE_OFFICE_C`, `CAFE_QUIET_A`) have a non-empty `patches` dict (`phase2_split.py:196-231`) because the required feature combination doesn't exist in the source data. For example, the source data doesn't contain any `ARRIVE_OFFICE` row with `cal_hasUpcoming=0` — so the splitter grabs the nearest match and overwrites the calendar fields.

### 3.3 Patching logic (`patch_and_verify`, `phase2_split.py:118-142`)

When a patch is applied:
1. Copy the candidate row.
2. Overwrite the feature fields listed in `patches`.
3. If `ps_time` is patched, also **repair `hour` and `timestep`**: the middle hour of the new ps_time range is assigned, and `timestep` is recomputed as `hour*3600 + (old_ts % 3600)` to preserve the minute/second offset.
4. Run `verify_sample()` on the result.

### 3.4 Row verification (`verify_sample`, `phase2_split.py:63-115`)

Five consistency rules applied to every test row (exact or patched):

1. `hour` must fall inside the range of `TIME_SLOT_HOURS[ps_time]` (`phase2_split.py:78-81`).
2. `timestep // 3600 == hour` (`phase2_split.py:83-87`).
3. `state_current` state-code rules from `STATE_CODE_CHECKS` enforce location/time/day/sound compatibility (`phase2_split.py:89-109`, rules table at `:43-49`). Examples:
   - `office_arriving` ⇒ `ps_location=work`, `ps_time ∈ {dawn, morning}`, `ps_dayType=workday`
   - `at_cafe_quiet` ⇒ `ps_location=cafe`, `ps_sound=quiet`
4. Calendar consistency: `cal_hasUpcoming=0 ⇔ cal_eventCount=0`, and `cal_hasUpcoming=1` forbids `cal_eventCount=0` (`phase2_split.py:101-106`).
5. Soft check: if `cal_hasUpcoming=1`, `cal_nextLocation` shouldn't be `unknown` (warning only; `phase2_split.py:109-113`).

Violations are appended to an `issues` list and surface in the summary report.

### 3.5 Profile normalisation

To avoid leaking user-id-specific signal into the split, every train and test row is overwritten to a single canonical profile (`phase2_split.py:19-26`, `286-291`, `342-350`):
```
user_id_hash_bucket = "b11"
age_bucket = "25_34"
sex = "male"
has_kids = 0
```
This is on by default (`--no-profile-normalize` disables).

### 3.6 Execution

```bash
python scripts/phase2_split.py --samples-per-context 3
```

Outputs:
- `artifacts/phase2/test_phase2.raw.jsonl` — 27 tagged rows
- `artifacts/phase2/train_phase2.raw.jsonl` — all other rows, profile-normalised
- `artifacts/phase2/phase2_split_report.json` — per-context candidate counts, patched fields, verification issues

### 3.7 Tags written to test rows

Each test row gets three extra fields (`phase2_split.py:347-351`), plus a 4th one if it was patched:

```json
{
  "phase2_context_id": "ARRIVE_OFFICE_A",
  "phase2_gt_ro": "O_SHOW_SCHEDULE",
  "phase2_label": "Meeting-heavy arrival",
  "phase2_patched_fields": ["cal_hasUpcoming", "cal_eventCount"]   // only if patched
}
```

These tags are the sole link between the test rows and the experiment — the workflow reads them verbatim.

---

## 4. Workflow — `simulate_feedback_propagation_phase2_on_frozen_neural_linear`

Defined at `recommendation_agents/workflows.py:1823-2290`. CLI wrapper: `simulate-feedback-propagation-phase2` at `recommendation_agents/cli.py:424-447` / `cli.py:955-1004`.

Algorithm (one execution):

### 4.1 Load & encode (workflows.py:1866-1952)

1. Load the frozen `MaskedNeuralLinearUCB` from `ro_model/` (`workflows.py:1868`). Require the model to expose `encode_context` and `partial_fit` (`:1870`).
2. Load V6 relevance catalog (used for "V6 before/after" coverage metric, `:1872`).
3. Load raw train+test JSONL, validate that every distinct `phase2_gt_ro` is in the model's 46-action space (`:1879-1885`).
4. For each test row: encode features → `x` → `latent = encoder(x)` via `encode_context`. Store `(episode_id, scenario_id, x, latent, phase2_*)` (`:1887-1914`).
5. Same for train rows (`:1916-1945`). Stack latents into a float32 matrix for cosine-similarity search (`:1954`). Build a `scenario_id → train_indices` lookup for the same-scenario mode (`:1955-1958`).

### 4.2 Anchor selection (workflows.py:1960-2019)

Group test rows by `phase2_context_id` (`:1961-1963`). For each context, sort by `episode_id` and pick the first row as the **anchor** (`:1970`). The anchor is the single point whose latent neighbourhood drives propagation, and where the "did we succeed?" rank-after check is applied.

- With `--use-all-test-samples` (`:1973`), every sample of every context becomes its own anchor ⇒ 27 feedback items instead of 9. Default is 9.

For each anchor, the workflow:
1. Runs `base_model.rank(x, candidate_action_ids, default_action_id, top_k=5)` to capture the baseline top-5 (`:1979-1983`).
2. Resolves the **target action** based on `--target-mode` (`:1985-1993`):
   - `phase2_gt` (default): `target = phase2_gt_ro` from the spec
   - `model_top1`: `target = baseline top-1` (trivial sanity baseline)
   - `random`: uniform random legal action (noise baseline)
3. Records baseline rank & score of the target under the frozen model (`:1992`).

The final feedback-item structure `workflows.py:1994-2018` captures:
```
anchor_key, phase2_context_id, scenario_id, target_action_id,
reward = reward_scale (default 1.0), feedback_type="like",
anchor_episode_id, anchor_top5, baseline_anchor_gt_rank, baseline_anchor_gt_score
```

### 4.3 Baseline eval (workflows.py:2022-2065)

Before any updates, record for every anchor:
- **Anchor rank** of the target under the frozen model (`_rank_action_details`, `workflows.py:265-283`)
- **Same-scenario eval** — avg rank/score/top-3 rate of the target across the *other* two rows of the same `phase2_context_id` + all rows of other contexts in the same `scenario_id` (`workflows.py:2029-2032`)
- **Cross-scenario eval** — same metrics across the test rows of every *different* scenario (`workflows.py:2033`)
- **V6 coverage "before"** — `_evaluate_ro_quality_on_rows` (`workflows.py:2038-2044`) runs the model on all 27 test rows, counts how many V6 `most_relevant_3` actions fall in top-3 per row, averaged. This is the safety metric (see §6.5).

Result: `baseline_target_stats[anchor_key]` + `baseline_quality`, written to `locked_feedback_items_phase2.json` and `baseline_anchor_predictions_phase2.json`.

### 3.4 Condition sweep

The workflow iterates through conditions (`workflows.py:2094-2102`):
- `("single", 1)` — update only the anchor
- `("global-latent-nearest", N)` for each N in `n_values` (default `[1, 5, 10, 20, 50, 100]`)
- `("same-scenario-nearest", N)` for each N
- `("entire-scenario-all", "all")` — update every train row sharing the anchor scenario

Unless `--anchor-only` is set, in which case the whole list is replaced by a single `("anchor-only", anchor_repeat_k)` condition (`workflows.py:2096-2097`).

For each condition, load a fresh copy of the model (`workflows.py:2165`) so updates do not leak across conditions — each condition starts from the frozen baseline.

### 3.5 Propagation row selection (`_propagation_rows_for_item`, `workflows.py:2072-2092`)

| Mode | Rows returned |
|---|---|
| `single` | `[anchor]` |
| `global-latent-nearest` | top-N train rows by cosine sim to anchor latent, across the entire train pool |
| `same-scenario-nearest` | top-N train rows by cosine sim, restricted to rows with the same scenario_id |
| `entire-scenario-all` | every train row with the anchor's scenario_id |
| `anchor-only` | `[anchor] * anchor_repeat_k` (bypasses the propagation mode entirely) |

Cosine similarity is computed by `_sorted_neighbor_indices` (`workflows.py:249-262`): ranked desc, ties broken by `episode_id`.

### 4.4 Contrast targets (workflows.py:2104-2145)

If `--contrast-mode` is non-`none`, a list of `(action_id, negative_reward)` pairs is computed **once per feedback item** (from the model state at the start of that item's update loop, not per propagation row):

- `top1`: apply a negative update to the current rank-1 if it's not the target. If the rank-1 *is* the target, fall back to rank-2.
- `top3`: apply a negative update to whichever of the current top-3 are not the target, split evenly (`-negative_magnitude / len(competitors)`).

Negative magnitude = `positive_reward * contrast_reward_scale` (default 1.0).

### 4.5 Update loop (workflows.py:2163-2181)

For each propagation row:
```python
for _ in range(repeat_k):                                    # repeat-k
    model.partial_fit(row["x"], target_action_id, +reward)   # positive update
    for cid, neg_r in contrast_targets:                      # contrast (optional)
        model.partial_fit(row["x"], cid, neg_r)
```

Where `reward = 1.0 * reward_scale` (default 1.0).

`partial_fit` on `MaskedNeuralLinearUCB` (`linucb.py:930-940`) does:

```python
h = encode_context(x)                          # frozen trunk forward pass
A_inv_h = A_inv[a] @ h                         # precompute
A_inv[a] -= outer(A_inv_h, A_inv_h) / (1 + h @ A_inv_h)   # Sherman-Morrison
b[a] += reward * h
```

Only `A_inv[a]` and `b[a]` change, for the single action `a`. The encoder trunk is not touched. After all updates, `theta[a] = A_inv[a] @ b[a]` is the new linear head for that action.

### 4.6 Post-update evaluation (workflows.py:2178-2258)

For each feedback item the workflow records, per condition:

| field | meaning |
|-------|---------|
| `anchor.before.rank` | target's rank under the frozen model (1 = best) |
| `anchor.after.rank` | target's rank after propagation |
| `anchor.rank_delta` | `before - after` (positive = improvement) |
| `anchor.after.score` | raw UCB score of the target post-update |
| `same_scenario.avg_rank_delta` | avg improvement across same-scenario non-anchor test rows |
| `cross_scenario.avg_rank_delta` | avg improvement across test rows of *other* scenarios |
| `selected_scenarios_quality_before/after` | V6 top-3 most-relevant coverage (0-3), averaged across all 27 test rows |

Aggregates per condition (`workflows.py:2236-2258`):
- `avg_anchor_rank_delta`, `avg_same_scenario_rank_delta`, `avg_cross_scenario_rank_delta`
- `num_anchors_improved` (rank_delta > 0)
- `num_anchors_regressed` (rank_delta < 0)
- `num_anchors_in_top3_after`
- `max_anchor_regression` (min rank_delta)

### 4.7 Serving-alpha override (workflows.py:2147-2161)

If `--serving-alpha` is passed, the workflow temporarily overrides `model.alpha` while running the *after* evaluation (`workflows.py:2189-2234`). This changes only the UCB exploration multiplier `α · √(hᵀ A⁻¹ h)` at ranking time — it does not affect the updates themselves. The original alpha is restored inside `_AlphaOverride.__exit__`.

---

## 5. CLI — `simulate-feedback-propagation-phase2`

Defined in `recommendation_agents/cli.py:424-462` (argument parser) and `cli.py:955-1004` (dispatch).

```
python -m recommendation_agents.cli simulate-feedback-propagation-phase2 \
    --artifact-dir   artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split \
    --train-raw      artifacts/phase2/train_phase2.raw.jsonl \
    --test-raw       artifacts/phase2/test_phase2.raw.jsonl \
    --relevance-markdown docs/scenario_recommendation_actions_v6.md \
    --output-dir     <path> \
    [--n-values 1,5,10,20,50,100] \
    [--device cpu|cuda|auto] \
    # Ablation knobs (all optional):
    [--reward-scale FLOAT]          # multiply the +1 like (default 1.0)
    [--repeat-k INT]                # partial_fit repetitions per row (default 1)
    [--contrast-mode none|top1|top3] # negative updates for competitors (default none)
    [--contrast-reward-scale FLOAT]  # magnitude of negative update (default 1.0)
    [--serving-alpha FLOAT]          # override α at eval time (default: use trained α)
    [--anchor-only]                  # skip propagation, update anchor only
    [--anchor-repeat-k INT]          # repetitions when --anchor-only set (default 1)
    [--target-mode phase2_gt|model_top1|random]  # which action is "liked"
    [--random-target-seed INT]       # seed for target-mode=random (default 42)
    [--use-all-test-samples]         # treat every test row as an anchor (27 items)
```

### 5.1 Output files (in `--output-dir`)

- `results_phase2.json` — the full machine-readable result (all conditions, all feedback items, per-row before/after)
- `locked_feedback_items_phase2.json` — the 9 (or 27) locked feedback items with baseline
- `baseline_anchor_predictions_phase2.json` — simplified baseline top-5 per anchor
- `summary_table_phase2.md` — condition × context rank-after matrix
- `propagation_summary_tables_phase2.md` — per-item detailed before/after tables

---

## 6. Metrics

### 6.1 Anchor metrics (per feedback item)

- **`anchor.before.rank / .after.rank`**: where the target action sits in the 46-action R/O ranking *at the anchor context*. 1 = top choice.
- **`anchor.rank_delta = before - after`**: positive = the target moved up. The primary success indicator.
- **`anchor.score / .mean_reward / .uncertainty`**: raw UCB components on the anchor, useful for diagnosing whether a rank change came from mean-reward increase or uncertainty shrinkage.

### 6.2 Same-scenario generalisation (per feedback item)

Computed over test rows with the same `scenario_id` as the anchor but *different* `phase2_context_id` (or different sample of the same context).

- `same_scenario.avg_rank_delta` — did the update help or hurt on rows that share the scenario but have different feature context? A positive number suggests the propagation generalised in a context-sensitive way rather than over-fitting the anchor.

### 6.3 Cross-scenario collateral (per feedback item)

Over test rows with *different* `scenario_id`.

- `cross_scenario.avg_rank_delta` — by how much did the target action's rank change in completely unrelated scenarios?

**Interpretation note.** A *negative* cross-scenario delta is actually desirable: if you teach `O_SHOW_SCHEDULE` is the right answer for `ARRIVE_OFFICE_A`, you want its rank in `LEAVE_OFFICE`/`CAFE_QUIET` to *drop*, not rise. The `S1` success criterion (§7.3) that flags `|cross_delta| > 0.2` as a failure is therefore misleading — see `artifacts/phase2/ablations/ablations_comparison.md:34-37`.

### 6.4 V6 coverage (safety metric) — `_evaluate_ro_quality_on_rows` (`workflows.py:2490-2670`)

For every test row across the full 27-row test set:
1. Get the model's top-3 predictions.
2. Look up the V6 catalog's `most_relevant_3` for that row's scenario.
3. Count the intersection (`most_count = |model_top3 ∩ v6_most_relevant_3| ∈ [0, 3]`).
4. Average across all 27 rows.

This produces `avg_most_relevant_covered_in_topk` ∈ [0.0, 3.0], reported as `V6 before` / `V6 after`. **This is the real safety metric**: does feedback propagation preserve (or improve) the frozen model's ability to surface V6-most-relevant actions on every test row?

Computed before any updates (`baseline_quality`, `workflows.py:2038`) and again per condition after updates (`selected_scenarios_quality_after`, `workflows.py:2236-2243`). The post-update run respects `--serving-alpha` since it's inside the `_AlphaOverride` block.

### 6.5 Aggregate columns in results.json

```
conditions[*].aggregate = {
    avg_anchor_rank_delta,
    avg_same_scenario_rank_delta,
    avg_cross_scenario_rank_delta,
    num_anchors_improved,
    num_anchors_regressed,
    num_anchors_in_top3_after,
    max_anchor_regression,       # most-negative rank delta across items
}
conditions[*].selected_scenarios_quality_before  = baseline V6 coverage
conditions[*].selected_scenarios_quality_after   = post-update V6 coverage
```

---

## 7. Ablation matrix — `scripts/phase2_run_ablations.sh`

28 experiments covering 8 groups. Each writes `artifacts/phase2/ablations/<name>/results_phase2.json` and an accompanying log in `artifacts/phase2/ablations/logs/<name>.log`. Reruns skip if `results_phase2.json` already exists (delete the directory to force rerun — `phase2_run_ablations.sh:30-33`).

### 7.1 Groups

| Group | Count | Hypothesis / knob | Runs |
|---|---:|---|---|
| Vanilla | 1 | reference point, r=1, k=1 | `vanilla_r1_k1` |
| A. Reward magnitude | 5 | Single +1 update is weaker than ~200 accumulated offline updates; scale it | `reward_r{5,10,25,50,100}_n50` |
| B. Repeat-k | 3 | Can more updates per row substitute for higher reward? | `repeat_k{3,5,10}_n50` |
| C. Contrastive negative | 4 | Also push the current top-1 down, not just the target up | `contrast_top1_r{1,5,10}_n50`, `contrast_top3_r5_n50` |
| D. Large N | 3 | Does the signal strengthen with more neighbours? | `large_n_{250,500,1000}` |
| E. Serving alpha | 3 | Change UCB exploration bonus at ranking time only | `serving_alpha_{0p1,0p5,1p0}` |
| F. Anchor-only | 4 | Without propagation, how many repeats on the anchor alone? | `anchor_only_k{1,10,50,100}` |
| G. Combos | 3 | Stack reward + contrast + higher N | `combo_r10_contr_n{50,100}`, `combo_r25_contr_n100` |
| H. Target-mode baselines | 2 | Sanity: what if we teach the model's current top-1 or a random action? | `baseline_target_model_top1`, `baseline_target_random` |

Total: **28 runs** (plus the standalone `vanilla_r1_k1` entry = **29 with vanilla** counted separately; the aggregator pulls vanilla from `artifacts/phase2/feedback_propagation_phase2_v1/results_phase2.json` by default — see `phase2_aggregate_ablations.py:13`).

### 7.2 Invocation

```bash
cd recommendation_agents/
bash scripts/phase2_run_ablations.sh
```

The script pins the environment to `/data00/liyao/projects/sfm/Bandit_SFM/recommendation_agents/.venv/bin/python` (override with `PYTHON=... bash scripts/phase2_run_ablations.sh`) and runs each experiment with `set -e` so the first failure aborts the sweep.

Each experiment is skipped if its results already exist (idempotent reruns).

---

## 8. Aggregator — `scripts/phase2_aggregate_ablations.py`

```
python scripts/phase2_aggregate_ablations.py
```

Writes `artifacts/phase2/ablations/ablations_comparison.md`. Three tables:

### 8.1 Table 1 — Global ranking

Columns (`phase2_aggregate_ablations.py:125-126`):
```
ablation | mode | N | anchor Δ | improved/9 | top3/9 | same Δ | cross Δ | max reg | V6 before | V6 after
```

The "best condition" per ablation is picked by `pick_best_condition` (`:67-90`) using a 4-tuple ordering:
```
(avg_anchor_rank_delta, num_anchors_in_top3_after, v6_delta, num_anchors_improved)
```
Rows are sorted descending by `avg_anchor_rank_delta`.

### 8.2 Table 2: per-context rank-after

For each ablation's best condition, print the post-update rank of each of the 9 contexts. Formatting rules (`:175-181`):
- **Bold** = better than baseline
- ~~Strike~~ = worse than baseline
- Plain = unchanged

### 8.3 Table 3: Success criteria

Four criteria (`phase2_aggregate_ablations.py:184-210`):

| Code | Definition | Intent |
|------|------------|--------|
| **P1** | ≥ 3 of 6 contexts in `NEEDS_PROMOTION` reach rank ≤ 2 | Promote the under-ranked anchors |
| **P2** | `OFFICE_LUNCH_OUT_B` (plausible-tier diagnostic) reaches rank ≤ 3 | Reach the stubborn plausible-tier case |
| **S1** | `|avg cross-scenario rank delta| ≤ 0.2` | Don't disturb unrelated scenarios *(misleading — see §6.5)* |
| **S2** | `v6_coverage_after ≥ v6_coverage_before - 1e-9` | Don't damage the V6 most-relevant top-3 coverage |

`NEEDS_PROMOTION = [ARRIVE_OFFICE_C, CAFE_QUIET_A, CAFE_QUIET_B, LEAVE_OFFICE_A, OFFICE_LUNCH_OUT_A, OFFICE_LUNCH_OUT_B]` (`:15-22`). These are the 6 contexts the frozen model already gets "wrong enough" that improving them is the point of phase-2. (The other 3 already have rank 1 under the frozen model.)

An ablation **PASSes** only if all four are satisfied (`:205`). An ablation counts as part of "best" only if its condition's anchor delta isn't worse than vanilla.

**Important caveat:** S1 flags negative cross-deltas as failure even though negative cross-deltas are desirable. In practice, S2 (V6 coverage) is the only reliable safety metric — the comparison markdown's executive summary says so explicitly.

### 5.4 `pick_best_condition` ordering

From `phase2_aggregate_ablations.py:67-90`, the ordering key is a 4-tuple:
```
( anchor_delta,  num_top3_after,  v6_delta,  num_improved )
```
lexicographic, so ties on `anchor_delta` break on `num_top3_after`, then `v6_delta`, then `num_improved`. This is what gets written to the "best condition" row for each ablation.

### 8.5 Table 2 — per-context ranks

For each ablation, a row showing the post-update rank of the target at each of the 9 contexts. Formatting:
- `**n**` (bold) when rank improved vs. baseline
- `~~n~~` (strike) when rank regressed
- plain otherwise
- Baseline row on top labelled `**BASELINE**` (`:161-164`)

### 6.3 Table 3 — pass/fail grid

Columns `P1 | P2 | S1 | S2 | overall`. `overall = PASS` only if all four pass. (`:193-210`)

---

## 9. Reproducing the end-to-end experiment

```bash
# 1. Build the phase-2 split (one-time, produces ~27 test rows + normalised train file)
python scripts/phase2_split.py --samples-per-context 3

# 2. Run the vanilla baseline (produces artifacts/phase2/feedback_propagation_phase2_v1/)
python3 -m recommendation_agents.cli simulate-feedback-propagation-phase2 \
    --artifact-dir artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split \
    --train-raw  artifacts/phase2/train_phase2.raw.jsonl \
    --test-raw   artifacts/phase2/test_phase2.raw.jsonl \
    --n-values 1,5,10,20,50,100 \
    --device cpu

# 3. Run the 28-experiment ablation matrix (~25 min on CPU)
bash scripts/phase2_run_ablations.sh

# 4. Build the comparison markdown
python scripts/phase2_aggregate_ablations.py
```

The final comparison lives at `artifacts/phase2/ablations/ablations_comparison.md`.

---

## 10. Key findings (from the committed `ablations_comparison.md`)

1. **Reward scaling is the dominant lever.** `reward_r100_n50` (global-latent, N=50, reward=100) promotes 6/6 needs-promo anchors to top-2 and raises V6 coverage from **2.444 → 2.556**. This is because the frozen `b[a]` vectors have absorbed roughly 200 training updates each at reward=1; a single +100 update finally contributes comparable "voice" to the accumulated offline evidence.
2. **Contrast top-1 helps the anchor but hurts V6 coverage.** Aggressively de-ranking the current rank-1 penalises actions that are `most_relevant` *for other scenarios*, so V6 coverage drops to ~2.04. S2 fails.
3. **Repeat-k is strictly weaker than reward-scaling at comparable update strength.** Each repeat also updates `A`, which via Sherman-Morrison dampens subsequent updates (diminishing returns).
4. **Serving-alpha has no effect.** The frozen `A⁻¹` is already tight enough that the `√(hᵀ A⁻¹ h)` bonus is too small to flip ranks.
5. **Anchor-only (even k=100) matches vanilla.** Spreading updates across latent neighbours matters — repeated updates on a single row cannot generalise.
6. **Large-N alone (no reward scaling) is inefficient.** `large_n_1000` matches `vanilla` on anchor delta.

---

## 11. Known caveats and footguns

### 11.1 `CAFE_QUIET_A` GT mismatch

`docs/phase2_multi_recommendation_scenarios.md:164-180` specifies `O_SHOW_MEETING_DETAILS` as the recommendation for the "Upcoming meeting" context, but `scripts/phase2_split.py:225` hardcodes `"gt_ro": "O_SHOW_SCHEDULE"`. These are both plausible outputs for the same user intent, but the experiment and the spec document currently disagree. Results under `phase2_gt` mode are being compared against `O_SHOW_SCHEDULE`, not `O_SHOW_MEETING_DETAILS`. Either update the split script or the spec markdown before publishing.

### 11.2 S1 criterion is misleading

The aggregator flags an ablation as failing S1 if `|avg_cross_scenario_rank_delta| > 0.2`. But a *negative* cross delta (the target action moving down in unrelated scenarios) is *desirable* — it means the propagation is context-specific. Use S2 (V6 coverage) as the actual safety gate. The writeup in `artifacts/phase2/ablations/ablations_comparison.md:36-43` acknowledges this and re-scores ablations ignoring S1.

### 11.3 Baseline-fairness is subtle

`baseline_target_model_top1` is not a "no-op" baseline — it *does* call `partial_fit` on the model's current top-1, which generally nudges the top-1 higher. It passes P1/P2 trivially because the model "was already right". Treat it as a lower-bound floor, not a neutral control.

`baseline_target_random` picks a uniformly random legal action per item (reproducible with `--random-target-seed`). It should underperform vanilla on anchor delta; if it doesn't, something is wrong.

### 11.4 `single` vs `anchor_only` are not the same

- `single` mode runs **one** `partial_fit` on just the anchor (`workflows.py:2076`). It is a condition inside the normal sweep.
- `--anchor-only` skips the entire propagation sweep and does `anchor_repeat_k` back-to-back updates on the anchor (`workflows.py:2095-2097, 2170-2171`). It produces a different `condition_summary` shape (mode = `"anchor-only"`).

### 11.5 Propagation pool leaks test rows

`train_phase2.raw.jsonl` is explicitly constructed to exclude the 27 test row indices (`phase2_split.py:293, 382-393`), so there is no anchor-vs-anchor leakage. But the test split has ONLY 3 samples per context — cross-scenario metrics may be noisy.

### 11.6 Idempotent re-runs

`phase2_run_ablations.sh:30-33` skips any ablation whose output directory already contains `results_phase2.json`. To rerun a single ablation, delete its directory. Do **not** use `--output-dir` to point different runs at the same directory — you will overwrite the vanilla baseline used by the aggregator (`phase2_aggregate_ablations.py:13`).

---

## 12. Key file and line references

| Component | Path | Anchor |
|---|---|---|
| Context spec (9 contexts) | `docs/phase2_multi_recommendation_scenarios.md` | — |
| Splitter | `scripts/phase2_split.py` | `CONTEXTS`: 148, `verify_sample`: 63, `patch_and_verify`: 118, main: 257 |
| Workflow fn | `recommendation_agents/workflows.py` | `simulate_feedback_propagation_phase2_on_frozen_neural_linear`: 1823 |
| Latent neighbours | `recommendation_agents/workflows.py` | `_sorted_neighbor_indices`: 249 |
| Per-action ranker | `recommendation_agents/workflows.py` | `_rank_action_details`: 265 |
| Same/cross eval | `recommendation_agents/workflows.py` | `_evaluate_target_action_on_rows`: 293 |
| V6 coverage eval | `recommendation_agents/workflows.py` | `_evaluate_ro_quality_on_rows`: 334, 2614-2648 |
| Result dataclass | `recommendation_agents/workflows.py` | `FeedbackPropagationPhase2ExperimentSummary`: 151 |
| Model `partial_fit` | `recommendation_agents/linucb.py` | `MaskedNeuralLinearUCB.partial_fit`: 930-940 |
| Model `encode_context` | `recommendation_agents/linucb.py` | 872-880 |
| CLI parser | `recommendation_agents/cli.py` | 424-462 |
| CLI dispatch | `recommendation_agents/cli.py` | 955-1004 |
| Splitter entry | `scripts/phase2_split.py` | `main`: 234 |
| Ablation sweep | `scripts/phase2_run_ablations.sh` | 28 runs: 53-97 |
| Aggregator | `scripts/phase2_aggregate_ablations.py` | `pick_best_condition`: 67, `render_markdown`: 97 |
| Baseline vanilla path | `scripts/phase2_aggregate_ablations.py:13` | `artifacts/phase2/feedback_propagation_phase2_v1/results_phase2.json` |

---

## 13. Quick reference — end-to-end example

```bash
# Setup
cd recommendation_agents/
conda activate sfm

# Build split (only needed once)
python scripts/phase2_split.py --samples-per-context 3

# Run the winning ablation manually
python3 -m recommendation_agents.cli simulate-feedback-propagation-phase2 \
    --artifact-dir artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split \
    --train-raw artifacts/phase2/train_phase2.raw.jsonl \
    --test-raw  artifacts/phase2/test_phase2.raw.jsonl \
    --relevance-markdown docs/scenario_recommendation_actions_v6.md \
    --output-dir artifacts/phase2/ablations/reward_r100_n50 \
    --n-values 50 --reward-scale 100 \
    --device cpu

# Inspect the per-context rank table
cat artifacts/phase2/ablations/reward_r100_n50/summary_table_phase2.md

# Run the full sweep + aggregation
bash scripts/phase2_run_ablations.sh
python scripts/phase2_aggregate_ablations.py
open artifacts/phase2/ablations/ablations_comparison.md
```
