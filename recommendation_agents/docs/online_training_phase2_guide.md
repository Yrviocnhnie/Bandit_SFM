# Online Training Phase 2 Guide

This document is the **replication handbook** companion to:

- [report_2026_04_13.md](/home/liyao/data00/projects/sfm/recommendation_agents/docs/report_2026_04_13.md)

Use this guide for:

- exact inputs
- implementation details
- reproduction commands
- output file definitions

Use the report for:

- the phase-2 goal
- the chosen method
- headline results
- the product-level interpretation

This document explains the **phase-2 balanced online-learning setup** for the
recommendation bandit model.

It is written so another teammate can:

- understand the phase-2 design
- reproduce the exact balanced experiment
- inspect the saved outputs
- understand how the input feedback data is prepared
- continue iterating on phase-2 multi-anchor personalization

This guide assumes the current working directory is the repository root:

```bash
Bandit_SFM/
```

Most commands then enter the package directory:

```bash
cd recommendation_agents
```

## 1. Goal

Phase 1 showed that a **single feedback anchor** can locally personalize the
ranking for similar contexts.

Phase 2 asks the harder and more product-relevant question:

> If the same scenario contains multiple different feedback anchors, can the
> model learn different local action preferences for different contexts within
> that same scenario?

Examples:

- one `OFFICE_LUNCH_OUT` context may like `O_SHOW_PAYMENT_QR`
- another `OFFICE_LUNCH_OUT` context may like `R_ENJOY_LEISURE_MOMENT`
- one `LEAVE_OFFICE` context may like `O_SHOW_COMMUTE_TRAFFIC`
- another `LEAVE_OFFICE` context may dislike that same action

The phase-2 experiment keeps the encoder frozen and updates only the
lightweight LinUCB heads, so it remains conservative and easy to reason about.

## 2. Base Model

Phase 2 starts from the current best offline artifact:

- `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split`

Model type:

- `NeuralEncoder LinUCB`
- CLI model type: `neural-linear`

Structure:

- a frozen neural encoder
- disjoint per-action linear UCB heads

During phase-2 online adaptation:

- the encoder is frozen
- the latent space is fixed
- only the UCB heads are updated through replay on propagated local contexts

This means phase-2 tests **online personalization on top of a stable offline
representation**, rather than retraining the full model.

## 3. Required Inputs

### 3.1 Offline Artifact

Required directory:

- `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split`

Important files inside:

- `ro_model/`
- `ro_metadata.json`
- `train.raw.jsonl`
- `test.raw.jsonl`

These provide:

- the frozen R/O model
- the action metadata and candidate sets
- the offline train contexts used for local propagation replay
- the offline test contexts used for evaluation

Phase 2 currently focuses on **R/O actions only**.

### 3.2 Relevance Catalog

Required file:

- `recommendation_agents/docs/scenario_recommendation_actions_v6.md`

This is used for selected-scenario quality evaluation. It is not used to choose
the simulated feedback action.

### 3.3 Phase-2 Feedback Markdown

Required file:

- `Bandit_SFM/data/phase2_multi_recommendation_scenarios_v1_2.md`

This file defines the explicit multi-anchor phase-2 simulation data.

Current scope:

- scenarios: `4`
- anchor contexts: `9`
- feedback items: `18`

Scenarios included:

- `ARRIVE_OFFICE`
- `LEAVE_OFFICE`
- `OFFICE_LUNCH_OUT`
- `CAFE_QUIET`

Each anchor context has:

- one `like`
- one `dislike`

Important detail:

- unlike phase 1, phase 2 does **not** lock a target action by rank position
- the markdown provides explicit `like` and `dislike` `target_action_id`s
- some `like` actions are intentionally outside the baseline top-3, so the
  experiment can test whether online learning can pull a preferred action up
  from outside the shown set

### 3.4 Conda Environment

The run script expects the existing `sfm` conda environment:

```bash
conda run --no-capture-output -n sfm ...
```

## 4. Files And Entry Points

Main run script:

- [run_feedback_margin_policy_fixed_radius_v1_phase2_balanced.sh](/home/liyao/data00/projects/sfm/recommendation_agents/run_feedback_margin_policy_fixed_radius_v1_phase2_balanced.sh)

Main CLI command:

```bash
python -m recommendation_agents.cli simulate-feedback-propagation
```

Main implementation files:

- [recommendation_agents/cli.py](/home/liyao/data00/projects/sfm/recommendation_agents/recommendation_agents/cli.py)
- [recommendation_agents/workflows.py](/home/liyao/data00/projects/sfm/recommendation_agents/recommendation_agents/workflows.py)
- [recommendation_agents/feedback_specs.py](/home/liyao/data00/projects/sfm/recommendation_agents/recommendation_agents/feedback_specs.py)

What each file does:

- `feedback_specs.py`
  - parses the phase-2 markdown into explicit feedback specs
  - builds fully specified anchor contexts by filling missing raw fields with
    stable defaults
- `cli.py`
  - exposes `simulate-feedback-propagation`
  - accepts `--feedback-spec-markdown`
  - exposes balanced allocation knobs:
    - `--min-neighbors`
    - `--max-neighbors`
- `workflows.py`
  - loads the frozen model and offline raw data
  - caches train/test latent encodings
  - builds baseline target stats
  - allocates local neighbors
  - replays feedback into the UCB heads
  - evaluates before/after rank movement

## 5. How The Phase-2 Input Data Is Prepared

### 5.1 Source Markdown Structure

The phase-2 markdown file is organized as:

- scenario section
- context section
- explicit partial features
- explicit `like`
- explicit `dislike`

Each context includes:

- `Features`
- `Offline baseline top-3`
- `Simulated feedback`
- `Reason`
- optional `Caveat`

### 5.2 Anchor Context Construction

The phase-2 markdown only lists the features needed to describe each context.
It does not list every raw field from the offline schema.

To turn each context into a valid model input, `build_phase2_anchor_context(...)`
fills missing fields with stable defaults, such as:

- default hour from `ps_time`
- default phone/location/network/activity values
- default user demographic buckets
- default calendar/SMS/battery fields

This produces a full raw-style context that can be encoded by the offline
feature pipeline.

### 5.3 Parsed Feedback Items

`parse_phase2_feedback_markdown(...)` converts the markdown into explicit
feedback specs with fields such as:

- `feedback_id`
- `anchor_id`
- `scenario_id`
- `feedback_type`
- `target_action_id`
- `anchor_context`
- `anchor_context_id`
- `anchor_context_title`

Because `like` and `dislike` for the same context share one `anchor_id`, they
share the same anchor latent and local neighbor region.

That is important for phase 2: a single context can express both positive and
negative preference signals over the same local region.

## 6. Phase-2 Balanced Policy

The final phase-2 setup uses:

- propagation mode: `hard-assigned-local-balanced`
- reward policy: `margin-aware-fixed-radius-v1`
- `min_neighbors = 500`
- `max_neighbors = 2000`
- `similarity_threshold = 0.80`

This is the phase-2 balanced setup documented here.

### 6.1 Why Balanced Allocation Is Needed

The original phase-2 local mode was:

- `hard-assigned-local-cutoff`

Failure mode:

- some anchors owned no local region at all
- those anchors got `effective_n = 0`
- those feedback items could not learn

Balanced allocation fixes this by ensuring each anchor gets a minimum support
floor before dominant anchors absorb the rest of the scenario.

### 6.2 Exact Balanced Neighbor Allocation Logic

Balanced local assignment is implemented by
`_build_balanced_local_assignment(...)`.

For each scenario:

1. collect all same-scenario offline train contexts
2. compute cosine similarity from every anchor to every same-scenario train
   context in frozen latent space
3. floor stage:
   - each anchor receives up to `target_min` contexts
   - `target_min = min(min_neighbors, max_neighbors, floor(context_count / anchor_count))`
   - contexts are assigned one time only
   - anchors are served in increasing current-count order, so smaller anchors
     are filled first
   - this stage ignores the similarity threshold
4. fill stage:
   - remaining unassigned contexts are processed in order of strongest maximum
     anchor similarity
   - each context is assigned to the most similar anchor that:
     - has not reached `max_neighbors`
     - has similarity `>= similarity_threshold`
5. final local contexts for each anchor are sorted by anchor similarity

Important detail:

- the floor stage is what prevents anchor starvation
- the threshold only applies during the fill stage
- one train context belongs to only one anchor in this balanced mode

## 7. Reward Policy

Phase 2 reuses the same reward logic as the final phase-1 policy:

- `margin-aware-fixed-radius-v1`

The workflow first ranks the target action on the anchor context using the
frozen base model, then chooses reward from the anchor's local score margin.

Reward table:

| gap bin | condition | like reward | dislike reward |
| --- | --- | ---: | ---: |
| boundary | like rank 1 or dislike last rank | `+2` | `-1` |
| small | gap `< 0.05` | `+2` | `-0.1` |
| medium | `0.05 <= gap < 0.5` | `+5` | `-1` |
| large | gap `>= 0.5` | `+10` | `-10` |

This is intentionally conservative for small-gap dislike and more aggressive
only when the baseline model is strongly confident in the opposite direction.

Important implementation detail:

- in the current phase-2 implementation, the score gap is computed from the
  original frozen offline base model
- the target action and reward for every feedback item are locked before the
  sequential update loop starts
- the reward is **not** recomputed after earlier feedback items have already
  changed the model

So the current design is:

```text
lock all margins from offline base model -> choose rewards -> run sequential online updates
```

This makes the experiment reproducible and prevents reward selection from being
entangled with update order.

Potential future improvement:

- before processing each feedback item, recompute the score gap using the
  latest updated model

That could be better when multiple anchors interact or conflict, because later
feedback items would use a reward strength matched to the model's current state
rather than the original offline state. The tradeoff is that results would
depend more strongly on feedback ordering.

## 8. End-To-End Pipeline

The balanced phase-2 run goes through three major stages.

### 8.1 Preparation

Preparation includes:

1. load the frozen offline artifact
2. read `train.raw.jsonl` and `test.raw.jsonl`
3. encode train/test rows into:
   - `x`
   - frozen latent
4. save or reload the encoding cache
5. parse the phase-2 markdown
6. construct explicit anchor contexts and their latents
7. compute or reload baseline target stats
8. compute or reload neighbor ownership/order caches

With caches enabled, the balanced phase-2 run recorded:

- `preparation_seconds = 1.998305082321167`

### 8.2 Update

For each feedback item:

1. select the propagated offline train rows owned by that anchor
2. assign the chosen reward to the target action on those propagated rows
3. call the model's online update path on the disjoint LinUCB head for that
   action

Current update order:

- feedback items are processed sequentially in the parsed markdown order
- for one anchor context, `like` is processed before `dislike`
- within one feedback item, propagated rows are replayed in descending latent
  similarity order to that anchor
- the model is updated cumulatively, so later feedback items see the model
  after earlier feedback items have already modified it

Because the encoder is frozen, this is a light online replay step rather than
full neural retraining.

For the documented balanced run:

- `update_elapsed_seconds = 0.8745777606964111`

### 8.3 Evaluation

After the update, the workflow measures before/after target behavior on:

- the anchor context itself
- neighbor contexts
- held-out same-scenario test contexts
- sampled cross-scenario test contexts

It also computes selected-scenario quality metrics against the relevance
catalog.

For the documented balanced run:

- `evaluation_elapsed_seconds = 12.913280725479126`

Other recorded timing values:

- `condition_elapsed_seconds = 13.787909984588623`
- `total_seconds = 19.174080848693848`

## 9. How To Reproduce The Balanced Phase-2 Run

From `Bandit_SFM/`:

```bash
cd recommendation_agents
bash ./run_feedback_margin_policy_fixed_radius_v1_phase2_balanced.sh
```

The script runs:

```bash
conda run --no-capture-output -n sfm python -m recommendation_agents.cli simulate-feedback-propagation \
  --artifact-dir artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split \
  --relevance-markdown docs/scenario_recommendation_actions_v6.md \
  --feedback-spec-markdown ../data/phase2_multi_recommendation_scenarios_v1_2.md \
  --propagation-modes hard-assigned-local-balanced \
  --n-values 2000 \
  --similarity-thresholds 0.80 \
  --min-neighbors 500 \
  --max-neighbors 2000 \
  --feedback-reward-policy margin-aware-fixed-radius-v1 \
  --cross-scenario-sample-size 2000 \
  --output-dir artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_margin_policy_fixed_radius_v1_phase2_balanced \
  --device cpu
```

Optional GPU run:

```bash
cd recommendation_agents
DEVICE=cuda:0 bash ./run_feedback_margin_policy_fixed_radius_v1_phase2_balanced.sh
```

## 10. Saved Outputs

Balanced phase-2 output directory:

- `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_margin_policy_fixed_radius_v1_phase2_balanced`

Files produced:

- [results.json](/home/liyao/data00/projects/sfm/recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_margin_policy_fixed_radius_v1_phase2_balanced/results.json)
  - full machine-readable output
  - timing, aggregate metrics, per-feedback metrics
- [summary_table.md](/home/liyao/data00/projects/sfm/recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_margin_policy_fixed_radius_v1_phase2_balanced/summary_table.md)
  - one-table summary of the run
- [locked_feedback_spec.json](/home/liyao/data00/projects/sfm/recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_margin_policy_fixed_radius_v1_phase2_balanced/locked_feedback_spec.json)
  - exact frozen anchor contexts, locked actions, rewards, and baseline rankings
- [baseline_anchor_predictions.json](/home/liyao/data00/projects/sfm/recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_margin_policy_fixed_radius_v1_phase2_balanced/baseline_anchor_predictions.json)
  - baseline top predictions for each anchor
- [phase2_balanced_compare.md](/home/liyao/data00/projects/sfm/recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_margin_policy_fixed_radius_v1_phase2_balanced/phase2_balanced_compare.md)
  - comparison against the original `hard-assigned-local-cutoff` phase-2 run

Output sizes for the documented run:

- balanced phase-2 output dir: about `680 KB`
- reused feedback cache: about `54 MB`

The encoding cache inside `feedback_propagation_cache` includes both:

- `x_values`
- `latent_values`

For the current offline artifact, the saved encoding cache files are roughly:

- train encodings: about `26.1 MiB` on disk
- test encodings: about `6.5 MiB` on disk

## 11. What The Evaluation Measures

For each feedback item, the result tracks:

- anchor rank before/after
- anchor score delta
- neighbor movement
- same-scenario movement
- cross-scenario movement

Interpretation:

- anchor movement
  - whether the exact feedback context learned in the requested direction
- neighbor movement
  - whether directly propagated local train contexts moved in the intended
    direction
- same-scenario movement
  - whether held-out contexts in the same scenario also shifted
- cross-scenario movement
  - how much global drift leaked into other scenarios because the action head is
    shared globally

The current balanced phase-2 setup does **not** separately report
same-scenario non-neighbor regions. The `same_scenario` bucket is the held-out
same-scenario test region as currently implemented.

## 12. Results From The Documented Balanced Run

### 12.1 Headline Numbers

Final balanced phase-2 result:

- feedback items: `18`
- anchor contexts: `9`
- scenarios: `4`
- anchor success: `16 / 18`
- zero-neighbor anchors: `0`
- average effective neighbors per feedback item: `707.3`

Compared with the original phase-2 local cutoff mode:

- success improved from `12 / 18` to `16 / 18`
- zero-neighbor anchors dropped from `6` to `0`
- average effective neighbors stayed the same at `707.3`

This means the gain came from **better allocation**, not from using more data.

### 12.2 Feedback Items

| feedback_id | scenario | feedback | target position |
| --- | --- | --- | ---: |
| `arrive_office__context_a__like` | `ARRIVE_OFFICE` | like | 1 |
| `arrive_office__context_a__dislike` | `ARRIVE_OFFICE` | dislike | 3 |
| `arrive_office__context_b__like` | `ARRIVE_OFFICE` | like | 1 |
| `arrive_office__context_b__dislike` | `ARRIVE_OFFICE` | dislike | 3 |
| `arrive_office__context_c__like` | `ARRIVE_OFFICE` | like | 26 |
| `arrive_office__context_c__dislike` | `ARRIVE_OFFICE` | dislike | 3 |
| `leave_office__context_a__like` | `LEAVE_OFFICE` | like | 1 |
| `leave_office__context_a__dislike` | `LEAVE_OFFICE` | dislike | 3 |
| `leave_office__context_b__like` | `LEAVE_OFFICE` | like | 1 |
| `leave_office__context_b__dislike` | `LEAVE_OFFICE` | dislike | 2 |
| `office_lunch_out__context_a__like` | `OFFICE_LUNCH_OUT` | like | 3 |
| `office_lunch_out__context_a__dislike` | `OFFICE_LUNCH_OUT` | dislike | 2 |
| `office_lunch_out__context_b__like` | `OFFICE_LUNCH_OUT` | like | 6 |
| `office_lunch_out__context_b__dislike` | `OFFICE_LUNCH_OUT` | dislike | 3 |
| `cafe_quiet__context_a__like` | `CAFE_QUIET` | like | 3 |
| `cafe_quiet__context_a__dislike` | `CAFE_QUIET` | dislike | 2 |
| `cafe_quiet__context_b__like` | `CAFE_QUIET` | like | 3 |
| `cafe_quiet__context_b__dislike` | `CAFE_QUIET` | dislike | 2 |

### 12.3 Result Table

| feedback_id | feedback | reward | effective N | anchor rank before -> after | score delta | outcome |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `arrive_office__context_a__like` | like | `+2` | 566 | `1 -> 1` | `+0.109` | already top1; score margin increased |
| `arrive_office__context_a__dislike` | dislike | `-0.1` | 566 | `3 -> 3` | `+0.068` | did not move down |
| `arrive_office__context_b__like` | like | `+2` | 500 | `1 -> 1` | `+0.021` | already top1; score margin increased |
| `arrive_office__context_b__dislike` | dislike | `-0.1` | 500 | `3 -> 4` | `-0.047` | disliked action moved down |
| `arrive_office__context_c__like` | like | `+2` | 500 | `26 -> 17` | `+0.025` | liked action moved up, but remains outside top3 |
| `arrive_office__context_c__dislike` | dislike | `-0.1` | 500 | `3 -> 4` | `-0.004` | disliked action moved down slightly |
| `leave_office__context_a__like` | like | `+2` | 1079 | `1 -> 2` | `+0.134` | score increased, but lost top1 due to conflicting update |
| `leave_office__context_a__dislike` | dislike | `-10` | 1079 | `3 -> 46` | `-3.776` | disliked action moved strongly down |
| `leave_office__context_b__like` | like | `+2` | 521 | `1 -> 1` | `+0.209` | already top1; score margin increased |
| `leave_office__context_b__dislike` | dislike | `-0.1` | 521 | `2 -> 2` | `+0.132` | did not move down |
| `office_lunch_out__context_a__like` | like | `+2` | 1065 | `3 -> 1` | `+0.349` | liked action became top1 |
| `office_lunch_out__context_a__dislike` | dislike | `-0.1` | 1065 | `2 -> 46` | `-2.077` | disliked action moved strongly down |
| `office_lunch_out__context_b__like` | like | `+2` | 535 | `6 -> 3` | `+0.481` | liked action was pulled into top3 |
| `office_lunch_out__context_b__dislike` | dislike | `-10` | 535 | `3 -> 46` | `-2.337` | disliked action moved strongly down |
| `cafe_quiet__context_a__like` | like | `+2` | 500 | `3 -> 2` | `+0.061` | liked action moved up |
| `cafe_quiet__context_a__dislike` | dislike | `-0.1` | 500 | `2 -> 3` | `-0.003` | disliked action moved down slightly |
| `cafe_quiet__context_b__like` | like | `+2` | 1100 | `3 -> 1` | `+0.384` | liked action became top1 |
| `cafe_quiet__context_b__dislike` | dislike | `-0.1` | 1100 | `2 -> 3` | `-0.254` | disliked action moved down |

### 12.4 Aggregate Comparison Against Original Phase-2 Local Cutoff

| metric | original hard local cutoff | balanced local assignment |
| --- | ---: | ---: |
| feedback items | `18` | `18` |
| anchor success | `12 / 18` | `16 / 18` |
| active-anchor success | `11 / 12` | `16 / 18` |
| zero-neighbor anchors | `6` | `0` |
| average effective neighbors | `707.3` | `707.3` |
| aligned anchor rank movement | `+7.11` | `+8.33` |
| aligned anchor score movement | `+0.491` | `+0.560` |
| aligned same-scenario rank movement | `+6.80` | `+9.86` |
| aligned same-scenario score movement | `+0.510` | `+0.604` |

### 12.5 Neighbor, same-scenario, and other-scenario effects

In the phase-2 balanced results:

- `neighbors` means the propagated local train contexts actually replayed during
  the update
- `same-scenario` means held-out test contexts from the same scenario
- `other-scenario` means the sampled cross-scenario test contexts

Rank movement summary:

| feedback_id | anchor rank | neighbors avg rank | same-scenario avg rank | other-scenario avg rank |
| --- | --- | --- | --- | --- |
| `arrive_office__context_a__like` | `1 -> 1` | `1.18 -> 1.01` | `1.20 -> 1.18` | `6.87 -> 6.86` |
| `arrive_office__context_a__dislike` | `3 -> 3` | `3.04 -> 3.00` | `3.01 -> 3.00` | `16.67 -> 16.90` |
| `arrive_office__context_b__like` | `1 -> 1` | `1.78 -> 1.98` | `1.80 -> 1.82` | `8.88 -> 8.87` |
| `arrive_office__context_b__dislike` | `3 -> 4` | `5.32 -> 5.99` | `5.41 -> 5.40` | `19.76 -> 20.39` |
| `arrive_office__context_c__like` | `26 -> 17` | `3.38 -> 3.00` | `3.01 -> 3.00` | `17.45 -> 17.54` |
| `arrive_office__context_c__dislike` | `3 -> 4` | `28.48 -> 39.34` | `27.20 -> 28.79` | `24.71 -> 25.55` |
| `leave_office__context_a__like` | `1 -> 2` | `1.83 -> 2.00` | `1.82 -> 2.00` | `21.92 -> 21.95` |
| `leave_office__context_a__dislike` | `3 -> 46` | `2.44 -> 46.00` | `2.50 -> 46.00` | `29.72 -> 28.47` |
| `leave_office__context_b__like` | `1 -> 1` | `1.60 -> 1.00` | `1.68 -> 1.00` | `11.16 -> 11.31` |
| `leave_office__context_b__dislike` | `2 -> 2` | `1.87 -> 2.00` | `1.82 -> 2.00` | `22.26 -> 22.19` |
| `office_lunch_out__context_a__like` | `3 -> 1` | `1.74 -> 1.00` | `1.90 -> 1.00` | `34.17 -> 34.07` |
| `office_lunch_out__context_a__dislike` | `2 -> 46` | `2.19 -> 46.00` | `2.24 -> 46.00` | `20.88 -> 22.72` |
| `office_lunch_out__context_b__like` | `6 -> 3` | `4.98 -> 3.00` | `4.87 -> 3.00` | `13.32 -> 13.91` |
| `office_lunch_out__context_b__dislike` | `3 -> 46` | `2.44 -> 46.00` | `2.24 -> 46.00` | `20.66 -> 22.41` |
| `cafe_quiet__context_a__like` | `3 -> 2` | `36.32 -> 4.11` | `36.86 -> 4.44` | `26.73 -> 25.89` |
| `cafe_quiet__context_a__dislike` | `2 -> 3` | `32.28 -> 42.38` | `36.46 -> 42.86` | `29.95 -> 29.67` |
| `cafe_quiet__context_b__like` | `3 -> 1` | `2.93 -> 1.31` | `2.94 -> 1.31` | `24.87 -> 24.10` |
| `cafe_quiet__context_b__dislike` | `2 -> 3` | `2.07 -> 3.03` | `2.06 -> 3.03` | `18.23 -> 18.80` |

Score movement summary:

| feedback_id | anchor score delta | neighbors avg score delta | same-scenario avg score delta | other-scenario avg score delta |
| --- | ---: | ---: | ---: | ---: |
| `arrive_office__context_a__like` | `+0.109` | `+0.113` | `+0.012` | `+0.002` |
| `arrive_office__context_a__dislike` | `+0.068` | `+0.072` | `+0.007` | `+0.001` |
| `arrive_office__context_b__like` | `+0.021` | `+0.102` | `+0.010` | `+0.002` |
| `arrive_office__context_b__dislike` | `-0.047` | `-0.027` | `-0.001` | `-0.004` |
| `arrive_office__context_c__like` | `+0.025` | `+0.074` | `+0.007` | `+0.001` |
| `arrive_office__context_c__dislike` | `-0.004` | `-0.012` | `-0.001` | `-0.000` |
| `leave_office__context_a__like` | `+0.134` | `+0.161` | `+0.159` | `+0.000` |
| `leave_office__context_a__dislike` | `-3.776` | `-4.438` | `-4.412` | `-0.001` |
| `leave_office__context_b__like` | `+0.209` | `+0.242` | `+0.243` | `+0.000` |
| `leave_office__context_b__dislike` | `+0.132` | `+0.155` | `+0.159` | `+0.000` |
| `office_lunch_out__context_a__like` | `+0.349` | `+0.361` | `+0.371` | `+0.001` |
| `office_lunch_out__context_a__dislike` | `-2.077` | `-2.107` | `-2.192` | `-0.002` |
| `office_lunch_out__context_b__like` | `+0.481` | `+0.488` | `+0.454` | `+0.000` |
| `office_lunch_out__context_b__dislike` | `-2.337` | `-2.361` | `-2.192` | `-0.002` |
| `cafe_quiet__context_a__like` | `+0.061` | `+0.343` | `+0.333` | `+0.002` |
| `cafe_quiet__context_a__dislike` | `-0.003` | `-0.017` | `-0.016` | `-0.000` |
| `cafe_quiet__context_b__like` | `+0.384` | `+0.384` | `+0.385` | `+0.003` |
| `cafe_quiet__context_b__dislike` | `-0.254` | `-0.254` | `-0.254` | `-0.002` |

Interpretation:

- for successful feedback items, the neighbor and same-scenario regions usually
  move in the same direction as the anchor
- other-scenario effects are much smaller in magnitude, which is what we want
  from a local personalization update
- a few difficult items still show conflicting movement because phase-2 is now
  allowing multiple nearby anchors with different preferences to interact

## 13. Interpretation

The phase-2 balanced run supports the main product claim:

> Multiple different feedback anchors inside the same scenario can shape
> different local action orderings.

Three strong examples:

1. `OFFICE_LUNCH_OUT / Context A`
   - liked action `O_SHOW_PAYMENT_QR`: `3 -> 1`
   - disliked action `R_MEAL_BREAK`: `2 -> 46`

2. `OFFICE_LUNCH_OUT / Context B`
   - liked action `R_ENJOY_LEISURE_MOMENT`: `6 -> 3`
   - this is the clearest "outside baseline top-3 like" success case

3. `CAFE_QUIET / Context B`
   - liked action `O_TURN_ON_SILENT_MODE`: `3 -> 1`
   - disliked action `R_DEEP_WORK_WINDOW`: `2 -> 3`

The two remaining difficult cases are:

- `arrive_office__context_a__dislike`
- `leave_office__context_b__dislike`

These are useful future stress cases for phase-2 v3.

## 14. Recommendation For Future Work

Use `hard-assigned-local-balanced` as the new phase-2 baseline:

- propagation mode: `hard-assigned-local-balanced`
- `min_neighbors = 500`
- `max_neighbors = 2000`
- `similarity_threshold = 0.80`
- reward policy: `margin-aware-fixed-radius-v1`

Reason:

- it preserves locality
- it removes anchor starvation
- it improves total phase-2 success without increasing average data usage
- it is easier to extend to future multi-anchor refinement

Most natural next step:

- test limited soft overlap for conflict-heavy anchors, especially when nearby
  contexts express opposite preferences on the same action

## 15. Recommended Next Iterations

The current phase-2 balanced setup is a strong baseline, but there are three
especially promising next directions.

### 15.1 Recompute Score Gap On-The-Fly

Current behavior:

- score gap is computed once from the original frozen offline base model
- reward is locked before the sequential update loop

Potential next version:

- before each feedback item is applied, rank the anchor context using the
  **current updated model**
- recompute the latest local score gap
- choose the reward from that updated margin instead of the original offline
  margin

Why this may help:

- later feedback items would use a reward strength matched to the model's
  current state
- this is especially relevant in phase 2, where multiple nearby anchors can
  interact or conflict

Tradeoff:

- the experiment becomes more order-dependent
- the same feedback set may produce different rewards if processed in a
  different sequence

A reasonable phase-2 v3 comparison would be:

- `locked-offline-margin` reward selection
- `on-the-fly-updated-margin` reward selection

### 15.2 Better Alternatives To Minimum-Neighbor Balancing

Current behavior:

- balanced assignment guarantees each anchor a minimum support floor
- this solves the original anchor-starvation problem, where one dominant anchor
  could absorb nearly the whole scenario

This is already a strong practical fix, but it is not the only option.

Better or more flexible alternatives to test:

1. Soft overlap assignment

- instead of assigning each context to exactly one anchor, allow a context to
  contribute to the top-2 or top-3 nearest anchors
- use similarity-based weights
- this is especially useful when two nearby anchors express different
  preferences on related contexts

2. Distance-weighted replay

- keep hard ownership, but weight each propagated replay update by similarity
- closer contexts produce a larger update
- farther floor-assigned contexts produce a smaller update

3. Adaptive minimum support

- instead of a fixed `min_neighbors = 500`, set the minimum per scenario using:
  - scenario size
  - number of anchors
  - similarity concentration
- this would avoid over-forcing support in tiny or highly imbalanced scenarios

4. Shared plus local update

- split the propagated set into:
  - a very local anchor-owned region
  - a small shared scenario buffer
- use the local region for strong updates and the shared region for weak
  regularizing updates

5. Cluster-first local regions

- instead of direct anchor competition, first partition same-scenario contexts
  in latent space
- then let anchors update within their nearest cluster region
- this can reduce pathological winner-take-all ownership

The most natural next experiment after the current balanced baseline is:

- `soft overlap` or `distance-weighted replay`

These preserve locality while avoiding the brittleness of purely hard
assignment.

### 15.3 Improve The Latent Space And Offline Training

Current offline encoder training:

- supervised dense target over the full 46-action R/O space
- target values are roughly:
  - relevant = `1.0`
  - acceptable = `0.1`
  - other = `0.0`
  - irrelevant = `-0.1`
- the frozen encoder is then reused by the online LinUCB heads

This worked well enough to produce the current best offline artifact, but there
are several promising ways to improve it.

#### Better encoder objectives

1. Preference-margin training

- instead of only regressing to dense target values, explicitly train the
  encoder so:
  - relevant > acceptable
  - acceptable > other
  - other > irrelevant
- examples:
  - pairwise ranking loss
  - triplet / contrastive loss
  - margin-based listwise loss

This would train the latent space around **relative preference ordering**, which
may align better with both ranking and online bandit adaptation.

2. Weighted supervision

- increase the gap between label groups during offline training
- for example, test stronger separation than `1 / 0.1 / 0 / -0.1`

This may make the latent geometry cleaner, especially if acceptable and other
are currently too close.

3. Scenario-consistency or metric learning

- contexts from the same scenario or same action-preference pattern could be
  pulled closer in latent space
- contexts with very different suitable actions could be pushed apart

This may help phase-2 neighbor retrieval directly.

4. Multi-task encoder training

- keep the action-score supervision
- add an auxiliary objective such as:
  - scenario prediction
  - relevance-group prediction
  - contrastive context grouping

This could produce a more structured latent space for online propagation.

#### Better offline UCB training

1. Use the supervised head more directly

- current neural-linear style training freezes the encoder and then builds
  disjoint LinUCB heads on top
- one possible extension is to initialize the UCB head statistics from stronger
  supervised priors, not only from latent replay

2. Train with data that better reflects future online updates

- currently the offline target is dense action supervision
- a future version could add synthetic single-feedback or multi-feedback replay
  episodes during offline training

This may make the online update behavior more stable.

3. Train with stronger context diversity inside scenario

- phase 2 is sensitive to within-scenario latent geometry
- if future offline data covers more diverse contexts per scenario, the local
  propagation behavior should improve

#### Practical recommendation

If only one latent-space improvement is tried next, the highest-signal choice is
probably:

- keep the current encoder architecture
- add a **preference-ranking objective** on top of the dense target regression

If only one online-allocation improvement is tried next, the highest-signal
choice is probably:

- keep the current balanced allocation
- add **distance-weighted replay** or **soft overlap assignment**
