# Online Training Phase 1 Guide

This document explains the current **phase-1 online learning / user-feedback adaptation** design for the recommendation bandit model.

It is written so another teammate can:

- understand the design
- reproduce the current experiment
- inspect the outputs
- continue toward phase 2, where multiple feedback anchors within the same scenario should personalize different contexts differently

This guide assumes your current working directory is the repository root:

```bash
Bandit_SFM/
```

Most commands then enter the package directory:

```bash
cd recommendation_agents
```

## 1. Goal

The best offline model has learned to recommend reasonable actions for each scenario/context. Phase 1 asks a different question:

> If a user gives explicit feedback on one shown action, can we update only the lightweight UCB layer so future recommendations reflect that user's preference?

Example:

- baseline top-3 for `OFFICE_WORKING` includes action A at rank 2
- user likes action A
- after online update, action A should move toward rank 1 for similar `OFFICE_WORKING` contexts

Another example:

- baseline top-3 for `OFFICE_TO_CAFE` includes action B at rank 3
- user dislikes action B
- after online update, action B should move down for similar `OFFICE_TO_CAFE` contexts

## 2. Base Model

Phase 1 starts from the current best offline artifact:

- `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split`

Model type:

- `NeuralEncoder LinUCB`
- CLI model type: `neural-linear`

Model structure:

- frozen neural encoder
- per-action disjoint linear UCB heads

In phase 1:

- the encoder is not updated
- the latent space is stable
- only the UCB head for the feedback action is updated

This is intentionally conservative. It gives us online adaptation without invalidating existing UCB statistics.

## 3. Required Inputs

### 3.1 Offline Artifact

Required directory:

- `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split`

Important files inside:

- `train.raw.jsonl`
- `test.raw.jsonl`
- `ro_metadata.json`
- `ro_model/`

The online phase-1 experiment currently focuses on **R/O actions only**.

### 3.2 Relevance Catalog

Required file:

- `recommendation_agents/docs/scenario_recommendation_actions_v6.md`

This is used only for evaluation, not for choosing the feedback action. Feedback actions are selected from the frozen model's baseline ranking.

### 3.3 Feedback Case Spec

Required file:

- `recommendation_agents/docs/feedback_propagation_policy_v1_cases.json`

Current six feedback items:

| feedback_id | scenario | feedback | target position |
| --- | --- | --- | ---: |
| `arrive_office_like_rank1` | `ARRIVE_OFFICE` | like | 1 |
| `office_lunch_out_dislike_rank1` | `OFFICE_LUNCH_OUT` | dislike | 1 |
| `office_working_like_rank2` | `OFFICE_WORKING` | like | 2 |
| `leave_office_dislike_rank2` | `LEAVE_OFFICE` | dislike | 2 |
| `cafe_stay_like_rank3` | `CAFE_STAY` | like | 3 |
| `office_to_cafe_dislike_rank3` | `OFFICE_TO_CAFE` | dislike | 3 |

Important detail:

- the JSON specifies `target_position`, not a hard-coded action id
- the workflow first runs the frozen model on the anchor context
- then it locks the action currently at that rank

This makes the experiment test actual model behavior rather than a hand-picked action.

### 3.4 Scenario id in real online feedback

The current simulation uses known `scenario_id` because the anchor contexts come from `test.raw.jsonl`.

In a real online system, we should not assume the incoming feedback event has a reliable scenario id. A realistic feedback event may only contain:

- raw context features
- the displayed action id
- feedback type, such as like/dislike
- optional user/session metadata

To recover an anchor scenario for phase-1 propagation:

1. encode the online feedback context with the frozen encoder
2. search nearest neighbors in the offline training latent index
3. collect the `scenario_id` values from the top nearest neighbors
4. assign the anchor scenario by majority vote
5. use that inferred scenario for `hard-assigned-local-cutoff` propagation

This works because the offline training data already contains scenario ids, and the frozen encoder maps similar contexts near each other.

A practical default for scenario inference:

```text
nearest_k_for_scenario_vote = 50 or 100
minimum_vote_share = 0.50
```

If the top scenario does not reach the minimum vote share, the system can either:

- skip propagation and update only the exact feedback context
- lower confidence and use a smaller radius
- queue the feedback for batch processing

This scenario inference step is important for deployment: phase 1 should not depend on a production scenario classifier.

## 4. Phase-1 Policy

The final phase-1 policy is:

- name: `Margin-Aware Fixed-Radius Policy v1`
- propagation mode: `hard-assigned-local-cutoff`
- fixed propagation radius: `N=2000`
- cosine similarity threshold: `0.80`
- reward policy: `margin-aware-fixed-radius-v1`

### 4.1 Why fixed radius?

Each scenario in the current offline data has roughly 2k+ unique contexts. Using `N=2000` means:

- with one feedback anchor, the update can cover most of the same-scenario context region
- with multiple feedback anchors in phase 2, the same scenario can be split into anchor-local regions

This is cleaner than tuning both reward and `N` at the same time.

### 4.2 What `similarity_threshold=0.80` means

For each feedback anchor, the workflow:

1. encodes the anchor context into frozen latent space
2. encodes offline training contexts into the same latent space
3. keeps only same-scenario contexts assigned to that anchor
4. filters to contexts with cosine similarity >= `0.80`
5. caps the result at `N=2000`

In the previous grid run, `0.80` gave a slightly wider coverage than `0.95` while behaving similarly in rank movement and drift. It is a reasonable phase-1 default.

### 4.3 Margin-aware reward selection

The policy first measures the score gap around the feedback target action.

For `like`:

- if the target is not rank 1, use:
  - `up_gap = previous_rank_score - target_score`
- this is the score amount needed to pass the action immediately above it

For `dislike`:

- if the target is not the last action, use:
  - `down_gap = target_score - next_rank_score`
- this is the score amount needed to fall below the action immediately below it

Reward table:

| gap bin | gap range | like reward | dislike reward |
| --- | ---: | ---: | ---: |
| boundary | like rank 1 or dislike last rank | `+2` | `-1` |
| small | `< 0.05` | `+2` | `-0.1` |
| medium | `0.05 - 0.5` | `+5` | `-1` |
| large | `>= 0.5` | `+10` | `-10` |

The small-gap dislike reward is intentionally `-0.1`, not `-1.0`. With fixed `N=2000`, `-1.0` was too aggressive for a small rank gap and could push a disliked top action far below the intended local adjustment.

## 5. Implementation Entry Points

Main CLI:

```bash
python -m recommendation_agents.cli simulate-feedback-propagation
```

Main workflow function:

- `simulate_feedback_propagation_on_frozen_neural_linear(...)`

Main files:

- `recommendation_agents/cli.py`
- `recommendation_agents/workflows.py`

The CLI argument added for the final policy:

```bash
--feedback-reward-policy margin-aware-fixed-radius-v1
```

Default behavior remains unchanged:

```bash
--feedback-reward-policy fixed
```

## 6. How To Reproduce The Final Phase-1 Run

From `Bandit_SFM/`:

```bash
cd recommendation_agents
bash ./run_feedback_margin_policy_fixed_radius_v1.sh
```

The script runs:

```bash
conda run --no-capture-output -n sfm python -m recommendation_agents.cli simulate-feedback-propagation \
  --artifact-dir artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split \
  --relevance-markdown docs/scenario_recommendation_actions_v6.md \
  --feedback-spec-json docs/feedback_propagation_policy_v1_cases.json \
  --propagation-modes hard-assigned-local-cutoff \
  --n-values 2000 \
  --similarity-thresholds 0.80 \
  --feedback-reward-policy margin-aware-fixed-radius-v1 \
  --cross-scenario-sample-size 2000 \
  --output-dir artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_margin_policy_fixed_radius_v1 \
  --device cpu
```

## 7. Output Files

Final output directory:

- `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_margin_policy_fixed_radius_v1`

Key files:

### `locked_feedback_spec.json`

The resolved feedback items after locking action ids from frozen baseline predictions.

Includes:

- `feedback_id`
- `scenario_id`
- `feedback_type`
- `target_position`
- locked `target_action_id`
- selected reward
- reward policy metadata
- margin gap and gap bin
- anchor context
- baseline top-3/top-5 predictions

### `baseline_anchor_predictions.json`

The frozen model's top predictions for each anchor before online adaptation.

### `results.json`

Full machine-readable output.

Includes:

- before/after rank for every feedback target action
- before/after score
- same-scenario propagation effects
- cross-scenario effects
- selected-scenario offline relevance metrics
- timing and cache information

### `summary_table.md`

Auto-generated compact condition summary.

### `phase1_policy_summary.md`

Human-readable summary for the final phase-1 policy run.

This is the best file to open first.

## 8. Current Results

Final run:

- `N=2000`
- `similarity_threshold=0.80`
- six feedback items applied together

| feedback_id | feedback | reward | effective N | anchor rank | interpretation |
| --- | --- | ---: | ---: | --- | --- |
| `arrive_office_like_rank1` | like rank1 | `+2` | 1566 | `1 -> 1` | already rank 1; score margin increased |
| `office_lunch_out_dislike_rank1` | dislike rank1 | `-0.1` | 1600 | `1 -> 3` | disliked action moved down but stayed in top3 |
| `office_working_like_rank2` | like rank2 | `+2` | 1640 | `2 -> 1` | liked action moved to top1 |
| `leave_office_dislike_rank2` | dislike rank2 | `-0.1` | 1600 | `2 -> 3` | disliked action moved down one slot |
| `cafe_stay_like_rank3` | like rank3 | `+2` | 2000 | `3 -> 1` | liked action moved to top1 |
| `office_to_cafe_dislike_rank3` | dislike rank3 | `-10` | 2000 | `3 -> 8` | large-gap disliked action moved out of top3 |

Aggregate selected-scenario offline metrics:

| metric | before | after |
| --- | ---: | ---: |
| most-relevant covered in top3 | 2.952 | 2.826 |
| acceptable covered in top3 | 2.972 | 2.972 |
| irrelevant in top3 | 0.000 | 0.000 |

The small decrease in offline most-relevant@3 is expected when the user dislikes an action that was originally labeled as offline most-relevant. For personalized online learning, target-action movement and replacement quality are more important than strict agreement with the original offline label.

Runtime with caches:

- preparation: about 2.0 seconds
- update: about 0.8 seconds
- evaluation: about 6.4 seconds
- total: about 12.8 seconds

Cache and update volume:

- current feedback cache size: about `45 MB`
- final policy output size: about `92 KB`
- calibration grid output size: about `7.6 MB`
- effective training contexts per feedback item: `[1566, 1600, 1640, 1600, 2000, 2000]`
- average effective training contexts per feedback item: `1734.3`
- total replay/update points for the six feedback items: `10406`

The cache size is dominated by latent encoding caches:

- train encodings: about `27 MB`
- test encodings: about `6.9 MB`
- final phase-1 neighbor-order cache: about `3.3 MB`

## 9. How To Interpret The Metrics

### Anchor rank

This is the main demonstration metric.

- like should move the target action up
- dislike should move the target action down
- rank1 like cannot move up, so inspect score and margin instead

### Score delta

This shows whether the UCB head actually changed.

- positive for like
- negative for dislike

### Same-scenario average rank delta

This shows whether the feedback generalized to other contexts in the same scenario.

For a like feedback:

- positive means the target action moved up

For a dislike feedback:

- negative means the target action moved down

### Cross-scenario drift

This measures side effects on other scenarios.

Some drift is expected because action heads are global, but large drift should be monitored before deployment.

### Neighbor, same-scenario, and other-scenario effects

The final phase-1 run reports three useful propagation regions:

| region | meaning |
| --- | --- |
| `neighbors` | training contexts selected for feedback propagation |
| `same_scenario` | held-out test contexts from the same scenario, excluding the anchor |
| `cross_scenario` | sampled held-out test contexts from other scenarios |

The current `same_scenario_outside_neighbors` field is not yet a strict non-neighbor split; for the current run, it should be treated as equivalent to `same_scenario`.

Rank movement summary:

| feedback_id | anchor rank | neighbor avg rank | same-scenario avg rank | other-scenario avg rank |
| --- | --- | --- | --- | --- |
| `arrive_office_like_rank1` | `1 -> 1` | `1.14 -> 1.00` | `1.20 -> 1.06` | `6.59 -> 6.75` |
| `office_lunch_out_dislike_rank1` | `1 -> 3` | `1.88 -> 3.00` | `1.86 -> 3.00` | `21.37 -> 25.53` |
| `office_working_like_rank2` | `2 -> 1` | `1.52 -> 1.00` | `1.54 -> 1.01` | `6.12 -> 6.30` |
| `leave_office_dislike_rank2` | `2 -> 3` | `2.47 -> 3.00` | `2.50 -> 3.00` | `29.57 -> 30.31` |
| `cafe_stay_like_rank3` | `3 -> 1` | `2.63 -> 1.00` | `2.91 -> 1.57` | `34.01 -> 33.83` |
| `office_to_cafe_dislike_rank3` | `3 -> 8` | `2.53 -> 5.94` | `2.46 -> 5.08` | `21.87 -> 26.38` |

Score movement summary:

| feedback_id | neighbor score delta | same-scenario score delta | other-scenario score delta |
| --- | ---: | ---: | ---: |
| `arrive_office_like_rank1` | `+0.2678` | `+0.0241` | `+0.0124` |
| `office_lunch_out_dislike_rank1` | `-0.5552` | `-0.5544` | `-0.1293` |
| `office_working_like_rank2` | `+0.2386` | `+0.2315` | `+0.0095` |
| `leave_office_dislike_rank2` | `-0.5454` | `-0.5451` | `-0.0001` |
| `cafe_stay_like_rank3` | `+0.3646` | `+0.3080` | `+0.0014` |
| `office_to_cafe_dislike_rank3` | `-1.0310` | `-1.0097` | `-0.0920` |

The main readout:

- neighbor contexts move strongly in the intended direction
- same-scenario held-out contexts also move in the intended direction, which shows generalization beyond the anchor
- other-scenario contexts can drift because action heads are global, especially for actions reused across many scenarios
- cross-scenario drift should be monitored, but the current phase-1 result still keeps irrelevant@3 at `0.000`

### Offline relevance metrics

These metrics compare against the offline scenario labels.

They are useful but should not be treated as absolute truth after user feedback, because online user preference can override offline weak labels.

## 10. Calibration Grid

Before the final policy, we ran a broader calibration grid:

- script: `recommendation_agents/run_feedback_propagation_margin_policy_v1_grid.sh`
- output: `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_policy_v1`

The grid tested:

- `N = 100, 500, 1000, 2000, 4000`
- `similarity_threshold = 0.70, 0.80, 0.85, 0.90, 0.95`
- like rewards: `+1, +2, +5, +10, +20`
- dislike rewards: `-1, -2, -5, -10, -20, -40`

Main conclusions:

- threshold had relatively small impact in the current latent space
- `0.80` is a good practical default
- fixed `N=2000` is useful for scenario-level personalization
- small-gap dislike must use a much smaller reward than large-gap dislike
- large-gap dislike needs a strong reward to move the action out of top3

## 11. Phase-2 Direction

Phase 1 validates a single-anchor-per-scenario style of online adaptation.

Phase 2 should test multiple feedback anchors inside the same scenario.

Target behavior:

- one user may like action A in one `OFFICE_WORKING` context
- the same user may dislike action A or prefer action B in another `OFFICE_WORKING` context
- the system should learn different local preferences within the same scenario

The current `hard-assigned-local-cutoff` mode is already designed for this:

1. group feedback anchors by scenario
2. compute latent similarity from every same-scenario training context to every anchor
3. assign each context to its nearest anchor
4. apply the feedback update only through that anchor's assigned local region
5. cap each anchor region at `N=2000`
6. filter with `similarity_threshold=0.80`

Suggested phase-2 experiment:

- create 2-3 feedback anchors within the same scenario
- use different target actions and feedback directions
- run them together in one feedback spec
- verify that each anchor changes its local context region without collapsing the whole scenario into a single preference

Recommended phase-2 metrics:

- per-anchor target rank before/after
- local assigned-region target rank before/after
- same-scenario outside-anchor-region drift
- cross-scenario drift
- replacement action quality after dislikes
- personalized top3 quality where disliked actions are removed from the preferred set

## 12. Related Files

- offline training guide: `recommendation_agents/docs/offline_training_guide_EN.md`
- feedback case spec: `recommendation_agents/docs/feedback_propagation_policy_v1_cases.json`
- final phase-1 script: `recommendation_agents/run_feedback_margin_policy_fixed_radius_v1.sh`
- calibration grid script: `recommendation_agents/run_feedback_propagation_margin_policy_v1_grid.sh`
- final phase-1 results: `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_margin_policy_fixed_radius_v1`
- calibration grid results: `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_policy_v1`
