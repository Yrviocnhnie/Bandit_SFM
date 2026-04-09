# Online Training Phase 1 Guide

This document describes the current **phase-1 online training / online adaptation** setup for the best offline model:

- [v6_2k_neural_linear_3p3a_hardneg_other_stratified_split](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split)

Phase 1 is intentionally simple:

- the neural encoder is **frozen**
- only the **disjoint linear UCB head** is updated
- user feedback is simulated or replayed as `(context, action, reward)` updates

This is the safest first step because:

- online updates are cheap
- the latent space stays stable
- we do not need to refresh or rebuild the encoder

## 1. What “online training” means in phase 1

The base offline model has already learned:

- a frozen encoder: `306 -> 128 -> 64`
- one disjoint LinUCB head per R/O action

In phase 1, online learning means:

1. the system shows recommended actions for a context
2. the user gives explicit feedback on one action
   - for example: like / dislike
3. that feedback is converted into a reward
4. the UCB head for that action is updated

The encoder is not retrained in this phase.

## 2. Current experiment entrypoint

The current implementation entrypoint is:

- `python -m recommendation_agents.cli simulate-feedback-propagation`

Implementation lives in:

- [cli.py](../recommendation_agents/cli.py)
- [workflows.py](../recommendation_agents/workflows.py)

Main workflow function:

- `simulate_feedback_propagation_on_frozen_neural_linear(...)`

## 3. What this phase-1 workflow needs

### 3.1 A trained offline artifact

The workflow starts from an existing offline artifact directory, for example:

- `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split`

This artifact must already contain:

- `train.raw.jsonl`
- `test.raw.jsonl`
- `ro_metadata.json`
- `ro_model/`

### 3.2 A relevance catalog

The current workflow also needs:

- [scenario_recommendation_actions_v6.md](scenario_recommendation_actions_v6.md)

This is used to measure whether propagation helps or hurts overall recommendation quality on the selected scenarios.

## 4. What the current phase-1 experiment actually does

The current implementation is an **R/O-only feedback propagation experiment**.

It uses four office-related scenarios:

- `ARRIVE_OFFICE`
- `OFFICE_LUNCH_OUT`
- `OFFICE_WORKING`
- `LEAVE_OFFICE`

For each scenario, it selects one anchor context from `test.raw.jsonl`, gets the frozen model’s baseline ranking, and then defines a feedback item from that baseline ranking.

Current feedback pattern:

- `ARRIVE_OFFICE`: like the baseline rank-1 action
- `OFFICE_LUNCH_OUT`: dislike the baseline rank-1 action
- `OFFICE_WORKING`: like the baseline rank-2 action
- `LEAVE_OFFICE`: dislike the baseline rank-2 action

Important:

- the target action ids are **not manually specified**
- they are locked from the frozen model’s own baseline predictions

## 5. Propagation modes

Phase 1 currently evaluates four propagation modes.

### `single`

Only the original feedback context is used.

This is the most realistic online baseline:

- one real feedback
- one update

### `global-latent-nearest-N`

The feedback is propagated to the `N` nearest contexts in the **entire offline training set**, using cosine similarity in frozen latent space.

This is the most realistic propagation heuristic when real online data does not have a reliable scenario label.

### `same-scenario-nearest-N`

The feedback is propagated to the `N` nearest contexts within the **same scenario** in the offline training set.

This is a stronger oracle-style version because it uses scenario membership.

### `entire-scenario-all`

The feedback is propagated to **all training contexts in the same scenario**.

This is best treated as an upper bound or aggressive propagation setting, not as the default production behavior.

## 6. Reward definition in phase 1

Current phase-1 reward mapping:

- `like = +1.0`
- `dislike = -1.0`

This reward is applied to the selected action head only.

Because the encoder is frozen and the model uses disjoint linear UCB heads:

- only the feedback target action is directly updated
- other action heads are not directly changed
- ranking can still change because one action moves up or down relative to the others

## 7. How to run the current phase-1 experiment

Run from the repository root `Bandit_SFM/`:

```bash
conda run --no-capture-output -n sfm python -m recommendation_agents.cli simulate-feedback-propagation \
  --artifact-dir recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split \
  --relevance-markdown recommendation_agents/docs/scenario_recommendation_actions_v6.md \
  --n-values 1,5,10,20,50,100 \
  --device cpu
```

## 8. Output files

By default, the outputs are written to:

- `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/feedback_propagation_ro_v1/`

Key files:

### `locked_feedback_spec.json`

The final locked feedback items used in the run.

This includes:

- scenario
- feedback type
- target action id
- anchor episode id
- baseline top-3 / top-5

### `baseline_anchor_predictions.json`

The frozen model’s baseline predictions for the anchor contexts before any adaptation.

### `results.json`

The full experiment output.

This includes:

- all propagation modes
- all `N` values
- before/after rank details for every feedback item
- same-scenario and cross-scenario effects
- overall selected-scenario quality before/after

### `summary_table.md`

A compact summary table across all conditions.

### `propagation_summary_tables.md`

A more readable derived summary that includes:

- one table per feedback item
- an aligned aggregate table

## 9. How to interpret the results

### Per-feedback-item view

This is the most important view.

For each feedback item, inspect:

- anchor rank before/after
- same-scenario rank change
- cross-scenario rank change

This tells you:

- whether the target action moved in the intended direction
- how much that change generalized to similar contexts
- whether the update spilled over too much into unrelated contexts

### Aligned aggregate view

The aligned aggregate table uses a direction-aware metric:

- for `like`, moving the action forward is positive
- for `dislike`, moving the action backward is positive

So positive aligned values always mean the model changed in the direction preferred by the user feedback.

### Overall quality view

Also inspect:

- `selected most_rel before`
- `selected most_rel after`

This tells you whether a more aggressive propagation setting is starting to damage the original ranking quality on those scenarios.

## 10. Current practical takeaway

Based on the current phase-1 results:

- `single` is usually too weak
- `global-latent-nearest-N` is the most practical propagation strategy
- `same-scenario-nearest-N` behaves very similarly, which suggests frozen latent similarity is already strong
- `N = 10` or `N = 20` is the current sweet spot
- `entire-scenario-all` is too aggressive and starts to hurt overall quality

## 11. Why this is called “phase 1”

This is phase 1 because it avoids encoder updates.

If the encoder stays frozen:

- latent space stays stable
- old UCB statistics remain valid
- online adaptation is easy and cheap

Later phases may explore:

- stronger positive feedback rewards
- additional feedback items
- periodic encoder refresh
- rebuilding UCB heads after encoder updates

## 12. Related files

- offline model training:
  - [offline_training_guide.md](offline_training_guide.md)
  - [offline_training_guide_EN.md](offline_training_guide_EN.md)
- main experiment summary:
  - [report_2026_04_01.md](report_2026_04_01.md)
- relevance catalog:
  - [scenario_recommendation_actions_v6.md](scenario_recommendation_actions_v6.md)
