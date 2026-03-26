# Recommendation Agents

This repo contains the V0 shared-action LinUCB setup for two agents:

- `R/O` agent: ranks the shared global recommendation-action space
- `App` agent: ranks the 10 app categories

Both agents use the same context feature encoder and the same trainer. The only difference is the action space.

## What This V0 Setup Assumes

The current setup matches the shared-action V0 design:

- there is no per-scenario action mask at normal serving time
- `scenario_id` is not a model input
- the model input is only the context features, including `state_current` and `precondition`
- training data is context/action pairs
- `scenario_id` can still be stored in training rows for offline analysis and reporting

The catalog sources checked into this repo are:

- `docs/scenario_recommendation_actions_v5.md`
- `docs/scenario_recommendation_actions_v6.md`

The metadata builder reads that catalog and produces:

- `R/O` metadata with `46` global actions
- `App` metadata with `10` app categories
- per-scenario default rankings only as offline metadata

## Model Summary

Current V0 model:

- feature encoder: hand-built `306`-dim context vector
- model: disjoint LinUCB, one linear scorer per action
- serving action space:
  - `R/O`: all `46` global actions
  - `App`: all `10` app categories
- optional `shown_actions`: may still be passed for special tests, but is not needed in the normal v5 flow

This is a linear contextual bandit, not an MLP.

## Model I/O

Input:

- one context vector of dimension `306`
- this is the same for both heads

Output:

- `R/O` head:
  - one score per global R/O action
  - current output space size: `46`
  - at inference time, the model ranks all `46` actions and usually returns `top-3`
- `App` head:
  - one score per app category
  - current output space size: `10`
  - at inference time, the model ranks all `10` categories and usually returns `top-3`

So the model shape is:

- `R/O`: `306 -> 46`
- `App`: `306 -> 10`

These are not MLP layers. This is still a linear contextual bandit:

- input: `306`-dim feature vector
- output: per-action linear score
- ranking: sort by score and return top-k

## Environment Setup

Create a conda env and install PyTorch:

```bash
conda create -n sfm python=3.11 -y
conda activate sfm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Device behavior:

- `--device auto`: use GPU if available, else CPU
- `--device cuda`: require CUDA
- `--device cpu`: force CPU

## Quickstart

From `Bandit_SFM/recommendation_agents`, the recommended flow today is:

1. use `Plan A` if you want a fair comparison against yesterday's model
2. use `Plan B` if you want the new scenario-stratified split
3. both commands do data prep, training, and evaluation in one run

Training defaults for the V6 runs:

- `default_bonus = 0.0`
- `alpha = 0.05`
- `most_relevant / plausible / irrelevant rewards = 1.0 / 0.6 / 0.0`
- `track_train_hit_rate = False` by default, because it is expensive and not very useful for the main V6 runs

Current strongest candidate config from the recent experiments:

- `alpha = 0.0`
- `default_bonus = 0.0`
- `most_relevant / plausible / irrelevant rewards = 1.0 / 0.1 / 0.0`
- `most_relevant / plausible / irrelevant repeat = 1 / 1 / 1`
- `epochs = 1`

This is the current `v6_candidate` setup.

Current candidate artifacts:

- `Plan A`: `artifacts/v6_candidate_same_split_expanded`
- `Plan B`: `artifacts/v6_candidate_stratified_split`

Short status report:

- [report_2026_03_26.md](/home/liyao/data00/projects/sfm/recommendation_agents/docs/report_2026_03_26.md)

## Recommended V6 Workflow

### Plan A: Reuse Yesterday's Split

Use this when you want the cleanest comparison against yesterday's model.

- reuse the old `train.raw.jsonl` and `test.raw.jsonl`
- expand only the training side
- keep the test side unchanged

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n sfm python -m recommendation_agents.cli run-v6-plan-a \
  --input-data-dir artifacts/v5_1000eps_each_scenario_updated \
  --output-dir artifacts/v6_candidate_same_split_expanded \
  --relevance-markdown docs/scenario_recommendation_actions_v6.md \
  --most-relevant-reward 1.0 \
  --plausible-reward 0.1 \
  --irrelevant-reward 0.0 \
  --most-relevant-repeat 1 \
  --plausible-repeat 1 \
  --irrelevant-repeat 1 \
  --alpha 0.0 \
  --default-bonus 0.0 \
  --epochs 1 \
  --top-k 3 \
  --progress-every 10000 \
  --device cuda \
  --no-track-train-hit-rate
```

This writes the current candidate artifacts to:

- `artifacts/v6_candidate_same_split_expanded`

### Plan B: New Scenario-Stratified Split

Use this when you want the cleaner long-term split:

- split raw contexts by `scenario_id` at the raw row / episode level
- target `80/20` train/test inside each scenario
- expand only the train side
- keep test contexts unexpanded

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n sfm python -m recommendation_agents.cli run-v6-plan-b \
  --input ../data/bandit_v0_v5_1000eps_each_scenario_updated_preconditions_and_updated_state_current.jsonl \
  --catalog-markdown docs/scenario_recommendation_actions_v5.md \
  --output-dir artifacts/v6_candidate_stratified_split \
  --relevance-markdown docs/scenario_recommendation_actions_v6.md \
  --test-ratio 0.2 \
  --most-relevant-reward 1.0 \
  --plausible-reward 0.1 \
  --irrelevant-reward 0.0 \
  --most-relevant-repeat 1 \
  --plausible-repeat 1 \
  --irrelevant-repeat 1 \
  --alpha 0.0 \
  --default-bonus 0.0 \
  --epochs 1 \
  --top-k 3 \
  --progress-every 10000 \
  --device cuda \
  --no-track-train-hit-rate
```

This writes the current candidate artifacts to:

- `artifacts/v6_candidate_stratified_split`

Main outputs for both plans:

- `.../ro_train_samples_expanded.jsonl`
- `.../app_train_samples_expanded.jsonl`
- `.../ro_model/`
- `.../app_model/`
- `.../eval_both_top3.json`
- `.../run_v6_plan_a_summary.json` or `.../run_v6_plan_b_summary.json`

Training logs now focus on:

- `interval avg reward`
- `cumulative avg reward`
- throughput and ETA

If you explicitly enable `--track-train-hit-rate`, the logs will also print `top1`, but training will be slower.

## Legacy V5 Workflow

If you still need the older one-label-per-context workflow, use the commands below.

### 1. Prepare data

This one command does all of the following:

- builds `R/O` metadata
- builds `App` metadata
- splits the raw file into train/test by `episode_id`
- converts train/test raw rows into trainer-ready sample JSONL files for both agents

```bash
python3 -m recommendation_agents.cli prepare-v0-data \
  --input ../data/bandit_v0_v5_1000eps_each_scenario_updated_preconditions_and_updated_state_current.jsonl \
  --catalog-markdown docs/scenario_recommendation_actions_v5.md \
  --output-dir artifacts/v5_1000eps_each_scenario_updated \
  --test-ratio 0.2
```

Main outputs:

- `artifacts/v5_1000eps_each_scenario_updated/ro_metadata.json`
- `artifacts/v5_1000eps_each_scenario_updated/app_metadata.json`
- `artifacts/v5_1000eps_each_scenario_updated/ro_train_samples.jsonl`
- `artifacts/v5_1000eps_each_scenario_updated/ro_test_samples.jsonl`
- `artifacts/v5_1000eps_each_scenario_updated/app_train_samples.jsonl`
- `artifacts/v5_1000eps_each_scenario_updated/app_test_samples.jsonl`
- `artifacts/v5_1000eps_each_scenario_updated/prepare_summary.json`

### 2. Train both models

This one command trains both `R/O` and `App` from the prepared data directory.
Use `conda run --no-capture-output` if you want live progress logs, throughput, and ETA while training is running.

```bash
conda run --no-capture-output -n sfm python -m recommendation_agents.cli train-v0-both \
  --data-dir artifacts/v5_1000eps_each_scenario_updated \
  --alpha 0.05 \
  --progress-every 1000 \
  --device auto
```

During training you will see lines like:

```text
[RO] Processed 1000/48691 samples | ... | elapsed=1m42s | eta=1h06m | 9.8 samples/s
```

Main outputs:

- `artifacts/v5_1000eps_each_scenario_updated/ro_model/`
- `artifacts/v5_1000eps_each_scenario_updated/app_model/`
- `artifacts/v5_1000eps_each_scenario_updated/train_both_summary.json`

### 3. Evaluate both models

This one command evaluates both models on the prepared test split using `top_k=3`.
Use `conda run --no-capture-output` if you want live evaluation progress, throughput, and ETA while scoring the test set.

```bash
conda run --no-capture-output -n sfm python -m recommendation_agents.cli eval-v0-both \
  --data-dir artifacts/v5_1000eps_each_scenario_updated \
  --catalog-markdown docs/scenario_recommendation_actions_v6.md \
  --top-k 3 \
  --progress-every 1000 \
  --device auto
```

- `artifacts/v5_1000eps_each_scenario_updated/eval_both_top3.json`

The eval report now uses the `v6` scenario relevance definitions and puts the main metrics first.

Core metrics for both `R/O` and `App`:

- `avg_most_relevant_covered_in_topk`
  - average number of `most relevant 3` items that appear in the predicted top-3
  - example: `1.6` means the model recovers `1.6 / 3` of the scenario's most relevant items on average
- `avg_acceptable_covered_in_topk`
  - average number of predicted top-3 items that belong to `most relevant 3 + other plausible 3`
  - higher is better
- `avg_irrelevant_in_topk`
  - average number of predicted top-3 items that belong to the scenario's `irrelevant 2`
  - lower is better

Useful detail fields after the core metrics:

- `top6_predicted_action_distribution`
  - the most frequently predicted actions or app categories across the whole test set
- `per_scenario`
  - per-scenario averages for the same three metrics
  - the scenario-defined `most_relevant_3`, `other_plausible_3`, and `irrelevant_2`
  - `predicted_top6_action_distribution` for that scenario

This format is intentionally unified so you can compare:

- the older model trained with one sampled label per context
- the newer model trained with expanded multi-action samples per context

## Files Teammates Need To Provide

### 1. Raw synthetic JSONL

This is the only input your teammate must generate.

Each line should represent one emitted recommendation event with:

- `scenario_id` or `scenarioId`
- `episode_id`
- `scenario_elapsed_sec` or similar offset field
- `emit_recommendation`
- `gt_ro` or `gt_ro_action_id`
- `gt_app` or `gt_app_category`
- context fields either at top level or nested under `features`

The converter currently supports both:

- the older format
- the newer finalized format with nested `features`

### 2. Test request JSON

This is a single JSON file used for scoring or choosing.

It contains:

- `context`
- optional `shown_actions`
- no `scenario_id` is required for serving

## Raw Data Contract

A typical raw row looks like this:

```json
{
  "episode_id": "arrive_office_ep01",
  "scenario_id": "ARRIVE_OFFICE",
  "scenario_elapsed_sec": 0,
  "emit_recommendation": 1,
  "gt_ro": "O_SHOW_SCHEDULE",
  "gt_app": "productivity",
  "features": {
    "state_current": "office_arriving",
    "precondition": "commuting_walk_out",
    "state_duration_sec": 180,
    "ps_time": "morning",
    "hour": 9,
    "cal_hasUpcoming": 1,
    "ps_dayType": "workday",
    "ps_motion": "stationary",
    "wifiLost": 0,
    "wifiLostCategory": "work",
    "cal_eventCount": 3,
    "cal_inMeeting": 0,
    "cal_nextLocation": "work",
    "ps_sound": "quiet",
    "sms_delivery_pending": 0,
    "sms_train_pending": 0,
    "sms_flight_pending": 0,
    "sms_hotel_pending": 0,
    "sms_movie_pending": 0,
    "sms_hospital_pending": 0,
    "sms_ride_pending": 0,
    "timestep": 32400,
    "ps_location": "work",
    "ps_phone": "on_desk",
    "batteryLevel": 88,
    "isCharging": 1,
    "networkType": "wifi",
    "activityState": "sitting",
    "activityDuration": 900,
    "user_id_hash_bucket": "b07",
    "age_bucket": "25_34",
    "sex": "female",
    "has_kids": 0
  }
}
```

Important rules:

- keep only rows with real labels when training
- use valid v5 global R/O action IDs for `gt_ro`
- use valid app categories for `gt_app`
- `scenario_id` can be included for reporting, but it is not a model feature
- `precondition` should be provided if possible

## Trainer-Ready Sample Contract

After conversion, each training row looks like this:

```json
{
  "event_id": "arrive_office_ep01:0",
  "scenario_id": "ARRIVE_OFFICE",
  "context": {
    "state_current": "office_arriving",
    "precondition": "commuting_walk_out"
  },
  "selected_action": "O_SHOW_SCHEDULE",
  "reward": 1.0,
  "propensity": 1.0
}
```

Notes:

- `context` is the only model input
- `selected_action` is the supervised label
- for `R/O`, `selected_action` is one of the `46` global actions
- for `App`, `selected_action` is one of the `10` app categories

## Score Request Contract

The same request format works for both agents:

```json
{
  "context": {
    "state_current": "office_arriving",
    "precondition": "commuting_walk_out",
    "state_duration_sec": 220,
    "ps_time": "morning",
    "hour": 9,
    "cal_hasUpcoming": 1,
    "ps_dayType": "workday",
    "ps_motion": "stationary",
    "wifiLost": 0,
    "wifiLostCategory": "work",
    "cal_eventCount": 3,
    "cal_inMeeting": 0,
    "cal_nextLocation": "work",
    "ps_sound": "quiet",
    "sms_delivery_pending": 0,
    "sms_train_pending": 0,
    "sms_flight_pending": 0,
    "sms_hotel_pending": 0,
    "sms_movie_pending": 0,
    "sms_hospital_pending": 0,
    "sms_ride_pending": 0,
    "timestep": 32100,
    "ps_location": "work",
    "ps_phone": "on_desk",
    "batteryLevel": 79,
    "isCharging": 1,
    "networkType": "wifi",
    "activityState": "sitting",
    "activityDuration": 1500,
    "user_id_hash_bucket": "b7",
    "age_bucket": "25_34",
    "sex": "female",
    "has_kids": 0
  }
}
```

Optional test-only restriction:

```json
{
  "context": {
    "state_current": "office_arriving",
    "precondition": "commuting_walk_out"
  },
  "shown_actions": ["O_SHOW_SCHEDULE", "O_SHOW_TODAY_TODO"]
}
```

In the real v5 flow, you usually should not pass `shown_actions`.

## Step 1: Build Metadata

Run these once per catalog version.

### Build `R/O` metadata

```bash
python3 -m recommendation_agents.cli build-ro-metadata \
  --input-markdown docs/scenario_recommendation_actions_v5.md \
  --output-metadata artifacts/ro_metadata.json
```

### Build `App` metadata

```bash
python3 -m recommendation_agents.cli build-app-metadata \
  --input-markdown docs/scenario_recommendation_actions_v5.md \
  --output-metadata artifacts/app_metadata.json
```

## Step 2: Convert Raw Data

Replace `<raw.jsonl>` with the path to the teammate-provided synthetic file. For the current shared dataset in this repo, use `../data/bandit_v0_v5_1000eps_each_scenario_updated_preconditions_and_updated_state_current.jsonl`.

### Convert `R/O` labels

```bash
python3 -m recommendation_agents.cli convert-v0-raw \
  --input <raw.jsonl> \
  --output-samples artifacts/ro_samples.jsonl \
  --output-metadata artifacts/ro_metadata_inferred.json
```

### Convert `App` labels

```bash
python3 -m recommendation_agents.cli convert-v0-raw-app \
  --input <raw.jsonl> \
  --output-samples artifacts/app_samples.jsonl \
  --output-metadata artifacts/app_metadata_inferred.json
```

What the converters do:

- keep rows with real labels
- map raw fields into the V0 context schema
- validate that the context can be encoded
- write one trainer-ready event per recommendation call

Use the catalog-based metadata for real training. The inferred metadata files are only useful for debugging.

## Step 3: Train The Models

### Train `R/O`

```bash
python3 -m recommendation_agents.cli train-v0 \
  --metadata artifacts/ro_metadata.json \
  --samples artifacts/ro_samples.jsonl \
  --output artifacts/ro_model \
  --device auto
```

### Train `App`

```bash
python3 -m recommendation_agents.cli train-v0 \
  --metadata artifacts/app_metadata.json \
  --samples artifacts/app_samples.jsonl \
  --output artifacts/app_model \
  --device auto
```

Useful knobs:

- `--alpha`: LinUCB exploration strength
- `--l2`: linear-model regularization
- `--progress-every`: training log interval
- `--device`: `auto`, `cpu`, `cuda`, `cuda:0`

One-step raw-to-model commands:

```bash
python3 -m recommendation_agents.cli train-v0-raw \
  --input <raw.jsonl> \
  --output artifacts/ro_model \
  --metadata artifacts/ro_metadata.json \
  --label-type ro \
  --device auto
```

```bash
python3 -m recommendation_agents.cli train-v0-raw \
  --input <raw.jsonl> \
  --output artifacts/app_model \
  --metadata artifacts/app_metadata.json \
  --label-type app \
  --device auto
```

Training outputs:

- `manifest.json`
- `weights.npz`
- `training_report.json`
- `metadata.snapshot.json`

## Step 4: Test The Models

Create one request file, for example `artifacts/score_request.json`, then run either ranking or single-choice commands.

### Rank `R/O` actions

```bash
python3 -m recommendation_agents.cli score-v0 \
  --artifact artifacts/ro_model \
  --metadata artifacts/ro_metadata.json \
  --sample artifacts/score_request.json \
  --top-k 5 \
  --device auto
```

### Choose one `R/O` action

```bash
python3 -m recommendation_agents.cli choose-v0 \
  --artifact artifacts/ro_model \
  --metadata artifacts/ro_metadata.json \
  --sample artifacts/score_request.json \
  --device auto
```

### Rank `App` categories

```bash
python3 -m recommendation_agents.cli score-v0 \
  --artifact artifacts/app_model \
  --metadata artifacts/app_metadata.json \
  --sample artifacts/score_request.json \
  --top-k 5 \
  --device auto
```

### Choose one `App` category

```bash
python3 -m recommendation_agents.cli choose-v0 \
  --artifact artifacts/app_model \
  --metadata artifacts/app_metadata.json \
  --sample artifacts/score_request.json \
  --device auto
```

Optional smoke-test randomness:

```bash
python3 -m recommendation_agents.cli choose-v0 \
  --artifact artifacts/ro_model \
  --metadata artifacts/ro_metadata.json \
  --sample artifacts/score_request.json \
  --epsilon 0.2 \
  --seed 7 \
  --device auto
```

## Example End-To-End Run

```bash
python3 -m recommendation_agents.cli build-ro-metadata \
  --input-markdown docs/scenario_recommendation_actions_v5.md \
  --output-metadata artifacts/ro_metadata.json
```

```bash
python3 -m recommendation_agents.cli build-app-metadata \
  --input-markdown docs/scenario_recommendation_actions_v5.md \
  --output-metadata artifacts/app_metadata.json
```

```bash
python3 -m recommendation_agents.cli convert-v0-raw \
  --input ../data/bandit_v0_v5_1000eps_each_scenario_updated_preconditions_and_updated_state_current.jsonl \
  --output-samples artifacts/v5_1000eps_each_scenario_updated/ro_samples.jsonl \
  --output-metadata artifacts/v5_1000eps_each_scenario_updated/ro_metadata_inferred.json
```

```bash
python3 -m recommendation_agents.cli convert-v0-raw-app \
  --input ../data/bandit_v0_v5_1000eps_each_scenario_updated_preconditions_and_updated_state_current.jsonl \
  --output-samples artifacts/v5_1000eps_each_scenario_updated/app_samples.jsonl \
  --output-metadata artifacts/v5_1000eps_each_scenario_updated/app_metadata_inferred.json
```

```bash
python3 -m recommendation_agents.cli train-v0 \
  --metadata artifacts/ro_metadata.json \
  --samples artifacts/v5_1000eps_each_scenario_updated/ro_samples.jsonl \
  --output artifacts/v5_1000eps_each_scenario_updated/ro_model \
  --alpha 0.05 \
  --device auto
```

```bash
python3 -m recommendation_agents.cli train-v0 \
  --metadata artifacts/app_metadata.json \
  --samples artifacts/v5_1000eps_each_scenario_updated/app_samples.jsonl \
  --output artifacts/v5_1000eps_each_scenario_updated/app_model \
  --alpha 0.05 \
  --device auto
```

```bash
conda run --no-capture-output -n sfm python -m recommendation_agents.cli eval-v0-both \
  --data-dir artifacts/v5_1000eps_each_scenario_updated \
  --top-k 3 \
  --progress-every 1000 \
  --device auto
```
This writes:

- `artifacts/v5_1000eps_each_scenario_updated/eval_both_top3.json`

## Notes For Teammates

- `scenario_id` is optional bookkeeping. Do not rely on it at inference time.
- The most important part of the data is the quality of the context features.
- If two contexts are similar, the model can rank similar actions highly even if the scenario label was never passed in.
- `default_bonus` is mostly inactive in the shared-action v5 setup, because serving does not usually provide one concrete default action.
- If the data grows much larger or becomes strongly nonlinear, we can revisit whether V1 should move beyond disjoint LinUCB.

## Validation

Run unit tests:

```bash
python3 -m unittest discover -s tests -v
```
