# Recommendation Agents

This repo contains two V0 contextual bandit agents built on the same feature space and the same LinUCB trainer:

- R/O agent: recommends a scenario-specific R/O action
- App agent: recommends 1 of 10 app categories

Both agents use:

- the V0 feature encoder from [bandit_feature_space_report.md](/home/liyao/data00/projects/sfm/recommendation_agents/docs/bandit_feature_space_report.md)
- a masked disjoint LinUCB model in [linucb.py](/home/liyao/data00/projects/sfm/recommendation_agents/recommendation_agents/linucb.py)
- offline replay training in [trainer.py](/home/liyao/data00/projects/sfm/recommendation_agents/recommendation_agents/trainer.py)

The difference is only the action space:

- R/O agent: uses the scenario-specific action sets from the current catalog document, [scenario_recommendation_actions_v4.md](/home/liyao/data00/projects/sfm/recommendation_agents/docs/scenario_recommendation_actions_v4.md)
- App agent: uses the 10 app categories, with one default app category per scenario from the same catalog document

## Model Summary

Current V0 model:

- feature encoder: hand-built 314-dim vector
- model: disjoint LinUCB, one linear scorer per action
- masking: only rank legal actions for the current scenario
- default prior: give the scenario default action an extra `default_bonus`
- exploration: add the LinUCB uncertainty term

This is not an MLP model. It is a linear contextual bandit.

## Setup

If you want to train or score, install PyTorch first. Example conda setup:

```bash
conda create -n sfm python=3.11 -y
conda activate sfm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

The code uses:

- GPU if you run with `--device auto` and CUDA is available
- CPU otherwise

## The Three Things You Plug In

For either agent, the practical flow is:

1. build metadata
2. convert raw JSONL into training samples
3. train and test the model

If you want a one-step path from raw JSONL to a trained artifact, use `train-v0-raw`.
If you want synchronized R/O and app training from the same shuffled raw file, use `train-v0-raw-dual`.

There are two kinds of files:

- `metadata.json`: the legal action space for each scenario
- `samples.jsonl`: one training event per emitted recommendation

## Raw Input File

Your colleague’s raw file can be in either of these formats:

- older sequential format with fields like `scenarioId`, `gt_ro_action_id`, `gt_app_category`
- newer finalized format with fields like `scenario_id`, nested `features`, `gt_ro`, `gt_app`

The converters in [raw_synthetic.py](/home/liyao/data00/projects/sfm/recommendation_agents/recommendation_agents/raw_synthetic.py) support both.

Current finalized sample:

- [synthetic_bandit_v0_two_scenarios_firststep_no_triggers.jsonl](/home/liyao/data00/projects/sfm/recommendation_agents/docs/synthetic_bandit_v0_two_scenarios_firststep_no_triggers.jsonl)
- ready-made score requests:
  [score_arrive_office.json](/home/liyao/data00/projects/sfm/recommendation_agents/artifacts/two_scenarios_firststep/score_arrive_office.json)
  [score_office_lunch_out.json](/home/liyao/data00/projects/sfm/recommendation_agents/artifacts/two_scenarios_firststep/score_office_lunch_out.json)

## R/O Agent: End-to-End

### 1. Build R/O metadata from the catalog doc

```bash
python3 -m recommendation_agents.cli build-ro-metadata \
  --input-markdown docs/scenario_recommendation_actions_v4.md \
  --output-metadata artifacts/ro_metadata.json
```

This creates metadata for:

- 65 scenarios
- 167 scenario-scoped R/O arms
- one default R/O action per scenario

### 2. Convert raw data into R/O training samples

```bash
python3 -m recommendation_agents.cli convert-v0-raw \
  --input docs/synthetic_bandit_v0_two_scenarios_firststep_no_triggers.jsonl \
  --output-samples artifacts/two_scenarios_firststep/v0_samples.jsonl \
  --output-metadata artifacts/two_scenarios_firststep/v0_metadata_inferred.json
```

What this does:

- keeps rows with a real R/O label
- maps raw fields into the V0 feature context
- writes one training row per emitted recommendation

Important:

- for real R/O training, use `artifacts/ro_metadata.json`
- the inferred metadata file is just a debugging/bootstrap artifact

### 3. Train the R/O model

```bash
python3 -m recommendation_agents.cli train-v0 \
  --metadata artifacts/ro_metadata.json \
  --samples artifacts/two_scenarios_firststep/v0_samples.jsonl \
  --output artifacts/two_scenarios_firststep/ro_model \
  --device auto
```

Or do conversion plus training in one command:

```bash
python3 -m recommendation_agents.cli train-v0-raw \
  --input docs/synthetic_bandit_v0_two_scenarios_firststep_no_triggers.jsonl \
  --output artifacts/two_scenarios_firststep/ro_model \
  --metadata artifacts/ro_metadata.json \
  --label-type ro \
  --device auto
```

This writes:

- `manifest.json`
- `weights.npz`
- `training_report.json`
- `metadata.snapshot.json`

Recommended output directory name:

- `artifacts/two_scenarios_firststep/ro_model`

### 4. Test the R/O model

Prepare a score request JSON like:

```json
{
  "scenario_id": "ARRIVE_OFFICE",
  "context": {
    "state_current": "office_arriving",
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
    "transportMode": "stationary",
    "activityState": "sitting",
    "activityDuration": 1500,
    "user_id_hash_bucket": "b7",
    "age_bucket": "25_34",
    "sex": "female",
    "has_kids": 0
  }
}
```

Best practice:

- use the full context field set from a converted training row
- keep the field names exactly the same as the V0 feature spec
- do not rely on partial context unless you know the missing fields are safe defaults

Then rank candidates:

```bash
python3 -m recommendation_agents.cli score-v0 \
  --artifact artifacts/two_scenarios_firststep/ro_model \
  --metadata artifacts/ro_metadata.json \
  --sample artifacts/two_scenarios_firststep/score_arrive_office.json \
  --top-k 3 \
  --device auto
```

Or choose one final action:

```bash
python3 -m recommendation_agents.cli choose-v0 \
  --artifact artifacts/two_scenarios_firststep/ro_model \
  --metadata artifacts/ro_metadata.json \
  --sample artifacts/two_scenarios_firststep/score_arrive_office.json \
  --device auto
```

Optional smoke-test randomness:

```bash
python3 -m recommendation_agents.cli choose-v0 \
  --artifact artifacts/two_scenarios_firststep/ro_model \
  --metadata artifacts/ro_metadata.json \
  --sample artifacts/two_scenarios_firststep/score_arrive_office.json \
  --epsilon 0.8 \
  --seed 7 \
  --device auto
```

## App Agent: End-to-End

### 1. Build app metadata from the catalog doc

```bash
python3 -m recommendation_agents.cli build-app-metadata \
  --input-markdown docs/scenario_recommendation_actions_v4.md \
  --output-metadata artifacts/app_metadata.json
```

This creates metadata for:

- 65 scenarios
- 10 app categories
- one default app category per scenario

### 2. Convert raw data into app training samples

```bash
python3 -m recommendation_agents.cli convert-v0-raw-app \
  --input docs/synthetic_bandit_v0_two_scenarios_firststep_no_triggers.jsonl \
  --output-samples artifacts/two_scenarios_firststep/v0_app_samples.jsonl \
  --output-metadata artifacts/two_scenarios_firststep/v0_app_metadata_inferred.json
```

What this does:

- keeps rows with a real app label
- converts raw fields into the same V0 context
- writes one training row per emitted app recommendation

Important:

- for real app training, use `artifacts/app_metadata.json`
- the inferred app metadata file is just a debugging/bootstrap artifact

### 3. Train the app model

```bash
python3 -m recommendation_agents.cli train-v0 \
  --metadata artifacts/app_metadata.json \
  --samples artifacts/two_scenarios_firststep/v0_app_samples.jsonl \
  --output artifacts/two_scenarios_firststep/app_model \
  --device auto
```

Or do conversion plus training in one command:

```bash
python3 -m recommendation_agents.cli train-v0-raw \
  --input docs/synthetic_bandit_v0_two_scenarios_firststep_no_triggers.jsonl \
  --output artifacts/two_scenarios_firststep/app_model \
  --metadata artifacts/app_metadata.json \
  --label-type app \
  --device auto
```

Recommended output directory name:

- `artifacts/two_scenarios_firststep/app_model`

### 4. Test the app model

The score request format is the same as the R/O agent:

- same `scenario_id`
- same `context`

Best practice:

- start from a real converted row and edit only the values you want to test

The only difference is the returned actions are app categories like `productivity`, `music`, or `navigation`.

Rank app categories:

```bash
python3 -m recommendation_agents.cli score-v0 \
  --artifact artifacts/two_scenarios_firststep/app_model \
  --metadata artifacts/app_metadata.json \
  --sample artifacts/two_scenarios_firststep/score_arrive_office.json \
  --top-k 5 \
  --device auto
```

Choose one final app category:

```bash
python3 -m recommendation_agents.cli choose-v0 \
  --artifact artifacts/two_scenarios_firststep/app_model \
  --metadata artifacts/app_metadata.json \
  --sample artifacts/two_scenarios_firststep/score_arrive_office.json \
  --device auto
```

## Dual Training

If you want both models from the same raw JSONL in one run, train them with one shared shuffled stream and interleaved train/eval windows:

```bash
python3 -m recommendation_agents.cli train-v0-raw-dual \
  --input "docs/bandit_v0_firststep_no_triggers_1000eps(1).jsonl" \
  --output artifacts/bandit_v0_firststep_dual \
  --ro-metadata artifacts/ro_metadata.json \
  --app-metadata artifacts/app_metadata.json \
  --alpha 0.15 \
  --alpha-end 0.01 \
  --device cpu \
  --train-window 100 \
  --eval-window 10 \
  --progress-every 100 \
  --shuffle-seed 0 \
  --tensorboard-logdir artifacts/bandit_v0_firststep_dual/tensorboard
```

This writes:

- `artifacts/bandit_v0_firststep_dual/ro_model`
- `artifacts/bandit_v0_firststep_dual/app_model`
- `artifacts/bandit_v0_firststep_dual/dual_training_summary.json`
- `artifacts/bandit_v0_firststep_dual/tensorboard`

What this dual command does:

- shuffles the raw rows once
- uses the same sample order for `RO` and `APP`
- trains for `100` rows, then evaluates top-1 for `10` rows
- decays LinUCB exploration from `alpha` to `alpha-end`
- logs separate `[RO]` and `[APP]` metrics after each eval block
- writes TensorBoard scalars for `RO` and `APP` train/eval reward and accuracy

Launch TensorBoard with:

```bash
tensorboard --logdir artifacts/bandit_v0_firststep_dual/tensorboard
```

## What the Sample Files Look Like After Conversion

Converted R/O samples look like:

```json
{
  "event_id": "arrive_office_ep01:0",
  "scenario_id": "ARRIVE_OFFICE",
  "context": { "...": "..." },
  "selected_action": "arrive_office_schedule",
  "reward": 1.0,
  "propensity": 1.0
}
```

Converted app samples look like:

```json
{
  "event_id": "arrive_office_ep01:0",
  "scenario_id": "ARRIVE_OFFICE",
  "context": { "...": "..." },
  "selected_action": "productivity",
  "reward": 1.0,
  "propensity": 1.0
}
```

## Notes and Caveats

- `shown_actions` is optional in training samples. If missing, the trainer uses the full legal action set from metadata.
- For the current finalized sample file, there are only 4 rows, so training is only a smoke test.
- The model is a linear contextual bandit, not an MLP.
- The default prior means the scenario default action gets an extra score bonus during ranking.
- If another action gets enough evidence, it can still beat the default.

## Useful Files

- R/O design: [bandit_design_note_zh.md](/home/liyao/data00/projects/sfm/recommendation_agents/docs/bandit_design_note_zh.md)
- App design: [app_category_agent_design_zh.md](/home/liyao/data00/projects/sfm/recommendation_agents/docs/app_category_agent_design_zh.md)
- Feature spec: [bandit_feature_space_report.md](/home/liyao/data00/projects/sfm/recommendation_agents/docs/bandit_feature_space_report.md)
- Action catalog: [scenario_recommendation_actions_v4.md](/home/liyao/data00/projects/sfm/recommendation_agents/docs/scenario_recommendation_actions_v4.md)
