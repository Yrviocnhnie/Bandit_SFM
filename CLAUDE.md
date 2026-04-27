# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo contains a Python prototype of a contextual bandit recommender system for HarmonyOS app scenarios. There are three subsystems:

1. **Root-level Dual UCB prototype** — an MLP-based Dual UCB model (726-dim features) that mirrors the C++ app-side implementation for debugging/training outside the app.
2. **`recommendation_agents/`** — a production-oriented V0 LinUCB scaffold (314-dim features) with two agents: R/O (scenario-specific actions) and App (app category recommendations). This is a *linear* contextual bandit, not an MLP.
3. **`app_usage_data/`** — single-user next-app prediction demo with two tasks (next-app + 15-min window set prediction) on 42 days of real HarmonyOS data. Independent of the above subsystems; shipped as v1 (GRU/TGT-lite baselines) → v2 (per-task hierarchical encoder) → v3 (feature enrichment + Markov fusion) → v4 (recency/periodicity ablation, falsified). Production picks: v1 GRU for Task A; v3 R6 for Task B (test EventHit@5 = 0.746).

These are separate model implementations with different feature spaces and architectures.

## Environment Setup

```bash
conda create -n sfm python=3.11 -y
conda activate sfm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

The `recommendation_agents/` package is installable via `pip install -e recommendation_agents/` (uses `pyproject.toml`, requires `numpy>=2.0` and `tensorboard`).

## Common Commands

### Root-level Dual UCB prototype

```bash
# Demo run
python ucb_cpp_port.py

# Synthetic training loop
python train_runner.py --steps 10 --topk 3
```

### recommendation_agents CLI

All commands run from inside the `recommendation_agents/` directory:

```bash
# Build metadata from catalog markdown
python3 -m recommendation_agents.cli build-ro-metadata --input-markdown docs/scenario_recommendation_actions_v4.md --output-metadata artifacts/ro_metadata.json
python3 -m recommendation_agents.cli build-app-metadata --input-markdown docs/scenario_recommendation_actions_v4.md --output-metadata artifacts/app_metadata.json

# Dual training (R/O + App from same raw JSONL)
python3 -m recommendation_agents.cli train-v0-raw-dual \
  --input <raw.jsonl> --output artifacts/bandit_v0_firststep_dual \
  --ro-metadata artifacts/ro_metadata.json --app-metadata artifacts/app_metadata.json \
  --device cpu

# Score / choose actions
python3 -m recommendation_agents.cli score-v0 --artifact <model_dir> --metadata <metadata.json> --sample <request.json> --top-k 3 --device auto
python3 -m recommendation_agents.cli choose-v0 --artifact <model_dir> --metadata <metadata.json> --sample <request.json> --device auto
```

See `recommendation_agents/run.sh` for a ready-made dual training invocation.

### app_usage_data (next-app prediction demo)

All commands run from inside `app_usage_data/`:

```bash
# v1 — baselines + shared-backbone GRU/TGT-lite
python scripts/01_prep_data.py        # build splits, vocab, sessions
python scripts/02_run_baselines.py    # MFU / MRU / HourMFU / Markov-1
python scripts/03_train_gru.py        # GRU-64 checkpoint
python scripts/04_train_tgt.py        # TGT-lite checkpoint
python scripts/05_window_eval.py      # neural Task A + Task B on the anchor grid

# v2 — per-task models (Local + Global + Profile, gated fusion)
python scripts/10_train_task_a_v2.py
python scripts/11_train_task_b_v2.py
python scripts/12_eval_v2.py

# v3 — v2 + feature enrichment (category, location, daypart, multi-window) + Markov prior fusion
python scripts/20_build_v3_features.py            # train-only fits: cat map, loc vocab, Markov table
python scripts/21_train_task_a_v3.py --tag task_a_v3_R4
python scripts/22_train_task_b_v3.py --use-markov --tag task_b_v3_R6   # SOTA Task B (test EH@5 = 0.746)

# v4 — adds per-app recency (8d) + periodicity priors (18d). Did NOT improve over v3.
python scripts/27_train_v4.py --task a --tag task_a_v4_full
python scripts/27_train_v4.py --task b --use-markov --tag task_b_v4_full
```

Documentation:
- `app_usage_data/README.md` — entry point with layout, reproduction, headline numbers
- `app_usage_data/REPORT.md` — v1 (baselines + shared backbone)
- `app_usage_data/REPORT_v2.md` — v2 (per-task hierarchical encoder)
- `app_usage_data/REPORT_v3.md` — v3 (feature enrichment + Markov prior)
- `app_usage_data/REPORT_v4.md` — v4 ablation experiment, full baseline comparison, production picks
- `app_usage_data/FEATURES.md` — every input feature explained (current v3 implementation)
- `app_usage_data/FEATURES_v2.md` — proposed feature redesign (audit + drops + adds)

### Tests

```bash
cd recommendation_agents && python -m pytest tests/

# Run a single test file
cd recommendation_agents && python -m pytest tests/test_catalog.py

# Run a single test by name
cd recommendation_agents && python -m pytest tests/test_catalog.py -k "test_name"
```

## Architecture

### Data Flow (Root-level prototype)

`PhysicalState / state chain` → `RuleEngine` (rule_engine.py) → `ContextSnapshot` → `DualBanditFeatureEncoderPy` (ucb_input_encoding.py) → `DualUCBModel` (dual_ucb_model.py)

- `feature_space.py`: defines the 726-dim feature vocabulary (state codes, location/time/day/motion/phone/sound classes, scenario IDs, action IDs)
- `state_and_scenario.py`: core data structures (`PhysicalState`, `ContextSnapshot`, `ScenarioMatch`, etc.) — loads scenario defs from `default_rules.json`
- `rule_engine.py`: encodes physical context → state code, matches rules/state-chain transitions → scenario
- `ucb_input_encoding.py`: encodes `ContextSnapshot` + scenario → 726-dim feature vector
- `dual_ucb_model.py`: Dual UCB with MLP prior, one head for R/O actions, one for APP categories

### Data Flow (recommendation_agents)

`raw JSONL` → `raw_synthetic.py` (convert) → `trainer.py` (LinUCB train/eval) → model artifacts (`manifest.json`, `weights.npz`, `training_report.json`)

- `catalog.py`: parses the catalog markdown to build metadata (legal action spaces per scenario)
- `feature_space.py`: 314-dim V0 feature encoder
- `linucb.py`: disjoint masked LinUCB model — one linear scorer per action, scenario-masked ranking
- `schemas.py`: data schemas (`ScoreRequest`, etc.)
- `metadata.py`: scenario/action metadata schemas (`ScenarioMeta`, `ActionMetadata`) loaded from catalog-built JSON
- `taxonomies.py`: action/category taxonomy definitions
- `cli.py`: argparse-based CLI entrypoint (subcommands: `train-v0`, `train-v0-raw`, `train-v0-raw-dual`, `score-v0`, `choose-v0`, `build-ro-metadata`, `build-app-metadata`, `convert-v0-raw`, `convert-v0-raw-app`)

### Key Design Points

- The LinUCB model uses **masking**: only legal actions for the current scenario are ranked
- A **default bonus** gives the scenario's default action an extra score prior, but other actions can still win with enough evidence
- Dual training shares the same shuffled sample stream for R/O and App agents with interleaved train/eval windows
- `data/generate_bandit_v0_firststep_no_triggers.py` generates synthetic training data

### Data Flow (app_usage_data — next-app prediction)

`app_usage_cleaned_dictionary_mapped.xlsx` (raw, 47k events / 7k targets) → `lib/data.py` (dedup, sessionize 300s idle cutoff with 32-event cap, chronological split 30/5/5 with 60-min embargo) → `artifacts/splits/{train,val,test}.parquet` → encoders → models → `artifacts/results/*.json`.

Layered architecture by version (each layer adds to the previous):

- `lib/` (v1): `data.py` (load XLSX, dedup, sessionize, vocab — collapse <5-event apps to <RARE>), `features.py` (per-event 28-d numeric + 32-d learned app embedding), `baselines.py` (MFU/MRU/HourMFU/Markov-1), `models.py` (GRU-64 and TGT-lite shared backbone, dual head), `train.py` (class weights, window counts), `metrics.py` (Hit@K, MRR, P/R/F1, EventHit@K, Wilson/bootstrap CI)
- `lib/v2/`: per-task models with three branches (LocalEncoder over last 16 in-session events + GlobalEncoder over last 64 cross-session target events + ProfileEncoder over 38-d hand-crafted statistics), combined via 3-way softmax-gated fusion. Profile stats fit on train only via `fit_profile_stats`.
- `lib/v3/`: feature enrichment — `categories.py` (11-class hand-built app taxonomy), `location.py` (parses WiFi SSID / Cell ID from `device_state_update_payload` with train-only vocab), `daypart.py` (10 hour×weekday bins), `window_rollups.py` (causal multi-window aggregates), `markov_prior.py` (V×V log P(next|last) on train), `models_v3.py` (TaskBModelV3 adds frozen Markov-prior buffer with one learnable α scalar). Also `recency.py` + `periodicity.py` for the v4 ablation.

All train-only stats are marked `fit_split="train"` and asserted on load — that's the leakage barrier. Causality is enforced via strict `<` comparison in `searchsorted` for all multi-window aggregates and recency lookups.

The next-app subsystem is **independent** of the bandit/recommendation_agents subsystems above: different data source, different eval protocol, different model. Unrelated to the LinUCB or Dual UCB pipelines.

## Data Generation

There are two separate synthetic data generation systems, one for each subsystem.

### V0 Production Data Generator (`data/generate_bandit_v0_firststep_no_triggers.py`)

This is the primary data generator for the LinUCB recommendation agents. It produces the JSONL files consumed by the `recommendation_agents` training pipeline.

**Inputs required:**
- `--rules`: scenario rules JSON (`data/default_rules_1_english.json`) — defines conditions, preconditions, and categories for each scenario
- `--actions`: action catalog markdown (`data/scenario_recommendation_actions_v4_english.md`) — defines R/O actions, default actions, and app categories per scenario

**Generation mode (current agreed setup):**
- Emits exactly **one row per episode** (first prediction timestep only — no multi-step sequences)
- **Filters out trigger-based scenarios**: any scenario with a non-`none` recommendation trigger is excluded (currently 4 of 65: `OFFICE_AFTERNOON`, `WEEKEND_OVERTIME`, `HOME_EVENING`, `GYM_WORKOUT`), leaving **61 included scenarios**
- Uses **70/30 default-vs-alternative** ground-truth label mix (configurable via `--default-ratio`), allocated with exact shuffled counts per scenario (not purely random)
- Excludes `transportMode`, `cal_nextMinutes`, and history features from the v0 feature set

**How each row is built (`generate_firststep_row`):**
1. **State selection**: picks a `state_current` from the scenario's rule conditions/preconditions (e.g., `office_arriving` for `ARRIVE_OFFICE`). Falls back to hardcoded state lists for scenarios without explicit state conditions (e.g., calendar-based or SMS-based scenarios).
2. **Context inference from state**: location, motion, activity, phone position, and sound are all inferred from the chosen state via heuristic mapping functions (`infer_location_from_state`, `infer_motion_from_state`, etc.). These use prefix matching on the state string (e.g., `home_*` → location `home`, `commuting_walk*` → motion `walking`).
3. **Time selection**: a time window is inferred from the scenario ID and category (`infer_time_window`), then an hour is sampled from the valid range for that window. `timestep` = seconds since midnight.
4. **Contextual features**: battery level, charging status, network type, user profile (age/sex/kids/hash bucket) are sampled with location- and time-aware weighted distributions (e.g., WiFi more likely at home/work, cellular more likely at transit).
5. **Rule overrides** (`apply_rule_overrides`): after the base features are generated, scenario-specific constraints are enforced. This ensures generated data satisfies the scenario's rule conditions (e.g., `OFFICE_LUNCH_OUT` forces `wifiLost=1, wifiLostCategory=work, ps_time=lunch`; `IN_MEETING` forces `cal_inMeeting=1`; SMS-based scenarios force their respective `sms_*_pending=1`).
6. **Label assignment** (`choose_labels`): for each episode, a pre-shuffled boolean flag determines whether the ground-truth R/O action is the scenario default or a random alternative. Same for app category. R/O and app labels are independent.

**Output per row:**
```json
{
  "episode_id": "arrive_office_ep0001",
  "scenario_id": "ARRIVE_OFFICE",
  "scenario_name": "Arrive at Office",
  "t_in_scenario_sec": 0,
  "dt_sec": null,
  "features": { /* 30 v0 feature fields */ },
  "gt_ro": "arrive_office_schedule",
  "gt_app": "productivity"
}
```

**V0 feature fields** (30 total): `scenarioId`, `precondition`, `state_current`, `state_duration_sec`, `ps_time`, `hour`, `cal_hasUpcoming`, `ps_dayType`, `ps_motion`, `wifiLost`, `wifiLostCategory`, `cal_eventCount`, `cal_inMeeting`, `cal_nextLocation`, `ps_sound`, `timestep`, `ps_location`, `ps_phone`, `batteryLevel`, `isCharging`, `networkType`, `activityState`, `activityDuration`, `user_id_hash_bucket`, `age_bucket`, `sex`, `has_kids`, 7x `sms_*_pending` fields.

**Generate command:**
```bash
cd data && python generate_bandit_v0_firststep_no_triggers.py \
  --rules default_rules_1_english.json \
  --actions scenario_recommendation_actions_v4_english.md \
  --output bandit_v0_firststep_no_triggers_1000eps_default_data.jsonl \
  --metadata bandit_v0_firststep_no_triggers_1000eps_default_metadata.json \
  --plan-output scenario_generation_plan_v0_firststep_no_deault_triggers.json \
  --episodes-per-scenario 1000 --default-ratio 0.70 --seed 7
```

### Raw-to-Training Conversion (`recommendation_agents/recommendation_agents/raw_synthetic.py`)

The generated JSONL is not directly consumed by the trainer. It must be converted to training samples:

- `convert_raw_sequence_to_v0`: extracts R/O training rows — keeps rows with a non-NONE `gt_ro`/`gt_ro_action_id`, maps raw feature fields into the V0 context schema, outputs one `{event_id, scenario_id, context, selected_action, reward, propensity}` per row
- `convert_raw_sequence_to_v0_app`: same but for app category labels (`gt_app`/`gt_app_category`)
- Supports both older format (flat fields like `scenarioId`, `gt_ro_action_id`) and newer format (nested `features`, `scenario_id`, `gt_ro`)
- The converter also infers bootstrap metadata (action sets per scenario) but the canonical metadata should come from `build-ro-metadata` / `build-app-metadata` parsed from the catalog markdown

### In-between Scenario Data Generator (`data/in-between_scenario_data_gen/`)

Generates JSONL data for transitional/in-between scenarios (states between primary scenarios, e.g., commuting between home and work). These are separate from the main V0 training data.

- `generate_in_between_scenarios.py` / `generate_new_in_between_scenarios.py`: generate rows using spec files (`*_spec.json`) that define state profiles, context distributions, and time windows for each in-between scenario
- Metadata stored in `metainfo/`
- Output: `in_between_scenarios_1000_each.jsonl` (and `new_` variants)

### Unique Sample Data Generator (`data/bandit_train_data_unique_samples/`)

- `state_verified_generator.py` / `unique_support_to_2k_generator_state_verified_v0.py`: generates training data with verified unique feature combinations per scenario, ensuring diversity in the training set

### Dual UCB Prototype Data Generator (`random_training_features.py`)

Used only by the root-level Dual UCB prototype (`train_runner.py`), not by the LinUCB pipeline.

- `RandomFeatureFactory.build_sample()`: generates a `SyntheticSample` by creating a random `PhysicalState`, running it through `RuleEngine` for scenario matching, then encoding via `DualBanditFeatureEncoderPy` into a 726-dim vector
- 70% of samples use "scenario-shaped" states (hardcoded realistic combinations like morning/work/stationary), 30% fully random
- Builds a random state chain (1-4 entries) and full `ContextSnapshot` with randomized calendar, SMS, battery, network, and activity features
- Target distributions (`ro_target`, `app_target`) come from the rule engine based on the matched scenario
