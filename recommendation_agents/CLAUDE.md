# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two contextual bandit agents for HarmonyOS scenario-based recommendation:

- **R/O agent**: ranks 46 global R/O actions (recommendations + operations)
- **App agent**: ranks 10 app categories

Both use a shared 306-dim feature encoder over 65 scenarios. Supports multiple model architectures: disjoint LinUCB (strongest), shared-linear, neural-linear, and neural-UCB variants.

## Setup

```bash
conda create -n sfm python=3.11 -y && conda activate sfm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

Dependencies: `numpy>=2.0`, `tensorboard`. PyTorch is optional (required only for neural model types).

## Common Commands

All commands run from the `recommendation_agents/` directory.

```bash
# Build metadata from catalog markdown
python3 -m recommendation_agents.cli build-ro-metadata \
  --input-markdown docs/scenario_recommendation_actions_v6.md \
  --output-metadata artifacts/ro_metadata.json
python3 -m recommendation_agents.cli build-app-metadata \
  --input-markdown docs/scenario_recommendation_actions_v6.md \
  --output-metadata artifacts/app_metadata.json

# V6 training (Plan B — standard end-to-end: raw JSONL → stratified split → V6 expansion → train)
python3 -m recommendation_agents.cli run-v6-plan-b \
  --input <raw.jsonl> --output artifacts/<output_dir> \
  --relevance-markdown docs/scenario_recommendation_actions_v6.md \
  --catalog-markdown docs/scenario_recommendation_actions_v5.md \
  --ro-metadata artifacts/ro_metadata.json --app-metadata artifacts/app_metadata.json \
  --alpha 0.05 --default-bonus 0.0 --top-k 3 --device cpu

# Legacy V0 dual training (simple 1:1 labels)
python3 -m recommendation_agents.cli train-v0-raw-dual \
  --input <raw.jsonl> --output artifacts/<output_dir> \
  --ro-metadata artifacts/ro_metadata.json --app-metadata artifacts/app_metadata.json \
  --device cpu

# Inference
python3 -m recommendation_agents.cli score-v0 --artifact <model_dir> --metadata <meta.json> --sample <request.json> --top-k 3 --device auto
python3 -m recommendation_agents.cli choose-v0 --artifact <model_dir> --metadata <meta.json> --sample <request.json> --device auto

# Evaluation
python3 -m recommendation_agents.cli eval-v0-both --output <artifact_dir> --top-k 3
python3 -m recommendation_agents.cli eval-soft-scenarios-both --output <artifact_dir> ...

# Phase-1 online feedback propagation (locks targets from the frozen model's top-k)
python3 -m recommendation_agents.cli simulate-feedback-propagation \
  --artifact-dir <artifact_dir> \
  --relevance-markdown docs/scenario_recommendation_actions_v6.md \
  --n-values 1,5,10,20,50,100 --device cpu

# Phase-2: build the profile-normalized train/test split (9 context variations)
python scripts/phase2_split.py --samples-per-context 3

# Phase-2: single feedback-propagation run (uses phase2_gt_ro as the liked target)
python3 -m recommendation_agents.cli simulate-feedback-propagation-phase2 \
  --artifact-dir artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split \
  --train-raw artifacts/phase2/train_phase2.raw.jsonl \
  --test-raw artifacts/phase2/test_phase2.raw.jsonl \
  --n-values 1,5,10,20,50,100 --reward-scale 100 --device cpu

# Phase-2: full ablation matrix (28 experiments, ~35 min on CPU) + comparison markdown
bash scripts/phase2_run_ablations.sh
python scripts/phase2_aggregate_ablations.py
```

See `run.sh` for V0 dual training and `run_v6_*.sh` for V6 training variants (GPU/CPU, ablations).

## Tests

```bash
python -m pytest tests/                          # all tests
python -m pytest tests/test_catalog.py            # single file
python -m pytest tests/test_catalog.py -k "name"  # single test by name
```

Some tests in `test_workflows.py` and `test_training_pipeline.py` require PyTorch.

## Architecture

### Data Flow (V6 Pipeline)

```
Raw JSONL → raw_synthetic.py (V6 expansion) → JSONL samples
                                                    ↓
                                              trainer.py (LinUCB/neural replay)
                                                    ↓
                                              Model artifacts (manifest.json, weights.npz, training_report.json)
```

Orchestrated by `workflows.py` via CLI subcommands.

### V6 Expansion

One raw context row → multiple labeled training samples per the V6 relevance catalog (`scenario_recommendation_actions_v6.md`):
- **most_relevant** (3 actions, reward=1.0) — correct actions for this scenario
- **plausible** (3 actions, reward=0.6) — acceptable but suboptimal
- **irrelevant** (2 actions, reward=0.0) — wrong actions

Rewards and repeat counts are configurable per tier. Samples share a `source_base_event_id` for grouped dense supervision in neural pretraining.

### Key Modules

- **`cli.py`**: argparse-based CLI with ~15 subcommands covering metadata build, data conversion, training, evaluation, and experimental workflows
- **`feature_space.py`**: 306-dim feature encoder — 33 context fields (categorical one-hot, binary, log-scaled numeric) → float32 vector. No scenario ID in the feature vector (shared-action design).
- **`linucb.py`**: 7 bandit model variants — `disjoint` (per-action linear UCB heads), `shared-linear`, `neural-linear` (MLP encoder 306→128→64 + linear heads), `neural-scorer`, `neural-ucb-lite`, `neural-ucb`, `neural-ucb-direct`. All implement `rank(context)`, `update(context, action, reward)`, `save()`/`load()`.
- **`trainer.py`**: offline replay training loop. Supports neural pretraining (encoder pre-training via MSE on grouped supervision), alpha annealing (`--alpha`/`--alpha-end`), train/eval windowing, TensorBoard logging.
- **`workflows.py`**: high-level pipelines — `run_v6_plan_a()` (reuse existing split), `run_v6_plan_b()` (raw → stratified split → expand → train), `run_v6_plan_all_data()`, `evaluate_v0_topk()`, `eval_soft_scenarios_both()`, `mine_v6_hard_negative_candidates()`, `simulate_feedback_propagation_on_frozen_neural_linear()` (phase-1 online learning), `simulate_feedback_propagation_phase2_on_frozen_neural_linear()` (phase-2 online learning with pre-labeled GTs and ablation knobs).
- **`raw_synthetic.py`**: converters from raw JSONL to training samples. V0 converters do 1:1 mapping; V6 converters expand using relevance tiers.
- **`v6_relevance.py`**: parses V6 relevance markdown (3+3+2 structure per scenario for R/O and app)
- **`in_between_relevance.py`**: parses in-between scenario relevance (evaluation-only, ro_top10 + app_top5)
- **`catalog.py`**: parses catalog markdown → metadata JSON (action spaces, defaults per scenario)
- **`metadata.py`**: `BanditMetadata` / `ScenarioMetadata` — runtime metadata for action resolution, arm mapping, scenario masking
- **`schemas.py`**: `TrainingEvent`, `ScoreRequest` data contracts
- **`taxonomies.py`**: canonical scenario IDs, state codes, app categories

### Key Design Points

- **Shared global action space**: 46 R/O + 10 app actions ranked for all 65 scenarios (no per-scenario masking at inference in V6)
- **V6 expansion**: one raw context → multiple samples per relevance tier (most_relevant/plausible/irrelevant) with configurable rewards and repeats
- **Dual-head training**: R/O and App agents train independently on the same context encoding
- **Default bonus**: additive prior for scenario's default action (configurable, often 0.0 in V6)
- **Neural pretraining**: for neural-linear models, encoder is pretrained via MSE on grouped dense supervision before bandit training
- **Phase-1 online learning** (experimental): freeze encoder, update only linear heads via `simulate-feedback-propagation` on locked model-top-1/top-2 targets (4 hardcoded office scenarios, mixed like/dislike feedback)
- **Phase-2 online learning** (experimental): same mechanism but with pre-labeled `phase2_gt_ro` targets from `test_phase2.raw.jsonl`, always `like` (+reward). Supports 8 ablation knobs (`--reward-scale`, `--repeat-k`, `--contrast-mode`, `--contrast-reward-scale`, `--serving-alpha`, `--anchor-only`, `--anchor-repeat-k`, `--target-mode`) via the `simulate-feedback-propagation-phase2` CLI. Use `bash scripts/phase2_run_ablations.sh` to sweep the ablation matrix

### V6 Training Workflows

- **Plan A** (`run-v6-plan-a`): reuse existing train/test split, apply V6 expansion
- **Plan B** (`run-v6-plan-b`): raw JSONL → 80/20 stratified split → V6 expansion → train (standard flow)
- **Plan All Data** (`run-v6-plan-all-data`): train on all data, no holdout (upper-bound check)

### Phase-2 Online Learning

Phase-2 tests whether feedback propagation can teach the frozen neural-linear model **context-specific recommendations** that the V6 training labels alone can't convey (e.g., `ARRIVE_OFFICE` with `cal_eventCount=3` → `O_SHOW_SCHEDULE`, but the same scenario with `cal_eventCount=0` → `O_SHOW_TODAY_TODO`). The target actions are pre-labeled in the test set instead of being locked from the frozen model's top-k (as in phase-1).

Inputs:
- `artifacts/phase2/train_phase2.raw.jsonl` — propagation pool (profile-normalized via `phase2_split.py`)
- `artifacts/phase2/test_phase2.raw.jsonl` — 27 tagged test rows (9 contexts × 3 samples; each row carries `phase2_context_id`, `phase2_gt_ro`, `phase2_label`)
- `docs/phase2_multi_recommendation_scenarios.md` — canonical definition of the 9 context variations
- Base model: a frozen neural-linear artifact (e.g., `artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/`)

Workflow function: `simulate_feedback_propagation_phase2_on_frozen_neural_linear` in `workflows.py`. Same 4 propagation modes as phase-1 (`single`, `global-latent-nearest-N`, `same-scenario-nearest-N`, `entire-scenario-all`), but the target action is the pre-labeled `phase2_gt_ro` (always `like`, reward = `+1.0 × --reward-scale`).

CLI flags on `simulate-feedback-propagation-phase2`:
- `--reward-scale` (default 1.0): linear multiplier on the positive reward
- `--repeat-k` (default 1): partial_fit repetitions per propagation row
- `--contrast-mode` (`none`/`top1`/`top3`): apply contrastive negative to current top competitors
- `--contrast-reward-scale` (default 1.0): magnitude multiplier for the negative update
- `--serving-alpha`: override `model.alpha` at ranking time (UCB exploration bonus)
- `--anchor-only` + `--anchor-repeat-k`: apply feedback only to the anchor, skip propagation
- `--target-mode` (`phase2_gt`/`model_top1`/`random`): baseline comparisons

Scripts (in `recommendation_agents/scripts/`):
- `phase2_split.py` — builds the phase-2 train/test split from the original unique-samples JSONL, filtering/patching rows to match each context definition and normalizing to a single user profile
- `phase2_run_ablations.sh` — runs the 28-experiment ablation matrix; each run writes to `artifacts/phase2/ablations/<name>/`
- `phase2_aggregate_ablations.py` — produces `artifacts/phase2/ablations/ablations_comparison.md` with global aggregates, per-context rank-after tables, and success-criteria pass/fail

### Model Artifacts

Each trained model produces:
- `manifest.json`: model type, hyperparameters, action list
- `weights.npz`: model parameters
- `training_report.json`: training metrics
- `metadata.snapshot.json`: copy of metadata used
- Optional: `encoder_state.pt` (neural models)
