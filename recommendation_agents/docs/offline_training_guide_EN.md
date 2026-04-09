# Offline Training Guide (English)

This document explains how to start from raw JSONL data and reproduce the current best offline-trained artifact:

- [v6_2k_neural_linear_3p3a_hardneg_other_stratified_split](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split)

The model in that artifact is:

- `NeuralEncoder LinUCB (UCB random init)`
- CLI name: `--model-type neural-linear`

This guide assumes:

- your current working directory is the repository root `Bandit_SFM/`
- the current directory contains:
  - `data/`
  - `recommendation_agents/`

## 1. Final Artifact

After training, you will get an output directory such as:

- `recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split`

Key files inside include:

- `train.raw.jsonl`
- `test.raw.jsonl`
- `ro_train_samples_expanded.jsonl`
- `app_train_samples_expanded.jsonl`
- `ro_model/`
- `app_model/`
- `eval_both_top3.json`
- `train_both_summary.json`
- `v6_training_config.json`

## 2. Required Data

### 2.1 Raw input JSONL

The current best model was trained from:

- `data/bandit_v0_65_scenaroios_unique_support_to_2k_samples_office_working_alias.jsonl`

Each line is a raw context sample and should include at least:

- `scenario_id`
- `episode_id`
- `features`

The raw file may also contain:

- `gt_ro`
- `gt_app`

But in the `run-v6-plan-b` workflow, these fields are **not used directly as training labels**.

Instead, training uses:

- `scenario_id`
- `features`
- the scenario-level relevance definitions from:
  - [scenario_recommendation_actions_v6.md](scenario_recommendation_actions_v6.md)

That means each raw context is expanded into full action-space training samples based on the scenario definition.

### 2.2 Relevance catalog

You also need:

- [scenario_recommendation_actions_v6.md](scenario_recommendation_actions_v6.md)

This file defines, for each scenario:

- `most_relevant`
- `plausible`
- `irrelevant`

During training, each raw context is expanded into:

- R/O: `46` action samples
- App: `10` category samples

### 2.3 Global action catalog

You also need:

- [scenario_recommendation_actions_v5.md](scenario_recommendation_actions_v5.md)

This file is used to build:

- `ro_metadata.json`
- `app_metadata.json`

## 3. Why this run uses the alias file

In the original 65-scenario dataset, one scenario is named:

- `OFFICE_AFTERNOON`

But in the current v6 relevance catalog, the matching scenario is:

- `OFFICE_WORKING`

To keep the training data aligned with the v6 catalog, this run uses an alias file that rewrites:

- `OFFICE_AFTERNOON -> OFFICE_WORKING`

If you only have the original raw file:

- `data/bandit_v0_65_scenaroios_unique_support_to_2k_samples.jsonl`

you can generate the alias file with:

```bash
python - <<'PY'
import json
from pathlib import Path

src = Path('data/bandit_v0_65_scenaroios_unique_support_to_2k_samples.jsonl')
dst = Path('data/bandit_v0_65_scenaroios_unique_support_to_2k_samples_office_working_alias.jsonl')

with src.open() as fin, dst.open('w') as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('scenario_id') == 'OFFICE_AFTERNOON':
            row['scenario_id'] = 'OFFICE_WORKING'
            if row.get('scenario_name') == 'Office afternoon':
                row['scenario_name'] = 'Office working'
        fout.write(json.dumps(row, sort_keys=True))
        fout.write('\n')

print(dst)
PY
```

## 4. Training entrypoint

The recommended entrypoint is:

- `python -m recommendation_agents.cli run-v6-plan-b`

Implementation lives in:

- [cli.py](../recommendation_agents/cli.py)
- [workflows.py](../recommendation_agents/workflows.py)

`run-v6-plan-b` performs the full offline workflow:

1. Read the raw input JSONL
2. Create an `80/20` scenario-stratified split
3. Write:
   - `train.raw.jsonl`
   - `test.raw.jsonl`
4. Build:
   - `ro_metadata.json`
   - `app_metadata.json`
5. Expand training samples using `scenario_recommendation_actions_v6.md`:
   - `ro_train_samples_expanded.jsonl`
   - `app_train_samples_expanded.jsonl`
6. Train:
   - `ro_model/`
   - `app_model/`
7. Evaluate on the test split and write:
   - `eval_both_top3.json`

## 5. Training configuration

The current best model uses:

- model type: `neural-linear`
- split: `Plan B`
- train/test split: `80/20 scenario-stratified`
- reward config:
  - `most_relevant = 1.0`
  - `plausible = 0.1`
  - `irrelevant = -0.1`
  - `other = 0.0`

This configuration is also saved in:

- [v6_training_config.json](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/v6_training_config.json)

`other = 0.0` is implemented with:

- `--other-zero-mode exclude-all-labeled`

Meaning:

- actions labeled as `most_relevant / plausible / irrelevant` keep their assigned rewards
- all remaining unlabeled actions are automatically assigned `0 reward`

So for each raw context:

- R/O expands to `46` training rows
- App expands to `10` training rows

## 6. One-command reproduction

To reproduce this best artifact from raw data:

```bash
conda run --no-capture-output -n sfm python -m recommendation_agents.cli run-v6-plan-b \
  --input data/bandit_v0_65_scenaroios_unique_support_to_2k_samples_office_working_alias.jsonl \
  --catalog-markdown recommendation_agents/docs/scenario_recommendation_actions_v5.md \
  --output-dir recommendation_agents/artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split \
  --relevance-markdown recommendation_agents/docs/scenario_recommendation_actions_v6.md \
  --test-ratio 0.2 \
  --most-relevant-reward 1.0 \
  --plausible-reward 0.1 \
  --irrelevant-reward -0.1 \
  --most-relevant-repeat 1 \
  --plausible-repeat 1 \
  --irrelevant-repeat 1 \
  --other-zero-mode exclude-all-labeled \
  --alpha 0.0 \
  --default-bonus 0.0 \
  --epochs 1 \
  --top-k 3 \
  --progress-every 10000 \
  --device cpu \
  --model-type neural-linear \
  --no-track-train-hit-rate
```

If you want to train on GPU, replace:

- `--device cpu`

with:

- `--device cuda`

## 7. How to verify the result

After training, the main files to inspect are:

- [train_both_summary.json](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/train_both_summary.json)
- [eval_both_top3.json](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/eval_both_top3.json)
- [ro_model/manifest.json](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/ro_model/manifest.json)
- [app_model/manifest.json](../artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split/app_model/manifest.json)

The current best artifact achieved roughly:

- `R/O`: `2.90 / 2.95 / 0.01`
- `App`: `2.92 / 2.96 / 0.01`

## 8. What the key output files mean

### `train.raw.jsonl`

The raw training split. One line per raw context.

### `test.raw.jsonl`

The raw test split. One line per raw context.

### `ro_train_samples_expanded.jsonl`

Expanded training rows for the R/O model.

Meaning:

- each raw training context is duplicated into `46` rows
- each row corresponds to one specific R/O action
- each row carries that action’s reward

So this file is expected to be large.

### `app_train_samples_expanded.jsonl`

Expanded training rows for the App model.

- each raw training context is duplicated into `10` rows
- each row corresponds to one app category

### `ro_model/` and `app_model/`

The trained model artifacts.

For `neural-linear`, this means:

- encoder: `306 -> 128 -> 64`
- plus one disjoint LinUCB head per action

## 9. Related documents

- experiment summary:
  - [report_2026_04_01.md](report_2026_04_01.md)
- scenario-to-action relevance labels:
  - [scenario_recommendation_actions_v6.md](scenario_recommendation_actions_v6.md)
- global action catalog:
  - [scenario_recommendation_actions_v5.md](scenario_recommendation_actions_v5.md)
