#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-sfm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_JSONL="${INPUT_JSONL:-../data/bandit_v0_v5_1000eps_each_scenario_updated_preconditions_and_updated_state_current.jsonl}"
CATALOG_MARKDOWN="${CATALOG_MARKDOWN:-docs/scenario_recommendation_actions_v5.md}"
RELEVANCE_MARKDOWN="${RELEVANCE_MARKDOWN:-docs/scenario_recommendation_actions_v6.md}"
TOP_K="${TOP_K:-3}"
TEST_RATIO="${TEST_RATIO:-0.2}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10000}"

run_case() {
  local output_dir="$1"
  shift

  echo
  echo "============================================================"
  echo "Running ${output_dir}"
  echo "============================================================"

  conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" -m recommendation_agents.cli run-v6-plan-b \
    --input "${INPUT_JSONL}" \
    --catalog-markdown "${CATALOG_MARKDOWN}" \
    --output-dir "${output_dir}" \
    --relevance-markdown "${RELEVANCE_MARKDOWN}" \
    --test-ratio "${TEST_RATIO}" \
    --alpha 0.0 \
    --default-bonus 0.0 \
    --epochs 1 \
    --top-k "${TOP_K}" \
    --progress-every "${PROGRESS_EVERY}" \
    --device cpu \
    --model-type neural-ucb-lite \
    --no-track-train-hit-rate \
    "$@"
}

# 1. 3 relevant + 43 other
run_case "artifacts/v6_neural_ucb_lite_3p43_stratified_split" \
  --most-relevant-reward 1.0 \
  --plausible-reward 0.0 \
  --irrelevant-reward 0.0 \
  --most-relevant-repeat 1 \
  --plausible-repeat 0 \
  --irrelevant-repeat 0 \
  --other-zero-mode exclude-most-only

# 2. 3 relevant only
run_case "artifacts/v6_neural_ucb_lite_3p_only_stratified_split" \
  --most-relevant-reward 1.0 \
  --plausible-reward 0.0 \
  --irrelevant-reward 0.0 \
  --most-relevant-repeat 1 \
  --plausible-repeat 0 \
  --irrelevant-repeat 0 \
  --other-zero-mode none

# 3. 3 relevant + 3 acceptable + N irrelevant (2+extra)
run_case "artifacts/v6_neural_ucb_lite_3p3a_hardneg_stratified_split" \
  --most-relevant-reward 1.0 \
  --plausible-reward 0.1 \
  --irrelevant-reward -0.1 \
  --most-relevant-repeat 1 \
  --plausible-repeat 1 \
  --irrelevant-repeat 1 \
  --other-zero-mode none

# 4. 3 relevant + 3 acceptable + N irrelevant (2+extra) + other
run_case "artifacts/v6_neural_ucb_lite_3p3a_hardneg_other_stratified_split" \
  --most-relevant-reward 1.0 \
  --plausible-reward 0.1 \
  --irrelevant-reward -0.1 \
  --most-relevant-repeat 1 \
  --plausible-repeat 1 \
  --irrelevant-repeat 1 \
  --other-zero-mode exclude-all-labeled

# 5. 3 relevant + 3 acceptable + other
run_case "artifacts/v6_neural_ucb_lite_3p3a_other_stratified_split" \
  --most-relevant-reward 1.0 \
  --plausible-reward 0.1 \
  --irrelevant-reward 0.0 \
  --most-relevant-repeat 1 \
  --plausible-repeat 1 \
  --irrelevant-repeat 0 \
  --other-zero-mode exclude-most-plausible

echo
echo "All NeuralUCB-lite ablations finished."
