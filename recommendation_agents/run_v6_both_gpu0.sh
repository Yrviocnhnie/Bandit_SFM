#!/bin/bash

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-sfm}"

PLAN_A_OUTPUT="${PLAN_A_OUTPUT:-artifacts/v6_same_split_expanded}"
PLAN_B_OUTPUT="${PLAN_B_OUTPUT:-artifacts/v6_stratified_split}"

mkdir -p "$PLAN_A_OUTPUT" "$PLAN_B_OUTPUT"

echo "Starting V6 Plan A and Plan B on GPU ${GPU_ID}"
echo "Plan A output: ${PLAN_A_OUTPUT}"
echo "Plan B output: ${PLAN_B_OUTPUT}"
echo ""

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n "$CONDA_ENV" python -m recommendation_agents.cli run-v6-plan-a \
  --input-data-dir artifacts/v5_1000eps_each_scenario_updated \
  --output-dir "$PLAN_A_OUTPUT" \
  --relevance-markdown docs/scenario_recommendation_actions_v6.md \
  --alpha 0.05 \
  --default-bonus 0.0 \
  --top-k 3 \
  --progress-every 10000 \
  --device cuda \
  --no-track-train-hit-rate \
  > "${PLAN_A_OUTPUT}/run.log" 2>&1 &
PLAN_A_PID=$!

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n "$CONDA_ENV" python -m recommendation_agents.cli run-v6-plan-b \
  --input ../data/bandit_v0_v5_1000eps_each_scenario_updated_preconditions_and_updated_state_current.jsonl \
  --catalog-markdown docs/scenario_recommendation_actions_v5.md \
  --output-dir "$PLAN_B_OUTPUT" \
  --relevance-markdown docs/scenario_recommendation_actions_v6.md \
  --test-ratio 0.2 \
  --alpha 0.05 \
  --default-bonus 0.0 \
  --top-k 3 \
  --progress-every 10000 \
  --device cuda \
  --no-track-train-hit-rate \
  > "${PLAN_B_OUTPUT}/run.log" 2>&1 &
PLAN_B_PID=$!

echo "Plan A PID: ${PLAN_A_PID}"
echo "Plan B PID: ${PLAN_B_PID}"
echo ""
echo "Tail logs with:"
echo "  tail -f ${PLAN_A_OUTPUT}/run.log"
echo "  tail -f ${PLAN_B_OUTPUT}/run.log"
echo ""

wait "$PLAN_A_PID"
wait "$PLAN_B_PID"

echo "Both runs finished."
