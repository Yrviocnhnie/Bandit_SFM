#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-sfm}"

cd "$(dirname "$0")"

mkdir -p artifacts/v6_candidate_all_data

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n "$CONDA_ENV" \
  python -m recommendation_agents.cli run-v6-plan-all-data \
  --input ../data/bandit_v0_v5_1000eps_each_scenario_updated_preconditions_and_updated_state_current.jsonl \
  --catalog-markdown docs/scenario_recommendation_actions_v5.md \
  --output-dir artifacts/v6_candidate_all_data \
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
  --no-track-train-hit-rate \
  | tee artifacts/v6_candidate_all_data/run.log
