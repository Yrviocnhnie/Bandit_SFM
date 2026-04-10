#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split}"
RELEVANCE_MARKDOWN="${RELEVANCE_MARKDOWN:-docs/scenario_recommendation_actions_v6.md}"
FEEDBACK_SPEC_JSON="${FEEDBACK_SPEC_JSON:-docs/feedback_propagation_policy_v1_cases.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_DIR}/feedback_margin_policy_fixed_radius_v1}"
DEVICE="${DEVICE:-cpu}"
CROSS_SCENARIO_SAMPLE_SIZE="${CROSS_SCENARIO_SAMPLE_SIZE:-2000}"

conda run --no-capture-output -n sfm python -m recommendation_agents.cli simulate-feedback-propagation \
  --artifact-dir "${ARTIFACT_DIR}" \
  --relevance-markdown "${RELEVANCE_MARKDOWN}" \
  --feedback-spec-json "${FEEDBACK_SPEC_JSON}" \
  --propagation-modes hard-assigned-local-cutoff \
  --n-values 2000 \
  --similarity-thresholds 0.80 \
  --feedback-reward-policy margin-aware-fixed-radius-v1 \
  --cross-scenario-sample-size "${CROSS_SCENARIO_SAMPLE_SIZE}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}"
