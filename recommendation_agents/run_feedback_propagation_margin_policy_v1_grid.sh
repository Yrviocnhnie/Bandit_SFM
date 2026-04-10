#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split}"
RELEVANCE_MARKDOWN="${RELEVANCE_MARKDOWN:-docs/scenario_recommendation_actions_v6.md}"
FEEDBACK_SPEC_JSON="${FEEDBACK_SPEC_JSON:-docs/feedback_propagation_policy_v1_cases.json}"
DEVICE="${DEVICE:-cpu}"
PROPAGATION_MODES="${PROPAGATION_MODES:-hard-assigned-local-cutoff}"
N_VALUES="${N_VALUES:-100,500,1000,2000,4000}"
SIMILARITY_THRESHOLDS="${SIMILARITY_THRESHOLDS:-0.70,0.80,0.85,0.90,0.95}"
CROSS_SCENARIO_SAMPLE_SIZE="${CROSS_SCENARIO_SAMPLE_SIZE:-2000}"
BASE_LIKE_REWARD="${BASE_LIKE_REWARD:-2.0}"
BASE_DISLIKE_REWARD="${BASE_DISLIKE_REWARD:--1.0}"

LIKE_REWARDS=(
  "1.0"
  "2.0"
  "5.0"
  "10.0"
  "20.0"
)

DISLIKE_REWARDS=(
  "-1.0"
  "-2.0"
  "-5.0"
  "-10.0"
  "-20.0"
  "-40.0"
)

CASE_IDS=(
  "arrive_office_like_rank1"
  "office_lunch_out_dislike_rank1"
  "office_working_like_rank2"
  "leave_office_dislike_rank2"
  "cafe_stay_like_rank3"
  "office_to_cafe_dislike_rank3"
)

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

for CASE_ID in "${CASE_IDS[@]}"; do
  CASE_SPEC_JSON="${TMP_DIR}/${CASE_ID}.json"
  python - <<'PY' "${FEEDBACK_SPEC_JSON}" "${CASE_ID}" "${CASE_SPEC_JSON}"
import json
import sys
from pathlib import Path

spec_path = Path(sys.argv[1])
feedback_id = sys.argv[2]
output_path = Path(sys.argv[3])
items = json.loads(spec_path.read_text())
selected = [item for item in items if item.get("feedback_id") == feedback_id]
if len(selected) != 1:
    raise SystemExit(f"Expected exactly one feedback item for {feedback_id!r}, found {len(selected)}")
output_path.write_text(json.dumps(selected, indent=2))
PY

  if [[ "${CASE_ID}" == *"_like_"* ]]; then
    for LIKE_REWARD in "${LIKE_REWARDS[@]}"; do
      RUN_START_SECONDS=${SECONDS}
      SAFE_REWARD="${LIKE_REWARD/./p}"
      OUTPUT_DIR="${ARTIFACT_DIR}/feedback_policy_v1/${CASE_ID}/like${SAFE_REWARD}"

      echo "============================================================"
      echo "Running margin-policy v1 grid point"
      echo "  feedback_id=${CASE_ID}"
      echo "  like_reward=${LIKE_REWARD}"
      echo "  dislike_reward=${BASE_DISLIKE_REWARD}"
      echo "  modes=${PROPAGATION_MODES}"
      echo "  n_values=${N_VALUES}"
      echo "  similarity_thresholds=${SIMILARITY_THRESHOLDS}"
      echo "  cross_scenario_sample_size=${CROSS_SCENARIO_SAMPLE_SIZE}"
      echo "  output_dir=${OUTPUT_DIR}"
      echo "============================================================"

      conda run --no-capture-output -n sfm python -m recommendation_agents.cli simulate-feedback-propagation \
        --artifact-dir "${ARTIFACT_DIR}" \
        --relevance-markdown "${RELEVANCE_MARKDOWN}" \
        --feedback-spec-json "${CASE_SPEC_JSON}" \
        --propagation-modes "${PROPAGATION_MODES}" \
        --n-values "${N_VALUES}" \
        --similarity-thresholds "${SIMILARITY_THRESHOLDS}" \
        --like-reward "${LIKE_REWARD}" \
        --dislike-reward "${BASE_DISLIKE_REWARD}" \
        --cross-scenario-sample-size "${CROSS_SCENARIO_SAMPLE_SIZE}" \
        --output-dir "${OUTPUT_DIR}" \
        --device "${DEVICE}"

      RUN_ELAPSED=$((SECONDS - RUN_START_SECONDS))
      printf 'Finished feedback_id=%s like_reward=%s | elapsed=%02dh:%02dm:%02ds\n' \
        "${CASE_ID}" \
        "${LIKE_REWARD}" \
        "$((RUN_ELAPSED/3600))" \
        "$(((RUN_ELAPSED%3600)/60))" \
        "$((RUN_ELAPSED%60))"
    done
  else
    for DISLIKE_REWARD in "${DISLIKE_REWARDS[@]}"; do
      RUN_START_SECONDS=${SECONDS}
      SAFE_REWARD="${DISLIKE_REWARD#-}"
      SAFE_REWARD="${SAFE_REWARD/./p}"
      OUTPUT_DIR="${ARTIFACT_DIR}/feedback_policy_v1/${CASE_ID}/dislike${SAFE_REWARD}"

      echo "============================================================"
      echo "Running margin-policy v1 grid point"
      echo "  feedback_id=${CASE_ID}"
      echo "  like_reward=${BASE_LIKE_REWARD}"
      echo "  dislike_reward=${DISLIKE_REWARD}"
      echo "  modes=${PROPAGATION_MODES}"
      echo "  n_values=${N_VALUES}"
      echo "  similarity_thresholds=${SIMILARITY_THRESHOLDS}"
      echo "  cross_scenario_sample_size=${CROSS_SCENARIO_SAMPLE_SIZE}"
      echo "  output_dir=${OUTPUT_DIR}"
      echo "============================================================"

      conda run --no-capture-output -n sfm python -m recommendation_agents.cli simulate-feedback-propagation \
        --artifact-dir "${ARTIFACT_DIR}" \
        --relevance-markdown "${RELEVANCE_MARKDOWN}" \
        --feedback-spec-json "${CASE_SPEC_JSON}" \
        --propagation-modes "${PROPAGATION_MODES}" \
        --n-values "${N_VALUES}" \
        --similarity-thresholds "${SIMILARITY_THRESHOLDS}" \
        --like-reward "${BASE_LIKE_REWARD}" \
        --dislike-reward "${DISLIKE_REWARD}" \
        --cross-scenario-sample-size "${CROSS_SCENARIO_SAMPLE_SIZE}" \
        --output-dir "${OUTPUT_DIR}" \
        --device "${DEVICE}"

      RUN_ELAPSED=$((SECONDS - RUN_START_SECONDS))
      printf 'Finished feedback_id=%s dislike_reward=%s | elapsed=%02dh:%02dm:%02ds\n' \
        "${CASE_ID}" \
        "${DISLIKE_REWARD}" \
        "$((RUN_ELAPSED/3600))" \
        "$(((RUN_ELAPSED%3600)/60))" \
        "$((RUN_ELAPSED%60))"
    done
  fi
done

echo "Finished all margin-policy-v1 grid runs."
