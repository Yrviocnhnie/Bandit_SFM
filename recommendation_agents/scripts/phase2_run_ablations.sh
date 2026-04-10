#!/usr/bin/env bash
#
# Phase-2 feedback propagation ablation matrix.
#
# Each ablation writes to artifacts/phase2/ablations/<ablation_name>/.
# Run phase2_aggregate_ablations.py afterwards to produce a comparison markdown.
#
# Usage: from the recommendation_agents/ directory:
#   bash scripts/phase2_run_ablations.sh
#
set -euo pipefail

PYTHON="${PYTHON:-/data00/liyao/anaconda3/envs/sfm/bin/python}"
ARTIFACT_DIR="artifacts/v6_2k_neural_linear_3p3a_hardneg_other_stratified_split"
TRAIN_RAW="artifacts/phase2/train_phase2.raw.jsonl"
TEST_RAW="artifacts/phase2/test_phase2.raw.jsonl"
RELEVANCE="docs/scenario_recommendation_actions_v6.md"
OUT_ROOT="artifacts/phase2/ablations"
LOG_ROOT="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

# Move to recommendation_agents/ root
cd "$(dirname "$0")/.."

run_experiment() {
  local name="$1"
  shift
  local out_dir="${OUT_ROOT}/${name}"
  local log_file="${LOG_ROOT}/${name}.log"
  if [[ -f "${out_dir}/results_phase2.json" ]]; then
    echo "[skip] ${name} (already has results_phase2.json; delete dir to rerun)"
    return
  fi
  mkdir -p "${out_dir}"
  local start=$(date +%s)
  echo "[run ] ${name}"
  "${PYTHON}" -m recommendation_agents.cli simulate-feedback-propagation-phase2 \
    --artifact-dir "${ARTIFACT_DIR}" \
    --train-raw "${TRAIN_RAW}" \
    --test-raw "${TEST_RAW}" \
    --relevance-markdown "${RELEVANCE}" \
    --output-dir "${out_dir}" \
    --device cpu \
    "$@" \
    > "${log_file}" 2>&1 || {
      echo "[FAIL] ${name}: see ${log_file}"
      return 1
    }
  local end=$(date +%s)
  echo "[ok  ] ${name} finished in $((end-start))s"
}

# --- Vanilla baseline ---------------------------------------------------
run_experiment "vanilla_r1_k1"              --n-values "1,5,10,20,50,100"

# --- Group A: reward magnitude scan ------------------------------------
run_experiment "reward_r5_n50"              --n-values "50"  --reward-scale 5
run_experiment "reward_r10_n50"             --n-values "50"  --reward-scale 10
run_experiment "reward_r25_n50"             --n-values "50"  --reward-scale 25
run_experiment "reward_r50_n50"             --n-values "50"  --reward-scale 50
run_experiment "reward_r100_n50"            --n-values "50"  --reward-scale 100

# --- Group B: repeat-k scan -------------------------------------------
run_experiment "repeat_k3_n50"              --n-values "50"  --repeat-k 3
run_experiment "repeat_k5_n50"              --n-values "50"  --repeat-k 5
run_experiment "repeat_k10_n50"             --n-values "50"  --repeat-k 10

# --- Group C: contrastive negative -------------------------------------
run_experiment "contrast_top1_r1_n50"       --n-values "50"  --contrast-mode top1
run_experiment "contrast_top1_r5_n50"      --n-values "50" --contrast-mode top1 --reward-scale 5
run_experiment "contrast_top1_r10_n50"     --n-values "50" --contrast-mode top1 --reward-scale 10
run_experiment "contrast_top3_r5_n50"      --n-values "50" --contrast-mode top3 --reward-scale 5

# --- Group D: larger N values -----------------------------------------
run_experiment "large_n_250"  --n-values "250"
run_experiment "large_n_500"  --n-values "500"
run_experiment "large_n_1000" --n-values "1000"

# --- Group E: serving alpha -------------------------------------------
run_experiment "serving_alpha_0p1" --n-values "50" --serving-alpha 0.1
run_experiment "serving_alpha_0p5" --n-values "50" --serving-alpha 0.5
run_experiment "serving_alpha_1p0" --n-values "50" --serving-alpha 1.0

# --- Group F: anchor-only warmup (no propagation) ---------------------
run_experiment "anchor_only_k1"     --n-values "1" --anchor-only --anchor-repeat-k 1
run_experiment "anchor_only_k10"    --n-values "1" --anchor-only --anchor-repeat-k 10
run_experiment "anchor_only_k50"    --n-values "1" --anchor-only --anchor-repeat-k 50
run_experiment "anchor_only_k100"   --n-values "1" --anchor-only --anchor-repeat-k 100

# --- Group G: combined best (predicted winners) -----------------------
run_experiment "combo_r10_contr_n50"     --n-values "50"  --reward-scale 10 --contrast-mode top1
run_experiment "combo_r10_contr_n100"    --n-values "100" --reward-scale 10 --contrast-mode top1
run_experiment "combo_r25_contr_n100"    --n-values "100" --reward-scale 25 --contrast-mode top1

# --- Group H: target-mode baselines -----------------------------------
run_experiment "baseline_target_model_top1" --n-values "50" --target-mode model_top1
run_experiment "baseline_target_random"     --n-values "50" --target-mode random

echo
echo "All ablations complete."
echo "Output root: ${OUT_BASE:-${OUT_ROOT}}"
