#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-sfm}"

BASE_ARTIFACT="${BASE_ARTIFACT:-artifacts/v6_neural_linear_3p43_stratified_split}"
RERUN_ARTIFACT="${RERUN_ARTIFACT:-artifacts/v6_neural_linear_3p43_stratified_split_rerun}"

echo "== Sanity check: existing artifact =="
python3 - <<'PY' "$BASE_ARTIFACT"
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
train_raw = base / "train.raw.jsonl"
test_raw = base / "test.raw.jsonl"
ro_samples = base / "ro_train_samples_expanded.jsonl"
summary = base / "run_v6_plan_b_summary.json"

def read_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

train_rows = read_jsonl(train_raw)
test_rows = read_jsonl(test_raw)
sample_rows = read_jsonl(ro_samples)

train_ids = set()
for row in train_rows:
    sid = row.get("scenario_id")
    eid = row.get("event_id") or row.get("episode_id")
    train_ids.add((sid, eid))

test_ids = set()
for row in test_rows:
    sid = row.get("scenario_id")
    eid = row.get("event_id") or row.get("episode_id")
    test_ids.add((sid, eid))

sample_base_ids = {row.get("source_base_event_id") for row in sample_rows}

print("artifact:", base)
print("train_raw_rows:", len(train_rows))
print("test_raw_rows:", len(test_rows))
print("ro_train_samples:", len(sample_rows))
print("unique_source_base_event_ids_in_ro_train_samples:", len(sample_base_ids))
print("train/test overlap:", len(train_ids & test_ids))
print("summary_exists:", summary.exists())
PY

echo
echo "== Rerun neural-linear Plan B =="
CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n "$CONDA_ENV" python -m recommendation_agents.cli run-v6-plan-b \
  --input ../data/bandit_v0_v5_1000eps_each_scenario_updated_preconditions_and_updated_state_current.jsonl \
  --catalog-markdown docs/scenario_recommendation_actions_v5.md \
  --output-dir "$RERUN_ARTIFACT" \
  --relevance-markdown docs/scenario_recommendation_actions_v6.md \
  --test-ratio 0.2 \
  --most-relevant-reward 1.0 \
  --plausible-reward 0.0 \
  --irrelevant-reward 0.0 \
  --most-relevant-repeat 1 \
  --plausible-repeat 0 \
  --irrelevant-repeat 0 \
  --other-zero-mode exclude-most-only \
  --alpha 0.0 \
  --default-bonus 0.0 \
  --epochs 1 \
  --top-k 3 \
  --progress-every 10000 \
  --device cuda \
  --model-type neural-linear \
  --no-track-train-hit-rate

echo
echo "== Sanity check: rerun artifact =="
python3 - <<'PY' "$RERUN_ARTIFACT"
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
train_raw = base / "train.raw.jsonl"
test_raw = base / "test.raw.jsonl"
ro_samples = base / "ro_train_samples_expanded.jsonl"
summary = base / "run_v6_plan_b_summary.json"
eval_json = base / "eval_both_top3.json"

def read_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

train_rows = read_jsonl(train_raw)
test_rows = read_jsonl(test_raw)
sample_rows = read_jsonl(ro_samples)
eval_data = json.loads(eval_json.read_text())

train_ids = set()
for row in train_rows:
    sid = row.get("scenario_id")
    eid = row.get("event_id") or row.get("episode_id")
    train_ids.add((sid, eid))

test_ids = set()
for row in test_rows:
    sid = row.get("scenario_id")
    eid = row.get("event_id") or row.get("episode_id")
    test_ids.add((sid, eid))

sample_base_ids = {row.get("source_base_event_id") for row in sample_rows}

print("artifact:", base)
print("train_raw_rows:", len(train_rows))
print("test_raw_rows:", len(test_rows))
print("ro_train_samples:", len(sample_rows))
print("unique_source_base_event_ids_in_ro_train_samples:", len(sample_base_ids))
print("train/test overlap:", len(train_ids & test_ids))
print("summary_exists:", summary.exists())
print("ro avg_most_relevant_covered_in_topk:", eval_data["ro"]["avg_most_relevant_covered_in_topk"])
print("app avg_most_relevant_covered_in_topk:", eval_data["app"]["avg_most_relevant_covered_in_topk"])
PY
