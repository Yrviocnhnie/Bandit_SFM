# #!/bin/bash
# #SBATCH --job-name=slurm_gpu_test
# #SBATCH --output=we_math_test.out
# #SBATCH --gres=gpu:1
# #SBATCH --partition=LocalQ
# #SBATCH --time=24:00:00

# echo "==== SLURM GPU SANITY CHECK ===="
# echo "Hostname: $(hostname)"
# echo "Job ID: $SLURM_JOB_ID"
# echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
# echo ""

python -m recommendation_agents.cli train-v0-raw-dual \
    --input "docs/bandit_v0_firststep_no_triggers_1000eps_default_data.jsonl" \
    --output artifacts/bandit_v0_firststep_dual \
    --ro-metadata artifacts/ro_metadata.json \
    --app-metadata artifacts/app_metadata.json \
    --alpha 0.15 \
    --alpha-end 0.01 \
    --device cpu \
    --train-window 1000 \
    --eval-window 100 \
    --progress-every 1000 \
    --shuffle-seed 8 \
    --tensorboard-logdir artifacts/bandit_v0_firststep_dual/tensorboard

# echo ""
# echo "Test completed."