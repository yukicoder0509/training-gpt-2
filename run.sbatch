#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --job-name=gpt2-c4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out


source $HOME/.bashrc
source ${VENV:-$HOME/training-gpt-2/.venv}/bin/activate

export HF_HOME=/work/$USER/hf_cache
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Log to the team W&B server (cerulean-labs/gpt2-training) instead of the personal wandb.ai account.
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}
export WANDB_ENTITY=${WANDB_ENTITY:-cerulean-labs}

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)

# Any TrainArguments field in train.py can be passed as --flag. Extra args to sbatch are forwarded,
# e.g. `sbatch run.sbatch --learning_rate 1e-3 --run_name lr1e-3`.
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    train.py \
    --data_dir /work/$USER/c4_gpt2 \
    --output_dir /work/$USER/gpt2_runs/$SLURM_JOB_ID \
    --global_batch_size 512 \
    --per_device_batch_size 64 \
    --max_steps 1800 \
    --warmup_steps 100 \
    --learning_rate 6e-4 \
    --time_budget_minutes 25 \
    "$@"

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"
