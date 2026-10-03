#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --job-name=gpt2-c4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --exclude=25a-hgpn003
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out


source $HOME/.bashrc
source ${VENV:-$HOME/training-gpt-2/.venv}/bin/activate
source $HOME/training-gpt-2/.env

# Log to the team W&B server (cerulean-labs/gpt2-training) instead of the personal wandb.ai account.
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}
export WANDB_ENTITY=${WANDB_ENTITY:-cerulean-labs}
export WANDB_PROJECT=${WANDB_PROJECT:-gpt2-training}  # HF Trainer otherwise defaults to project "huggingface"

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv
echo "========================================"
START_TIME=$(date +%s)

# reference https://huggingface.co/docs/lerobot/multi_gpu_training
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=bf16 \
    train.py --model_name=$MODEL_NAME

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"

# Note:
# --nodelist=25a-hgpn008  # hgpn003 breaks multi-GPU jobs (jobs 493252, 493275, 493290)