#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --job-name=gpt2-c4-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out

# One-off: download + tokenize C4 into /work/$USER/c4_gpt2/{train,val}.bin (no GPU used,
# but the partition requires one). ~20 shards -> ~3B tokens, ~6GB on disk + HF cache.

source $HOME/.bashrc
source ${VENV:-$HOME/training-gpt-2/.venv}/bin/activate

export HF_HOME=/work/$USER/hf_cache

python prepare_data.py --out_dir /work/$USER/c4_gpt2 --num_train_shards 20 "$@"
