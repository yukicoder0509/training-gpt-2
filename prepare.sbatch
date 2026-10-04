#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --job-name=gpt2-c4-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --output=logs/%x-%j.out

# One-off: score 200 C4 shards (~30B tokens) with the DCLM fastText classifier, keep the
# top ~10% (~3B tokens) and tokenize them into $OUT_DIR/{train,val}.bin (no GPU used, but the
# partition requires one). Output and the 2.3GB classifier go to /home: the /work quota is full.

source $HOME/.bashrc
source ${VENV:-$HOME/training-gpt-2/.venv}/bin/activate

export HF_HOME=/work/$USER/hf_cache

OUT_DIR=${OUT_DIR:-$HOME/c4_gpt2_dclm}
python prepare_data.py --out_dir $OUT_DIR --num_train_shards 200 --target_train_tokens 3000000000 \
    --quality_model_cache_dir $HOME/hf_models "$@"
