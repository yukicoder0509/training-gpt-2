# Quick Start
1. Install virtual environments
2. Activate virtual environments
3. Login to Hugging Face and WandB
4. Create `.env` file. Put `MODEL_NAME=` variable. This is the Hub repo `push_model.py` pushes to.

# Usage

Run heavy work (data prep, training) through Slurm, not on the login node.

## 1. Prepare data

```
sbatch prepare.sbatch
```

Writes `train.bin` / `val.bin` to `$OUT_DIR` (default `~/c4_gpt2_dclm`): the top ~3B tokens of
200 C4 shards by DCLM fastText quality score. Scores are cached in `~/c4_gpt2_dclm/dclm_scores`, so
a re-cut only re-tokenizes (~10–15 min):

```
OUT_DIR=~/c4_gpt2_dclm_5b sbatch prepare.sbatch --target_train_tokens 5000000000
```

`--target_train_tokens 0` keeps every doc (no filtering).

## 2. Train

```
sbatch --job-name=gpt2-c4-wsd20 run.sbatch --learning_rate=1.25e-3 --decay_frac=0.2 --run_name=wsd20-lr1.25e-3
```

| Flag | Default | Meaning |
|---|---|---|
| `--learning_rate` | `1.25e-3` | Peak LR |
| `--decay_frac` | `0` | `0`: constant after warmup; `>0`: warmup-stable-decay, linear decay to 0 over this fraction of the final steps |
| `--data_dir` | `/work/$USER/c4_gpt2` | Directory with `train.bin` and `val.bin` |
| `--run_name` | none | W&B run name; also names the save directory |
| `--save_dir` | `~/gpt2_models/<run_name or latest>` | Where the final model is saved |

The step count is derived from the 30-min job limit (`TIME_BUDGET` / `SECOND_PER_STEP` in
`train.py`). Warmup is 10% of the steps. If training runs slow, it stops 90 s before the limit and
still saves the model. Losses go to W&B (`cerulean-labs/gpt2-training`) and `logs/<job-name>-<job-id>.out`.

## 3. Push to the Hub

Training only saves the model locally, so the upload doesn't count against the job's time
limit. After the job finishes, push from the login node. Uploading only uses the network,
so it doesn't need Slurm:

```
source .venv/bin/activate && source .env
python push_model.py --model_dir ~/gpt2_models/wsd20-lr1.25e-3 --repo_id $MODEL_NAME --message "WSD20 lr1.25e-3, eval 4.21"
```

The last lines of the training log print this command with the right paths.

| Flag | Default | Meaning |
|---|---|---|
| `--model_dir` | required | Folder written by `train.py` |
| `--repo_id` | `$MODEL_NAME` | Hub repo to push to |
| `--message` | `Upload <folder name>` | Commit message |

It uploads `config.json`, `generation_config.json` and `model.safetensors` as one commit. Each
saved model takes ~500 MB of the 100 G `/home` quota; delete old ones from `~/gpt2_models/` once pushed.

Experiment history and results: [`experiments.md`](experiments.md).
