from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset
import torch
import numpy as np
import math
from torch.utils.data import Dataset
import os
import argparse
import time
from adam_mini import Adam_mini

START_TIME = time.time()

# Check GPU availability
print(torch.__version__, "Cuda:", torch.cuda.is_available(), torch.version.cuda)

# Constants
BLOCK_SIZE = 1024

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Hub repo the saved model is meant for (pushed separately with push_model.py)")
parser.add_argument("--learning_rate", type=float, default=1.25e-3, help="Peak LR, held constant after warmup")
parser.add_argument("--decay_frac", type=float, default=0.0, help="0: constant after warmup; >0: warmup-stable-decay, linear decay to 0 over this fraction of the final steps")
parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay (GPT-1: 0.01, GPT-3/nanoGPT: 0.1)")
parser.add_argument("--optimizer", type=str, default="adam_mini", choices=["adam_mini", "adamw"], help="adam_mini (Adam-mini, pip adam-mini) or adamw (torch AdamW)")
parser.add_argument("--data_dir", type=str, default=os.path.expandvars("/work/$USER/c4_gpt2"), help="Dir with train.bin and val.bin")
parser.add_argument("--run_name", type=str, default=None, help="W&B run name; also isolates checkpoints in results/<run_name>")
parser.add_argument("--save_dir", type=str, default=None, help="Where to save the final model (default: ~/gpt2_models/<run_name or latest>)")
args = parser.parse_args()
MODEL_NAME = args.model_name
DATA_DIR = os.path.expanduser(args.data_dir)
SAVE_DIR = os.path.expanduser(args.save_dir or os.path.join("~/gpt2_models", args.run_name or "latest"))

print("Model name:", MODEL_NAME, "LR:", args.learning_rate, "Optimizer:", args.optimizer, "WD:", args.weight_decay, "Run name:", args.run_name, "Data dir:", DATA_DIR)

# The Hub upload runs after the job (push_model.py), so the 30-min limit only covers startup,
# training and a local save. Measured over jobs 496865-496949: ~40 s startup, 2.13 s/step incl.
# eval every 20 steps (631 steps -> 1345 s), local save + W&B finish < 30 s.
JOB_TIME_LIMIT = 1800
SECOND_PER_STEP = 2.13
TIME_BUDGET = JOB_TIME_LIMIT - 60 - 30 - 120  # startup, save + W&B finish, safety margin
MAX_STEPS = int(TIME_BUDGET / SECOND_PER_STEP)
WARMUP_STEPS = int(0.1 * MAX_STEPS)
DECAY_STEPS = int(args.decay_frac * MAX_STEPS)

# Prepare tokenizer and model
print("=== Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

config = AutoConfig.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_config(config)

# Load English C4 dataset
print("=== Loading English C4 dataset...")
class TokenBlocks(Dataset):
    def __init__(self, path, block_size):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")  # lazy, no RAM blowup
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, i):
        x = torch.from_numpy(self.tokens[i * self.block_size : (i + 1) * self.block_size].astype(np.int64))
        return {"input_ids": x, "labels": x}  # model shifts labels internally)

train_ds = TokenBlocks(os.path.join(DATA_DIR, "train.bin"), BLOCK_SIZE)
val_ds = TokenBlocks(os.path.join(DATA_DIR, "val.bin"), BLOCK_SIZE)

# Training arg
training_args = TrainingArguments(
    output_dir=os.path.join("./results", args.run_name or ""),
    run_name=args.run_name,
    # Global batch = per_device * grad_accum * num_gpus = 32 * 8 * 2 = 512 sequences  [gpt1]
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
    bf16=True,
    num_train_epochs=1,
    max_steps=MAX_STEPS,

    # Logging, eval and reporting
    logging_steps=10,
    report_to="wandb",  # Log to W&B
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="no",

    # Optimizer
    optim="adamw_torch",
    learning_rate=args.learning_rate,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    weight_decay=args.weight_decay,  # [choice] GPT-1 used 0.01; 0.1 is the GPT-3/nanoGPT value

    # Scheduler: linear warmup then constant at peak (get_constant_schedule_with_warmup),
    # or with --decay_frac, warmup-stable-decay (get_wsd_schedule) with a final linear decay to 0.
    # Was: cosine to 10% of peak ("cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": 0.1})
    lr_scheduler_type="warmup_stable_decay" if DECAY_STEPS else "constant_with_warmup",
    lr_scheduler_kwargs={"num_decay_steps": DECAY_STEPS, "decay_type": "linear"} if DECAY_STEPS else {},
    warmup_steps=WARMUP_STEPS,  # ~10% of the run
)
print("=== Training arguments: ", training_args)

# Safety net: if training runs slower than estimated (e.g. a slow node), stop early so the model
# is still saved before Slurm kills the job. The LR decay is cut short in that case.
class DeadlineCallback(TrainerCallback):
    def __init__(self, deadline):
        self.deadline = deadline

    def on_step_end(self, args, state, control, **kwargs):
        stop = torch.tensor(float(time.time() > self.deadline), device=args.device)
        if torch.distributed.is_initialized():  # all ranks must stop at the same step or the next collective hangs
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
        if stop.item():
            print(f"=== Deadline reached at step {state.global_step}/{state.max_steps}, stopping to save the model")
            control.should_training_stop = True
        return control

# Optimizer. Adam-mini groups params by name: one lr per row for embeddings/MLP, one per tensor for the
# rest. HF GPT-2 names: wte (matched), wpe (added below), attn.c_attn (fused QKV) and attn.c_proj are not
# matched, so they get one lr per tensor; mlp.c_fc/c_proj are matched but are Conv1D (in, out), so
# "per row" is per input feature rather than per output neuron.
optimizer = None
if args.optimizer == "adam_mini":
    optimizer = Adam_mini(
        named_parameters=model.named_parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,  # Adam-mini skips decay on norm and bias params itself
        dim=config.n_embd,
        n_heads=config.n_head,
        verbose=int(os.environ.get("LOCAL_RANK", 0)) == 0,
    )
    optimizer.embd_names.add("wpe")

# Trainer
class CustomTrainer(Trainer):
    # Log PPL
    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            logs["ppl"] = math.exp(logs["loss"])
        if "eval_loss" in logs:
            logs["eval_ppl"] = math.exp(logs["eval_loss"])
        super().log(logs, *args, **kwargs) # Override the log to include perplexity (ppl)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[DeadlineCallback(START_TIME + JOB_TIME_LIMIT - 90)],
    optimizers=(optimizer, None),  # None -> Trainer builds AdamW from args; scheduler always from args
)

# Train
print("=== Starting training...")
trainer.train()
trainer.save_model(SAVE_DIR)  # main process only; config + generation_config + safetensors
print(f"=== Training finished. Saved to {SAVE_DIR} at {time.time() - START_TIME:.0f}s. Push with:")
print(f"    python push_model.py --model_dir {SAVE_DIR} --repo_id {MODEL_NAME}")

# OpenAI team hyperparameters
"""
Batch size: 512 sequences.
Adam with a max learning rate of 2.5e-4
Linear warmup over 2,000 updates, then cosine annealing
Dropout of 0.1
Modified L2 weight decay of 0.01

If you're trying to reproduce GPT-2 training,
 community reproductions like Karpathy's nanoGPT and llm.c 
 are the practical reference. 
 They fill in the missing values, 
 for example a peak learning rate around 6e-4 for 124M and AdamW with betas (0.9, 0.95). Those values are their choices, not OpenAI's.
"""