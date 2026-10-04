from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset
import torch
import numpy as np
import math
from torch.utils.data import Dataset
import os
import argparse

# Check GPU availability
print(torch.__version__, "Cuda:", torch.cuda.is_available(), torch.version.cuda)

# Constants
DATA_DIR = os.path.expandvars("/work/$USER/c4_gpt2")
BLOCK_SIZE = 1024

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Name of the model to be used (also the Hub repo to push to)")
args = parser.parse_args()
MODEL_NAME = args.model_name

print("Model name:", MODEL_NAME)

SECOND_PER_STEP = 1.9  # Estimated
TIME_BUDGET = 1800 - 300 # 30 min (in second) - 300 estimated setup + upload time
MAX_STEPS = int(TIME_BUDGET / SECOND_PER_STEP)

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
    output_dir="./results",
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
    save_steps=100,
    save_total_limit=3,

    # Optimizer
    optim="adamw_torch",
    learning_rate=2.5e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    weight_decay=0.1,  # [choice] GPT-1 used 0.01; 0.1 is the GPT-3/nanoGPT value

    # Scheduler: linear warmup then cosine decay  [gpt1]
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},   # decay to 10% of peak; use "cosine" for ->0
    warmup_steps=100,
)
print("=== Training arguments: ", training_args)

# Trainer
class PPLTrainer(Trainer):
    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            logs["ppl"] = math.exp(logs["loss"])
        if "eval_loss" in logs:
            logs["eval_ppl"] = math.exp(logs["eval_loss"])
        super().log(logs, *args, **kwargs) # Override the log to include perplexity (ppl)

trainer = PPLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

# Train
print("=== Starting training...")
trainer.train()
model.push_to_hub(MODEL_NAME)
print("=== Training finished.")

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