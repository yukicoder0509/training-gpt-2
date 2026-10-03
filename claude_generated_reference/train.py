"""GPT-2 small pretraining on C4 `en` with Accelerate.

Launch with:  accelerate launch --num_processes 2 --mixed_precision bf16 train.py [--flags]
Every field of TrainArguments is a CLI flag (HfArgumentParser), e.g. --learning_rate 1e-3.
"""

import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, HfArgumentParser, get_scheduler


@dataclass
class TrainArguments:
    # data
    data_dir: str = field(default=os.path.expandvars("/work/$USER/c4_gpt2"))
    block_size: int = 1024

    # model (defaults = GPT-2 small, 124M). HF's GPT2 init is N(0, 0.02) like GPT-1,
    # plus the GPT-2 1/sqrt(2*n_layer) scaling on residual projections.
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # applied to resid/embd/attn dropout; HF default is 0.1
    initializer_range: float = 0.02
    attn_implementation: str = field(default="sdpa", metadata={"help": "sdpa | flash_attention_2 | eager"})

    # optimization (defaults follow GPT-3 small, since the GPT-2 paper omits most of these)
    global_batch_size: int = field(default=512, metadata={"help": "Sequences per optimizer step (GPT-2 paper: 512)."})
    per_device_batch_size: int = 64
    max_steps: int = 1800
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1  # only used by cosine_with_min_lr
    lr_scheduler_type: str = field(
        default="cosine_with_min_lr", metadata={"help": "Any transformers.get_scheduler name."}
    )
    warmup_steps: int = 100
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # runtime
    seed: int = 42
    torch_compile: bool = True
    num_workers: int = 4
    time_budget_minutes: float = field(
        default=25.0, metadata={"help": "Stop training after this much wall-clock time, then eval/save."}
    )

    # logging / eval / saving
    wandb_project: str = "gpt2-training"
    run_name: Optional[str] = None
    logging_steps: int = 10
    eval_steps: int = 250
    eval_batches: int = 20  # per device; -1 = whole val.bin
    output_dir: str = "outputs/gpt2-c4"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "e.g. Cerulean/gpt2-c4-yourname"})


class TokenBlocks(Dataset):
    """Non-overlapping block_size windows over a flat uint16 token file."""

    def __init__(self, path, block_size):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, i):
        x = torch.from_numpy(self.tokens[i * self.block_size : (i + 1) * self.block_size].astype(np.int64))
        return {"input_ids": x, "labels": x}


@torch.no_grad()
def evaluate(model, loader, accelerator, max_batches):
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if max_batches >= 0 and i >= max_batches:
            break
        loss = model(**batch).loss
        losses.append(accelerator.gather_for_metrics(loss.detach().reshape(1)))
    model.train()
    loss = torch.cat(losses).mean().item()
    return {"eval/loss": loss, "eval/perplexity": math.exp(loss)}


def main():
    (args,) = HfArgumentParser(TrainArguments).parse_args_into_dataclasses()
    start_time = time.time()

    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    micro_batches = args.per_device_batch_size * num_processes
    assert args.global_batch_size % micro_batches == 0, "global_batch_size must divide per_device_batch_size * num_gpus"
    grad_accum = args.global_batch_size // micro_batches

    accelerator = Accelerator(gradient_accumulation_steps=grad_accum, log_with="wandb")
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- data ----
    train_ds = TokenBlocks(os.path.join(args.data_dir, "train.bin"), args.block_size)
    val_ds = TokenBlocks(os.path.join(args.data_dir, "val.bin"), args.block_size)
    train_loader = DataLoader(
        train_ds, batch_size=args.per_device_batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.per_device_batch_size, drop_last=True, pin_memory=True)

    # ---- model ----
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = GPT2Config(
        n_positions=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        initializer_range=args.initializer_range,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    config._attn_implementation = args.attn_implementation
    model = GPT2LMHeadModel(config)

    # No weight decay on biases and LayerNorm weights (same rule as HF Trainer).
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon, fused=True,
    )
    scheduler_kwargs = {"min_lr_rate": args.min_lr_ratio} if args.lr_scheduler_type == "cosine_with_min_lr" else {}
    # Not passed through accelerator.prepare: AcceleratedScheduler would step num_processes times per step.
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type, optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps, scheduler_specific_kwargs=scheduler_kwargs,
    )

    if args.torch_compile:
        model = torch.compile(model)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    accelerator.init_trackers(
        args.wandb_project,
        config={**asdict(args), "grad_accum": grad_accum, "num_processes": accelerator.num_processes},
        init_kwargs={"wandb": {"name": args.run_name}} if args.run_name else {},
    )
    tokens_per_step = args.global_batch_size * args.block_size
    accelerator.print(
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  grad_accum={grad_accum}  "
        f"tokens/step={tokens_per_step:,}  train_tokens={len(train_ds) * args.block_size:,}  "
        f"planned_tokens={tokens_per_step * args.max_steps:,}"
    )

    # ---- train ----
    model.train()
    global_step = 0
    loss_sum = torch.zeros((), device=accelerator.device)
    log_t0, log_step0 = time.time(), 0
    done = False
    while not done:
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                loss_sum += loss.detach()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            lr_scheduler.step()
            global_step += 1

            if global_step % args.logging_steps == 0:
                train_loss = accelerator.reduce(loss_sum, reduction="mean").item() / (grad_accum * args.logging_steps)
                dt = time.time() - log_t0
                steps = global_step - log_step0
                accelerator.log(
                    {
                        "train/loss": train_loss,
                        "train/perplexity": math.exp(train_loss),
                        "train/grad_norm": grad_norm.item(),
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/epoch": global_step * args.global_batch_size / len(train_ds),
                        "train/tokens_seen": global_step * tokens_per_step,
                        "perf/tokens_per_sec": steps * tokens_per_step / dt,
                        "perf/step_time_sec": dt / steps,
                        "perf/elapsed_min": (time.time() - start_time) / 60,
                    },
                    step=global_step,
                )
                accelerator.print(
                    f"step {global_step}/{args.max_steps} loss {train_loss:.4f} "
                    f"lr {lr_scheduler.get_last_lr()[0]:.2e} gnorm {grad_norm.item():.3f} "
                    f"tok/s {steps * tokens_per_step / dt:,.0f}"
                )
                loss_sum.zero_()
                log_t0, log_step0 = time.time(), global_step

            if args.eval_steps > 0 and global_step % args.eval_steps == 0 and global_step < args.max_steps:
                metrics = evaluate(model, val_loader, accelerator, args.eval_batches)
                accelerator.log(metrics, step=global_step)
                accelerator.print(f"step {global_step} eval {metrics}")

            # Every rank must agree on stopping, otherwise the next collective hangs.
            out_of_time = torch.tensor(
                float(time.time() - start_time > args.time_budget_minutes * 60), device=accelerator.device
            )
            if global_step >= args.max_steps or accelerator.reduce(out_of_time, reduction="sum").item() > 0:
                if global_step < args.max_steps:
                    accelerator.print(f"time budget hit at step {global_step}, stopping before LR fully decayed")
                done = True
                break

    # ---- final eval + save ----
    metrics = evaluate(model, val_loader, accelerator, -1)
    accelerator.log(metrics, step=global_step)
    accelerator.print(f"final eval {metrics}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        if args.push_to_hub:
            unwrapped.push_to_hub(args.hub_model_id)
            tokenizer.push_to_hub(args.hub_model_id)
    accelerator.end_training()
    accelerator.print(f"total time {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
