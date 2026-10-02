"""Tokenize C4 `en` with the GPT-2 tokenizer into flat uint16 token files.

Writes `train.bin` and `val.bin` under --out_dir. Each document is followed by
an EOS token so documents can be packed into fixed-length training blocks.

Each C4 `en` train shard is ~150M GPT-2 tokens, so 20 shards ~= 3B tokens.
"""

import os
from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class PrepareArguments:
    out_dir: str = field(default=os.path.expandvars("/work/$USER/c4_gpt2"))
    num_train_shards: int = field(default=20, metadata={"help": "Out of 1024 C4 en train shards."})
    num_val_docs: int = field(
        default=5000,
        metadata={"help": "Docs from the validation split (shuffled with seed 42, like the OJ) used for in-training eval."},
    )
    num_proc: int = field(default=len(os.sched_getaffinity(0)))


def write_bin(ds, path):
    # Sequential writes, not np.memmap: memmap writes to the /work network FS were silently
    # lost (train.bin came out with its last ~28% all zeros).
    total = int(np.sum(ds["len"], dtype=np.uint64))
    num_chunks = max(1, min(1024, len(ds)))
    written = 0
    with open(path + ".tmp", "wb") as f:
        for i in range(num_chunks):
            chunk = ds.shard(num_shards=num_chunks, index=i, contiguous=True).with_format("numpy")
            if len(chunk):
                ids = np.concatenate(chunk["ids"]).astype(np.uint16)
                f.write(ids.tobytes())
                written += len(ids)
        f.flush()
        os.fsync(f.fileno())
    assert written == total, f"wrote {written:,} tokens, expected {total:,}"
    check = np.memmap(path + ".tmp", dtype=np.uint16, mode="r")
    assert len(check) == total and check[-4096:].any(), f"{path}.tmp failed verification"
    os.replace(path + ".tmp", path)  # only a complete file ever appears under the final name
    print(f"wrote {written:,} tokens to {path}")


def main():
    (args,) = HfArgumentParser(PrepareArguments).parse_args_into_dataclasses()
    os.makedirs(args.out_dir, exist_ok=True)
    os.chdir(args.out_dir)  # datasets resolves paths against cwd; don't depend on the (NFS) submit dir

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e9)  # silence the >1024 length warning; we pack ourselves
    eos = tokenizer.eos_token_id

    def tokenize(batch):
        ids = [x + [eos] for x in tokenizer(batch["text"])["input_ids"]]
        return {"ids": [np.asarray(x, dtype=np.uint16) for x in ids], "len": [len(x) for x in ids]}

    train_files = [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(args.num_train_shards)]
    val_files = [f"en/c4-validation.{i:05d}-of-00008.json.gz" for i in range(8)]

    val_path, train_path = os.path.join(args.out_dir, "val.bin"), os.path.join(args.out_dir, "train.bin")
    if os.path.exists(val_path) and os.path.exists(train_path):
        print("val.bin and train.bin already exist, nothing to do")
        return

    val = load_dataset("allenai/c4", data_files={"validation": val_files}, split="validation")
    val = val.shuffle(seed=42).select(range(args.num_val_docs))
    val = val.map(tokenize, batched=True, num_proc=args.num_proc, remove_columns=val.column_names)
    write_bin(val, val_path)

    train = load_dataset("allenai/c4", data_files={"train": train_files}, split="train", num_proc=args.num_proc)
    train = train.map(tokenize, batched=True, num_proc=args.num_proc, remove_columns=train.column_names)
    write_bin(train, train_path)


if __name__ == "__main__":
    main()
