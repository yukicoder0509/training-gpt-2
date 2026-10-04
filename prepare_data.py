"""Tokenize C4 `en` with the GPT-2 tokenizer into flat uint16 token files.

Writes `train.bin` and `val.bin` under --out_dir. Each document is followed by
an EOS token so documents can be packed into fixed-length training blocks.

Train docs are filtered with the DCLM fastText quality classifier
(`mlfoundations/fasttext-oh-eli5`): every doc in --num_train_shards is scored, then
the highest-scoring docs are kept until --target_train_tokens is reached. Each C4 `en`
train shard is ~150M GPT-2 tokens, so 200 shards -> top ~10% ~= 3B tokens (the DCLM
paper's keep fraction). --target_train_tokens 0 keeps every doc (no filtering).

Train shards are streamed straight from the downloaded .json.gz files (two passes:
score, then tokenize the kept docs) instead of `load_dataset`, whose arrow cache for
200 shards would not fit in the /work quota. The val split is never filtered.
"""

import gzip
import json
import os
from dataclasses import dataclass, field
from multiprocessing import Pool

import fasttext
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class PrepareArguments:
    out_dir: str = field(default=os.path.expandvars("/work/$USER/c4_gpt2"))
    num_train_shards: int = field(default=200, metadata={"help": "Out of 1024 C4 en train shards."})
    target_train_tokens: int = field(
        default=3_000_000_000,
        metadata={"help": "Keep the top-scoring train docs up to this many tokens; 0 keeps every doc."},
    )
    num_val_docs: int = field(
        default=5000,
        metadata={"help": "Docs from the validation split (shuffled with seed 42, like the OJ) used for in-training eval."},
    )
    num_proc: int = field(default=len(os.sched_getaffinity(0)))
    score_dir: str = field(
        default=os.path.expandvars("$HOME/c4_gpt2_dclm/dclm_scores"),
        metadata={"help": "Per-shard DCLM score cache, shared across --out_dir experiments (scoring is the slow pass)."},
    )
    quality_model_cache_dir: str = field(
        default=None, metadata={"help": "Where to cache the 2.3GB fastText model (default: the HF hub cache)."}
    )


# Set in main() before the worker pool forks, so workers share one copy of the 2.3GB model.
tokenizer = None
quality_model = None


def tokenize(texts, batch_size=1000):
    # Batched: a whole shard of token lists as Python ints would take ~6GB per worker.
    out = []
    for i in range(0, len(texts), batch_size):
        ids = tokenizer(texts[i : i + batch_size])["input_ids"]
        out.extend(np.asarray(x + [tokenizer.eos_token_id], dtype=np.uint16) for x in ids)
    return out


def read_shard(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line)["text"] for line in f]


def score_shard(args):
    """Pass 1: DCLM quality score and token count (incl. EOS) of every doc, cached as npz."""
    path, score_dir = args
    if quality_model is None:  # no filtering: only the token counts are needed
        lens = np.array([len(x) for x in tokenize(read_shard(path))], dtype=np.int64)
        return np.zeros(len(lens), dtype=np.float32), lens
    cache = os.path.join(score_dir, os.path.basename(path).replace(".json.gz", ".npz"))
    if not os.path.exists(cache):
        texts = read_shard(path)
        labels, probs = quality_model.predict([t.replace("\n", " ") for t in texts], k=2)
        hq = np.array([dict(zip(l, p)).get("__label__hq", 0.0) for l, p in zip(labels, probs)], dtype=np.float32)
        lens = np.array([len(x) for x in tokenize(texts)], dtype=np.int64)
        np.savez(cache + ".tmp.npz", hq=hq, len=lens)
        os.replace(cache + ".tmp.npz", cache)
    with np.load(cache) as d:
        return d["hq"], d["len"]


def tokenize_kept(args):
    """Pass 2: tokens of the docs selected by `keep`, concatenated."""
    path, keep = args
    texts = read_shard(path)
    ids = tokenize([texts[i] for i in np.flatnonzero(keep)])
    return np.concatenate(ids) if ids else np.zeros(0, dtype=np.uint16)


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
    verify_and_publish(path, written, total)


def verify_and_publish(path, written, total):
    assert written == total, f"wrote {written:,} tokens, expected {total:,}"
    check = np.memmap(path + ".tmp", dtype=np.uint16, mode="r")
    assert len(check) == total and check[-4096:].any(), f"{path}.tmp failed verification"
    os.replace(path + ".tmp", path)  # only a complete file ever appears under the final name
    print(f"wrote {written:,} tokens to {path}")


def write_train_bin(train_files, args, path):
    score_dir = args.score_dir
    os.makedirs(score_dir, exist_ok=True)
    # Resolve (and if needed download) shards here: hf_hub_download inside forked workers deadlocks.
    train_files = [hf_hub_download("allenai/c4", f, repo_type="dataset") for f in train_files]
    with Pool(args.num_proc) as pool:
        scores = []
        for s in pool.imap(score_shard, [(f, score_dir) for f in train_files]):  # in shard order
            scores.append(s)
            print(f"scored shard {len(scores)}/{len(train_files)}", flush=True)
        hq = np.concatenate([s[0] for s in scores])
        lens = np.concatenate([s[1] for s in scores])

        # Lowest score whose top-down cumulative token count reaches the target.
        if args.target_train_tokens and args.target_train_tokens < lens.sum():
            order = np.argsort(-hq, kind="stable")
            cutoff = np.searchsorted(np.cumsum(lens[order]), args.target_train_tokens)
            threshold = hq[order[cutoff]]
        else:
            threshold = -np.inf
        keep = [s[0] >= threshold for s in scores]
        total = int(sum(s[1][k].sum() for s, k in zip(scores, keep)))
        print(
            f"scored {len(hq):,} docs / {lens.sum():,} tokens; hq threshold {threshold:.4f} keeps "
            f"{sum(k.sum() for k in keep):,} docs ({np.mean(hq >= threshold):.1%}) / {total:,} tokens"
        )

        written = 0
        with open(path + ".tmp", "wb") as f:
            for ids in pool.imap(tokenize_kept, zip(train_files, keep)):  # in shard order
                f.write(ids.tobytes())
                written += len(ids)
            f.flush()
            os.fsync(f.fileno())
    verify_and_publish(path, written, total)


def main():
    global tokenizer, quality_model
    (args,) = HfArgumentParser(PrepareArguments).parse_args_into_dataclasses()
    os.makedirs(args.out_dir, exist_ok=True)
    os.chdir(args.out_dir)  # datasets resolves paths against cwd; don't depend on the (NFS) submit dir

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e9)  # silence the >1024 length warning; we pack ourselves
    eos = tokenizer.eos_token_id

    def tokenize_batch(batch):
        ids = [x + [eos] for x in tokenizer(batch["text"])["input_ids"]]
        return {"ids": [np.asarray(x, dtype=np.uint16) for x in ids], "len": [len(x) for x in ids]}

    train_files = [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(args.num_train_shards)]
    val_files = [f"en/c4-validation.{i:05d}-of-00008.json.gz" for i in range(8)]

    val_path, train_path = os.path.join(args.out_dir, "val.bin"), os.path.join(args.out_dir, "train.bin")
    if os.path.exists(val_path) and os.path.exists(train_path):
        print("val.bin and train.bin already exist, nothing to do")
        return

    if not os.path.exists(val_path):
        val = load_dataset("allenai/c4", data_files={"validation": val_files}, split="validation")
        val = val.shuffle(seed=42).select(range(args.num_val_docs))
        val = val.map(tokenize_batch, batched=True, num_proc=args.num_proc, remove_columns=val.column_names)
        write_bin(val, val_path)

    if args.target_train_tokens:
        quality_model = fasttext.load_model(
            hf_hub_download(
                "mlfoundations/fasttext-oh-eli5",
                "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
                cache_dir=args.quality_model_cache_dir,
            )
        )
    write_train_bin(train_files, args, train_path)


if __name__ == "__main__":
    main()
