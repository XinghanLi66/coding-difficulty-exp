"""
Prepare codeforces-cots dataset for difficulty-vs-size SFT experiments.

Difficulty proxy: character length of the <think>...</think> block.
  - Fast (no tokenizer needed)
  - Highly correlated with token count (~94% of output is CoT, ~3.5 chars/token)
  - Consistent with the paper's CoT-length proxy

Output format: LLaMA-Factory sharegpt JSONL
  {"conversations": [{"from": "human", "value": <prompt>},
                     {"from": "gpt",   "value": <generation>}]}

Usage:
    python prepare_data.py [--output_dir DATA_DIR] [--pool_size N] [--seed 42]
"""

import argparse
import json
import os
import re
import random
from collections import defaultdict
from datasets import load_dataset

# ── Bucket boundaries (think-block character length) ──────────────────────────
# Derived from sampling 10k examples:
#   completion_tokens: p5=2175, p25=6646, p50=12886, p75=18218, p95=23279
#   CoT ≈ 94% of completion, ~3.5 chars/token  →  think chars ≈ tokens * 3.5
# We target 6 buckets, each with a pool of ≥ POOL_SIZE examples.
# Boundaries are in think-block character counts.
BUCKET_DEFS = [
    ("2k_4k",   7_000,  14_000),   # ~2k–4k tokens
    ("4k_6k",  14_000,  21_000),   # ~4k–6k tokens
    ("6k_8k",  21_000,  28_000),   # ~6k–8k tokens
    ("8k_12k", 28_000,  42_000),   # ~8k–12k tokens
    ("12k_16k",42_000,  56_000),   # ~12k–16k tokens
    ("16k_20k",56_000,  70_000),   # ~16k–20k tokens
]

# Training set sizes to sample per bucket
TRAIN_SIZES = [1_000, 2_000, 4_000, 8_000, 16_000]

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_think_len(generation: str) -> int:
    """Return character length of the <think> block, or 0 if not found."""
    m = THINK_RE.search(generation)
    return len(m.group(1)) if m else 0


def to_sharegpt(prompt: str, generation: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt",   "value": generation},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/newcpfs/lxh/coding-difficulty-exp/data/processed")
    parser.add_argument("--pool_size", type=int, default=20_000,
                        help="Max pool size per bucket (capped by available data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_cache", default=None,
                        help="Optional HuggingFace cache dir")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache

    print("Loading dataset (streaming)…")
    ds = load_dataset(
        "open-r1/codeforces-cots",
        "solutions",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # ── Pass 1: bucket all examples ───────────────────────────────────────────
    buckets: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    n_no_think = 0

    print("Bucketing examples by CoT (think-block) character length…")
    for ex in ds:
        gen = ex.get("generation", "")
        prompt = ex.get("prompt", "")
        if not gen or not prompt:
            continue

        think_len = extract_think_len(gen)
        if think_len == 0:
            n_no_think += 1
            continue

        for name, lo, hi in BUCKET_DEFS:
            if lo <= think_len < hi:
                buckets[name].append({"prompt": prompt, "generation": gen,
                                      "think_len": think_len})
                break

        n_total += 1
        if n_total % 10_000 == 0:
            sizes = {k: len(v) for k, v in buckets.items()}
            print(f"  processed {n_total:,} | buckets: {sizes}")

    print(f"\nDone. Total processed: {n_total:,}  |  no-think skipped: {n_no_think}")
    print("Bucket sizes:")
    for name, lo, hi in BUCKET_DEFS:
        print(f"  [{name}] ({lo//1000}k–{hi//1000}k chars): {len(buckets[name]):,}")

    # ── Pass 2: save pool + sampled subsets ───────────────────────────────────
    stats = {}
    for name, lo, hi in BUCKET_DEFS:
        pool = buckets[name]
        random.shuffle(pool)
        pool = pool[:args.pool_size]   # cap pool size

        bucket_dir = os.path.join(args.output_dir, f"bucket_{name}")
        os.makedirs(bucket_dir, exist_ok=True)

        # Save full pool
        pool_path = os.path.join(bucket_dir, "pool.jsonl")
        with open(pool_path, "w") as f:
            for ex in pool:
                f.write(json.dumps(to_sharegpt(ex["prompt"], ex["generation"])) + "\n")
        print(f"\n[{name}] pool saved: {len(pool):,} examples → {pool_path}")

        # Save subsets
        saved_sizes = []
        for n in TRAIN_SIZES:
            if n > len(pool):
                print(f"  skip train_{n//1000}k (pool only has {len(pool):,})")
                continue
            subset = pool[:n]
            out_path = os.path.join(bucket_dir, f"train_{n//1000}k.jsonl")
            with open(out_path, "w") as f:
                for ex in subset:
                    f.write(json.dumps(to_sharegpt(ex["prompt"], ex["generation"])) + "\n")
            saved_sizes.append(n)

        stats[name] = {"pool": len(pool), "train_sizes": saved_sizes}
        print(f"  subsets saved: {[f'{n//1000}k' for n in saved_sizes]}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "bucket_defs": {n: {"char_lo": lo, "char_hi": hi}
                            for n, lo, hi in BUCKET_DEFS},
            "train_sizes": TRAIN_SIZES,
            "seed": args.seed,
            "buckets": stats,
        }, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
