#!/usr/bin/env python3
"""
Generate results/1.5B_results.tsv — a clean summary of all Coder-1.5B
training NLL and pass@1 scores.  Run after each eval to keep it current.

Usage:  python scripts/gen_results_table.py
Output: results/1.5B_results.tsv   (also printed to stdout)
"""

import json, os, glob

PROJECT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR  = os.path.join(PROJECT, "results", "eval")
RES_DIR   = os.path.join(PROJECT, "results", "Coder-1.5B")
OUT_PATH  = os.path.join(PROJECT, "results", "1.5B_results.tsv")

BUCKETS   = ["2k_4k", "4k_6k", "6k_8k", "8k_12k", "12k_16k", "16k_20k"]
SIZES     = ["1k", "2k", "4k", "8k"]


def get_train_loss(bucket: str, size: str):
    state = os.path.join(RES_DIR, f"bucket_{bucket}_{size}", "trainer_state.json")
    if not os.path.exists(state):
        return None
    d = json.load(open(state))
    for e in reversed(d.get("log_history", [])):
        if "train_loss" in e:
            return e["train_loss"]
    return None


def get_pass1(bucket: str, size: str):
    path = os.path.join(EVAL_DIR, f"Coder-1.5B_bucket_{bucket}_{size}.json")
    if not os.path.exists(path):
        return None
    m = json.load(open(path))["metrics"]
    # HumanEval stores pass@1 directly; LCB used pass@1_total
    return m.get("pass@1") or m.get("pass@1_total")


header = ["bucket", "size", "train_nll", "humaneval_pass@1"]
rows = []

for bucket in BUCKETS:
    for size in SIZES:
        model_dir = os.path.join(RES_DIR, f"bucket_{bucket}_{size}")
        if not os.path.exists(model_dir):
            continue
        nll = get_train_loss(bucket, size)
        p1  = get_pass1(bucket, size)

        def fmt(v, pct=False):
            if v is None: return "—"
            return f"{v*100:.2f}%" if pct else f"{v:.4f}"

        rows.append([
            bucket, size,
            fmt(nll),
            fmt(p1, pct=True),
        ])

# Write TSV
with open(OUT_PATH, "w") as f:
    f.write("\t".join(header) + "\n")
    for r in rows:
        f.write("\t".join(r) + "\n")

# Print aligned table
col_w = [max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))]
sep   = "  ".join("-" * w for w in col_w)
fmt_row = lambda r: "  ".join(r[i].ljust(col_w[i]) for i in range(len(r)))

print(fmt_row(header))
print(sep)
for r in rows:
    print(fmt_row(r))

print(f"\nWritten to {OUT_PATH}")
