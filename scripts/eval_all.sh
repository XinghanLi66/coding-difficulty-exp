#!/bin/bash
# Evaluate all trained models and collect results into a summary CSV.
# Usage: bash eval_all.sh [model_tag_filter] [gpu_ids]
# Example: bash eval_all.sh Coder-1.5B 0,1,2,3

set -e

FILTER="${1:-}"
GPU_IDS="${2:-0,1,2,3}"
PROJECT_DIR="/newcpfs/lxh/coding-difficulty-exp"

for model_dir in "$PROJECT_DIR"/results/*/; do
    name=$(basename "$model_dir")
    [ "$name" = "eval" ] && continue
    [ "$name" = "smoke_test" ] && continue

    if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then
        continue
    fi

    result_file="$PROJECT_DIR/results/eval/${name}.json"
    if [ -f "$result_file" ]; then
        echo "SKIP (exists): $name"
        continue
    fi

    if [ ! -f "$model_dir/config.json" ]; then
        echo "SKIP (no model): $name"
        continue
    fi

    echo ">>> Evaluating: $name"
    bash "$PROJECT_DIR/scripts/eval_one.sh" "$model_dir" "$GPU_IDS"
done

# Aggregate results into summary CSV
PYTHON=/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python
$PYTHON - << 'EOF'
import json, glob, csv, os, re

project = "/newcpfs/lxh/coding-difficulty-exp"
rows = []

for f in sorted(glob.glob(f"{project}/results/eval/*.json")):
    with open(f) as fh:
        d = json.load(fh)
    name = os.path.basename(f).replace(".json", "")
    m = re.match(r"(Coder-\d+\.?\d*B)_bucket_(\w+)_(\d+k)", name)
    if not m:
        continue
    model, bucket, size = m.groups()
    metrics = d.get("metrics", {})
    rows.append({
        "model": model, "bucket": bucket, "size": size,
        "pass@1_easy":   metrics.get("pass@1_easy",   ""),
        "pass@1_medium": metrics.get("pass@1_medium", ""),
        "pass@1_hard":   metrics.get("pass@1_hard",   ""),
        "pass@1_total":  metrics.get("pass@1_total",  ""),
    })

out = f"{project}/results/eval/summary.csv"
if rows:
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written: {out}  ({len(rows)} rows)")
else:
    print("No eval results found yet.")
EOF
