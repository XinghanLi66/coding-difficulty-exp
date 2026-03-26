#!/bin/bash
# Priority pipeline: finish 1.5B training → eval 1.5B pass@1 → train 7B
# Usage: bash run_priority.sh
# Logs: logs/run_priority.log

set -uo pipefail

PROJECT_DIR="/newcpfs/lxh/coding-difficulty-exp"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
LOGS_DIR="$PROJECT_DIR/logs"
PYTHON=/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─── Phase 1: Finish all remaining 1.5B training ────────────────────────────
log "=========================================="
log "PHASE 1: Training — Coder-1.5B (remaining)"
log "=========================================="
bash "$SCRIPTS_DIR/train_all.sh" Coder-1.5B

log ""
log "=========================================="
log "PHASE 2: Eval — Coder-1.5B pass@1"
log "=========================================="

# Use all 8 GPUs split 4+4 would allow parallel eval, but vLLM with tp=8 is
# simpler and faster for generation. Run sequentially with all 8 GPUs per job.
ALL_GPUS="0,1,2,3,4,5,6,7"
eval_count=0
skip_count=0

for model_dir in "$PROJECT_DIR"/results/Coder-1.5B/*/; do
    [ -f "$model_dir/config.json" ] || continue  # skip incomplete training runs
    run_name=$(basename "$model_dir")
    result_file="$PROJECT_DIR/results/eval/Coder-1.5B_${run_name}.json"
    if [ -f "$result_file" ]; then
        log "SKIP eval (exists): $run_name"
        ((skip_count++)) || true
        continue
    fi
    log ">>> Evaluating: $run_name"
    if CUDA_VISIBLE_DEVICES="$ALL_GPUS" \
       USE_TORCH=1 \
       DISABLE_VERSION_CHECK=1 \
       $PYTHON "$SCRIPTS_DIR/run_lcb_eval.py" \
           --model_path "$model_dir" \
           --output_path "$result_file" \
           --tensor_parallel_size 8 \
           --n_samples 1 \
           --temperature 0.0 \
           --max_tokens 8192 \
           2>&1 | tee "$LOGS_DIR/eval_Coder-1.5B_${run_name}.log"; then
        ((eval_count++)) || true
    else
        log "FAILED eval: $run_name (check $LOGS_DIR/eval_Coder-1.5B_${run_name}.log)"
    fi
done

log "Eval done. Ran: $eval_count  Skipped: $skip_count"

# Aggregate CSV
$PYTHON - << 'EOF'
import json, glob, csv, os, re

project = "/newcpfs/lxh/coding-difficulty-exp"
rows = []
for f in sorted(glob.glob(f"{project}/results/eval/Coder-1.5B_*.json")):
    with open(f) as fh:
        d = json.load(fh)
    name = os.path.basename(f).replace(".json", "")
    m = re.match(r"(Coder-\d+\.?\d*B)_bucket_(\w+)_(\d+k)", name)
    if not m:
        continue
    model, bucket, size = m.groups()
    metrics = d.get("metrics", {})
    rows.append({"model": model, "bucket": bucket, "size": size,
                 "pass@1_easy":   metrics.get("pass@1_easy",   ""),
                 "pass@1_medium": metrics.get("pass@1_medium", ""),
                 "pass@1_hard":   metrics.get("pass@1_hard",   ""),
                 "pass@1_total":  metrics.get("pass@1_total",  "")})
out = f"{project}/results/eval/summary_1.5B.csv"
if rows:
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] Written {out}  ({len(rows)} rows)")
else:
    print("[summary] No eval results found.")
EOF

log ""
log "=========================================="
log "PHASE 3: Training — Coder-7B (low priority)"
log "=========================================="
bash "$SCRIPTS_DIR/train_all.sh" Coder-7B

log ""
log "=========================================="
log "All phases complete."
log "=========================================="
