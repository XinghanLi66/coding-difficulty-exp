#!/bin/bash
# Run eval for all Coder-1.5B models, skipping already-evaluated ones.
# Usage: bash scripts/run_all_1.5B_evals.sh [gpu_ids]
#   gpu_ids defaults to "0,1,2,3"

set -e

GPU_IDS="${1:-0,1,2,3}"
PROJECT_DIR="/newcpfs/lxh/coding-difficulty-exp"
PYTHON=/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python

log() { echo "[$(date '+%H:%M:%S')] $*"; }

for model_dir in "$PROJECT_DIR"/results/Coder-1.5B/bucket_*/; do
    model_dir="${model_dir%/}"  # strip trailing slash
    bucket=$(basename "$model_dir")
    result_file="$PROJECT_DIR/results/eval/Coder-1.5B_${bucket}.json"

    if [ -f "$result_file" ]; then
        log "SKIP (exists): $bucket"
        continue
    fi

    if [ ! -f "$model_dir/config.json" ]; then
        log "SKIP (no model): $bucket"
        continue
    fi

    log ">>> Evaluating: $bucket"
    bash "$PROJECT_DIR/scripts/eval_one.sh" "$model_dir" "$GPU_IDS"
    log "Done: $bucket"

    # Update summary after each eval
    $PYTHON "$PROJECT_DIR/scripts/gen_results_table.py" 2>/dev/null || true
done

log "All evals complete. Final table:"
$PYTHON "$PROJECT_DIR/scripts/gen_results_table.py"
