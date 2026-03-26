#!/bin/bash
# Run all training jobs sequentially.
# Skips runs where the output model already exists.
# Usage: bash train_all.sh [model_tag_filter]
# Example: bash train_all.sh Coder-1.5B   (only run 1.5B configs)

FILTER="${1:-}"
PROJECT_DIR="/newcpfs/lxh/coding-difficulty-exp"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
PYTHON=/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python

# Wait until all GPUs have < threshold MiB in use, then proceed
wait_for_gpus() {
    local threshold=2000  # MiB — anything below this is "idle"
    local max_wait=120    # seconds
    local elapsed=0
    while true; do
        local busy
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk -v t="$threshold" '$1 > t {count++} END {print count+0}')
        [ "$busy" -eq 0 ] && return 0
        [ "$elapsed" -ge "$max_wait" ] && { echo "GPU wait timeout after ${max_wait}s"; return 0; }
        echo "  Waiting for GPUs to free (${busy} busy, ${elapsed}s elapsed)..."
        sleep 10
        elapsed=$((elapsed + 10))
    done
}

run_pass() {
    local total=0 skipped=0 failed=0
    for cfg in "$PROJECT_DIR"/configs/train_*.yaml; do
        local name
        name=$(basename "$cfg" .yaml)

        if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then
            continue
        fi

        local out_dir
        out_dir=$(grep "^output_dir:" "$cfg" | awk '{print $2}')
        if [ -d "$out_dir" ] && [ -f "$out_dir/config.json" ]; then
            echo "SKIP (exists): $name"
            ((skipped++)) || true
            continue
        fi

        wait_for_gpus

        echo ""
        echo ">>> Running: $name"
        total=$((total + 1))
        if ! bash "$SCRIPTS_DIR/train_one.sh" "$cfg"; then
            echo "FAILED: $name"
            failed=$((failed + 1))
        fi
    done
    echo ""
    echo "Pass done. Ran: $total  Skipped: $skipped  Failed: $failed"
    echo "$failed"
}

echo "========================================"
echo "Pass 1"
echo "========================================"
failed=$(run_pass)

# Retry pass for any runs that failed (e.g. transient OOM)
if [ "${failed:-0}" -gt 0 ]; then
    echo ""
    echo "========================================"
    echo "Pass 2 (retrying $failed failed runs)"
    echo "========================================"
    wait_for_gpus
    run_pass
fi

echo ""
echo "========================================"
echo "All passes done."
echo "========================================"
