#!/bin/bash
# Evaluate a trained model on HumanEval using vLLM.
# Usage: bash eval_one.sh <model_dir> [gpu_ids]
# Example: bash eval_one.sh results/Coder-1.5B/bucket_2k_4k_1k 0,1

set -e

MODEL_DIR="$1"
GPU_IDS="${2:-0}"

if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <model_dir> [gpu_ids]"
    exit 1
fi

PYTHON=/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python
PROJECT_DIR="/newcpfs/lxh/coding-difficulty-exp"
RUN_NAME=$(echo "$MODEL_DIR" | sed 's|.*/results/||' | tr '/' '_')
RESULT_FILE="$PROJECT_DIR/results/eval/${RUN_NAME}.json"

mkdir -p "$PROJECT_DIR/results/eval"

echo "========================================"
echo "Evaluating: $RUN_NAME"
echo "Model:  $MODEL_DIR"
echo "Output: $RESULT_FILE"
echo "========================================"

# Count GPUs for tensor parallelism
N_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
USE_TORCH=1 \
DISABLE_VERSION_CHECK=1 \
$PYTHON "$PROJECT_DIR/scripts/run_humaneval.py" \
    --model_path "$MODEL_DIR" \
    --output_path "$RESULT_FILE" \
    --tensor_parallel_size "$N_GPUS" \
    --n_samples 1 \
    --temperature 0.0 \
    --max_tokens 32768

echo "Done: $RUN_NAME  →  $RESULT_FILE"
