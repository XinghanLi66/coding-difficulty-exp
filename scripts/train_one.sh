#!/bin/bash
# Train a single (model, bucket, size) configuration.
# Usage: bash train_one.sh <config_yaml>
# Example: bash train_one.sh configs/train_Coder-1.5B_bucket_2k_4k_1k.yaml

set -e

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "Usage: $0 <config_yaml>"
    exit 1
fi

PYTHON=/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python
LLAMAFACTORY=/newcpfs/lxh/my_LLaMAFactory
CONDA_BIN=/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin
export PATH="$CONDA_BIN:$PATH"

RUN_NAME=$(basename "$CONFIG" .yaml)
LOG_FILE="/newcpfs/lxh/coding-difficulty-exp/logs/${RUN_NAME}.log"

echo "========================================"
echo "Starting: $RUN_NAME"
echo "Config:   $CONFIG"
echo "Log:      $LOG_FILE"
echo "========================================"

cd "$LLAMAFACTORY"

FORCE_TORCHRUN=1 \
DISABLE_VERSION_CHECK=1 \
USE_TORCH=1 \
PYTHONPATH="$LLAMAFACTORY/src:$PYTHONPATH" \
    $PYTHON -m llamafactory.cli train "$CONFIG" 2>&1 | tee "$LOG_FILE"

echo "Done: $RUN_NAME"
