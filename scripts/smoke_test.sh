#!/bin/bash
# Smoke test: one tiny training run (50 samples, 1 epoch) to verify the full pipeline.
# Uses GPU 0 only, no DeepSpeed, cutoff_len=2048 for speed.

set -e

PROJECT_DIR="/newcpfs/lxh/coding-difficulty-exp"
LLAMA_FACTORY_DIR="/newcpfs/lxh/my_LLaMAFactory"
PYTHON=/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python

SMOKE_CFG="$PROJECT_DIR/configs/smoke_test.yaml"
SMOKE_OUT="$PROJECT_DIR/results/smoke_test"
LOG="$PROJECT_DIR/logs/smoke_test.log"

cat > "$SMOKE_CFG" << 'YAML'
### Smoke test: Coder-1.5B, bucket_2k_4k, 50 samples
model_name_or_path: /newcpfs/user/sujianghao/model/Qwen/Qwen2.5-Coder-1.5B
trust_remote_code: true

stage: sft
do_train: true
use_dft_loss: false
finetuning_type: full

dataset: bucket_2k_4k_1k
dataset_dir: /newcpfs/lxh/coding-difficulty-exp/data
template: qwen
cutoff_len: 2048
max_samples: 50
overwrite_cache: true
preprocessing_num_workers: 4

output_dir: /newcpfs/lxh/coding-difficulty-exp/results/smoke_test
logging_steps: 5
save_steps: 9999
overwrite_output_dir: true
save_only_model: true
report_to: none

per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 5.0e-5
num_train_epochs: 1.0
lr_scheduler_type: constant
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
flash_attn: fa2
YAML

echo "========================================"
echo "Smoke test starting"
echo "Config: $SMOKE_CFG"
echo "Output: $SMOKE_OUT"
echo "Log:    $LOG"
echo "========================================"

cd "$LLAMA_FACTORY_DIR"

CUDA_VISIBLE_DEVICES=0 \
DISABLE_VERSION_CHECK=1 \
PYTHONPATH="$LLAMA_FACTORY_DIR/src:$PYTHONPATH" \
    $PYTHON -m llamafactory.cli train "$SMOKE_CFG" 2>&1 | tee "$LOG"

echo ""
echo "========================================"
echo "Smoke test complete. Checking output..."
echo "========================================"

if [ -f "$SMOKE_OUT/config.json" ]; then
    echo "PASS: model config saved at $SMOKE_OUT/config.json"
else
    echo "FAIL: no config.json found in $SMOKE_OUT"
    exit 1
fi

echo ""
echo "Smoke test PASSED."
