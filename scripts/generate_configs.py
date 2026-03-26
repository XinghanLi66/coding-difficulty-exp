"""
Generate one LLaMA-Factory YAML config per (model, bucket, size) combination.
"""
import os
import yaml

PROJECT_DIR = "/newcpfs/lxh/coding-difficulty-exp"
LLAMA_FACTORY_DIR = "/newcpfs/lxh/my_LLaMAFactory"

MODELS = {
    "Coder-1.5B": "/newcpfs/user/sujianghao/model/Qwen/Qwen2.5-Coder-1.5B",
    "Coder-7B":   "/newcpfs/user/sujianghao/model/Qwen/Qwen2.5-Coder-7B",
}

BUCKETS = ["2k_4k", "4k_6k", "6k_8k", "8k_12k", "12k_16k", "16k_20k"]
SIZES   = ["1k", "2k", "4k", "8k"]   # 8k only exists for 16k_20k bucket

DEEPSPEED_CONFIG = f"{LLAMA_FACTORY_DIR}/examples/deepspeed/ds_z3_config.json"

BASE_CFG = dict(
    trust_remote_code=True,
    stage="sft",
    do_train=True,
    use_dft_loss=False,
    finetuning_type="full",
    deepspeed=DEEPSPEED_CONFIG,
    dataset_dir=f"{PROJECT_DIR}/data",
    template="qwen",
    cutoff_len=20480,
    overwrite_cache=True,
    preprocessing_num_workers=1,
    output_dir=None,          # filled per run
    logging_steps=10,
    save_steps=500,
    overwrite_output_dir=True,
    save_only_model=True,
    report_to="none",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5.0e-5,
    num_train_epochs=1.0,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    flash_attn="fa2",
)


def make_config(model_tag, model_path, bucket, size):
    dataset_key = f"bucket_{bucket}_{size}"
    out_dir = f"{PROJECT_DIR}/results/{model_tag}/{dataset_key}"
    cfg = dict(BASE_CFG)
    cfg["model_name_or_path"] = model_path
    cfg["dataset"] = dataset_key
    cfg["output_dir"] = out_dir
    return cfg


def main():
    cfg_dir = f"{PROJECT_DIR}/configs"
    os.makedirs(cfg_dir, exist_ok=True)

    generated = []
    for model_tag, model_path in MODELS.items():
        for bucket in BUCKETS:
            for size in SIZES:
                # Check if dataset exists
                jsonl = f"{PROJECT_DIR}/data/processed/bucket_{bucket}/train_{size}.jsonl"
                if not os.path.exists(jsonl):
                    continue
                cfg = make_config(model_tag, model_path, bucket, size)
                fname = f"train_{model_tag}_bucket_{bucket}_{size}.yaml"
                fpath = os.path.join(cfg_dir, fname)
                with open(fpath, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                generated.append(fname)

    print(f"Generated {len(generated)} configs in {cfg_dir}:")
    for f in generated:
        print(f"  {f}")


if __name__ == "__main__":
    main()
