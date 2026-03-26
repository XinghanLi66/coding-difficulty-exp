#!/usr/bin/env python3
"""
Export training progress data to dashboard_data.json.

Usage:
    python scripts/export_data.py

Reads trainer_state.json, trainer_log.jsonl, eval results, and log files
for each (model, bucket, size) combination and writes a single JSON file
consumed by index.html.
"""

import json
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT = "/newcpfs/lxh/coding-difficulty-exp"
MODELS  = ["Coder-1.5B", "Coder-7B"]
BUCKETS = ["2k_4k", "4k_6k", "6k_8k", "8k_12k", "12k_16k", "16k_20k"]
SIZES   = ["1k", "2k", "4k", "8k"]  # all possible sizes (superset)

# Per-bucket valid sizes loaded from summary.json
def _load_bucket_sizes() -> dict:
    summary_path = os.path.join(PROJECT, "data", "processed", "summary.json")
    try:
        with open(summary_path) as f:
            s = json.load(f)
        result = {}
        for b in BUCKETS:
            raw = s.get("buckets", {}).get(b, {}).get("train_sizes", [])
            result[b] = [f"{n//1000}k" for n in raw]
        return result
    except Exception:
        return {b: SIZES for b in BUCKETS}

BUCKET_SIZES = _load_bucket_sizes()   # e.g. {"2k_4k": ["1k","2k","4k"], "16k_20k": ["1k","2k","4k","8k"]}

RESULTS_DIR = os.path.join(PROJECT, "results")
LOGS_DIR    = os.path.join(PROJECT, "logs")
EVAL_DIR    = os.path.join(RESULTS_DIR, "eval")
OUTPUT_PATH = os.path.join(PROJECT, "dashboard_data.json")

# Hyperparameter keys to surface from any YAML config (ordered for display)
HPARAM_KEYS = [
    "learning_rate",
    "lr_scheduler_type",
    "warmup_ratio",
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "bf16",
    "flash_attn",
    "finetuning_type",
    "deepspeed",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_dir(model: str, bucket: str, size: str) -> str:
    return os.path.join(RESULTS_DIR, model, f"bucket_{bucket}_{size}")


def read_json(path: str):
    """Return parsed JSON or None if the file doesn't exist / is unreadable."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def read_jsonl_last(path: str):
    """Return the last valid JSON line from a .jsonl file, or None."""
    try:
        last = None
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        pass
        return last
    except Exception:
        return None


def log_has_oom(model: str, bucket: str, size: str) -> bool:
    log_path = os.path.join(LOGS_DIR, f"train_{model}_bucket_{bucket}_{size}.log")
    try:
        with open(log_path, "r") as f:
            return "OutOfMemoryError" in f.read()
    except Exception:
        return False


def mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Per-run status + data extraction
# ---------------------------------------------------------------------------

def get_run_info(model: str, bucket: str, size: str) -> dict:
    """
    Returns a dict with keys:
        status, metric, metric_label, loss_curve, total_steps,
        _trainer_log_path (internal, used for current-run detection),
        _trainer_log_mtime (internal)
    """
    rdir = run_dir(model, bucket, size)
    config_path       = os.path.join(rdir, "config.json")
    state_path        = os.path.join(rdir, "trainer_state.json")
    trainer_log_path  = os.path.join(rdir, "trainer_log.jsonl")
    eval_path         = os.path.join(EVAL_DIR, f"{model}_bucket_{bucket}_{size}.json")

    has_config      = os.path.exists(config_path)
    has_trainer_log = os.path.exists(trainer_log_path)

    # ---- Determine status (priority order) --------------------------------
    if has_config:
        status = "done"
    elif has_trainer_log:
        # Will be refined to "running" for current run after all statuses computed
        if log_has_oom(model, bucket, size):
            status = "retry"
        else:
            status = "failed"
    else:
        status = "pending"

    # ---- trainer_state data -----------------------------------------------
    state = read_json(state_path)
    total_steps = None
    loss_curve  = []
    train_loss  = None

    if state:
        log_history = state.get("log_history", [])

        # Deduplicate by step, keep first occurrence
        seen_steps = set()
        for entry in log_history:
            step = entry.get("step")
            orig = entry.get("original_loss")
            if step is not None and step > 0 and orig is not None:
                if step not in seen_steps:
                    loss_curve.append([step, orig])
                    seen_steps.add(step)

        # Final entry has train_loss
        if log_history:
            final = log_history[-1]
            if "train_loss" in final:
                train_loss  = final["train_loss"]
                total_steps = final.get("step")

        # total_steps fallback: last step in loss_curve or trainer_log
        if total_steps is None and loss_curve:
            total_steps = loss_curve[-1][0]

    # ---- trainer_log total_steps (fallback / in-progress) -----------------
    trainer_log_last = None
    if has_trainer_log:
        trainer_log_last = read_jsonl_last(trainer_log_path)
        if trainer_log_last and total_steps is None:
            total_steps = trainer_log_last.get("total_steps")

    # ---- Eval metric -------------------------------------------------------
    eval_data = read_json(eval_path)
    if eval_data and "overall_pass@1" in eval_data:
        metric       = eval_data["overall_pass@1"]
        metric_label = "pass@1"
    elif train_loss is not None:
        metric       = train_loss
        metric_label = "train_loss"
    else:
        metric       = None
        metric_label = "train_loss"

    return {
        "status":              status,
        "metric":              metric,
        "metric_label":        metric_label,
        "loss_curve":          loss_curve,
        "total_steps":         total_steps,
        # Internal fields for current-run detection
        "_trainer_log_path":   trainer_log_path if has_trainer_log else None,
        "_trainer_log_mtime":  mtime(trainer_log_path) if has_trainer_log else 0.0,
        "_trainer_log_last":   trainer_log_last,
    }


# ---------------------------------------------------------------------------
# Hyperparameter extraction
# ---------------------------------------------------------------------------

def load_hparams() -> list:
    """
    Read the first available YAML config and extract key hyperparameters.
    Returns a list of [key, value] pairs.
    """
    configs_dir = os.path.join(PROJECT, "configs")
    if not os.path.isdir(configs_dir):
        return []

    # Pick the first non-smoke-test yaml
    yaml_files = sorted(
        f for f in os.listdir(configs_dir)
        if f.endswith(".yaml") and "smoke" not in f
    )
    if not yaml_files:
        return []

    yaml_path = os.path.join(configs_dir, yaml_files[0])
    hparams = {}
    try:
        with open(yaml_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip()
                    if key in HPARAM_KEYS:
                        hparams[key] = val
    except Exception:
        pass

    # Return in display order; mark absent keys
    result = []
    for k in HPARAM_KEYS:
        if k in hparams:
            result.append([k, hparams[k]])

    # Add a pseudo-row to emphasize fixed hparams
    result.append(["grid_search", "None — fixed hparams across all runs"])
    return result


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export():
    all_runs   = {}    # model -> {key -> info}
    total_runs = 0

    # Detect the active run from running llamafactory process (most reliable)
    def _detect_active_run():
        """Return (model, bucket, size) from the live llamafactory process, or None."""
        import subprocess, re
        try:
            out = subprocess.check_output(
                ["pgrep", "-a", "-f", "llamafactory.cli train"],
                text=True, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            return None
        # Extract config path from the process command line
        m = re.search(r"train\s+(\S+\.yaml)", out)
        if not m:
            return None
        cfg = m.group(1)
        # Parse model/bucket/size from config filename
        m2 = re.search(r"train_(Coder-[\d.]+B)_bucket_(\w+)_(\d+k)\.yaml", cfg)
        if not m2:
            return None
        return m2.group(1), m2.group(2), m2.group(3)

    active_run = _detect_active_run()

    # Collect all non-done runs that have a trainer_log, for fallback detection
    candidate_running = []  # (mtime, model, bucket, size)

    # First pass: collect statuses
    for model in MODELS:
        all_runs[model] = {}
        for bucket in BUCKETS:
            for size in BUCKET_SIZES[bucket]:
                total_runs += 1
                key  = f"{bucket}_{size}"
                info = get_run_info(model, bucket, size)
                all_runs[model][key] = info

                # Candidate if not done/pending and has trainer_log
                # Include "retry" runs since those are the ones being re-run
                if (info["status"] not in ("done", "pending")
                        and info["_trainer_log_path"] is not None):
                    candidate_running.append(
                        (info["_trainer_log_mtime"], model, bucket, size)
                    )

    # Identify current run: prefer process detection, fall back to newest mtime
    current_run = None
    if active_run:
        cur_model, cur_bucket, cur_size = active_run
    elif candidate_running:
        candidate_running.sort(key=lambda x: x[0], reverse=True)
        _, cur_model, cur_bucket, cur_size = candidate_running[0]
    else:
        cur_model = cur_bucket = cur_size = None

    if cur_model:
        cur_key  = f"{cur_bucket}_{cur_size}"
        cur_info = all_runs.get(cur_model, {}).get(cur_key)
        if cur_info is None:
            cur_model = cur_bucket = cur_size = None
    if cur_model:
        cur_info["status"] = "running"

        log_last = cur_info["_trainer_log_last"]
        if log_last:
            current_run = {
                "model":       cur_model,
                "bucket":      cur_bucket,
                "size":        cur_size,
                "step":        log_last.get("current_steps"),
                "total_steps": log_last.get("total_steps"),
                "pct":         log_last.get("percentage"),
                "eta":         log_last.get("remaining_time"),
                "elapsed":     log_last.get("elapsed_time"),
            }

    # Count completed
    completed = sum(
        1
        for model in MODELS
        for bucket in BUCKETS
        for size in BUCKET_SIZES[bucket]
        if all_runs[model][f"{bucket}_{size}"]["status"] == "done"
    )

    # Build clean output (strip internal fields)
    clean_runs = {}
    for model in MODELS:
        clean_runs[model] = {}
        for bucket in BUCKETS:
            for size in BUCKET_SIZES[bucket]:
                key  = f"{bucket}_{size}"
                info = all_runs[model][key]
                clean_runs[model][key] = {
                    "status":       info["status"],
                    "metric":       info["metric"],
                    "metric_label": info["metric_label"],
                    "loss_curve":   info["loss_curve"],
                    "total_steps":  info["total_steps"],
                }

    output = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_runs":   total_runs,
        "completed":    completed,
        "current_run":  current_run,
        "hparams":      load_hparams(),
        "runs":         clean_runs,
        "buckets":      BUCKETS,
        "bucket_sizes": BUCKET_SIZES,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[export_data] Written to {OUTPUT_PATH}")
    print(f"  total={total_runs}  completed={completed}  current={current_run}")


if __name__ == "__main__":
    export()
