#!/usr/bin/env python3
"""
Training progress dashboard.
Usage: python dashboard.py [--out dashboard.png]
Reads trainer_state.json / trainer_log.jsonl from results/ and generates a PNG.
"""

import argparse
import json
import os
import re
import glob
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── config ───────────────────────────────────────────────────────────────────
PROJECT   = "/newcpfs/lxh/coding-difficulty-exp"
RESULTS   = f"{PROJECT}/results"
LOGS      = f"{PROJECT}/logs"
EVAL_DIR  = f"{PROJECT}/results/eval"
CONFIGS   = f"{PROJECT}/configs"

MODELS  = ["Coder-1.5B", "Coder-7B"]
BUCKETS = ["2k_4k", "4k_6k", "6k_8k", "8k_12k", "12k_16k", "16k_20k"]
SIZES   = ["1k", "2k", "4k", "8k"]  # superset; per-bucket valid sizes in BUCKET_SIZES

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

BUCKET_SIZES = _load_bucket_sizes()

# fixed hyperparameters (same for all runs)
HPARAMS = [
    ("learning_rate",          "5e-5"),
    ("lr_scheduler",           "constant"),
    ("warmup_ratio",           "0.03"),
    ("num_epochs",             "1"),
    ("per_device_batch_size",  "1"),
    ("gradient_accum_steps",   "8"),
    ("effective_batch_size",   "64 (8 GPUs × 1 × 8)"),
    ("cutoff_len",             "20480 tokens"),
    ("finetuning_type",        "full (ZeRO-3)"),
    ("flash_attn",             "fa2"),
    ("bf16",                   "True"),
    ("grid_search",            "None — fixed hparams across all runs"),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def result_dir(model, bucket, size):
    return f"{RESULTS}/{model}/bucket_{bucket}_{size}"

def is_completed(model, bucket, size):
    return os.path.exists(f"{result_dir(model, bucket, size)}/config.json")

def load_trainer_state(model, bucket, size):
    path = f"{result_dir(model, bucket, size)}/trainer_state.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def get_final_train_loss(model, bucket, size):
    state = load_trainer_state(model, bucket, size)
    if state is None:
        return None
    for entry in reversed(state.get("log_history", [])):
        if "train_loss" in entry:
            return entry["train_loss"]
    return None

def get_loss_curve(model, bucket, size):
    """Returns (steps[], losses[]) from trainer_state log_history."""
    state = load_trainer_state(model, bucket, size)
    if state is None:
        return [], []
    steps, losses = [], []
    for entry in state.get("log_history", []):
        if "original_loss" in entry and entry.get("step", 0) > 0:
            steps.append(entry["step"])
            losses.append(entry["original_loss"])
    return steps, losses

def get_eval_pass1(model, bucket, size):
    path = f"{EVAL_DIR}/{model}_bucket_{bucket}_{size}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("overall_pass@1")

def find_current_run():
    """Return (model, bucket, size) by checking the most recently updated trainer_log.jsonl."""
    # prefer a run whose trainer_log.jsonl was modified in the last 5 min (actively writing)
    import time
    now = time.time()
    best_mtime, best = 0, (None, None, None)
    for model in MODELS:
        for b in BUCKETS:
            for s in BUCKET_SIZES[b]:
                rdir = result_dir(model, b, s)
                tlog = f"{rdir}/trainer_log.jsonl"
                if os.path.exists(tlog) and not is_completed(model, b, s):
                    mt = os.path.getmtime(tlog)
                    if mt > best_mtime:
                        best_mtime = mt
                        best = (model, b, s)
    # fall back to most recently modified training log file
    if best[0] is None:
        logs = glob.glob(f"{LOGS}/train_Coder-*.log")
        if logs:
            logs.sort(key=os.path.getmtime, reverse=True)
            for log in logs:
                name = os.path.basename(log).replace(".log", "").replace("train_", "")
                m = re.match(r"(Coder-\d+\.?\d*B)_bucket_(.+)_(\d+k)$", name)
                if m:
                    model, bucket, size = m.group(1), m.group(2), m.group(3)
                    if model in MODELS and bucket in BUCKETS and size in SIZES:
                        best = (model, bucket, size)
                        break
    return best

def load_current_progress(model, bucket, size):
    """Read trainer_log.jsonl for the most recent step/ETA."""
    path = f"{result_dir(model, bucket, size)}/trainer_log.jsonl"
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                pass
    return last

def get_current_loss_curve(model, bucket, size):
    """Loss curve from trainer_state (if exists) or by parsing log file."""
    if is_completed(model, bucket, size):
        return get_loss_curve(model, bucket, size)
    # Parse log file for in-progress run
    log_path = f"{LOGS}/train_{model}_bucket_{bucket}_{size}.log"
    if not os.path.exists(log_path):
        return [], []
    # Read trainer_log.jsonl from the (possibly partial) result dir
    state_path = f"{result_dir(model, bucket, size)}/trainer_state.json"
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        steps, losses = [], []
        for entry in state.get("log_history", []):
            if "original_loss" in entry and entry.get("step", 0) > 0:
                steps.append(entry["step"])
                losses.append(entry["original_loss"])
        return steps, losses
    return [], []

_current_run_cache = None
def _get_current_run():
    global _current_run_cache
    if _current_run_cache is None:
        _current_run_cache = find_current_run()
    return _current_run_cache

def _log_has_oom(model, bucket, size):
    log = f"{LOGS}/train_{model}_bucket_{bucket}_{size}.log"
    if not os.path.exists(log):
        return False
    with open(log) as f:
        return "OutOfMemoryError" in f.read()

def run_status(model, bucket, size):
    """'done' | 'running' | 'retry' | 'failed' | 'pending' | 'n/a'"""
    if size not in BUCKET_SIZES.get(bucket, SIZES):
        return "n/a"
    if is_completed(model, bucket, size):
        return "done"
    rdir = result_dir(model, bucket, size)
    tlog_exists = os.path.exists(f"{rdir}/trainer_log.jsonl")
    log = f"{LOGS}/train_{model}_bucket_{bucket}_{size}.log"
    log_exists = os.path.exists(log)
    cur_m, cur_b, cur_s = _get_current_run()
    if (model, bucket, size) == (cur_m, cur_b, cur_s):
        return "running"
    if tlog_exists:
        # Directory with trainer_log but no config.json = attempted but incomplete
        if _log_has_oom(model, bucket, size):
            return "retry"   # OOM — was GPU contention, will retry
        return "failed"
    return "pending"


# ── plotting helpers ──────────────────────────────────────────────────────────

STATUS_COLOR = {"done": "#4CAF50", "running": "#FFC107", "retry": "#FF9800", "failed": "#F44336", "pending": "#9E9E9E", "n/a": "#2a2a2a"}

def build_metric_matrix(model):
    """Return (matrix, mask_done) shaped (len(BUCKETS), len(SIZES))."""
    mat = np.full((len(BUCKETS), len(SIZES)), np.nan)
    for bi, b in enumerate(BUCKETS):
        for si, s in enumerate(SIZES):
            # prefer eval pass@1, fall back to train loss
            v = get_eval_pass1(model, b, s)
            if v is None:
                v = get_final_train_loss(model, b, s)
            if v is not None:
                mat[bi, si] = v
    return mat

def has_eval():
    return any(
        get_eval_pass1(m, b, s) is not None
        for m in MODELS for b in BUCKETS for s in BUCKET_SIZES[b]
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main(out_path):
    use_eval = has_eval()
    metric_label = "pass@1 (↑)" if use_eval else "train loss (↓)"
    cur_m, cur_b, cur_s = find_current_run()

    # count stats
    done  = sum(1 for m in MODELS for b in BUCKETS for s in BUCKET_SIZES[b] if is_completed(m, b, s))
    total = sum(len(BUCKET_SIZES[b]) for b in BUCKETS) * len(MODELS)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#1a1a2e")

    gs_top   = gridspec.GridSpec(1, 2, figure=fig, left=0.05, right=0.97, top=0.92, bottom=0.58, wspace=0.08)
    gs_mid   = gridspec.GridSpec(1, 2, figure=fig, left=0.05, right=0.97, top=0.54, bottom=0.28, wspace=0.25)
    gs_bot   = gridspec.GridSpec(1, 1, figure=fig, left=0.05, right=0.97, top=0.23, bottom=0.02)

    ax_hm = [fig.add_subplot(gs_top[0, i]) for i in range(2)]  # 2D heatmaps
    ax_prog  = fig.add_subplot(gs_mid[0, 0])   # progress table
    ax_loss  = fig.add_subplot(gs_mid[0, 1])   # loss curve current run
    ax_hp    = fig.add_subplot(gs_bot[0, 0])   # hparam table

    DARK_BG  = "#16213e"
    TEXT_COL = "#e0e0e0"
    for ax in ax_hm + [ax_prog, ax_loss, ax_hp]:
        ax.set_facecolor(DARK_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # title
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.5, 0.965, f"Coding Difficulty Experiment — Dashboard  [{ts}]",
             ha="center", va="top", fontsize=15, fontweight="bold", color=TEXT_COL)
    fig.text(0.5, 0.945, f"Progress: {done}/{total} runs completed  |  metric: {metric_label}",
             ha="center", va="top", fontsize=11, color="#aaaaaa")

    # ── 2D heatmaps ───────────────────────────────────────────────────────────
    cmap_loss = LinearSegmentedColormap.from_list("rg", ["#d32f2f", "#ff9800", "#4CAF50"])
    cmap_eval = LinearSegmentedColormap.from_list("gy", ["#1565C0", "#42A5F5", "#4CAF50"])
    cmap = cmap_eval if use_eval else cmap_loss

    for ax, model in zip(ax_hm, MODELS):
        mat = build_metric_matrix(model)

        # determine vmin/vmax from non-nan values
        valid = mat[~np.isnan(mat)]
        if len(valid) > 0:
            if use_eval:
                vmin, vmax = 0, max(valid.max(), 0.3)
            else:
                vmin, vmax = valid.min() * 0.95, valid.max() * 1.05
        else:
            vmin, vmax = 0, 1

        # grey background for NaN cells
        nan_mat = np.where(np.isnan(mat), 1.0, np.nan)
        ax.imshow(nan_mat, aspect="auto", cmap="gray_r", alpha=0.15,
                  vmin=0, vmax=1, origin="upper",
                  extent=[-0.5, len(SIZES)-0.5, len(BUCKETS)-0.5, -0.5])

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="upper", alpha=0.9,
                       extent=[-0.5, len(SIZES)-0.5, len(BUCKETS)-0.5, -0.5])

        # cell annotations
        for bi, b in enumerate(BUCKETS):
            for si, s in enumerate(SIZES):
                status = run_status(model, b, s)
                if status == "n/a":
                    ax.text(si, bi, "—", ha="center", va="center", fontsize=9, color="#444")
                    continue
                v = mat[bi, si]
                if not np.isnan(v):
                    txt = f"{v:.3f}"
                    brightness = (v - vmin) / (vmax - vmin + 1e-9)
                    tc = "white" if (use_eval and brightness < 0.5) or (not use_eval and brightness > 0.5) else "#222"
                    ax.text(si, bi, txt, ha="center", va="center",
                            fontsize=8.5, color=tc, fontweight="bold")
                else:
                    sym = "▶" if status == "running" else ("·" if status == "pending" else "✗")
                    col = STATUS_COLOR.get(status, "gray")
                    ax.text(si, bi, sym, ha="center", va="center", fontsize=11, color=col)

        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels(SIZES, color=TEXT_COL, fontsize=9)
        ax.set_yticks(range(len(BUCKETS)))
        ax.set_yticklabels([f"CoT {b}" for b in BUCKETS], color=TEXT_COL, fontsize=9)
        ax.set_xlabel("Data size (N)", color=TEXT_COL, fontsize=10)
        ax.set_title(f"Qwen2.5-{model}", color=TEXT_COL, fontsize=12, fontweight="bold", pad=6)
        ax.tick_params(colors="#888")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=TEXT_COL, labelcolor=TEXT_COL)

    # ── progress table ─────────────────────────────────────────────────────────
    ax_prog.set_xlim(0, len(SIZES))
    ax_prog.set_ylim(len(BUCKETS), 0)
    ax_prog.set_xticks(np.arange(len(SIZES)) + 0.5)
    ax_prog.set_xticklabels(SIZES, color=TEXT_COL, fontsize=9)
    ax_prog.set_yticks(np.arange(len(BUCKETS)) + 0.5)
    ax_prog.set_yticklabels([f"CoT {b}" for b in BUCKETS], color=TEXT_COL, fontsize=9)
    ax_prog.tick_params(colors="#888")

    legend_patches = [mpatches.Patch(facecolor=v, label=k) for k, v in STATUS_COLOR.items()]
    ax_prog.legend(handles=legend_patches, loc="upper right", fontsize=8,
                   facecolor="#222", labelcolor=TEXT_COL, framealpha=0.8)
    ax_prog.set_title("Run Status  (Coder-1.5B — top, Coder-7B — bottom)", color=TEXT_COL, fontsize=10, pad=5)

    for mi, model in enumerate(MODELS):
        row_offset = mi * len(BUCKETS) / 2   # stack both models vertically in half-height
        for bi, b in enumerate(BUCKETS):
            for si, s in enumerate(SIZES):
                status = run_status(model, b, s)
                col = STATUS_COLOR.get(status, "#9E9E9E")
                y0 = (bi + mi * len(BUCKETS)) / (len(MODELS) * len(BUCKETS)) * len(BUCKETS)
                h  = 1.0 / len(MODELS)
                rect = plt.Rectangle((si, bi/len(MODELS) + mi*len(BUCKETS)/len(MODELS)/1),
                                     1, 1/len(MODELS), color=col, alpha=0.7)
                ax_prog.add_patch(rect)
                if is_completed(model, b, s):
                    ax_prog.text(si + 0.5, bi/len(MODELS) + mi*len(BUCKETS)/len(MODELS)/1 + 0.5/len(MODELS),
                                 "✓", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
                elif status == "running":
                    ax_prog.text(si + 0.5, bi/len(MODELS) + mi*len(BUCKETS)/len(MODELS)/1 + 0.5/len(MODELS),
                                 "▶", ha="center", va="center", fontsize=9, color="white")

    ax_prog.set_ylim(len(BUCKETS), 0)
    ax_prog.set_xlim(0, len(SIZES))

    # separator line between models
    ax_prog.axhline(y=len(BUCKETS)/2, color="#888", linestyle="--", linewidth=0.8)
    ax_prog.text(-0.05, len(BUCKETS)/4, "1.5B", ha="right", va="center",
                 color="#aaa", fontsize=8, transform=ax_prog.transData)
    ax_prog.text(-0.05, 3*len(BUCKETS)/4, "7B", ha="right", va="center",
                 color="#aaa", fontsize=8, transform=ax_prog.transData)

    # ── loss curve of current run ──────────────────────────────────────────────
    ax_loss.set_title("Loss Curve — Current / Most Recent Run", color=TEXT_COL, fontsize=11, pad=5)
    ax_loss.set_xlabel("Training Step", color=TEXT_COL, fontsize=9)
    ax_loss.set_ylabel("original_loss", color=TEXT_COL, fontsize=9)
    ax_loss.tick_params(colors="#888", labelcolor=TEXT_COL)

    plotted_any = False
    # plot all completed runs' loss curves (faint grey)
    for model in MODELS:
        for b in BUCKETS:
            for s in BUCKET_SIZES[b]:
                if is_completed(model, b, s):
                    steps, losses = get_loss_curve(model, b, s)
                    if steps:
                        # smooth
                        ax_loss.plot(steps, losses, color="#555", linewidth=0.4, alpha=0.3)
                        plotted_any = True

    # highlight current run
    if cur_m and cur_b and cur_s:
        steps, losses = get_current_loss_curve(cur_m, cur_b, cur_s)
        if steps:
            ax_loss.plot(steps, losses, color="#FFC107", linewidth=2.0,
                         label=f"{cur_m} bucket_{cur_b} {cur_s}")
            plotted_any = True

        # progress info
        prog = load_current_progress(cur_m, cur_b, cur_s)
        if prog:
            pct  = prog.get("percentage", 0)
            eta  = prog.get("remaining_time", "?")
            cstep = prog.get("current_steps", "?")
            tstep = prog.get("total_steps", "?")
            ax_loss.text(0.02, 0.96, f"  {cur_m} | bucket_{cur_b} | {cur_s}\n"
                         f"  Step {cstep}/{tstep}  ({pct:.1f}%)  ETA: {eta}",
                         transform=ax_loss.transAxes, va="top", ha="left",
                         fontsize=9, color="#FFC107",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.8))
        ax_loss.legend(fontsize=8, facecolor="#222", labelcolor=TEXT_COL, framealpha=0.8)

    if not plotted_any:
        ax_loss.text(0.5, 0.5, "No training data yet", ha="center", va="center",
                     transform=ax_loss.transAxes, color="#888", fontsize=12)
    ax_loss.grid(True, color="#333", linewidth=0.5)

    # ── hyperparameter table ───────────────────────────────────────────────────
    ax_hp.axis("off")
    ax_hp.set_title("Hyperparameters  (fixed — no local grid search)", color=TEXT_COL, fontsize=11, pad=5)

    col_labels = ["Hyperparameter", "Value"]
    rows = HPARAMS

    n_cols = 2  # side-by-side two HP tables to save vertical space
    half = len(rows) // 2 + len(rows) % 2
    left_rows  = rows[:half]
    right_rows = rows[half:]
    # pad to equal length
    while len(right_rows) < len(left_rows):
        right_rows.append(("", ""))

    combined = [[f"{l[0]}",  f"{l[1]}",  f"{r[0]}",  f"{r[1]}"]
                for l, r in zip(left_rows, right_rows)]

    tbl = ax_hp.table(
        cellText=combined,
        colLabels=["Parameter", "Value", "Parameter", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#0d3b66" if row == 0 else ("#1e2a40" if row % 2 == 0 else "#162032"))
        cell.set_text_props(color=TEXT_COL)
        cell.set_edgecolor("#333")
        if col in (0, 2) and row > 0:
            cell.set_text_props(color="#80BFFF", fontweight="bold")

    # save
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Dashboard saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=f"{PROJECT}/dashboard.png")
    args = parser.parse_args()
    main(args.out)
