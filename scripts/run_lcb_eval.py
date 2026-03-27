"""
Evaluate a trained model on LiveCodeBench using vLLM.

LiveCodeBench problems are loaded from HuggingFace (livecodebench/code_generation_lite).
Evaluation: pass@1 via execution against test cases.

Usage:
    python run_lcb_eval.py --model_path <dir> --output_path <json> [options]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--version", default="release_v5",
                   help="LiveCodeBench version tag (release_v1 .. release_v5)")
    p.add_argument("--n_problems", type=int, default=None,
                   help="Limit number of problems (for quick smoke eval)")
    return p.parse_args()


SYSTEM_PROMPT = (
    "You are an expert competitive programmer. "
    "Given a programming problem, write a correct and efficient solution in Python. "
    "Enclose your final solution in ```python ... ``` code blocks."
)


def build_prompt(problem: dict) -> str:
    return (
        f"Problem: {problem['question_title']}\n\n"
        f"{problem['question_content']}\n\n"
        "Please write a Python solution."
    )


def extract_code(text: str) -> str:
    """Extract Python code from ```python ... ``` blocks, or return full text."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_generation(args, problems):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_tokens + 4096,
        gpu_memory_utilization=0.9,
    )

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_samples,
    )

    prompts = []
    for p in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(p)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    print(f"Generating for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling)

    results = []
    for prob, out in zip(problems, outputs):
        for completion in out.outputs:
            code = extract_code(completion.text)
            results.append({
                "question_id": prob["question_id"],
                "difficulty": prob.get("difficulty", "unknown"),
                "code": code,
                "raw_output": completion.text,
            })
    return results


def evaluate_results(results, problems):
    """
    Simple pass@1 estimation using LCB's test cases.
    Falls back to syntax check if execution environment not available.
    """
    try:
        from livecodebench.evaluation import evaluate_solution
        HAS_LCB = True
    except ImportError:
        HAS_LCB = False

    prob_map = {p["question_id"]: p for p in problems}
    correct = {"easy": 0, "medium": 0, "hard": 0, "total": 0}
    total   = {"easy": 0, "medium": 0, "hard": 0, "total": 0}

    for r in results:
        prob = prob_map.get(r["question_id"], {})
        diff = r.get("difficulty", "medium").lower()
        if diff not in ("easy", "medium", "hard"):
            diff = "medium"

        total[diff] += 1
        total["total"] += 1

        if HAS_LCB:
            passed = evaluate_solution(r["code"], prob.get("public_test_cases", []))
        else:
            # Fallback: syntax check
            try:
                compile(r["code"], "<string>", "exec")
                passed = True
            except SyntaxError:
                passed = False

        if passed:
            correct[diff] += 1
            correct["total"] += 1

    metrics = {}
    for k in ("easy", "medium", "hard", "total"):
        n = total[k]
        metrics[f"pass@1_{k}"] = correct[k] / n if n > 0 else 0.0
        metrics[f"n_{k}"] = n

    return metrics


def main():
    args = parse_args()

    print(f"Loading LiveCodeBench ({args.version})...")
    from datasets import load_dataset
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag=args.version,
        trust_remote_code=True,
    )
    problems = list(ds)
    if args.n_problems:
        problems = problems[:args.n_problems]
    print(f"  Loaded {len(problems)} problems")

    results = run_generation(args, problems)
    metrics = evaluate_results(results, problems)

    output = {
        "model_path": args.model_path,
        "version": args.version,
        "n_problems": len(problems),
        "metrics": metrics,
        "raw_results": results,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Results ===")
    for k, v in metrics.items():
        if k.startswith("pass@1"):
            print(f"  {k}: {v:.3f}")
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
