"""
Evaluate a trained model on HumanEval using vLLM.

HumanEval problems are loaded from HuggingFace (openai/openai_humaneval).
Evaluation: pass@1 via subprocess execution against the bundled test function.

Usage:
    python run_humaneval.py --model_path <dir> --output_path <json> [options]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=32768)
    p.add_argument("--n_problems", type=int, default=None,
                   help="Limit number of problems (for quick smoke eval)")
    return p.parse_args()


SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Complete the given Python function. "
    "Your answer must contain the complete function implementation "
    "(including the function signature) in a ```python ... ``` code block."
)


def build_prompt(problem: dict) -> str:
    return (
        f"Complete the following Python function:\n\n"
        f"```python\n{problem['prompt']}```\n\n"
        "Write the complete function in a ```python ... ``` code block."
    )


def extract_python_code(text: str) -> str:
    """
    Extract Python code from model output.
    Strips <think>...</think> block first, then finds ```python...``` block.
    """
    # Strip thinking block
    think_end = text.find("</think>")
    body = text[think_end + len("</think>"):].strip() if think_end != -1 else text

    # Try ```python...``` block first
    m = re.search(r"```python\s*(.*?)```", body, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Try any code block
    m = re.search(r"```\s*(.*?)```", body, re.DOTALL)
    if m:
        return m.group(1).strip()

    # No block — return body as-is
    return body.strip()


def run_generation(args, problems):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=min(args.max_tokens + 4096, 32768),
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
            code = extract_python_code(completion.text)
            results.append({
                "task_id":     prob["task_id"],
                "code":        code,
                "raw_output":  completion.text,
            })
    return results


def evaluate_one(task_id: str, code: str, prompt: str, test: str, entry_point: str,
                 timeout: float = 10.0) -> bool:
    """
    Run the generated code + test function in a subprocess.
    Returns True if the solution passes all tests.
    """
    # If the extracted code doesn't define the function (just the body),
    # prepend the original stub so the function exists.
    fn_def = f"def {entry_point}"
    if fn_def not in code:
        full_code = prompt + "\n" + code
    else:
        full_code = code

    script = f"{full_code}\n\n{test}\n\ncheck({entry_point})\n"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            fname = f.name
        r = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout
        )
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


def evaluate_results(results, problems):
    prob_map = {p["task_id"]: p for p in problems}
    correct = 0
    total = len(results)

    for r in results:
        prob = prob_map.get(r["task_id"], {})
        passed = evaluate_one(
            task_id=r["task_id"],
            code=r["code"],
            prompt=prob.get("prompt", ""),
            test=prob.get("test", ""),
            entry_point=prob.get("entry_point", ""),
        )
        r["passed"] = passed
        if passed:
            correct += 1

    metrics = {
        "pass@1": correct / total if total > 0 else 0.0,
        "n_correct": correct,
        "n_total": total,
    }
    return metrics


def main():
    args = parse_args()

    print("Loading HumanEval...")
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    problems = list(ds)
    if args.n_problems:
        problems = problems[:args.n_problems]
    print(f"  Loaded {len(problems)} problems")

    results = run_generation(args, problems)
    metrics = evaluate_results(results, problems)

    output = {
        "model_path":  args.model_path,
        "benchmark":   "humaneval",
        "n_problems":  len(problems),
        "metrics":     metrics,
        "raw_results": results,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    p1 = metrics["pass@1"]
    print(f"\n=== HumanEval Results ===")
    print(f"  pass@1: {p1:.1%}  ({metrics['n_correct']}/{metrics['n_total']})")
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
