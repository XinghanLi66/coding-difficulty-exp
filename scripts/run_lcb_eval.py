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
    p.add_argument("--max_tokens", type=int, default=32768)
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


def extract_code(text: str) -> tuple[str, str]:
    """
    Extract code and detect language from model output.
    Returns (code, lang) where lang is 'cpp', 'python', or 'unknown'.
    Strips the <think>...</think> block first.
    """
    # Strip thinking block
    think_end = text.find("</think>")
    body = text[think_end + len("</think>"):].strip() if think_end != -1 else text

    # Try language-tagged blocks in priority order
    for tag, lang in [("cpp", "cpp"), ("c++", "cpp"), ("python", "python"), ("py", "python")]:
        m = re.search(rf"```{tag}\s*(.*?)```", body, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip(), lang

    # Untagged block
    m = re.search(r"```\s*(.*?)```", body, re.DOTALL)
    if m:
        code = m.group(1).strip()
        lang = "cpp" if "#include" in code else "python"
        return code, lang

    # No block — heuristic from body
    lang = "cpp" if "#include" in body else "python"
    return body.strip(), lang


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
            code, lang = extract_code(completion.text)
            results.append({
                "question_id": prob["question_id"],
                "difficulty":  prob.get("difficulty", "unknown"),
                "code":        code,
                "lang":        lang,
                "raw_output":  completion.text,
            })
    return results


def run_code(code: str, lang: str, stdin_data: str, timeout: float = 10.0) -> str | None:
    """
    Execute code against stdin_data.  Returns stdout string on success, None on error/timeout.
    lang: 'cpp' | 'python'
    """
    import subprocess, tempfile, os
    try:
        if lang == "cpp":
            with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as src:
                src.write(code.encode())
                src_path = src.name
            exe_path = src_path.replace(".cpp", "")
            try:
                r = subprocess.run(
                    ["g++", "-O2", "-o", exe_path, src_path],
                    capture_output=True, timeout=30
                )
                if r.returncode != 0:
                    return None
                r2 = subprocess.run(
                    [exe_path], input=stdin_data, capture_output=True,
                    text=True, timeout=timeout
                )
                return r2.stdout if r2.returncode == 0 else None
            finally:
                os.unlink(src_path)
                if os.path.exists(exe_path): os.unlink(exe_path)
        else:
            r = subprocess.run(
                [sys.executable, "-c", code],
                input=stdin_data, capture_output=True, text=True, timeout=timeout
            )
            return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def outputs_match(got: str, expected: str) -> bool:
    """Compare outputs ignoring trailing whitespace per line."""
    def norm(s):
        return [ln.rstrip() for ln in s.strip().splitlines()]
    return norm(got) == norm(expected)


def evaluate_results(results, problems):
    """Execute generated code against public test cases (C++ and Python)."""
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

        code = r.get("code", "")
        lang = r.get("lang", "unknown")
        if lang == "unknown":
            lang = "cpp" if "#include" in code else "python"

        raw_tc = prob.get("public_test_cases", "[]")
        try:
            test_cases = json.loads(raw_tc) if isinstance(raw_tc, str) else raw_tc
        except Exception:
            test_cases = []

        passed = False
        if test_cases and lang in ("cpp", "python"):
            passed = True
            for tc in test_cases:
                got = run_code(code, lang, tc.get("input", ""))
                if got is None or not outputs_match(got, tc.get("output", "")):
                    passed = False
                    break

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
