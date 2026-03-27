#!/usr/bin/env python3
"""
Re-evaluate all existing Coder-1.5B eval JSONs using proper C++/Python execution.
Loads raw_results, re-runs code against test cases, updates metrics in-place.

Usage:  python scripts/re_evaluate.py [--bucket bucket_X_Y_Z]
"""

import json, os, sys, argparse, glob

# Reuse execution helpers from run_lcb_eval
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_lcb_eval import run_code, outputs_match, extract_code

PROJECT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(PROJECT, "results", "eval")


def load_problems():
    """Load LCB problems (with test cases) from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("livecodebench/code_generation_lite", trust_remote_code=True,
                      name="release_v5", split="test")
    return {p["question_id"]: p for p in ds}


def re_evaluate_file(path: str, prob_map: dict) -> dict:
    data = json.load(open(path))
    results = data["raw_results"]
    if not results:
        print(f"  skip {os.path.basename(path)} — no raw_results")
        return data

    correct = {"easy": 0, "medium": 0, "hard": 0, "total": 0}
    total   = {"easy": 0, "medium": 0, "hard": 0, "total": 0}
    langs   = {"cpp": 0, "python": 0, "unknown": 0}

    for r in results:
        diff = r.get("difficulty", "medium").lower()
        if diff not in ("easy", "medium", "hard"):
            diff = "medium"
        total[diff] += 1
        total["total"] += 1

        # Re-extract code and language from raw_output
        raw = r.get("raw_output", r.get("code", ""))
        code, lang = extract_code(raw)
        langs[lang] = langs.get(lang, 0) + 1

        prob = prob_map.get(r["question_id"], {})
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

        # Update result entry in-place
        r["code"] = code
        r["lang"] = lang

    metrics = {}
    for k in ("easy", "medium", "hard", "total"):
        n = total[k]
        metrics[f"pass@1_{k}"] = correct[k] / n if n > 0 else 0.0
        metrics[f"n_{k}"] = n

    data["metrics"] = metrics
    n = total["total"]
    print(f"  {os.path.basename(path)}: pass@1={correct['total']/n*100:.2f}%  "
          f"({correct['total']}/{n})  langs={langs}")
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=None, help="limit to one bucket, e.g. 2k_4k_1k")
    args = p.parse_args()

    print("Loading LCB problems...")
    prob_map = load_problems()
    print(f"Loaded {len(prob_map)} problems.")

    pattern = f"Coder-1.5B_bucket_{args.bucket}.json" if args.bucket else "Coder-1.5B_bucket_*.json"
    files = sorted(glob.glob(os.path.join(EVAL_DIR, pattern)))
    if not files:
        print("No matching eval files found.")
        return

    for path in files:
        print(f"Re-evaluating {os.path.basename(path)}...")
        data = re_evaluate_file(path, prob_map)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    print("\nDone. Run gen_results_table.py to update the summary.")


if __name__ == "__main__":
    main()
