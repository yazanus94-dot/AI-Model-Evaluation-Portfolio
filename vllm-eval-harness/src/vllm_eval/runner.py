"""Run a test suite against a backend and score it."""
import json
import time
from collections import defaultdict

from .checks import FAILURE_CLASS, run_check

SEVERITY_WEIGHT = {"critical": 3, "major": 2, "minor": 1}


def load_suite(path):
    with open(path, encoding="utf-8") as f:
        suite = json.load(f)
    seen = set()
    for case in suite:
        if case["id"] in seen:
            raise ValueError(f"duplicate case id {case['id']}")
        seen.add(case["id"])
        if case.get("severity", "major") not in SEVERITY_WEIGHT:
            raise ValueError(f"bad severity in {case['id']}")
    return suite


def score_case(case, text):
    """Case passes only if every check passes. Score = fraction of checks passed."""
    details = []
    for check in case["checks"]:
        ok, why = run_check(check, text)
        details.append({"type": check["type"], "passed": ok, "detail": why})
    passed = all(d["passed"] for d in details)
    return passed, sum(d["passed"] for d in details) / len(details), details


def evaluate(backend, suite, max_tokens=128, temperature=0.0, repeats=1):
    """Batch every prompt (x repeats) in a single backend call, then score."""
    prompts = [c["prompt"] for c in suite for _ in range(repeats)]
    t0 = time.perf_counter()
    gens = backend.generate(prompts, max_tokens, temperature)
    elapsed = time.perf_counter() - t0

    results, total_tokens = [], 0
    for i, case in enumerate(suite):
        runs = []
        for r in range(repeats):
            g = gens[i * repeats + r]
            total_tokens += g.n_tokens
            passed, frac, details = score_case(case, g.text)
            runs.append({"output": g.text, "passed": passed, "check_fraction": frac, "checks": details})
        pass_rate = sum(r["passed"] for r in runs) / repeats
        results.append({
            "id": case["id"], "category": case["category"], "severity": case.get("severity", "major"),
            "prompt": case["prompt"], "runs": runs, "pass_rate": pass_rate,
            "flaky": 0 < pass_rate < 1,
        })

    return {"results": results, "summary": summarize(results, elapsed, total_tokens, repeats)}


def summarize(results, elapsed, total_tokens, repeats):
    by_cat = defaultdict(lambda: [0.0, 0])
    weighted, weight_total = 0.0, 0
    failures = defaultdict(int)
    for r in results:
        w = SEVERITY_WEIGHT[r["severity"]]
        weighted += w * r["pass_rate"]
        weight_total += w
        by_cat[r["category"]][0] += r["pass_rate"]
        by_cat[r["category"]][1] += 1
        if r["pass_rate"] < 1:
            for run in r["runs"]:
                for d in run["checks"]:
                    if not d["passed"]:
                        failures[FAILURE_CLASS[d["type"]]] += 1
    n = len(results)
    return {
        "cases": n,
        "repeats": repeats,
        "pass_rate": sum(r["pass_rate"] for r in results) / n,
        "weighted_score": weighted / weight_total,
        "by_category": {c: v[0] / v[1] for c, v in sorted(by_cat.items())},
        "failure_taxonomy": dict(sorted(failures.items(), key=lambda kv: -kv[1])),
        "flaky_cases": [r["id"] for r in results if r["flaky"]],
        "seconds": elapsed,
        "generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else None,
    }
