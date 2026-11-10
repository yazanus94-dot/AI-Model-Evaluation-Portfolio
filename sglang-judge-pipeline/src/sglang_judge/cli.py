import argparse
import json
import pathlib

from .judge import MockJudge, SGLangJudge
from .pipeline import judge_pairs, load_pairs, markdown_report, summarize


def main():
    ap = argparse.ArgumentParser(description="Pairwise LLM-as-judge with SGLang.")
    ap.add_argument("--backend", choices=["sglang", "mock"], default="sglang")
    ap.add_argument("--url", default="http://localhost:30000", help="SGLang server URL")
    ap.add_argument("--pairs", default="data/pairs.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--label", default=None, help="model name to show in the report")
    args = ap.parse_args()

    pairs = load_pairs(args.pairs)
    if args.backend == "mock":
        truth = {}
        for p in pairs:
            truth[(p["question"], p["answer_a"], p["answer_b"])] = p["human_label"]
            flip = {"A": "B", "B": "A", "tie": "tie"}[p["human_label"]]
            truth[(p["question"], p["answer_b"], p["answer_a"])] = flip
        judge, label = MockJudge(truth), "SIMULATED mock judge (not a real LLM)"
    else:
        judge, label = SGLangJudge(args.url), args.label or args.url

    rows = judge_pairs(judge, pairs)
    summary = summarize(rows)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "judgements.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    (out / "report.md").write_text(markdown_report(summary, rows, label), encoding="utf-8")
    print(f"consistency {summary['consistency_rate']:.0%} | "
          f"agreement single {summary['agreement_naive']:.0%} -> swapped {summary['agreement_debiased']:.0%} | "
          f"kappa {summary['kappa_naive']:.2f} -> {summary['kappa_debiased']:.2f}")
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
