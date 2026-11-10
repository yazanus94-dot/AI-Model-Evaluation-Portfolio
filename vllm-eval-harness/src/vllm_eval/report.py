"""Write results.json and a Markdown defect report."""
import json
import pathlib


def write_reports(report, out_dir, model_label):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["summary"]
    lines = [f"# LLM Evaluation Report - {model_label}", ""]
    lines += [
        f"- Cases: **{s['cases']}** (repeats per case: {s['repeats']})",
        f"- Pass rate: **{s['pass_rate']:.0%}**",
        f"- Severity-weighted score: **{s['weighted_score']:.0%}**",
        "- Throughput: n/a (simulated backend)" if "SIMULATED" in model_label else
        f"- Throughput: {s['generated_tokens']} tokens in {s['seconds']:.2f}s"
        + (f" ({s['tokens_per_second']:.0f} tok/s)" if s["tokens_per_second"] else ""),
        "",
        "## Pass rate by category",
        "",
        "| Category | Pass rate |",
        "|---|---|",
    ]
    lines += [f"| {c} | {v:.0%} |" for c, v in s["by_category"].items()]

    lines += ["", "## Failure taxonomy", ""]
    if s["failure_taxonomy"]:
        lines += ["| Failure type | Count |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in s["failure_taxonomy"].items()]
    else:
        lines.append("No failures.")

    if s["flaky_cases"]:
        lines += ["", "## Flaky cases (pass/fail changed between repeats)", ""]
        lines += [f"- {cid}" for cid in s["flaky_cases"]]

    lines += ["", "## Defects", ""]
    failed = [r for r in report["results"] if r["pass_rate"] < 1]
    if not failed:
        lines.append("None.")
    for r in failed:
        run = next(x for x in r["runs"] if not x["passed"])
        lines += [
            f"### {r['id']} ({r['category']}, {r['severity']})",
            f"- **Prompt:** {r['prompt']}",
            f"- **Model output:** {run['output'].strip()[:300]!r}",
            "- **Failed checks:**",
        ]
        lines += [f"  - `{d['type']}`: {d['detail']}" for d in run["checks"] if not d["passed"]]
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out
