"""Position-debiased pairwise judging and agreement with human labels."""
import json
from collections import Counter

from .metrics import accuracy, cohens_kappa

FLIP = {"A": "B", "B": "A", "tie": "tie"}


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    for p in pairs:
        assert p["human_label"] in ("A", "B", "tie"), p["id"]
    return pairs


def swapped(pair):
    return {**pair, "answer_a": pair["answer_b"], "answer_b": pair["answer_a"]}


def judge_pairs(judge, pairs):
    """Judge each pair in both orders and reconcile (verdicts mapped back to the original orientation):
      - same verdict both times            -> accept it (consistent)
      - one decisive verdict + one 'tie'   -> accept the decisive one (soft: tie treated as abstention)
      - A in one order, B in the other     -> contradiction: the judge followed position, not content -> 'tie'
    """
    triples = [(p["question"], p["answer_a"], p["answer_b"]) for p in pairs]
    triples_sw = [(p["question"], p["answer_b"], p["answer_a"]) for p in pairs]
    forward = judge.verdict_batch(triples)
    backward_raw = judge.verdict_batch(triples_sw)
    backward = [FLIP[v] for v in backward_raw]

    rows = []
    for p, f, b, b_raw in zip(pairs, forward, backward, backward_raw):
        consistent = f == b
        contradiction = {f, b} == {"A", "B"}
        final = f if consistent else (b if f == "tie" else f if b == "tie" else "tie")
        rows.append({
            "id": p["id"], "human": p["human_label"],
            "forward": f, "backward_mapped": b,
            "raw_first_position_choice": (f == "A") + (b_raw == "A"),
            "consistent": consistent,
            "contradiction": contradiction,
            "final": final,
        })
    return rows


def summarize(rows):
    human = [r["human"] for r in rows]
    final = [r["final"] for r in rows]
    naive = [r["forward"] for r in rows]  # what you would get with a single, un-swapped call
    n_calls = 2 * len(rows)
    return {
        "pairs": len(rows),
        "consistency_rate": sum(r["consistent"] for r in rows) / len(rows),
        "contradiction_rate": sum(r["contradiction"] for r in rows) / len(rows),
        "agreement_on_consistent": (lambda c: accuracy([r["final"] for r in c], [r["human"] for r in c]) if c else None)(
            [r for r in rows if r["consistent"]]),
        "first_position_rate": sum(r["raw_first_position_choice"] for r in rows) / n_calls,
        "agreement_naive": accuracy(naive, human),
        "agreement_debiased": accuracy(final, human),
        "kappa_naive": cohens_kappa(naive, human),
        "kappa_debiased": cohens_kappa(final, human),
        "verdict_counts": dict(Counter(final)),
        "inconsistent_ids": [r["id"] for r in rows if not r["consistent"]],
        "contradiction_ids": [r["id"] for r in rows if r["contradiction"]],
    }


def markdown_report(summary, rows, label):
    s = summary
    lines = [
        f"# LLM-as-Judge Report - {label}", "",
        f"- Pairs judged: **{s['pairs']}** (each judged twice, answer order swapped)",
        f"- Order-consistency (same verdict in both orders): **{s['consistency_rate']:.0%}**",
        f"- Contradictions (A in one order, B in the other): **{s['contradiction_rate']:.0%}**",
        f"- Agreement with humans on order-consistent pairs only: **{s['agreement_on_consistent']:.0%}**"
        if s["agreement_on_consistent"] is not None else "- No order-consistent pairs",
        f"- First-position preference: **{s['first_position_rate']:.0%}** of raw verdicts chose the answer shown first "
        "(an unbiased judge should be near the share of pairs where the first answer is truly better)",
        "",
        "| Method | Agreement with humans | Cohen's kappa |", "|---|---|---|",
        f"| Single call (no swap) | {s['agreement_naive']:.0%} | {s['kappa_naive']:.2f} |",
        f"| Swap + reconcile | {s['agreement_debiased']:.0%} | {s['kappa_debiased']:.2f} |",
        "", "## Order-sensitive pairs", "",
    ]
    bad = [r for r in rows if not r["consistent"]]
    if not bad:
        lines.append("None.")
    for r in bad:
        kind = "CONTRADICTION" if r["contradiction"] else "soft"
        lines.append(f"- `{r['id']}` [{kind}]: forward={r['forward']}, swapped={r['backward_mapped']}, "
                     f"final={r['final']}, human={r['human']}")
    return "\n".join(lines) + "\n"
