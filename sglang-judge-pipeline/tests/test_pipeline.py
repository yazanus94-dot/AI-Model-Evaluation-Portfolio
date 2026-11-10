"""Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests -v"""
import unittest

from sglang_judge.judge import MockJudge
from sglang_judge.metrics import accuracy, cohens_kappa
from sglang_judge.pipeline import FLIP, judge_pairs, load_pairs, summarize

PAIRS = load_pairs("data/pairs.json")


def truth_map(pairs):
    t = {}
    for p in pairs:
        t[(p["question"], p["answer_a"], p["answer_b"])] = p["human_label"]
        t[(p["question"], p["answer_b"], p["answer_a"])] = FLIP[p["human_label"]]
    return t


class PerfectJudge:
    def __init__(self, truth): self.truth = truth
    def verdict_batch(self, triples): return [self.truth[t] for t in triples]


class AlwaysA:
    def verdict_batch(self, triples): return ["A"] * len(triples)


class TestMetrics(unittest.TestCase):
    def test_kappa_known_value(self):
        a = ["x"] * 20 + ["x"] * 5 + ["y"] * 10 + ["y"] * 15
        b = ["x"] * 20 + ["y"] * 5 + ["x"] * 10 + ["y"] * 15
        self.assertAlmostEqual(cohens_kappa(a, b), 0.4, places=6)
        self.assertAlmostEqual(accuracy(a, b), 0.7)

    def test_kappa_perfect(self):
        self.assertAlmostEqual(cohens_kappa(["a", "b", "a"], ["a", "b", "a"]), 1.0)


class TestPipeline(unittest.TestCase):
    def test_data_is_well_formed(self):
        self.assertEqual(len({p["id"] for p in PAIRS}), len(PAIRS))
        labels = {p["human_label"] for p in PAIRS}
        self.assertTrue({"A", "B"} <= labels)  # both positions occur, so position bias is detectable

    def test_perfect_judge_is_consistent_and_correct(self):
        rows = judge_pairs(PerfectJudge(truth_map(PAIRS)), PAIRS)
        s = summarize(rows)
        self.assertEqual(s["consistency_rate"], 1.0)
        self.assertEqual(s["agreement_debiased"], 1.0)
        self.assertAlmostEqual(s["kappa_debiased"], 1.0)

    def test_always_a_judge_is_flagged_as_position_biased(self):
        rows = judge_pairs(AlwaysA(), PAIRS)
        s = summarize(rows)
        self.assertEqual(s["first_position_rate"], 1.0)
        self.assertEqual(s["consistency_rate"], 0.0)  # every pair flips when order is swapped
        self.assertEqual(s["contradiction_rate"], 1.0)
        self.assertEqual(set(s["verdict_counts"]), {"tie"})

    def test_tie_plus_decisive_is_accepted(self):
        class Abstainer:  # 1st call (original order): decisive & right; 2nd call (swapped): 'tie'
            def __init__(self): self.calls = 0
            def verdict_batch(self, triples):
                self.calls += 1
                return ["A" if self.calls == 1 else "tie"] * len(triples)
        pairs = [{"id": "x", "question": "q", "answer_a": "good", "answer_b": "bad", "human_label": "A"}]
        rows = judge_pairs(Abstainer(), pairs)
        self.assertEqual(rows[0]["final"], "A")
        self.assertFalse(rows[0]["consistent"])
        self.assertFalse(rows[0]["contradiction"])

    def test_swap_reconciliation_does_not_hurt_biased_mock(self):
        judge = MockJudge(truth_map(PAIRS), noise=0.1, first_position_bias=0.3, seed=1)
        s = summarize(judge_pairs(judge, PAIRS))
        self.assertGreaterEqual(s["pairs"], 12)
        self.assertGreater(s["first_position_rate"], 0.3)  # bias shows up in the metric
        self.assertLessEqual(len(s["inconsistent_ids"]), s["pairs"])


if __name__ == "__main__":
    unittest.main()
