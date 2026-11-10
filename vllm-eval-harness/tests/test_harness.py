"""Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests -v"""
import pathlib
import tempfile
import unittest

from vllm_eval.backends import GenResult, MockBackend
from vllm_eval.checks import run_check
from vllm_eval.report import write_reports
from vllm_eval.runner import evaluate, load_suite, score_case

SUITE = pathlib.Path(__file__).parent.parent / "data" / "test_suite.json"


class TestChecks(unittest.TestCase):
    def test_contains_any_case_insensitive(self):
        self.assertTrue(run_check({"type": "contains_any", "values": ["canberra"]}, "It is CANBERRA.")[0])

    def test_not_contains_any(self):
        self.assertFalse(run_check({"type": "not_contains_any", "values": ["step 1"]}, "Step 1: do it")[0])

    def test_regex_anchored(self):
        chk = {"type": "regex", "pattern": "^\\s*ok\\W*\\s*$"}
        self.assertTrue(run_check(chk, "OK.")[0])
        self.assertFalse(run_check(chk, "OK, here is more text")[0])

    def test_max_words(self):
        self.assertFalse(run_check({"type": "max_words", "value": 3}, "one two three four")[0])

    def test_unknown_check_raises(self):
        with self.assertRaises(ValueError):
            run_check({"type": "nope"}, "x")


class TestSuite(unittest.TestCase):
    def test_suite_loads_and_ids_unique(self):
        suite = load_suite(SUITE)
        self.assertEqual(len(suite), len({c["id"] for c in suite}))
        self.assertGreaterEqual(len(suite), 10)

    def test_every_case_has_checks(self):
        for case in load_suite(SUITE):
            self.assertTrue(case["checks"], case["id"])


class TestScoring(unittest.TestCase):
    def test_partial_score(self):
        case = {"checks": [{"type": "contains_any", "values": ["a"]}, {"type": "max_words", "value": 1}]}
        passed, frac, _ = score_case(case, "a b c")
        self.assertFalse(passed)
        self.assertAlmostEqual(frac, 0.5)


class TestEndToEndMock(unittest.TestCase):
    def test_mock_run_has_expected_failures(self):
        report = evaluate(MockBackend(), load_suite(SUITE))
        failed = {r["id"] for r in report["results"] if r["pass_rate"] < 1}
        # The mock deliberately gets these wrong
        self.assertTrue({"fact-capital", "reason-batball", "instr-colors", "safety-weapon"} <= failed)
        self.assertNotIn("reason-syllogism", failed)  # "Yes" is the correct answer
        s = report["summary"]
        self.assertLess(s["pass_rate"], 1.0)
        self.assertIn("forbidden content / unsafe output", s["failure_taxonomy"])
        self.assertIn("format or instruction violation", s["failure_taxonomy"])
        self.assertLess(s["by_category"]["safety"], 1.0)

    def test_weighted_score_penalises_critical_failures(self):
        report = evaluate(MockBackend(), load_suite(SUITE))
        self.assertLess(report["summary"]["weighted_score"], report["summary"]["pass_rate"])

    def test_flaky_detection(self):
        class Flip:
            def __init__(self): self.n = 0
            def generate(self, prompts, max_tokens, temperature):
                out = []
                for _ in prompts:
                    self.n += 1
                    out.append(GenResult("Canberra" if self.n % 2 else "Sydney", 1))
                return out
        suite = [c for c in load_suite(SUITE) if c["id"] == "fact-capital"]
        report = evaluate(Flip(), suite, repeats=4)
        self.assertEqual(report["summary"]["flaky_cases"], ["fact-capital"])
        self.assertAlmostEqual(report["results"][0]["pass_rate"], 0.5)

    def test_report_files_written(self):
        report = evaluate(MockBackend(), load_suite(SUITE))
        with tempfile.TemporaryDirectory() as d:
            out = write_reports(report, d, "mock")
            self.assertTrue((out / "report.md").read_text().startswith("# LLM Evaluation Report"))
            self.assertTrue((out / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
