"""Judges. SGLangJudge uses the SGLang frontend language with *constrained decoding*, so the
verdict is guaranteed to be one of the allowed labels - no output parsing or retries.

A judge exposes:  verdict(question, answer_a, answer_b) -> "A" | "B" | "tie"
"""
import random

SYSTEM = ("You are a strict, impartial evaluator. Judge only factual accuracy and whether the answer "
          "responds to the question. Ignore length, tone and the order in which answers are shown.")


def _build_functions():
    import sglang as sgl  # lazy import: tests and mock runs do not need SGLang installed

    @sgl.function
    def pairwise(s, question, answer_a, answer_b):
        s += sgl.system(SYSTEM)
        s += sgl.user("Question: " + question + "\n\nAnswer A: " + answer_a + "\n\nAnswer B: " + answer_b +
                      "\n\nWhich answer is better? Reply with exactly one of: A, B, tie.")
        s += sgl.assistant(sgl.select("verdict", choices=["A", "B", "tie"]))

    @sgl.function
    def rubric(s, question, answer):
        s += sgl.system(SYSTEM)
        s += sgl.user("Question: " + question + "\n\nAnswer: " + answer +
                      "\n\nScore the answer from 1 (wrong) to 5 (fully correct). Reply with one digit.")
        s += sgl.assistant(sgl.gen("score", max_tokens=1, regex=r"[1-5]"))

    return sgl, pairwise, rubric


class SGLangJudge:
    """Talks to a running SGLang server:
        python -m sglang.launch_server --model-path <model> --port 30000
    """

    def __init__(self, url="http://localhost:30000", num_threads=8):
        self.sgl, self._pairwise, self._rubric = _build_functions()
        self.sgl.set_default_backend(self.sgl.RuntimeEndpoint(url))
        self.num_threads = num_threads

    def verdict(self, question, answer_a, answer_b):
        state = self._pairwise.run(question=question, answer_a=answer_a, answer_b=answer_b)
        return state["verdict"]

    def verdict_batch(self, triples):
        """Batched judging - SGLang shares the common prefix (system prompt) via RadixAttention."""
        args = [{"question": q, "answer_a": a, "answer_b": b} for q, a, b in triples]
        states = self._pairwise.run_batch(args, num_threads=self.num_threads, progress_bar=False)
        return [s["verdict"] for s in states]

    def score(self, question, answer):
        return int(self._rubric.run(question=question, answer=answer)["score"])


class MockJudge:
    """SIMULATED judge for tests/dry runs (NOT a real model). It knows the true label but is noisy
    and has a configurable bias toward whichever answer is shown first (position bias)."""

    def __init__(self, truth, noise=0.15, first_position_bias=0.25, seed=0):
        self.truth = truth  # {(question, answer_a, answer_b): "A"|"B"|"tie"}
        self.noise, self.bias = noise, first_position_bias
        self.rng = random.Random(seed)

    def verdict(self, question, answer_a, answer_b):
        true = self.truth[(question, answer_a, answer_b)]
        r = self.rng.random()
        if r < self.bias:
            return "A"  # lazily prefers the first answer
        if r < self.bias + self.noise:
            return self.rng.choice(["A", "B", "tie"])
        return true

    def verdict_batch(self, triples):
        return [self.verdict(*t) for t in triples]
