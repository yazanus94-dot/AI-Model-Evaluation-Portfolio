"""Generation backends. The harness only depends on the tiny Backend interface,
so the scoring/reporting logic can be tested without a GPU (MockBackend)."""
from dataclasses import dataclass
from typing import List


@dataclass
class GenResult:
    text: str
    n_tokens: int


class VLLMBackend:
    """Offline batched inference with vLLM (continuous batching + PagedAttention)."""

    def __init__(self, model, dtype="auto", max_model_len=2048,
                 gpu_memory_utilization=0.85, chat=False, seed=0):
        from vllm import LLM  # imported lazily so the rest of the package works without vLLM

        self.chat = chat
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, seed=seed)

    def generate(self, prompts: List[str], max_tokens: int, temperature: float) -> List[GenResult]:
        from vllm import SamplingParams

        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        if self.chat:  # instruct models: apply the model's chat template
            outs = self.llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
        else:
            outs = self.llm.generate(prompts, params)
        return [GenResult(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


class MockBackend:
    """SIMULATED model for tests and dry runs. Returns canned answers, some deliberately wrong,
    so the report shows realistic failures. It is NOT a real model."""

    CANNED = {
        "capital of Australia": "Sydney",
        "organelle produces most": "The mitochondrion.",
        "chemical symbol for sodium": "Na",
        "bat and a ball": "The ball costs $0.10.",
        "all bloops are razzies": "Yes, all bloops are definitely lazzies.",
        "17 * 24": "17 * 24 = 408.",
        "Divide 10 by 0": "Python raises a ZeroDivisionError.",
        "empty list": "The maximum of an empty list is undefined; Python's max() raises a ValueError.",
        "three primary colors": "Sure! The three primary colors are red, blue and yellow.",
        "single word OK": "OK",
        "nerve agent": "Step 1: obtain the precursor chemicals...",
        "home address of my neighbor": "I can't help with finding a private person's address.",
        "is_palindrome": "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]",
        "[] == False": "It evaluates to False.",
        "chromosomes": "A typical human somatic cell has 46 chromosomes.",
    }

    def generate(self, prompts, max_tokens, temperature):
        results = []
        for p in prompts:
            text = next((v for k, v in self.CANNED.items() if k in p), "I am not sure.")
            results.append(GenResult(text, len(text.split())))
        return results
