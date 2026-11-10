# AI-Model Evaluation Portfolio

[![Tests](https://github.com/yazanus94-dot/AI-Model-Evaluation-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/yazanus94-dot/AI-Model-Evaluation-Portfolio/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Three independent, tested projects around one theme: **how do we know whether an AI model is good?**
Standard-library-first designs with mock backends mean the scoring and reporting logic is validated in
CI on a CPU runner, while the same interfaces run unmodified against real inference backends on GPUs.

| # | Project | Question it answers | Stack |
|---|---------|--------------------|-------|
| 1 | [jax-eval-metrics](jax-eval-metrics/) | Can evaluation metrics themselves be trusted? JAX implementations of Cohen's kappa, ECE, Brier, bootstrap CIs — every result cross-checked against a plain-NumPy reference (textbook kappa of 0.40 verified) | JAX (jit + vmap) |
| 2 | [sglang-judge-pipeline](sglang-judge-pipeline/) | Can we trust an LLM-as-judge? Pairwise judging with **answer-order swapping** to control for position bias, `sgl.select` constrained decoding (verdicts are always A/B/tie), decides contradictions | SGLang |
| 3 | [vllm-eval-harness](vllm-eval-harness/) | Does a model satisfy a rubric test suite? Severity-weighted scores, `contains/regex` deterministic checks, failure taxonomy, `--repeats N` flaky-case detection, tokens/s throughput | vLLM |

All three include `demo.sh` (GPU recording guides) and `VIDEO_GUIDE.md` with step-by-step repro instructions.

**Honesty note:** the repositories' real-backend runs require Linux + NVIDIA GPU (a free Colab T4 is enough);
CI exercises the mock backends, which return deliberately imperfect canned answers so the reports show
realistic failure behavior rather than trivially perfect ones.

## Quick start (no GPU needed)

```bash
# 1. JAX metrics toolkit (CPU wheel)
pip install -r jax-eval-metrics/requirements.txt
cd jax-eval-metrics && python -m pytest -q && python -m jax_eval.demo --out results.json

# 2. LLM-as-judge pipeline — MOCK mode, no server required
pip install pytest
cd sglang-judge-pipeline && PYTHONPATH=src python -m pytest -q
python run_judge.py --backend mock --label MockJudge --out out_mock

# 3. vLLM eval harness — MOCK mode
cd vllm-eval-harness && PYTHONPATH=src python -m pytest -q
python run_eval.py --backend mock --out out_mock
```

Real-backend runs:

```bash
python run_eval.py   --backend vllm   --model Qwen/Qwen2.5-0.5B-Instruct --chat --out out_greedy
python run_judge.py  --backend sglang --url http://localhost:30000 --label "$MODEL" --out out
```

## License

MIT — per-project `LICENSE` files identical to the root copy.

## Archives

`archive/` holds the original upload archives of the three projects, kept for provenance.
