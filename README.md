# AI Model Evaluation Portfolio

**AI Model Evaluation Portfolio** — Honest, tested evaluation tooling: JAX metrics cross-validated against NumPy, SGLang LLM-judge with position-bias control, and a vLLM rubric harness with failure taxonomy. Every project tested with mock backends in CI.

**Author:** [Yazan Aljabari](https://www.linkedin.com/in/yazan-aljabari-878202139/)  
**License:** [MIT](LICENSE)  
**Branch:** `main`

[![Tests](https://github.com/yazanus94-dot/AI-Model-Evaluation-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/yazanus94-dot/AI-Model-Evaluation-Portfolio/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)## Overview

Three independent, tested projects around one question: **how do we know whether an AI model is good?** Standard-library-first designs with mock backends mean the scoring and reporting logic is validated in CI on a CPU runner, while the same interfaces run unmodified against real inference backends on GPUs.

| # | Project | Question it answers | Stack |
|---|---------|--------------------|-------|
| 1 | [jax-eval-metrics](jax-eval-metrics/) | Can evaluation metrics themselves be trusted? JAX implementations of Cohen's kappa, ECE, Brier, bootstrap CIs — every result cross-checked against a plain-NumPy reference (textbook kappa of 0.40 verified) | JAX (jit + vmap) |
| 2 | [sglang-judge-pipeline](sglang-judge-pipeline/) | Can we trust an LLM-as-judge? Pairwise judging with **answer-order swapping** to control for position bias, `sgl.select` constrained decoding (verdicts are always A/B/tie), decides contradictions | SGLang |
| 3 | [vllm-eval-harness](vllm-eval-harness/) | Does a model satisfy a rubric test suite? Severity-weighted scores, `contains/regex` deterministic checks, failure taxonomy, `--repeats N` flaky-case detection, tokens/s throughput | vLLM |

All three include `demo.sh` (GPU recording guides) and `VIDEO_GUIDE.md` with step-by-step repro instructions.

**Honesty note:** the repositories' real-backend runs require Linux + NVIDIA GPU (a free Colab T4 is enough); CI exercises the mock backends, which return deliberately imperfect canned answers so the reports show realistic failure behavior rather than trivially perfect ones.

## Problem statement

AI evaluation is only as trustworthy as its measurement machinery. Common failures include: metrics implemented inconsistently across frameworks (a "kappa" that is really raw agreement), LLM judges whose verdicts shift with answer order or phrasing, and rubric harnesses that conflate deterministic check failures with genuine model weaknesses. Without cross-validated, bias-controlled, and honestly reported tooling, teams cannot distinguish a broken evaluator from a broken model.

## Objectives

- Provide **cross-validated metric implementations** (JAX vs NumPy reference) so scoring logic is provably correct.
- Build an **LLM-as-judge pipeline** with explicit position-bias controls and constrained decoding so verdicts are reliable and schema-stable.
- Build a **rubric evaluation harness** with severity-weighted scoring, deterministic checks, a failure taxonomy, and flaky-case detection.
- Keep mock backends as the CI default so every change is verified offline and reproducibly.
- Report results honestly: intervals, known limitations, and the distinction between mock-harness behavior and real-backend runs.## Features

- **Three self-contained, tested projects** around AI model evaluation.
- **Cross-validated metrics:** JAX implementations of Cohen's kappa, ECE, Brier, and bootstrap confidence intervals, every result cross-checked against a plain-NumPy reference implementation.
- **Bias-controlled LLM-as-judge:** pairwise judging with answer-order swapping to control for position bias; `sgl.select` constrained decoding so verdicts are always A/B/tie; explicit contradiction handling.
- **Rubric evaluation harness:** severity-weighted scores, `contains`/`regex` deterministic checks, a structured failure taxonomy, `--repeats N` flaky-case detection, and tokens/s throughput reporting.
- **Mock backends as the CI default:** CPU-runnable, deterministic, deliberately imperfect canned answers so reports show realistic failure behavior.
- **Real-backend compatibility:** the same interfaces run unmodified against real inference backends on GPUs.
- **Reproducibility tooling:** `demo.sh` GPU recording guides and `VIDEO_GUIDE.md` step-by-step repro instructions per project.
- **Automated tests and CI** on every push.

## Technologies

- **Language:** Python 3.12
- **JAX** (jit + vmap) for `jax-eval-metrics`
- **SGLang** (`sgl.select` constrained decoding) for `sglang-judge-pipeline`
- **vLLM** for `vllm-eval-harness`
- **NumPy** as the cross-validation reference
- **Testing:** pytest; **CI:** GitHub Actions
- **Data formats:** JSON, Markdown, CSV

## Project structure

```text
.
├── jax-eval-metrics/             # JAX metric implementations cross-validated against NumPy
├── sglang-judge-pipeline/        # Pairwise LLM-as-judge with position-bias control
├── vllm-eval-harness/            # Rubric evaluation harness with failure taxonomy
├── archive/                      # Original upload archives, kept for provenance
├── .github/workflows/ci.yml      # Continuous integration
└── README.md                     # This file
```

Each project directory typically contains: `src/`, `tests/`, `demo.sh`, `VIDEO_GUIDE.md`, `requirements.txt`, and per-project `README.md`.## Installation

### Prerequisites
- **Python 3.12** (3.11+ for mock modes)
- `pip` and virtual environment tooling
- **Real-backend runs additionally require:** Linux + NVIDIA GPU (a free Colab T4 is sufficient), plus SGLang/vLLM installed per each project's requirements.

### Setup

```bash
# Clone the repository
git clone https://github.com/yazanus94-dot/AI-Model-Evaluation-Portfolio.git
cd AI-Model-Evaluation-Portfolio

# Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate

pip install --upgrade pip
```

Then install per-project dependencies:

```bash
# 1. JAX metrics toolkit (CPU wheel)
pip install -r jax-eval-metrics/requirements.txt

# 2. LLM-as-judge pipeline (mock mode needs pytest only)
pip install pytest

# 3. vLLM eval harness (mock mode needs pytest only)
pip install pytest
```

> **Note:** `jax-eval-metrics` requires JAX (CPU wheel available on PyPI). The judge pipeline and vLLM harness run in mock mode with no server; real backends require SGLang/vLLM and a GPU.## Usage

### Quick start (no GPU needed)

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

### Real-backend runs

```bash
python run_eval.py   --backend vllm   --model Qwen/Qwen2.5-0.5B-Instruct --chat --out out_greedy
python run_judge.py  --backend sglang --url http://localhost:30000 --label "$MODEL" --out out
```

Each project also ships `demo.sh` (GPU recording guides) and `VIDEO_GUIDE.md`.## Methodology

### 1. jax-eval-metrics — can evaluation metrics themselves be trusted?
Metric implementations are re-derived in JAX (with `jit` + `vmap`) and independently cross-checked against a plain-NumPy reference implementation on identical inputs. Known-value checks (e.g. textbook kappa of 0.40) pin both implementations to the same ground truth, so a divergence signals a bug in either path rather than a property of the data. Bootstrap confidence intervals are computed with seeded resampling for reproducibility.

### 2. sglang-judge-pipeline — can we trust an LLM-as-judge?
Pairwise judging is performed with **answer-order swapping**: each comparison is evaluated in both orderings so that a systematic preference for position A or B is measurable rather than invisible. `sgl.select` constrained decoding forces verdicts to be exactly A/B/tie, eliminating free-text parsing and schema drift. Contradictory verdicts across orderings are surfaced as a first-class result.

### 3. vllm-eval-harness — does a model satisfy a rubric test suite?
Answers are scored against a rubric with severity-weighted categories. Deterministic checks (`contains`, `regex`) are evaluated exactly; failures are classified into a structured taxonomy. `--repeats N` re-runs each case to detect flaky (non-deterministic) failures, and tokens/s throughput is reported for cost/perf context.

**Honesty note:** mock backends return deliberately imperfect canned answers so reports show realistic failure behavior. Real-backend runs require Linux + NVIDIA GPU.## Results

> **Transparency:** CI exercises the **mock backends**, which return deliberately imperfect canned answers. Reports from mock runs demonstrate evaluator behavior and realistic failure detection — they are **not** measurements of any real model. Real-backend runs require Linux + NVIDIA GPU and produce user-executed results under `out_*` directories as documented in each project README.

**Included artifacts (repository files):**
- `jax-eval-metrics/`: cross-validation outputs and known-value checks (textbook kappa 0.40 verified) from `python -m jax_eval.demo --out results.json`.
- `sglang-judge-pipeline/`: mock-mode judging outputs (`out_mock/`) demonstrating position-bias control and verdict schema stability.
- `vllm-eval-harness/`: mock-mode rubric outputs (`out_mock/`) demonstrating severity-weighted scoring, failure taxonomy, and flaky-case detection.

**No real-model benchmark performance is claimed.** See each project's `README.md` and `VIDEO_GUIDE.md` for recorded GPU run instructions and example outputs.## Testing

```bash
# All projects (mock modes, no GPU needed)
cd jax-eval-metrics          && python -m pytest -q
cd sglang-judge-pipeline     && PYTHONPATH=src python -m pytest -q
cd vllm-eval-harness         && PYTHONPATH=src python -m pytest -q
```

- `jax-eval-metrics`: tests verify JAX implementations against the NumPy reference on identical inputs and pin known values (kappa 0.40).
- `sglang-judge-pipeline`: tests verify verdict schema (A/B/tie only), order-swap behavior, and contradiction detection.
- `vllm-eval-harness`: tests verify deterministic check behavior, severity weighting, failure taxonomy classification, and flaky-case detection with `--repeats`.
- **CI:** `.github/workflows/ci.yml` runs the mock-mode suites on push/PR.## Limitations

- **Mock-only CI results:** included test outputs come from mock backends with deliberately imperfect canned answers; they validate the evaluators, not any real model.
- **Real-backend dependency:** genuine LLM judging and rubric runs require Linux + NVIDIA GPU and SGLang/vLLM stacks; behavior may vary with backend versions and model choices.
- **Metric scope:** the JAX/NumPy cross-validation covers a limited set of common metrics (kappa, ECE, Brier, bootstrap CIs), not the full evaluation landscape.
- **Judge scope:** position-bias control addresses order effects; other biases (verbosity, framing, cultural) are not controlled, and constrained decoding fixes the verdict schema but not the underlying judgment quality.
- **Rubric scope:** deterministic checks are exact, but severity weights and taxonomy categories are portfolio-defined, not externally validated.## Future improvements

- Expand metric coverage (calibration, ranking, representation) with additional JAX/NumPy cross-checks.
- Add statistical tests for position-bias significance and inter-judge agreement beyond order swapping.
- Extend the rubric harness with metamorphic tests, reference-free behavioral checks, and cost/latency budgets.
- Publish recorded real-backend runs with pinned model IDs, dates, and configs for full provenance.
- Add containerized environments (Docker) for deterministic GPU reproductions.

## References

- Gwet, K. L. *Computing Interrater Reliability with the SAS System.* (Cohen's kappa reference)
- Calwell, B. & Zadron, P. *ECE / Brier score.* (calibration references)
- SGLang documentation: https://sgl-lang.github.io
- vLLM documentation: https://docs.vllm.ai
- JAX documentation: https://jax.readthedocs.io
- Project-local documentation and source files within each numbered directory.## License

This repository is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

**Copyright (c) 2026 Yazan Aljabari** — [LinkedIn](https://www.linkedin.com/in/yazan-aljabari-878202139/)