# vLLM Evaluation Harness

A batch **LLM evaluation harness** built on [vLLM](https://github.com/vllm-project/vllm). It runs a rubric-based
test suite against any Hugging Face model, scores every response with deterministic checks, and writes a
**defect report** with a failure taxonomy - the same workflow used in RLHF and model-evaluation work.

## Features
- **Rubric test suite** (`data/test_suite.json`): 15 cases across factual, reasoning, edge-case, instruction-following, safety and code categories, each with a severity (`critical` / `major` / `minor`).
- **Deterministic checks** (`src/vllm_eval/checks.py`): `contains_any`, `contains_all`, `not_contains_any`, `regex`, `max_words`.
- **Severity-weighted score** - a failed safety case hurts more than a failed trivia case.
- **Failure taxonomy** - missing expected content / forbidden or unsafe content / format violation.
- **Consistency testing** - `--repeats N` at `temperature > 0` flags *flaky* cases whose pass/fail changes between samples.
- **Throughput** (tokens/s) from vLLM's continuous batching: all prompts go through one `generate` call.
- **Backend interface** - the scoring and reporting logic is tested with a `MockBackend`, so tests need no GPU.

## Demo video

_Add your video link here._

## Run it

```bash
pip install -r requirements.txt     # vLLM needs Linux + NVIDIA GPU (free Colab T4 works)
export PYTHONPATH=src

python -m unittest discover -s tests -v                       # no GPU needed
python run_eval.py --backend mock                             # dry run with a SIMULATED model
python run_eval.py --backend vllm --model Qwen/Qwen2.5-0.5B-Instruct --chat
python run_eval.py --backend vllm --model Qwen/Qwen2.5-0.5B-Instruct --chat --temperature 0.7 --repeats 5
```

Outputs: `out/results.json` (every output + every check) and `out/report.md` (summary + defects).

## Layout

| File | Role |
|---|---|
| `backends.py` | `VLLMBackend` (`LLM`, `SamplingParams`, `llm.chat`) and `MockBackend` |
| `checks.py` | rubric check implementations + failure classes |
| `runner.py` | batching, scoring, severity weighting, flaky detection, summaries |
| `report.py` | JSON + Markdown reports |
| `cli.py` / `run_eval.py` | command line interface |

## Results

> Paste your real report summary here after running on a GPU (model name, pass rate, weighted score, top failures, tokens/s).
> `sample_output_MOCK/` shows the report format. It was generated with the **simulated** backend and says nothing about any real model.

## Notes
- The suite uses simple string/regex checks, so it is fast and reproducible but cannot judge open-ended quality. See the companion `sglang-judge-pipeline` project for LLM-as-judge scoring.
- vLLM's API evolves between releases; if `llm.chat` or a constructor argument is missing in your version, check the vLLM docs for your installed version.
