# SGLang LLM-as-Judge Pipeline

A pairwise **LLM-as-judge** pipeline on [SGLang](https://github.com/sgl-project/sglang) that compares two answers,
controls for **position bias**, and measures how well the judge agrees with **human labels**.

## Why this design
LLM judges are known to be influenced by *which answer is shown first*. This pipeline:
1. Uses SGLang **constrained decoding** (`sgl.select`, `regex=`) so the verdict is always `A`, `B` or `tie` - no parsing, no retries.
2. Judges every pair **twice, with the answer order swapped**, and reconciles:
   - same verdict both times -> accept
   - one decisive verdict + one `tie` -> accept the decisive one (tie treated as abstention)
   - `A` in one order and `B` in the other -> **contradiction** (the judge followed position, not content) -> `tie`
3. Reports **agreement with human labels, Cohen's kappa**, order-consistency, contradiction rate and first-position preference.

## SGLang features used
- `@sgl.function` programs with `sgl.system / user / assistant`
- `sgl.select(..., choices=[...])` and `sgl.gen(..., regex=...)` for constrained output
- `run_batch(...)` for parallel judging; shared system-prompt prefix benefits from SGLang's RadixAttention cache

## Demo video

_Add your video link here._

## Run it

```bash
pip install -r requirements.txt          # SGLang needs Linux + NVIDIA GPU (free Colab T4 works)
export PYTHONPATH=src
python -m unittest discover -s tests -v                      # no GPU needed
python run_judge.py --backend mock                           # dry run with a SIMULATED judge

# real run
python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --port 30000 &
python run_judge.py --backend sglang --url http://localhost:30000 --label Qwen2.5-1.5B-Instruct
```

Outputs: `out/judgements.json` and `out/report.md`.

## Data
`data/pairs.json`: 12 question/answer pairs (biology, CS, math, medicine, web) with a human label (`A`, `B`, `tie`).
Correct answers are placed in both positions so position bias is measurable. Extend the file with your own pairs - the more pairs, the more meaningful the kappa.

## Results

> Paste your real report here after running against a live SGLang server.
> `sample_output_MOCK/` shows the report format only; it was produced by a **simulated** judge.

## Limitations
- 12 pairs is a demo-size set; agreement and kappa from so few items have wide uncertainty.
- Small judge models are weak judges. Try a larger model and compare - that comparison makes a good write-up.
- SGLang's frontend API changes between releases; check the docs for your installed version if a call fails.
