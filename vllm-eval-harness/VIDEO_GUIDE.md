# Video recording guide - vllm-eval-harness

I cannot record videos for you, so follow this with any screen recorder (OBS Studio, QuickTime, Win+G).

## General rules
- Record your terminal at a large font (18pt+), 1080p, with notifications off.
- Speak in your own words. Use the bullets below as prompts, not a script to read.
- Show real output. Run the commands live (or at least do not edit the numbers).
- Keep each video under 3 minutes. Put your face or just voice - either is fine.
- Host on YouTube as **Unlisted** (or Loom), then paste the link in the "Demo video" section of this project's `README.md`.
  Do not commit large video files to Git. If you want the video inside the repo page, GitHub lets you drag an
  .mp4 (under 10 MB) into a README edit box, and it generates a hosted link.

## vLLM Evaluation Harness (GPU, run `vllm-eval-harness/demo.sh`)
1. (0:00) Explain: a rubric-based harness that produces a defect report, like RLHF/eval work.
2. (0:20) Show `data/test_suite.json` (categories, severities) and `checks.py` in one sentence each.
3. (0:50) Run the tests (no GPU needed), then start the real run; mention all prompts go in one batched `generate` call.
4. (1:30) Open `out_greedy/report.md`: pass rate, weighted score, failure taxonomy, one real defect with its failed check.
5. (2:10) Show the `--repeats 5 --temperature 0.7` run and point at any flaky cases.
6. (2:40) Close: limits of string checks, and that the SGLang project covers LLM-as-judge.

## Running on Google Colab (free GPU)
1. Runtime -> Change runtime type -> T4 GPU.
2. Upload the zip, unzip it, `!pip install vllm` (or `sglang[all]`) in a cell.
3. Run the commands from `demo.sh` in cells (SGLang: start the server with `&`, wait, then run the judge).
4. If a version mismatch breaks an API call, check the library's current docs and adjust that one call.

## Before you publish
- Run each demo yourself and paste your real numbers into each README's **Results** section.
- Be ready to explain every file in an interview. Read the code until you can walk through it unaided.
- Delete the `sample_output_MOCK` folders if you do not want the simulated samples in your repo.
