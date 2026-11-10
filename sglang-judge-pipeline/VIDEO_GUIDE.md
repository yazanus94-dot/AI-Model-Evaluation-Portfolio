# Video recording guide - sglang-judge-pipeline

I cannot record videos for you, so follow this with any screen recorder (OBS Studio, QuickTime, Win+G).

## General rules
- Record your terminal at a large font (18pt+), 1080p, with notifications off.
- Speak in your own words. Use the bullets below as prompts, not a script to read.
- Show real output. Run the commands live (or at least do not edit the numbers).
- Keep each video under 3 minutes. Put your face or just voice - either is fine.
- Host on YouTube as **Unlisted** (or Loom), then paste the link in the "Demo video" section of this project's `README.md`.
  Do not commit large video files to Git. If you want the video inside the repo page, GitHub lets you drag an
  .mp4 (under 10 MB) into a README edit box, and it generates a hosted link.

## SGLang Judge Pipeline (GPU, run `sglang-judge-pipeline/demo.sh`)
1. (0:00) Explain position bias in LLM judges and why swapping answer order matters.
2. (0:25) Open `judge.py`: show `sgl.select(..., choices=["A","B","tie"])` - constrained decoding means no parsing.
3. (0:55) Open `pipeline.py`: explain the 3 reconciliation rules (consistent / tie as abstention / contradiction).
4. (1:25) Run `demo.sh`: server starts, judge runs.
5. (2:00) Open `out/report.md`: consistency, contradictions, first-position preference, agreement and kappa vs the human labels.
6. (2:40) Close: 12 pairs is small; say what you would do to scale it and try a larger judge model.

## Running on Google Colab (free GPU)
1. Runtime -> Change runtime type -> T4 GPU.
2. Upload the zip, unzip it, `!pip install vllm` (or `sglang[all]`) in a cell.
3. Run the commands from `demo.sh` in cells (SGLang: start the server with `&`, wait, then run the judge).
4. If a version mismatch breaks an API call, check the library's current docs and adjust that one call.

## Before you publish
- Run each demo yourself and paste your real numbers into each README's **Results** section.
- Be ready to explain every file in an interview. Read the code until you can walk through it unaided.
- Delete the `sample_output_MOCK` folders if you do not want the simulated samples in your repo.
