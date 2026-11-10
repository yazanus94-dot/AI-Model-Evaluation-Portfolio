# Video recording guide - jax-eval-metrics

I cannot record videos for you, so follow this with any screen recorder (OBS Studio, QuickTime, Win+G).

## General rules
- Record your terminal at a large font (18pt+), 1080p, with notifications off.
- Speak in your own words. Use the bullets below as prompts, not a script to read.
- Show real output. Run the commands live (or at least do not edit the numbers).
- Keep each video under 3 minutes. Put your face or just voice - either is fine.
- Host on YouTube as **Unlisted** (or Loom), then paste the link in the "Demo video" section of this project's `README.md`.
  Do not commit large video files to Git. If you want the video inside the repo page, GitHub lets you drag an
  .mp4 (under 10 MB) into a README edit box, and it generates a hosted link.

## JAX Evaluation Metrics (CPU, run `jax-eval-metrics/demo.sh`)
1. (0:00) State the problem: accuracy alone hides calibration and uncertainty. Show the repo tree.
2. (0:20) Open `metrics.py`: point at `@jax.jit` + `static_argnames`, then `bootstrap_ci` using `jax.vmap`.
3. (1:00) Run the tests: JAX results match the NumPy reference, and the known kappa = 0.40 case passes.
4. (1:30) Run the demo: read out accuracy with its 95% CI, kappa, and ECE before vs after temperature scaling.
5. (2:15) Run the harder `--spread 2.4` case: explain why the CI widens / metrics drop.
6. (2:40) Close: what you would extend next (more metrics, real model outputs).

## Before you publish
- Run each demo yourself and paste your real numbers into each README's **Results** section.
- Be ready to explain every file in an interview. Read the code until you can walk through it unaided.
- Delete the `sample_output_MOCK` folders if you do not want the simulated samples in your repo.
