# JAX Evaluation Metrics Toolkit

A small, tested library of **model-evaluation metrics written in JAX** (jit-compiled and `vmap`-parallel),
plus an end-to-end demo that trains a classifier with `jax.grad`, evaluates it with confidence intervals,
and fixes overconfidence with temperature scaling.

Why this matters for AI evaluation work: accuracy alone hides problems. This toolkit reports
**chance-corrected agreement (Cohen's kappa)**, **calibration (ECE, Brier)**, and **uncertainty (bootstrap CIs)**.

## What's inside

| File | What it does |
|---|---|
| `src/jax_eval/metrics.py` | accuracy, confusion matrix, macro-F1, Cohen's kappa, ECE, Brier score, bootstrap CI (via `jax.vmap`) |
| `src/jax_eval/reference.py` | plain-NumPy versions used as ground truth in the tests |
| `src/jax_eval/train.py` | softmax regression with `jax.value_and_grad` + `jax.jit`; temperature scaling |
| `src/jax_eval/demo.py` | train -> evaluate -> bootstrap -> calibrate, writes `results.json` |
| `tests/test_metrics.py` | JAX results must match NumPy references; known textbook kappa (0.40) |

## Demo video

_Add your video link here._

## Run it

```bash
pip install -r requirements.txt          # CPU JAX is enough
export PYTHONPATH=src
python -m unittest discover -s tests -v  # verify the metrics
python -m jax_eval.demo                  # run the demo
```

## JAX techniques demonstrated
- `jax.jit` with `static_argnames` for shape-determining arguments (`num_classes`, `n_bins`)
- `jax.vmap` to run 1000 bootstrap resamples in parallel instead of a Python loop
- `jax.random` explicit PRNG keys (`split`) for reproducibility
- `jax.grad` / `value_and_grad` for training and for fitting the temperature parameter
- Pytrees (`tree_map`) for parameter updates

## Results

> Paste the output of your own run here after you have run `demo.sh` (accuracy, kappa with CI, ECE before/after).

## Design notes
- Metrics return JAX scalars so they compose with `jit`/`vmap`; `bootstrap_ci` converts to Python floats at the end.
- Kappa is computed from the confusion matrix with `bincount` (no Python loops), so it is jittable.
- Tests compare against an independent NumPy implementation rather than re-using the same code path.
