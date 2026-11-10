#!/usr/bin/env bash
# Record this one: ~2 minutes. Runs on CPU.
set -e
export PYTHONPATH=src
echo "== 1. Unit tests: JAX metrics vs NumPy reference =="
python -m unittest discover -s tests -v
echo
echo "== 2. Train + evaluate + bootstrap CIs + calibration =="
python -m jax_eval.demo --spread 1.6 --out results.json
echo
echo "== 3. Harder problem (more class overlap) =="
python -m jax_eval.demo --spread 2.4 --out results_hard.json
