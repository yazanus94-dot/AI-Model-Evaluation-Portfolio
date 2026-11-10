#!/usr/bin/env bash
# Record this one: ~3 minutes. Needs a Linux machine with an NVIDIA GPU (a free Colab T4 works).
set -e
export PYTHONPATH=src
echo "== 1. Tests (no GPU needed) =="
python -m unittest discover -s tests -v
echo
echo "== 2. Real run on vLLM: greedy, one pass =="
python run_eval.py --backend vllm --model Qwen/Qwen2.5-0.5B-Instruct --chat --out out_greedy
echo
echo "== 3. Consistency check: sample 5x at temperature 0.7 to find flaky cases =="
python run_eval.py --backend vllm --model Qwen/Qwen2.5-0.5B-Instruct --chat --temperature 0.7 --repeats 5 --out out_sampled
echo
echo "Open out_greedy/report.md and out_sampled/report.md"
