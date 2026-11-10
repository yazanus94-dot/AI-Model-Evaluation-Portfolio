#!/usr/bin/env bash
# Record this one: ~3 minutes. Needs a Linux machine with an NVIDIA GPU (a free Colab T4 works).
set -e
export PYTHONPATH=src
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
echo "== 1. Tests (no GPU needed) =="
python -m unittest discover -s tests -v
echo
echo "== 2. Start the SGLang server in the background =="
python -m sglang.launch_server --model-path "$MODEL" --port 30000 > server.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID" EXIT
until curl -s localhost:30000/health >/dev/null; do sleep 3; done
echo "server ready"
echo
echo "== 3. Run the pairwise judge with answer-order swapping =="
python run_judge.py --backend sglang --url http://localhost:30000 --label "$MODEL" --out out
echo
echo "Open out/report.md"
