import argparse

from .backends import MockBackend, VLLMBackend
from .report import write_reports
from .runner import evaluate, load_suite


def main():
    ap = argparse.ArgumentParser(description="Evaluate an LLM on a rubric test suite.")
    ap.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--suite", default="data/test_suite.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--chat", action="store_true", help="apply the chat template (instruct models)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=1, help="repeat each prompt N times (use temperature > 0)")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    suite = load_suite(args.suite)
    if args.backend == "mock":
        backend, label = MockBackend(), "SIMULATED mock model (not a real LLM)"
    else:
        backend = VLLMBackend(args.model, max_model_len=args.max_model_len,
                              gpu_memory_utilization=args.gpu_mem, chat=args.chat)
        label = args.model

    report = evaluate(backend, suite, args.max_tokens, args.temperature, args.repeats)
    out = write_reports(report, args.out, label)
    s = report["summary"]
    print(f"pass rate {s['pass_rate']:.0%} | weighted {s['weighted_score']:.0%} | "
          f"{s['generated_tokens']} tokens in {s['seconds']:.2f}s")
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
