"""End-to-end demo: train -> evaluate with JAX metrics -> bootstrap CIs -> calibration."""
import argparse
import json
import pathlib

import jax
import jax.numpy as jnp

from . import metrics as M
from .data import make_blobs, train_val_test_split
from .train import fit, fit_temperature, logits_fn


def evaluate(logits, y, k, key, n_boot):
    probs = jax.nn.softmax(logits)
    pred = jnp.argmax(probs, axis=-1)
    conf = jnp.max(probs, axis=-1)
    correct = pred == y
    _, acc_lo, acc_hi = M.bootstrap_ci(M.accuracy, key, (y, pred), n_boot=n_boot)
    _, kap_lo, kap_hi = M.bootstrap_ci(lambda t, p: M.cohens_kappa(t, p, k), key, (y, pred), n_boot=n_boot)
    return {
        "accuracy": float(M.accuracy(y, pred)),
        "accuracy_ci95": [acc_lo, acc_hi],
        "macro_f1": float(M.macro_f1(y, pred, k)),
        "cohens_kappa": float(M.cohens_kappa(y, pred, k)),
        "kappa_ci95": [kap_lo, kap_hi],
        "ece": float(M.expected_calibration_error(conf, correct)),
        "brier": float(M.brier_score(probs, y)),
        "confusion_matrix": M.confusion_matrix(y, pred, k).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--spread", type=float, default=1.6, help="class overlap (higher = harder)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    k_data, k_init, k_boot = jax.random.split(key, 3)
    x, y = make_blobs(k_data, args.n, args.classes, spread=args.spread)
    (xtr, ytr), (xva, yva), (xte, yte) = train_val_test_split(x, y)

    params, losses = fit(k_init, xtr, ytr, args.classes, steps=args.steps)
    print(f"trained: loss {losses[0]:.3f} -> {losses[-1]:.3f} in {args.steps} jit-compiled SGD steps")

    test_logits = logits_fn(params, xte)
    before = evaluate(test_logits, yte, args.classes, k_boot, args.boot)

    temp = fit_temperature(logits_fn(params, xva), yva)
    after = evaluate(test_logits / temp, yte, args.classes, k_boot, args.boot)

    print(f"\nTest metrics (n={xte.shape[0]}, {args.boot} bootstrap resamples via vmap)")
    print(f"  accuracy   {before['accuracy']:.3f}  95% CI [{before['accuracy_ci95'][0]:.3f}, {before['accuracy_ci95'][1]:.3f}]")
    print(f"  macro F1   {before['macro_f1']:.3f}")
    print(f"  kappa      {before['cohens_kappa']:.3f}  95% CI [{before['kappa_ci95'][0]:.3f}, {before['kappa_ci95'][1]:.3f}]")
    print(f"  Brier      {before['brier']:.3f}")
    print(f"\nCalibration: fitted temperature T = {temp:.3f}")
    print(f"  ECE before {before['ece']:.4f}  ->  after {after['ece']:.4f}")

    pathlib.Path(args.out).write_text(json.dumps(
        {"args": vars(args), "temperature": temp, "before_scaling": before, "after_scaling": after}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
