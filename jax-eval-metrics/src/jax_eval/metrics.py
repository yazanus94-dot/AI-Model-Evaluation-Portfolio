"""Evaluation metrics written in JAX (jit-compiled, vmap-able).

All label arrays are integer class ids in [0, num_classes).
"""
from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def accuracy(y_true, y_pred):
    return jnp.mean(y_true == y_pred)


@partial(jax.jit, static_argnames="num_classes")
def confusion_matrix(y_true, y_pred, num_classes):
    """Rows = true class, columns = predicted class."""
    idx = y_true * num_classes + y_pred
    return jnp.bincount(idx, length=num_classes * num_classes).reshape(num_classes, num_classes)


@partial(jax.jit, static_argnames="num_classes")
def macro_f1(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, num_classes).astype(jnp.float32)
    tp = jnp.diag(cm)
    precision = tp / jnp.maximum(cm.sum(axis=0), 1.0)
    recall = tp / jnp.maximum(cm.sum(axis=1), 1.0)
    f1 = 2 * precision * recall / jnp.maximum(precision + recall, 1e-12)
    return jnp.mean(f1)


@partial(jax.jit, static_argnames="num_classes")
def cohens_kappa(y_true, y_pred, num_classes):
    """Chance-corrected agreement: (p_o - p_e) / (1 - p_e)."""
    cm = confusion_matrix(y_true, y_pred, num_classes).astype(jnp.float32)
    n = cm.sum()
    p_o = jnp.trace(cm) / n
    p_e = jnp.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (n * n)
    return jnp.where(p_e >= 1.0, 1.0, (p_o - p_e) / jnp.maximum(1.0 - p_e, 1e-12))


@partial(jax.jit, static_argnames="n_bins")
def expected_calibration_error(confidences, correct, n_bins=10):
    """ECE: weighted mean |accuracy - confidence| over equal-width confidence bins."""
    bins = jnp.clip((confidences * n_bins).astype(jnp.int32), 0, n_bins - 1)
    counts = jnp.bincount(bins, length=n_bins)
    conf_sum = jnp.bincount(bins, weights=confidences, length=n_bins)
    acc_sum = jnp.bincount(bins, weights=correct.astype(jnp.float32), length=n_bins)
    safe = jnp.maximum(counts, 1)
    gap = jnp.abs(acc_sum / safe - conf_sum / safe)
    return jnp.sum(gap * counts) / confidences.shape[0]


@jax.jit
def brier_score(probs, y_true):
    """Multi-class Brier score: mean squared distance to the one-hot label."""
    onehot = jax.nn.one_hot(y_true, probs.shape[-1])
    return jnp.mean(jnp.sum((probs - onehot) ** 2, axis=-1))


def bootstrap_ci(metric_fn, key, arrays, n_boot=1000, alpha=0.05):
    """Percentile bootstrap CI. Resamples are drawn and scored in parallel with vmap.

    metric_fn takes the arrays (resampled the same way) and returns a scalar.
    Returns (mean, lower, upper) as Python floats.
    """
    n = arrays[0].shape[0]
    keys = jax.random.split(key, n_boot)

    def one_resample(k):
        idx = jax.random.randint(k, (n,), 0, n)
        return metric_fn(*[a[idx] for a in arrays])

    stats = jax.vmap(one_resample)(keys)
    lo, hi = jnp.quantile(stats, jnp.array([alpha / 2, 1 - alpha / 2]))
    return float(stats.mean()), float(lo), float(hi)
