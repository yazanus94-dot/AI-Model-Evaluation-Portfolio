"""Plain-NumPy reference implementations, used by the tests to verify the JAX versions."""
import numpy as np


def confusion_matrix(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred, k):
    cm = confusion_matrix(y_true, y_pred, k).astype(float)
    f1s = []
    for c in range(k):
        tp = cm[c, c]
        prec = tp / max(cm[:, c].sum(), 1.0)
        rec = tp / max(cm[c, :].sum(), 1.0)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def cohens_kappa(y_true, y_pred, k):
    cm = confusion_matrix(y_true, y_pred, k).astype(float)
    n = cm.sum()
    p_o = np.trace(cm) / n
    p_e = float((cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n))
    return 1.0 if p_e >= 1.0 else float((p_o - p_e) / (1.0 - p_e))


def ece(confidences, correct, n_bins=10):
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.clip((confidences * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.sum() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(total / len(confidences))


def brier(probs, y_true):
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(probs.shape[1])[np.asarray(y_true)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
