"""Pure-Python agreement metrics (no dependencies)."""
from collections import Counter


def accuracy(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohens_kappa(a, b):
    n = len(a)
    p_o = accuracy(a, b)
    ca, cb = Counter(a), Counter(b)
    p_e = sum((ca[l] / n) * (cb[l] / n) for l in set(a) | set(b))
    return 1.0 if p_e >= 1.0 else (p_o - p_e) / (1.0 - p_e)
