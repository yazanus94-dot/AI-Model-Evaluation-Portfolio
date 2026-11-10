"""Synthetic, overlapping Gaussian-blob classification data (reproducible via a PRNG key)."""
import jax


def make_blobs(key, n=3000, num_classes=3, dim=2, spread=1.6):
    k_c, k_y, k_x = jax.random.split(key, 3)
    centers = 3.0 * jax.random.normal(k_c, (num_classes, dim))
    y = jax.random.randint(k_y, (n,), 0, num_classes)
    x = centers[y] + spread * jax.random.normal(k_x, (n, dim))
    return x, y


def train_val_test_split(x, y, fracs=(0.6, 0.2, 0.2)):
    n = x.shape[0]
    a = int(n * fracs[0])
    b = a + int(n * fracs[1])
    return (x[:a], y[:a]), (x[a:b], y[a:b]), (x[b:], y[b:])
