"""Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests -v"""
import unittest

import numpy as np

from jax_eval import reference as R

try:
    import jax
    import jax.numpy as jnp
    from jax_eval import metrics as M
    HAVE_JAX = True
except ImportError:  # reference-only run
    HAVE_JAX = False


def labels_from_matrix(cm):
    yt, yp = [], []
    for t, row in enumerate(cm):
        for p, count in enumerate(row):
            yt += [t] * count
            yp += [p] * count
    return np.array(yt), np.array(yp)


class TestReference(unittest.TestCase):
    def test_known_kappa(self):
        # Textbook example: p_o=0.70, p_e=0.50 -> kappa=0.40
        yt, yp = labels_from_matrix([[20, 5], [10, 15]])
        self.assertAlmostEqual(R.cohens_kappa(yt, yp, 2), 0.4, places=6)

    def test_perfect_agreement(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        self.assertAlmostEqual(R.cohens_kappa(y, y, 3), 1.0)
        self.assertAlmostEqual(R.accuracy(y, y), 1.0)

    def test_ece_calibrated_bin(self):
        conf = np.array([0.8] * 10)
        correct = np.array([1] * 8 + [0] * 2)
        self.assertAlmostEqual(R.ece(conf, correct), 0.0, places=6)

    def test_ece_overconfident(self):
        conf = np.array([0.9] * 10)
        correct = np.array([1] * 5 + [0] * 5)
        self.assertAlmostEqual(R.ece(conf, correct), 0.4, places=6)

    def test_brier_perfect(self):
        probs = np.eye(3)[[0, 1, 2]]
        self.assertAlmostEqual(R.brier(probs, [0, 1, 2]), 0.0)


@unittest.skipUnless(HAVE_JAX, "jax not installed")
class TestJaxMatchesReference(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.k = 4
        self.yt = rng.integers(0, self.k, 500)
        self.yp = np.where(rng.random(500) < 0.7, self.yt, rng.integers(0, self.k, 500))
        raw = rng.random((500, self.k))
        self.probs = raw / raw.sum(1, keepdims=True)

    def test_accuracy(self):
        got = float(M.accuracy(jnp.array(self.yt), jnp.array(self.yp)))
        self.assertAlmostEqual(got, R.accuracy(self.yt, self.yp), places=5)

    def test_macro_f1(self):
        got = float(M.macro_f1(jnp.array(self.yt), jnp.array(self.yp), self.k))
        self.assertAlmostEqual(got, R.macro_f1(self.yt, self.yp, self.k), places=4)

    def test_kappa(self):
        got = float(M.cohens_kappa(jnp.array(self.yt), jnp.array(self.yp), self.k))
        self.assertAlmostEqual(got, R.cohens_kappa(self.yt, self.yp, self.k), places=4)

    def test_known_kappa(self):
        yt, yp = labels_from_matrix([[20, 5], [10, 15]])
        self.assertAlmostEqual(float(M.cohens_kappa(jnp.array(yt), jnp.array(yp), 2)), 0.4, places=5)

    def test_ece(self):
        conf = self.probs.max(1).astype(np.float32)
        correct = self.probs.argmax(1) == self.yt
        got = float(M.expected_calibration_error(jnp.array(conf), jnp.array(correct)))
        self.assertAlmostEqual(got, R.ece(conf, correct), places=4)

    def test_brier(self):
        got = float(M.brier_score(jnp.array(self.probs, dtype=jnp.float32), jnp.array(self.yt)))
        self.assertAlmostEqual(got, R.brier(self.probs, self.yt), places=4)

    def test_bootstrap_ci_brackets_point_estimate(self):
        _, lo, hi = M.bootstrap_ci(M.accuracy, jax.random.PRNGKey(0),
                                   (jnp.array(self.yt), jnp.array(self.yp)), n_boot=500)
        point = R.accuracy(self.yt, self.yp)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)
        self.assertLess(hi - lo, 0.2)


if __name__ == "__main__":
    unittest.main()
