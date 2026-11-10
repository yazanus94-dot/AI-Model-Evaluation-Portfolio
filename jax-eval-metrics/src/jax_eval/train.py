"""Softmax regression trained with jax.grad + jit, plus post-hoc temperature scaling."""
import jax
import jax.numpy as jnp


def init_params(key, dim, num_classes):
    return {"W": 0.01 * jax.random.normal(key, (dim, num_classes)), "b": jnp.zeros(num_classes)}


def logits_fn(params, x):
    return x @ params["W"] + params["b"]


def loss_fn(params, x, y):
    log_probs = jax.nn.log_softmax(logits_fn(params, x))
    return -jnp.mean(log_probs[jnp.arange(y.shape[0]), y])


@jax.jit
def sgd_step(params, x, y, lr):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return params, loss


def fit(key, x, y, num_classes, steps=300, lr=0.5):
    params = init_params(key, x.shape[1], num_classes)
    losses = []
    for _ in range(steps):
        params, loss = sgd_step(params, x, y, lr)
        losses.append(float(loss))
    return params, losses


def fit_temperature(logits, y, steps=200, lr=0.1):
    """Learn a single scalar T minimising validation NLL of softmax(logits / T)."""

    def nll(log_t):
        lp = jax.nn.log_softmax(logits / jnp.exp(log_t))
        return -jnp.mean(lp[jnp.arange(y.shape[0]), y])

    grad = jax.jit(jax.grad(nll))
    log_t = jnp.array(0.0)
    for _ in range(steps):
        log_t = log_t - lr * grad(log_t)
    return float(jnp.exp(log_t))
