"""Standard black-box optimization benchmark functions (all global minimum 0).

Each function accepts a batch of candidates shaped (batch_size, n_dim) and
returns fitness shaped (batch_size,), so the harness can evaluate an entire
ask() batch in one vectorized call.
"""

import numpy as np


def sphere(x: np.ndarray) -> np.ndarray:
    """Convex, unimodal. Global minimum: 0 at the origin."""
    return np.sum(x**2, axis=-1)


def rastrigin(x: np.ndarray, scale: float = 5.0) -> np.ndarray:
    """Highly multimodal with regularly-spaced local minima. Global minimum: 0 at the origin."""
    n = x.shape[-1]
    z = scale * x
    return 10 * n + np.sum(z**2 - 10 * np.cos(2 * np.pi * z), axis=-1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """Narrow curved valley, non-convex. Global minimum: 0 at (1, 1, ..., 1)."""
    x_i, x_next = x[..., :-1], x[..., 1:]
    return np.sum(100 * (x_next - x_i**2) ** 2 + (1 - x_i) ** 2, axis=-1)


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> np.ndarray:
    """Nearly flat outer region with a deep central well. Global minimum: 0 at the origin."""
    n = x.shape[-1]
    sum_sq = np.sum(x**2, axis=-1)
    sum_cos = np.sum(np.cos(c * x), axis=-1)
    return (
        -a * np.exp(-b * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + a
        + np.e
    )


OBJECTIVE_FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ackley": ackley,
}
