import numpy as np

from .base import Optimizer


class SPSA(Optimizer):
    """Simultaneous Perturbation Stochastic Approximation.

    Estimates a descent direction from exactly 2 evaluations per step,
    regardless of n_dim -- the cheapest "energy" cost per step in this
    library, at the price of noisier gradient estimates in high dimensions.
    Uses the classic decaying step-size / perturbation-size schedule
    (a_k = a / (k+1)^alpha, c_k = c / (k+1)^gamma).
    """

    def __init__(
        self,
        n_dim: int,
        x_init: np.ndarray,
        a: float = 0.1,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        seed: int | None = None,
    ):
        super().__init__(n_dim)
        self._rng = np.random.default_rng(seed)

        self.x = np.asarray(x_init, dtype=float).reshape(n_dim)
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.k = 0

        self._delta: np.ndarray | None = None
        self._c_k: float = c
        self._last_candidates: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        self._delta = self._rng.choice([-1.0, 1.0], size=self.n_dim)
        self._c_k = self.c / (self.k + 1) ** self.gamma
        x_plus = self.x + self._c_k * self._delta
        x_minus = self.x - self._c_k * self._delta
        self._last_candidates = np.stack([x_plus, x_minus])
        return self._last_candidates

    def tell(self, fitness: np.ndarray) -> None:
        fitness = np.asarray(fitness, dtype=float)
        loss_plus, loss_minus = fitness[0], fitness[1]

        gradient_estimate = (loss_plus - loss_minus) / (2 * self._c_k * self._delta)

        a_k = self.a / (self.k + 1) ** self.alpha
        self.x = self.x - a_k * gradient_estimate
        self.k += 1

        self._update_best(self._last_candidates, fitness)
