import numpy as np

from .base import Optimizer


class FiniteDifferenceGradient(Optimizer):
    """Coordinate-wise (or randomly-subset) finite-difference gradient descent.

    Each step perturbs ``n_perturb`` coordinates individually (+/- epsilon),
    costing ``2 * n_perturb`` evaluations per step -- the most expensive
    "energy" cost in this library when n_perturb scales with n_dim, since
    unlike SPSA the cost grows with dimensionality rather than staying
    constant. Included as the "energy-inefficient but low-variance" baseline
    for the query-count comparison.
    """

    def __init__(
        self,
        n_dim: int,
        x_init: np.ndarray,
        n_perturb: int | None = None,
        learning_rate: float = 0.05,
        epsilon: float = 1e-3,
        seed: int | None = None,
    ):
        super().__init__(n_dim)
        self._rng = np.random.default_rng(seed)

        self.x = np.asarray(x_init, dtype=float).reshape(n_dim)
        self.n_perturb = n_perturb or n_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self._perturb_idx: np.ndarray | None = None
        self._last_candidates: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        self._perturb_idx = self._rng.choice(self.n_dim, size=self.n_perturb, replace=False)

        plus = np.tile(self.x, (self.n_perturb, 1))
        minus = np.tile(self.x, (self.n_perturb, 1))
        rows = np.arange(self.n_perturb)
        plus[rows, self._perturb_idx] += self.epsilon
        minus[rows, self._perturb_idx] -= self.epsilon

        self._last_candidates = np.vstack([plus, minus])
        return self._last_candidates

    def tell(self, fitness: np.ndarray) -> None:
        fitness = np.asarray(fitness, dtype=float)
        loss_plus = fitness[: self.n_perturb]
        loss_minus = fitness[self.n_perturb :]

        gradient = np.zeros(self.n_dim)
        gradient[self._perturb_idx] = (loss_plus - loss_minus) / (2 * self.epsilon)

        self.x = self.x - self.learning_rate * gradient

        self._update_best(self._last_candidates, fitness)
