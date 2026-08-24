import numpy as np

from .base import Optimizer


class PEPG(Optimizer):
    """Parameter-Exploring Policy Gradients with antithetic (mirrored) sampling.

    Population-based, O(n_dim) per-sample cost, no covariance matrix -- avoids
    CMA-ES's quadratic memory / cubic eigendecomposition cost, at the expense
    of only modeling a diagonal (axis-aligned) search distribution. Uses
    rank-based fitness shaping for robustness to reward scale, following
    Wierstra et al. (2014), "Natural Evolution Strategies", JMLR.
    """

    def __init__(
        self,
        n_dim: int,
        mean_init: np.ndarray,
        sigma_init: float = 0.5,
        population_size: int = 20,
        learning_rate_mu: float = 0.15,
        learning_rate_sigma: float = 0.10,
        sigma_decay: float = 0.999,
        sigma_limit: float = 1e-3,
        seed: int | None = None,
    ):
        super().__init__(n_dim)
        self._rng = np.random.default_rng(seed)

        self.batch_size = population_size // 2
        self.pop_size = 2 * self.batch_size

        self.mu = np.asarray(mean_init, dtype=float).reshape(n_dim)
        self.sigma = np.full(n_dim, sigma_init, dtype=float)

        self.lr_mu = learning_rate_mu
        self.lr_sigma = learning_rate_sigma
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit

        self._epsilon: np.ndarray | None = None
        self._candidates: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        self._epsilon = self._rng.standard_normal((self.batch_size, self.n_dim)) * self.sigma
        self._candidates = np.vstack([self.mu + self._epsilon, self.mu - self._epsilon])
        return self._candidates

    @staticmethod
    def _centered_ranks(x: np.ndarray) -> np.ndarray:
        ranks = np.empty(len(x))
        ranks[np.argsort(x)] = np.arange(len(x))
        return ranks / (len(x) - 1) - 0.5

    def tell(self, fitness: np.ndarray) -> None:
        fitness = np.asarray(fitness, dtype=float)
        # Fitness shaping expects "higher is better"; this library evaluates
        # minimization problems, so negate before ranking.
        shaped = self._centered_ranks(-fitness)

        reward_plus = shaped[: self.batch_size]
        reward_minus = shaped[self.batch_size :]

        mu_grad = self._epsilon.T @ (reward_plus - reward_minus) / self.batch_size
        self.mu = self.mu + self.lr_mu * mu_grad

        avg_reward = (reward_plus + reward_minus) / 2.0
        sigma_signal = (self._epsilon**2 - self.sigma**2) / self.sigma
        sigma_grad = sigma_signal.T @ avg_reward / self.batch_size
        self.sigma = self.sigma + self.lr_sigma * sigma_grad
        self.sigma = np.maximum(self.sigma * self.sigma_decay, self.sigma_limit)

        self._update_best(self._candidates, fitness)
