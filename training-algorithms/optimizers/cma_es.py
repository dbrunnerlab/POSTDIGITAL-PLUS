import numpy as np

from .base import Optimizer


class CMAES(Optimizer):
    """Covariance Matrix Adaptation Evolution Strategy.

    Standard rank-mu / rank-one update CMA-ES, following the strategy
    parameter defaults from Hansen's "The CMA Evolution Strategy: A Tutorial"
    (arXiv:1604.00772). Population-based: each ask()/tell() cycle costs
    ``population_size`` evaluations, and the covariance matrix update is
    O(n_dim^3) (periodic eigendecomposition) with O(n_dim^2) storage.
    """

    def __init__(
        self,
        n_dim: int,
        mean_init: np.ndarray,
        sigma_init: float = 0.5,
        population_size: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(n_dim)
        self._rng = np.random.default_rng(seed)

        self.mean = np.asarray(mean_init, dtype=float).reshape(n_dim)
        self.sigma = sigma_init

        self.pop_size = population_size or (4 + int(3 * np.log(n_dim)))
        self.n_select = self.pop_size // 2

        raw_weights = np.log(self.n_select + 0.5) - np.log(np.arange(1, self.n_select + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights**2)

        n = n_dim
        self.c_sigma = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (n + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
        self.c_1 = 2 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((n + 2) ** 2 + self.mu_eff),
        )
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        self.p_sigma = np.zeros(n)
        self.p_c = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self._eigen_stale_after = max(1, int(1 / ((self.c_1 + self.c_mu) * n * 10)))
        self._generations_since_eigen = 0
        self.generation = 0

        self._population: np.ndarray | None = None
        self._steps: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        z = self._rng.standard_normal((self.pop_size, self.n_dim))
        self._steps = z @ (self.B * self.D).T
        self._population = self.mean + self.sigma * self._steps
        return self._population

    def tell(self, fitness: np.ndarray) -> None:
        fitness = np.asarray(fitness, dtype=float)
        order = np.argsort(fitness)
        selected = order[: self.n_select]

        mean_old = self.mean.copy()
        self.mean = self.weights @ self._population[selected]

        y_w = (self.mean - mean_old) / self.sigma
        C_inv_sqrt = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff
        ) * (C_inv_sqrt @ y_w)

        self.generation += 1
        h_sigma_lhs = np.linalg.norm(self.p_sigma) / np.sqrt(
            1 - (1 - self.c_sigma) ** (2 * self.generation)
        )
        h_sigma = h_sigma_lhs < (1.4 + 2 / (self.n_dim + 1)) * self.chi_n
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * np.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * y_w

        y_selected = self._steps[selected]
        rank_mu = (self.weights[:, None] * y_selected).T @ y_selected
        delta_h = (1 - h_sigma) * self.c_c * (2 - self.c_c)
        self.C = (
            (1 - self.c_1 - self.c_mu) * self.C
            + self.c_1 * (np.outer(self.p_c, self.p_c) + delta_h * self.C)
            + self.c_mu * rank_mu
        )

        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_n - 1)
        )
        self.sigma = float(np.clip(self.sigma, 1e-12, 1e12))

        self._generations_since_eigen += 1
        if self._generations_since_eigen >= self._eigen_stale_after:
            self._generations_since_eigen = 0
            self.C = (self.C + self.C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.clip(eigvals, 1e-20, None)
            self.D = np.sqrt(eigvals)
            self.B = eigvecs

        self._update_best(self._population, fitness)
