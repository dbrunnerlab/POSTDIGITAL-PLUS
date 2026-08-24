from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """Common ask/tell interface for every black-box optimizer in this library.

    ``ask()`` returns a batch of candidate solutions to evaluate, shaped
    ``(batch_size, n_dim)``. ``batch_size`` is algorithm-specific (population
    size for population-based methods, 2 for SPSA, etc.) and may vary between
    calls. ``tell(fitness)`` reports the objective value for each row of the
    most recent batch, in the same order, and updates internal state.

    Evaluation-count bookkeeping is intentionally left to the caller (see
    ``experiments/runner.py``): because every algorithm reports exactly the
    batch it wants evaluated, the harness can derive the "energy" metric
    (evaluation count) for free, without each optimizer tracking it itself.
    """

    def __init__(self, n_dim: int):
        if n_dim <= 0:
            raise ValueError("n_dim must be positive")
        self.n_dim = n_dim
        self.best_solution: np.ndarray | None = None
        self.best_fitness: float | None = None

    @abstractmethod
    def ask(self) -> np.ndarray:
        """Return the next batch of candidates to evaluate, shape (batch_size, n_dim)."""

    @abstractmethod
    def tell(self, fitness: np.ndarray) -> None:
        """Report fitness (lower is better) for the batch returned by the last ask()."""

    def _update_best(self, candidates: np.ndarray, fitness: np.ndarray) -> None:
        idx = int(np.argmin(fitness))
        if self.best_fitness is None or fitness[idx] < self.best_fitness:
            self.best_fitness = float(fitness[idx])
            self.best_solution = candidates[idx].copy()
