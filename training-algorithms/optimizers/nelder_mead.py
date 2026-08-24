import numpy as np

from .base import Optimizer


class NelderMead(Optimizer):
    """Downhill simplex method (Nelder-Mead) with reflect/expand/contract/shrink.

    Nelder-Mead is inherently sequential -- each proposal depends on the
    previous evaluation's outcome -- so ask() always returns a single
    candidate (batch_size=1). It never builds a population or a gradient
    estimate, so its "energy" cost per accepted step is the lowest in this
    library, at the cost of poor scaling to high dimensions (the simplex has
    n_dim+1 vertices and shrink steps cost n_dim evaluations).
    """

    _INIT, _REFLECT, _EXPAND, _CONTRACT, _SHRINK = range(5)

    def __init__(
        self,
        n_dim: int,
        x_init: np.ndarray,
        step: float = 0.5,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        shrink_coef: float = 0.5,
    ):
        super().__init__(n_dim)

        x0 = np.asarray(x_init, dtype=float).reshape(n_dim)
        self.simplex = np.tile(x0, (n_dim + 1, 1))
        for i in range(n_dim):
            self.simplex[i + 1, i] += step
        self.fitness = np.full(n_dim + 1, np.nan)

        self.alpha, self.gamma, self.rho, self.shrink_coef = alpha, gamma, rho, shrink_coef

        self._state = self._INIT
        self._init_idx = 0
        self._shrink_idx = 1
        self._centroid: np.ndarray | None = None
        self._x_reflect: np.ndarray | None = None
        self._f_reflect: float | None = None
        self._contract_outside: bool | None = None
        self._pending_point: np.ndarray | None = None

    def _sort_simplex(self) -> None:
        order = np.argsort(self.fitness)
        self.simplex = self.simplex[order]
        self.fitness = self.fitness[order]

    def _start_reflection(self) -> np.ndarray:
        self._centroid = self.simplex[:-1].mean(axis=0)
        self._x_reflect = self._centroid + self.alpha * (self._centroid - self.simplex[-1])
        self._state = self._REFLECT
        return self._x_reflect

    def ask(self) -> np.ndarray:
        if self._state == self._INIT:
            self._pending_point = self.simplex[self._init_idx]
        elif self._state == self._SHRINK:
            self._pending_point = self.simplex[self._shrink_idx]
        elif self._state == self._REFLECT:
            self._pending_point = self._start_reflection()
        elif self._state == self._EXPAND:
            self._pending_point = self._centroid + self.gamma * (self._x_reflect - self._centroid)
        elif self._state == self._CONTRACT:
            if self._contract_outside:
                self._pending_point = self._centroid + self.rho * (self._x_reflect - self._centroid)
            else:
                self._pending_point = self._centroid + self.rho * (self.simplex[-1] - self._centroid)
        return self._pending_point.reshape(1, self.n_dim)

    def tell(self, fitness: np.ndarray) -> None:
        f = float(np.asarray(fitness).reshape(-1)[0])
        self._update_best(self._pending_point.reshape(1, self.n_dim), np.array([f]))

        if self._state == self._INIT:
            self.fitness[self._init_idx] = f
            self._init_idx += 1
            if self._init_idx == self.n_dim + 1:
                self._sort_simplex()
                self._state = self._REFLECT
            return

        if self._state == self._SHRINK:
            self.fitness[self._shrink_idx] = f
            self._shrink_idx += 1
            if self._shrink_idx == self.n_dim + 1:
                self._sort_simplex()
                self._state = self._REFLECT
            return

        if self._state == self._REFLECT:
            self._f_reflect = f
            if self.fitness[0] <= f < self.fitness[-2]:
                self.simplex[-1], self.fitness[-1] = self._x_reflect, f
                self._sort_simplex()
                self._state = self._REFLECT
            elif f < self.fitness[0]:
                self._state = self._EXPAND
            else:
                self._contract_outside = f < self.fitness[-1]
                self._state = self._CONTRACT
            return

        if self._state == self._EXPAND:
            if f < self._f_reflect:
                self.simplex[-1], self.fitness[-1] = self._pending_point, f
            else:
                self.simplex[-1], self.fitness[-1] = self._x_reflect, self._f_reflect
            self._sort_simplex()
            self._state = self._REFLECT
            return

        if self._state == self._CONTRACT:
            accept = f <= self._f_reflect if self._contract_outside else f < self.fitness[-1]
            if accept:
                self.simplex[-1], self.fitness[-1] = self._pending_point, f
                self._sort_simplex()
                self._state = self._REFLECT
            else:
                best = self.simplex[0]
                self.simplex[1:] = best + self.shrink_coef * (self.simplex[1:] - best)
                self._shrink_idx = 1
                self._state = self._SHRINK
            return
