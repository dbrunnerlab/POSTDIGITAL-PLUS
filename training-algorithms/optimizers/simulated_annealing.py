import numpy as np

from .base import Optimizer


class SimulatedAnnealing(Optimizer):
    """Simulated Annealing with a Gaussian proposal and geometric cooling schedule.

    Sequential: exactly 1 evaluation per step, the same as a single SPSA
    coordinate would cost but with no gradient information at all -- included
    as the "pure random search with acceptance criterion" baseline. Accepts
    worsening moves with Metropolis probability exp(-delta / T), letting it
    escape local minima early (high T) while converging to greedy descent
    late (low T).
    """

    def __init__(
        self,
        n_dim: int,
        x_init: np.ndarray,
        step_size: float = 0.5,
        temperature_init: float = 1.0,
        cooling_rate: float = 0.995,
        temperature_min: float = 1e-6,
        seed: int | None = None,
    ):
        super().__init__(n_dim)
        self._rng = np.random.default_rng(seed)

        self.x = np.asarray(x_init, dtype=float).reshape(n_dim)
        self.current_fitness: float | None = None
        self.step_size = step_size
        self.temperature = temperature_init
        self.cooling_rate = cooling_rate
        self.temperature_min = temperature_min

        self._candidate: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        proposal = self.x + self._rng.standard_normal(self.n_dim) * self.step_size
        self._candidate = proposal.reshape(1, self.n_dim)
        return self._candidate

    def tell(self, fitness: np.ndarray) -> None:
        candidate_fitness = float(np.asarray(fitness).reshape(-1)[0])

        if self.current_fitness is None:
            accept = True
        else:
            delta = candidate_fitness - self.current_fitness
            accept = delta < 0 or self._rng.random() < np.exp(-delta / max(self.temperature, 1e-12))

        if accept:
            self.x = self._candidate[0]
            self.current_fitness = candidate_fitness

        self.temperature = max(self.temperature * self.cooling_rate, self.temperature_min)

        self._update_best(self._candidate, np.array([candidate_fitness]))
