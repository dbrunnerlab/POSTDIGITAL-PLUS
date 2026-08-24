import numpy as np

from .base import Optimizer


class ParticleSwarmOptimization(Optimizer):
    """Particle Swarm Optimization with velocity clamping and boundary reflection.

    Population-based: each ask()/tell() cycle costs ``population_size``
    evaluations. O(n_dim) memory and update cost per particle.
    """

    def __init__(
        self,
        n_dim: int,
        bounds: tuple[float, float],
        population_size: int = 30,
        inertia: float = 0.7,
        cognitive: float = 1.4,
        social: float = 1.4,
        seed: int | None = None,
    ):
        super().__init__(n_dim)
        self._rng = np.random.default_rng(seed)

        self.pop_size = population_size
        self.lower, self.upper = bounds
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        span = self.upper - self.lower
        self.velocity_max = 0.2 * span

        self.position = self._rng.uniform(self.lower, self.upper, size=(self.pop_size, n_dim))
        self.velocity = self._rng.uniform(-self.velocity_max, self.velocity_max, size=(self.pop_size, n_dim))

        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = np.full(self.pop_size, np.inf)
        self.global_best_position = self.position[0].copy()
        self.global_best_fitness = np.inf

    def ask(self) -> np.ndarray:
        r1 = self._rng.random((self.pop_size, self.n_dim))
        r2 = self._rng.random((self.pop_size, self.n_dim))

        self.velocity = (
            self.inertia * self.velocity
            + self.cognitive * r1 * (self.personal_best_position - self.position)
            + self.social * r2 * (self.global_best_position - self.position)
        )
        self.velocity = np.clip(self.velocity, -self.velocity_max, self.velocity_max)

        self.position = self.position + self.velocity

        exceed_upper = self.position > self.upper
        exceed_lower = self.position < self.lower
        self.velocity[exceed_upper | exceed_lower] *= -1
        self.position = np.clip(self.position, self.lower, self.upper)

        return self.position

    def tell(self, fitness: np.ndarray) -> None:
        fitness = np.asarray(fitness, dtype=float)

        improved = fitness < self.personal_best_fitness
        self.personal_best_fitness[improved] = fitness[improved]
        self.personal_best_position[improved] = self.position[improved]

        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < self.global_best_fitness:
            self.global_best_fitness = float(fitness[best_idx])
            self.global_best_position = self.position[best_idx].copy()

        self._update_best(self.position, fitness)
