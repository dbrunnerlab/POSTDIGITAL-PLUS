import numpy as np

from .base import Optimizer


class GeneticAlgorithm(Optimizer):
    """Real-coded Genetic Algorithm with tournament selection, blend crossover
    and Gaussian mutation.

    Population-based: each ask()/tell() cycle costs ``population_size``
    evaluations, O(n_dim) per individual for crossover/mutation.
    """

    def __init__(
        self,
        n_dim: int,
        bounds: tuple[float, float],
        population_size: int = 40,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        mutation_scale: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__(n_dim)
        self._rng = np.random.default_rng(seed)

        self.pop_size = population_size
        self.lower, self.upper = bounds
        self.n_elite = max(1, int(elite_fraction * population_size))
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = np.full(n_dim, mutation_scale * (self.upper - self.lower))

        self.population = self._rng.uniform(self.lower, self.upper, size=(self.pop_size, n_dim))

    def ask(self) -> np.ndarray:
        return self.population

    def _tournament_select(self, fitness: np.ndarray) -> np.ndarray:
        contenders = self._rng.integers(0, self.pop_size, size=self.tournament_size)
        winner = contenders[np.argmin(fitness[contenders])]
        return self.population[winner]

    def tell(self, fitness: np.ndarray) -> None:
        fitness = np.asarray(fitness, dtype=float)
        self._update_best(self.population, fitness)

        order = np.argsort(fitness)
        next_population = [self.population[i].copy() for i in order[: self.n_elite]]

        while len(next_population) < self.pop_size:
            parent_a = self._tournament_select(fitness)
            parent_b = self._tournament_select(fitness)

            if self._rng.random() < self.crossover_rate:
                blend = self._rng.uniform(-0.25, 1.25, size=self.n_dim)
                child = blend * parent_a + (1 - blend) * parent_b
            else:
                child = parent_a.copy()

            mutate_mask = self._rng.random(self.n_dim) < self.mutation_rate
            child[mutate_mask] += self._rng.standard_normal(np.sum(mutate_mask)) * self.mutation_scale[mutate_mask]

            child = np.clip(child, self.lower, self.upper)
            next_population.append(child)

        self.population = np.array(next_population[: self.pop_size])
