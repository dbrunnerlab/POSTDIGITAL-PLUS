import numpy as np
import pytest

from experiments.objective_functions import sphere
from optimizers import (
    CMAES,
    PEPG,
    SPSA,
    FiniteDifferenceGradient,
    GeneticAlgorithm,
    NelderMead,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
)

N_DIM = 4
X_INIT = np.full(N_DIM, 3.0)
BOUNDS = (-5.0, 5.0)
BUDGET = 4000

FACTORIES = {
    "cma_es": lambda: CMAES(N_DIM, mean_init=X_INIT.copy(), seed=0),
    "pso": lambda: ParticleSwarmOptimization(N_DIM, bounds=BOUNDS, seed=0),
    "spsa": lambda: SPSA(N_DIM, x_init=X_INIT.copy(), seed=0),
    "pepg": lambda: PEPG(N_DIM, mean_init=X_INIT.copy(), seed=0),
    "finite_difference": lambda: FiniteDifferenceGradient(N_DIM, x_init=X_INIT.copy(), seed=0),
    "simulated_annealing": lambda: SimulatedAnnealing(N_DIM, x_init=X_INIT.copy(), seed=0),
    "genetic_algorithm": lambda: GeneticAlgorithm(N_DIM, bounds=BOUNDS, seed=0),
    "nelder_mead": lambda: NelderMead(N_DIM, x_init=X_INIT.copy()),
}


def run_to_budget(optimizer, objective_fn, budget):
    n_evaluations = 0
    while n_evaluations < budget:
        candidates = optimizer.ask()
        fitness = objective_fn(candidates)
        optimizer.tell(fitness)
        n_evaluations += candidates.shape[0]
    return optimizer


@pytest.mark.parametrize("name", FACTORIES)
def test_ask_returns_2d_array_of_correct_width(name):
    optimizer = FACTORIES[name]()
    candidates = optimizer.ask()
    assert candidates.ndim == 2
    assert candidates.shape[1] == N_DIM
    assert candidates.shape[0] >= 1


@pytest.mark.parametrize("name", FACTORIES)
def test_tell_accepts_ask_output_shape(name):
    optimizer = FACTORIES[name]()
    candidates = optimizer.ask()
    fitness = sphere(candidates)
    optimizer.tell(fitness)  # should not raise


@pytest.mark.parametrize("name", FACTORIES)
def test_converges_on_sphere(name):
    optimizer = FACTORIES[name]()
    initial_fitness = float(sphere(X_INIT[None, :])[0])

    optimizer = run_to_budget(optimizer, sphere, BUDGET)

    assert optimizer.best_fitness is not None
    assert optimizer.best_solution is not None
    assert optimizer.best_solution.shape == (N_DIM,)
    assert optimizer.best_fitness < initial_fitness * 0.5


@pytest.mark.parametrize("name", FACTORIES)
def test_best_fitness_matches_best_solution(name):
    optimizer = FACTORIES[name]()
    optimizer = run_to_budget(optimizer, sphere, BUDGET)

    recomputed = float(sphere(optimizer.best_solution[None, :])[0])
    assert recomputed == pytest.approx(optimizer.best_fitness, rel=1e-6, abs=1e-9)
