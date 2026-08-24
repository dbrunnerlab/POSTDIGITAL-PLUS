"""Runs (optimizer, objective, dimension, seed) trials and records the
evaluation-count / wall-time / best-fitness trace needed for the energy,
complexity and convergence metrics.
"""

import time
from typing import Callable

import numpy as np
import pandas as pd

from optimizers.base import Optimizer

OptimizerFactory = Callable[[int, int], Optimizer]  # (n_dim, seed) -> Optimizer
ObjectiveFn = Callable[[np.ndarray], np.ndarray]


def run_trial(
    optimizer: Optimizer,
    objective_fn: ObjectiveFn,
    max_evaluations: int,
) -> pd.DataFrame:
    """Runs one optimizer to a fixed evaluation budget, returning a per-step trace."""
    records = []
    n_evaluations = 0
    best_so_far = np.inf
    start = time.perf_counter()

    while n_evaluations < max_evaluations:
        candidates = optimizer.ask()
        fitness = objective_fn(candidates)
        optimizer.tell(fitness)

        n_evaluations += candidates.shape[0]
        best_so_far = min(best_so_far, float(np.min(fitness)))
        elapsed = time.perf_counter() - start

        records.append(
            {"evaluations": n_evaluations, "best_fitness": best_so_far, "wall_time": elapsed}
        )

    return pd.DataFrame.from_records(records)


def run_benchmark_suite(
    optimizer_factories: dict[str, OptimizerFactory],
    objectives: dict[str, ObjectiveFn],
    dims: list[int],
    n_trials: int,
    max_evaluations: int,
) -> pd.DataFrame:
    """Runs every (optimizer, objective, dim) combination for n_trials seeds.

    Returns a long-form DataFrame with columns:
    optimizer, objective, n_dim, trial, evaluations, best_fitness, wall_time.
    """
    all_traces = []

    for optimizer_name, factory in optimizer_factories.items():
        for objective_name, objective_fn in objectives.items():
            for n_dim in dims:
                for trial in range(n_trials):
                    seed = 1000 * trial + n_dim
                    optimizer = factory(n_dim, seed)
                    trace = run_trial(optimizer, objective_fn, max_evaluations)
                    trace["optimizer"] = optimizer_name
                    trace["objective"] = objective_name
                    trace["n_dim"] = n_dim
                    trace["trial"] = trial
                    all_traces.append(trace)

    return pd.concat(all_traces, ignore_index=True)
