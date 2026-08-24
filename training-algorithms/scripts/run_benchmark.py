"""CLI entry point: runs the full benchmark suite (all algorithms x all
objective functions x all dimensions x several seeds) and writes the
energy / complexity / convergence plots and summary CSV to results/.

Usage:
    python scripts/run_benchmark.py [--trials N] [--max-evaluations N] [--dims D1 D2 ...]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import metrics, report
from experiments.objective_functions import OBJECTIVE_FUNCTIONS
from experiments.runner import run_benchmark_suite
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

BOUNDS = (-5.0, 5.0)
INIT_POINT_SCALE = 4.0

TARGETS = {
    "sphere": 1e-4,
    "rastrigin": 10.0,
    "rosenbrock": 1.0,
    "ackley": 1.0,
}


def build_optimizer_factories() -> dict:
    def init_point(n_dim: int, seed: int) -> np.ndarray:
        return np.random.default_rng(seed + 777).uniform(-INIT_POINT_SCALE, INIT_POINT_SCALE, n_dim)

    return {
        "cma_es": lambda n_dim, seed: CMAES(n_dim, mean_init=init_point(n_dim, seed), seed=seed),
        "pso": lambda n_dim, seed: ParticleSwarmOptimization(n_dim, bounds=BOUNDS, seed=seed),
        "spsa": lambda n_dim, seed: SPSA(n_dim, x_init=init_point(n_dim, seed), seed=seed),
        "pepg": lambda n_dim, seed: PEPG(n_dim, mean_init=init_point(n_dim, seed), seed=seed),
        "finite_difference": lambda n_dim, seed: FiniteDifferenceGradient(
            n_dim, x_init=init_point(n_dim, seed), seed=seed
        ),
        "simulated_annealing": lambda n_dim, seed: SimulatedAnnealing(
            n_dim, x_init=init_point(n_dim, seed), seed=seed
        ),
        "genetic_algorithm": lambda n_dim, seed: GeneticAlgorithm(n_dim, bounds=BOUNDS, seed=seed),
        "nelder_mead": lambda n_dim, seed: NelderMead(n_dim, x_init=init_point(n_dim, seed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-evaluations", type=int, default=4000)
    parser.add_argument("--dims", type=int, nargs="+", default=[5, 20, 50])
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    factories = build_optimizer_factories()

    print(
        f"Running {len(factories)} optimizers x {len(OBJECTIVE_FUNCTIONS)} objectives x "
        f"{len(args.dims)} dims x {args.trials} trials (budget={args.max_evaluations} evals/trial)..."
    )
    start = time.perf_counter()
    results = run_benchmark_suite(
        optimizer_factories=factories,
        objectives=OBJECTIVE_FUNCTIONS,
        dims=args.dims,
        n_trials=args.trials,
        max_evaluations=args.max_evaluations,
    )
    print(f"Suite finished in {time.perf_counter() - start:.1f}s, {len(results)} rows.")

    energy_df = metrics.energy_summary(results, TARGETS)
    complexity_df = metrics.complexity_scaling(results)

    for objective_name in OBJECTIVE_FUNCTIONS:
        for n_dim in args.dims:
            path = report.plot_convergence(results, objective_name, n_dim, args.output_dir)
            print(f"Wrote {path}")

    print(f"Wrote {report.plot_evals_to_target(energy_df, args.output_dir)}")
    print(f"Wrote {report.plot_complexity_scaling(complexity_df, args.output_dir)}")
    print(f"Wrote {report.write_summary_csv(energy_df, complexity_df, args.output_dir)}")


if __name__ == "__main__":
    main()
