# Black-box optimizer library

A library of black-box (zeroth-order) optimization algorithms with a unified
`ask()`/`tell()` interface, plus a benchmark harness comparing them on three
axes:

- **Energy efficiency** -- evaluations needed to reach a target fitness.
  Physical/non-digital hardware forward-passes are the expensive, slow
  resource in this project, so evaluation count is used as the energy proxy
  throughout.
- **Complexity** -- documented Big-O per algorithm (memory and per-step cost)
  plus empirically measured wall-clock scaling with problem dimension.
- **Convergence performance** -- best-fitness-vs-evaluations curves, averaged
  over multiple random seeds.

## Why a unified interface

Every algorithm implements the same two methods:

```python
candidates = optimizer.ask()      # np.ndarray, shape (batch_size, n_dim)
fitness = objective_fn(candidates)  # evaluate however you like (simulation, hardware, ...)
optimizer.tell(fitness)           # np.ndarray, shape (batch_size,), lower is better
```

`batch_size` varies by algorithm (population size for CMA-ES/PSO/PEPG/GA, 2
for SPSA, `2*n_perturb` for finite-difference, 1 for simulated annealing, up
to 2 for Nelder-Mead's reflect/expand/contract steps), but the contract is
always the same. Because the harness evaluates exactly what `ask()` returns,
evaluation-count bookkeeping (the energy metric) falls out for free instead
of being tracked separately per algorithm.

## Algorithms

| Algorithm | File | Evals/step | Notes |
|---|---|---|---|
| CMA-ES | `optimizers/cma_es.py` | population | Full covariance adaptation; O(n²) memory, periodic O(n³) eigendecomposition |
| PSO | `optimizers/pso.py` | population | Velocity + inertia, boundary reflection |
| PEPG | `optimizers/pepg.py` | population | Antithetic sampling, diagonal covariance only -- avoids CMA-ES's cubic cost |
| Genetic Algorithm | `optimizers/genetic_algorithm.py` | population | Tournament selection, blend crossover, Gaussian mutation |
| SPSA | `optimizers/spsa.py` | 2 (constant) | Cheapest per-step energy cost regardless of `n_dim` |
| Finite-difference gradient | `optimizers/finite_difference.py` | `2*n_perturb` | Cost scales with dimension when `n_perturb ≈ n_dim` -- the energy-inefficient baseline |
| Simulated Annealing | `optimizers/simulated_annealing.py` | 1 | Metropolis acceptance, geometric cooling |
| Nelder-Mead | `optimizers/nelder_mead.py` | 1-2 | Sequential simplex; no gradient or population, poor high-dimensional scaling |

Reimplemented from standard references (Hansen's CMA-ES tutorial,
arXiv:1604.00772; Wierstra et al., "Natural Evolution Strategies", JMLR 2014)
rather than ported from any single source, so the interface could be made
consistent across all eight algorithms.

**Not included:** ADAM. It requires true gradients (backprop), which isn't
available in a black-box/hardware-in-the-loop setting -- out of scope for
this library by design, not an oversight.

## Benchmark harness

- `experiments/objective_functions.py` -- Sphere (convex), Rastrigin
  (multimodal), Rosenbrock (curved valley), Ackley (flat outer region, sharp
  central well).
- `experiments/runner.py` -- runs one (optimizer, objective, dimension, seed)
  trial to a fixed evaluation budget, recording evaluations / best-fitness /
  wall-time at every step.
- `experiments/metrics.py` -- energy (evaluations-to-target, success rate),
  convergence curves (mean/std over trials), complexity scaling (time per
  evaluation vs. dimension).
- `experiments/report.py` -- renders the plots and summary CSV under
  `results/`.

## Usage

```bash
pip install -r requirements.txt
pytest tests/                       # unit tests: ask/tell shapes + sphere convergence sanity checks
python scripts/run_benchmark.py     # full suite -> results/*.png, results/summary.csv
```

Options: `--trials`, `--max-evaluations`, `--dims`, `--output-dir` (see
`python scripts/run_benchmark.py --help`).

## Results

`results/` contains one committed snapshot from an actual local run (8
algorithms x 4 objectives x dims {5, 20, 50} x 5 seeds each):

- `convergence_<objective>_<dim>d.png` -- best-fitness-vs-evaluations per
  objective/dimension, one line per algorithm.
- `evals_to_target.png` -- mean evaluations-to-target per algorithm/objective
  (the energy efficiency comparison).
- `complexity_scaling.png` -- mean wall-clock time per evaluation vs. `n_dim`
  per algorithm (empirical complexity).
- `summary.csv` -- the underlying numbers for both plots.

Re-run `scripts/run_benchmark.py` to regenerate against different budgets,
dimensions, or objective functions (e.g. a physical-hardware forward pass
instead of a synthetic test function).

## Credits

API design and algorithm selection were inspired by
[ASkalli/learning_strategies](https://github.com/ASkalli/learning_strategies)
and its associated paper (arXiv:2503.16943), which benchmarked CMA-ES, PSO,
SPSA, PEPG and finite-difference gradients on a physical optical neural
network. This library reimplements the algorithms from standard references
under a single consistent interface, adds Simulated Annealing, a Genetic
Algorithm and Nelder-Mead, and adds the energy/complexity/convergence
benchmark harness.
