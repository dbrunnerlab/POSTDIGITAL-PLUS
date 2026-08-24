# POSTDIGITAL-PLUS

Training algorithms and benchmarking data for physical neural networks and
non-digital computing systems.

## Repository layout

- [`training-algorithms/`](training-algorithms/) library of black-box
  (zeroth-order) optimization algorithms (CMA-ES, PSO, PEPG, SPSA,
  finite-difference gradient, simulated annealing, genetic algorithm,
  Nelder-Mead) behind a unified `ask()`/`tell()` interface, with a benchmark
  harness comparing them on energy efficiency (evaluation count; the
  relevant cost when evaluations come from physical hardware), complexity,
  and convergence performance. See its
  [README](training-algorithms/README.md) for details and usage.
- [`benchmarking/`](benchmarking/) physical neural network comparison
  data and datasets used to evaluate training algorithms on real hardware.
  - `PNN-comparison-database/` results database for comparing
    optimization algorithms on physical neural network hardware.
  - `data-sets/` datasets used across benchmarking experiments.
