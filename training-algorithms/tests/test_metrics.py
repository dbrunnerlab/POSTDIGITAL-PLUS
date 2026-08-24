import numpy as np
import pandas as pd
import pytest

from experiments import metrics


def _fake_trial(evaluations, best_fitness, optimizer="opt_a", objective="sphere", n_dim=4, trial=0):
    return pd.DataFrame(
        {
            "evaluations": evaluations,
            "best_fitness": best_fitness,
            "wall_time": np.linspace(0.001, 0.001 * len(evaluations), len(evaluations)),
            "optimizer": optimizer,
            "objective": objective,
            "n_dim": n_dim,
            "trial": trial,
        }
    )


def test_evaluations_to_target_reached():
    trace = _fake_trial([10, 20, 30, 40], [5.0, 2.0, 0.5, 0.05])
    assert metrics.evaluations_to_target(trace, target=1.0) == 30


def test_evaluations_to_target_never_reached():
    trace = _fake_trial([10, 20, 30], [5.0, 4.0, 3.0])
    assert np.isnan(metrics.evaluations_to_target(trace, target=1.0))


def test_energy_summary_success_rate():
    results = pd.concat(
        [
            _fake_trial([10, 20], [5.0, 0.5], trial=0),
            _fake_trial([10, 20], [5.0, 4.0], trial=1),
        ],
        ignore_index=True,
    )
    summary = metrics.energy_summary(results, targets={"sphere": 1.0})
    row = summary.iloc[0]
    assert row["success_rate"] == pytest.approx(0.5)
    assert row["evals_to_target_mean"] == pytest.approx(20.0)


def test_convergence_curve_shape():
    results = pd.concat(
        [
            _fake_trial([10, 20, 30], [5.0, 2.0, 1.0], trial=0),
            _fake_trial([10, 25, 30], [6.0, 1.5, 1.2], trial=1),
        ],
        ignore_index=True,
    )
    curve = metrics.convergence_curve(results, "opt_a", "sphere", 4)
    assert set(curve.columns) == {"evaluations", "mean_best_fitness", "std_best_fitness"}
    assert len(curve) == results["evaluations"].nunique()


def test_complexity_scaling_columns():
    results = pd.concat(
        [
            _fake_trial([10, 20], [5.0, 1.0], n_dim=4),
            _fake_trial([10, 20], [5.0, 1.0], n_dim=8),
        ],
        ignore_index=True,
    )
    scaling = metrics.complexity_scaling(results)
    assert set(scaling.columns) == {"optimizer", "n_dim", "mean_time_per_evaluation"}
    assert len(scaling) == 2
