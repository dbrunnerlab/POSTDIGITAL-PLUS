"""Derives the three headline metrics -- energy efficiency, complexity and
convergence performance -- from the raw traces produced by experiments.runner.
"""

import numpy as np
import pandas as pd


def evaluations_to_target(trace: pd.DataFrame, target: float) -> float:
    """Energy metric: evaluation count at which best_fitness first reaches
    ``target``. Returns NaN if the target was never reached within the budget.
    """
    reached = trace[trace["best_fitness"] <= target]
    if reached.empty:
        return np.nan
    return float(reached["evaluations"].iloc[0])


def energy_summary(results: pd.DataFrame, targets: dict[str, float]) -> pd.DataFrame:
    """Per (optimizer, objective, n_dim): mean/std evaluations-to-target and
    success rate (fraction of trials that reached the target) across trials.
    """
    rows = []
    group_cols = ["optimizer", "objective", "n_dim"]
    for keys, group in results.groupby(group_cols):
        optimizer_name, objective_name, n_dim = keys
        target = targets[objective_name]
        per_trial = group.groupby("trial").apply(
            lambda trace: evaluations_to_target(trace, target), include_groups=False
        )
        rows.append(
            {
                "optimizer": optimizer_name,
                "objective": objective_name,
                "n_dim": n_dim,
                "evals_to_target_mean": per_trial.mean(),
                "evals_to_target_std": per_trial.std(),
                "success_rate": per_trial.notna().mean(),
            }
        )
    return pd.DataFrame(rows)


def convergence_curve(results: pd.DataFrame, optimizer: str, objective: str, n_dim: int) -> pd.DataFrame:
    """Mean and std of best-fitness-so-far vs. evaluations, averaged over trials,
    on a shared evaluation grid (piecewise-constant interpolation of each
    trial's step function).
    """
    subset = results[
        (results["optimizer"] == optimizer)
        & (results["objective"] == objective)
        & (results["n_dim"] == n_dim)
    ]
    grid = np.sort(subset["evaluations"].unique())

    per_trial_curves = []
    for _, trial_trace in subset.groupby("trial"):
        trial_trace = trial_trace.sort_values("evaluations")
        interpolated = np.interp(
            grid, trial_trace["evaluations"], trial_trace["best_fitness"]
        )
        per_trial_curves.append(interpolated)

    curves = np.array(per_trial_curves)
    return pd.DataFrame(
        {"evaluations": grid, "mean_best_fitness": curves.mean(axis=0), "std_best_fitness": curves.std(axis=0)}
    )


def complexity_scaling(results: pd.DataFrame) -> pd.DataFrame:
    """Per (optimizer, n_dim): mean wall-clock time per evaluation, averaged
    across objectives and trials -- the empirical counterpart to each
    algorithm's documented Big-O complexity.
    """
    per_step = results.copy()
    per_step["time_per_eval"] = per_step["wall_time"] / per_step["evaluations"]
    return (
        per_step.groupby(["optimizer", "n_dim"])["time_per_eval"]
        .mean()
        .reset_index()
        .rename(columns={"time_per_eval": "mean_time_per_evaluation"})
    )
