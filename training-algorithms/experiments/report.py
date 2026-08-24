"""Turns the metrics in experiments.metrics into the plots and summary table
that get committed under results/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics


def plot_convergence(results: pd.DataFrame, objective: str, n_dim: int, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for optimizer_name in sorted(results["optimizer"].unique()):
        curve = metrics.convergence_curve(results, optimizer_name, objective, n_dim)
        ax.plot(curve["evaluations"], curve["mean_best_fitness"], label=optimizer_name)
        ax.fill_between(
            curve["evaluations"],
            curve["mean_best_fitness"] - curve["std_best_fitness"],
            curve["mean_best_fitness"] + curve["std_best_fitness"],
            alpha=0.15,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Evaluations (energy proxy)")
    ax.set_ylabel("Best fitness so far (log scale)")
    ax.set_title(f"Convergence on {objective} (n_dim={n_dim})")
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = output_dir / f"convergence_{objective}_{n_dim}d.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_evals_to_target(energy_df: pd.DataFrame, output_dir: Path) -> Path:
    pivot = energy_df.pivot_table(
        index="optimizer", columns="objective", values="evals_to_target_mean"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean evaluations to reach target (energy proxy)")
    ax.set_title("Energy efficiency by algorithm and objective")
    ax.legend(title="objective", fontsize=8)
    fig.tight_layout()

    output_path = output_dir / "evals_to_target.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_complexity_scaling(complexity_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for optimizer_name, group in complexity_df.groupby("optimizer"):
        group = group.sort_values("n_dim")
        ax.plot(group["n_dim"], group["mean_time_per_evaluation"], marker="o", label=optimizer_name)

    ax.set_yscale("log")
    ax.set_xlabel("n_dim")
    ax.set_ylabel("Mean wall-clock time per evaluation (s, log scale)")
    ax.set_title("Empirical complexity scaling")
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = output_dir / "complexity_scaling.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_summary_csv(energy_df: pd.DataFrame, complexity_df: pd.DataFrame, output_dir: Path) -> Path:
    summary = energy_df.merge(complexity_df, on=["optimizer", "n_dim"], how="left")
    output_path = output_dir / "summary.csv"
    summary.to_csv(output_path, index=False)
    return output_path
