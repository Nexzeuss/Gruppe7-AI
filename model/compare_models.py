"""Create model-comparison plots from per-model cross-validation CSV files."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from model_utils import PERFORMANCE_FOLDER


METRICS = ["rmse", "mae", "r2", "cv_rmse"]


def main():
    files = sorted(PERFORMANCE_FOLDER.glob("*_performance.csv"))
    if not files:
        raise FileNotFoundError(f"No model performance CSV files found in {PERFORMANCE_FOLDER}")
    results = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if not {"model", "fold", *METRICS}.issubset(results.columns):
        raise ValueError("Performance CSV files must contain model, fold, and all metric columns.")

    PERFORMANCE_FOLDER.mkdir(parents=True, exist_ok=True)
    summary = results.groupby("model")[METRICS].agg(["mean", "std"])
    summary.to_csv(PERFORMANCE_FOLDER / "model_performance_summary.csv")

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(9, 5))
        groups = [group[metric].dropna().to_numpy() for _, group in results.groupby("model", sort=True)]
        labels = [name for name, group in results.groupby("model", sort=True) if group[metric].notna().any()]
        ax.boxplot(groups, labels=labels, showmeans=True)
        ax.set_title(f"Cross-validation model comparison: {metric.upper()}")
        ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(PERFORMANCE_FOLDER / f"model_comparison_{metric}.png", dpi=160)
        plt.close(fig)

    print(f"Saved comparison summary and plots in {PERFORMANCE_FOLDER}")


if __name__ == "__main__":
    main()