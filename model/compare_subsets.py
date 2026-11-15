"""Summarise and plot the feature-subset results from run_subset_A/B/C.py."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from subset_utils import MODELS, SUBSET_FOLDER, SUBSETS


METRICS = ["rmse", "mae", "r2", "cv_rmse"]


def main():
    files = sorted(SUBSET_FOLDER.glob("subset_*_performance.csv"))
    if not files:
        raise FileNotFoundError(f"No subset results in {SUBSET_FOLDER}; run run_subset_A/B/C.py first.")
    results = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    summary = results.groupby(["subset", "model"])[METRICS].agg(["mean", "std"])
    summary.to_csv(SUBSET_FOLDER / "subset_performance_summary.csv")
    print(summary.round(3).to_string())

    subsets = [subset for subset in SUBSETS if subset in set(results["subset"])]
    for metric in METRICS:
        means = results.groupby(["subset", "model"])[metric].mean().unstack()[list(MODELS)].loc[subsets]
        stds = results.groupby(["subset", "model"])[metric].std().unstack()[list(MODELS)].loc[subsets]
        fig, ax = plt.subplots(figsize=(8, 5))
        means.plot.bar(yerr=stds, capsize=3, ax=ax, rot=0)
        ax.set_xticklabels([f"{subset} ({len(SUBSETS[subset])} features)" for subset in subsets])
        ax.set_xlabel("Feature subset")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Five-fold CV by feature subset: {metric.upper()}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(SUBSET_FOLDER / f"subset_comparison_{metric}.png", dpi=160)
        plt.close(fig)

    print(f"Saved summary and plots in {SUBSET_FOLDER}")


if __name__ == "__main__":
    main()
