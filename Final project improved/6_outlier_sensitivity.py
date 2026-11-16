"""How the Random Forest score depends on how many buildings are removed as outliers.

For each share p, the lowest p % and the highest p % of electricity densities are removed (p = 25 % keeps [Q1, Q3],
the recipe's rule). The best setup from 4_experiments.py (step 4) is then trained and scored with 5-fold CV on the
remaining buildings. Each share is scored on its own set of buildings, so a lower MAE partly reflects an easier set.
"""

import importlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from settings import N_SPLITS, OUTPUT_FOLDER, RANDOM_STATE, TARGET


experiments = importlib.import_module("4_experiments")
figures = importlib.import_module("5_figures")
plt = figures.plt

TRIM_SHARES = [0, 5, 10, 15, 20, 25]


def run_share(data, share):
    density = data[TARGET]
    low, high = density.quantile([share / 100, 1 - share / 100])
    kept = data[density.between(low, high)]
    features = experiments.FEATURES_METERS
    columns = features["onehot"] + features["target"] + features["numeric"]
    x, actual = kept[columns], kept[TARGET]
    y = np.log1p(actual.clip(lower=0))
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(
            KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(kept), start=1):
        pipeline = experiments.make_pipeline_for(features, experiments.FOREST)
        pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
        predicted = np.maximum(0, np.expm1(pipeline.predict(x.iloc[valid_idx])))
        rows.append({"trim_share": share, "fold": fold, "buildings": len(kept), "low": low, "high": high,
                     "target_std": actual.std(), **experiments.scores(actual.iloc[valid_idx], predicted)})
    result = pd.DataFrame(rows)
    print(f"  {share:>2} % removed at each end: {len(kept)} buildings in [{low:.1f}, {high:.1f}], "
          f"R2={result.r2.mean():.3f}, MAE={result.mae.mean():.1f}")
    return result


def plot(summary):
    fig, axes = plt.subplots(1, 2, figsize=(figures.FULL_WIDTH, 3.0))
    # Each tick shows the share removed and how many buildings are left.
    tick_labels = [f"{share} %\n{n}" for share, n in summary[("buildings", "first")].items()]
    for ax, metric, label in [(axes[0], "r2", "R$^2$"), (axes[1], "mae", "MAE (kWh/m$^2$/year)")]:
        ax.errorbar(summary.index, summary[(metric, "mean")], yerr=summary[(metric, "std")], color=figures.BLUE,
                    marker="o", markersize=5, linewidth=2, capsize=3, elinewidth=0.8, ecolor=figures.INK_SECONDARY)
        ax.set_xticks(summary.index, tick_labels, fontsize=7)
        ax.set_xlabel("Removed at each end (buildings left)")
        ax.set_ylabel(label)
        figures.style(ax, "both")
    axes[0].axhline(0, color=figures.AXIS, linewidth=0.8)
    axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    figures.save(fig, "outlier_sensitivity")


def main():
    data = experiments.load_data()
    print(f"Outlier sensitivity: Random Forest, step 4 features, {len(data)} buildings")
    results = pd.concat([run_share(data, share) for share in TRIM_SHARES], ignore_index=True)
    results.to_csv(OUTPUT_FOLDER / "outlier_sensitivity_results.csv", index=False)
    summary = results.groupby("trim_share").agg({
        "buildings": "first", "low": "first", "high": "first", "target_std": "first",
        "r2": ["mean", "std"], "mae": ["mean", "std"], "mse": ["mean", "std"], "mape": ["mean", "std"],
    })
    summary.to_csv(OUTPUT_FOLDER / "outlier_sensitivity_summary.csv")
    plot(summary)


if __name__ == "__main__":
    main()
