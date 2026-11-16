"""Figures for the improved final project. Run after 4_experiments.py; figures are saved in outputs/figures."""

import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from settings import N_SPLITS, OUTLIER_IQR_FACTOR, OUTPUT_FOLDER, RANDOM_STATE, TARGET


experiments = importlib.import_module("4_experiments")

FIGURE_FOLDER = OUTPUT_FOLDER / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_SECONDARY, AXIS, GRID = "#0b0b0b", "#52514e", "#b5b4ae", "#e4e3df"
FULL_WIDTH = 6.3  # inches, matches \textwidth of an A4 article
# Q1 and Q3 used by the recipe version, shown for comparison.
RECIPE_RANGE = (72.3, 193.8)
SITE_GROUPED_LABEL = "7 Best setup, new sites (site-grouped CV)"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": AXIS,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_FOLDER / f"{name}.png")
    fig.savefig(FIGURE_FOLDER / f"{name}.pdf")
    plt.close(fig)
    print(f"Saved {name}")


def plausible_range(density):
    """The same IQR rule on log(density) as 3_electricity_density.py."""
    q1, q3 = np.log(density[density > 0]).quantile([0.25, 0.75])
    return np.exp(q1 - OUTLIER_IQR_FACTOR * (q3 - q1)), np.exp(q3 + OUTLIER_IQR_FACTOR * (q3 - q1))


def ablation():
    summary = pd.read_csv(OUTPUT_FOLDER / "experiment_summary.csv", header=[0, 1], index_col=[0, 1])
    summary = summary.xs("plausible buildings", level="scored_on")
    summary.index = [SITE_GROUPED_LABEL if name.startswith("7") else name for name in summary.index]
    order = summary.index[::-1]
    # Row 7 uses a different validation (new sites), so it gets a neutral color.
    colors = [INK_SECONDARY if name.startswith("7") else BLUE for name in order]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 1.25, 3.8), sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 1]})
    for ax, metric, label in [(axes[0], "r2", "R$^2$"), (axes[1], "mae", "MAE (kWh/m$^2$/year)")]:
        ax.barh(order, summary.loc[order, (metric, "mean")], xerr=summary.loc[order, (metric, "std")],
                height=0.6, color=colors, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK_SECONDARY})
        ax.axvline(0, color=AXIS, linewidth=0.8)
        ax.set_xlabel(f"{label}, mean over 5 folds")
        ax.tick_params(axis="y", length=0)
        style(ax, "x")
    save(fig, "ablation")


def target_distribution(data):
    density = data[TARGET]
    low, high = plausible_range(density)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.8))
    bins = np.logspace(np.log10(density[density > 0].min()), np.log10(density.max()), 60)
    ax.hist(density[density > 0], bins=bins, color=BLUE, edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    top = ax.get_ylim()[1] * 1.2
    ax.set_ylim(0, top)
    ax.axvspan(*RECIPE_RANGE, color=GRID, alpha=0.6, zorder=0)
    ax.text(np.sqrt(RECIPE_RANGE[0] * RECIPE_RANGE[1]), top * 0.98, "recipe [Q1, Q3]", ha="center", va="top",
            fontsize=7.5, color=INK_SECONDARY)
    for limit, side in [(low, "right"), (high, "left")]:
        ax.axvline(limit, color=INK, linestyle="--", linewidth=1)
        offset = 0.92 if side == "right" else 1.08
        ax.text(limit * offset, top * 0.98, f"{limit:.1f}", ha=side, va="top", fontsize=8, color=INK)
    ax.text(0.02, 0.6, f"{int(data['is_outlier'].sum())} buildings outside\nthe plausible range",
            transform=ax.transAxes, fontsize=8, color=INK_SECONDARY)
    ax.set_xlabel("Electricity density (kWh/m$^2$/year, log scale)")
    ax.set_ylabel("Number of buildings")
    style(ax)
    save(fig, "target_distribution")


def best_setup_out_of_fold(data):
    """Out-of-fold predictions and permutation importance for step 4 (Random Forest with all added features)."""
    features = experiments.FEATURES_METERS
    columns = features["onehot"] + features["target"] + features["numeric"]
    x, density = data[columns], data[TARGET]
    y = np.log1p(density.clip(lower=0))
    plausible = ~data["is_outlier"].to_numpy()
    predictions = pd.Series(np.nan, index=data.index)
    importances = []

    def mae_score(model, x_valid, y_valid):
        return -mean_absolute_error(np.expm1(y_valid), np.maximum(0, np.expm1(model.predict(x_valid))))

    for train_idx, valid_idx in KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(data):
        train_idx = train_idx[plausible[train_idx]]
        pipeline = experiments.make_pipeline_for(features, experiments.FOREST)
        pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions.iloc[valid_idx] = np.maximum(0, np.expm1(pipeline.predict(x.iloc[valid_idx])))
        scored_idx = valid_idx[plausible[valid_idx]]
        result = permutation_importance(pipeline, x.iloc[scored_idx], y.iloc[scored_idx], scoring=mae_score,
                                        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        importances.append(pd.Series(result.importances_mean, index=columns))
    return predictions, pd.concat(importances, axis=1)


def predicted_vs_measured(data, predictions):
    density = data[TARGET]
    outlier = data["is_outlier"]
    limits = [1, density.max() * 1.25]
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.scatter(density[~outlier], predictions[~outlier], s=9, color=BLUE, alpha=0.5, edgecolor="none",
               label="Plausible (scored)")
    ax.scatter(density[outlier], predictions[outlier], s=14, color=ORANGE, alpha=0.9, edgecolor="none",
               label="Outlier (not scored)")
    ax.plot(limits, limits, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Measured (kWh/m$^2$/year)")
    ax.set_ylabel("Predicted (kWh/m$^2$/year)")
    mae = mean_absolute_error(density[~outlier], predictions[~outlier])
    ax.text(0.04, 0.96, f"MAE {mae:.1f} (plausible)", transform=ax.transAxes, va="top", fontsize=8, color=INK)
    ax.legend(loc="lower right")
    style(ax, "both")
    save(fig, "predicted_vs_measured")


def feature_importance(importances):
    mean = importances.mean(axis=1).sort_values()
    std = importances.std(axis=1)[mean.index]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.75, 0.24 * len(mean) + 0.8))
    ax.barh(mean.index, mean, xerr=std, height=0.65, color=BLUE,
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK_SECONDARY})
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("Increase in MAE when feature is shuffled (kWh/m$^2$/year)")
    ax.tick_params(axis="y", length=0)
    style(ax, "x")
    save(fig, "feature_importance")


def site_climate(data):
    """Median electricity density per site against cooling degree-days."""
    plausible = data[~data["is_outlier"]]
    sites = plausible.groupby("site_id").agg(median=(TARGET, "median"), CDD=("CDD", "first"), n=(TARGET, "size"))
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.75, 3.4))
    ax.scatter(sites["CDD"], sites["median"], s=24, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    # Wolf and Bear sit close together, so Wolf's label goes below its point.
    offsets = {"Wolf": (4, -10)}
    for site, row in sites.iterrows():
        ax.annotate(f"{site} (n={int(row.n)})", (row.CDD, row["median"]), xytext=offsets.get(site, (4, 2)),
                    textcoords="offset points", fontsize=7, color=INK_SECONDARY)
    ax.set_xlabel("Cooling degree-days per year (base 18 $^\\circ$C)")
    ax.set_ylabel("Median density (kWh/m$^2$/year)")
    style(ax, "both")
    save(fig, "site_climate")


def main():
    data = experiments.load_data()
    ablation()
    target_distribution(data)
    predictions, importances = best_setup_out_of_fold(data)
    predicted_vs_measured(data, predictions)
    feature_importance(importances)
    site_climate(data)
    print(f"Saved figures in {FIGURE_FOLDER}")


if __name__ == "__main__":
    main()
