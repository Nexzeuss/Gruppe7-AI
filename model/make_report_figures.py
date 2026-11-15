"""Figures for the report: target distribution, use types, model comparison, tuning, predictions, importance.

Run after run_subset_A/B/C.py and tune_hyperparameters.py; figures are saved in subset_performance/figures.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from model_utils import N_SPLITS, RANDOM_STATE, make_preprocessor
from subset_utils import MODELS, SUBSET_FOLDER, SUBSETS, TARGET, load_metadata


FIGURE_FOLDER = SUBSET_FOLDER / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MODEL_COLORS = dict(zip(MODELS, SERIES))
INK, INK_SECONDARY, AXIS, GRID = "#0b0b0b", "#52514e", "#b5b4ae", "#e4e3df"
FULL_WIDTH = 6.3  # inches, matches \textwidth of an A4 article

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


def out_of_fold(data, features, estimator, importance=True):
    """Out-of-fold predictions and per-fold permutation importance (MAE increase in kWh/m2/year)."""
    x, actual = data[features], data[TARGET]
    y = np.log1p(actual.clip(lower=0))
    predictions = pd.Series(np.nan, index=data.index)
    importances = []

    def mae_score(model, x_valid, y_valid):
        return -mean_absolute_error(np.expm1(y_valid), np.maximum(0, np.expm1(model.predict(x_valid))))

    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, valid_idx in splitter.split(x):
        pipeline = make_pipeline(make_preprocessor(x.iloc[train_idx], features), clone(estimator))
        pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions.iloc[valid_idx] = np.maximum(0, np.expm1(pipeline.predict(x.iloc[valid_idx])))
        if not importance:
            continue
        result = permutation_importance(
            pipeline, x.iloc[valid_idx], y.iloc[valid_idx], scoring=mae_score,
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
        )
        importances.append(pd.Series(result.importances_mean, index=features))
    return predictions, pd.concat(importances, axis=1) if importances else None


def mean_baseline(data):
    """Predict the training-fold mean for every validation building, using the same folds as the models."""
    actual = data[TARGET].to_numpy()
    scores = []
    for train_idx, valid_idx in KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(actual):
        prediction = np.full(len(valid_idx), actual[train_idx].mean())
        scores.append({"mae": mean_absolute_error(actual[valid_idx], prediction),
                       "r2": r2_score(actual[valid_idx], prediction)})
    return pd.DataFrame(scores)


def target_distribution(data):
    values = data[TARGET]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.8))
    bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 60)
    ax.hist(values, bins=bins, color=SERIES[0], edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    top = ax.get_ylim()[1]
    for limit, side, text in [(5, "right", f"{(values < 5).sum()} buildings\nbelow 5"),
                              (1000, "left", f"{(values > 1000).sum()} buildings\nabove 1 000")]:
        ax.axvline(limit, color=INK_SECONDARY, linestyle="--", linewidth=1)
        offset = 0.92 if side == "right" else 1.08
        ax.text(limit * offset, top * 0.92, text, ha=side, va="top", color=INK_SECONDARY, fontsize=8)
    median = values.median()
    ax.axvline(median, color=INK, linewidth=1.2)
    ax.text(median * 1.08, top * 0.72, f"median\n{median:.0f}", va="top", color=INK, fontsize=8)
    ax.set_xlabel("Electricity density (kWh/m$^2$/year, log scale)")
    ax.set_ylabel("Number of buildings")
    style(ax)
    save(fig, "target_distribution")


def feature_coverage(data):
    coverage = data[SUBSETS["A"]].notna().mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.8, 0.22 * len(coverage) + 0.8))
    ax.barh(coverage.index, coverage, height=0.65, color=SERIES[0])
    for name, share in coverage.items():
        ax.text(share + 1, name, f"{share:.0f} %", va="center", fontsize=7.5, color=INK_SECONDARY)
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Buildings with a non-missing value (%)")
    ax.tick_params(axis="y", length=0)
    style(ax, "x")
    save(fig, "feature_coverage")


def density_by_use(data):
    counts = data["primaryspaceusage"].value_counts()
    categories = counts[counts >= 5].index
    medians = data[data["primaryspaceusage"].isin(categories)].groupby("primaryspaceusage")[TARGET].median()
    order = medians.sort_values().index
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.8))
    ax.boxplot(
        [data.loc[data["primaryspaceusage"] == c, TARGET] for c in order], vert=False, widths=0.6,
        patch_artist=True, boxprops={"facecolor": "#d6e5f7", "edgecolor": SERIES[0]},
        medianprops={"color": SERIES[0], "linewidth": 2}, whiskerprops={"color": AXIS}, capprops={"color": AXIS},
        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": AXIS, "markeredgecolor": "none"},
    )
    ax.set_yticks(range(1, len(order) + 1), [f"{c} (n={counts[c]})" for c in order])
    ax.set_xscale("log")
    ax.set_xlabel("Electricity density (kWh/m$^2$/year, log scale)")
    style(ax, "x")
    save(fig, "density_by_use_type")


def subset_comparison(baseline):
    results = pd.concat([pd.read_csv(SUBSET_FOLDER / f"subset_{s}_performance.csv") for s in SUBSETS])
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))
    positions = np.arange(len(SUBSETS))
    width = 0.26
    for ax, metric, label in [(axes[0], "mae", "MAE (kWh/m$^2$/year)"), (axes[1], "r2", "R$^2$")]:
        for i, model in enumerate(MODELS):
            scores = results[results["model"] == model].groupby("subset")[metric]
            ax.bar(positions + (i - 1) * width, scores.mean()[list(SUBSETS)], width * 0.92,
                   yerr=scores.std()[list(SUBSETS)], color=MODEL_COLORS[model], label=model,
                   error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK_SECONDARY})
        ax.axhline(baseline[metric].mean(), color=INK_SECONDARY, linestyle="--", linewidth=1, label="Mean baseline")
        ax.set_xticks(positions, [f"{s} ({len(SUBSETS[s])})" for s in SUBSETS])
        ax.set_xlabel("Feature subset (number of features)")
        ax.set_ylabel(label)
        style(ax)
    axes[0].set_ylim(0, 140)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "subset_comparison")


def tuning_comparison():
    default = pd.concat([pd.read_csv(SUBSET_FOLDER / f"subset_{s}_performance.csv") for s in SUBSETS])
    tuned = pd.concat([pd.read_csv(SUBSET_FOLDER / f"tuned_subset_{s}_performance.csv") for s in SUBSETS])
    rows = [(s, m) for s in SUBSETS for m in MODELS]
    y = np.arange(len(rows))[::-1] + np.repeat(np.arange(len(SUBSETS))[::-1], len(MODELS)) * 0.8
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.6), sharey=True)
    for ax, metric, label in [(axes[0], "mae", "MAE (kWh/m$^2$/year)"), (axes[1], "r2", "R$^2$")]:
        before = default.groupby(["subset", "model"])[metric].mean().loc[rows].to_numpy()
        after = tuned.groupby(["subset", "model"])[metric].mean().loc[rows].to_numpy()
        ax.hlines(y, before, after, color=AXIS, linewidth=1.5)
        ax.scatter(before, y, s=36, facecolor="white", edgecolor=INK_SECONDARY, linewidth=1.2, label="Default", zorder=3)
        ax.scatter(after, y, s=36, color=SERIES[0], edgecolor="white", linewidth=1, label="Tuned", zorder=3)
        ax.set_xlabel(label)
        style(ax, "x")
    axes[0].set_yticks(y, [f"{s}: {m}" for s, m in rows])
    axes[0].tick_params(axis="y", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.05))
    fig.tight_layout()
    save(fig, "tuning_comparison")


def predicted_vs_actual(data, predictions):
    actual = data[TARGET]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.scatter(actual, predictions, s=10, color=SERIES[0], alpha=0.55, edgecolor="none")
    limits = [actual.min() * 0.8, actual.max() * 1.25]
    ax.plot(limits, limits, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.text(limits[1] * 0.7, limits[1] * 0.45, "perfect\nprediction", ha="right", color=INK_SECONDARY, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Measured (kWh/m$^2$/year)")
    ax.set_ylabel("Predicted (kWh/m$^2$/year)")
    ax.text(0.04, 0.96, f"MAE {mean_absolute_error(actual, predictions):.1f}", transform=ax.transAxes,
            va="top", fontsize=8, color=INK)
    style(ax, "both")
    save(fig, "predicted_vs_actual_rf_B")


def predicted_vs_actual_all_models(data, subset="B"):
    actual = data[TARGET]
    # Zoom on the bulk of the buildings; the few below 1 kWh/m2/year fall outside the axes.
    limits = [1, actual.max() * 1.25]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(FULL_WIDTH, 2.6), sharex=True, sharey=True)
    for ax, (model_name, estimator) in zip(axes, MODELS.items()):
        predictions, _ = out_of_fold(data, SUBSETS[subset], estimator, importance=False)
        ax.scatter(actual, predictions, s=7, color=MODEL_COLORS[model_name], alpha=0.5, edgecolor="none")
        ax.plot(limits, limits, color=INK_SECONDARY, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title(model_name, color=INK)
        ax.text(0.05, 0.95, f"MAE {mean_absolute_error(actual, predictions):.1f}", transform=ax.transAxes,
                va="top", fontsize=8, color=INK)
        ax.set_xlabel("Measured (kWh/m$^2$/year)")
        style(ax, "both")
    axes[0].set_ylabel("Predicted (kWh/m$^2$/year)")
    fig.tight_layout()
    save(fig, f"predicted_vs_actual_all_models_{subset}")


def feature_importance(importances, name):
    mean = importances.mean(axis=1).sort_values()
    std = importances.std(axis=1)[mean.index]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.75, 0.25 * len(mean) + 0.8))
    ax.barh(mean.index, mean, xerr=std, height=0.65, color=SERIES[0],
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK_SECONDARY})
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("Increase in MAE when feature is shuffled (kWh/m$^2$/year)")
    ax.tick_params(axis="y", length=0)
    style(ax, "x")
    save(fig, name)


def main():
    data = load_metadata()
    values = data[TARGET]
    print(f"Median {values.median():.1f}, Q1 {values.quantile(0.25):.1f}, Q3 {values.quantile(0.75):.1f}")
    target_distribution(data)
    feature_coverage(data)
    density_by_use(data)
    baseline = mean_baseline(data)
    baseline.to_csv(SUBSET_FOLDER / "mean_baseline_performance.csv", index=False)
    print(f"Mean baseline: MAE={baseline.mae.mean():.1f}, R2={baseline.r2.mean():.3f}")
    subset_comparison(baseline)
    tuning_comparison()

    # Default random forest, so the figures match the comparative study (Table 4).
    forest = MODELS["Random Forest"]
    predictions_b, importance_b = out_of_fold(data, SUBSETS["B"], forest)
    predicted_vs_actual(data, predictions_b)
    predicted_vs_actual_all_models(data)
    feature_importance(importance_b, "feature_importance_rf_B")
    _, importance_a = out_of_fold(data, SUBSETS["A"], forest)
    feature_importance(importance_a, "feature_importance_rf_A")


if __name__ == "__main__":
    main()
