"""One Random Forest per building category against one shared Random Forest for all buildings.

Two groupings are tested: primary use and sub-primary use. Categories with more than MIN_GROUP_SIZE buildings get
their own model; the rest are pooled into "Other". All approaches use the step 4 setup from 4_experiments.py
(features, forest, training on plausible buildings), the same five folds, and are scored on the same plausible
validation buildings, so they compare directly. Per-group scores use MAE: R2 inside one group is lower by
construction, because a group varies less.
"""

import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from settings import N_SPLITS, OUTPUT_FOLDER, RANDOM_STATE, TARGET


experiments = importlib.import_module("4_experiments")
figures = importlib.import_module("5_figures")
plt = figures.plt

MIN_GROUP_SIZE = 10  # a category needs more than this many buildings to get its own model
GREEN = "#1baf7a"  # third series color, as in the model assignment figures
SHARED = "One shared model"
# column: (label in results, output name, bar color)
GROUPINGS = {
    "primaryspaceusage": ("One model per primary use", "category_models", figures.ORANGE),
    "sub_primaryspaceusage": ("One model per sub-use", "subcategory_models", GREEN),
}
# The same sub-use spelled in two ways; merged before grouping so these buildings are not sent to "Other".
SAME_SUB_USE = {
    "Fire Station, Fire Stations": "Fire Station",
    "Community Center, Community Centers": "Community Center",
    "Public Service": "Public services",
}


def make_groups(data, column):
    labels = data[column].replace(SAME_SUB_USE)
    counts = labels.value_counts()
    large = counts[counts > MIN_GROUP_SIZE].index.drop("Other", errors="ignore")
    return labels.where(labels.isin(large), "Other").rename("group")


def out_of_fold_predictions(data, groupings):
    """Predictions for every building from the shared model and from each grouping, plus its validation fold."""
    features = experiments.FEATURES_METERS
    columns = features["onehot"] + features["target"] + features["numeric"]
    x, y = data[columns], np.log1p(data[TARGET].clip(lower=0))
    plausible = ~data["is_outlier"].to_numpy()
    predictions = pd.DataFrame(index=data.index, columns=[SHARED, *groupings, "fold"], dtype=float)
    # TargetEncoder cross-fits with five internal folds, so very small training sets one-hot encode instead.
    few_buildings = {**features, "onehot": features["onehot"] + features["target"], "target": []}

    def fit_predict(train_idx, valid_idx):
        chosen = features if len(train_idx) >= 2 * N_SPLITS else few_buildings
        pipeline = experiments.make_pipeline_for(chosen, experiments.FOREST)
        pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
        return np.maximum(0, np.expm1(pipeline.predict(x.iloc[valid_idx])))

    splits = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(data)
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        train_idx = train_idx[plausible[train_idx]]
        predictions.iloc[valid_idx, predictions.columns.get_loc(SHARED)] = fit_predict(train_idx, valid_idx)
        predictions.iloc[valid_idx, predictions.columns.get_loc("fold")] = fold
        for label, groups in groupings.items():
            column = predictions.columns.get_loc(label)
            for group in groups.unique():
                in_group = (groups == group).to_numpy()
                group_valid = valid_idx[in_group[valid_idx]]
                if len(group_valid):  # a small group can be missing from a validation fold
                    predictions.iloc[group_valid, column] = fit_predict(train_idx[in_group[train_idx]], group_valid)
    return predictions


def overall_scores(data, predictions, approaches):
    """R2, MAE, MSE and MAPE per fold on the plausible validation buildings, as in 4_experiments.py."""
    scored = predictions[~data["is_outlier"]]
    rows = []
    for approach in approaches:
        for fold, part in scored.groupby("fold"):
            rows.append({"approach": approach, "fold": int(fold),
                         **experiments.scores(data.loc[part.index, TARGET], part[approach])})
    return pd.DataFrame(rows)


def group_scores(data, groups, predictions, label):
    """MAE per group over all folds, on the plausible buildings of that group."""
    plausible = ~data["is_outlier"]
    rows = []
    for group, members in groups[plausible].groupby(groups):
        actual = data.loc[members.index, TARGET]
        rows.append({
            "group": group, "buildings": int((groups == group).sum()), "scored": len(members),
            "median_density": actual.median(),
            "mae_shared": mean_absolute_error(actual, predictions.loc[members.index, SHARED]),
            "mae_per_group": mean_absolute_error(actual, predictions.loc[members.index, label]),
        })
    result = pd.DataFrame(rows).sort_values("scored", ascending=False).set_index("group")
    result["mae_change"] = result["mae_per_group"] - result["mae_shared"]
    return result


def plot(by_group, summary, label, name, color):
    # Largest group at the top; "Other" last, since it is a mix of small categories.
    order = [group for group in by_group.index if group != "Other"] + ["Other"]
    order = order[::-1]
    positions = np.arange(len(order))
    height = 0.38
    fig, ax = plt.subplots(figsize=(figures.FULL_WIDTH, 0.42 * len(order) + 1.2))
    for offset, column, bar_label, bar_color in [(height / 2, "mae_shared", SHARED, figures.BLUE),
                                                 (-height / 2, "mae_per_group", label, color)]:
        ax.barh(positions + offset, by_group.loc[order, column], height=height, color=bar_color, label=bar_label)
    ax.set_yticks(positions, [f"{group} ({by_group.loc[group, 'scored']})" for group in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("MAE (kWh/m$^2$/year), plausible buildings, all folds")
    shared, grouped = summary.loc[SHARED], summary.loc[label]
    ax.set_title(f"All buildings: shared R$^2$ {shared[('r2', 'mean')]:.3f}, MAE {shared[('mae', 'mean')]:.1f}   |   "
                 f"per group R$^2$ {grouped[('r2', 'mean')]:.3f}, MAE {grouped[('mae', 'mean')]:.1f}",
                 fontsize=8.5, color=figures.INK_SECONDARY, loc="left")
    # Free space above the top group for the legend.
    ax.set_ylim(-0.6, len(order) + 0.5)
    ax.legend(loc="upper right", ncol=2)
    figures.style(ax, "x")
    fig.tight_layout()
    figures.save(fig, name)


def main():
    data = experiments.load_data()
    groupings = {label: make_groups(data, column) for column, (label, _, _) in GROUPINGS.items()}
    for label, groups in groupings.items():
        print(f"{label}: {groups.nunique()} groups (more than {MIN_GROUP_SIZE} buildings, rest in Other)")
    predictions = out_of_fold_predictions(data, groupings)

    results = overall_scores(data, predictions, [SHARED, *groupings])
    summary = results.groupby("approach", sort=False)[["r2", "mae", "mse", "mape"]].agg(["mean", "std"])
    print(summary.round(3).to_string())
    results.to_csv(OUTPUT_FOLDER / "category_models_results.csv", index=False)
    summary.to_csv(OUTPUT_FOLDER / "category_models_summary.csv")

    for label, name, color in GROUPINGS.values():
        by_group = group_scores(data, groupings[label], predictions, label)
        print(f"\n{label}\n{by_group.round(1).to_string()}")
        by_group.to_csv(OUTPUT_FOLDER / f"{name}_by_group.csv")
        plot(by_group, summary, label, name, color)


if __name__ == "__main__":
    main()
