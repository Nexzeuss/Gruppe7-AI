"""Step-by-step improvement experiments (ablation) for electricity density, with 5-fold CV.

Every experiment uses the same five folds over all buildings, and every score is computed on the same buildings:
the validation buildings inside the plausible range. Only the training data, features or model change between
rows, so the rows can be compared directly. Models are trained on log(1 + density) and scored in kWh/m2/year.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from settings import N_SPLITS, OUTPUT_FOLDER, RANDOM_STATE, TARGET


BASE_CATEGORICAL = ["site_id", "timezone", "rating", "primaryspaceusage", "sub_primaryspaceusage", "heatingtype"]
BASE_NUMERIC = ["sqm", "yearbuilt", "numberoffloors", "occupants", "T_amb"]
TARGET_ENCODED = ["site_id", "sub_primaryspaceusage"]
CLIMATE = ["HDD", "CDD", "T_std", "dew_mean"]
METERS = ["has_chilledwater", "has_steam", "has_hotwater", "has_gas"]

# Each step adds to the one before it.
FEATURES_BASE = {"onehot": BASE_CATEGORICAL, "target": [], "numeric": BASE_NUMERIC}
FEATURES_ENCODED = {
    "onehot": [column for column in BASE_CATEGORICAL if column not in TARGET_ENCODED],
    "target": TARGET_ENCODED,
    "numeric": BASE_NUMERIC,
}
FEATURES_CLIMATE = {**FEATURES_ENCODED, "numeric": BASE_NUMERIC + CLIMATE}
FEATURES_METERS = {**FEATURES_CLIMATE, "numeric": BASE_NUMERIC + CLIMATE + METERS}

# Default settings from the model assignment, so row 0 matches that setup.
FOREST = RandomForestRegressor(
    n_estimators=150, max_depth=24, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE,
)
EXTRA_TREES = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)
BOOSTING = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True,
    validation_fraction=0.1, random_state=RANDOM_STATE,
)
SEARCH_SPACES = {
    "Random Forest": (FOREST, {
        "n_estimators": [200, 400],
        "max_depth": [8, 12, 16, 24, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", 0.3, 0.5, 1.0],
    }),
    "HistGradientBoosting": (BOOSTING, {
        "max_iter": [100, 300, 600],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_leaf_nodes": [7, 15, 31],
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [5, 10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    }),
}
N_ITER = 20


def tuned(name):
    """Nested tuning: the search runs its own 5-fold CV inside each outer training fold."""
    model, space = SEARCH_SPACES[name]
    prefix = type(model).__name__.lower()
    return RandomizedSearchCV(
        make_pipeline_for(FEATURES_METERS, model), {f"{prefix}__{key}": value for key, value in space.items()},
        n_iter=N_ITER, cv=N_SPLITS, scoring="neg_mean_absolute_error", random_state=RANDOM_STATE, n_jobs=-1,
    )


def make_pipeline_for(features, model):
    """Preprocessing fitted on the training buildings only, then the model.

    TargetEncoder replaces a category by the mean log density of the training buildings in that category,
    using internal cross-fitting so a building never sees its own target.
    """
    transformers = []
    if features["onehot"]:
        transformers.append(("onehot", OneHotEncoder(handle_unknown="ignore"), features["onehot"]))
    if features["target"]:
        transformers.append(("target", TargetEncoder(random_state=RANDOM_STATE), features["target"]))
    transformers.append(("numeric", SimpleImputer(strategy="median"), features["numeric"]))
    return make_pipeline(ColumnTransformer(transformers), clone(model))


EXPERIMENTS = [
    # (name, features, model or search, train on outlier-free buildings only)
    ("0 Starting point (model assignment setup)", FEATURES_BASE, FOREST, False),
    ("1 + outlier rule for training", FEATURES_BASE, FOREST, True),
    ("2 + target encoding of site and sub-use", FEATURES_ENCODED, FOREST, True),
    ("3 + climate features", FEATURES_CLIMATE, FOREST, True),
    ("4 + meter indicators", FEATURES_METERS, FOREST, True),
    ("5a ExtraTrees", FEATURES_METERS, EXTRA_TREES, True),
    ("5b HistGradientBoosting", FEATURES_METERS, BOOSTING, True),
    ("6a Random Forest, tuned", FEATURES_METERS, "Random Forest", True),
    ("6b HistGradientBoosting, tuned", FEATURES_METERS, "HistGradientBoosting", True),
]


def load_data():
    data = pd.read_csv(OUTPUT_FOLDER / "electricity_density_dataset.csv", index_col="building_id")
    categorical = sorted(set(BASE_CATEGORICAL))
    # One type per categorical column; missing values become their own category.
    data[categorical] = data[categorical].astype("object").fillna("missing").astype(str)
    return data


def scores(actual, predicted):
    return {
        "r2": r2_score(actual, predicted),
        "mae": mean_absolute_error(actual, predicted),
        "mse": mean_squared_error(actual, predicted),
        "mape": mean_absolute_percentage_error(actual, predicted) * 100,
    }


def run_experiment(name, features, model, train_plausible, data, splits):
    columns = features["onehot"] + features["target"] + features["numeric"]
    x, density = data[columns], data[TARGET]
    y = np.log1p(density.clip(lower=0))
    plausible = ~data["is_outlier"].to_numpy()
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        if train_plausible:
            train_idx = train_idx[plausible[train_idx]]
        estimator = tuned(model) if isinstance(model, str) else make_pipeline_for(features, model)
        estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
        predicted = np.maximum(0, np.expm1(estimator.predict(x.iloc[valid_idx])))
        actual = density.iloc[valid_idx].to_numpy()
        inside = plausible[valid_idx]
        rows.append({"experiment": name, "fold": fold, "scored_on": "plausible buildings",
                     **scores(actual[inside], predicted[inside])})
        if not train_plausible:
            rows.append({"experiment": name, "fold": fold, "scored_on": "all buildings",
                         **scores(actual, predicted)})
    result = pd.DataFrame(rows)
    main_scores = result[result["scored_on"] == "plausible buildings"]
    print(f"  {name}: R2={main_scores.r2.mean():.3f} +/- {main_scores.r2.std():.3f}, MAE={main_scores.mae.mean():.1f}")
    return result


def plot_ablation(summary):
    order = summary.index[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ax.barh(order, summary.loc[order, ("r2", "mean")], xerr=summary.loc[order, ("r2", "std")], height=0.6,
            color="#2a78d6", error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#52514e"})
    ax.axvline(0, color="#b5b4ae", linewidth=0.8)
    ax.set_xlabel("R$^2$ (mean over 5 folds, scored on plausible buildings)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.savefig(OUTPUT_FOLDER / "ablation_r2.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_FOLDER / "ablation_r2.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_data()
    print(f"Experiments: {len(data)} buildings, {int(data['is_outlier'].sum())} flagged as outliers")
    splits = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(data))

    results = pd.concat([run_experiment(*experiment, data, splits) for experiment in EXPERIMENTS], ignore_index=True)

    # Robustness check: the best setup again, but with whole sites held out (a building at a new site).
    main_results = results[results["scored_on"] == "plausible buildings"]
    best = main_results.groupby("experiment")["r2"].mean().idxmax()
    name, features, model, train_plausible = next(e for e in EXPERIMENTS if e[0] == best)
    site_splits = list(GroupKFold(n_splits=N_SPLITS).split(data, groups=data["site_id"]))
    print(f"Site-grouped CV of the best setup ({best}):")
    results = pd.concat([results, run_experiment(f"7 Best setup ({best.split()[0]}), new sites (site-grouped CV)",
                                                 features, model, train_plausible, data, site_splits)],
                        ignore_index=True)

    results.to_csv(OUTPUT_FOLDER / "experiment_results.csv", index=False)
    summary = (results.groupby(["experiment", "scored_on"], sort=False)[["r2", "mae", "mse", "mape"]]
               .agg(["mean", "std"]))
    summary.to_csv(OUTPUT_FOLDER / "experiment_summary.csv")
    print(summary.round(3).to_string())
    plot_ablation(summary.xs("plausible buildings", level="scored_on"))
    print(f"Saved results and ablation_r2.png in {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
