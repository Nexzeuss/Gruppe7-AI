"""Nested CV hyperparameter tuning for all three models on all three feature subsets.

Outer loop: the same five folds as run_subset_A/B/C.py, so tuned and default scores are comparable.
Inner loop: RandomizedSearchCV with five-fold CV on each outer training fold picks the settings.
"""

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from model_utils import N_SPLITS, RANDOM_STATE
from subset_utils import MODELS, SUBSET_FOLDER, SUBSETS, run_subset


N_ITER = 25

SEARCH_SPACES = {
    "Decision Tree": {
        "max_depth": [3, 5, 8, 12, 16, 20, None],
        "min_samples_leaf": [1, 2, 5, 10, 20, 40],
        "min_samples_split": [2, 5, 10, 20],
        "max_features": [None, "sqrt", 0.5],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [8, 12, 16, 24, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", 0.3, 0.5, 1.0],
    },
    "HistGradientBoosting": {
        "max_iter": [100, 300, 600],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "max_leaf_nodes": [7, 15, 31, 63],
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [5, 10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    },
}


def tuned_models():
    # Selection uses MAE on the log target, so a few extreme buildings do not decide the settings.
    return {
        name: RandomizedSearchCV(
            model, SEARCH_SPACES[name], n_iter=N_ITER, cv=N_SPLITS,
            scoring="neg_mean_absolute_error", random_state=RANDOM_STATE, n_jobs=-1,
        )
        for name, model in MODELS.items()
    }


def main():
    for subset in SUBSETS:
        run_subset(subset, models=tuned_models(), prefix="tuned_subset")

    metrics = ["rmse", "mae", "r2"]
    default = pd.concat([pd.read_csv(SUBSET_FOLDER / f"subset_{s}_performance.csv") for s in SUBSETS])
    tuned = pd.concat([pd.read_csv(SUBSET_FOLDER / f"tuned_subset_{s}_performance.csv") for s in SUBSETS])
    summary = pd.concat(
        {
            "default": default.groupby(["subset", "model"])[metrics].mean(),
            "tuned": tuned.groupby(["subset", "model"])[metrics].mean(),
        },
        axis=1,
    )
    summary.to_csv(SUBSET_FOLDER / "tuned_vs_default_summary.csv")
    tuned[["subset", "model", "fold", "best_params"]].to_csv(SUBSET_FOLDER / "tuned_best_params.csv", index=False)
    print(summary.round(3).to_string())
    print(f"Saved tuned results in {SUBSET_FOLDER}")


if __name__ == "__main__":
    main()
