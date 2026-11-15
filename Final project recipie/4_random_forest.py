"""Problem 1, step 11: Random Forest pipeline, evaluated with MAE, MSE, MAPE and R2 in 5-fold CV."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from settings import N_SPLITS, OUTPUT_FOLDER, RANDOM_STATE, TARGET


CATEGORICAL = ["site_id", "timezone", "rating", "primaryspaceusage", "sub_primaryspaceusage"]
NUMERIC = ["sqm", "yearbuilt", "numberoffloors", "occupants", "T_amb"]


def make_pipeline():
    """One-hot encode categorical variables, impute missing numeric values with the mean, then the forest."""
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", SimpleImputer(strategy="mean"), NUMERIC),
    ])
    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def evaluate(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "mse": mean_squared_error(actual, predicted),
        "mape": mean_absolute_percentage_error(actual, predicted) * 100,
        "r2": r2_score(actual, predicted),
    }


def main():
    data = pd.read_csv(OUTPUT_FOLDER / "electricity_density_dataset.csv", index_col="building_id")
    # OneHotEncoder needs one type per column; missing categories become their own category.
    x = data[CATEGORICAL + NUMERIC].astype({column: "object" for column in CATEGORICAL})
    x[CATEGORICAL] = x[CATEGORICAL].fillna("missing")
    y = data[TARGET]
    print(f"Random Forest: {len(data)} buildings, {len(CATEGORICAL) + len(NUMERIC)} features")

    results = []
    predictions = pd.Series(np.nan, index=data.index, name="predicted")
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x), start=1):
        pipeline = make_pipeline().fit(x.iloc[train_idx], y.iloc[train_idx])
        predicted = pipeline.predict(x.iloc[valid_idx])
        predictions.iloc[valid_idx] = predicted
        results.append({"fold": fold, **evaluate(y.iloc[valid_idx], predicted)})
        scores = results[-1]
        print(f"  Fold {fold}: MAE={scores['mae']:.2f}, MSE={scores['mse']:.1f}, "
              f"MAPE={scores['mape']:.1f} %, R2={scores['r2']:.3f}")

    results = pd.DataFrame(results)
    summary = results.drop(columns="fold").agg(["mean", "std"]).T
    print(summary.round(3).to_string())

    results.to_csv(OUTPUT_FOLDER / "random_forest_cv_results.csv", index=False)
    summary.to_csv(OUTPUT_FOLDER / "random_forest_cv_summary.csv")
    pd.concat([y.rename("measured"), predictions], axis=1).to_csv(OUTPUT_FOLDER / "random_forest_predictions.csv")
    print(f"Saved results in {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
