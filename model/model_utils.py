"""Shared loading, feature preparation, and cross-validation for model scripts."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "building-data-genome-project-2" / "data"
METER_FILE = DATA_FOLDER / "meters" / "my_cleand_custom" / "electricity_cleaned.csv"
WEATHER_FILE = DATA_FOLDER / "meters" / "my_cleand_custom" / "weather_cleaned.csv"
METADATA_FILE = DATA_FOLDER / "metadata" / "my_cleand_custom" / "metadata_cleaned.csv"
PERFORMANCE_FOLDER = Path(__file__).resolve().parent / "model_performace"

TARGET = "meter_reading"
RANDOM_STATE = 42
N_BUILDINGS = 100
N_SPLITS = 5
WEATHER_FEATURES = ["airTemperature", "dewTemperature", "windSpeed", "cloudCoverage", "seaLvlPressure", "precipDepth1HR"]
TIME_FEATURES = ["hour", "weekday", "month", "dayofyear"]


def load_data():
    """Join custom-cleaned electricity, weather, and all eligible metadata features."""
    meters = pd.read_csv(METER_FILE, parse_dates=["timestamp"], index_col="timestamp")
    building_ids = meters.columns.to_series().sample(
        n=min(N_BUILDINGS, meters.shape[1]), random_state=RANDOM_STATE
    )
    meters = meters[building_ids].copy()
    data = meters.reset_index().melt(
        id_vars="timestamp", var_name="building_id", value_name=TARGET
    )
    data = data.dropna(subset=[TARGET])

    metadata = pd.read_csv(METADATA_FILE)
    if "building_id" not in metadata:
        raise ValueError(f"Metadata file must include building_id: {METADATA_FILE}")
    # Identifier and energy_intensity (derived from this electricity target) are not predictors.
    metadata_features = [
        column for column in metadata.columns
        if column not in {"building_id", "energy_intensity"}
    ]
    data = data.merge(metadata[["building_id", *metadata_features]], on="building_id", how="left", validate="many_to_one")

    weather = pd.read_csv(WEATHER_FILE, parse_dates=["timestamp"])
    available_weather = [column for column in WEATHER_FEATURES if column in weather.columns]
    data = data.merge(
        weather[["timestamp", "site_id", *available_weather]],
        on=["timestamp", "site_id"], how="left", validate="many_to_one",
    )
    data["hour"] = data["timestamp"].dt.hour
    data["weekday"] = data["timestamp"].dt.weekday
    data["month"] = data["timestamp"].dt.month
    data["dayofyear"] = data["timestamp"].dt.dayofyear

    features = [*TIME_FEATURES, *available_weather, *metadata_features]
    # Prevent leakage from target-derived features and avoid using timestamps/identifiers.
    features = [column for column in features if column not in {"building_id", "energy_intensity", "timestamp"}]
    data = data.replace([np.inf, -np.inf], np.nan)
    return data, features


def make_preprocessor(data, features, *, scale_numeric=False):
    categorical = [column for column in features if not pd.api.types.is_numeric_dtype(data[column])]
    numeric = [column for column in features if column not in categorical]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = [("numeric", make_pipeline(*[step for _, step in numeric_steps]), numeric)]
    if categorical:
        transformers.append((
            "categorical",
            make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def evaluate_cv(model_name, estimator, *, scale_numeric=False):
    """Evaluate log1p readings with five folds grouped by building; save row-level metrics."""
    from sklearn.base import clone
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold

    data, features = load_data()
    if data["building_id"].nunique() < N_SPLITS:
        raise ValueError(f"Need at least {N_SPLITS} buildings for grouped CV.")
    splitter = GroupKFold(n_splits=N_SPLITS)
    results = []
    y = np.log1p(data[TARGET].clip(lower=0))
    groups = data["building_id"]
    print(f"{model_name}: {len(data):,} rows, {groups.nunique()} buildings, {len(features)} features")

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(data[features], y, groups), start=1):
        train_x, valid_x = data.iloc[train_idx][features], data.iloc[valid_idx][features]
        train_y, valid_y = y.iloc[train_idx], y.iloc[valid_idx]
        pipeline = make_pipeline(
            make_preprocessor(data.iloc[train_idx], features, scale_numeric=scale_numeric),
            clone(estimator),
        )
        pipeline.fit(train_x, train_y)
        predictions = np.maximum(0, np.expm1(pipeline.predict(valid_x)))
        actual = data.iloc[valid_idx][TARGET].to_numpy()
        rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
        results.append({
            "model": model_name,
            "fold": fold,
            "train_rows": len(train_idx),
            "validation_rows": len(valid_idx),
            "validation_buildings": groups.iloc[valid_idx].nunique(),
            "rmse": rmse,
            "mae": float(mean_absolute_error(actual, predictions)),
            "r2": float(r2_score(actual, predictions)),
            "cv_rmse": float(rmse / actual.mean()) if actual.mean() else np.nan,
        })
        print(f"  Fold {fold}: RMSE={rmse:.3f}, MAE={results[-1]['mae']:.3f}, R2={results[-1]['r2']:.3f}")

    PERFORMANCE_FOLDER.mkdir(parents=True, exist_ok=True)
    output = PERFORMANCE_FOLDER / f"{model_name.lower().replace(' ', '_')}_performance.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"Saved {output}")
    return pd.DataFrame(results)