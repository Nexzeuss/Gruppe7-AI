"""Shared setup for the feature-subset sensitivity analysis on energy intensity (kWh/m2/year)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

from model_utils import METADATA_FILE, RANDOM_STATE, N_SPLITS, make_preprocessor


TARGET = "energy_intensity"
SUBSET_FOLDER = Path(__file__).resolve().parent / "subset_performance"

SUBSETS = {
    "A": [
        "primaryspaceusage", "sub_primaryspaceusage", "heatingtype", "sqm", "numberoffloors",
        "building_age", "yearbuilt", "log_sqm", "sqm_per_floor", "occupants", "T_amb",
        "site_id", "timezone", "lat", "lng", "n_meter_types", "has_chilledwater", "has_steam",
        "has_hotwater", "has_gas", "has_water", "has_irrigation", "has_solar",
    ],
    "B": [
        "primaryspaceusage", "sub_primaryspaceusage", "heatingtype", "sqm", "numberoffloors",
        "building_age", "T_amb",
    ],
    "C": ["primaryspaceusage", "sqm", "T_amb"],
}

MODELS = {
    "Decision Tree": DecisionTreeRegressor(max_depth=20, min_samples_leaf=10, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, max_depth=24, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE,
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True,
        validation_fraction=0.1, random_state=RANDOM_STATE,
    ),
}


def load_metadata():
    """One row per building; total_energy and energy_intensity are never used as inputs."""
    metadata = pd.read_csv(METADATA_FILE, index_col="building_id")
    return metadata.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])


def run_subset(subset, models=MODELS, prefix="subset"):
    """Run every model with five-fold CV on one feature subset and save fold-level metrics."""
    features = SUBSETS[subset]
    data = load_metadata()
    x = data[features]
    # Energy intensity is right-skewed, so models are trained on log1p and scored in kWh/m2/year.
    y = np.log1p(data[TARGET].clip(lower=0))
    actual_all = data[TARGET].to_numpy()
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print(f"Subset {subset}: {len(data)} buildings, {len(features)} features")

    results = []
    for model_name, estimator in models.items():
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x), start=1):
            pipeline = make_pipeline(make_preprocessor(x.iloc[train_idx], features), clone(estimator))
            pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
            predictions = np.maximum(0, np.expm1(pipeline.predict(x.iloc[valid_idx])))
            actual = actual_all[valid_idx]
            rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
            results.append({
                "subset": subset,
                "model": model_name,
                "fold": fold,
                "n_features": len(features),
                "train_buildings": len(train_idx),
                "validation_buildings": len(valid_idx),
                "rmse": rmse,
                "mae": float(mean_absolute_error(actual, predictions)),
                "r2": float(r2_score(actual, predictions)),
                "cv_rmse": rmse / actual.mean(),
            })
            # Tuned models are searches; keep the settings each outer fold picked.
            if hasattr(pipeline[-1], "best_params_"):
                results[-1]["best_params"] = str(pipeline[-1].best_params_)
        scores = pd.DataFrame(results[-N_SPLITS:])
        print(f"  {model_name}: RMSE={scores.rmse.mean():.1f}, MAE={scores.mae.mean():.1f}, R2={scores.r2.mean():.3f}")

    SUBSET_FOLDER.mkdir(parents=True, exist_ok=True)
    output = SUBSET_FOLDER / f"{prefix}_{subset}_performance.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"Saved {output}")
    return pd.DataFrame(results)
