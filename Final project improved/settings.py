"""Paths, choices and helpers shared by the improved final-assignment scripts (Problem 1: electricity density).

Run the scripts in order: 1_metadata_preparation.py, 2_weather_preparation.py,
3_electricity_density.py, 4_experiments.py. Each one reads the previous output from OUTPUT_FOLDER.

Compared with the recipe version, the target is computed the same way (steps 1-6), but the Q1-Q3 filter is
replaced by an outlier rule on the log scale, and the experiments add features and models one step at a time.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "building-data-genome-project-2" / "data"
METADATA_FILE = DATA_FOLDER / "metadata" / "metadata.csv"
WEATHER_FILE = DATA_FOLDER / "weather" / "weather.csv"
ELECTRICITY_FILE = DATA_FOLDER / "meters" / "cleaned" / "electricity_cleaned.csv"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "outputs"

# Metadata: the assignment's columns plus heating type and the meter columns used for the meter indicators.
METADATA_COLUMNS = [
    "building_id", "site_id", "sqm", "timezone", "yearbuilt", "numberoffloors",
    "occupants", "rating", "primaryspaceusage", "sub_primaryspaceusage", "heatingtype",
]
METER_COLUMNS = ["chilledwater", "steam", "hotwater", "gas"]
ESSENTIAL_COLUMNS = ["sqm"]

# Target, steps 1-6 as in the recipe.
MAX_MISSING_SHARE = 0.05
YEARS = [2016, 2017]

# Outlier rule, fixed before looking at any model scores: the IQR rule on log(density).
# Electricity density is log-normal, so the rule is symmetric on the log scale.
OUTLIER_IQR_FACTOR = 1.5

# Climate features: degree-days use the common 18 degC base temperature.
DEGREE_DAY_BASE = 18.0

TARGET = "electricity_density"
RANDOM_STATE = 42
N_SPLITS = 5


def log_step(log, script, step, description, buildings_before, buildings_after):
    """Record how many buildings each step keeps, for the data-preparation overview."""
    log.append({
        "script": script,
        "step": step,
        "description": description,
        "buildings_before": buildings_before,
        "buildings_after": buildings_after,
        "buildings_removed": buildings_before - buildings_after,
    })
    print(f"  Step {step}: {description} ({buildings_before} -> {buildings_after} buildings)")


def save_log(log, script):
    """Replace this script's rows in the shared steps overview."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_FOLDER / "preparation_steps.csv"
    frames = [pd.DataFrame(log)]
    if path.exists():
        previous = pd.read_csv(path)
        frames.insert(0, previous[previous["script"] != script])
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["script", "step"], kind="stable").to_csv(path, index=False)
