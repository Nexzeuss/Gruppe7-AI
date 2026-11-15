"""Paths, choices and helpers shared by the final-assignment scripts (Problem 1: electricity density).

Run the scripts in order: 1_metadata_preparation.py, 2_weather_preparation.py,
3_electricity_density.py, 4_random_forest.py. Each one reads the previous output from OUTPUT_FOLDER.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "building-data-genome-project-2" / "data"
METADATA_FILE = DATA_FOLDER / "metadata" / "metadata.csv"
WEATHER_FILE = DATA_FOLDER / "weather" / "weather.csv"
# The assignment names BDG2's electricity_cleaned.csv; swap in data/meters/my_cleand_custom to use our own cleaning.
ELECTRICITY_FILE = DATA_FOLDER / "meters" / "cleaned" / "electricity_cleaned.csv"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "outputs"

# Choices from the assignment text.
METADATA_COLUMNS = [
    "building_id", "site_id", "sqm", "timezone", "yearbuilt", "numberoffloors",
    "occupants", "rating", "primaryspaceusage", "sub_primaryspaceusage",
]
ESSENTIAL_COLUMNS = ["sqm"]
MAX_MISSING_SHARE = 0.05  # Problem 1 step 2: drop buildings with more than 5 % missing hours
YEARS = [2016, 2017]
OUTLIER_QUANTILES = (0.25, 0.75)  # Problem 1 step 7: keep buildings within [Q1, Q3]

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
