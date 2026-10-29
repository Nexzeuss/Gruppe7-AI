import numpy as np
import pandas as pd
from pathlib import Path

#### cleans metadata
#### - generates building features, incl. energy intensity (kWh/sqm/year) from cleaned electricity
#### - keeps only the columns needed before feature selection
#### - indexes by building_id so it aligns with the meter columns
#### - drops buildings missing important data
#### - saves to data/metadata/my_cleand_custom

#### tunig variables
reference_year = 2016  # first year of meter data, used for building age
align_with_cleaned_meters = True  # drop buildings with no column left in any cleaned meter file
hours_per_year = 365 * 24
min_energy_coverage = 0.5  # share of the hours (2016-2017) that must have readings, else NaN

meter_types = ["electricity", "chilledwater", "steam", "hotwater", "gas", "water", "irrigation", "solar"]

# columns kept before feature selection, the rest are duplicates (sqft, kaggle ids)
# or not selected for this output (sqft, Kaggle IDs, ratings, etc.).
columns_to_keep = [
    "site_id", "primaryspaceusage", "sub_primaryspaceusage", "heatingtype", "timezone",
    "sqm", "yearbuilt", "numberoffloors", "lat", "lng", "occupants",
]

# rows missing any of these are dropped
# yearbuilt/numberoffloors/lat are missing for entire sites, so they are left as NaN instead
important_columns = ["site_id", "primaryspaceusage", "sqm"]


## data paths/managment
project_folder = Path(__file__).resolve().parent.parent
metadata_file = project_folder / "data" / "metadata" / "metadata.csv"
cleaned_meter_folder = project_folder / "data" / "meters" / "my_cleand_custom"
weather_file = cleaned_meter_folder / "weather_cleaned.csv"
output_folder = project_folder / "data" / "metadata" / "my_cleand_custom"
report_folder = Path(__file__).resolve().parent / "new_data_report"


#### feature generation
def generate_features(metadata):
    """Add derived building features, columns that can't be computed stay NaN."""
    features = metadata.copy()

    ## which meters each building has, "Yes"/NaN -> 1/0
    for meter in meter_types:
        features[f"has_{meter}"] = features[meter].notna().astype("int8")
    features["n_meter_types"] = features[[f"has_{meter}" for meter in meter_types]].sum(axis=1)

    ## size and age
    features["log_sqm"] = np.log1p(features["sqm"])
    features["building_age"] = reference_year - features["yearbuilt"]
    features["sqm_per_floor"] = features["sqm"] / features["numberoffloors"]

    return features


def total_energy(meter):
    """Total kWh per building over the whole period (2016-2017), NaN if too few readings.
    Also returns the length of the period in hours."""
    meter_file = cleaned_meter_folder / f"{meter}_cleaned.csv"
    readings = pd.read_csv(meter_file, parse_dates=["timestamp"], index_col="timestamp")

    # outliers and gaps are NaN after cleaning, so scale the mean hourly reading up to the whole period
    coverage = readings.notna().sum() / len(readings)
    total = (readings.mean() * len(readings)).where(coverage >= min_energy_coverage)
    return total, len(readings)


def add_energy_intensity(metadata):
    """Add total electricity use (kWh, 2016-2017) and energy intensity (kWh/sqm/year)."""
    features = metadata.copy()
    if not (cleaned_meter_folder / "electricity_cleaned.csv").exists():
        print("No cleaned electricity file, energy intensity left as NaN")
        features["total_energy"] = np.nan
        features["energy_intensity"] = np.nan
        return features

    # only electricity, chilledwater/hotwater use other units at some sites (Eagle, Fox)
    # and give impossible values like 100 000+ kWh/sqm/year
    total, hours = total_energy("electricity")
    features["total_energy"] = total.reindex(features.index)
    years = hours / hours_per_year
    features["energy_intensity"] = features["total_energy"] / years / features["sqm"]
    print(f"Energy intensity found for {features['energy_intensity'].notna().sum()} buildings")
    return features


def add_ambient_temperature(metadata):
    """Add T_amb, the site's average air temperature over the whole period (2016-2017)."""
    features = metadata.copy()
    weather = pd.read_csv(weather_file, usecols=["site_id", "airTemperature"])
    site_temperature = weather.groupby("site_id")["airTemperature"].mean()
    features["T_amb"] = features["site_id"].map(site_temperature)
    print(f"T_amb found for {features['T_amb'].notna().sum()} buildings")
    return features


def drop_missing_energy_intensity(metadata):
    missing = metadata["energy_intensity"].isna()
    print(f"{missing.sum()} buildings dropped without energy intensity")
    return metadata[~missing]


#### keep only neccecery columns
def select_columns(metadata):
    generated_columns = [f"has_{meter}" for meter in meter_types]
    generated_columns += ["n_meter_types", "log_sqm", "building_age", "sqm_per_floor"]
    return metadata[["building_id"] + columns_to_keep + generated_columns]


#### index by building id
def index_by_building(metadata):
    if metadata["building_id"].duplicated().any():
        raise ValueError("Metadata has duplicate building_id values.")
    return metadata.set_index("building_id").sort_index()


#### drop rows missing important data
def drop_missing_rows(metadata):
    missing = metadata[important_columns].isna().any(axis=1)
    print(f"{missing.sum()} buildings dropped for missing {important_columns}")
    return metadata[~missing]


def drop_buildings_without_meters(metadata):
    """Keep buildings that still have a column in at least one cleaned meter file."""
    meter_files = list(cleaned_meter_folder.glob("*_cleaned.csv"))
    meter_files = [file for file in meter_files if file.stem != "weather_cleaned"]
    if not meter_files:
        print(f"No cleaned meter files in {cleaned_meter_folder}, skipping alignment")
        return metadata

    # only read the header, the columns are the building ids
    buildings_with_meters = set()
    for file in meter_files:
        buildings_with_meters.update(pd.read_csv(file, nrows=0).columns)

    has_meters = metadata.index.isin(buildings_with_meters)
    print(f"{(~has_meters).sum()} buildings dropped with no cleaned meter data")
    return metadata[has_meters]


#### overview of each step
def building_ids(metadata):
    return metadata["building_id"] if "building_id" in metadata.columns else metadata.index.to_series()


def log_step(steps_log, step, before, after):
    """Record which buildings and columns a step removed and which columns it added."""
    # building_id moves from a column to the index, so it isn't counted as removed
    columns_before = [column for column in before.columns if column != "building_id"]
    columns_after = [column for column in after.columns if column != "building_id"]
    columns_added = [column for column in columns_after if column not in columns_before]
    columns_removed = [column for column in columns_before if column not in columns_after]
    removed_buildings = sorted(set(building_ids(before)) - set(building_ids(after)))

    steps_log.append({
        "step": step,
        "buildings_before": len(before),
        "buildings_after": len(after),
        "buildings_removed": len(removed_buildings),
        "columns_before": len(columns_before),
        "columns_after": len(columns_after),
        "columns_added": "; ".join(columns_added),
        "columns_removed": "; ".join(columns_removed),
        # how many buildings got a value in each added column
        "values_added": "; ".join(f"{column}: {after[column].notna().sum()}" for column in columns_added),
        "removed_building_ids": "; ".join(removed_buildings),
    })


#### main
def main():
    metadata = pd.read_csv(metadata_file)
    print(f"Loaded {len(metadata)} buildings, {metadata.shape[1]} columns")

    steps = [
        ("feature generation", generate_features),
        ("select columns", select_columns),
        ("index by building_id", index_by_building),
        ("drop rows missing important data", drop_missing_rows),
    ]
    if align_with_cleaned_meters:
        steps.append(("drop buildings without cleaned meters", drop_buildings_without_meters))
    steps.append(("energy intensity", add_energy_intensity))
    steps.append(("ambient temperature", add_ambient_temperature))
    steps.append(("drop buildings without energy intensity", drop_missing_energy_intensity))

    steps_log = []
    for step, function in steps:
        before = metadata
        metadata = function(metadata)
        log_step(steps_log, step, before, metadata)

    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "metadata_cleaned.csv"
    metadata.to_csv(output_file)
    report_folder.mkdir(parents=True, exist_ok=True)
    steps_file = report_folder / "metadata_cleaning_steps.csv"
    pd.DataFrame(steps_log).to_csv(steps_file, index=False)
    print(f"Saved {steps_file}")
    print(f"Finished, {len(metadata)} buildings, {metadata.shape[1]} columns")
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
