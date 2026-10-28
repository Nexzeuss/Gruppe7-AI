import numpy as np
import pandas as pd
from pathlib import Path

#### cleans metadata
#### - generates building features
#### - keeps only the columns needed before feature selection
#### - indexes by building_id so it aligns with the meter columns
#### - drops buildings missing important data
#### - saves to data/metadata/my_cleand_custom

#### tunig variables
reference_year = 2016  # first year of meter data, used for building age
align_with_cleaned_meters = True  # drop buildings with no column left in any cleaned meter file

meter_types = ["electricity", "chilledwater", "steam", "hotwater", "gas", "water", "irrigation", "solar"]

# columns kept before feature selection, the rest are duplicates (sqft, kaggle ids)
# or too sparse to be useful (<15% filled: eui, occupants, rating, energystarscore, ...)
columns_to_keep = [
    "site_id", "primaryspaceusage", "sub_primaryspaceusage", "timezone",
    "sqm", "yearbuilt", "numberoffloors", "lat", "lng",
]

# rows missing any of these are dropped
# yearbuilt/numberoffloors/lat are missing for entire sites, so they are left as NaN instead
important_columns = ["site_id", "primaryspaceusage", "sqm"]


## data paths/managment
project_folder = Path(__file__).resolve().parent.parent
metadata_file = project_folder / "data" / "metadata" / "metadata.csv"
cleaned_meter_folder = project_folder / "data" / "meters" / "my_cleand_custom"
output_folder = project_folder / "data" / "metadata" / "my_cleand_custom"


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


#### main
def main():
    metadata = pd.read_csv(metadata_file)
    print(f"Loaded {len(metadata)} buildings, {metadata.shape[1]} columns")

    metadata = generate_features(metadata)
    metadata = select_columns(metadata)
    metadata = index_by_building(metadata)
    metadata = drop_missing_rows(metadata)
    if align_with_cleaned_meters:
        metadata = drop_buildings_without_meters(metadata)

    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "metadata_cleaned.csv"
    metadata.to_csv(output_file)
    print(f"Finished, {len(metadata)} buildings, {metadata.shape[1]} columns")
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
