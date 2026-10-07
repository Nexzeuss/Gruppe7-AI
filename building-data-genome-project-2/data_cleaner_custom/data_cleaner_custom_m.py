import numpy as np
import pandas as pd
from pathlib import Path

#### cleans data 
#### - defined by variables
#### - creates new forlder my_cleand_custom

#### tunig variables
# uniqe value threshold, def = 1
uniqe_value_threshold = 1
consecutive_missing_threshold = 100*24
lower_quantile = 0.25
upper_quantile = 0.75


## data paths/managment
data = "raw"
project_folder = Path(__file__).resolve().parent.parent
output_folder = project_folder / "data" / "meters" / "my_cleand_custom"

electricity_file = project_folder / "data" / "meters" / "raw" / "electricity.csv"
chilledwater_file = project_folder / "data" / "meters" / "raw" / "chilledwater.csv"
weather_file = project_folder / "data" / "weather" / "weather.csv"
metadata_file = project_folder / "data" / "metadata" / "metadata.csv"

#### handle missing data

def find_missing_data_ids(data, *, check_negative_and_runs=False):

    ## remove columns with only a single value
    ids_to_remove = []
    for col in data.columns:  
        if data[col].nunique() <= uniqe_value_threshold:
            ids_to_remove.append(col)

    ## remove duplicate columns
    duplicate_columns = data.T.duplicated()
    ids_to_remove.extend(data.columns[duplicate_columns])

    if check_negative_and_runs:
        numeric_data = data.select_dtypes(include="number")

        ## handle columns whose non-missing numeric values are all negative
        has_values = numeric_data.notna().any()
        only_negative = numeric_data.lt(0).where(numeric_data.notna(), True).all()
        ids_to_remove.extend(numeric_data.columns[has_values & only_negative])

        ## handle long consecutive runs of missing or zero readings
        invalid_readings = numeric_data.isna() | numeric_data.eq(0)
        long_invalid_run = invalid_readings.apply(
            lambda values: values.astype("int8")
            .groupby((~values).cumsum())
            .sum()
            .ge(consecutive_missing_threshold)
            .any()
        )
        ids_to_remove.extend(long_invalid_run[long_invalid_run].index)

    return remove_id_dupes(ids_to_remove)


def clean_data(data, *, check_negative_and_runs=False):
    """Drop flagged columns, then replace numeric outliers with NaN."""
    ids_to_remove = find_missing_data_ids(
        data, check_negative_and_runs=check_negative_and_runs
    )
    return filter_quantile_outliers(data.drop(columns=ids_to_remove))


def filter_quantile_outliers(data):
    """Replace numeric readings outside the configured quantiles with NaN."""
    filtered_data = data.copy()
    numeric_columns = filtered_data.select_dtypes(include="number").columns
    lower_bounds = filtered_data[numeric_columns].quantile(lower_quantile)
    upper_bounds = filtered_data[numeric_columns].quantile(upper_quantile)
    filtered_data[numeric_columns] = filtered_data[numeric_columns].where(
        filtered_data[numeric_columns].ge(lower_bounds)
        & filtered_data[numeric_columns].le(upper_bounds)
    )
    return filtered_data


def pad_hourly_sequence(data, group_columns=None):
    """Add missing hourly timestamps, leaving newly created readings as NaN."""
    group_columns = group_columns or []
    padded_groups = []

    groups = data.groupby(group_columns, dropna=False) if group_columns else [(None, data)]
    for key, group in groups:
        group = group.drop(columns=group_columns).drop_duplicates(
            "timestamp", keep="first"
        )
        group = group.set_index("timestamp").sort_index()
        if group.empty:
            continue

        full_range = pd.date_range(
            group.index.min(), group.index.max(), freq="h", name="timestamp"
        )
        group = group.reindex(full_range)

        if group_columns:
            key_values = key if isinstance(key, tuple) else (key,)
            for column, value in zip(group_columns, key_values):
                group[column] = value

        padded_groups.append(group.reset_index())

    if not padded_groups:
        return data.copy()
    return pd.concat(padded_groups, ignore_index=True)[data.columns]

def remove_id_dupes(ids):
    cleaned_ids = []
    for meter_id in ids:
        if meter_id not in cleaned_ids:
            cleaned_ids.append(meter_id)
    return cleaned_ids

#### detect & handle outliers


#### tranform/normalize


#### sequense


#### pad where needed



#### main
def main():
    #### get data
    electricity_data = pd.read_csv(electricity_file, parse_dates=(["timestamp"]))
    chilledwater_data = pd.read_csv(chilledwater_file, parse_dates=(["timestamp"]))
    weather_data = pd.read_csv(weather_file, parse_dates=(["timestamp"]))
    metadata = pd.read_csv(metadata_file)

    #### pad missing hourly timestamps
    electricity_data = pad_hourly_sequence(electricity_data)
    chilledwater_data = pad_hourly_sequence(chilledwater_data)
    weather_data = pad_hourly_sequence(weather_data, group_columns=["site_id"])

    output_folder.mkdir(parents=True, exist_ok=True)

    #### find and report columns to remove
    ids = []

    print("Analysing electricity data")
    ids_electrical = find_missing_data_ids(
        electricity_data, check_negative_and_runs=True
    )
    previous_count = len(ids)
    ids.extend(ids_electrical)
    ids = remove_id_dupes(ids)
    print(f"{len(ids) - previous_count} new solutions found")

    print("Analysing chilled-water data")
    ids_chilledwater = find_missing_data_ids(chilledwater_data)
    previous_count = len(ids)
    ids.extend(ids_chilledwater)
    ids = remove_id_dupes(ids)
    print(f"{len(ids) - previous_count} new solutions found")

    print("Analysing weather data")
    ids_weather = find_missing_data_ids(
        weather_data, check_negative_and_runs=True
    )
    previous_count = len(ids)
    ids.extend(ids_weather)
    ids = remove_id_dupes(ids)
    print(f"{len(ids) - previous_count} new solutions found")

    print(f"Finished, total count {len(ids)}, ids: {ids}")

    #### clean electricity data
    print("Cleaning electricity data")
    electricity_data = filter_quantile_outliers(
        electricity_data.drop(columns=ids_electrical)
    )
    electricity_data.to_csv(output_folder / "electricity_cleaned.csv", index=False)
    print(f"Saved {output_folder / 'electricity_cleaned.csv'}")

    #### clean chilled-water data
    print("Cleaning chilled-water data")
    chilledwater_data = filter_quantile_outliers(
        chilledwater_data.drop(columns=ids_chilledwater)
    )
    chilledwater_data.to_csv(output_folder / "chilledwater_cleaned.csv", index=False)
    print(f"Saved {output_folder / 'chilledwater_cleaned.csv'}")

    #### clean weather data
    print("Cleaning weather data")
    weather_data = filter_quantile_outliers(
        weather_data.drop(columns=ids_weather)
    )
    weather_data.to_csv(output_folder / "weather_cleaned.csv", index=False)
    print(f"Saved {output_folder / 'weather_cleaned.csv'}")




if __name__ == "__main__":
    main()


