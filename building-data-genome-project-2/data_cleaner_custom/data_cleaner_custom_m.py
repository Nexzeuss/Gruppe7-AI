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
iqr_multiplier = 1.5
rolling_window = 720  # 30 days of hourly readings
min_periods = 24  # allow rolling bounds after one day of readings
remove_meters_over_missing_threshold = False
missing_data_threshold = 0.05


## data paths/managment
data = "raw"
project_folder = Path(__file__).resolve().parent.parent
output_folder = project_folder / "data" / "meters" / "my_cleand_custom"
report_folder = Path(__file__).resolve().parent / "new_data_report"
removal_report_rows = []

electricity_file = project_folder / "data" / "meters" / "raw" / "electricity.csv"
chilledwater_file = project_folder / "data" / "meters" / "raw" / "chilledwater.csv"
gas_file = project_folder / "data" / "meters" / "raw" / "gas.csv"
hotwater_file = project_folder / "data" / "meters" / "raw" / "hotwater.csv"
irrigation_file = project_folder / "data" / "meters" / "raw" / "irrigation.csv"
solar_file = project_folder / "data" / "meters" / "raw" / "solar.csv"
steam_file = project_folder / "data" / "meters" / "raw" / "steam.csv"
water_file = project_folder / "data" / "meters" / "raw" / "water.csv"
weather_file = project_folder / "data" / "weather" / "weather.csv"
metadata_file = project_folder / "data" / "metadata" / "metadata.csv"

#### handle missing data

def find_missing_data_ids(data, *, check_negative_and_runs=False, report=None):

    protected_columns = [column for column in ("timestamp", "site_id") if column in data]
    measurement_data = data.drop(columns=protected_columns)

    ## remove columns with only a single value
    ids_to_remove = []
    for col in measurement_data.columns:
        if measurement_data[col].nunique() <= uniqe_value_threshold:
            ids_to_remove.append(col)
    if report is not None:
        report("single-value columns", ids_to_remove)

    ## remove duplicate columns
    duplicate_columns = measurement_data.T.duplicated()
    duplicate_ids = list(measurement_data.columns[duplicate_columns])
    ids_to_remove.extend(duplicate_ids)
    if report is not None:
        report("duplicate columns", duplicate_ids)

    if check_negative_and_runs:
        numeric_data = measurement_data.select_dtypes(include="number")

        ## handle columns whose non-missing numeric values are all negative
        has_values = numeric_data.notna().any()
        only_negative = numeric_data.lt(0).where(numeric_data.notna(), True).all()
        negative_ids = list(numeric_data.columns[has_values & only_negative])
        ids_to_remove.extend(negative_ids)
        if report is not None:
            report("all-negative readings", negative_ids)

        ## handle long consecutive runs of missing or zero readings
        invalid_readings = numeric_data.isna() | numeric_data.eq(0)
        long_invalid_run = invalid_readings.apply(
            lambda values: values.astype("int8")
            .groupby((~values).cumsum())
            .sum()
            .ge(consecutive_missing_threshold)
            .any()
        )
        run_ids = list(long_invalid_run[long_invalid_run].index)
        ids_to_remove.extend(run_ids)
        if report is not None:
            report(f"long missing/zero run (≥{consecutive_missing_threshold} hours)", run_ids)

    return remove_id_dupes(ids_to_remove)


def clean_data(data, *, check_negative_and_runs=False):
    """Drop unusable columns, then replace rolling-IQR outliers with NaN."""
    ids_to_remove = find_missing_data_ids(
        data, check_negative_and_runs=check_negative_and_runs
    )
    return filter_rolling_iqr_outliers(data.drop(columns=ids_to_remove))


def filter_rolling_iqr_outliers(data):
    """Apply a centered rolling IQR fence independently to each numeric column."""
    filtered_data = data.copy()
    numeric_columns = filtered_data.select_dtypes(include="number").columns
    if numeric_columns.empty:
        return filtered_data

    # Rolling bounds adapt to seasonal and operating changes instead of using
    # one global range for a meter's entire time series. Centering also avoids
    # applying a past-only baseline at the end of the sequence.
    readings = filtered_data[numeric_columns]
    rolling = readings.rolling(
        window=rolling_window, min_periods=min_periods, center=True
    )
    first_quartile = rolling.quantile(lower_quantile)
    third_quartile = rolling.quantile(upper_quantile)
    iqr = third_quartile - first_quartile
    lower_bounds = first_quartile - iqr_multiplier * iqr
    upper_bounds = third_quartile + iqr_multiplier * iqr
    filtered_data[numeric_columns] = readings.where(
        readings.ge(lower_bounds) & readings.le(upper_bounds)
    )
    return filtered_data


def filter_weather_by_site(data):
    """Apply rolling IQR per site and numeric weather variable only."""
    if "site_id" not in data.columns:
        raise ValueError("Weather data must include a 'site_id' column.")

    filtered_sites = []
    for _, site_data in data.groupby("site_id", sort=False, dropna=False):
        site_data = site_data.sort_values("timestamp").copy()
        timestamps = site_data["timestamp"].copy()
        site_ids = site_data["site_id"].copy()
        readings = site_data.drop(columns=["timestamp", "site_id"])
        readings = filter_rolling_iqr_outliers(readings)
        readings.insert(0, "site_id", site_ids.to_numpy())
        readings.insert(0, "timestamp", timestamps.to_numpy())
        filtered_sites.append(readings)

    if not filtered_sites:
        return data.copy()
    return pd.concat(filtered_sites, ignore_index=True)[data.columns]


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


def clean_and_save_meter(file_path, ids_already_found):
    """Clean one meter type, report newly flagged columns, and save it."""
    meter_data = pd.read_csv(file_path, parse_dates=["timestamp"])
    meter_data = pad_hourly_sequence(meter_data)
    check_negative_and_runs = file_path.stem == "electricity"

    print(f"Analysing {file_path.stem} data")
    def log_removals(reason, ids):
        unique_ids = remove_id_dupes(ids)
        print(f"  Removed by {reason}: {len(unique_ids)} meter(s): {unique_ids}")
        removal_report_rows.append({
            "data_type": file_path.stem,
            "process": reason,
            "quantity": len(unique_ids),
        })

    ids_to_remove = find_missing_data_ids(
        meter_data, check_negative_and_runs=check_negative_and_runs,
        report=log_removals,
    )
    if remove_meters_over_missing_threshold:
        measurement_columns = [c for c in meter_data.columns if c not in ("timestamp", "site_id")]
        missing_share = meter_data[measurement_columns].isna().mean()
        missing_ids = list(missing_share[missing_share >= missing_data_threshold].index)
        log_removals(f"missing data ≥{missing_data_threshold:.0%}", missing_ids)
        ids_to_remove.extend(missing_ids)
    ids_to_remove = remove_id_dupes(ids_to_remove)
    new_ids = [meter_id for meter_id in ids_to_remove if meter_id not in ids_already_found]
    ids_already_found.extend(new_ids)
    cleaned_data = filter_rolling_iqr_outliers(meter_data.drop(columns=ids_to_remove))
    numeric_columns = cleaned_data.select_dtypes(include="number").columns
    before_outlier_filter = meter_data.drop(columns=ids_to_remove)[numeric_columns]
    outlier_count = int((before_outlier_filter.notna() & cleaned_data[numeric_columns].isna()).sum().sum())
    print(f"  Outlier readings replaced with NaN: {outlier_count}")
    removal_report_rows.append({
        "data_type": file_path.stem,
        "process": "rolling-IQR outlier readings removed",
        "quantity": outlier_count,
    })
    if "timestamp" in cleaned_data:
        indexed = cleaned_data.set_index("timestamp")
        indexed[numeric_columns] = indexed[numeric_columns].interpolate(method="time").bfill()
        cleaned_data = indexed.reset_index()
    output_file = output_folder / f"{file_path.stem}_cleaned.csv"
    cleaned_data.to_csv(output_file, index=False)
    print(f"Saved {output_file}")
    return ids_already_found

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
    #### get weather data
    weather_data = pd.read_csv(weather_file, parse_dates=(["timestamp"]))

    #### pad missing hourly timestamps for each weather site
    weather_data = pad_hourly_sequence(weather_data, group_columns=["site_id"])

    output_folder.mkdir(parents=True, exist_ok=True)

    #### clean and save all meter types
    ids = []
    ids = clean_and_save_meter(electricity_file, ids)
    ids = clean_and_save_meter(chilledwater_file, ids)
    ids = clean_and_save_meter(gas_file, ids)
    ids = clean_and_save_meter(hotwater_file, ids)
    ids = clean_and_save_meter(irrigation_file, ids)
    ids = clean_and_save_meter(solar_file, ids)
    ids = clean_and_save_meter(steam_file, ids)
    ids = clean_and_save_meter(water_file, ids)

    #### clean weather data
    print("Analysing weather data")
    def log_weather_removals(reason, ids):
        unique_ids = remove_id_dupes(ids)
        print(f"  Removed by {reason}: {len(unique_ids)} column(s): {unique_ids}")
        removal_report_rows.append({
            "data_type": "weather",
            "process": reason,
            "quantity": len(unique_ids),
        })

    ids_weather = find_missing_data_ids(
        weather_data, check_negative_and_runs=True, report=log_weather_removals
    )
    new_weather_ids = [meter_id for meter_id in ids_weather if meter_id not in ids]
    ids.extend(new_weather_ids)
    print(f"{len(new_weather_ids)} new solutions found")
    weather_data = filter_weather_by_site(weather_data.drop(columns=ids_weather))
    numeric_columns = weather_data.select_dtypes(include="number").columns.drop("site_id", errors="ignore")
    weather_data = weather_data.sort_values(["site_id", "timestamp"])
    weather_data[numeric_columns] = weather_data.groupby("site_id", dropna=False)[numeric_columns].transform(
        lambda values: values.interpolate().bfill()
    )
    weather_data.to_csv(output_folder / "weather_cleaned.csv", index=False)
    print(f"Saved {output_folder / 'weather_cleaned.csv'}")

    report_folder.mkdir(parents=True, exist_ok=True)
    report_file = report_folder / "meter_removal_quantities.csv"
    pd.DataFrame(removal_report_rows, columns=["data_type", "process", "quantity"]).to_csv(
        report_file, index=False
    )
    print(f"Saved removal quantities report: {report_file}")

    print(f"Finished, total count {len(ids)}, ids: {ids}")



if __name__ == "__main__":
    main()


