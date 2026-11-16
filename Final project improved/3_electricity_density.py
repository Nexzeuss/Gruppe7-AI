"""Electricity density: the recipe's steps 1-6, an outlier flag instead of the Q1-Q3 filter, and the merge."""

import numpy as np
import pandas as pd

from settings import (
    ELECTRICITY_FILE, MAX_MISSING_SHARE, OUTLIER_IQR_FACTOR, OUTPUT_FOLDER, TARGET, YEARS, log_step, save_log,
)


SCRIPT = "3_electricity_density"


def step_1_read_data():
    """Read the electricity time series and the prepared metadata."""
    electricity = pd.read_csv(ELECTRICITY_FILE, parse_dates=["timestamp"], index_col="timestamp")
    electricity = electricity[electricity.index.year.isin(YEARS)]
    metadata = pd.read_csv(OUTPUT_FOLDER / "metadata_with_climate.csv", index_col="building_id")
    return electricity, metadata


def step_2_filter_missing(electricity):
    """Drop buildings with more than 5 % missing hourly values."""
    return electricity.loc[:, electricity.isna().mean() <= MAX_MISSING_SHARE]


def step_3_interpolate_and_backfill(electricity):
    """Interpolate gaps linearly, then backward-fill missing values at the start of the series."""
    return electricity.interpolate(method="linear").bfill()


def step_4_sum_per_year(electricity):
    """Total electricity use (kWh) for each building in 2016 and in 2017."""
    return electricity.groupby(electricity.index.year).sum()


def step_5_average_years(yearly):
    """Average annual electricity use over the two years."""
    return yearly.loc[YEARS].mean()


def step_6_divide_by_area(annual, metadata):
    """Divide by floor area to get kWh/m2/year; buildings without metadata get no value."""
    density = (annual / metadata["sqm"].reindex(annual.index)).rename(TARGET)
    return density.rename_axis("building_id")


def step_7_flag_outliers(density):
    """Flag implausible densities with the IQR rule on log(density); the experiments decide whether to drop them.

    Buildings with zero consumption cannot be logged and are always flagged.
    """
    positive = density[density > 0]
    q1, q3 = np.log(positive).quantile([0.25, 0.75])
    low, high = np.exp(q1 - OUTLIER_IQR_FACTOR * (q3 - q1)), np.exp(q3 + OUTLIER_IQR_FACTOR * (q3 - q1))
    print(f"    Plausible range: [{low:.1f}, {high:.1f}] kWh/m2/year")
    is_outlier = ~density.between(low, high)
    return is_outlier.rename("is_outlier"), low, high


def step_8_merge_features(density, is_outlier, metadata):
    """Merge the target, the outlier flag and all metadata and climate features."""
    return pd.concat([density, is_outlier], axis=1).join(metadata, how="left")


def main():
    log = []
    print("Electricity density")
    electricity, metadata = step_1_read_data()
    n = electricity.shape[1]
    log_step(log, SCRIPT, 1, f"Read electricity ({len(electricity):,} hours) and metadata", n, n)

    electricity = step_2_filter_missing(electricity)
    log_step(log, SCRIPT, 2, f"Drop buildings with more than {MAX_MISSING_SHARE:.0%} missing hours", n, electricity.shape[1])
    n = electricity.shape[1]

    missing_values = int(electricity.isna().sum().sum())
    electricity = step_3_interpolate_and_backfill(electricity)
    log_step(log, SCRIPT, 3, f"Interpolate and backward-fill {missing_values:,} missing hourly values", n, n)

    yearly = step_4_sum_per_year(electricity)
    log_step(log, SCRIPT, 4, "Sum electricity use per year", n, n)

    annual = step_5_average_years(yearly)
    log_step(log, SCRIPT, 5, "Average the two years", n, n)

    density = step_6_divide_by_area(annual, metadata).dropna()
    log_step(log, SCRIPT, 6, "Divide by sqm (buildings without prepared metadata removed)", n, len(density))
    n = len(density)

    is_outlier, low, high = step_7_flag_outliers(density)
    log_step(log, SCRIPT, 7, f"Flag outliers outside [{low:.1f}, {high:.1f}] ({int(is_outlier.sum())} flagged, kept)", n, n)

    data = step_8_merge_features(density, is_outlier, metadata)
    log_step(log, SCRIPT, 8, "Merge metadata and climate features with the target", n, len(data))

    data.to_csv(OUTPUT_FOLDER / "electricity_density_dataset.csv")
    save_log(log, SCRIPT)
    print(f"Saved {OUTPUT_FOLDER / 'electricity_density_dataset.csv'} ({len(data)} buildings)")


if __name__ == "__main__":
    main()
