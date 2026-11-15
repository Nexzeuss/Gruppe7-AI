"""Problem 1, steps 1-10: compute the electricity density (kWh/m2/year) and merge it with the features."""

import pandas as pd

from settings import (
    ELECTRICITY_FILE, MAX_MISSING_SHARE, OUTLIER_QUANTILES, OUTPUT_FOLDER, TARGET, YEARS, log_step, save_log,
)


SCRIPT = "3_electricity_density"


def step_1_read_data():
    """Read the electricity time series and the prepared metadata."""
    electricity = pd.read_csv(ELECTRICITY_FILE, parse_dates=["timestamp"], index_col="timestamp")
    electricity = electricity[electricity.index.year.isin(YEARS)]
    metadata = pd.read_csv(OUTPUT_FOLDER / "metadata_with_tamb.csv", index_col="building_id")
    return electricity, metadata


def step_2_filter_missing(electricity):
    """Drop buildings with more than 5 % missing hourly values."""
    missing_share = electricity.isna().mean()
    return electricity.loc[:, missing_share <= MAX_MISSING_SHARE]


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


def step_7_remove_outliers(density):
    """Keep only buildings whose density lies within [Q1, Q3]."""
    q1, q3 = density.quantile(list(OUTLIER_QUANTILES))
    print(f"    Q1 = {q1:.1f}, Q3 = {q3:.1f} kWh/m2/year")
    return density[density.between(q1, q3)], q1, q3


def step_8_add_temperature(density, metadata):
    """Add the site-level average outdoor temperature T_amb through site_id."""
    site_temperature = pd.read_csv(OUTPUT_FOLDER / "site_temperature.csv", index_col="site_id")["T_amb"]
    data = density.to_frame()
    data["site_id"] = metadata["site_id"].reindex(data.index)
    data["T_amb"] = data["site_id"].map(site_temperature)
    return data


def step_9_merge_features(data, metadata):
    """Merge the remaining metadata features (area, year built, use, ...) with the target."""
    features = metadata.drop(columns=["site_id", "T_amb"])
    return data.join(features, how="left")


def step_10_drop_missing_target(data):
    """Drop buildings without an electricity density."""
    return data.dropna(subset=[TARGET])


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
    density.to_frame().to_csv(OUTPUT_FOLDER / "electricity_density_before_outliers.csv")

    density, q1, q3 = step_7_remove_outliers(density)
    log_step(log, SCRIPT, 7, f"Keep density within [Q1, Q3] = [{q1:.1f}, {q3:.1f}]", n, len(density))
    n = len(density)

    data = step_8_add_temperature(density, metadata)
    log_step(log, SCRIPT, 8, "Add T_amb through site_id", n, len(data))

    data = step_9_merge_features(data, metadata)
    log_step(log, SCRIPT, 9, "Merge metadata features with the target", n, len(data))

    data = step_10_drop_missing_target(data)
    log_step(log, SCRIPT, 10, "Drop rows with missing target", n, len(data))

    data.to_csv(OUTPUT_FOLDER / "electricity_density_dataset.csv")
    save_log(log, SCRIPT)
    print(f"Saved {OUTPUT_FOLDER / 'electricity_density_dataset.csv'} ({len(data)} buildings)")


if __name__ == "__main__":
    main()
