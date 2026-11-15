"""Weather Data Preparation, steps 1-6 of the assignment: adds T_amb to the metadata."""

import pandas as pd

from settings import OUTPUT_FOLDER, WEATHER_FILE, log_step, save_log


SCRIPT = "2_weather_preparation"


def step_1_load_weather():
    """Load weather.csv and parse the datetime index."""
    return pd.read_csv(WEATHER_FILE, parse_dates=["timestamp"], index_col="timestamp")


def step_2_group_by_site(weather):
    """Group the weather data by site_id."""
    return weather.groupby("site_id")


def step_3_average_temperature(grouped):
    """Average airTemperature for each site over the entire dataset."""
    return grouped["airTemperature"].mean()


def step_4_map_to_buildings(metadata, site_temperature):
    """Map each building's site_id to its site's average air temperature."""
    return metadata["site_id"].map(site_temperature)


def step_5_add_t_amb(metadata, mapped_temperature):
    """Create the column T_amb with the mapped average temperature."""
    metadata = metadata.copy()
    metadata["T_amb"] = mapped_temperature
    return metadata


def step_6_ensure_valid_mapping(metadata):
    """Keep only buildings with a valid site_id and temperature."""
    return metadata.dropna(subset=["site_id", "T_amb"])


def main():
    log = []
    print("Weather data preparation")
    metadata = pd.read_csv(OUTPUT_FOLDER / "metadata_prepared.csv", index_col="building_id")
    n = len(metadata)

    weather = step_1_load_weather()
    log_step(log, SCRIPT, 1, f"Load weather.csv ({len(weather):,} hourly rows)", n, n)

    grouped = step_2_group_by_site(weather)
    log_step(log, SCRIPT, 2, f"Group weather by site_id ({grouped.ngroups} sites)", n, n)

    site_temperature = step_3_average_temperature(grouped)
    log_step(log, SCRIPT, 3, "Average airTemperature per site", n, n)

    mapped_temperature = step_4_map_to_buildings(metadata, site_temperature)
    log_step(log, SCRIPT, 4, "Map site temperature to each building", n, n)

    metadata = step_5_add_t_amb(metadata, mapped_temperature)
    log_step(log, SCRIPT, 5, "Add column T_amb", n, n)

    metadata = step_6_ensure_valid_mapping(metadata)
    log_step(log, SCRIPT, 6, "Drop buildings without valid site_id or T_amb", n, len(metadata))

    site_temperature.rename("T_amb").to_csv(OUTPUT_FOLDER / "site_temperature.csv")
    metadata.to_csv(OUTPUT_FOLDER / "metadata_with_tamb.csv")
    save_log(log, SCRIPT)
    print(f"Saved {OUTPUT_FOLDER / 'metadata_with_tamb.csv'} and site_temperature.csv")


if __name__ == "__main__":
    main()
