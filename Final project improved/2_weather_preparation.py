"""Weather preparation: the assignment's T_amb, plus site climate features (degree-days, variation, humidity)."""

import pandas as pd

from settings import DEGREE_DAY_BASE, OUTPUT_FOLDER, WEATHER_FILE, YEARS, log_step, save_log


SCRIPT = "2_weather_preparation"
CLIMATE_FEATURES = ["T_amb", "HDD", "CDD", "T_std", "dew_mean"]


def step_1_load_weather():
    """Load weather.csv and parse the datetime index."""
    weather = pd.read_csv(WEATHER_FILE, parse_dates=["timestamp"], index_col="timestamp")
    return weather[weather.index.year.isin(YEARS)]


def step_2_site_climate(weather):
    """Climate features per site_id.

    T_amb:    mean air temperature (the assignment's feature).
    HDD/CDD:  heating and cooling degree-days per year with an 18 degC base. A mild site and a site with hot
              summers and cold winters can share the same mean temperature but need very different cooling.
    T_std:    standard deviation of the hourly temperature (how much the climate swings).
    dew_mean: mean dew point temperature, a measure of humidity, which adds to the cooling load.
    """
    temperature = weather["airTemperature"]
    hours = weather.assign(
        heating=(DEGREE_DAY_BASE - temperature).clip(lower=0) / 24,
        cooling=(temperature - DEGREE_DAY_BASE).clip(lower=0) / 24,
    )
    grouped = hours.groupby("site_id")
    # Degree-days per year: the mean hourly contribution times the hours in a year, so gaps do not lower them.
    return pd.DataFrame({
        "T_amb": grouped["airTemperature"].mean(),
        "HDD": grouped["heating"].mean() * 8760,
        "CDD": grouped["cooling"].mean() * 8760,
        "T_std": grouped["airTemperature"].std(),
        "dew_mean": grouped["dewTemperature"].mean(),
    })


def step_3_map_to_buildings(metadata, climate):
    """Map each building's site_id to its site's climate features."""
    return metadata.join(climate, on="site_id")


def step_4_ensure_valid_mapping(metadata):
    """Keep only buildings with a valid site_id and temperature."""
    return metadata.dropna(subset=["site_id", "T_amb"])


def main():
    log = []
    print("Weather preparation")
    metadata = pd.read_csv(OUTPUT_FOLDER / "metadata_prepared.csv", index_col="building_id")
    n = len(metadata)

    weather = step_1_load_weather()
    log_step(log, SCRIPT, 1, f"Load weather.csv ({len(weather):,} hourly rows)", n, n)

    climate = step_2_site_climate(weather)
    log_step(log, SCRIPT, 2, f"Climate features for {len(climate)} sites: {', '.join(CLIMATE_FEATURES)}", n, n)

    metadata = step_3_map_to_buildings(metadata, climate)
    log_step(log, SCRIPT, 3, "Map site climate features to each building", n, n)

    metadata = step_4_ensure_valid_mapping(metadata)
    log_step(log, SCRIPT, 4, "Drop buildings without valid site_id or T_amb", n, len(metadata))

    climate.to_csv(OUTPUT_FOLDER / "site_climate.csv")
    metadata.to_csv(OUTPUT_FOLDER / "metadata_with_climate.csv")
    save_log(log, SCRIPT)
    print(f"Saved {OUTPUT_FOLDER / 'metadata_with_climate.csv'} and site_climate.csv")


if __name__ == "__main__":
    main()
