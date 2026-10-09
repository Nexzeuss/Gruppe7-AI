"""Compare selected raw and custom-cleaned weather measurements."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


project_folder = Path(__file__).resolve().parent.parent
raw_weather_file = project_folder / "data" / "weather" / "weather.csv"
clean_weather_file = (
    project_folder / "data" / "meters" / "my_cleand_custom" / "weather_cleaned.csv"
)
output_folder = project_folder / "figures" / "my_cleand_custom" / "weather_comparison"

weather_variables = ["airTemperature", "dewTemperature", "windSpeed"]


def load_weather(path):
    """Load weather data and keep only comparison fields available in the file."""
    data = pd.read_csv(path, parse_dates=["timestamp"])
    columns = [
        column for column in weather_variables
        if column in data.columns
    ]
    if not columns:
        raise ValueError(f"No selected weather variables found in {path}.")
    return data[["timestamp", "site_id", *columns]]


def main():
    for path in (raw_weather_file, clean_weather_file):
        if not path.is_file():
            raise FileNotFoundError(f"Weather data file not found: {path}")

    raw = load_weather(raw_weather_file)
    cleaned = load_weather(clean_weather_file)
    variables = [column for column in weather_variables if column in raw and column in cleaned]
    comparison = raw.merge(
        cleaned,
        on=["timestamp", "site_id"],
        how="inner",
        suffixes=("_raw", "_cleaned"),
    )
    if comparison.empty:
        raise ValueError("Raw and cleaned weather files have no matching site/timestamp rows.")

    output_folder.mkdir(parents=True, exist_ok=True)
    site_id = comparison["site_id"].dropna().iloc[0]
    site_data = comparison[comparison["site_id"] == site_id].sort_values("timestamp")

    for variable in variables:
        raw_column = f"{variable}_raw"
        cleaned_column = f"{variable}_cleaned"
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(site_data["timestamp"], site_data[raw_column], label="Raw", alpha=0.7)
        axes[0].plot(site_data["timestamp"], site_data[cleaned_column], label="Cleaned", alpha=0.8)
        axes[0].set_title(f"{variable} over time — site {site_id}")
        axes[0].set_ylabel(variable)
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        difference = site_data[cleaned_column] - site_data[raw_column]
        axes[1].plot(site_data["timestamp"], difference, color="#c44e52")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title(f"Cleaning difference (cleaned − raw) — site {site_id}")
        axes[1].set_xlabel("Timestamp")
        axes[1].set_ylabel(f"Change in {variable}")
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(output_folder / f"{variable}_raw_vs_cleaned.png", dpi=180)
        plt.close(fig)

    print(f"Saved weather comparison plots for site {site_id} in {output_folder}")


if __name__ == "__main__":
    main()