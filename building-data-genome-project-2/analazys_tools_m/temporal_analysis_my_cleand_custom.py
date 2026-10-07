"""Plot custom-cleaned electricity power density by time and building characteristics."""

from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a desktop display.
import matplotlib.pyplot as plt
import seaborn as sns


# Keep the data source easy to switch when a cleaned-data version is needed.
data_type = "my_cleand_custom"
project_folder = Path(__file__).resolve().parent.parent
meter_file = project_folder / "data" / "meters" / data_type / "electricity_cleaned.csv"
metadata_file = project_folder / "data" / "metadata" / "metadata.csv"
output_folder = project_folder / "figures" / "my_cleand_custom" / f"temporal_analysis_{data_type}"


def check_csv(path):
	"""Fail clearly if a data path contains an LFS pointer instead of a CSV."""
	if not path.is_file():
		raise FileNotFoundError(f"Required data file not found: {path}")
	with path.open(encoding="utf-8") as csv_file:
		if csv_file.readline().strip() == "version https://git-lfs.github.com/spec/v1":
			raise RuntimeError(f"{path.name} is a Git LFS pointer. Run `git lfs pull` to fetch the data.")


def save_boxplot(data, category, order, title, output_path):
	"""Save distributions of building-hour power density grouped by time period."""
	fig, ax = plt.subplots(figsize=(13, 6))
	sns.boxplot(
		data=data,
		x=category,
		y="power_per_sqm",
		order=order,
		showfliers=False,
		ax=ax,
	)
	ax.set_title(title)
	ax.set_xlabel(category.replace("_", " ").title())
	ax.set_ylabel("Average electricity power per m²")
	ax.grid(axis="y", alpha=0.25)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def save_histogram(data, column, title, output_path):
	"""Save a histogram, excluding missing values."""
	values = pd.to_numeric(data[column], errors="coerce").dropna()
	if values.empty:
		print(f"Skipping {column}: no numeric values available.")
		return
	fig, ax = plt.subplots(figsize=(9, 6))
	sns.histplot(values, bins="auto", kde=False, ax=ax)
	ax.set(title=title, xlabel=column.replace("_", " ").title(), ylabel="Buildings")
	ax.grid(axis="y", alpha=0.25)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def main():
	check_csv(meter_file)
	check_csv(metadata_file)
	meters = pd.read_csv(meter_file, parse_dates=["timestamp"])
	metadata = pd.read_csv(metadata_file)
	metadata["sqm"] = pd.to_numeric(metadata["sqm"], errors="coerce")
	metadata = metadata[metadata["sqm"] > 0].drop_duplicates("building_id").set_index("building_id")
	building_columns = [column for column in meters.columns if column != "timestamp" and column in metadata.index]
	if not building_columns:
		raise ValueError("No electricity meter columns match buildings with valid sqm metadata.")

	# The input is hourly electricity energy; for one-hour intervals, kWh/hour
	# numerically equals average kW. Divide each reading by its building area.
	power_per_sqm = meters[building_columns].div(metadata.loc[building_columns, "sqm"], axis=1)
	long = power_per_sqm.assign(timestamp=meters["timestamp"]).melt(
		id_vars="timestamp", var_name="building_id", value_name="power_per_sqm"
	)
	long = long.dropna(subset=["timestamp", "power_per_sqm"])
	long["hour"] = long["timestamp"].dt.hour
	long["weekday"] = long["timestamp"].dt.dayofweek
	long["month"] = long["timestamp"].dt.month

	output_folder.mkdir(parents=True, exist_ok=True)
	# Monday=0 through Sunday=6, labeled for readability.
	weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	long["weekday"] = pd.Categorical(long["weekday"].map(dict(enumerate(weekday_names))), categories=weekday_names, ordered=True)
	month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
	long["month"] = pd.Categorical(long["month"].map(dict(enumerate(month_names, start=1))), categories=month_names, ordered=True)

	save_boxplot(long, "hour", list(range(24)), "Electricity power density by hour of day", output_folder / "power_density_by_hour.png")
	save_boxplot(long, "weekday", weekday_names, "Electricity power density by weekday", output_folder / "power_density_by_weekday.png")
	save_boxplot(long, "month", month_names, "Electricity power density by month", output_folder / "power_density_by_month.png")

	metadata_histograms = metadata.reset_index()
	metadata_histograms["electricity_density"] = long.groupby("building_id")["power_per_sqm"].mean().reindex(metadata_histograms["building_id"]).to_numpy()
	for column, label, filename in [
		("electricity_density", "Mean electricity power density per building", "histogram_electricity_density.png"),
		("sqm", "Building floor area", "histogram_sqm.png"),
		("yearbuilt", "Building year built", "histogram_yearbuilt.png"),
		("numberoffloors", "Number of floors", "histogram_numberoffloors.png"),
		("occupants", "Number of occupants", "histogram_occupants.png"),
	]:
		if column in metadata_histograms:
			save_histogram(metadata_histograms, column, label, output_folder / filename)
		else:
			print(f"Skipping {column}: metadata column is unavailable.")

	print(f"Analyzed {len(building_columns)} electricity meters and {len(long):,} valid hourly observations.")
	print(f"Temporal plots and histograms saved in: {output_folder}")


if __name__ == "__main__":
	main()