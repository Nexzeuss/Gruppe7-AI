"""Correlate cleaned energy use per square metre with metadata, weather, and time."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a desktop display.
import matplotlib.pyplot as plt
import seaborn as sns


# Select one cleaned meter dataset. Values are assumed to use the units in the
# repository's cleaned files; divide by metadata sqm to compare per floor area.
data_type = "cleaned"
meter_type = "electricity"  # e.g. electricity, gas, chilledwater, solar

project_folder = Path(__file__).resolve().parent.parent
meter_file = project_folder / "data" / "meters" / data_type / f"{meter_type}_cleaned.csv"
weather_file = project_folder / "data" / "weather" / "weather.csv"
metadata_file = project_folder / "data" / "metadata" / "metadata.csv"
output_folder = project_folder / "figures" / f"correlation_{data_type}_{meter_type}"


def save_correlation_plot(matrix, title, output_path):
	"""Save one full square, annotated correlation heatmap."""
	fig, ax = plt.subplots(figsize=(max(8, len(matrix.columns) * 0.8), max(6, len(matrix) * 0.6)))
	image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
	ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
	ax.set_yticks(range(len(matrix.index)), matrix.index, rotation=0)
	for row in range(len(matrix.index)):
		for column in range(len(matrix.columns)):
			value = matrix.iloc[row, column]
			label = "NA" if pd.isna(value) else f"{value:.2f}"
			ax.text(column, row, label, ha="center", va="center", fontsize=8)
	fig.colorbar(image, ax=ax, label="Correlation")
	ax.set_title(title)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def add_cycle_features(frame, values, period, prefix):
	"""Add sine/cosine columns without changing the observations' order."""
	frame[f"{prefix}_sin"] = np.sin(2 * np.pi * values / period)
	frame[f"{prefix}_cos"] = np.cos(2 * np.pi * values / period)


def save_category_boxplot(data, category, title, output_path):
	"""Plot per-building usage distributions for a categorical metadata field."""
	plot_data = data[[category, "mean_usage_per_sqm"]].dropna().copy()
	order = plot_data.groupby(category)["mean_usage_per_sqm"].median().sort_values().index
	fig, ax = plt.subplots(figsize=(11, max(6, len(order) * 0.4)))
	sns.boxplot(
		data=plot_data,
		y=category,
		x="mean_usage_per_sqm",
		order=list(order),
		showfliers=False,
		ax=ax,
	)
	ax.set_title(title)
	ax.set_xlabel("Mean energy usage per m²")
	ax.set_ylabel(category.replace("_", " ").title())
	fig.tight_layout()
	fig.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def mean_usage_per_sqm(file_path, metadata):
	"""Load one meter type and calculate each building's mean usage per m²."""
	data = pd.read_csv(file_path)
	columns = [column for column in data.columns if column != "timestamp" and column in metadata.index]
	if not columns:
		return pd.Series(dtype=float)
	per_sqm = data[columns].div(metadata.loc[columns, "sqm"], axis=1)
	return per_sqm.mean().rename(file_path.stem.removesuffix("_cleaned") + "_usage_per_sqm")


def main():
	for path in (meter_file, weather_file, metadata_file):
		if not path.is_file():
			raise FileNotFoundError(f"Required data file not found: {path}")

	meters = pd.read_csv(meter_file, parse_dates=["timestamp"])
	weather = pd.read_csv(weather_file, parse_dates=["timestamp"])
	metadata = pd.read_csv(metadata_file)
	usage_columns = [column for column in meters.columns if column != "timestamp"]

	# Keep buildings with a known, positive floor area so usage can be normalized.
	metadata["sqm"] = pd.to_numeric(metadata["sqm"], errors="coerce")
	metadata = metadata[metadata["sqm"] > 0].copy()
	metadata = metadata.drop_duplicates("building_id").set_index("building_id")
	usage_columns = [column for column in usage_columns if column in metadata.index]
	if not usage_columns:
		raise ValueError("No meter columns match buildings with valid sqm metadata.")
	output_folder.mkdir(parents=True, exist_ok=True)

	# Normalize each building's readings first, so each building contributes on
	# a per-area basis rather than large buildings dominating the totals.
	usage_per_sqm = meters[usage_columns].div(metadata.loc[usage_columns, "sqm"], axis=1)
	mean_usage = usage_per_sqm.mean().rename("mean_usage_per_sqm")
	building_data = mean_usage.rename_axis("building_id").reset_index().merge(
		metadata.reset_index(), on="building_id", how="left"
	)
	additional_meter_types = ("steam", "chilledwater", "hotwater", "solar")
	for additional_type in additional_meter_types:
		additional_file = project_folder / "data" / "meters" / "cleaned" / f"{additional_type}_cleaned.csv"
		if not additional_file.is_file():
			print(f"Skipping {additional_type}: file not found ({additional_file.name}).")
			continue
		additional_usage = mean_usage_per_sqm(additional_file, metadata)
		building_data = building_data.merge(
			additional_usage.rename_axis("building_id").reset_index(),
			on="building_id",
			how="left",
		)

	# Matrix 1: general numeric building characteristics.
	general = building_data[
		[
			"mean_usage_per_sqm", "steam_usage_per_sqm", "chilledwater_usage_per_sqm",
			"hotwater_usage_per_sqm", "solar_usage_per_sqm", "numberoffloors",
			"occupants", "yearbuilt", "lat", "lng",
		]
	].apply(pd.to_numeric, errors="coerce")
	general_matrix = general.corr()

	# Matrix 2: heating type. Dummy columns make categories numeric for correlation.
	heating = pd.get_dummies(building_data["heatingtype"], prefix="heating", dummy_na=False, dtype=float)
	heating_matrix = pd.concat([building_data["mean_usage_per_sqm"], heating], axis=1).corr()
	heating_usage_matrix = heating_matrix.loc[heating.columns, ["mean_usage_per_sqm"]].T

	# Matrix 3: weather and cyclical time features. Do not use raw hour, weekday,
	# or month numbers: adjacent values at the cycle boundary must remain close.
	building_sites = metadata["site_id"]
	site_hour_usage = {}
	for site, site_buildings in building_sites.groupby(building_sites).groups.items():
		site_columns = [column for column in usage_columns if column in site_buildings]
		if site_columns:
			# Mean across buildings gives sites equal building-level contributions.
			site_hour_usage[site] = usage_per_sqm[site_columns].mean(axis=1)

	usage_by_site = pd.DataFrame(site_hour_usage)
	usage_by_site["timestamp"] = meters["timestamp"]
	usage_long = usage_by_site.melt(
		id_vars="timestamp", var_name="site_id", value_name="usage_per_sqm"
	)
	weather_joined = usage_long.merge(weather, on=["timestamp", "site_id"], how="inner")
	timestamps = weather_joined["timestamp"]
	add_cycle_features(weather_joined, timestamps.dt.hour, 24, "hour")
	add_cycle_features(weather_joined, timestamps.dt.dayofweek, 7, "weekday")
	add_cycle_features(weather_joined, timestamps.dt.dayofyear - 1, 365.2425, "season")

	# Wind direction is also circular (0 degrees is adjacent to 360 degrees).
	weather_joined["windDirection"] = pd.to_numeric(weather_joined["windDirection"], errors="coerce")
	weather_joined["wind_direction_sin"] = np.sin(np.deg2rad(weather_joined["windDirection"]))
	weather_joined["wind_direction_cos"] = np.cos(np.deg2rad(weather_joined["windDirection"]))
	weather_columns = [
		column for column in weather.select_dtypes(include="number").columns
		if column not in ("site_id", "windDirection")
	]
	time_columns = [
		"hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
		"season_sin", "season_cos", "wind_direction_sin", "wind_direction_cos",
	]
	weather_time = weather_joined[["usage_per_sqm", *weather_columns, *time_columns]]
	weather_matrix = weather_time.corr()

	# Matrix 4: primary building usage as indicator columns.
	primary_usage = pd.get_dummies(
		building_data["primaryspaceusage"], prefix="usage", dummy_na=False, dtype=float
	)
	usage_matrix = pd.concat([building_data["mean_usage_per_sqm"], primary_usage], axis=1).corr()
	primary_usage_matrix = usage_matrix.loc[primary_usage.columns, ["mean_usage_per_sqm"]].T

	# Latitude may affect energy use nonlinearly. Save its rank correlation and
	# a scatter plot with binned averages to reveal curved or otherwise nonlinear trends.
	latitude_data = building_data[["lat", "mean_usage_per_sqm"]].dropna()
	latitude_spearman = latitude_data.corr(method="spearman").iloc[0, 1]
	pd.DataFrame({
		"measure": ["pearson", "spearman"],
		"correlation_with_usage_per_sqm": [
			latitude_data.corr(method="pearson").iloc[0, 1], latitude_spearman,
		],
	}).to_csv(output_folder / "latitude_correlations.csv", index=False)

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.scatter(
		latitude_data["lat"], latitude_data["mean_usage_per_sqm"],
		alpha=0.25, s=18, color="#3572A5", label="Buildings",
	)
	latitude_data = latitude_data.copy()
	latitude_data["latitude_bin"] = pd.qcut(
		latitude_data["lat"], q=min(12, latitude_data["lat"].nunique()),
		duplicates="drop",
	)
	binned = latitude_data.groupby("latitude_bin", observed=True).agg(
		latitude=("lat", "mean"), usage=("mean_usage_per_sqm", "mean")
	)
	ax.plot(binned["latitude"], binned["usage"], color="#C44E52", marker="o", label="Binned mean")
	ax.set(
		title=f"Latitude and {meter_type} usage per m² (Spearman r={latitude_spearman:.2f})",
		xlabel="Latitude (degrees)", ylabel="Mean energy usage per m²",
	)
	ax.legend()
	ax.grid(alpha=0.25)
	fig.tight_layout()
	fig.savefig(output_folder / "latitude_usage_relationship.png", dpi=180)
	plt.close(fig)

	general_matrix.to_csv(output_folder / "matrix_1_general.csv")
	heating_usage_matrix.to_csv(output_folder / "matrix_2_heating_type.csv")
	weather_matrix.to_csv(output_folder / "matrix_3_weather_time.csv")
	primary_usage_matrix.to_csv(output_folder / "matrix_4_primary_usage.csv")

	matrices = [
		("General building features", general_matrix, "matrix_1_general.png"),
		("Weather and time", weather_matrix, "matrix_3_weather_time.png"),
	]
	for title, matrix, filename in matrices:
		save_correlation_plot(
			matrix,
			f"{meter_type.title()} usage per m² — {title}",
			output_folder / filename,
		)

	save_category_boxplot(
		building_data,
		"heatingtype",
		f"{meter_type.title()} usage per m² by heating type",
		output_folder / "matrix_2_heating_type_boxplot.png",
	)
	save_category_boxplot(
		building_data,
		"primaryspaceusage",
		f"{meter_type.title()} usage per m² by primary usage",
		output_folder / "matrix_4_primary_usage_boxplot.png",
	)

	save_correlation_plot(
		heating_usage_matrix,
		f"{meter_type.title()} usage-per-m² correlations by heating type",
		output_folder / "matrix_2_heating_type.png",
	)
	save_correlation_plot(
		primary_usage_matrix,
		f"{meter_type.title()} usage-per-m² correlations by primary usage",
		output_folder / "matrix_4_primary_usage.png",
	)

	print(f"Analyzed {data_type}/{meter_type}: {len(usage_columns)} buildings with valid floor area.")
	print(f"Weather/time matrix uses {len(weather_joined):,} site-hour observations.")
	print(f"Four correlation matrices, heatmaps, and latitude analysis saved in: {output_folder}")


if __name__ == "__main__":
	main()