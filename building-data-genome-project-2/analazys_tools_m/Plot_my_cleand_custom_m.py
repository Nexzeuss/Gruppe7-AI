from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd


project_folder = Path(__file__).resolve().parent.parent
dataset_name = "my_cleand_custom"
data_folder = project_folder / "data" / "meters" / dataset_name
weather_file = data_folder / "weather_cleaned.csv"
output_folder = project_folder / "figures" / "my_cleand_custom" / f"sequential_analysis_{dataset_name}"

LINE_COLOR = "#2a78d6"
BAND_COLOR = "#86b6ef"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e6e5e0"


def daily_per_meter(file_path):
	"""Sum each meter's hourly readings into one value per day."""
	with file_path.open(encoding="utf-8") as csv_file:
		first_line = csv_file.readline().strip()
	if first_line == "version https://git-lfs.github.com/spec/v1":
		raise RuntimeError(
			f"{file_path.name} is a Git LFS pointer, not the actual CSV data. "
			"Install Git LFS if needed, then run `git lfs pull` from the repository root."
		)

	data = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
	# min_count=1 keeps days without any readings as NaN instead of 0
	return data.resample("D").sum(min_count=1)


def plot_meter_type(ax, name, daily):
	meters_with_data = daily.notna().any().sum()
	# Median across meters is robust to a few huge buildings and to meters that drop out
	median = daily.median(axis=1)
	lower = daily.quantile(0.25, axis=1)
	upper = daily.quantile(0.75, axis=1)

	ax.fill_between(daily.index, lower, upper, color=BAND_COLOR, alpha=0.35, linewidth=0)
	ax.plot(daily.index, median, color=LINE_COLOR, linewidth=1.5)

	ax.set_title(f"{name}  ({meters_with_data} meters)", loc="left", fontsize=11)
	ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
	ax.set_axisbelow(True)
	ax.set_ylim(bottom=0)
	ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(GRID_COLOR)
	ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def meter_files():
	"""Return CSVs for the selected input layout, with a clear missing-data error."""
	if dataset_name not in {"raw", "cleaned", "my_cleaned", "my_cleand_custom"}:
		raise ValueError("Unsupported dataset_name.")
	if not data_folder.is_dir():
		raise FileNotFoundError(f"Meter data folder not found: {data_folder}")
	files = sorted(data_folder.glob("*.csv"))
	if dataset_name in {"my_cleaned", "my_cleand_custom"}:
		files = [path for path in files if path.name.endswith("_cleaned.csv")]
	files = [path for path in files if path.name != weather_file.name]
	if not files:
		raise FileNotFoundError(f"No meter CSV files found in {data_folder}.")
	return files


def main():
	files = meter_files()
	if not weather_file.is_file():
		raise FileNotFoundError(f"Weather data file not found: {weather_file}")
	with weather_file.open(encoding="utf-8") as csv_file:
		if csv_file.readline().strip() == "version https://git-lfs.github.com/spec/v1":
			raise RuntimeError(f"{weather_file.name} is a Git LFS pointer, not CSV data.")
	output_folder.mkdir(parents=True, exist_ok=True)
	columns = 2
	rows = math.ceil(len(files) / columns)
	figure, axes = plt.subplots(rows, columns, figsize=(16, 3.5 * rows), sharex=True, squeeze=False)
	axes = axes.flatten()

	for ax, file_path in zip(axes, files):
		name = file_path.stem.removesuffix("_cleaned")
		print(f"Loading {file_path.name} ...")
		daily = daily_per_meter(file_path)
		plot_meter_type(ax, name, daily)

		# Also save each meter type as a standalone sequential-data plot.
		individual_figure, individual_ax = plt.subplots(figsize=(12, 5))
		plot_meter_type(individual_ax, name, daily)
		individual_ax.set_xlabel("Date")
		individual_figure.suptitle(f"Daily meter readings — {name} ({dataset_name} data)")
		individual_figure.tight_layout()
		individual_figure.savefig(output_folder / f"daily_{name}.png", dpi=180, bbox_inches="tight")
		plt.close(individual_figure)

	for ax in axes[len(files):]:
		ax.set_visible(False)

	figure.suptitle(f"Daily meter readings per meter type — {dataset_name} data", fontsize=14, x=0.01, ha="left")
	figure.text(
		0.01, 0.955,
		"Line: median meter's daily total. Band: middle 50% of meters (25th–75th percentile). "
		"Each panel has its own y-scale.",
		fontsize=9, color=TEXT_SECONDARY, ha="left",
	)
	figure.tight_layout(rect=(0, 0, 1, 0.95))
	figure.savefig(output_folder / f"daily_meter_readings_{dataset_name}.png", dpi=180, bbox_inches="tight")
	plt.close(figure)
	print(f"Saved sequential plots for {len(files)} meter types in: {output_folder}")


if __name__ == "__main__":
	main()
