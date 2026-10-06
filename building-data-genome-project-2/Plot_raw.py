from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd


data_folder = Path(__file__).parent / "data" / "meters" / "raw"
dataset_name = "raw"

LINE_COLOR = "#2a78d6"
BAND_COLOR = "#86b6ef"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e6e5e0"


def daily_per_meter(file_path):
	"""Sum each meter's hourly readings into one value per day."""
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


files = sorted(data_folder.glob("*.csv"))
columns = 2
rows = math.ceil(len(files) / columns)

figure, axes = plt.subplots(rows, columns, figsize=(16, 3 * rows), sharex=True)
axes = axes.flatten()

for ax, file_path in zip(axes, files):
	name = file_path.stem.replace("_cleaned", "")
	print(f"Loading {file_path.name} ...")
	plot_meter_type(ax, name, daily_per_meter(file_path))

# Hide empty panels if the number of files is odd
for ax in axes[len(files):]:
	ax.set_visible(False)

figure.suptitle(f"Daily meter readings per meter type - {dataset_name} data", fontsize=14, x=0.01, ha="left")
figure.text(
	0.01, 0.955,
	"Line: median meter's daily total.  Band: middle 50% of meters (25th-75th percentile).  "
	"Each panel has its own y-scale.",
	fontsize=9, color=TEXT_SECONDARY, ha="left",
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()
