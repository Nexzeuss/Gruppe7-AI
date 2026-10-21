from pathlib import Path

import numpy as np
import pandas as pd


raw_folder = Path(__file__).parent / "data" / "meters" / "raw"
output_folder = Path(__file__).parent / "data" / "meters" / "my_cleaned"

# A building always uses some electricity, so a zero reading means the meter dropped out.
# For the other meter types zero is a real value (no heating in summer, no irrigation in winter, ...)
ZERO_MEANS_MISSING = {"electricity"}

FLAT_LINE_HOURS = 168       # same non-zero value for a week = stuck meter. Shorter is not safe:
                            # some sites (e.g. Gator) store one value per day, repeated for 24 hours
OUTLIER_WINDOW_HOURS = 720  # 30-day moving window, so seasons do not count as outliers
OUTLIER_IQR_FACTOR = 3      # 3 x IQR is the usual fence for "extreme" outliers
MAX_GAP_TO_FILL = 6         # only interpolate gaps of up to 6 hours
MIN_DATA_SHARE = 0.5        # drop meters that end up with less than half of the hours


def run_lengths(values):
	"""For every position, the length of the run of equal values it belongs to (NaN counts as equal to NaN)."""
	same_as_previous = np.r_[False, (values[1:] == values[:-1]) | (np.isnan(values[1:]) & np.isnan(values[:-1]))]
	run_id = np.cumsum(~same_as_previous)
	return np.bincount(run_id)[run_id]


def remove_flat_lines(data):
	"""Set long runs of the same non-zero value to NaN."""
	flat = pd.DataFrame(False, index=data.index, columns=data.columns)
	for meter in data.columns:
		values = data[meter].to_numpy()
		flat[meter] = (run_lengths(values) >= FLAT_LINE_HOURS) & (values != 0) & ~np.isnan(values)
	return data.mask(flat), flat


def remove_outliers(data):
	"""IQR filter on a moving window: values far outside the local Q1-Q3 range become NaN."""
	# Quartiles are taken from hours where the meter is in use (> 0). Otherwise meters that are
	# zero half the time (solar at night, gas in summer) get a fence so tight that normal peaks are cut
	in_use = data.where(data > 0)
	window = in_use.rolling(OUTLIER_WINDOW_HOURS, center=True, min_periods=OUTLIER_WINDOW_HOURS // 4)
	q1 = window.quantile(0.25)
	q3 = window.quantile(0.75)
	iqr = q3 - q1
	# Zeros are only tested against the in-use range (a shutdown is not a spike), so compare in_use, not data.
	# When IQR is 0 or unknown (meter almost never in use) the fence is useless, so skip those hours
	outliers = ((in_use > q3 + OUTLIER_IQR_FACTOR * iqr) | (in_use < q1 - OUTLIER_IQR_FACTOR * iqr)) & (iqr > 0)
	return data.mask(outliers), outliers


def fill_short_gaps(data):
	"""Linear interpolation, but only for gaps of at most MAX_GAP_TO_FILL hours. Longer gaps stay NaN."""
	interpolated = data.interpolate(method="time", limit_area="inside")
	short_gap = pd.DataFrame(False, index=data.index, columns=data.columns)
	for meter in data.columns:
		values = data[meter].to_numpy()
		short_gap[meter] = np.isnan(values) & (run_lengths(values) <= MAX_GAP_TO_FILL)
	filled = short_gap & interpolated.notna()
	return data.where(~filled, interpolated), filled


def clean_meter_type(file_path):
	name = file_path.stem
	data = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")

	# 1. Structure: sorted, no duplicate hours, and every hour in the period present
	data = data.sort_index()
	data = data[~data.index.duplicated(keep="first")]
	full_range = pd.date_range(data.index.min(), data.index.max(), freq="h", name="timestamp")
	data = data.reindex(full_range)
	missing_in_raw = data.isna()

	# 2. Impossible values
	invalid = data < 0
	if name in ZERO_MEANS_MISSING:
		invalid |= data == 0
	data = data.mask(invalid)

	# 3. Stuck meters and 4. spikes
	data, flat = remove_flat_lines(data)
	data, outliers = remove_outliers(data)

	# 5. Fill short holes, leave long ones as NaN
	data, filled = fill_short_gaps(data)

	# 6. Drop meters that have too little data left to be useful
	data_share = data.notna().mean()
	kept = data_share >= MIN_DATA_SHARE

	report = pd.DataFrame({
		"meter_type": name,
		"hours": len(data),
		"missing_in_raw": missing_in_raw.sum(),
		"invalid_removed": invalid.sum(),
		"flat_line_removed": flat.sum(),
		"outliers_removed": outliers.sum(),
		"gaps_interpolated": filled.sum(),
		"missing_after": data.isna().sum(),
		"data_share_after": data_share.round(3),
		"kept": kept,
	})
	report.index.name = "meter"
	return data.loc[:, kept], report


output_folder.mkdir(exist_ok=True)
reports = []

for file_path in sorted(raw_folder.glob("*.csv")):
	print(f"Cleaning {file_path.name} ...")
	cleaned, report = clean_meter_type(file_path)
	cleaned.round(4).to_csv(output_folder / f"{file_path.stem}_cleaned.csv")
	reports.append(report)

	total = report["hours"].sum()
	print(
		f"  raw missing {report['missing_in_raw'].sum() / total:.1%}, "
		f"invalid {report['invalid_removed'].sum() / total:.1%}, "
		f"flat lines {report['flat_line_removed'].sum() / total:.1%}, "
		f"outliers {report['outliers_removed'].sum() / total:.1%}, "
		f"interpolated {report['gaps_interpolated'].sum() / total:.1%}, "
		f"missing after {report['missing_after'].sum() / total:.1%}, "
		f"meters kept {report['kept'].sum()}/{len(report)}"
	)

pd.concat(reports).to_csv(output_folder / "cleaning_report.csv")
print(f"Done. Files and cleaning_report.csv written to {output_folder}")
