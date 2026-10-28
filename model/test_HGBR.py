from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data_folder = Path(__file__).parent.parent / "building-data-genome-project-2" / "data"
meter_file = data_folder / "meters" / "my_cleaned" / "electricity_cleaned.csv"
weather_file = data_folder / "weather" / "weather.csv"
metadata_file = data_folder / "metadata" / "metadata.csv"

N_BUILDINGS = 100        # all ~1500 buildings x 17544 hours is ~26M rows, so test on a random sample
RANDOM_STATE = 42
SPLIT_DATE = "2017-01-01"  # train on 2016, test on 2017

WEATHER_FEATURES = ["airTemperature", "dewTemperature", "windSpeed", "cloudCoverage", "seaLvlPressure", "precipDepth1HR"]
METADATA_FEATURES = ["sqm", "yearbuilt", "numberoffloors", "lat"]
CATEGORICAL_FEATURES = ["site_id", "primaryspaceusage"]
TIME_FEATURES = ["hour", "weekday", "month", "dayofyear"]
FEATURES = TIME_FEATURES + WEATHER_FEATURES + METADATA_FEATURES + CATEGORICAL_FEATURES


def load_data():
	"""Meter data in long format (one row per building and hour) joined with metadata and weather."""
	meters = pd.read_csv(meter_file, parse_dates=["timestamp"], index_col="timestamp")
	buildings = meters.columns.to_series().sample(min(N_BUILDINGS, meters.shape[1]), random_state=RANDOM_STATE)
	meters = meters[buildings].copy()

	data = meters.reset_index().melt(id_vars="timestamp", var_name="building_id", value_name="meter_reading")
	data = data.dropna(subset=["meter_reading"])  # long gaps were left as NaN by the cleaning, nothing to learn from

	metadata = pd.read_csv(metadata_file, usecols=["building_id", "site_id", "primaryspaceusage"] + METADATA_FEATURES)
	data = data.merge(metadata, on="building_id", how="left")

	weather = pd.read_csv(weather_file, parse_dates=["timestamp"], usecols=["timestamp", "site_id"] + WEATHER_FEATURES)
	data = data.merge(weather, on=["timestamp", "site_id"], how="left")

	data["hour"] = data["timestamp"].dt.hour
	data["weekday"] = data["timestamp"].dt.weekday
	data["month"] = data["timestamp"].dt.month
	data["dayofyear"] = data["timestamp"].dt.dayofyear

	for column in CATEGORICAL_FEATURES:
		data[column] = data[column].astype("category")
	return data


def evaluate(name, y_true, y_pred):
	rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	print(
		f"{name:6s} RMSE {rmse:9.2f}  MAE {mean_absolute_error(y_true, y_pred):9.2f}  "
		f"R2 {r2_score(y_true, y_pred):6.3f}  CV(RMSE) {rmse / y_true.mean():6.1%}"
	)


print(f"Loading data for {N_BUILDINGS} buildings ...")
data = load_data()
train = data[data["timestamp"] < SPLIT_DATE]
test = data[data["timestamp"] >= SPLIT_DATE]
print(f"  train rows {len(train):,}, test rows {len(test):,}")

# Meter readings are very skewed (small offices vs. big campuses), so the model learns log1p(reading)
model = HistGradientBoostingRegressor(
	max_iter=500,
	learning_rate=0.1,
	max_leaf_nodes=63,
	categorical_features="from_dtype",
	early_stopping=True,
	validation_fraction=0.1,
	random_state=RANDOM_STATE,
)
print("Training HistGradientBoostingRegressor ...")
model.fit(train[FEATURES], np.log1p(train["meter_reading"]))
print(f"  stopped after {model.n_iter_} iterations")

train_pred = np.expm1(model.predict(train[FEATURES]))
test_pred = np.expm1(model.predict(test[FEATURES]))
evaluate("train", train["meter_reading"], train_pred)
evaluate("test", test["meter_reading"], test_pred)

# Per building error on the test year, to see if a few buildings dominate the total error
test = test.assign(prediction=test_pred)
per_building = test.groupby("building_id", observed=True).apply(
	lambda g: np.sqrt(mean_squared_error(g["meter_reading"], g["prediction"])) / g["meter_reading"].mean()
)
print("\nCV(RMSE) per building on the test year:")
print(per_building.describe().to_string(float_format=lambda x: f"{x:.3f}"))

# Plot two weeks of actual vs. predicted for the building closest to the median error
example = (per_building - per_building.median()).abs().idxmin()
example_data = test[test["building_id"] == example].set_index("timestamp").sort_index()
example_data = example_data.loc["2017-07-01":"2017-07-14"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(example_data.index, example_data["meter_reading"], label="actual")
axes[0].plot(example_data.index, example_data["prediction"], label="predicted")
axes[0].set_title(f"{example} (median building), July 2017")
axes[0].set_ylabel("electricity [kWh]")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=45)

axes[1].scatter(test["meter_reading"], test["prediction"], s=1, alpha=0.1)
limit = max(test["meter_reading"].max(), test["prediction"].max())
axes[1].plot([0, limit], [0, limit], color="black", linewidth=1)
axes[1].set_xscale("symlog")
axes[1].set_yscale("symlog")
axes[1].set_xlabel("actual [kWh]")
axes[1].set_ylabel("predicted [kWh]")
axes[1].set_title("Test year 2017, all buildings")

fig.tight_layout()
plt.show()
