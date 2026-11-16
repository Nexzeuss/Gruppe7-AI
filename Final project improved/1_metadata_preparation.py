"""Metadata preparation: the assignment's steps 1-4, plus heating type and meter indicators."""

import pandas as pd

from settings import (
    ESSENTIAL_COLUMNS, METADATA_COLUMNS, METADATA_FILE, METER_COLUMNS, OUTPUT_FOLDER, log_step, save_log,
)


SCRIPT = "1_metadata_preparation"


def step_1_load_metadata():
    """Load metadata.csv into a DataFrame."""
    return pd.read_csv(METADATA_FILE)


def step_2_keep_columns(metadata):
    """Keep the assignment's columns, heating type, and the meter columns."""
    return metadata[METADATA_COLUMNS + METER_COLUMNS]


def step_3_set_index(metadata):
    """Set the index to building_id to align with the time series."""
    return metadata.set_index("building_id")


def step_4_drop_missing_essentials(metadata):
    """Drop rows with missing building_id or essential fields like sqm."""
    metadata = metadata[metadata.index.notna()]
    return metadata.dropna(subset=ESSENTIAL_COLUMNS)


def step_5_meter_indicators(metadata):
    """Turn the "Yes"/empty meter columns into 0/1 indicators, e.g. has_chilledwater.

    A chilled water, steam or hot water meter suggests district supply, which moves heating or cooling
    off the electricity meter. The indicators describe equipment, not consumption, so they do not leak the target.
    """
    metadata = metadata.copy()
    for meter in METER_COLUMNS:
        metadata[f"has_{meter}"] = metadata.pop(meter).notna().astype(int)
    return metadata


def main():
    log = []
    print("Metadata preparation")
    metadata = step_1_load_metadata()
    log_step(log, SCRIPT, 1, "Load metadata.csv", len(metadata), len(metadata))

    metadata = step_2_keep_columns(metadata)
    log_step(log, SCRIPT, 2, "Keep the assignment's columns, heatingtype and meter columns", len(metadata), len(metadata))

    metadata = step_3_set_index(metadata)
    log_step(log, SCRIPT, 3, "Set index to building_id", len(metadata), len(metadata))

    before = len(metadata)
    metadata = step_4_drop_missing_essentials(metadata)
    log_step(log, SCRIPT, 4, f"Drop rows missing building_id or {', '.join(ESSENTIAL_COLUMNS)}", before, len(metadata))

    metadata = step_5_meter_indicators(metadata)
    log_step(log, SCRIPT, 5, "Add meter indicators has_<meter>", len(metadata), len(metadata))

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(OUTPUT_FOLDER / "metadata_prepared.csv")
    save_log(log, SCRIPT)
    print(f"Saved {OUTPUT_FOLDER / 'metadata_prepared.csv'}")


if __name__ == "__main__":
    main()
