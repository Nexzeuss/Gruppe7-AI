"""Feature subset C (minimal: floor area, use type and climate): run all three models with five-fold CV."""

from subset_utils import run_subset


if __name__ == "__main__":
    run_subset("C")
