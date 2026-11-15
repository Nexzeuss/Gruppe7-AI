"""Feature subset A (all metadata features including generated ones): run all three models with five-fold CV."""

from subset_utils import run_subset


if __name__ == "__main__":
    run_subset("A")
