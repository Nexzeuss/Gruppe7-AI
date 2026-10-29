"""Five-fold, building-grouped validation for HistGradientBoostingRegressor."""

from sklearn.ensemble import HistGradientBoostingRegressor

from model_utils import RANDOM_STATE, evaluate_cv


def main():
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.1,
        max_leaf_nodes=31,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    evaluate_cv("HistGradientBoosting", model)


if __name__ == "__main__":
    main()