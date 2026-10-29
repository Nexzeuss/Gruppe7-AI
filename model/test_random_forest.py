"""Five-fold, building-grouped validation for a random forest regressor."""

from sklearn.ensemble import RandomForestRegressor

from model_utils import RANDOM_STATE, evaluate_cv


def main():
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=24,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    evaluate_cv("Random Forest", model)


if __name__ == "__main__":
    main()