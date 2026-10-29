"""Five-fold, building-grouped validation for a decision tree regressor."""

from sklearn.tree import DecisionTreeRegressor

from model_utils import RANDOM_STATE, evaluate_cv


def main():
    model = DecisionTreeRegressor(
        max_depth=20,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )
    evaluate_cv("Decision Tree", model)


if __name__ == "__main__":
    main()