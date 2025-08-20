# pylint: disable=invalid-name, line-too-long, too-many-locals, too-many-statements, redefined-outer-name
"""
This module contains tests for regression models using the CalibratedExplainer from the calibrated_explanations package.
The tests cover various scenarios including failure cases, probabilistic explanations, conditional explanations,
and explanations with difficulty estimators. The tests use pytest for testing and include fixtures for generating
regression datasets.
Functions:
    regression_dataset: Generates a regression dataset from a CSV file.
    get_regression_model: Returns a regression model based on the provided model name.
    test_failure_regression: Tests failure cases for the CalibratedExplainer.
    test_regression_ce: Tests the CalibratedExplainer for regression models.
    test_probabilistic_regression_ce: Tests probabilistic explanations for regression models.
    test_regression_conditional_ce: Tests conditional explanations for regression models.
    test_probabilistic_regression_conditional_ce: Tests probabilistic conditional explanations for regression models.
    test_knn_normalized_regression_ce: Tests KNN normalized explanations for regression models.
    test_knn_normalized_probabilistic_regression_ce: Tests KNN normalized probabilistic explanations for regression models.
    test_var_normalized_regression_ce: Tests variance normalized explanations for regression models.
    test_var_normalized_probabilistic_regression_ce: Tests variance normalized probabilistic explanations for regression models.
    test_regression_fast_ce: Tests fast explanations for regression models.
    test_probabilistic_regression_fast_ce: Tests fast probabilistic explanations for regression models.
    test_regression_conditional_fast_ce: Tests conditional perturbed explanations for regression models.
    test_probabilistic_regression_conditional_fast_ce: Tests probabilistic conditional perturbed explanations for regression models.
    test_knn_normalized_regression_fast_ce: Tests KNN normalized fast explanations for regression models.
    test_knn_normalized_probabilistic_regression_fast_ce: Tests KNN normalized fast probabilistic explanations for regression models.
    test_var_normalized_regression_fast_ce: Tests variance normalized fast explanations for regression models.
    test_var_normalized_probabilistic_regression_fast_ce: Tests variance normalized fast probabilistic explanations for regression models.
"""

import numpy as np
import pandas as pd
import pytest
from calibrated_explanations import CalibratedExplainer
from crepes.extras import DifficultyEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from tests.test_classification import initiate_explainer


@pytest.fixture
def regression_dataset():
    """
    Generates a regression dataset from a CSV file.
    The function reads a dataset from a CSV file, processes it, and splits it into training, calibration, and test sets.
    It also identifies the number of features and categorical features.
    Returns:
        tuple: A tuple containing the following elements:
            - X_prop_train (numpy.ndarray): The training features for the model.
            - y_prop_train (numpy.ndarray): The training labels for the model.
            - X_cal (numpy.ndarray): The calibration features.
            - y_cal (numpy.ndarray): The calibration labels.
            - X_test (numpy.ndarray): The test features.
            - y_test (numpy.ndarray): The test labels.
            - no_of_features (int): The number of features in the dataset.
            - categorical_features (list): A list of indices of categorical features.
            - columns (pandas.Index): The column names of the features.
    """
    num_to_test = 2
    calibration_size = 1000
    dataset = "abalone.txt"

    ds = pd.read_csv(f"data/reg/{dataset}")
    X = ds.drop("REGRESSION", axis=1).values[:2000, :]
    y = ds["REGRESSION"].values[:2000]
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    no_of_features = X.shape[1]
    categorical_features = [i for i in range(no_of_features) if len(np.unique(X[:, i])) < 10]
    columns = ds.drop("REGRESSION", axis=1).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=num_to_test, random_state=42
    )
    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(
        X_train, y_train, test_size=calibration_size, random_state=42
    )
    return (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        no_of_features,
        categorical_features,
        columns,
    )


def get_regression_model(model_name, X_prop_train, y_prop_train):
    """
    Initializes and fits a regression model.
    Args:
        model_name (str): The name of the model to initialize.
        X_prop_train (np.ndarray): Training features.
        y_prop_train (np.ndarray): Training labels.
    Returns:
        tuple: A tuple containing the fitted model and its name.
    """
    t1 = DecisionTreeRegressor()
    r1 = RandomForestRegressor(n_estimators=10)
    model_dict = {"RF": (r1, "RF"), "DT": (t1, "DT")}

    model, model_name = model_dict[model_name]
    model.fit(X_prop_train, y_prop_train)
    return model, model_name


def test_failure_regression(regression_dataset):
    """
    Tests failure cases for the CalibratedExplainer.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, _, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, X_cal, y_cal, feature_names, categorical_features, mode="regression"
    )
    with pytest.raises(RuntimeError):
        cal_exp.set_difficulty_estimator(DifficultyEstimator())
    with pytest.raises(RuntimeError):
        cal_exp.set_difficulty_estimator(DifficultyEstimator)


def test_regression_ce(regression_dataset):
    """
    Tests the CalibratedExplainer for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, X_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    factual_explanation = cal_exp.explain_factual(X_test)
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)
    factual_explanation.plot(show=False, filename="test.png")

    factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(0.1, np.inf))
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(-np.inf, 0.9))
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_alternatives(X_test)
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_alternatives(
        X_test, low_high_percentiles=(0.1, np.inf)
    )
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_alternatives(
        X_test, low_high_percentiles=(-np.inf, 0.9)
    )
    alternative_explanation.plot(show=False)
    alternative_explanation.semi_explanations()
    alternative_explanation.counter_explanations()


def test_probabilistic_regression_ce(regression_dataset):
    """
    Tests probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, X_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    cal_exp.initialize_reject_learner(threshold=0.5)
    cal_exp.predict_reject(X_test)

    factual_explanation = cal_exp.explain_factual(X_test, y_test)
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation.plot(show=False, uncertainty=True)

    factual_explanation = cal_exp.explain_factual(X_test, y_test[0])
    factual_explanation = cal_exp.explain_factual(X_test, (0.4, 0.6))

    alternative_explanation = cal_exp.explore_alternatives(X_test, y_test)
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_alternatives(X_test, y_test[0])
    alternative_explanation.plot(show=False)
    alternative_explanation.super_explanations()
    alternative_explanation.semi_explanations()
    alternative_explanation.counter_explanations()


def test_regression_as_classification_ce(regression_dataset):
    """
    Tests probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)

    def predict_function(x):
        """Convert regression predictions to binary classification."""
        return np.asarray([[float(p > 0.5), float(p <= 0.5)] for p in model.predict(x)])

    cal_exp = CalibratedExplainer(
        model,
        X_cal,
        np.asarray([1 if y <= 0.5 else 0 for y in y_cal]),
        feature_names=feature_names,
        categorical_features=categorical_features,
        mode="classification",
        predict_function=predict_function,
    )

    factual_explanation = cal_exp.explain_factual(X_test)
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation.plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_alternatives(X_test)
    alternative_explanation.plot(show=False)


def test_regression_conditional_ce(regression_dataset):
    """
    Tests conditional explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    bin_feature = 0
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        bins=X_cal[:, bin_feature],
    )

    factual_explanation = cal_exp.explain_factual(X_test, bins=X_test[:, bin_feature])
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation.plot(show=False, uncertainty=True)
    repr(factual_explanation)

    factual_explanation = cal_exp.explain_factual(
        X_test, low_high_percentiles=(0.1, np.inf), bins=X_test[:, bin_feature]
    )
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    factual_explanation = cal_exp.explain_factual(
        X_test, low_high_percentiles=(-np.inf, 0.9), bins=X_test[:, bin_feature]
    )
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_alternatives(X_test, bins=X_test[:, bin_feature])
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_alternatives(
        X_test, low_high_percentiles=(0.1, np.inf), bins=X_test[:, bin_feature]
    )
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_alternatives(
        X_test, low_high_percentiles=(-np.inf, 0.9), bins=X_test[:, bin_feature]
    )
    alternative_explanation.plot(show=False)
    repr(alternative_explanation)


def test_probabilistic_regression_conditional_ce(regression_dataset):
    """
    Tests probabilistic conditional explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        bins=X_cal[:, 0],
    )

    cal_exp.initialize_reject_learner(threshold=0.5)
    cal_exp.predict_reject(X_test, bins=X_test[:, 0])

    factual_explanation = cal_exp.explain_factual(X_test, y_test, bins=X_test[:, 0])
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)

    factual_explanation = cal_exp.explain_factual(X_test, y_test[0], bins=X_test[:, 0])

    alternative_explanation = cal_exp.explore_alternatives(X_test, y_test, bins=X_test[:, 0])
    alternative_explanation.plot(show=False)

    cal_exp.explore_alternatives(X_test, y_test[0], bins=X_test[:, 0])


def test_knn_normalized_regression_ce(regression_dataset):
    """
    Tests KNN normalized explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, y=y_prop_train, scaler=True),
    )

    factual_explanation = cal_exp.explain_factual(X_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(0.1, np.inf))

    factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(-np.inf, 0.9))

    cal_exp.explore_alternatives(X_test)

    cal_exp.explore_alternatives(X_test, low_high_percentiles=(0.1, np.inf))

    cal_exp.explore_alternatives(X_test, low_high_percentiles=(-np.inf, 0.9))


def test_knn_normalized_probabilistic_regression_ce(regression_dataset):
    """
    Tests KNN normalized probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, y=y_prop_train, scaler=True),
    )

    factual_explanation = cal_exp.explain_factual(X_test, y_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_factual(X_test, y_test[0])

    cal_exp.explore_alternatives(X_test, y_test)

    cal_exp.explore_alternatives(X_test, y_test[0])


def test_var_normalized_regression_ce(regression_dataset):
    """
    Tests variance normalized explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, learner=model, scaler=True),
    )

    factual_explanation = cal_exp.explain_factual(X_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(0.1, np.inf))

    factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(-np.inf, 0.9))

    cal_exp.explore_alternatives(X_test)

    cal_exp.explore_alternatives(X_test, low_high_percentiles=(0.1, np.inf))

    cal_exp.explore_alternatives(X_test, low_high_percentiles=(-np.inf, 0.9))


def test_var_normalized_probabilistic_regression_ce(regression_dataset):
    """
    Tests variance normalized probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, learner=model, scaler=True),
    )

    factual_explanation = cal_exp.explain_factual(X_test, y_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_factual(X_test, y_test[0])

    cal_exp.explore_alternatives(X_test, y_test)

    cal_exp.explore_alternatives(X_test, y_test[0])


def test_regression_fast_ce(regression_dataset):
    """
    Tests fast explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, X_cal, y_cal, feature_names, categorical_features, mode="regression", fast=True
    )

    fast_explanation = cal_exp.explain_fast(X_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()
    fast_explanation.plot(show=False)
    fast_explanation.plot(show=False, uncertainty=True)

    fast_explanation = cal_exp.explain_fast(X_test, low_high_percentiles=(0.1, np.inf))
    fast_explanation.plot(show=False)
    with pytest.raises(Warning):
        fast_explanation.plot(show=False, uncertainty=True)

    fast_explanation = cal_exp.explain_fast(X_test, low_high_percentiles=(-np.inf, 0.9))
    fast_explanation.plot(show=False)
    with pytest.raises(Warning):
        fast_explanation.plot(show=False, uncertainty=True)


def test_probabilistic_regression_fast_ce(regression_dataset):
    """
    Tests fast probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, X_cal, y_cal, feature_names, categorical_features, mode="regression", fast=True
    )

    fast_explanation = cal_exp.explain_fast(X_test, y_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()
    fast_explanation.plot(show=False)
    fast_explanation.plot(show=False, uncertainty=True)

    fast_explanation = cal_exp.explain_fast(X_test, y_test[0])
    fast_explanation.plot(show=False)
    fast_explanation.plot(show=False, uncertainty=True)


def test_regression_conditional_fast_ce(regression_dataset):
    """
    Tests conditional perturbed explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        bins=X_cal[:, 0],
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(X_test, bins=X_test[:, 0])
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(
        X_test, low_high_percentiles=(0.1, np.inf), bins=X_test[:, 0]
    )

    fast_explanation = cal_exp.explain_fast(
        X_test, low_high_percentiles=(-np.inf, 0.9), bins=X_test[:, 0]
    )


def test_probabilistic_regression_conditional_fast_ce(regression_dataset):
    """
    Tests probabilistic conditional perturbed explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        bins=y_cal > y_test[0],
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(X_test, y_test, bins=y_test > y_test[0])
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(X_test, y_test[0], bins=y_test > y_test[0])


def test_knn_normalized_regression_fast_ce(regression_dataset):
    """
    Tests KNN normalized fast explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, y=y_prop_train, scaler=True),
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(X_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(X_test, low_high_percentiles=(0.1, np.inf))

    fast_explanation = cal_exp.explain_fast(X_test, low_high_percentiles=(-np.inf, 0.9))


def test_knn_normalized_probabilistic_regression_fast_ce(regression_dataset):
    """
    Tests KNN normalized fast probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, y=y_prop_train, scaler=True),
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(X_test, y_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(X_test, y_test[0])


def test_var_normalized_regression_fast_ce(regression_dataset):
    """
    Tests variance normalized fast explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, learner=model, scaler=True),
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(X_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(X_test, low_high_percentiles=(0.1, np.inf))

    fast_explanation = cal_exp.explain_fast(X_test, low_high_percentiles=(-np.inf, 0.9))


def test_var_normalized_probabilistic_regression_fast_ce(regression_dataset):
    """
    Tests variance normalized fast probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, learner=model, scaler=True),
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(X_test, y_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(X_test, y_test[0])
