# pylint: disable=invalid-name, line-too-long, too-many-locals, too-many-statements, redefined-outer-name
"""
Module for testing the WrapCalibratedExplainer class for regression tasks.
This module contains test functions that verify the functionality of the WrapCalibratedExplainer class
using a RandomForestRegressor. The tests cover various aspects including fitting, calibration, prediction,
and explanation capabilities of the explainer.
Functions:
    regression_dataset: Generates a regression dataset from a CSV file.
    test_wrap_regression_ce: Tests the WrapCalibratedExplainer class for regression.
    test_wrap_conditional_regression_ce: Tests the WrapCalibratedExplainer class for conditional regression.
    test_wrap_regression_fast_ce: Tests the WrapCalibratedExplainer class for fast regression.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from crepes.extras import MondrianCategorizer
from calibrated_explanations import WrapCalibratedExplainer

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
    num_to_test = 1
    calibration_size = 200
    dataset = 'abalone.txt'

    ds = pd.read_csv(f'data/reg/{dataset}')
    X = ds.drop('REGRESSION', axis=1).values[:2000, :]
    y = ds['REGRESSION'].values[:2000]
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    no_of_features = X.shape[1]
    categorical_features = [i for i in range(no_of_features) if len(np.unique(X[:, i])) < 10]
    columns = ds.drop('REGRESSION', axis=1).columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_to_test, random_state=42)
    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train, test_size=calibration_size, random_state=42)
    return X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, no_of_features, categorical_features, columns

def test_wrap_regression_ce(regression_dataset):
    """
    Test the WrapCalibratedExplainer class for regression.
    This test function performs the following steps:
    1. Initializes the WrapCalibratedExplainer with a RandomForestRegressor.
    2. Checks that the explainer is neither fitted nor calibrated initially.
    3. Ensures that explain methods raise RuntimeError before fitting.
    4. Fits the explainer and verifies it is fitted but not calibrated.
    5. Tests various prediction methods (with and without calibration) and ensures consistency in the predictions.
    6. Tests the predict_proba method (with and without calibration) and ensures consistency in the probability predictions.
    7. Calibrates the explainer and verifies it is both fitted and calibrated.
    8. Re-tests the prediction methods to ensure consistency post-calibration.
    9. Re-fits the explainer and verifies it remains calibrated.
    10. Tests the ability to create new instances of WrapCalibratedExplainer with the same learner and explainer, ensuring they inherit the correct fitted and calibrated states.
    11. Plots the results to visually inspect the predictions.
    Args:
        regression_dataset (tuple): A tuple containing the training, calibration, and test datasets along with additional metadata such as categorical features and feature names.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, feature_names = regression_dataset
    cal_exp = WrapCalibratedExplainer(RandomForestRegressor())
    assert not cal_exp.fitted
    assert not cal_exp.calibrated

    with pytest.raises(RuntimeError):
        cal_exp.explain_factual(X_test)
    with pytest.raises(RuntimeError):
        cal_exp.explore_alternatives(X_test)

    cal_exp.fit(X_prop_train, y_prop_train)
    assert cal_exp.fitted
    assert not cal_exp.calibrated

    y_test_hat1 = cal_exp.predict(X_test)
    y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True)
    y_test_hat3 = cal_exp.predict(X_test, calibrated=False)
    y_test_hat4, (low4, high4) = cal_exp.predict(X_test, uq_interval=True, calibrated=False)

    for i, y_hat in enumerate(y_test_hat1):
        assert y_test_hat2[i] == y_hat
        assert y_test_hat3[i] == y_hat
        assert y_test_hat4[i] == y_hat
        assert low[i] == y_hat
        assert high[i] == y_hat
        assert low4[i] == y_hat
        assert high4[i] == y_hat

    with pytest.raises(ValueError):
        cal_exp.predict(X_test, threshold=y_test)
    with pytest.raises(ValueError):
        cal_exp.predict(X_test, uq_interval=True, threshold=y_test)
    with pytest.raises(ValueError):
        cal_exp.predict_proba(X_test)
    with pytest.raises(ValueError):
        cal_exp.predict_proba(X_test, uq_interval=True)
    with pytest.raises(RuntimeError):
        cal_exp.predict_proba(X_test, threshold=y_test)
    with pytest.raises(RuntimeError):
        cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test)
    with pytest.raises(RuntimeError):
        cal_exp.explain_factual(X_test)
    with pytest.raises(RuntimeError):
        cal_exp.explore_alternatives(X_test)
    with pytest.raises(RuntimeError):
        cal_exp.explain_factual(X_test, threshold=y_test)
    with pytest.raises(RuntimeError):
        cal_exp.explore_alternatives(X_test, threshold=y_test)

    cal_exp.calibrate(X_cal, y_cal, feature_names=feature_names)
    assert cal_exp.fitted
    assert cal_exp.calibrated

    y_test_hat3 = cal_exp.predict(X_test, calibrated=False)
    y_test_hat4, (low4, high4) = cal_exp.predict(X_test, uq_interval=True, calibrated=False)

    for i, y_hat in enumerate(y_test_hat1):
        assert y_test_hat3[i] == y_hat
        assert y_test_hat4[i] == y_hat
        assert low4[i] == y_hat
        assert high4[i] == y_hat

    y_test_hat1 = cal_exp.predict(X_test)
    y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True)

    for i, y_hat in enumerate(y_test_hat2):
        assert y_test_hat1[i] == y_hat
        assert low[i] <= y_hat <= high[i]

    y_test_hat1 = cal_exp.predict(X_test, threshold=y_test)
    y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True, threshold=y_test)

    cal_exp.explain_factual(X_test)
    cal_exp.explore_alternatives(X_test)
    cal_exp.explain_factual(X_test, threshold=y_test)
    cal_exp.explore_alternatives(X_test, threshold=y_test)

    with pytest.raises(ValueError):
        cal_exp.predict_proba(X_test)
    with pytest.raises(ValueError):
        cal_exp.predict_proba(X_test, uq_interval=True)
    y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test[0])
    y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test[0])

    for i, y_hat in enumerate(y_test_hat2):
        assert low[i] <= y_test_hat2[i, 1] <= high[i]

    y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test)
    y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test)

    for i, y_hat in enumerate(y_test_hat2):
        assert low[i] <= y_test_hat2[i, 1] <= high[i]

    cal_exp.fit(X_prop_train, y_prop_train)
    assert cal_exp.fitted
    assert cal_exp.calibrated

    learner = cal_exp.learner
    explainer = cal_exp.explainer

    new_exp = WrapCalibratedExplainer(learner)
    assert new_exp.fitted
    assert not new_exp.calibrated
    assert new_exp.learner == learner

    new_exp = WrapCalibratedExplainer(explainer)
    assert new_exp.fitted
    assert new_exp.calibrated
    assert new_exp.explainer == explainer
    assert new_exp.learner == learner

    cal_exp.plot(X_test)
    cal_exp.plot(X_test, y_test)
    cal_exp.plot(X_test, threshold=y_test[0])
    cal_exp.plot(X_test, y_test, threshold=y_test[0])

def test_wrap_conditional_regression_ce(regression_dataset):
    """
    Test the WrapCalibratedExplainer class for conditional regression.
    This test function performs the following steps:
    1. Initializes the WrapCalibratedExplainer with a RandomForestRegressor.
    2. Fits the explainer and verifies it is fitted but not calibrated.
    3. Calibrates the explainer using MondrianCategorizer and verifies it is both fitted and calibrated.
    4. Tests various prediction methods (with and without calibration) and ensures consistency in the predictions.
    5. Tests the predict_proba method (with and without calibration) and ensures consistency in the probability predictions.
    6. Re-fits the explainer and verifies it remains calibrated.
    7. Tests the ability to create new instances of WrapCalibratedExplainer with the same learner and explainer, ensuring they inherit the correct fitted and calibrated states.
    Args:
        regression_dataset (tuple): A tuple containing the training, calibration, and test datasets along with additional metadata such as categorical features and feature names.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, feature_names = regression_dataset
    cal_exp = WrapCalibratedExplainer(RandomForestRegressor())
    cal_exp.fit(X_prop_train, y_prop_train)

    mc = MondrianCategorizer()
    mc.fit(X_cal, f=cal_exp.learner.predict, no_bins=5)

    cal_exp.calibrate(X_cal, y_cal, mc=mc, feature_names=feature_names)
    assert cal_exp.fitted
    assert cal_exp.calibrated

    y_test_hat1 = cal_exp.predict(X_test)
    y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True)

    for i, y_hat in enumerate(y_test_hat2):
        assert y_test_hat1[i] == y_hat
        assert low[i] <= y_hat <= high[i]

    y_test_hat1 = cal_exp.predict(X_test, threshold=y_test)
    y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True, threshold=y_test)

    cal_exp.explain_factual(X_test)
    cal_exp.explore_alternatives(X_test)
    cal_exp.explain_factual(X_test, threshold=y_test)
    cal_exp.explore_alternatives(X_test, threshold=y_test)

    with pytest.raises(ValueError):
        cal_exp.predict_proba(X_test)
    with pytest.raises(ValueError):
        cal_exp.predict_proba(X_test, uq_interval=True)
    y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test[0])
    y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test[0])

    for i, y_hat in enumerate(y_test_hat2):
        assert low[i] <= y_test_hat2[i, 1] <= high[i]

    y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test)
    y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test)

    for i, y_hat in enumerate(y_test_hat2):
        assert low[i] <= y_test_hat2[i, 1] <= high[i]

    cal_exp.fit(X_prop_train, y_prop_train)
    assert cal_exp.fitted
    assert cal_exp.calibrated

    learner = cal_exp.learner
    explainer = cal_exp.explainer

    new_exp = WrapCalibratedExplainer(learner)
    assert new_exp.fitted
    assert not new_exp.calibrated
    assert new_exp.learner == learner

    new_exp = WrapCalibratedExplainer(explainer)
    assert new_exp.fitted
    assert new_exp.calibrated
    assert new_exp.explainer == explainer
    assert new_exp.learner == learner

def test_wrap_regression_fast_ce(regression_dataset):
    """
    Test the WrapCalibratedExplainer class for fast regression.
    This test function performs the following steps:
    1. Initializes the WrapCalibratedExplainer with a RandomForestRegressor.
    2. Fits the explainer and verifies it is fitted but not calibrated.
    3. Calibrates the explainer with perturbation and verifies it is both fitted and calibrated.
    4. Tests various prediction methods (with and without calibration) and ensures consistency in the predictions.
    5. Tests the predict_proba method (with and without calibration) and ensures consistency in the probability predictions.
    6. Re-fits the explainer and verifies it remains calibrated.
    7. Tests the ability to create new instances of WrapCalibratedExplainer with the same learner and explainer, ensuring they inherit the correct fitted and calibrated states.
    Args:
        regression_dataset (tuple): A tuple containing the training, calibration, and test datasets along with additional metadata such as categorical features and feature names.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, feature_names = regression_dataset
    cal_exp = WrapCalibratedExplainer(RandomForestRegressor())
    cal_exp.fit(X_prop_train, y_prop_train)
    cal_exp.calibrate(X_cal, y_cal, feature_names=feature_names, perturb=True)
    assert cal_exp.fitted
    assert cal_exp.calibrated

    y_test_hat1 = cal_exp.predict(X_test)
    y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True)

    for i, y_hat in enumerate(y_test_hat2):
        assert y_test_hat1[i] == y_hat
        assert low[i] <= y_hat <= high[i]

    y_test_hat1 = cal_exp.predict(X_test, threshold=y_test)
    y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True, threshold=y_test)

    cal_exp.explain_factual(X_test)
    cal_exp.explore_alternatives(X_test)
    cal_exp.explain_factual(X_test, threshold=y_test)
    cal_exp.explore_alternatives(X_test, threshold=y_test)
    cal_exp.explain_fast(X_test)
    cal_exp.explain_fast(X_test, threshold=y_test)

    with pytest.raises(ValueError):
        cal_exp.predict_proba(X_test)
    with pytest.raises(ValueError):
        cal_exp.predict_proba(X_test, uq_interval=True)
    y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test[0])
    y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test[0])

    for i, y_hat in enumerate(y_test_hat2):
        assert low[i] <= y_test_hat2[i, 1] <= high[i]

    y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test)
    y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test)

    for i, y_hat in enumerate(y_test_hat2):
        assert low[i] <= y_test_hat2[i, 1] <= high[i]

    cal_exp.fit(X_prop_train, y_prop_train)
    assert cal_exp.fitted
    assert cal_exp.calibrated

    learner = cal_exp.learner
    explainer = cal_exp.explainer

    new_exp = WrapCalibratedExplainer(learner)
    assert new_exp.fitted
    assert not new_exp.calibrated
    assert new_exp.learner == learner

    new_exp = WrapCalibratedExplainer(explainer)
    assert new_exp.fitted
    assert new_exp.calibrated
    assert new_exp.explainer == explainer
    assert new_exp.learner == learner

    cal_exp.plot(X_test)
    cal_exp.plot(X_test, y_test)
    cal_exp.plot(X_test, threshold=y_test[0])
    cal_exp.plot(X_test, y_test, threshold=y_test[0])
