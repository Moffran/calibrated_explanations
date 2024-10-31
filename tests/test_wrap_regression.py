# pylint: disable=invalid-name, line-too-long, too-many-locals, too-many-statements, redefined-outer-name, duplicate-code, unused-import
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
from sklearn.ensemble import RandomForestRegressor

from crepes.extras import MondrianCategorizer
from calibrated_explanations import WrapCalibratedExplainer

from tests.test_regression import regression_dataset

def generic_test(cal_exp, X_prop_train, y_prop_train, X_test, y_test):
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
    return cal_exp

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

    cal_exp = generic_test(cal_exp, X_prop_train, y_prop_train, X_test, y_test)
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

    generic_test(cal_exp, X_prop_train, y_prop_train, X_test, y_test)

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

    cal_exp = generic_test(cal_exp, X_prop_train, y_prop_train, X_test, y_test)
    cal_exp.plot(X_test, threshold=y_test[0])
    cal_exp.plot(X_test, y_test, threshold=y_test[0])
