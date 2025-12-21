# pylint: disable=invalid-name, line-too-long, too-many-locals, too-many-statements, redefined-outer-name, duplicate-code, unused-import, too-many-instance-attributes
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

import numpy as np
import pytest
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.utils.exceptions import NotFittedError, ValidationError
from crepes.extras import MondrianCategorizer
from sklearn.ensemble import RandomForestRegressor
import os
from tests._helpers import generic_test


class TestWrapRegressionExplainer:
    """Tests for WrapCalibratedExplainer in regression tasks."""

    # Class attributes instead of instance attributes initialized in __init__
    x_train = None
    y_train = None
    x_cal = None
    y_cal = None
    x_test = None
    y_test = None
    feature_names = None
    explainer = None

    @pytest.fixture(autouse=True)
    def setup(self, regression_dataset):
        """Setup the regression dataset and explainer."""
        (
            self.x_train,
            self.y_train,
            self.x_cal,
            self.y_cal,
            self.x_test,
            self.y_test,
            _,
            _,
            self.feature_names,
        ) = regression_dataset
        self.explainer = WrapCalibratedExplainer(RandomForestRegressor(random_state=42))

    def test_initial_state(self):
        """Test initial unfitted state"""
        assert not self.explainer.fitted, "Should not be fitted initially"
        assert not self.explainer.calibrated, "Should not be calibrated initially"
        with pytest.raises(NotFittedError, match="must be fitted"):
            self.explainer.explain_factual(self.x_test)

    @pytest.mark.parametrize(
        "threshold",
        [
            None,
            0.5,
            [0.5, 0.6],
            (0.4, 0.6),
            [(0.4, 0.6), (0.3, 0.4)],
            -1,
        ],
    )
    def test_prediction_with_thresholds(self, threshold):
        """Test predictions with different threshold values"""
        self.explainer.fit(self.x_train, self.y_train)
        self.explainer.calibrate(self.x_cal, self.y_cal)

        if threshold is not None:
            y_pred = self.explainer.predict(self.x_test, threshold=threshold)
            assert len(y_pred) == len(self.x_test)

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        self.explainer.fit(self.x_train, self.y_train)

        # Test empty input
        # sklearn raises ValueError on empty input during predict
        with pytest.raises(ValueError):
            self.explainer.predict(np.array([]), calibrated=False)

        # Test invalid feature count
        # sklearn raises ValueError when the number of features doesn't match
        with pytest.raises(ValueError):
            rng = np.random.default_rng()
            self.explainer.predict(rng.random((10, len(self.x_train[0]) + 1)), calibrated=False)

        # # Test NaN/Inf handling
        # X_invalid = self.x_test.copy()
        # X_invalid[0,0] = np.nan
        # with pytest.raises(ValueError):
        #     self.explainer.predict(X_invalid)


# generic_test moved to `tests/_helpers.py`


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
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, y_test, _, _, feature_names = (
        regression_dataset
    )
    cal_exp = WrapCalibratedExplainer(RandomForestRegressor())
    assert not cal_exp.fitted
    assert not cal_exp.calibrated

    with pytest.raises(NotFittedError):
        cal_exp.explain_factual(x_test)
    with pytest.raises(NotFittedError):
        cal_exp.explore_alternatives(x_test)

    cal_exp.fit(x_prop_train, y_prop_train)
    assert cal_exp.fitted
    assert not cal_exp.calibrated

    with pytest.warns(UserWarning):
        y_test_hat1 = cal_exp.predict(x_test)
    with pytest.warns(UserWarning):
        y_test_hat2, (low, high) = cal_exp.predict(x_test, uq_interval=True)
    y_test_hat3 = cal_exp.predict(x_test, calibrated=False)
    y_test_hat4, (low4, high4) = cal_exp.predict(x_test, uq_interval=True, calibrated=False)

    for i, y_hat in enumerate(y_test_hat1):
        assert y_test_hat2[i] == pytest.approx(y_hat)
        assert y_test_hat3[i] == pytest.approx(y_hat)
        assert y_test_hat4[i] == pytest.approx(y_hat)
        assert low[i] == pytest.approx(y_hat)
        assert high[i] == pytest.approx(y_hat)
        assert low4[i] == pytest.approx(y_hat)
        assert high4[i] == pytest.approx(y_hat)

    with pytest.raises(ValidationError):
        cal_exp.predict(x_test, threshold=y_test)
    with pytest.raises(ValidationError):
        cal_exp.predict(x_test, uq_interval=True, threshold=y_test)
    with pytest.raises(ValidationError):
        cal_exp.predict_proba(x_test)
    with pytest.raises(ValidationError):
        cal_exp.predict_proba(x_test, uq_interval=True)
    with pytest.raises(NotFittedError):
        cal_exp.predict_proba(x_test, threshold=y_test)
    with pytest.raises(NotFittedError):
        cal_exp.predict_proba(x_test, uq_interval=True, threshold=y_test)
    with pytest.raises(NotFittedError):
        cal_exp.explain_factual(x_test)
    with pytest.raises(NotFittedError):
        cal_exp.explore_alternatives(x_test)
    with pytest.raises(NotFittedError):
        cal_exp.explain_factual(x_test, threshold=y_test)
    with pytest.raises(NotFittedError):
        cal_exp.explore_alternatives(x_test, threshold=y_test)

    cal_exp.calibrate(x_cal, y_cal, feature_names=feature_names)
    assert cal_exp.fitted
    assert cal_exp.calibrated

    y_test_hat3 = cal_exp.predict(x_test, calibrated=False)
    y_test_hat4, (low4, high4) = cal_exp.predict(x_test, uq_interval=True, calibrated=False)

    for i, y_hat in enumerate(y_test_hat1):
        assert y_test_hat3[i] == pytest.approx(y_hat)
        assert y_test_hat4[i] == pytest.approx(y_hat)
        assert low4[i] == pytest.approx(y_hat)
        assert high4[i] == pytest.approx(y_hat)

    y_test_hat1 = cal_exp.predict(x_test)
    y_test_hat2, (low, high) = cal_exp.predict(x_test, uq_interval=True)

    for i, y_hat in enumerate(y_test_hat2):
        # Ensure the point prediction is consistent with the reported interval
        assert low[i] <= y_test_hat1[i] <= high[i]
        assert low[i] <= y_hat <= high[i]

    y_test_hat1 = cal_exp.predict(x_test, threshold=y_test)
    y_test_hat2, (low, high) = cal_exp.predict(x_test, uq_interval=True, threshold=y_test)

    cal_exp.explain_factual(x_test)
    cal_exp.explore_alternatives(x_test)
    cal_exp.explain_factual(x_test, threshold=y_test)
    cal_exp.explore_alternatives(x_test, threshold=y_test)

    with pytest.raises(ValidationError):
        cal_exp.predict_proba(x_test)
    with pytest.raises(ValidationError):
        cal_exp.predict_proba(x_test, uq_interval=True)
    y_test_hat1 = cal_exp.predict_proba(x_test, threshold=y_test[0])
    y_test_hat2, (low, high) = cal_exp.predict_proba(x_test, uq_interval=True, threshold=y_test[0])

    for i, y_hat in enumerate(y_test_hat2):
        assert low[i] <= y_test_hat2[i, 1] <= high[i]

    y_test_hat1 = cal_exp.predict_proba(x_test, threshold=y_test)
    y_test_hat2, (low, high) = cal_exp.predict_proba(x_test, uq_interval=True, threshold=y_test)

    for i, y_hat in enumerate(y_test_hat2):
        assert low[i] <= y_test_hat2[i, 1] <= high[i]

    cal_exp = generic_test(cal_exp, x_prop_train, y_prop_train, x_test, y_test)
    cal_exp.plot(x_test, show=False, threshold=y_test[0])
    cal_exp.plot(x_test, y_test, show=False, threshold=y_test[0])


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
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, y_test, _, _, feature_names = (
        regression_dataset
    )
    # In fast mode skip this long conditional test (external system may slow it)
    if bool(os.getenv("FAST_TESTS")):
        pytest.skip("Skipping long conditional regression test in FAST_TESTS mode")

    cal_exp = WrapCalibratedExplainer(RandomForestRegressor())
    cal_exp.fit(x_prop_train, y_prop_train)

    # test with MondrianCategorizer
    mc = MondrianCategorizer()
    mc.fit(x_cal, f=cal_exp.learner.predict, no_bins=5)

    cal_exp.calibrate(x_cal, y_cal, mc=mc, feature_names=feature_names)
    conditional_test(cal_exp, x_prop_train, y_prop_train, x_test, y_test)

    # test with predict as categorizer
    cal_exp.calibrate(
        x_cal, y_cal, mc=lambda x: cal_exp.learner.predict(x) > 0.5, feature_names=feature_names
    )
    conditional_test(cal_exp, x_prop_train, y_prop_train, x_test, y_test)


def conditional_test(cal_exp, x_prop_train, y_prop_train, x, y):
    """
    Tests the functionality of a calibrated explainer for conditional regression.
    This function performs a series of assertions to ensure that the
    calibrated explainer (`cal_exp`) is properly fitted and calibrated.
    It also checks the behavior of the `WrapCalibratedExplainer` class
    when initialized with the learner and explainer from `cal_exp`.
    Parameters:
    cal_exp (object): The calibrated explainer to be tested.
    x_prop_train (array-like): Training data features for the explainer.
    y_prop_train (array-like): Training data labels for the explainer.
    x (array-like): Test data features for plotting.
    y (array-like): Test data labels for plotting.
    Returns:
    object: The fitted and calibrated explainer (`cal_exp`).
    """
    assert cal_exp.fitted
    assert cal_exp.calibrated

    y_test_hat1 = cal_exp.predict(x)
    y_test_hat2, (low, high) = cal_exp.predict(x, uq_interval=True)

    for i, y_hat in enumerate(y_test_hat2):
        # Point prediction should lie within the reported uncertainty interval
        assert low[i] <= y_test_hat1[i] <= high[i]
        # And the conditional estimate should also be within the limits
        assert low[i] <= y_hat <= high[i]

    y_test_hat1 = cal_exp.predict(x, threshold=y)
    y_test_hat2, (low, high) = cal_exp.predict(x, uq_interval=True, threshold=y)

    cal_exp.explain_factual(x)
    cal_exp.explore_alternatives(x)
    cal_exp.explain_factual(x, threshold=y)
    cal_exp.explore_alternatives(x, threshold=y)

    with pytest.raises(ValidationError):
        cal_exp.predict_proba(x)
    with pytest.raises(ValidationError):
        cal_exp.predict_proba(x, uq_interval=True)
    y_test_hat1 = cal_exp.predict_proba(x, threshold=y[0])
    y_test_hat2, (low, high) = cal_exp.predict_proba(x, uq_interval=True, threshold=y[0])

    for i, y_hat in enumerate(y_test_hat2):
        assert low[i] <= y_test_hat2[i, 1] <= high[i]


@pytest.mark.viz
def test_wrap_regression_accepts_int_threshold(regression_dataset):
    """WrapCalibratedExplainer should accept integer thresholds without errors."""
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, _y_test, _, _, feature_names = (
        regression_dataset
    )
    cal_exp = WrapCalibratedExplainer(RandomForestRegressor())
    cal_exp.fit(x_prop_train, y_prop_train)
    cal_exp.calibrate(x_cal, y_cal, feature_names=feature_names)

    # Predict with integer threshold
    y_pred_int = cal_exp.predict(x_test, threshold=0)
    assert len(y_pred_int) == len(x_test)

    # Predict_proba with integer threshold and intervals
    proba_int, (low, high) = cal_exp.predict_proba(x_test, uq_interval=True, threshold=0)
    assert proba_int.shape == (len(x_test), 2)
    assert len(low) == len(high) == len(x_test)

    # Explanations with integer threshold
    fx = cal_exp.explain_factual(x_test, threshold=0)
    fx.plot(show=False)
    ax = cal_exp.explore_alternatives(x_test, threshold=0)
    ax.plot(show=False)


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
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, y_test, _, _, feature_names = (
        regression_dataset
    )
    cal_exp = WrapCalibratedExplainer(RandomForestRegressor())
    cal_exp.fit(x_prop_train, y_prop_train)
    cal_exp.calibrate(x_cal, y_cal, feature_names=feature_names, perturb=True)
    assert cal_exp.fitted
    assert cal_exp.calibrated

    y_test_hat1 = cal_exp.predict(x_test)
    y_test_hat2, (low, high) = cal_exp.predict(x_test, uq_interval=True)

    for i, y_hat in enumerate(y_test_hat2):
        assert y_test_hat1[i] == y_hat
        assert low[i] <= y_hat <= high[i]

    y_test_hat1 = cal_exp.predict(x_test, threshold=y_test)
    y_test_hat2, (low, high) = cal_exp.predict(x_test, uq_interval=True, threshold=y_test)

    cal_exp.explain_factual(x_test)
    cal_exp.explore_alternatives(x_test)
    cal_exp.explain_factual(x_test, threshold=y_test)
    cal_exp.explore_alternatives(x_test, threshold=y_test)
    cal_exp.explain_fast(x_test)
    cal_exp.explain_fast(x_test, threshold=y_test)

    with pytest.raises(ValidationError):
        cal_exp.predict_proba(x_test)
    with pytest.raises(ValidationError):
        cal_exp.predict_proba(x_test, uq_interval=True)
    y_test_hat1 = cal_exp.predict_proba(x_test, threshold=y_test[0])
    y_test_hat2, (low, high) = cal_exp.predict_proba(x_test, uq_interval=True, threshold=y_test[0])

    for i, y_hat in enumerate(y_test_hat2):
        assert low[i] <= y_test_hat2[i, 1] <= high[i]

    y_test_hat1 = cal_exp.predict_proba(x_test, threshold=y_test)
    y_test_hat2, (low, high) = cal_exp.predict_proba(x_test, uq_interval=True, threshold=y_test)

    for i, y_hat in enumerate(y_test_hat2):
        assert low[i] <= y_test_hat2[i, 1] <= high[i]

    cal_exp = generic_test(cal_exp, x_prop_train, y_prop_train, x_test, y_test)
    cal_exp.plot(x_test, show=False, threshold=y_test[0])
    cal_exp.plot(x_test, y_test, show=False, threshold=y_test[0])
