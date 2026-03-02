# pylint: disable=invalid-name, line-too-long, too-many-locals, too-many-statements, redefined-outer-name, duplicate-code, unused-import, too-many-instance-attributes
"""
Module for testing the WrapCalibratedExplainer class for regression tasks.

IMPORTANT: THESE TESTS MUST NOT BE REMOVED OR SILENTLY MODIFIED. They are
protected integration tests relied on by release gating and regression
protection tooling. See docs/improvement/test-quality-method/README.md.

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
import os
import pytest
import json
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.utils.exceptions import (
    IncompatibleStateError,
    NotFittedError,
    ValidationError,
)
from crepes.extras import MondrianCategorizer
from sklearn.ensemble import RandomForestRegressor

from tests.helpers.explainer_utils import generic_test

pytestmark = pytest.mark.integration


def assert_payload_close(left, right):
    """Recursively compare nested persistence payloads."""
    if isinstance(left, tuple) and isinstance(right, tuple):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            assert_payload_close(lhs, rhs)
        return
    np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=1e-9)


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
        """Test initial unfitted state

        IMPORTANT: THIS TEST MUST NOT BE REMOVED.
        """
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
        """Test predictions with different threshold values

        IMPORTANT: THIS TEST MUST NOT BE REMOVED.
        """
        self.explainer.fit(self.x_train, self.y_train)
        self.explainer.calibrate(self.x_cal, self.y_cal)

        if threshold is not None:
            y_pred = self.explainer.predict(self.x_test, threshold=threshold)
            assert len(y_pred) == len(self.x_test)

    def test_edge_cases(self):
        """Test edge cases and error conditions

        IMPORTANT: THIS TEST MUST NOT BE REMOVED.
        """
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

    IMPORTANT: THIS TEST MUST NOT BE REMOVED.

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

    fx = cal_exp.explain_factual(x_test)
    fx.add_conjunctions()
    alt = cal_exp.explore_alternatives(x_test)
    alt.add_conjunctions()
    fx = cal_exp.explain_factual(x_test, threshold=y_test)
    fx.add_conjunctions()
    alt = cal_exp.explore_alternatives(x_test, threshold=y_test)
    alt.add_conjunctions()
    # Basic sanity assertions to ensure the explainer produced results
    assert fx is not None
    assert alt is not None

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

    IMPORTANT: THIS TEST MUST NOT BE REMOVED.

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
    # Basic sanity assertions to ensure the explainer produced results in this conditional test
    assert cal_exp.fitted
    assert cal_exp.calibrated


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

    fx = cal_exp.explain_factual(x)
    fx.add_conjunctions()
    alt = cal_exp.explore_alternatives(x)
    alt.add_conjunctions()
    fx = cal_exp.explain_factual(x, threshold=y)
    fx.add_conjunctions()
    alt = cal_exp.explore_alternatives(x, threshold=y)
    alt.add_conjunctions()

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
    """WrapCalibratedExplainer should accept integer thresholds without errors.

    IMPORTANT: THIS TEST MUST NOT BE REMOVED.
    """
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
    fx.add_conjunctions()
    fx.plot(show=False)
    ax = cal_exp.explore_alternatives(x_test, threshold=0)
    ax.add_conjunctions()
    ax.plot(show=False)


def test_wrap_regression_fast_ce(regression_dataset):
    """
    Test the WrapCalibratedExplainer class for fast regression.

    IMPORTANT: THIS TEST MUST NOT BE REMOVED.

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

    fx = cal_exp.explain_factual(x_test)
    fx.add_conjunctions()
    alt = cal_exp.explore_alternatives(x_test)
    alt.add_conjunctions()
    fx = cal_exp.explain_factual(x_test, threshold=y_test)
    fx.add_conjunctions()
    alt = cal_exp.explore_alternatives(x_test, threshold=y_test)
    alt.add_conjunctions()
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


def test_should_roundtrip_state_with_native_regression_primitive_when_saved(
    tmp_path, regression_dataset
):
    """ADR-031 regression round-trip should persist interval_regressor primitives."""
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, y_test, _, _, feature_names = (
        regression_dataset
    )
    wrapper = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=24, random_state=13))
    wrapper.fit(x_prop_train, y_prop_train)
    wrapper.calibrate(x_cal, y_cal, feature_names=feature_names)
    threshold = float(np.median(y_test))
    baseline = wrapper.predict_proba(x_test[:14], threshold=threshold, uq_interval=True)

    state_dir = tmp_path / "regression_state"
    wrapper.save_state(state_dir)

    primitive = json.loads((state_dir / "calibrator_primitive.json").read_text(encoding="utf-8"))
    assert primitive["calibrator_type"] == "interval_regressor"
    assert primitive["schema_version"] == 1

    restored = WrapCalibratedExplainer.load_state(state_dir)
    reloaded = restored.predict_proba(x_test[:14], threshold=threshold, uq_interval=True)
    assert_payload_close(baseline, reloaded)


def test_should_roundtrip_state_with_pickle_fallback_when_fast_interval_calibrator_is_used(
    tmp_path,
    regression_dataset,
):
    """Fast regression should exercise the python_pickle calibrator primitive path."""
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, y_test, _, _, feature_names = (
        regression_dataset
    )
    wrapper = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=22, random_state=17))
    wrapper.fit(x_prop_train, y_prop_train)
    wrapper.calibrate(x_cal, y_cal, feature_names=feature_names, fast=True)
    threshold = float(np.median(y_test))
    baseline = wrapper.predict_proba(x_test[:12], threshold=threshold, uq_interval=True)

    state_dir = tmp_path / "regression_pickle_fallback_state"
    wrapper.save_state(state_dir)

    primitive = json.loads((state_dir / "calibrator_primitive.json").read_text(encoding="utf-8"))
    assert primitive["calibrator_type"] == "python_pickle"
    assert primitive["schema_version"] == 1
    assert "payload" in primitive and "pickle_b64" in primitive["payload"]

    restored = WrapCalibratedExplainer.load_state(state_dir)
    reloaded = restored.predict_proba(x_test[:12], threshold=threshold, uq_interval=True)
    assert_payload_close(baseline, reloaded)


def test_should_fail_load_when_pickle_fallback_payload_checksum_is_tampered(
    tmp_path,
    regression_dataset,
):
    """load_state should reject tampered python_pickle payloads even when manifest checksums are recomputed."""
    x_prop_train, y_prop_train, x_cal, y_cal, _, _, _, _, feature_names = regression_dataset
    wrapper = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=18, random_state=19))
    wrapper.fit(x_prop_train, y_prop_train)
    wrapper.calibrate(x_cal, y_cal, feature_names=feature_names, fast=True)

    state_dir = tmp_path / "regression_tamper_state"
    wrapper.save_state(state_dir)

    primitive_path = state_dir / "calibrator_primitive.json"
    primitive = json.loads(primitive_path.read_text(encoding="utf-8"))
    primitive["payload"]["pickle_b64"] = primitive["payload"]["pickle_b64"][:-4] + "AAAA"
    primitive_path.write_text(json.dumps(primitive, indent=2, sort_keys=True), encoding="utf-8")

    manifest_path = state_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with primitive_path.open("rb") as handle:
        import hashlib

        manifest["files"]["calibrator_primitive.json"] = hashlib.sha256(handle.read()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        IncompatibleStateError, match="Calibrator primitive checksum validation failed"
    ):
        WrapCalibratedExplainer.load_state(state_dir)
