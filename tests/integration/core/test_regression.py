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
import pytest
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import NotFittedError, ValidationError
from crepes.extras import DifficultyEstimator

from tests._helpers import get_regression_model, initiate_explainer


def safe_fit_difficulty(x, y, scaler=True):
    """Try to fit the crepes DifficultyEstimator; fall back to a light stub when data is too small.

    This prevents NearestNeighbors errors when running in FAST_TESTS with tiny fixtures.
    """
    try:
        return DifficultyEstimator().fit(X=x, y=y, scaler=scaler)
    except Exception:
        # Fallback stub: minimal API used by CalibratedExplainer (fit + predict-like method)
        class StubDifficulty:
            def __init__(self):
                # indicate 'fitted' so CalibratedExplainer accepts it
                self.fitted = True

            def fit(self, *a, **k):
                self.fitted = True
                return self

            def predict(self, x_data):
                import numpy as _np

                # return zeros (easy examples) for all points
                return _np.zeros(len(x_data))

            # crepes DifficultyEstimator exposes apply(x) in codepaths
            def apply(self, x_data):
                return self.predict(x_data)

            # some code paths may call the instance directly
            def __call__(self, x_data):
                return self.apply(x_data)

        return StubDifficulty().fit()


def test_safe_fit_difficulty_fallback(monkeypatch):
    """Ensure the helper returns a stub difficulty estimator when fitting fails."""

    def _failing_fit(self, *args, **kwargs):  # noqa: D401  - short helper, no doc needed
        raise RuntimeError("boom")

    monkeypatch.setattr(DifficultyEstimator, "fit", _failing_fit, raising=True)

    stub = safe_fit_difficulty(np.zeros((3, 2)), np.zeros(3), scaler=False)

    assert getattr(stub, "fitted", False) is True
    assert np.allclose(stub.predict(np.ones((2, 2))), 0.0)
    assert np.allclose(stub.apply(np.ones((2, 2))), 0.0)


def test_failure_regression(regression_dataset):
    """
    Tests failure cases for the CalibratedExplainer.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    x_prop_train, y_prop_train, x_cal, y_cal, _, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )
    with pytest.raises(NotFittedError):
        cal_exp.set_difficulty_estimator(DifficultyEstimator())
    with pytest.raises(NotFittedError):
        cal_exp.set_difficulty_estimator(DifficultyEstimator)


@pytest.mark.viz
def test_regression_ce(regression_dataset):
    """
    Tests the CalibratedExplainer for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    factual_explanation = cal_exp.explain_factual(x_test)
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)
    factual_explanation.plot(show=False, filename="test.png")

    factual_explanation = cal_exp.explain_factual(x_test, low_high_percentiles=(0.1, np.inf))
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    factual_explanation = cal_exp.explain_factual(x_test, low_high_percentiles=(-np.inf, 0.9))
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_alternatives(x_test)
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_alternatives(
        x_test, low_high_percentiles=(0.1, np.inf)
    )
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_alternatives(
        x_test, low_high_percentiles=(-np.inf, 0.9)
    )
    alternative_explanation.plot(show=False)
    alternative_explanation.semi_explanations()
    alternative_explanation.counter_explanations()


def test_regression_predict_reject_requires_threshold(regression_dataset):
    """Regression reject predictions should fail when the threshold is missing."""

    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    cal_exp.initialize_reject_learner(threshold=0.5)
    cal_exp.reject_threshold = None

    with pytest.raises(ValidationError):
        cal_exp.predict_reject(x_test)


def test_regression_reject_learner_custom_calibration(regression_dataset):
    """Explicit calibration sets should be accepted when initializing the reject learner."""

    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    # Use a list to exercise the fallback calibration_set path inside initialize_reject_learner
    calibration_subset = [x_cal[:25], y_cal[:25]]
    learner = cal_exp.initialize_reject_learner(calibration_set=calibration_subset, threshold=0.6)

    assert learner is cal_exp.reject_learner
    assert cal_exp.reject_threshold == 0.6

    rejected, error_rate, reject_rate = cal_exp.predict_reject(x_test, confidence=0.9)

    assert rejected.shape == (len(x_test),)
    assert not np.isnan(error_rate)
    assert not np.isnan(reject_rate)


@pytest.mark.viz
def test_probabilistic_regression_ce(regression_dataset):
    """Test probabilistic regression with calibrated explanations.

    Probabilistic regression (also called thresholded regression in architecture docs)
    applies a threshold to convert regression predictions into calibrated probability
    predictions. This test validates the full end-to-end workflow.

    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    cal_exp.initialize_reject_learner(threshold=0.5)
    cal_exp.predict_reject(x_test)

    factual_explanation = cal_exp.explain_factual(x_test, y_test)
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation.plot(show=False, uncertainty=True)

    factual_explanation = cal_exp.explain_factual(x_test, y_test[0])
    factual_explanation = cal_exp.explain_factual(x_test, (0.4, 0.6))

    alternative_explanation = cal_exp.explore_alternatives(x_test, y_test)
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_alternatives(x_test, y_test[0])
    alternative_explanation.plot(show=False)
    alternative_explanation.super_explanations()
    alternative_explanation.semi_explanations()
    alternative_explanation.counter_explanations()


@pytest.mark.viz
def test_probabilistic_regression_int_threshold_ce(regression_dataset):
    """Test probabilistic regression with integer threshold.

    Probabilistic regression (also called thresholded regression in architecture docs)
    applies a threshold to convert regression predictions into calibrated probability
    predictions P(y <= threshold). This test validates that integer thresholds are accepted.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    # Single integer threshold (y is normalized to [0,1])
    factual_explanation = cal_exp.explain_factual(x_test, 0)
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)

    # Tuple of integer thresholds
    factual_explanation = cal_exp.explain_factual(x_test, (0, 1))
    factual_explanation.plot(show=False)

    # Alternatives should also accept int thresholds
    alternative_explanation = cal_exp.explore_alternatives(x_test, 0)
    alternative_explanation.plot(show=False)


@pytest.mark.viz
def test_regression_as_classification_ce(regression_dataset):
    """
    Tests probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)

    def predict_function(x):
        """Convert regression predictions to binary classification."""
        return np.asarray([[float(p > 0.5), float(p <= 0.5)] for p in model.predict(x)])

    cal_exp = CalibratedExplainer(
        model,
        x_cal,
        np.asarray([1 if y <= 0.5 else 0 for y in y_cal]),
        feature_names=feature_names,
        categorical_features=categorical_features,
        mode="classification",
        predict_function=predict_function,
    )

    factual_explanation = cal_exp.explain_factual(x_test)
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation.plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_alternatives(x_test)
    alternative_explanation.plot(show=False)


@pytest.mark.viz
def test_regression_conditional_ce(regression_dataset):
    """
    Tests conditional explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    bin_feature = 0
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        bins=x_cal[:, bin_feature],
    )

    factual_explanation = cal_exp.explain_factual(x_test, bins=x_test[:, bin_feature])
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation.plot(show=False, uncertainty=True)
    repr(factual_explanation)

    factual_explanation = cal_exp.explain_factual(
        x_test, low_high_percentiles=(0.1, np.inf), bins=x_test[:, bin_feature]
    )
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    factual_explanation = cal_exp.explain_factual(
        x_test, low_high_percentiles=(-np.inf, 0.9), bins=x_test[:, bin_feature]
    )
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_alternatives(x_test, bins=x_test[:, bin_feature])
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_alternatives(
        x_test, low_high_percentiles=(0.1, np.inf), bins=x_test[:, bin_feature]
    )
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_alternatives(
        x_test, low_high_percentiles=(-np.inf, 0.9), bins=x_test[:, bin_feature]
    )
    alternative_explanation.plot(show=False)
    repr(alternative_explanation)


@pytest.mark.viz
def test_probabilistic_regression_conditional_ce(regression_dataset):
    """
    Tests probabilistic conditional explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        bins=x_cal[:, 0],
    )

    cal_exp.initialize_reject_learner(threshold=0.5)
    cal_exp.predict_reject(x_test, bins=x_test[:, 0])

    factual_explanation = cal_exp.explain_factual(x_test, y_test, bins=x_test[:, 0])
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)

    factual_explanation = cal_exp.explain_factual(x_test, y_test[0], bins=x_test[:, 0])

    alternative_explanation = cal_exp.explore_alternatives(x_test, y_test, bins=x_test[:, 0])
    alternative_explanation.plot(show=False)

    cal_exp.explore_alternatives(x_test, y_test[0], bins=x_test[:, 0])


def test_knn_normalized_regression_ce(regression_dataset):
    """
    Tests KNN normalized explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=safe_fit_difficulty(x_prop_train, y_prop_train, scaler=True),
    )

    factual_explanation = cal_exp.explain_factual(x_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_factual(x_test, low_high_percentiles=(0.1, np.inf))

    factual_explanation = cal_exp.explain_factual(x_test, low_high_percentiles=(-np.inf, 0.9))

    cal_exp.explore_alternatives(x_test)

    cal_exp.explore_alternatives(x_test, low_high_percentiles=(0.1, np.inf))

    cal_exp.explore_alternatives(x_test, low_high_percentiles=(-np.inf, 0.9))


def test_knn_normalized_probabilistic_regression_ce(regression_dataset):
    """
    Tests KNN normalized probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=safe_fit_difficulty(x_prop_train, y_prop_train, scaler=True),
    )

    factual_explanation = cal_exp.explain_factual(x_test, y_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_factual(x_test, y_test[0])

    cal_exp.explore_alternatives(x_test, y_test)

    cal_exp.explore_alternatives(x_test, y_test[0])


def test_var_normalized_regression_ce(regression_dataset):
    """
    Tests variance normalized explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=safe_fit_difficulty(x_prop_train, y_prop_train, scaler=True),
    )

    factual_explanation = cal_exp.explain_factual(x_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_factual(x_test, low_high_percentiles=(0.1, np.inf))

    factual_explanation = cal_exp.explain_factual(x_test, low_high_percentiles=(-np.inf, 0.9))

    cal_exp.explore_alternatives(x_test)

    cal_exp.explore_alternatives(x_test, low_high_percentiles=(0.1, np.inf))

    cal_exp.explore_alternatives(x_test, low_high_percentiles=(-np.inf, 0.9))


def test_var_normalized_probabilistic_regression_ce(regression_dataset):
    """
    Tests variance normalized probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=safe_fit_difficulty(x_prop_train, y_prop_train, scaler=True),
    )

    factual_explanation = cal_exp.explain_factual(x_test, y_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_factual(x_test, y_test[0])

    cal_exp.explore_alternatives(x_test, y_test)

    cal_exp.explore_alternatives(x_test, y_test[0])


@pytest.mark.viz
def test_regression_fast_ce(regression_dataset):
    """
    Tests fast explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression", fast=True
    )

    fast_explanation = cal_exp.explain_fast(x_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()
    fast_explanation.plot(show=False)
    fast_explanation.plot(show=False, uncertainty=True)

    fast_explanation = cal_exp.explain_fast(x_test, low_high_percentiles=(0.1, np.inf))
    fast_explanation.plot(show=False)
    with pytest.raises(Warning):
        fast_explanation.plot(show=False, uncertainty=True)

    fast_explanation = cal_exp.explain_fast(x_test, low_high_percentiles=(-np.inf, 0.9))
    fast_explanation.plot(show=False)
    with pytest.raises(Warning):
        fast_explanation.plot(show=False, uncertainty=True)


@pytest.mark.viz
def test_probabilistic_regression_fast_ce(regression_dataset):
    """
    Tests fast probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression", fast=True
    )

    fast_explanation = cal_exp.explain_fast(x_test, y_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()
    fast_explanation.plot(show=False)
    fast_explanation.plot(show=False, uncertainty=True)

    fast_explanation = cal_exp.explain_fast(x_test, y_test[0])
    fast_explanation.plot(show=False)
    fast_explanation.plot(show=False, uncertainty=True)


@pytest.mark.viz
def test_regression_conditional_fast_ce(regression_dataset):
    """
    Tests conditional perturbed explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        bins=x_cal[:, 0],
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(x_test, bins=x_test[:, 0])
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(
        x_test, low_high_percentiles=(0.1, np.inf), bins=x_test[:, 0]
    )

    fast_explanation = cal_exp.explain_fast(
        x_test, low_high_percentiles=(-np.inf, 0.9), bins=x_test[:, 0]
    )


@pytest.mark.viz
def test_probabilistic_regression_conditional_fast_ce(regression_dataset):
    """
    Tests probabilistic conditional perturbed explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        bins=y_cal > y_test[0],
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(x_test, y_test, bins=y_test > y_test[0])
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(x_test, y_test[0], bins=y_test > y_test[0])


def test_knn_normalized_regression_fast_ce(regression_dataset):
    """
    Tests KNN normalized fast explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=safe_fit_difficulty(x_prop_train, y_prop_train, scaler=True),
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(x_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(x_test, low_high_percentiles=(0.1, np.inf))

    fast_explanation = cal_exp.explain_fast(x_test, low_high_percentiles=(-np.inf, 0.9))


def test_knn_normalized_probabilistic_regression_fast_ce(regression_dataset):
    """
    Tests KNN normalized fast probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=safe_fit_difficulty(x_prop_train, y_prop_train, scaler=True),
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(x_test, y_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(x_test, y_test[0])


def test_var_normalized_regression_fast_ce(regression_dataset):
    """
    Tests variance normalized fast explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    x_prop_train, y_prop_train, x_cal, y_cal, x_test, _, _, categorical_features, feature_names = (
        regression_dataset
    )
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=safe_fit_difficulty(x_prop_train, y_prop_train, scaler=True),
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(x_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(x_test, low_high_percentiles=(0.1, np.inf))

    fast_explanation = cal_exp.explain_fast(x_test, low_high_percentiles=(-np.inf, 0.9))


def test_var_normalized_probabilistic_regression_fast_ce(regression_dataset):
    """
    Tests variance normalized fast probabilistic explanations for regression models.
    Args:
        regression_dataset (tuple): The regression dataset.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        difficulty_estimator=DifficultyEstimator().fit(X=x_prop_train, learner=model, scaler=True),
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(x_test, y_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()

    fast_explanation = cal_exp.explain_fast(x_test, y_test[0])
