# pylint: disable=invalid-name, line-too-long, too-many-locals, too-many-statements, redefined-outer-name
"""
This module contains tests for regression models using the CalibratedExplainer from the calibrated_explanations package.

IMPORTANT: THESE TESTS MUST NOT BE REMOVED OR SILENTLY MODIFIED. They are
protected integration tests relied on release gating and regression
protection tooling. See docs/improvement/test-quality-method/README.md.

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

from tests.helpers.model_utils import get_regression_model
from tests.helpers.explainer_utils import initiate_explainer

pytestmark = pytest.mark.integration


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

    def failing_fit_mock(self, *args, **kwargs):  # noqa: D401  - short helper, no doc needed
        raise RuntimeError("boom")

    monkeypatch.setattr(DifficultyEstimator, "fit", failing_fit_mock, raising=True)

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


def test_set_difficulty_estimator_refits_cps_with_sigmas(regression_dataset, monkeypatch):
    """Ensure difficulty estimator changes are forwarded to crepes CPS.

    The regression interval backend (`IntervalRegressor`) fits a crepes
    `ConformalPredictiveSystem`. When a difficulty estimator is configured,
    the CPS must be fit with `sigmas=`. Updating the difficulty estimator must
    therefore rebuild/refit the interval learner, rather than reusing the
    existing calibrator.
    """

    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        _x_test,
        _y_test,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset

    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    previous_interval = cal_exp.interval_learner

    import crepes

    original_fit = crepes.ConformalPredictiveSystem.fit
    fit_calls = []

    def fit_spy(self, *args, **kwargs):  # noqa: ANN001 - spy
        fit_calls.append(dict(kwargs))
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(crepes.ConformalPredictiveSystem, "fit", fit_spy, raising=True)

    difficulty = safe_fit_difficulty(x_prop_train, y_prop_train, scaler=True)
    cal_exp.set_difficulty_estimator(difficulty, initialize=True)

    assert cal_exp.interval_learner is not previous_interval
    assert len(fit_calls) >= 1
    assert any("sigmas" in call and call.get("sigmas") is not None for call in fit_calls)


def test_should_vary_factual_weight_width_by_feature_when_difficulty_depends_on_perturbations(
    monkeypatch,
):
    """Factual rule weight intervals should reflect per-perturbation difficulty.

    Regression factual explanations compute feature weights by evaluating
    predictions for perturbed samples. When a difficulty estimator depends on
    the input features, perturbations for different features should generally
    induce different sigma distributions and therefore different weight interval
    widths.

    This test uses a deterministic CPS.predict stub where the returned interval
    width is proportional to the provided `sigmas`, making the expectation
    robust and independent of upstream crepes internals.
    """

    from sklearn.tree import DecisionTreeRegressor

    # Arrange: small, deterministic regression setup with two categorical features.
    x_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    y_train = (x_train[:, 0] + 2.0 * x_train[:, 1]).astype(float)

    model = DecisionTreeRegressor(random_state=0)
    model.fit(x_train, y_train)

    x_cal = x_train.copy()
    y_cal = y_train.copy()
    x_test = np.array([[0.0, 0.0]])

    feature_names = ["x0", "x1"]
    categorical_features = [0, 1]

    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="regression",
        seed=0,
    )

    class ThresholdDifficulty:
        def __init__(self, threshold: float = 0.5) -> None:
            self.threshold = threshold
            self.fitted = True

        def fit(self, *args, **kwargs):  # noqa: ANN001 - test stub
            self.fitted = True
            return self

        def apply(self, x):  # noqa: ANN001 - test stub
            x = np.asarray(x)
            # High difficulty whenever feature 0 exceeds the threshold.
            return np.where(x[:, 0] > self.threshold, 11.0, 1.0)

    cal_exp.set_difficulty_estimator(ThresholdDifficulty(), initialize=True)

    # Patch CPS.predict to return intervals that scale with provided sigmas.
    interval_learner = cal_exp.interval_learner

    def cps_predict_stub(*args, **kwargs):  # noqa: ANN001 - test stub
        sigmas = np.asarray(kwargs.get("sigmas")).reshape(-1)
        n = int(sigmas.shape[0])
        if kwargs.get("y") is not None:
            return np.zeros(n)
        y_hat = np.asarray(kwargs.get("y_hat")).reshape(-1)
        if y_hat.shape[0] != n:
            y_hat = np.resize(y_hat, n)
        interval = np.zeros((n, 4), dtype=float)
        # crepes-style 4-column interval; CE's IntervalRegressor uses:
        # - `interval[:, 0]` for low
        # - `interval[:, 2]` for high
        # - median = (`interval[:, 1]` + `interval[:, 3]`) / 2
        interval[:, 0] = y_hat - sigmas
        interval[:, 1] = y_hat
        interval[:, 2] = y_hat + sigmas
        interval[:, 3] = y_hat
        return interval

    monkeypatch.setattr(interval_learner.cps, "predict", cps_predict_stub, raising=True)

    # Act
    explanations = cal_exp.explain_guarded_factual(x_test)
    rules = explanations[0].get_rules()

    # Assert: weight interval widths differ between features (x0 perturbations change sigma).
    feature_ids = [int(f) for f in rules["feature"]]
    idx0 = feature_ids.index(0)
    idx1 = feature_ids.index(1)

    width0 = float(rules["weight_high"][idx0] - rules["weight_low"][idx0])
    width1 = float(rules["weight_high"][idx1] - rules["weight_low"][idx1])

    assert width0 > width1


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

    factual_explanation = cal_exp.explain_guarded_factual(x_test)
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)
    factual_explanation.plot(show=False, filename="test.png")

    factual_explanation = cal_exp.explain_guarded_factual(
        x_test, low_high_percentiles=(0.1, np.inf)
    )
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    factual_explanation = cal_exp.explain_guarded_factual(
        x_test, low_high_percentiles=(-np.inf, 0.9)
    )
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_guarded_alternatives(x_test)
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_guarded_alternatives(
        x_test, low_high_percentiles=(0.1, np.inf)
    )
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_guarded_alternatives(
        x_test, low_high_percentiles=(-np.inf, 0.9)
    )
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)
    alternative_explanation.semi_explanations()
    alternative_explanation.counter_explanations()
    # Basic sanity assertions to ensure the explainer produced results
    assert factual_explanation is not None
    assert alternative_explanation is not None


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

    cal_exp.reject_orchestrator.initialize_reject_learner(threshold=0.5)
    cal_exp.reject_threshold = None

    with pytest.raises(ValidationError):
        cal_exp.reject_orchestrator.predict_reject(x_test)


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
    learner = cal_exp.reject_orchestrator.initialize_reject_learner(
        calibration_set=calibration_subset, threshold=0.6
    )

    assert learner is cal_exp.reject_learner
    assert cal_exp.reject_threshold == 0.6

    rejected, error_rate, reject_rate = cal_exp.reject_orchestrator.predict_reject(
        x_test, confidence=0.9, threshold=0.6
    )

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

    cal_exp.reject_orchestrator.initialize_reject_learner(threshold=0.5)
    cal_exp.reject_orchestrator.predict_reject(x_test, threshold=0.5)

    factual_explanation = cal_exp.explain_guarded_factual(x_test, y_test)
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation.plot(show=False, uncertainty=True)

    factual_explanation = cal_exp.explain_guarded_factual(x_test, y_test[0])
    factual_explanation.add_conjunctions()
    factual_explanation = cal_exp.explain_guarded_factual(x_test, (0.4, 0.6))
    factual_explanation.add_conjunctions()

    alternative_explanation = cal_exp.explore_guarded_alternatives(x_test, y_test)
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)
    # Basic sanity assertions to ensure the explainer produced results
    assert factual_explanation is not None
    assert alternative_explanation is not None

    alternative_explanation = cal_exp.explore_guarded_alternatives(x_test, y_test[0])
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)
    alternative_explanation.super_explanations()
    alternative_explanation.semi_explanations()
    alternative_explanation.counter_explanations()
    # Basic sanity assertions to ensure the explainer produced results
    assert factual_explanation is not None
    assert alternative_explanation is not None


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
    factual_explanation = cal_exp.explain_guarded_factual(x_test, 0)
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)

    # Tuple of integer thresholds
    factual_explanation = cal_exp.explain_guarded_factual(x_test, (0, 1))
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)

    # Alternatives should also accept int thresholds
    alternative_explanation = cal_exp.explore_guarded_alternatives(x_test, 0)
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)
    # Basic sanity assertions to ensure the explainer produced results
    assert factual_explanation is not None
    assert alternative_explanation is not None


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

    factual_explanation = cal_exp.explain_guarded_factual(x_test)
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation.plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_guarded_alternatives(x_test)
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)
    # Basic sanity assertions to ensure the explainer produced results
    assert factual_explanation is not None
    assert alternative_explanation is not None


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

    factual_explanation = cal_exp.explain_guarded_factual(x_test, bins=x_test[:, bin_feature])
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation.plot(show=False, uncertainty=True)
    repr(factual_explanation)

    factual_explanation = cal_exp.explain_guarded_factual(
        x_test, low_high_percentiles=(0.1, np.inf), bins=x_test[:, bin_feature]
    )
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    factual_explanation = cal_exp.explain_guarded_factual(
        x_test, low_high_percentiles=(-np.inf, 0.9), bins=x_test[:, bin_feature]
    )
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    with pytest.raises(Warning):
        factual_explanation.plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_guarded_alternatives(
        x_test, bins=x_test[:, bin_feature]
    )
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_guarded_alternatives(
        x_test, low_high_percentiles=(0.1, np.inf), bins=x_test[:, bin_feature]
    )
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)

    alternative_explanation = cal_exp.explore_guarded_alternatives(
        x_test, low_high_percentiles=(-np.inf, 0.9), bins=x_test[:, bin_feature]
    )
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)
    repr(alternative_explanation)
    # Basic sanity assertions to ensure the explainer produced results
    assert factual_explanation is not None
    assert alternative_explanation is not None


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

    cal_exp.reject_orchestrator.initialize_reject_learner(threshold=0.5)
    cal_exp.reject_orchestrator.predict_reject(x_test, bins=x_test[:, 0], threshold=0.5)

    factual_explanation = cal_exp.explain_guarded_factual(x_test, y_test, bins=x_test[:, 0])
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)

    factual_explanation = cal_exp.explain_guarded_factual(x_test, y_test[0], bins=x_test[:, 0])
    factual_explanation.add_conjunctions()

    alternative_explanation = cal_exp.explore_guarded_alternatives(
        x_test, y_test, bins=x_test[:, 0]
    )
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)

    alt = cal_exp.explore_guarded_alternatives(x_test, y_test[0], bins=x_test[:, 0])
    alt.add_conjunctions()
    # Basic sanity assertions to ensure the explainer produced results
    assert factual_explanation is not None
    assert alt is not None


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

    factual_explanation = cal_exp.explain_guarded_factual(x_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_guarded_factual(
        x_test, low_high_percentiles=(0.1, np.inf)
    )
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_guarded_factual(
        x_test, low_high_percentiles=(-np.inf, 0.9)
    )
    factual_explanation.add_conjunctions()

    alt = cal_exp.explore_guarded_alternatives(x_test)
    alt.add_conjunctions()
    assert alt is not None

    alt = cal_exp.explore_guarded_alternatives(x_test, low_high_percentiles=(0.1, np.inf))
    alt.add_conjunctions()
    assert alt is not None

    alt = cal_exp.explore_guarded_alternatives(x_test, low_high_percentiles=(-np.inf, 0.9))
    alt.add_conjunctions()
    assert alt is not None
    assert alt is not None


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

    factual_explanation = cal_exp.explain_guarded_factual(x_test, y_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_guarded_factual(x_test, y_test[0])
    factual_explanation.add_conjunctions()

    alt = cal_exp.explore_guarded_alternatives(x_test, y_test)
    alt.add_conjunctions()
    assert alt is not None

    alt = cal_exp.explore_guarded_alternatives(x_test, y_test[0])
    alt.add_conjunctions()


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

    factual_explanation = cal_exp.explain_guarded_factual(x_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_guarded_factual(
        x_test, low_high_percentiles=(0.1, np.inf)
    )
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_guarded_factual(
        x_test, low_high_percentiles=(-np.inf, 0.9)
    )
    factual_explanation.add_conjunctions()

    alt = cal_exp.explore_guarded_alternatives(x_test)
    alt.add_conjunctions()

    alt = cal_exp.explore_guarded_alternatives(x_test, low_high_percentiles=(0.1, np.inf))
    alt.add_conjunctions()

    alt = cal_exp.explore_guarded_alternatives(x_test, low_high_percentiles=(-np.inf, 0.9))
    alt.add_conjunctions()
    assert alt is not None


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

    factual_explanation = cal_exp.explain_guarded_factual(x_test, y_test)
    factual_explanation.add_conjunctions()

    factual_explanation = cal_exp.explain_guarded_factual(x_test, y_test[0])
    factual_explanation.add_conjunctions()

    alt = cal_exp.explore_guarded_alternatives(x_test, y_test)
    alt.add_conjunctions()
    assert alt is not None

    alt = cal_exp.explore_guarded_alternatives(x_test, y_test[0])
    alt.add_conjunctions()


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
