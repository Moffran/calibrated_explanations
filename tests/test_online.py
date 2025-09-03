"""Tests for OnlineCalibratedExplainer functionality."""

# pylint: disable=invalid-name, too-many-locals
import numpy as np
import pytest
from calibrated_explanations import OnlineCalibratedExplainer
from calibrated_explanations.core.exceptions import ModelNotSupportedError, NotFittedError
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler


def test_online_classification():
    """Test OnlineCalibratedExplainer with classification."""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )

    # Split into initial training, calibration and test sets
    X_train = X[:400]
    y_train = y[:400]
    X_cal = X[400:600]
    y_cal = y[400:600]
    X_test = X[600:800]
    X_stream = X[800:]
    y_stream = y[800:]

    # Create and initialize online learner
    sgd = SGDClassifier(loss="log_loss", random_state=42)
    explainer = OnlineCalibratedExplainer(sgd)

    # Initial fit and calibration
    explainer.fit(X_train, y_train)
    explainer.calibrate(X_cal, y_cal)

    # Get initial predictions
    initial_preds = explainer.predict(X_test)
    initial_probs = explainer.predict_proba(X_test)

    # Update with streaming data
    for i in range(0, len(X_stream), 10):
        batch_X = X_stream[i : i + 10]
        batch_y = y_stream[i : i + 10]
        explainer.partial_fit(batch_X, batch_y)
        explainer.calibrate_many(batch_X, batch_y)

    # Get updated predictions
    updated_preds = explainer.predict(X_test)
    updated_probs = explainer.predict_proba(X_test)

    # Test single instance prediction
    single_pred = explainer.predict_one(X_test[0])
    single_prob = explainer.predict_proba_one(X_test[0])

    # Verify outputs
    assert initial_preds.shape == updated_preds.shape
    assert initial_probs.shape == updated_probs.shape
    assert isinstance(single_pred, (np.ndarray, list))
    assert isinstance(single_prob, (np.ndarray, list))
    assert len(single_pred.shape) == 1
    assert len(single_prob.shape) <= 2


def test_online_regression():
    """Test OnlineCalibratedExplainer with regression."""
    # Generate synthetic data
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, random_state=42)

    # Scale targets to reasonable range
    scaler = StandardScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    # Split into initial training, calibration and test sets
    X_train = X[:400]
    y_train = y[:400]
    X_cal = X[400:600]
    y_cal = y[400:600]
    X_test = X[600:800]
    X_stream = X[800:]
    y_stream = y[800:]

    # Create and initialize online learner
    sgd = SGDRegressor(random_state=42)
    explainer = OnlineCalibratedExplainer(sgd)

    # Initial fit and calibration
    explainer.fit(X_train, y_train)
    explainer.calibrate(X_cal, y_cal)

    # Get initial predictions
    initial_preds = explainer.predict(X_test)
    initial_probs = explainer.predict_proba(X_test, threshold=0.0)  # Test probabilistic regression

    # Update with streaming data
    for i in range(0, len(X_stream), 10):
        batch_X = X_stream[i : i + 10]
        batch_y = y_stream[i : i + 10]
        explainer.partial_fit(batch_X, batch_y)
        explainer.calibrate_many(batch_X, batch_y)

    # Get updated predictions
    updated_preds = explainer.predict(X_test)
    updated_probs = explainer.predict_proba(X_test, threshold=0.0)

    # Test single instance prediction
    single_pred = explainer.predict_one(X_test[0])
    single_prob = explainer.predict_proba_one(X_test[0], threshold=0.0)

    # Test prediction intervals
    preds_with_intervals = explainer.predict(X_test, uq_interval=True)

    # Verify outputs
    assert initial_preds.shape == updated_preds.shape
    assert initial_probs.shape == updated_probs.shape
    assert isinstance(single_pred, (np.ndarray, list))
    assert isinstance(single_prob, (np.ndarray, list))
    assert len(single_pred.shape) == 1
    assert len(single_prob.shape) <= 2
    assert len(preds_with_intervals) == 2
    assert len(preds_with_intervals[1]) == 2  # (low, high) intervals


def test_online_explanations():
    """Test explanation generation with OnlineCalibratedExplainer."""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )

    # Split data
    X_train = X[:400]
    y_train = y[:400]
    X_cal = X[400:600]
    y_cal = y[400:600]
    X_test = X[600:605]  # Small test set for explanations
    X_stream = X[800:]
    y_stream = y[800:]

    # Create and initialize explainer
    sgd = SGDClassifier(loss="log_loss", random_state=42)
    explainer = OnlineCalibratedExplainer(sgd)

    # Initial fit and calibration
    explainer.fit(X_train, y_train)
    explainer.calibrate(X_cal, y_cal, feature_names=[f"f{i}" for i in range(20)])

    # Generate initial explanations
    initial_factual = explainer.explain_factual(X_test)
    initial_alternatives = explainer.explore_alternatives(X_test)

    # Update with streaming data
    for i in range(0, len(X_stream), 10):
        batch_X = X_stream[i : i + 10]
        batch_y = y_stream[i : i + 10]
        explainer.partial_fit(batch_X, batch_y)
        explainer.calibrate_many(batch_X, batch_y)

    # Generate updated explanations
    updated_factual = explainer.explain_factual(X_test)
    updated_alternatives = explainer.explore_alternatives(X_test)

    # Verify explanation outputs
    assert len(initial_factual.explanations) == len(X_test)
    assert len(updated_factual.explanations) == len(X_test)
    assert len(initial_alternatives.explanations) == len(X_test)
    assert len(updated_alternatives.explanations) == len(X_test)

    # Verify explanation attributes
    for exp in updated_factual.explanations:
        assert hasattr(exp, "feature_weights")
        assert hasattr(exp, "prediction")
        assert hasattr(exp, "X_test")

    for exp in updated_alternatives.explanations:
        assert hasattr(exp, "feature_weights")
        assert hasattr(exp, "prediction")
        assert hasattr(exp, "X_test")


def test_error_handling():
    """Test error handling in OnlineCalibratedExplainer."""

    # Create explainer with non-online learner
    class NonOnlineLearner:
        """Error class"""

        def fit(self, X, y):
            """Fit function"""
            pass  # pylint: disable=unnecessary-pass

        def predict(self, X):
            """Predict function"""
            return np.zeros(len(X))

    explainer = OnlineCalibratedExplainer(NonOnlineLearner())

    # Test partial_fit with non-online learner
    rng = np.random.default_rng()
    X = rng.standard_normal((10, 5))
    y = rng.integers(0, 2, 10)

    with pytest.raises(ModelNotSupportedError):
        explainer.partial_fit(X, y)

    # Test calibration before fitting
    with pytest.raises(NotFittedError):
        explainer.calibrate_one(X[0], y[0])

    # Test prediction before fitting/calibration
    with pytest.raises(NotFittedError):
        explainer.predict_one(X[0])
