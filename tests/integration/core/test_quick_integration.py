"""Integration test for quick_explain API."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations.api.quick import quick_explain
from calibrated_explanations.api.config import ExplainerBuilder
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer


def test_quick_explain_binary_classification():
    """Test quick_explain with binary classification."""
    # Generate simple binary classification data
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split into train, cal, test
    X_train = X[:60]
    y_train = y[:60]
    X_cal = X[60:80]
    y_cal = y[60:80]
    X_test = X[80:]
    y_test = y[80:]

    # Use RandomForest
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    # Call quick_explain
    result = quick_explain(
        model=model,
        x_train=X_train,
        y_train=y_train,
        x_cal=X_cal,
        y_cal=y_cal,
        x=X_test,
        task="classification",
    )

    # Check that result is returned
    assert result is not None
    # Check that it has some attributes
    assert hasattr(result, "explanations")


def test_quick_explain_regression():
    """Test quick_explain with regression."""
    # Generate simple regression data
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1

    # Split into train, cal, test
    X_train = X[:60]
    y_train = y[:60]
    X_cal = X[60:80]
    y_cal = y[60:80]
    X_test = X[80:]
    y_test = y[80:]

    # Use RandomForest
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=10, random_state=42)

    # Call quick_explain
    result = quick_explain(
        model=model,
        x_train=X_train,
        y_train=y_train,
        x_cal=X_cal,
        y_cal=y_cal,
        x=X_test,
        task="regression",
    )

    # Check that result is returned
    assert result is not None
    # Check that it has some attributes
    assert hasattr(result, "explanations")


def test_explainer_builder_integration():
    """Integration test for ExplainerBuilder with WrapCalibratedExplainer."""
    # Generate simple binary classification data
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split into train, cal, test
    X_train = X[:60]
    y_train = y[:60]
    X_cal = X[60:80]
    y_cal = y[60:80]
    X_test = X[80:]

    # Use RandomForest
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    # Use ExplainerBuilder
    builder = ExplainerBuilder(model)
    builder = builder.task("classification")
    builder = builder.low_high_percentiles((10, 90))
    # threshold only for regression
    cfg = builder.build_config()

    # Create WrapCalibratedExplainer from config
    w = WrapCalibratedExplainer.from_config(cfg)
    w.fit(X_train, y_train)
    w.calibrate(X_cal, y_cal)

    # Explain
    result = w.explain_factual(X_test)

    # Check
    assert result is not None
    assert hasattr(result, "explanations")


def test_explainer_builder_regression():
    """Integration test for ExplainerBuilder with regression."""
    # Generate simple regression data
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1

    # Split into train, cal, test
    X_train = X[:60]
    y_train = y[:60]
    X_cal = X[60:80]
    y_cal = y[60:80]
    X_test = X[80:]

    # Use RandomForest
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=10, random_state=42)

    # Use ExplainerBuilder
    builder = ExplainerBuilder(model)
    builder = builder.task("regression")
    builder = builder.threshold(0.5)
    cfg = builder.build_config()

    # Create WrapCalibratedExplainer from config
    w = WrapCalibratedExplainer.from_config(cfg)
    w.fit(X_train, y_train)
    w.calibrate(X_cal, y_cal)

    # Explain
    result = w.explain_factual(X_test)

    # Check
    assert result is not None
    assert hasattr(result, "explanations")
