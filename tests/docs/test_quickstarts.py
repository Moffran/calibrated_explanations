from __future__ import annotations

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer


def test_classification_quickstart() -> None:
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=0
    )

    explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal, feature_names=dataset.feature_names)

    batch = explainer.explain_factual(X_test[:5])
    assert len(batch) == 5
    telemetry = getattr(batch, "telemetry", {})
    assert "interval_source" in telemetry
    assert telemetry.get("mode") == "factual"


def test_regression_quickstart() -> None:
    dataset = load_diabetes()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=0
    )

    explainer = WrapCalibratedExplainer(RandomForestRegressor(random_state=0))
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(
        X_cal,
        y_cal,
        feature_names=dataset.feature_names,
    )

    batch = explainer.explore_alternatives(X_test[:3], threshold=2.5)
    assert len(batch) == 3
    telemetry = getattr(batch, "telemetry", {})
    assert "proba_source" in telemetry
    assert telemetry.get("task") == "regression"
