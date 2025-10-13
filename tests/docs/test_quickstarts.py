from __future__ import annotations

import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer


def _extract_threshold_value(threshold):
    if threshold is None:
        return None
    if isinstance(threshold, dict):
        for key in ("value", "threshold", "amount"):
            value = threshold.get(key)
            if isinstance(value, (int, float)):
                return value
        return None
    if isinstance(threshold, (list, tuple)):
        for item in threshold:
            value = _extract_threshold_value(item)
            if value is not None:
                return value
        return None
    if isinstance(threshold, (int, float)):
        return threshold
    return None


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

    first_instance = batch[0]
    prediction = first_instance.prediction
    uncertainty = telemetry.get("uncertainty", {})
    assert uncertainty.get("representation") == "venn_abers"
    assert _extract_threshold_value(uncertainty.get("threshold")) is None
    raw_percentiles = uncertainty.get("raw_percentiles")
    if isinstance(raw_percentiles, dict):
        assert all(value is None for value in raw_percentiles.values())
    else:
        assert raw_percentiles is None
    assert uncertainty.get("confidence_level") is None
    assert uncertainty.get("calibrated_value") == pytest.approx(prediction["predict"])
    assert uncertainty.get("lower_bound") == pytest.approx(prediction["low"])
    assert uncertainty.get("upper_bound") == pytest.approx(prediction["high"])
    legacy_interval = uncertainty.get("legacy_interval")
    assert isinstance(legacy_interval, (list, tuple))
    assert len(legacy_interval) == 2
    assert legacy_interval[0] == pytest.approx(uncertainty["lower_bound"])
    assert legacy_interval[1] == pytest.approx(uncertainty["upper_bound"])

    rules = telemetry.get("rules")
    expected_rules = first_instance.build_rules_payload()
    assert rules == expected_rules
    first_rule = rules[0]
    assert first_rule["prediction"]["representation"] == "venn_abers"
    assert _extract_threshold_value(first_rule["prediction"].get("threshold")) is None


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

    first_instance = batch[0]
    prediction = first_instance.prediction
    uncertainty = telemetry.get("uncertainty", {})
    assert uncertainty.get("representation") == "threshold"
    threshold_metadata = _extract_threshold_value(uncertainty.get("threshold"))
    assert threshold_metadata == pytest.approx(2.5)
    raw_percentiles = uncertainty.get("raw_percentiles")
    if isinstance(raw_percentiles, dict):
        assert raw_percentiles.get("low") in (None,)
        assert raw_percentiles.get("high") in (None,)
    else:
        assert raw_percentiles in (None, [None, None])
    assert uncertainty.get("confidence_level") is None
    assert uncertainty.get("calibrated_value") == pytest.approx(prediction["predict"])
    assert uncertainty.get("lower_bound") == pytest.approx(prediction["low"])
    assert uncertainty.get("upper_bound") == pytest.approx(prediction["high"])
    legacy_interval = uncertainty.get("legacy_interval")
    assert isinstance(legacy_interval, (list, tuple))
    assert len(legacy_interval) == 2
    assert legacy_interval[0] == pytest.approx(uncertainty["lower_bound"])
    assert legacy_interval[1] == pytest.approx(uncertainty["upper_bound"])

    rules = telemetry.get("rules")
    expected_rules = first_instance.build_rules_payload()
    assert rules == expected_rules
    first_rule = rules[0]
    assert first_rule["uncertainty"]["representation"] == "threshold"
    assert _extract_threshold_value(first_rule.get("threshold")) == pytest.approx(2.5)
