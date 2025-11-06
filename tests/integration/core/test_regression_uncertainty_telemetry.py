"""Regression telemetry schema regression tests.

This suite exercises calibrated regression explanations across factual,
alternative, percentile, and thresholded modes to ensure the telemetry payload
emitted by :class:`CalibratedExplanation` matches the ADR-022 interval schema.
"""

from __future__ import annotations

import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer


def _train_regression_explainer():
    dataset = load_diabetes()
    x, y = dataset.data, dataset.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_proper, x_cal, y_proper, y_cal = train_test_split(
        x_train, y_train, test_size=0.25, random_state=0
    )

    explainer = WrapCalibratedExplainer(RandomForestRegressor(random_state=0))
    explainer.fit(x_proper, y_proper)
    explainer.calibrate(x_cal, y_cal, feature_names=dataset.feature_names)
    return explainer, x_test


def _train_classification_explainer():
    dataset = load_breast_cancer()
    x, y = dataset.data, dataset.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=0
    )
    x_proper, x_cal, y_proper, y_cal = train_test_split(
        x_train, y_train, test_size=0.25, stratify=y_train, random_state=0
    )

    explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
    explainer.fit(x_proper, y_proper)
    explainer.calibrate(x_cal, y_cal, feature_names=dataset.feature_names)
    return explainer, x_test


def _extract_percentile_values(raw_percentiles):
    """Return normalized low/high percentile values from telemetry payloads."""
    if raw_percentiles is None:
        return None, None
    if isinstance(raw_percentiles, dict):
        low = raw_percentiles.get("low")
        high = raw_percentiles.get("high")
        if isinstance(low, dict):
            low = low.get("value", low.get("percentile", low.get("bound")))
        if isinstance(high, dict):
            high = high.get("value", high.get("percentile", high.get("bound")))
    else:
        assert isinstance(raw_percentiles, (list, tuple))
        assert len(raw_percentiles) == 2
        low, high = raw_percentiles
    return low, high


def _extract_threshold_value(threshold):
    """Return a scalar threshold from telemetry metadata."""
    if threshold is None:
        return None
    if isinstance(threshold, dict):
        for candidate_key in ("value", "threshold", "amount"):
            candidate = threshold.get(candidate_key)
            if isinstance(candidate, (int, float)):
                return candidate
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


def _assert_uncertainty_schema(
    payload, representation: str, expect_percentiles: bool, threshold_value
):
    expected_keys = {
        "representation",
        "calibrated_value",
        "lower_bound",
        "upper_bound",
        "legacy_interval",
        "threshold",
        "raw_percentiles",
        "confidence_level",
    }
    assert expected_keys.issubset(payload.keys())
    assert payload["representation"] == representation
    assert isinstance(payload["calibrated_value"], (int, float))
    assert isinstance(payload["lower_bound"], (int, float))
    assert isinstance(payload["upper_bound"], (int, float))
    legacy_interval = payload["legacy_interval"]
    assert isinstance(legacy_interval, (list, tuple))
    assert len(legacy_interval) == 2
    assert legacy_interval[0] == pytest.approx(payload["lower_bound"])
    assert legacy_interval[1] == pytest.approx(payload["upper_bound"])

    percentiles = payload.get("raw_percentiles")
    if expect_percentiles:
        assert percentiles is not None
        low, high = _extract_percentile_values(percentiles)
        for value in (low, high):
            assert value is None or isinstance(value, (int, float))
        confidence = payload.get("confidence_level")
        assert confidence is None or isinstance(confidence, (int, float))
    else:
        if percentiles is not None:
            low, high = _extract_percentile_values(percentiles)
            assert low is None and high is None
        assert payload.get("confidence_level") in (None,)

    extracted_threshold = _extract_threshold_value(payload.get("threshold"))
    if threshold_value is None:
        assert extracted_threshold is None
    else:
        assert extracted_threshold == pytest.approx(threshold_value)


def test_regression_batches_publish_full_uncertainty_schema():
    """Telemetry for regression explanations must follow ADR-022."""

    explainer, x_test = _train_regression_explainer()
    sample = x_test[:1]

    factual = explainer.explain_factual(sample)
    alternative = explainer.explore_alternatives(sample)
    thresholded = explainer.explain_factual(sample, threshold=2.5)
    thresholded_alt = explainer.explore_alternatives(sample, threshold=2.5)

    expectations = (
        (factual, "percentile", True, None),
        (alternative, "percentile", True, None),
        (thresholded, "threshold", False, 2.5),
        (thresholded_alt, "threshold", False, 2.5),
    )

    for batch, representation, expect_percentiles, threshold_value in expectations:
        telemetry = getattr(batch, "telemetry", {})
        assert "uncertainty" in telemetry
        _assert_uncertainty_schema(
            telemetry["uncertainty"], representation, expect_percentiles, threshold_value
        )

        telemetry_rules = telemetry.get("rules")
        expected_payload = batch[0].build_rules_payload()
        assert telemetry_rules == expected_payload

        metadata_rules = telemetry_rules["metadata"].get("feature_rules", [])
        assert metadata_rules
        first_metadata_rule = metadata_rules[0]
        _assert_uncertainty_schema(
            first_metadata_rule["prediction_uncertainty"],
            representation,
            expect_percentiles,
            threshold_value,
        )
        if telemetry.get("mode") != "factual" and threshold_value is not None:
            assert _extract_threshold_value(first_metadata_rule.get("threshold")) == pytest.approx(
                threshold_value
            )


def test_build_rules_payload_covers_probabilistic_and_thresholded_alternatives():
    """Rule payloads must include uncertainty metadata for all alternatives."""

    class_explainer, class_test = _train_classification_explainer()
    class_batch = class_explainer.explore_alternatives(class_test[:1])
    class_payload = class_batch[0].build_rules_payload()
    class_metadata_rule = class_payload["metadata"]["feature_rules"][0]
    _assert_uncertainty_schema(
        class_metadata_rule["prediction_uncertainty"], "venn_abers", False, None
    )
    _assert_uncertainty_schema(class_metadata_rule["weight_uncertainty"], "venn_abers", False, None)

    reg_explainer, reg_test = _train_regression_explainer()
    reg_batch = reg_explainer.explore_alternatives(reg_test[:1], threshold=2.5)
    reg_payload = reg_batch[0].build_rules_payload()
    reg_metadata_rule = reg_payload["metadata"]["feature_rules"][0]
    _assert_uncertainty_schema(reg_metadata_rule["prediction_uncertainty"], "threshold", False, 2.5)
    assert _extract_threshold_value(reg_metadata_rule.get("threshold")) == pytest.approx(2.5)
    # Feature-level weights retain probabilistic uncertainty blocks for thresholds.
    _assert_uncertainty_schema(reg_metadata_rule["weight_uncertainty"], "venn_abers", False, None)
