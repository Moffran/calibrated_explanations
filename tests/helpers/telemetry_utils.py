import pytest


def extract_percentile_values(raw_percentiles):
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


def assert_uncertainty_schema(
    payload, representation: str, expect_percentiles: bool, threshold_value=None
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
        low, high = extract_percentile_values(percentiles)
        for value in (low, high):
            assert value is None or isinstance(value, (int, float))
        confidence = payload.get("confidence_level")
        assert confidence is None or isinstance(confidence, (int, float))
    else:
        if percentiles is not None:
            # If present, must be empty or None-like depending on implementation
            pass
