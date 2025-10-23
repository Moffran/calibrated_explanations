import pytest

from calibrated_explanations.viz import (
    build_regression_bars_spec,
    plotspec_to_dict,
    plotspec_from_dict,
    validate_plotspec,
)


def test_plotspec_round_trip_minimal():
    spec = build_regression_bars_spec(
        title="roundtrip",
        predict={"predict": 0.5, "low": 0.2, "high": 0.8},
        feature_weights={
            "predict": [0.1, 0.2],
            "low": [0.05, 0.15],
            "high": [0.18, 0.25],
        },
        features_to_plot=[0, 1],
        column_names=["a", "b"],
        instance=None,
        y_minmax=(0.0, 1.0),
        interval=True,
    )
    d = plotspec_to_dict(spec)
    # Basic structural checks
    assert isinstance(d, dict)
    assert "body" in d and "bars" in d["body"]
    # Round-trip
    spec2 = plotspec_from_dict(d)
    assert spec2.title == spec.title
    assert len(spec2.body.bars) == len(spec.body.bars)


def test_validate_plotspec_missing_body_raises():
    bad = {"plotspec_version": "1.0.0", "title": "no body"}
    with pytest.raises(ValueError):
        validate_plotspec(bad)


def test_validate_plotspec_requires_version():
    bad = {"title": "missing version", "body": {"bars": []}}
    with pytest.raises(ValueError):
        validate_plotspec(bad)


def test_validate_plotspec_rejects_bar_without_value():
    bad = {
        "plotspec_version": "1.0.0",
        "body": {"bars": [{"label": "f0"}]},
    }
    with pytest.raises(ValueError):
        validate_plotspec(bad)


def test_validate_plotspec_rejects_incomplete_bars():
    missing_value = {"plotspec_version": "1.0.0", "body": {"bars": [{"label": "a"}]}}
    with pytest.raises(ValueError):
        validate_plotspec(missing_value)

    missing_label = {"plotspec_version": "1.0.0", "body": {"bars": [{"value": 0.2}]}}
    with pytest.raises(ValueError):
        validate_plotspec(missing_label)


def test_validate_plotspec_requires_bar_label_and_value():
    bad = {
        "plotspec_version": "1.0.0",
        "body": {"bars": [{"label": "f0"}]},
    }

    with pytest.raises(ValueError):
        validate_plotspec(bad)
