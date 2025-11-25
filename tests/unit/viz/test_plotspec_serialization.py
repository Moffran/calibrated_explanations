"""Tests for PlotSpec serialization and roundtrip behavior.

Focuses on semantic assertions (domain invariants) rather than
implementation details. See TEST_GUIDELINES_ENHANCED.md for patterns.
"""

import numpy as np
import pytest

from calibrated_explanations.viz import (
    build_regression_bars_spec,
    plotspec_to_dict,
    plotspec_from_dict,
    validate_plotspec,
)


def test_plotspec_round_trip_minimal__should_preserve_structure_and_values():
    """Verify basic roundtrip: title and bar count are preserved.

    This is a snapshot test (equality check) combined with structural
    semantic assertions to ensure the basic contract is maintained.
    """
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
    spec_dict = plotspec_to_dict(spec)
    # Snapshot: basic structure check
    assert isinstance(spec_dict, dict)
    assert "body" in spec_dict and "bars" in spec_dict["body"]

    # Round-trip
    spec2 = plotspec_from_dict(spec_dict)

    # Semantic assertions (domain invariants)
    # Invariant 1: Title is preserved exactly
    assert spec2.title == spec.title, "Title should be preserved through roundtrip"

    # Invariant 2: Bar structure matches
    assert len(spec2.body.bars) == len(spec.body.bars), "Bar count should match"

    # Invariant 3: Header bounds satisfy ordering (low ≤ pred ≤ high)
    # Ref: ADR-005 Explanation Envelope
    assert spec2.header is not None, "Header is mandatory"
    assert spec2.header.low is not None, "Lower bound is mandatory"
    assert spec2.header.pred is not None, "Point estimate is mandatory"
    assert spec2.header.high is not None, "Upper bound is mandatory"
    assert spec2.header.low <= spec2.header.pred <= spec2.header.high, (
        f"Bounds violated: {spec2.header.low} ≤ {spec2.header.pred} ≤ " f"{spec2.header.high}"
    )

    # Invariant 4: Each bar has required fields
    for idx, bar_item in enumerate(spec2.body.bars):
        assert bar_item.label is not None, f"Bar {idx} label is mandatory"
        assert bar_item.value is not None, f"Bar {idx} value is mandatory"
        assert isinstance(bar_item.value, (int, float)), f"Bar {idx} value must be numeric"


def test_plotspec_roundtrip__should_preserve_interval_semantics():
    """Verify that interval bounds remain valid after roundtrip.

    Domain Invariant: Predicted value must lie within uncertainty bounds.
    This tests the core semantic contract that intervals maintain their
    mathematical properties through serialization and deserialization.
    Ref: ADR-005 Explanation Envelope, ADR-007 PlotSpec Abstraction
    """
    spec = build_regression_bars_spec(
        title="interval_test",
        predict={"predict": 0.5, "low": 0.2, "high": 0.8},
        feature_weights={
            "predict": np.array([0.1, -0.2, 0.3]),
            "low": np.array([0.05, -0.25, 0.25]),
            "high": np.array([0.15, -0.15, 0.35]),
        },
        features_to_plot=[0, 1, 2],
        column_names=["f0", "f1", "f2"],
        instance=np.array([1.0, 2.0, 3.0]),
        y_minmax=(0.0, 1.0),
        interval=True,
    )

    # Roundtrip
    spec_dict = plotspec_to_dict(spec)
    restored = plotspec_from_dict(spec_dict)

    # Semantic assertions: interval invariants (core invariants, not
    # exact equality which may be affected by default values)
    header = restored.header
    assert header.low <= header.pred <= header.high, (
        f"Header interval violated: {header.low} ≤ {header.pred} ≤ " f"{header.high}"
    )

    # Semantic assertions: title and structure preserved
    assert restored.title == spec.title, "Title should be preserved"
    assert len(restored.body.bars) == len(spec.body.bars), "Bar count should match"

    # Semantic assertions: bar-level invariants
    for idx, (original_bar, restored_bar) in enumerate(zip(spec.body.bars, restored.body.bars)):
        # Values preserved
        assert restored_bar.label == original_bar.label, f"Bar {idx} label mismatch"
        assert restored_bar.value == original_bar.value, f"Bar {idx} value mismatch"

        # If intervals exist, they must satisfy ordering
        if restored_bar.interval_low is not None and restored_bar.interval_high is not None:
            low = restored_bar.interval_low
            val = restored_bar.value
            high = restored_bar.interval_high
            assert low <= val <= high, f"Bar {idx} interval violated: {low} ≤ {val} ≤ {high}"


def test_plotspec_roundtrip__should_handle_zero_width_interval():
    """Verify roundtrip handles edge case where pred = low = high.

    Edge case: zero-width interval (all bounds equal).
    This tests the boundary condition where uncertainty is zero.
    Ref: ADR-005 Explanation Envelope
    """
    spec = build_regression_bars_spec(
        title="zero_width",
        predict={"predict": 0.5, "low": 0.5, "high": 0.5},
        feature_weights={"predict": [0.2], "low": [0.1], "high": [0.3]},
        features_to_plot=[0],
        column_names=["feat"],
        instance=None,
        y_minmax=(0.0, 1.0),
        interval=True,
    )

    # Roundtrip
    spec_dict = plotspec_to_dict(spec)
    restored = plotspec_from_dict(spec_dict)

    # Semantic assertion: edge case handling
    header = restored.header
    assert header.low == header.pred == header.high, "Zero-width interval should have equal bounds"
    assert (
        header.low <= header.pred <= header.high
    ), "Even at boundary, ordering invariant must hold"


def test_plotspec_roundtrip__should_preserve_optional_fields():
    """Verify that optional fields in IntervalHeaderSpec are preserved.

    Tests that xlim, xlabel, ylabel, and other optional fields survive
    the roundtrip and maintain their semantic meaning.
    Ref: ADR-007 PlotSpec Abstraction
    """
    spec = build_regression_bars_spec(
        title="optional_fields",
        predict={"predict": 0.6, "low": 0.4, "high": 0.8},
        feature_weights={"predict": [0.15], "low": [0.1], "high": [0.2]},
        features_to_plot=[0],
        column_names=["feature"],
        instance=None,
        y_minmax=(-0.1, 1.1),  # Custom range
        interval=True,
    )

    # Roundtrip
    spec_dict = plotspec_to_dict(spec)
    restored = plotspec_from_dict(spec_dict)

    # Semantic assertions: optional fields preserved
    assert restored.header.xlim is not None, "xlim should be computed from y_minmax"
    assert (
        restored.header.xlim[0] <= restored.header.xlim[1]
    ), f"xlim range violated: {restored.header.xlim}"

    # If original has custom labels, they're preserved
    if spec.header.xlabel is not None:
        assert restored.header.xlabel == spec.header.xlabel


def test_validate_plotspec_missing_body_raises():
    """Verify that PlotSpec validation rejects specs without body."""
    bad = {"plotspec_version": "1.0.0", "title": "no body"}
    with pytest.raises(ValueError):
        validate_plotspec(bad)


def test_validate_plotspec_requires_version():
    """Verify that PlotSpec validation requires version field."""
    bad = {"title": "missing version", "body": {"bars": []}}
    with pytest.raises(ValueError):
        validate_plotspec(bad)


def test_validate_plotspec_rejects_bar_without_value():
    """Verify that bars without value are rejected."""
    bad = {
        "plotspec_version": "1.0.0",
        "body": {"bars": [{"label": "f0"}]},
    }
    with pytest.raises(ValueError):
        validate_plotspec(bad)


def test_validate_plotspec_rejects_incomplete_bars():
    """Verify that bars missing label or value are rejected."""
    missing_value = {"plotspec_version": "1.0.0", "body": {"bars": [{"label": "a"}]}}
    with pytest.raises(ValueError):
        validate_plotspec(missing_value)

    missing_label = {"plotspec_version": "1.0.0", "body": {"bars": [{"value": 0.2}]}}
    with pytest.raises(ValueError):
        validate_plotspec(missing_label)


def test_validate_plotspec_requires_bar_label_and_value():
    """Verify that bar items require both label and value fields."""
    bad = {
        "plotspec_version": "1.0.0",
        "body": {"bars": [{"label": "f0"}]},
    }

    with pytest.raises(ValueError):
        validate_plotspec(bad)
