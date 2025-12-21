"""Tests for PlotSpec serializers: to_dict and from_dict.

Focuses on semantic assertions (domain invariants) and roundtrip
verification. See .github/tests-guidance.md for patterns.
"""

import pytest

from calibrated_explanations.viz import (
    plotspec_to_dict,
    plotspec_from_dict,
    validate_plotspec,
)
from calibrated_explanations.viz import (
    PlotSpec,
    IntervalHeaderSpec,
    BarHPanelSpec,
    BarItem,
)


def test_plotspec_roundtrip_and_validate__should_preserve_semantics():
    """Verify roundtrip with semantic invariant checks.

    Domain Invariant: Interval bounds satisfy low ≤ pred ≤ high.
    Ref: ADR-005 Explanation Envelope
    """
    header = IntervalHeaderSpec(pred=0.3, low=0.1, high=0.9)
    bars = [BarItem(label="a", value=0.1), BarItem(label="b", value=0.2)]
    body = BarHPanelSpec(bars=bars)
    spec = PlotSpec(title="t", header=header, body=body)

    spec_dict = plotspec_to_dict(spec)
    assert spec_dict["plotspec_version"]

    # Roundtrip
    restored = plotspec_from_dict(spec_dict)

    # Snapshot equality
    assert restored == spec, "Roundtrip should preserve exact equality"

    # Semantic assertions: interval invariant
    assert restored.header.low <= restored.header.pred <= restored.header.high, (
        f"Interval violated: {restored.header.low} ≤ "
        f"{restored.header.pred} ≤ {restored.header.high}"
    )

    # Semantic assertions: bar integrity
    assert restored.title == "t", "Title should be preserved"
    assert len(restored.body.bars) == 2, "Bar count should match"
    for idx, bar_item in enumerate(restored.body.bars):
        assert bar_item.label is not None, f"Bar {idx} label is mandatory"
        assert bar_item.value is not None, f"Bar {idx} value is mandatory"


def test_validate_rejects_bad_payload():
    """Verify that validation rejects malformed payloads."""
    from calibrated_explanations.utils.exceptions import ValidationError

    with pytest.raises(ValidationError):
        validate_plotspec(["not", "a", "dict"])
    with pytest.raises(ValidationError):
        validate_plotspec({})
    with pytest.raises(ValidationError):
        validate_plotspec({"plotspec_version": "1.0.0", "body": {"bars": "notalist"}})
    with pytest.raises(ValidationError):
        validate_plotspec({"plotspec_version": "1.0.0", "title": "missing body"})
    with pytest.raises(ValidationError):
        validate_plotspec(
            {
                "plotspec_version": "1.0.0",
                "body": {"bars": [{"label": "a"}]},
            }
        )


def test_interval_header_spec_optional_fields():
    """Verify that optional fields are preserved in IntervalHeaderSpec."""
    header = IntervalHeaderSpec(
        pred=0.4,
        low=0.2,
        high=0.9,
        xlim=(0.0, 1.0),
        xlabel="prediction",
        ylabel="probability",
        dual=False,
        neg_label="negative",
        pos_label="positive",
        uncertainty_color="#ccc",
        uncertainty_alpha=0.75,
    )

    # Semantic assertions: all fields are preserved
    assert header.dual is False
    assert header.xlabel == "prediction"
    assert header.neg_label == "negative"
    assert header.uncertainty_alpha == 0.75

    # Domain invariant: bounds ordering
    assert header.low <= header.pred <= header.high


def test_bar_item_and_panel_configuration():
    """Verify BarItem and BarHPanelSpec configuration."""
    items = [
        BarItem(
            label="feature",
            value=0.35,
            interval_low=-0.05,
            interval_high=0.4,
            color_role="positive",
            instance_value=3.14,
            solid_on_interval_crosses_zero=False,
        ),
        BarItem(label="baseline", value=-0.1),
    ]
    panel = BarHPanelSpec(
        bars=items,
        xlabel="Contribution",
        ylabel="Feature",
        solid_on_interval_crosses_zero=False,
    )

    # Semantic assertions: configuration preserved
    assert panel.bars[0].color_role == "positive"
    assert panel.solid_on_interval_crosses_zero is False

    # Semantic assertion: bar invariants
    bar0 = panel.bars[0]
    if bar0.interval_low is not None and bar0.interval_high is not None:
        assert bar0.interval_low <= bar0.value <= bar0.interval_high, (
            f"Bar interval violated: {bar0.interval_low} ≤ " f"{bar0.value} ≤ {bar0.interval_high}"
        )


def test_plotspec_all_fields():
    """Verify PlotSpec with all fields set."""
    header = IntervalHeaderSpec(pred=0.5, low=0.25, high=0.75)
    panel = BarHPanelSpec(bars=[BarItem(label="feat", value=0.2)])
    spec = PlotSpec(
        title="Example",
        figure_size=(6.0, 4.0),
        header=header,
        body=panel,
    )

    # Semantic assertions: all fields present and valid
    assert spec.figure_size == (6.0, 4.0)
    assert spec.body.bars[0].label == "feat"

    # Domain invariant: interval bounds
    assert spec.header.low <= spec.header.pred <= spec.header.high


def test_plotspec_to_dict_with_optional_fields():
    """Verify serialization preserves optional header and body fields."""
    header = IntervalHeaderSpec(
        pred=0.2,
        low=-0.1,
        high=0.7,
        xlim=(-1.0, 1.0),
        xlabel="x",
        ylabel="y",
    )
    body = BarHPanelSpec(
        bars=[
            BarItem(
                label="f0",
                value=0.4,
                interval_low=-0.2,
                interval_high=0.5,
                color_role="positive",
                instance_value={"foo": "bar"},
            ),
        ],
        xlabel="xlabel",
        ylabel="ylabel",
    )
    spec = PlotSpec(title="Example", figure_size=(8, 3), header=header, body=body)

    serialized = plotspec_to_dict(spec)

    # Semantic assertions: optional fields preserved in serialization
    assert serialized["figure_size"] == (8, 3)
    assert serialized["header"]["xlim"] == (-1.0, 1.0)
    assert serialized["body"]["bars"][0]["interval_high"] == 0.5
    assert serialized["body"]["bars"][0]["instance_value"] == {"foo": "bar"}

    # Semantic assertion: verify roundtrip preserves values
    restored = plotspec_from_dict(serialized)
    assert restored.header.xlim == (-1.0, 1.0)
    assert restored.body.bars[0].interval_high == 0.5


def test_plotspec_from_dict_casts_values():
    """Verify that from_dict properly casts string values to correct types.

    This tests deserialization robustness when values come as strings
    or tuples instead of their expected types.
    """
    raw = {
        "plotspec_version": "1.0.0",
        "title": "dict",
        "figure_size": [5, 2],
        "header": {
            "pred": "0.5",
            "low": 0,
            "high": 1,
            "xlim": ["0", "2"],
            "xlabel": "pred",
        },
        "body": {
            "xlabel": "Contribution",
            "ylabel": "Feature",
            "bars": [
                {
                    "label": "a",
                    "value": "0.25",
                    "interval_low": 0,
                    "interval_high": "0.5",
                    "color_role": "positive",
                },
                {
                    "label": "b",
                    "value": 0.1,
                    "interval_low": None,
                    "interval_high": None,
                },
            ],
        },
    }

    spec = plotspec_from_dict(raw)

    # Semantic assertions: type coercion successful
    assert spec.figure_size == (5, 2)
    assert spec.header is not None and spec.header.pred == 0.5
    assert spec.body is not None and spec.body.bars[0].interval_high == 0.5
    assert spec.body.bars[1].interval_low is None

    # Semantic assertion: interval invariants hold after coercion
    assert spec.header.low <= spec.header.pred <= spec.header.high, (
        f"Bounds violated after coercion: {spec.header.low} ≤ "
        f"{spec.header.pred} ≤ {spec.header.high}"
    )

    # Semantic assertion: bar intervals valid
    bar0 = spec.body.bars[0]
    if bar0.interval_low is not None and bar0.interval_high is not None:
        assert bar0.interval_low <= bar0.value <= bar0.interval_high, (
            f"Bar interval violated: {bar0.interval_low} ≤ " f"{bar0.value} ≤ {bar0.interval_high}"
        )
