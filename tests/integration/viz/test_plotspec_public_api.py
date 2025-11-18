"""Integration tests for PlotSpec public API.

Testing behavior through public APIs rather than private helpers.

Instead of directly testing private functions like `_looks_like_probability_values()`,
this module tests the public plotting APIs and verifies the desired behavior
through observable outcomes.

Ref: ADR-005 (Explanation Envelope)
"""

from __future__ import annotations

from calibrated_explanations.viz import (
    build_regression_bars_spec,
    plotspec_from_dict,
    plotspec_to_dict,
)


class TestPlotSpecIntervalSemantics:
    """Test PlotSpec interval behavior through public API.

    Tests verify the domain invariant: low <= predict <= high.
    Tests private helper _looks_like_probability_values through its usage
    in the public build and serialization APIs.
    """

    def test_build_bars_creates_valid_intervals(self):
        """Verify that build_regression_bars_spec creates specs with valid intervals.

        Domain Invariant: The prediction must lie within the interval bounds.
        Ref: ADR-005 Explanation Envelope
        """
        # When building a spec with probability values
        w_low = [0.05, -0.1, 0.15]
        w_high = [0.15, -0.3, 0.45]
        spec = build_regression_bars_spec(
            title="Test Plot",
            predict={"predict": 0.5, "low": 0.2, "high": 0.8},
            feature_weights={
                "predict": [0.1, -0.2, 0.3],
                "low": w_low,
                "high": w_high,
            },
            features_to_plot=[0, 1, 2],
            column_names=["a", "b", "c"],
            rule_labels=None,
            instance=None,
            y_minmax=(0.0, 1.0),
            interval=True,
            sort_by=None,
            ascending=False,
        )

        # Then the spec should have valid intervals (domain invariant)
        assert spec.header is not None
        assert spec.header.low is not None
        assert spec.header.pred is not None
        assert spec.header.high is not None

        # Core invariant: prediction within bounds
        low_val = spec.header.low
        pred_val = spec.header.pred
        high_val = spec.header.high
        assert (
            low_val <= pred_val <= high_val
        ), f"Bounds: {low_val} <= {pred_val} <= {high_val}"

    def test_roundtrip_preserves_interval_semantics(self):
        """Verify that interval semantics survive serialization roundtrip.

        Domain Invariant: low <= pred <= high must hold after roundtrip.
        Ref: ADR-005 Explanation Envelope
        """
        # Given a spec with valid intervals
        original = build_regression_bars_spec(
            title="Roundtrip Test",
            predict={"predict": 0.6, "low": 0.3, "high": 0.9},
            feature_weights={
                "predict": [0.2],
                "low": [0.1],
                "high": [0.3],
            },
            features_to_plot=[0],
            column_names=["feature"],
            rule_labels=None,
            instance=None,
            y_minmax=(0.0, 1.0),
            interval=True,
            sort_by=None,
            ascending=False,
        )

        # When serialized and deserialized
        d_serialized = plotspec_to_dict(original)
        restored = plotspec_from_dict(d_serialized)

        # Then semantic invariants still hold
        assert (
            restored.header.low
            <= restored.header.pred
            <= restored.header.high
        ), "Interval invariant violated after roundtrip"
        assert isinstance(restored.header.pred, (int, float))
        assert len(restored.body.bars) > 0, "Must have at least one bar"
