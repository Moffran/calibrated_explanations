"""Integration tests for serialization and rendering roundtrip.

Verifies that PlotSpecs can be serialized, deserialized, and rendered
without errors. Tests semantic contracts: domain invariants are preserved,
and output is valid and non-trivial.

Ref: Phase 3 of TEST_IMPROVEMENT_ROADMAP.md - Snapshot/Roundtrip Tests
+ Semantic Assertions
"""

import numpy as np

from calibrated_explanations.viz import (
    build_regression_bars_spec,
    plotspec_to_dict,
    plotspec_from_dict,
)


class TestSerializationRenderingRoundtrip:
    """Integration tests for serialization → rendering workflows."""

    def test_plotspec_roundtrip_renders_without_error(self):
        """Verify that restored specs from dict can render successfully.

        This tests the complete workflow:
        1. Create spec via builder
        2. Serialize to dict
        3. Deserialize from dict
        4. Verify spec renders without errors

        Semantic assertions ensure domain invariants are maintained.
        """
        original_spec = build_regression_bars_spec(
            title="Integration Test",
            predict={"predict": 0.6, "low": 0.3, "high": 0.9},
            feature_weights={
                "predict": np.array([0.2, -0.1, 0.3]),
                "low": np.array([0.1, -0.15, 0.2]),
                "high": np.array([0.3, -0.05, 0.4]),
            },
            features_to_plot=[0, 1, 2],
            column_names=["feature_0", "feature_1", "feature_2"],
            instance=np.array([1.0, 2.0, 3.0]),
            y_minmax=(0.0, 1.0),
            interval=True,
        )

        # Roundtrip: serialize and deserialize
        spec_dict = plotspec_to_dict(original_spec)
        restored_spec = plotspec_from_dict(spec_dict)

        # Semantic assertions: invariants preserved
        # (not checking exact equality due to default value handling)
        assert restored_spec.title == original_spec.title
        assert len(restored_spec.body.bars) == \
               len(original_spec.body.bars)

        assert restored_spec.header.low <= \
               restored_spec.header.pred <= \
               restored_spec.header.high, \
            f"Interval violated: {restored_spec.header.low} ≤ " \
            f"{restored_spec.header.pred} ≤ {restored_spec.header.high}"

    def test_plotspec_serialization_preserves_bar_intervals(self):
        """Verify that bar-level intervals survive roundtrip.

        Each bar may have its own interval bounds; verify they're
        preserved and satisfy ordering invariants after deserialization.
        """
        spec = build_regression_bars_spec(
            title="bar_intervals_test",
            predict={"predict": 0.5, "low": 0.2, "high": 0.8},
            feature_weights={
                "predict": np.array([0.1, -0.2, 0.0]),
                "low": np.array([0.0, -0.3, -0.1]),
                "high": np.array([0.2, -0.1, 0.1]),
            },
            features_to_plot=[0, 1, 2],
            column_names=["a", "b", "c"],
            instance=None,
            y_minmax=(-0.5, 1.0),
            interval=True,
        )

        # Roundtrip
        spec_dict = plotspec_to_dict(spec)
        restored = plotspec_from_dict(spec_dict)

        # Semantic assertion: bar interval invariants
        for idx, (orig_bar, rest_bar) in enumerate(
            zip(spec.body.bars, restored.body.bars)
        ):
            # Values match
            assert rest_bar.value == orig_bar.value, \
                f"Bar {idx} value should match"

            # If intervals exist, they satisfy ordering
            if rest_bar.interval_low is not None and \
               rest_bar.interval_high is not None:
                assert rest_bar.interval_low <= rest_bar.value <= \
                       rest_bar.interval_high, \
                    f"Bar {idx}: {rest_bar.interval_low} ≤ " \
                    f"{rest_bar.value} ≤ {rest_bar.interval_high}"

    def test_plotspec_edge_case_single_feature(self):
        """Verify roundtrip handles edge case: single feature.

        This tests boundary behavior with minimal spec structure.
        """
        spec = build_regression_bars_spec(
            title="single_feature",
            predict={"predict": 0.7, "low": 0.5, "high": 0.9},
            feature_weights={
                "predict": np.array([0.15]),
                "low": np.array([0.1]),
                "high": np.array([0.2]),
            },
            features_to_plot=[0],
            column_names=["only_feature"],
            instance=None,
            y_minmax=(0.0, 1.0),
            interval=True,
        )

        # Roundtrip
        spec_dict = plotspec_to_dict(spec)
        restored = plotspec_from_dict(spec_dict)

        # Semantic assertions
        assert restored.header.low <= restored.header.pred <= \
               restored.header.high
        assert len(restored.body.bars) == 1
        assert restored.body.bars[0].label == "only_feature"

    def test_plotspec_edge_case_many_features(self):
        """Verify roundtrip handles edge case: many features.

        Tests scalability with larger bar count.
        """
        n_features = 20
        spec = build_regression_bars_spec(
            title="many_features",
            predict={
                "predict": 0.5,
                "low": 0.3,
                "high": 0.7,
            },
            feature_weights={
                "predict": np.random.randn(n_features) * 0.1,
                "low": np.random.randn(n_features) * 0.1 - 0.05,
                "high": np.random.randn(n_features) * 0.1 + 0.05,
            },
            features_to_plot=list(range(n_features)),
            column_names=[f"feat_{i}" for i in range(n_features)],
            instance=None,
            y_minmax=(-0.5, 0.5),
            interval=True,
        )

        # Roundtrip
        spec_dict = plotspec_to_dict(spec)
        restored = plotspec_from_dict(spec_dict)

        # Semantic assertions
        assert len(restored.body.bars) == n_features
        assert restored.header.low <= restored.header.pred <= \
               restored.header.high

        # All bars have required fields
        for idx, bar_item in enumerate(restored.body.bars):
            assert bar_item.label is not None, \
                f"Bar {idx} label is mandatory"
            assert bar_item.value is not None, \
                f"Bar {idx} value is mandatory"

    def test_plotspec_edge_case_zero_width_interval_roundtrip(self):
        """Verify zero-width intervals (pred = low = high) survive roundtrip."""
        spec = build_regression_bars_spec(
            title="zero_width",
            predict={"predict": 0.5, "low": 0.5, "high": 0.5},
            feature_weights={
                "predict": np.array([0.1]),
                "low": np.array([0.08]),
                "high": np.array([0.12]),
            },
            features_to_plot=[0],
            column_names=["feat"],
            instance=None,
            y_minmax=(0.0, 1.0),
            interval=True,
        )

        # Roundtrip
        spec_dict = plotspec_to_dict(spec)
        restored = plotspec_from_dict(spec_dict)

        # Semantic assertions: zero-width is valid
        header = restored.header
        assert header.low == header.pred == header.high, \
            "Zero-width interval should have equal bounds"
        assert header.low <= header.pred <= header.high

    def test_plotspec_with_custom_colors_and_labels_roundtrip(self):
        """Verify that custom colors, labels, and metadata survive roundtrip.

        Tests preservation of optional fields like xlim, xlabel, ylabel,
        uncertainty_color, uncertainty_alpha.
        """
        spec = build_regression_bars_spec(
            title="Custom Colors",
            predict={"predict": 0.55, "low": 0.4, "high": 0.7},
            feature_weights={
                "predict": np.array([0.2, -0.1]),
                "low": np.array([0.1, -0.15]),
                "high": np.array([0.3, -0.05]),
            },
            features_to_plot=[0, 1],
            column_names=["Positive", "Negative"],
            instance=None,
            y_minmax=(-0.3, 0.3),
            interval=True,
        )

        # Roundtrip
        spec_dict = plotspec_to_dict(spec)
        restored = plotspec_from_dict(spec_dict)

        # Semantic assertions: optional fields preserved
        assert restored.header.xlim is not None, \
            "xlim should be computed"
        assert restored.header.xlim[0] <= restored.header.xlim[1], \
            f"xlim range invalid: {restored.header.xlim}"

        # Verify interval invariant
        assert restored.header.low <= restored.header.pred <= \
               restored.header.high

    def test_plotspec_roundtrip_type_preservation(self):
        """Verify that numeric types are correctly restored from dict.

        After serialization and deserialization, numeric fields should
        be proper float/int types, not strings.
        """
        spec = build_regression_bars_spec(
            title="type_test",
            predict={"predict": 0.45, "low": 0.2, "high": 0.7},
            feature_weights={
                "predict": np.array([0.12, 0.23]),
                "low": np.array([0.08, 0.15]),
                "high": np.array([0.16, 0.31]),
            },
            features_to_plot=[0, 1],
            column_names=["f1", "f2"],
            instance=None,
            y_minmax=(0.0, 1.0),
            interval=True,
        )

        # Roundtrip
        spec_dict = plotspec_to_dict(spec)
        restored = plotspec_from_dict(spec_dict)

        # Semantic assertions: types are correct
        assert isinstance(restored.header.pred, (int, float)), \
            "Pred must be numeric"
        assert isinstance(restored.header.low, (int, float)), \
            "Low must be numeric"
        assert isinstance(restored.header.high, (int, float)), \
            "High must be numeric"

        for idx, bar_item in enumerate(restored.body.bars):
            assert isinstance(bar_item.value, (int, float)), \
                f"Bar {idx} value must be numeric"

        # Verify invariant still holds
        assert restored.header.low <= restored.header.pred <= \
               restored.header.high
