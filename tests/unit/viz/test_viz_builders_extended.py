import math

import numpy as np
import pytest

from calibrated_explanations.viz import builders


def test_probability_helper_utilities_and_segments():
    assert builders._looks_like_probability_values(0.0, 0.5, "1.0")
    assert not builders._looks_like_probability_values()
    assert not builders._looks_like_probability_values(0.1, math.inf)
    assert not builders._looks_like_probability_values("nan")

    builders._ensure_indexable_length("ok", [1, 2, 3], max_index=2)
    with pytest.raises(ValueError):
        builders._ensure_indexable_length("too_short", [1], max_index=1)

    low, high, xlim = builders._normalize_interval_bounds(
        float("nan"), float("inf"), y_minmax=(0.0, 2.0)
    )
    assert (low, high) == (0.0, 2.0)
    assert xlim == (0.0, 2.0)

    palette = builders._legacy_color_brew(4)
    assert len(palette) == 4 and palette[0] != palette[-1]

    assert builders._legacy_get_fill_color(1.0) == "#ff0000"
    assert builders._legacy_get_fill_color(0.2) != builders._legacy_get_fill_color(0.8)
    assert builders._legacy_get_fill_color(0.8, reduction=0.5) != builders._legacy_get_fill_color(
        0.8
    )

    two_segments = builders._build_probability_segments(
        low=0.2, high=0.8, center=0.5, reduction=1.0, pivot=0.5
    )
    assert len(two_segments) == 2 and two_segments[0].color != two_segments[1].color

    single_segment = builders._build_probability_segments(
        low=0.9, high=0.2, center=0.4, reduction=1.0, pivot=None
    )
    assert len(single_segment) == 1 and single_segment[0].low == min(0.2, 0.9)


def test_build_alternative_probabilistic_spec_interval_dict():
    feature_weights = {
        "predict": np.array([0.2, 0.8]),
        "low": np.array([0.1, 0.6]),
        "high": np.array([0.3, 0.9]),
    }
    spec = builders.build_alternative_probabilistic_spec(
        title="alt prob",
        predict={"predict": 0.6, "low": 0.1, "high": 0.9},
        feature_weights=feature_weights,
        features_to_plot=[0, 1],
        column_names=["first", "second"],
        rule_labels=["r0", "r1"],
        instance=[10, 20],
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by="interval",
        ascending=False,
        legacy_solid_behavior=False,
        neg_label="negative",
        pos_label="positive",
        uncertainty_color="#123456",
        uncertainty_alpha=0.8,
        xlabel="Probability mass",
        xlim=(0.0, 1.0),
        xticks=[0.0, 0.5, 1.0],
    )

    assert spec.header is None
    assert spec.body.is_alternative is True
    assert spec.body.pivot == 0.5
    assert spec.body.xticks == (0.0, 0.5, 1.0)
    assert spec.body.base_segments and len(spec.body.base_segments) == 2
    # Interval sorting should place the wider second bar first
    assert [bar.label for bar in spec.body.bars] == ["r1", "r0"]
    roles = {bar.color_role for bar in spec.body.bars}
    assert roles == {"positive", "negative"}


def test_build_alternative_probabilistic_spec_sequence_path():
    spec = builders.build_alternative_probabilistic_spec(
        title=None,
        predict={"predict": 0.4},
        feature_weights=[0.2, 0.7, 0.1],
        features_to_plot=[0, 1, 2],
        column_names=None,
        rule_labels=None,
        instance=None,
        y_minmax=None,
        interval=False,
        sort_by="label",
        ascending=True,
        legacy_solid_behavior=True,
        neg_label=None,
        pos_label=None,
        uncertainty_color=None,
        uncertainty_alpha=None,
        xlabel=None,
        xlim=(0.0, 1.0),
        xticks=None,
    )

    assert [bar.label for bar in spec.body.bars] == ["0", "1", "2"]
    for bar in spec.body.bars:
        assert bar.segments and bar.interval_low == bar.interval_high == bar.value


def test_build_alternative_regression_spec_paths():
    feature_weights = {
        "predict": np.array([0.3, 0.9]),
        "low": np.array([0.1, 0.6]),
        "high": np.array([0.4, 1.0]),
    }
    spec = builders.build_alternative_regression_spec(
        title="alt reg",
        predict={"predict": 0.5, "low": 0.2, "high": 0.8},
        feature_weights=feature_weights,
        features_to_plot=[0, 1],
        column_names=["left", "right"],
        rule_labels=None,
        instance=[1, 2],
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by="value",
        ascending=True,
        legacy_solid_behavior=False,
        neg_label=None,
        pos_label=None,
        uncertainty_color="#abcdef",
        uncertainty_alpha=0.6,
        xlabel="Interval",
        xlim=(0.0, 1.0),
        xticks=[0.0, 0.5, 1.0],
    )

    assert spec.body.is_alternative is True
    assert spec.body.base_segments[0].color == builders.REGRESSION_BASE_COLOR
    assert spec.body.base_lines[0][0] == pytest.approx(0.5)
    assert spec.body.xticks == (0.0, 0.5, 1.0)
    assert all(
        hasattr(bar, "line") and bar.line_color == builders.REGRESSION_BAR_COLOR
        for bar in spec.body.bars
    )

    seq_spec = builders.build_alternative_regression_spec(
        title=None,
        predict={"predict": 0.2, "low": 0.0, "high": 0.5},
        feature_weights=[0.1, -0.2],
        features_to_plot=[0, 1],
        column_names=None,
        rule_labels=None,
        instance=None,
        y_minmax=None,
        interval=False,
        sort_by=None,
        ascending=False,
        legacy_solid_behavior=True,
        neg_label=None,
        pos_label=None,
        uncertainty_color=None,
        uncertainty_alpha=None,
        xlabel=None,
        xlim=None,
        xticks=None,
    )

    assert seq_spec.body.base_segments[0].low <= seq_spec.body.base_segments[0].high
    for bar in seq_spec.body.bars:
        assert hasattr(bar, "segments") and hasattr(bar, "line")


def test_triangular_global_and_serialization_helpers():
    tri = builders.build_triangular_plotspec_dict(
        title="tri",  # unused but ensures signature parity
        proba=[0.2, 0.8],
        uncertainty=[0.1, 0.3],
        rule_proba=[0.25, 0.75],
        rule_uncertainty=[0.05, 0.1],
        num_to_show=5,
        is_probabilistic=False,
    )
    assert tri["plot_spec"]["kind"] == "triangular"
    assert tri["plot_spec"]["triangular"]["num_to_show"] == 5
    assert tri["plot_spec"]["mode"] == "regression"

    global_spec = builders.build_global_plotspec_dict(
        title="global",
        proba=[0.1, 0.9],
        predict=None,
        low=[0.0, 0.2],
        high=[0.3, 0.7],
        uncertainty=[0.05, 0.15],
        y_test=[1, 0],
        is_regularized=False,
    )
    assert global_spec["plot_spec"]["kind"] == "global_regression"
    assert global_spec["plot_spec"]["axis_hints"]["xlim"][0] <= 0.1

    fallback = builders.build_global_plotspec_dict(
        title=None,
        proba=["bad"],
        predict=None,
        low=[0.0],
        high=[1.0],
        uncertainty=["oops"],
        y_test=None,
        is_regularized=True,
    )
    assert fallback["plot_spec"]["axis_hints"] == {"xlim": [0.0, 1.0], "ylim": [0.0, 1.0]}

    regression_spec = builders.build_regression_bars_spec(
        title="serialize",
        predict={"predict": 1.0, "low": 0.5, "high": 1.5},
        feature_weights={
            "predict": [0.5, -0.4],
            "low": [0.4, -0.6],
            "high": [0.6, -0.2],
        },
        features_to_plot=[0, 1],
        column_names=["A", "B"],
        rule_labels=None,
        instance=[1, 2],
        y_minmax=(0.0, 2.0),
        interval=True,
        confidence=95.0,
    )
    regression_dict = builders.plotspec_to_dict(regression_spec)
    assert regression_dict["plot_spec"]["uncertainty"] is True
    assert regression_dict["plot_spec"]["feature_entries"][0]["name"] == "A"

    factual = builders.build_factual_probabilistic_plotspec_dict(
        title="prob",
        predict={"predict": 0.6, "low": 0.4, "high": 0.8},
        feature_weights={
            "predict": [0.9, 0.2],
            "low": [0.8, 0.1],
            "high": [0.95, 0.3],
        },
        features_to_plot=[0, 1],
        column_names=["x", "y"],
        rule_labels=None,
        instance=[1, 2],
        y_minmax=None,
        interval=True,
        sort_by="abs",
        ascending=False,
        legacy_solid_behavior=True,
        neg_label="neg",
        pos_label="pos",
        uncertainty_color="#ddd",
        uncertainty_alpha=0.3,
        neg_caption="neg caption",
        pos_caption="pos caption",
    )
    assert factual["plot_spec"]["kind"] == "factual_probabilistic"
    assert factual["plot_spec"]["uncertainty"] is True
