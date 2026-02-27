import numpy as np
import pytest

from calibrated_explanations.viz import builders


def test_probabilistic_spec_clamps_header_bounds():
    spec = builders.build_probabilistic_bars_spec(
        title="clamp",
        predict={"predict": float("nan"), "low": float("nan"), "high": float("inf")},
        feature_weights={"predict": [0.3], "low": [0.1], "high": [0.4]},
        features_to_plot=[0],
        column_names=["value"],
        rule_labels=["r0"],
        instance=[5],
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by=None,
        ascending=False,
        legacy_solid_behavior=True,
        neg_label=None,
        pos_label=None,
        uncertainty_color="#333333",
        uncertainty_alpha=0.5,
        neg_caption=None,
        pos_caption=None,
    )

    header = spec.header
    assert header.xlim == (0.0, 1.0)
    assert header.low == pytest.approx(0.0)
    assert header.high == pytest.approx(1.0)
    assert header.show_intervals

    bar = spec.body.bars[0]
    assert 0.0 <= bar.interval_low <= bar.interval_high <= 1.0


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


def test_triangular_global_and_serialization_helpers__should_produce_correct_plot_spec_dicts():
    """Verify that builders produce valid plot_spec dicts with correct structure.

    Domain Invariants:
    - Triangular plot spec must have 'kind' field = 'triangular'
    - Triangular must preserve num_to_show setting
    - Mode must be correctly set (regression, classification)
    - Global spec must have 'kind' field = 'global_regression' (for regression mode)
    - All returned dicts must have 'plot_spec' key (not dict key check, but semantic)
    Ref: ADR-005 Explanation Envelope, ADR-007 PlotSpec Abstraction
    """
    tri = builders.build_triangular_plotspec_dict(
        title="tri",  # unused but ensures signature parity
        proba=[0.2, 0.8],
        uncertainty=[0.1, 0.3],
        rule_proba=[0.25, 0.75],
        rule_uncertainty=[0.05, 0.1],
        num_to_show=5,
        is_probabilistic=False,
    )
    # Domain invariant: triangular plot spec has correct kind
    assert "plotspec_version" in tri, "Envelope must include 'plotspec_version'"
    assert "plot_spec" in tri, "Result must be wrapped in plot_spec"
    assert (
        tri["plot_spec"]["kind"] == "triangular"
    ), "Triangular builder must produce 'triangular' kind"

    # Domain invariant: triangular settings preserved
    assert "triangular" in tri["plot_spec"], "Triangular config must be present"
    assert (
        tri["plot_spec"]["triangular"]["num_to_show"] == 5
    ), "num_to_show parameter must be preserved exactly"

    # Domain invariant: mode correctly set
    assert (
        tri["plot_spec"]["mode"] == "regression"
    ), "Mode must be 'regression' for non-probabilistic setup"

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
    # Domain invariant: global regression spec has correct kind
    assert "plotspec_version" in global_spec, "Envelope must include 'plotspec_version'"
    assert (
        global_spec["plot_spec"]["kind"] == "global_regression"
    ), "Global builder must produce 'global_regression' kind"

    # Domain invariant: axis hints are sensible (xlim lower bound reasonable)
    assert "axis_hints" in global_spec["plot_spec"], "Global spec must include axis_hints"
    xlim = global_spec["plot_spec"]["axis_hints"]["xlim"]
    assert xlim[0] <= 0.1, f"xlim lower bound should be <= 0.1, got {xlim[0]}"

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
    # Domain invariant: fallback should have default axis hints
    assert "axis_hints" in fallback["plot_spec"]
    axis_hints = fallback["plot_spec"]["axis_hints"]
    # Default xlim and ylim should be [0, 1] for probabilistic bounds
    assert axis_hints.get("xlim") == [
        0.0,
        1.0,
    ], f"Default xlim should be [0.0, 1.0], got {axis_hints.get('xlim')}"
    assert axis_hints.get("ylim") == [
        0.0,
        1.0,
    ], f"Default ylim should be [0.0, 1.0], got {axis_hints.get('ylim')}"

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
    # Domain invariant: uncertainty flag is set when confidence is provided
    assert "plot_spec" in regression_dict
    assert (
        regression_dict["plot_spec"]["uncertainty"] is True
    ), "Uncertainty must be True when confidence interval is enabled"

    # Domain invariant: feature entries preserve column names
    assert "feature_entries" in regression_dict["plot_spec"]
    feature_entries = regression_dict["plot_spec"]["feature_entries"]
    assert len(feature_entries) >= 1, "Must have at least one feature entry"
    assert (
        feature_entries[0]["name"] == "A"
    ), "First feature entry must have name 'A' (from column_names)"

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
    # Domain invariant: factual probabilistic plot has correct kind
    assert (
        factual["plot_spec"]["kind"] == "factual_probabilistic"
    ), "Factual builder must produce 'factual_probabilistic' kind"

    # Domain invariant: uncertainty flag is set when interval is enabled
    assert (
        factual["plot_spec"]["uncertainty"] is True
    ), "Uncertainty must be True when interval is enabled in factual plot"


# ---------------------------------------------------------------------------
# Public builder wrappers
# ---------------------------------------------------------------------------


def test_factual_probabilistic_spec_delegates():
    """Verify build_factual_probabilistic_spec delegates correctly."""
    spec = builders.build_factual_probabilistic_spec(
        title="fp",
        predict={"predict": 0.6, "low": 0.4, "high": 0.8},
        feature_weights=[0.3, 0.7],
        features_to_plot=[0, 1],
        column_names=["a", "b"],
        instance=[1, 2],
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by=None,
        ascending=False,
        legacy_solid_behavior=True,
    )
    assert spec.kind == "factual_probabilistic"


def test_factual_regression_spec_delegates():
    """Verify build_factual_regression_spec delegates correctly."""
    spec = builders.build_factual_regression_spec(
        title="fr",
        predict={"predict": 1.0},
        feature_weights=[0.5, -0.3],
        features_to_plot=[0, 1],
        column_names=["a", "b"],
        instance=[1, 2],
        y_minmax=None,
        interval=False,
    )
    assert spec.kind == "factual_regression"


# ---------------------------------------------------------------------------
# Regression bars edge cases
# ---------------------------------------------------------------------------


def test_regression_bars_empty_features_raises():
    """Verify build_regression_bars_spec raises on empty features_to_plot."""
    from calibrated_explanations.utils.exceptions import ValidationError

    with pytest.raises(ValidationError, match="cannot be empty"):
        builders.build_regression_bars_spec(
            title=None,
            predict={"predict": 0.0},
            feature_weights=[0.1],
            features_to_plot=[],
            column_names=["a"],
            instance=None,
            y_minmax=None,
            interval=False,
        )


def test_regression_bars_short_feature_weights_raises():
    """Verify build_regression_bars_spec raises when weights are too short."""
    from calibrated_explanations.utils.exceptions import ValidationError

    with pytest.raises(ValidationError, match="does not cover"):
        builders.build_regression_bars_spec(
            title=None,
            predict={"predict": 0.0},
            feature_weights=[0.1],
            features_to_plot=[0, 5],
            column_names=["a", "b", "c", "d", "e", "f"],
            instance=None,
            y_minmax=None,
            interval=False,
        )


# ---------------------------------------------------------------------------
# Probabilistic bars edge cases and sort paths
# ---------------------------------------------------------------------------


def test_probabilistic_bars_empty_features_raises():
    """Verify build_probabilistic_bars_spec raises on empty features."""
    from calibrated_explanations.utils.exceptions import ValidationError

    with pytest.raises(ValidationError, match="cannot be empty"):
        builders.build_probabilistic_bars_spec(
            title=None,
            predict={"predict": 0.5},
            feature_weights=[0.3],
            features_to_plot=[],
            column_names=["a"],
            instance=None,
            y_minmax=None,
            interval=False,
        )


def make_prob_spec(sort_by, ascending=False):
    """Helper for probabilistic bar sort tests."""
    return builders.build_probabilistic_bars_spec(
        title=None,
        predict={"predict": 0.6, "low": 0.4, "high": 0.8},
        feature_weights={
            "predict": [0.3, 0.8, 0.1],
            "low": [0.2, 0.6, 0.0],
            "high": [0.4, 0.9, 0.2],
        },
        features_to_plot=[0, 1, 2],
        column_names=["a", "c", "b"],
        rule_labels=["r0", "r1", "r2"],
        instance=[1, 2, 3],
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by=sort_by,
        ascending=ascending,
        legacy_solid_behavior=True,
    )


def test_prob_sort_by_value():
    """Verify probabilistic bars can be sorted by value."""
    spec = make_prob_spec("value", ascending=True)
    values = [b.value for b in spec.body.bars]
    assert values == sorted(values)


def test_prob_sort_by_abs():
    """Verify probabilistic bars can be sorted by absolute value."""
    spec = make_prob_spec("abs", ascending=False)
    mags = [abs(b.value) for b in spec.body.bars]
    assert mags == sorted(mags, reverse=True)


def test_prob_sort_by_width():
    """Verify probabilistic bars can be sorted by interval width."""
    spec = make_prob_spec("width", ascending=False)
    assert len(spec.body.bars) == 3


def test_prob_sort_by_label():
    """Verify probabilistic bars can be sorted by label."""
    spec = make_prob_spec("label", ascending=True)
    labels = [b.label for b in spec.body.bars]
    assert labels == sorted(labels)


# ---------------------------------------------------------------------------
# is_valid_probability_values edge cases
# ---------------------------------------------------------------------------


def test_is_valid_probability_values_rejects_nan():
    """Verify NaN is rejected as invalid probability."""
    assert not builders.is_valid_probability_values(float("nan"))


def test_is_valid_probability_values_rejects_inf():
    """Verify infinity is rejected as invalid probability."""
    assert not builders.is_valid_probability_values(float("inf"))


# ---------------------------------------------------------------------------
# Regression bars interval with sort by none and feature_order
# ---------------------------------------------------------------------------


def test_regression_bars_interval_with_feature_order():
    """Verify regression bars with interval=True populates feature_order."""
    spec = builders.build_regression_bars_spec(
        title="of",
        predict={"predict": 1.0, "low": 0.5, "high": 1.5},
        feature_weights={
            "predict": [0.5, -0.4],
            "low": [0.3, -0.6],
            "high": [0.7, -0.2],
        },
        features_to_plot=[0, 1],
        column_names=["A", "B"],
        instance=[1, 2],
        y_minmax=(0.0, 2.0),
        interval=True,
        sort_by="none",
    )
    assert spec.body is not None
    assert spec.feature_order == (0, 1)


def test_regression_bars_width_sort_without_interval():
    """Verify width sort falls back when bars have no interval bounds."""
    spec = builders.build_regression_bars_spec(
        title=None,
        predict={"predict": 1.0},
        feature_weights=[0.5, -0.3, 0.1],
        features_to_plot=[0, 1, 2],
        column_names=["a", "b", "c"],
        instance=None,
        y_minmax=None,
        interval=False,
        sort_by="width",
        ascending=True,
    )
    assert len(spec.body.bars) == 3


def test_regression_bars_unknown_sort_key_fallback():
    """Verify unknown sort key produces stable output without error."""
    spec = builders.build_regression_bars_spec(
        title=None,
        predict={"predict": 1.0},
        feature_weights=[0.5, -0.3],
        features_to_plot=[0, 1],
        column_names=["a", "b"],
        instance=None,
        y_minmax=None,
        interval=False,
        sort_by="custom_unsupported",
        ascending=False,
    )
    assert len(spec.body.bars) == 2


def test_prob_sort_width_without_interval():
    """Verify prob width sort falls back when bars lack interval."""
    spec = builders.build_probabilistic_bars_spec(
        title=None,
        predict={"predict": 0.6},
        feature_weights=[0.3, 0.8],
        features_to_plot=[0, 1],
        column_names=["a", "b"],
        instance=None,
        y_minmax=None,
        interval=False,
        sort_by="width",
        ascending=True,
        legacy_solid_behavior=True,
    )
    assert len(spec.body.bars) == 2


def test_prob_sort_unknown_key_fallback():
    """Verify prob unknown sort key produces stable output."""
    spec = builders.build_probabilistic_bars_spec(
        title=None,
        predict={"predict": 0.6},
        feature_weights=[0.3, 0.8],
        features_to_plot=[0, 1],
        column_names=["a", "b"],
        instance=None,
        y_minmax=None,
        interval=False,
        sort_by="custom_unsupported",
        ascending=False,
        legacy_solid_behavior=True,
    )
    assert len(spec.body.bars) == 2
