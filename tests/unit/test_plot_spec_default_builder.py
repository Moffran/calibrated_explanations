from types import MappingProxyType, SimpleNamespace

import numpy as np

from calibrated_explanations.plugins import PlotRenderContext
from calibrated_explanations.plugins.builtins import PlotSpecDefaultBuilder
from calibrated_explanations.viz.builders import _legacy_get_fill_color
from calibrated_explanations.viz.plotspec import PlotSpec


def _make_context(intent, payload, explanation=None):
    return PlotRenderContext(
        explanation=explanation,
        instance_metadata=MappingProxyType({"type": "alternative"}),
        style="plot_spec.default",
        intent=MappingProxyType(intent),
        show=False,
        path=None,
        save_ext=None,
        options=MappingProxyType({"payload": payload}),
    )


def test_plot_spec_builder_handles_alternative_probabilistic():
    builder = PlotSpecDefaultBuilder()
    explanation = SimpleNamespace(
        get_mode=lambda: "classification",
        y_minmax=(0.0, 1.0),
        get_class_labels=lambda: ["No", "Yes"],
        prediction={"classes": 1},
    )
    payload = {
        "predict": {"predict": 0.6, "low": 0.45, "high": 0.7},
        "feature_predict": {
            "predict": [0.3, 0.7],
            "low": [0.2, 0.6],
            "high": [0.4, 0.8],
        },
        "features_to_plot": [0, 1],
        "column_names": ["rule_a", "rule_b"],
        "instance": [1.0, 2.0],
    }
    context = _make_context(
        {"type": "alternative", "mode": "classification", "title": "alt"},
        payload,
        explanation,
    )

    spec = builder.build(context)

    assert isinstance(spec, PlotSpec)
    assert spec.header is None
    assert spec.body is not None
    assert len(spec.body.bars) == 2
    assert {bar.color_role for bar in spec.body.bars} <= {"positive", "negative"}
    base_segments = getattr(spec.body, "base_segments", ())
    assert base_segments


def test_plot_spec_builder_handles_alternative_regression():
    builder = PlotSpecDefaultBuilder()
    explanation = SimpleNamespace(get_mode=lambda: "regression", y_minmax=(-2.0, 5.0))
    payload = {
        "predict": {"predict": 1.2, "low": 0.5, "high": 2.0},
        "feature_predict": {
            "predict": [0.4, -0.2],
            "low": [0.3, -0.4],
            "high": [0.6, 0.1],
        },
        "features_to_plot": [0, 1],
        "column_names": ["rule_a", "rule_b"],
        "instance": np.array([10.0, 5.0]),
    }
    context = _make_context(
        {"type": "alternative", "mode": "regression", "title": "alt"},
        payload,
        explanation,
    )

    spec = builder.build(context)

    assert isinstance(spec, PlotSpec)
    assert spec.header is None
    assert spec.body is not None
    assert len(spec.body.bars) == 2
    assert all(bar.color_role == REG_BAR_COLOR for bar in spec.body.bars)
    base_segments = getattr(spec.body, "base_segments", ())
    base_lines = getattr(spec.body, "base_lines", ())
    assert base_segments
    assert base_lines
    assert base_segments[0].color == REG_BASE_COLOR
    assert base_lines[0][1] == REG_BAR_COLOR


def test_plot_spec_builder_handles_alternative_regression_without_intervals():
    builder = PlotSpecDefaultBuilder()
    explanation = SimpleNamespace(get_mode=lambda: "regression", y_minmax=(-1.0, 3.0))
    payload = {
        "predict": {"predict": 0.5, "low": 0.2, "high": 0.8},
        "feature_predict": [1.1, -0.4],
        "features_to_plot": [0, 1],
        "column_names": ["rule_a", "rule_b"],
        "instance": [0.1, -0.2],
    }
    context = _make_context(
        {"type": "alternative", "mode": "regression", "title": "alt_no_interval"},
        payload,
        explanation,
    )

    spec = builder.build(context)

    assert isinstance(spec, PlotSpec)
    assert spec.header is None
    assert spec.body is not None
    assert len(spec.body.bars) == 2
    for bar in spec.body.bars:
        assert bar.color_role == REG_BAR_COLOR
        assert bar.interval_low <= bar.interval_high
        assert getattr(bar, "line_color", None) == REG_BAR_COLOR


def test_plot_spec_builder_normalizes_probability_regression_scale():
    builder = PlotSpecDefaultBuilder()
    explanation = SimpleNamespace(get_mode=lambda: "regression", y_minmax=(22500.0, 500001.0))
    payload = {
        "predict": {"predict": 0.22, "low": 0.18, "high": 0.24},
        "feature_predict": {
            "predict": [0.01, 0.3],
            "low": [0.0, 0.28],
            "high": [0.02, 0.32],
        },
        "features_to_plot": [0, 1],
        "column_names": ["rule_a", "rule_b"],
        "instance": [0.5, 1.2],
    }
    context = _make_context(
        {"type": "alternative", "mode": "regression", "title": "alt_prob_scale"},
        payload,
        explanation,
    )

    spec = builder.build(context)

    assert isinstance(spec, PlotSpec)
    assert spec.body is not None
    assert spec.body.xlim == (0.0, 1.0)
    assert spec.body.xlabel == "Probability"
    base_segments = getattr(spec.body, "base_segments", ())
    assert base_segments
    assert base_segments[0].low == 0.18 and base_segments[0].high == 0.24
    overlays = [getattr(bar, "segments", ()) for bar in spec.body.bars]
    assert overlays and all(seg[0].low >= 0.0 and seg[0].high <= 1.0 for seg in overlays if seg)


def test_plot_spec_builder_infers_missing_features_and_labels():
    builder = PlotSpecDefaultBuilder()
    explanation = SimpleNamespace(
        get_mode=lambda: "classification",
        y_minmax=(0.0, 1.0),
        get_class_labels=lambda: ["no", "yes"],
        prediction={"classes": 0},
    )

    payload = {
        "predict": {"predict": "0.4"},
        "feature_predict": {
            "predict": np.array([0.1, -0.2, 0.05]),
            "low": np.array([0.05, -0.25, 0.0]),
            "high": np.array([0.15, -0.15, 0.1]),
        },
        # features_to_plot and column_names intentionally omitted to exercise inference
        "instance": [10.0, 5.0, -2.0],
    }

    context = _make_context(
        {"type": "alternative", "mode": "classification", "title": "alt"},
        payload,
        explanation,
    )

    spec = builder.build(context)

    assert isinstance(spec, PlotSpec)
    assert spec.body is not None
    # Expect three inferred bars with default numeric labels
    assert [bar.label for bar in spec.body.bars] == ["0", "1", "2"]
    # Ensure intervals are respected after normalisation
    assert all(bar.interval_low is not None and bar.interval_high is not None for bar in spec.body.bars)
REG_BAR_COLOR = _legacy_get_fill_color(1.0, 1.0)
REG_BASE_COLOR = _legacy_get_fill_color(1.0, 0.15)
