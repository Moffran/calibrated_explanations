import math
from types import MappingProxyType, SimpleNamespace

import numpy as np

from calibrated_explanations.plugins import PlotRenderContext
from calibrated_explanations.plugins.builtins import PlotSpecDefaultBuilder
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
    assert spec.header is not None
    assert math.isclose(spec.header.pred, 0.6)
    assert spec.header.dual is True
    assert spec.body is not None
    assert len(spec.body.bars) == 2
    assert {bar.color_role for bar in spec.body.bars} <= {"positive", "negative"}


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
    assert spec.header is not None
    assert spec.header.dual is False
    assert spec.body is not None
    assert len(spec.body.bars) == 2
    assert all(bar.color_role == "regression" for bar in spec.body.bars)


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
