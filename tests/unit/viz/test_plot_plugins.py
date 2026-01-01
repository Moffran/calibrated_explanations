from calibrated_explanations.utils.exceptions import PlotPluginError
from calibrated_explanations.viz.plugins import BasePlotBuilder, BasePlotRenderer
from calibrated_explanations.viz.serializers import PLOTSPEC_VERSION


def test_plot_builder_should_require_build_implementation_when_called():
    builder = BasePlotBuilder()
    builder.initialize(context={"stage": "unit-test"})

    try:
        builder.build(context={"stage": "unit-test"})
    except PlotPluginError as exc:
        assert "build" in str(exc).lower()
    else:
        raise AssertionError("Expected PlotPluginError for unimplemented build()")


def test_plot_renderer_should_require_render_implementation_for_valid_plotspec():
    renderer = BasePlotRenderer()
    renderer.initialize(context={"stage": "unit-test"})

    artifact = {
        "plotspec_version": PLOTSPEC_VERSION,
        "plot_spec": {
            "kind": "factual_regression",
            "mode": "regression",
            "header": {"pred": 0.5, "low": 0.2, "high": 0.8},
            "feature_entries": [{"name": "f1", "weight": 0.1}],
        },
    }

    try:
        renderer.render(artifact, context={"stage": "unit-test"})
    except PlotPluginError as exc:
        assert "render" in str(exc).lower()
    else:
        raise AssertionError("Expected PlotPluginError for unimplemented render()")


def test_plot_renderer_should_surface_validation_error_when_plotspec_invalid():
    renderer = BasePlotRenderer()

    artifact = {"plot_spec": {"kind": "factual_regression"}}

    try:
        renderer.render(artifact, context={"stage": "unit-test"})
    except PlotPluginError as exc:
        assert "plotspec validation failed" in str(exc).lower()
    else:
        raise AssertionError("Expected PlotPluginError for invalid PlotSpec payload")
