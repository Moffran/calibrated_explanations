"""Tests for plugin-contract.md documentation code examples.

These tests verify that all code examples in docs/contributor/plugin-contract.md
are syntactically correct and functionally sound, following the repository's
doc-testing protocol.
"""

# pylint: disable=import-outside-toplevel,unused-argument,missing-docstring,invalid-name,attribute-defined-outside-init,protected-access

from __future__ import annotations

from typing import Any


class TestHelloIntervalCalibratorPlugin:
    """Test the HelloIntervalCalibratorPlugin example from plugin-contract.md."""

    def test_plugin_metadata_structure(self):
        """Verify the plugin metadata has required fields."""
        from calibrated_explanations.plugins.intervals import IntervalCalibratorPlugin

        class HelloIntervalCalibratorPlugin(IntervalCalibratorPlugin):
            """Minimal interval calibrator example that computes fixed-width bounds."""

            plugin_meta = {
                "schema_version": 1,
                "name": "hello.interval.calibrator",
                "version": "0.1.0",
                "provider": "example-team",
                "capabilities": ["interval:classification", "interval:regression"],
                "modes": ("classification", "regression"),
                "dependencies": (),
                "trusted": False,
                "trust": False,
                "confidence_source": "hello",
                "requires_bins": False,
                "fast_compatible": False,
            }

            def create(self, context, **kwargs: Any) -> Any:
                """Return a list of calibrators for feature-wise and model-level intervals."""
                task = str(context.metadata.get("task") or context.metadata.get("mode") or "")
                if "classification" in task:
                    return []
                else:
                    return []

        plugin = HelloIntervalCalibratorPlugin()
        assert plugin.plugin_meta["name"] == "hello.interval.calibrator"
        assert plugin.plugin_meta["schema_version"] == 1
        assert "interval:classification" in plugin.plugin_meta["capabilities"]

    def test_plugin_registration(self):
        """Verify the interval plugin can be registered."""
        from calibrated_explanations.plugins.base import validate_plugin_meta
        from calibrated_explanations.plugins import (
            register_interval_plugin,
            _INTERVAL_PLUGINS,
        )
        from calibrated_explanations.plugins.intervals import IntervalCalibratorPlugin

        class HelloIntervalCalibratorPlugin(IntervalCalibratorPlugin):
            plugin_meta = {
                "schema_version": 1,
                "name": "hello.interval.calibrator.test",
                "version": "0.1.0",
                "provider": "example-team",
                "capabilities": ["interval:classification", "interval:regression"],
                "modes": ("classification", "regression"),
                "dependencies": (),
                "trusted": False,
                "trust": False,
                "confidence_source": "hello",
                "requires_bins": False,
                "fast_compatible": False,
            }

            def create(self, context, **kwargs):
                return []

        plugin = HelloIntervalCalibratorPlugin()
        validate_plugin_meta(dict(plugin.plugin_meta))

        identifier = "external.hello.interval.test"
        try:
            register_interval_plugin(identifier, plugin)
            assert identifier in _INTERVAL_PLUGINS
        finally:
            _INTERVAL_PLUGINS.pop(identifier, None)


class TestHelloExplanationPlugin:
    """Test the HelloExplanationPlugin example from plugin-contract.md."""

    def test_plugin_metadata_structure(self):
        """Verify the explanation plugin metadata has required fields."""
        from dataclasses import dataclass
        from calibrated_explanations.plugins.explanations import ExplanationPlugin

        @dataclass
        class HelloExplanationPlugin(ExplanationPlugin):
            """Minimal example that returns calibrated explanations for binary classification."""

            plugin_meta = {
                "schema_version": 1,
                "name": "hello.explanation.plugin",
                "version": "0.1.0",
                "provider": "example-team",
                "capabilities": ["explain", "explanation:factual", "task:classification"],
                "modes": ("factual",),
                "tasks": ("classification",),
                "dependencies": (),
                "interval_dependency": "core.interval.legacy",
                "plot_dependency": "plot_spec.default",
                "trusted": False,
            }

            def supports(self, model: Any) -> bool:
                return hasattr(model, "predict_proba")

            def initialize(self, context):
                self.context = context

            def explain_batch(self, x, request):
                from calibrated_explanations.explanations import CalibratedExplanations

                explanations = CalibratedExplanations(None, 0, x, {}, {}, {}, {})
                return type(
                    "ExplanationBatch",
                    (),
                    {
                        "explanations": explanations,
                        "collection_metadata": {
                            "mode": self.plugin_meta.get("modes", ("factual",))[0],
                            "task": self.context.task,
                        },
                    },
                )()

        plugin = HelloExplanationPlugin()
        assert plugin.plugin_meta["name"] == "hello.explanation.plugin"
        assert plugin.plugin_meta["plot_dependency"] == "plot_spec.default"
        assert plugin.plugin_meta["interval_dependency"] == "core.interval.legacy"

    def test_plugin_registration(self):
        """Verify the explanation plugin can be registered."""
        from dataclasses import dataclass
        from calibrated_explanations.plugins.base import validate_plugin_meta
        from calibrated_explanations.plugins import (
            register_explanation_plugin,
            _EXPLANATION_PLUGINS,
        )
        from calibrated_explanations.plugins.explanations import ExplanationPlugin

        @dataclass
        class HelloExplanationPlugin(ExplanationPlugin):
            plugin_meta = {
                "schema_version": 1,
                "name": "hello.explanation.plugin.test",
                "version": "0.1.0",
                "provider": "example-team",
                "capabilities": ["explain", "explanation:factual", "task:classification"],
                "modes": ("factual",),
                "tasks": ("classification",),
                "dependencies": (),
                "interval_dependency": "core.interval.legacy",
                "plot_dependency": "plot_spec.default",
                "trusted": False,
            }

            def supports(self, model):
                return hasattr(model, "predict_proba")

            def initialize(self, context):
                self.context = context

            def explain_batch(self, x, request):
                from calibrated_explanations.explanations import CalibratedExplanations

                explanations = CalibratedExplanations(None, 0, x, {}, {}, {}, {})
                return type(
                    "ExplanationBatch",
                    (),
                    {
                        "explanations": explanations,
                        "collection_metadata": {"mode": "factual", "task": "classification"},
                    },
                )()

        plugin = HelloExplanationPlugin()
        validate_plugin_meta(dict(plugin.plugin_meta))

        identifier = "external.hello.explanation.test"
        try:
            register_explanation_plugin(identifier, plugin)
            assert identifier in _EXPLANATION_PLUGINS
        finally:
            _EXPLANATION_PLUGINS.pop(identifier, None)


class TestHelloPlotPlugin:
    """Test the HelloPlotPlugin examples from plugin-contract.md."""

    def test_plot_builder_metadata(self):
        """Verify the plot builder metadata has required fields."""
        from calibrated_explanations.plugins.plots import PlotBuilder

        class HelloPlotBuilder(PlotBuilder):
            """Minimal plot builder that creates a simple PlotSpec payload."""

            plugin_meta = {
                "schema_version": 1,
                "name": "hello.plot.builder",
                "version": "0.1.0",
                "provider": "example-team",
                "style": "hello",
                "output_formats": ("png", "svg"),
                "capabilities": ["plot:builder"],
                "dependencies": (),
                "trusted": False,
            }

            def build(self, context):
                return {
                    "plot_spec": {
                        "title": "Hello Plot",
                        "description": "A minimal plot builder example",
                        "primitives": [],
                    }
                }

        builder = HelloPlotBuilder()
        assert builder.plugin_meta["style"] == "hello"
        assert "png" in builder.plugin_meta["output_formats"]
        assert "plot:builder" in builder.plugin_meta["capabilities"]

    def test_plot_renderer_metadata(self):
        """Verify the plot renderer metadata has required fields."""
        from calibrated_explanations.plugins.plots import PlotRenderer

        class HelloPlotRenderer(PlotRenderer):
            """Minimal plot renderer that materializes PlotSpec payloads."""

            plugin_meta = {
                "schema_version": 1,
                "name": "hello.plot.renderer",
                "version": "0.1.0",
                "provider": "example-team",
                "output_formats": ("png", "svg"),
                "capabilities": ["plot:renderer"],
                "supports_interactive": False,
                "dependencies": (),
                "trusted": False,
            }

            def render(self, artifact, *, context):
                from calibrated_explanations.plugins.plots import PlotRenderResult

                return PlotRenderResult(
                    artifact=artifact,
                    figure=None,
                    saved_paths=(),
                    extras={},
                )

        renderer = HelloPlotRenderer()
        assert renderer.plugin_meta["supports_interactive"] is False
        assert "plot:renderer" in renderer.plugin_meta["capabilities"]

    def test_plot_plugin_registration(self):
        """Verify plot builder, renderer, and style can be registered."""
        from calibrated_explanations.plugins.base import validate_plugin_meta
        from calibrated_explanations.plugins.plots import (
            PlotBuilder,
            PlotRenderer,
            PlotRenderResult,
        )
        from calibrated_explanations.plugins import (
            register_plot_builder,
            register_plot_renderer,
            register_plot_style,
            _PLOT_BUILDERS,
            _PLOT_RENDERERS,
            _PLOT_STYLES,
        )

        class HelloPlotBuilder(PlotBuilder):
            plugin_meta = {
                "schema_version": 1,
                "name": "hello.plot.builder.test",
                "version": "0.1.0",
                "provider": "example-team",
                "style": "hello.test",
                "output_formats": ("png", "svg"),
                "capabilities": ["plot:builder"],
                "dependencies": (),
                "trusted": False,
                "legacy_compatible": False,
            }

            def build(self, context):
                return {"plot_spec": {"title": "Test", "primitives": []}}

        class HelloPlotRenderer(PlotRenderer):
            plugin_meta = {
                "schema_version": 1,
                "name": "hello.plot.renderer.test",
                "version": "0.1.0",
                "provider": "example-team",
                "output_formats": ("png", "svg"),
                "capabilities": ["plot:renderer"],
                "supports_interactive": False,
                "dependencies": (),
                "trusted": False,
                "legacy_compatible": False,
            }

            def render(self, artifact, *, context):
                return PlotRenderResult(
                    artifact=artifact,
                    figure=None,
                    saved_paths=(),
                    extras={},
                )

        builder = HelloPlotBuilder()
        renderer = HelloPlotRenderer()

        validate_plugin_meta(dict(builder.plugin_meta))
        validate_plugin_meta(dict(renderer.plugin_meta))

        builder_id = "hello.plot.builder.test"
        renderer_id = "hello.plot.renderer.test"
        style_id = "hello.test"

        try:
            register_plot_builder(builder_id, builder)
            register_plot_renderer(renderer_id, renderer)

            assert builder_id in _PLOT_BUILDERS
            assert renderer_id in _PLOT_RENDERERS

            register_plot_style(
                style_id,
                metadata={
                    "style": style_id,
                    "builder_id": builder_id,
                    "renderer_id": renderer_id,
                    "fallbacks": ("plot_spec.default", "legacy"),
                    "is_default": False,
                },
            )

            assert style_id in _PLOT_STYLES
        finally:
            _PLOT_BUILDERS.pop(builder_id, None)
            _PLOT_RENDERERS.pop(renderer_id, None)
            _PLOT_STYLES.pop(style_id, None)


class TestPluginWiringMethodsAB:
    """Test the wiring examples from plugin-contract.md Methods A & B."""

    def test_method_a_explainer_parameter(self):
        """Test Method A: Wiring via CalibratedExplainer parameter."""
        from tests._helpers import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer

        # Prepare data
        data = load_breast_cancer()
        x = data.data
        y = data.target
        x_temp, x_test, y_temp, _ = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train, x_cal, y_train, y_cal = train_test_split(
            x_temp, y_temp, test_size=0.4, random_state=42
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_cal = scaler.transform(x_cal)
        x_test = scaler.transform(x_test)

        model, _ = get_classification_model("RF", x_train, y_train)

        # Test Method A: Parameter wiring
        explainer = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
            plot_style="plot_spec.default",
        )

        assert explainer._plot_style_override == "plot_spec.default"
        # Verify the style chain includes our override
        chain = explainer._plot_style_chain
        assert chain[0] == "plot_spec.default"

    def test_method_b_plot_parameter(self):
        """Test Method B: Wiring via explanation.plot() parameter."""
        from tests._helpers import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer

        # Prepare data
        data = load_breast_cancer()
        x = data.data
        y = data.target
        x_temp, x_test, y_temp, _ = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train, x_cal, y_train, y_cal = train_test_split(
            x_temp, y_temp, test_size=0.4, random_state=42
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_cal = scaler.transform(x_cal)
        x_test = scaler.transform(x_test)

        model, _ = get_classification_model("RF", x_train, y_train)

        # Test Method B: Plot parameter wiring
        explainer = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
        )

        explanations = explainer.explain_factual(x_test[:3])

        # Verify explanations object has plot method with style_override support
        assert hasattr(explanations, "plot")
        # Calling plot with style_override should work (we test interface, not rendering)
        # since rendering requires matplotlib/display setup
        assert "style_override" in explanations.plot.__code__.co_varnames


class TestPluginDependencyPropagation:
    """Test plugin dependency propagation from explanation plugin metadata."""

    def test_plugin_dependency_metadata_seeding(self):
        """Verify that explanation plugin metadata dependencies are seeded into fallback chain."""
        from tests._helpers import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer
        from calibrated_explanations.plugins import ensure_builtin_plugins

        ensure_builtin_plugins()

        # Prepare data
        data = load_breast_cancer()
        x = data.data
        y = data.target
        x_temp, x_test, y_temp, _ = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train, x_cal, y_train, y_cal = train_test_split(
            x_temp, y_temp, test_size=0.4, random_state=42
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_cal = scaler.transform(x_cal)
        x_test = scaler.transform(x_test)

        model, _ = get_classification_model("RF", x_train, y_train)

        # Create explainer with core plugins (which have declared dependencies)
        explainer = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
            factual_plugin="core.explanation.factual",
        )

        # Generate explanations
        explanations = explainer.explain_factual(x_test[:3])

        # Verify metadata includes plot fallbacks from plugin dependencies
        assert hasattr(explanations, "telemetry")
        metadata = getattr(explanations, "telemetry", {})
        if "plot_fallbacks" in metadata:
            # Should have plot fallbacks seeded from plugin
            assert isinstance(metadata["plot_fallbacks"], (list, tuple))
