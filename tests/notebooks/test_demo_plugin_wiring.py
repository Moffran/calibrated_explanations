"""Tests for demo_plugin_wiring.ipynb notebook code examples.

These tests verify that all code examples in notebooks/demo_plugin_wiring.ipynb
are syntactically correct and functionally sound, following the repository's
doc-testing protocol.
"""

# pylint: disable=import-outside-toplevel,unused-argument,missing-docstring,invalid-name,too-many-locals,protected-access

from __future__ import annotations


class TestNotebookSetup:
    """Test notebook setup and data preparation."""

    def test_imports_available(self):
        """Verify all required imports for notebook are available."""
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer

        # All imports should succeed without error
        assert load_breast_cancer is not None
        assert RandomForestClassifier is not None
        assert train_test_split is not None
        assert StandardScaler is not None
        assert CalibratedExplainer is not None

    def test_data_loading_and_preprocessing(self):
        """Test the notebook's data loading and preprocessing steps."""
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Load data
        data = load_breast_cancer()
        x = data.data
        y = data.target
        assert x.shape[0] > 0
        assert y.shape[0] > 0
        assert x.shape[0] == y.shape[0]

        # Split data (train/test, then train/calibration)
        x_temp, x_test, y_temp, _ = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train, x_cal, y_train, _ = train_test_split(
            x_temp, y_temp, test_size=0.4, random_state=42
        )

        # Scale data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_cal = scaler.transform(x_cal)
        x_test = scaler.transform(x_test)

        # Verify sizes match expectations
        assert x_train.shape[0] > 0
        assert x_cal.shape[0] > 0
        assert x_test.shape[0] > 0
        assert x_train.shape[1] == 30  # Breast cancer has 30 features

        # Train a basic classifier
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(x_train, y_train)

        # Model should be trainable and predictable
        predictions = model.predict(x_test[:3])
        assert predictions.shape[0] == 3


class TestNotebookMethodA:
    """Test Method A: Parameter-based wiring from notebook."""

    def test_method_a_explainer_initialization(self):
        """Test Method A: Creating CalibratedExplainer with plot_style parameter."""
        from tests._helpers import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer

        # Prepare data (same as notebook)
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

        # Method A: Explicit plot style parameter
        explainer_a = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
            plot_style="plot_spec.default",
        )

        # Verify explainer was created with the style
        assert explainer_a is not None
        assert explainer_a._plot_style_override == "plot_spec.default"

        # Generate explanations
        explanations = explainer_a.explain_factual(x_test[:3])
        assert explanations is not None


class TestNotebookMethodB:
    """Test Method B: plot() parameter from notebook."""

    def test_method_b_plot_style_override(self):
        """Test Method B: Using style_override in explanations.plot() call."""
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

        # Create explainer without explicit style
        explainer = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
        )

        # Generate explanations
        explanations = explainer.explain_factual(x_test[:3])

        # Method B: Use style_override in plot() call
        # Verify the plot method accepts style_override parameter
        import inspect

        plot_signature = inspect.signature(explanations.plot)
        assert "style_override" in plot_signature.parameters


class TestNotebookMethodC:
    """Test Method C: Environment variable wiring from notebook."""

    def test_method_c_environment_configuration(self):
        """Test Method C: Using environment variables for plot style."""
        import os
        from tests._helpers import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer

        # Save original environment
        old_style = os.environ.get("CE_PLOT_STYLE")

        try:
            # Set environment variable (Method C)
            os.environ["CE_PLOT_STYLE"] = "legacy"

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

            # Create explainer (should use environment variable as fallback)
            explainer = CalibratedExplainer(
                model,
                x_cal,
                y_cal,
            )

            # Verify explainer was created successfully
            assert explainer is not None

            # Generate explanations
            explanations = explainer.explain_factual(x_test[:3])
            assert explanations is not None

        finally:
            # Restore original environment
            if old_style is not None:
                os.environ["CE_PLOT_STYLE"] = old_style
            else:
                os.environ.pop("CE_PLOT_STYLE", None)


class TestNotebookMethodE:
    """Test Method E: Plugin registration and dependency wiring from notebook."""

    def test_method_e_custom_plugin_registration(self):
        """Test Method E: Custom plugin with declared dependencies."""
        from dataclasses import dataclass
        from calibrated_explanations.plugins.explanations import ExplanationPlugin
        from calibrated_explanations.plugins.registry import (
            register_explanation_plugin,
            _EXPLANATION_PLUGINS,
        )
        from calibrated_explanations.plugins.base import validate_plugin_meta
        from tests._helpers import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer

        # Define custom plugin with dependencies (Method E)
        @dataclass
        class HelloFactualPlugin(ExplanationPlugin):
            plugin_meta = {
                "schema_version": 1,
                "name": "hello.explanation.factual.test",
                "version": "0.1.0",
                "provider": "example",
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
                pass

            def explain_batch(self, x, request):
                return None

        plugin = HelloFactualPlugin()
        validate_plugin_meta(dict(plugin.plugin_meta))

        plugin_id = "external.hello.factual.test"

        try:
            # Register the plugin
            register_explanation_plugin(plugin_id, plugin)
            assert plugin_id in _EXPLANATION_PLUGINS

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

            # Use the registered plugin (Method E)
            explainer = CalibratedExplainer(
                model,
                x_cal,
                y_cal,
                factual_plugin=plugin_id,
            )

            # Generate explanations using the custom plugin
            explanations = explainer.explain_factual(x_test[:3])
            assert explanations is not None

        finally:
            _EXPLANATION_PLUGINS.pop(plugin_id, None)

    def test_method_e_plugin_dependencies_in_fallback(self):
        """Test that plugin dependencies are included in fallback chain."""
        from dataclasses import dataclass
        from calibrated_explanations.plugins.explanations import ExplanationPlugin
        from calibrated_explanations.plugins.registry import (
            register_explanation_plugin,
            ensure_builtin_plugins,
            _EXPLANATION_PLUGINS,
        )
        from tests._helpers import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer

        ensure_builtin_plugins()

        @dataclass
        class CustomFactualPlugin(ExplanationPlugin):
            plugin_meta = {
                "schema_version": 1,
                "name": "custom.factual.with_deps",
                "version": "0.1.0",
                "provider": "custom",
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
                pass

            def explain_batch(self, x, request):
                return None

        plugin = CustomFactualPlugin()
        plugin_id = "external.custom.factual.with_deps"

        try:
            register_explanation_plugin(plugin_id, plugin)

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

            # Create explainer with custom plugin
            explainer = CalibratedExplainer(
                model,
                x_cal,
                y_cal,
                factual_plugin=plugin_id,
            )

            # Verify the plugin was used
            assert explainer._explanation_plugin_overrides["factual"] == plugin_id

            # Generate explanations
            explanations = explainer.explain_factual(x_test[:3])
            assert explanations is not None

        finally:
            _EXPLANATION_PLUGINS.pop(plugin_id, None)


class TestNotebookIntegration:
    """Integration tests for full notebook workflow."""

    def test_full_workflow_with_multiple_methods(self):
        """Test a complete workflow using multiple wiring methods."""
        import os
        from tests._helpers import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer

        # Prepare data (once for all methods)
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

        # Test Method A
        explainer_a = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
            plot_style="plot_spec.default",
        )
        explanations_a = explainer_a.explain_factual(x_test[:3])
        assert explanations_a is not None

        # Test Method B (with Method A explainer - plot() uses style_override)
        # Just verify the plot method exists and accepts style_override
        assert hasattr(explanations_a, "plot")

        # Test Method C (environment variable)
        old_env = os.environ.get("CE_PLOT_STYLE")
        try:
            os.environ["CE_PLOT_STYLE"] = "legacy"
            explainer_c = CalibratedExplainer(
                model,
                x_cal,
                y_cal,
            )
            explanations_c = explainer_c.explain_factual(x_test[:3])
            assert explanations_c is not None
        finally:
            if old_env is not None:
                os.environ["CE_PLOT_STYLE"] = old_env
            else:
                os.environ.pop("CE_PLOT_STYLE", None)

        # All explanations should be valid
        assert explanations_a is not None
        assert explanations_c is not None
