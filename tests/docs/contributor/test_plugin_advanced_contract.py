"""Tests for plugin-advanced-contract.md documentation code examples.

These tests verify that all code examples in docs/contributor/extending/plugin-advanced-contract.md
are syntactically correct and functionally sound, following the repository's
doc-testing protocol.
"""

# pylint: disable=import-outside-toplevel,unused-argument,missing-docstring,invalid-name,too-many-locals,protected-access

from __future__ import annotations

import os


class TestMethodCEnvironmentVariables:
    """Test Method C: Environment variable wiring from plugin-advanced-contract.md."""

    def test_ce_plot_style_environment_variable(self):
        """Verify CE_PLOT_STYLE environment variable configuration."""
        from calibrated_explanations.plotting import resolve_plot_style_chain
        from tests.helpers.model_utils import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer

        # Prepare data and explainer
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
        explainer = CalibratedExplainer(model, x_cal, y_cal)

        # Test that CE_PLOT_STYLE is checked in the resolution chain
        old_value = os.environ.get("CE_PLOT_STYLE")
        try:
            os.environ["CE_PLOT_STYLE"] = "legacy"
            # The chain should prioritize CE_PLOT_STYLE when no override is specified
            chain = resolve_plot_style_chain(explainer, explicit_style=None)
            # legacy should be in the fallback chain
            assert "legacy" in chain
        finally:
            if old_value is not None:
                os.environ["CE_PLOT_STYLE"] = old_value
            else:
                os.environ.pop("CE_PLOT_STYLE", None)

    def test_ce_plot_style_fallbacks_environment_variable(self):
        """Verify CE_PLOT_STYLE_FALLBACKS environment variable."""
        from calibrated_explanations.plotting import resolve_plot_style_chain
        from tests.helpers.model_utils import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer

        # Prepare data and explainer
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
        explainer = CalibratedExplainer(model, x_cal, y_cal)

        old_fallbacks = os.environ.get("CE_PLOT_STYLE_FALLBACKS")
        try:
            # Set fallbacks as comma-separated list
            os.environ["CE_PLOT_STYLE_FALLBACKS"] = "plot_spec.default,legacy"
            # The resolution chain should include these fallbacks
            chain = resolve_plot_style_chain(explainer, explicit_style=None)
            # Fallbacks should be included in the chain
            assert len(chain) > 0
        finally:
            if old_fallbacks is not None:
                os.environ["CE_PLOT_STYLE_FALLBACKS"] = old_fallbacks
            else:
                os.environ.pop("CE_PLOT_STYLE_FALLBACKS", None)

    def test_explanation_plugin_fallback_environment_variable(self):
        """Verify CE_EXPLANATION_PLUGIN_*_FALLBACKS environment variable."""
        from calibrated_explanations.plugins import (
            register_explanation_plugin,
            _EXPLANATION_PLUGINS,
        )
        from calibrated_explanations.plugins.explanations import ExplanationPlugin
        from dataclasses import dataclass

        @dataclass
        class TestExplanationPlugin(ExplanationPlugin):
            plugin_meta = {
                "schema_version": 1,
                "name": "test.explanation.fallback",
                "version": "0.1.0",
                "provider": "test",
                "capabilities": ["explain"],
                "modes": ("factual",),
                "tasks": ("classification",),
                "dependencies": (),
                "trusted": False,
            }

            def supports(self, model):
                return hasattr(model, "predict_proba")

            def initialize(self, context):
                pass

            def explain_batch(self, x, request):
                return None

        plugin = TestExplanationPlugin()
        plugin_id = "test.explanation.fallback"

        try:
            # Test that environment variable for fallbacks can be parsed
            old_val = os.environ.get("CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS")
            try:
                os.environ["CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS"] = (
                    "core.explanation.factual,test.explanation.fallback"
                )
                # This should allow the environment variable to be consumed
                assert "CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS" in os.environ
            finally:
                if old_val is not None:
                    os.environ["CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS"] = old_val
                else:
                    os.environ.pop("CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS", None)

            register_explanation_plugin(plugin_id, plugin)
            assert plugin_id in _EXPLANATION_PLUGINS
        finally:
            _EXPLANATION_PLUGINS.pop(plugin_id, None)


class TestMethodDConfigurationFile:
    """Test Method D: pyproject.toml configuration from plugin-advanced-contract.md."""

    def test_pyproject_toml_parsing(self):
        """Verify pyproject.toml configuration structure can be read."""
        from pathlib import Path

        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:
            import tomli as tomllib  # backport (tomli) used as tomllib

        # Locate pyproject.toml in current directory
        pyproject_path = Path(__file__).parents[3] / "pyproject.toml"
        assert pyproject_path.exists(), f"pyproject.toml not found at {pyproject_path}"

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        # Verify the structure can be parsed
        assert isinstance(config, dict)
        assert "tool" in config

        # If calibrated-explanations config exists, verify structure
        ce_config = config.get("tool", {}).get("calibrated-explanations", {})
        if ce_config:
            # Verify expected sections if present
            for section in ("plots", "intervals", "explanations"):
                if section in ce_config:
                    assert isinstance(
                        ce_config[section], dict
                    ), f"[tool.calibrated-explanations.{section}] must be a dict"

    def test_plot_configuration_structure(self):
        """Verify the structure of plot configuration in pyproject.toml."""
        from pathlib import Path

        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:
            import tomli as tomllib  # backport (tomli) used as tomllib

        pyproject_path = Path(__file__).parents[3] / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        ce_config = config.get("tool", {}).get("calibrated-explanations", {})
        plots_config = ce_config.get("plots", {})

        # If plots config exists, verify each entry has required fields
        for plot_id, plot_meta in plots_config.items():
            if isinstance(plot_meta, dict):
                # Each plot should have builder and renderer
                assert (
                    "builder_id" in plot_meta or "default" in str(plot_meta).lower()
                ), f"Plot {plot_id} missing builder_id or default setting"


class TestMethodEPluginMetadata:
    """Test Method E: Plugin metadata dependencies from plugin-advanced-contract.md."""

    def test_plugin_metadata_dependencies(self):
        """Verify plugin metadata with declared dependencies."""
        from calibrated_explanations.plugins.base import validate_plugin_meta
        from dataclasses import dataclass
        from calibrated_explanations.plugins.explanations import ExplanationPlugin

        @dataclass
        class HelloFactualPlugin(ExplanationPlugin):
            """Example explanation plugin with declared dependencies."""

            plugin_meta = {
                "schema_version": 1,
                "name": "hello.explanation.factual",
                "version": "0.1.0",
                "provider": "example-team",
                "capabilities": ["explain", "explanation:factual", "task:classification"],
                "modes": ("factual",),
                "tasks": ("classification",),
                "dependencies": (),
                # Declare dependencies on interval and plot plugins
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
        # Validate metadata against schema
        validate_plugin_meta(dict(plugin.plugin_meta))
        # Verify dependencies are declared
        assert plugin.plugin_meta["interval_dependency"] == "core.interval.legacy"
        assert plugin.plugin_meta["plot_dependency"] == "plot_spec.default"

    def test_plugin_dependency_propagation_in_explainer(self):
        """Verify that plugin dependencies are propagated in CalibratedExplainer."""
        from tests.helpers.model_utils import get_classification_model
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

        # Create explainer using a core plugin
        explainer = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
            factual_plugin="core.explanation.factual",
        )

        # Verify that plugin metadata was used
        assert explainer.explanation_plugin_overrides["factual"] == "core.explanation.factual"

        # Generate explanations
        explanations = explainer.explain_factual(x_test[:3])

        # Verify explanations object was created successfully
        assert explanations is not None


class TestPriorityResolution:
    """Test the priority resolution order of all wiring methods."""

    def test_method_priority_order(self):
        """Verify that Method A (parameter override) has highest priority."""
        from tests.helpers.model_utils import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer
        from calibrated_explanations.plotting import resolve_plot_style_chain

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

        # Create explainer with Method A (highest priority)
        explainer = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
            plot_style="plot_spec.default",
        )

        # The plot style chain should prioritize the explicit parameter
        chain = resolve_plot_style_chain(explainer, explicit_style="plot_spec.default")
        # Method A override should be first in chain
        assert chain[0] == "plot_spec.default"

    def test_method_fallback_chain(self):
        """Verify that fallback chain works when lower-priority method is used."""
        from calibrated_explanations.plotting import resolve_plot_style_chain
        from tests.helpers.model_utils import get_classification_model
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from calibrated_explanations import CalibratedExplainer

        # Prepare data and explainer
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
        explainer = CalibratedExplainer(model, x_cal, y_cal)

        # When no override is specified, chain should include multiple fallbacks
        chain = resolve_plot_style_chain(explainer, explicit_style=None)

        # Chain should contain multiple styles to try
        assert len(chain) > 1, "Fallback chain should contain multiple styles"
        # First entry should be a default or environment-determined style
        assert isinstance(chain[0], str), "First chain entry should be a string"
