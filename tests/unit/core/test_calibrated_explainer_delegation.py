from unittest.mock import MagicMock

import pytest
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


def make_stub_explainer_with_mock_pm():
    expl = object.__new__(CalibratedExplainer)
    # essential attrs
    expl.initialized = False
    expl.learner = MagicMock()

    # Initialize private attrs using setattr to avoid private member access checks
    # and to ensure getters don't fail.
    object.__setattr__(expl, "_lime_helper", "lime_stub")
    object.__setattr__(expl, "_shap_helper", "shap_stub")
    object.__setattr__(expl, "_perf_parallel", None)
    object.__setattr__(expl, "_preprocessor_metadata", None)

    # Mock plugin manager
    pm = MagicMock()
    expl.plugin_manager = pm
    return expl, pm


def test_removed_plugin_manager_aliases_fail_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CE_DEPRECATIONS", raising=False)
    expl, pm = make_stub_explainer_with_mock_pm()

    for name in (
        "interval_plugin_hints",
        "interval_plugin_fallbacks",
        "explanation_plugin_overrides",
        "interval_plugin_override",
        "fast_interval_plugin_override",
        "plot_style_override",
        "interval_preferred_identifier",
        "telemetry_interval_sources",
        "interval_plugin_identifiers",
        "interval_context_metadata",
    ):
        assert not hasattr(expl, name)

    expl.bridge_monitors = {}
    assert expl.bridge_monitors == {}

    expl.explanation_plugin_instances = {}
    assert pm.explanation_plugin_instances == {}

    expl.pyproject_explanations = {}
    assert pm.pyproject_explanations == {}

    expl.pyproject_intervals = {}
    assert pm.pyproject_intervals == {}

    expl.pyproject_plots = {}
    assert pm.pyproject_plots == {}


def test_internal_plugin_manager_aliases_delegate_to_manager():
    expl, pm = make_stub_explainer_with_mock_pm()

    alias_values = {
        "_interval_plugin_hints": {"factual": ("core",)},
        "_interval_plugin_fallbacks": {"default": ("legacy",)},
        "_interval_preferred_identifier": {"default": "legacy"},
        "_telemetry_interval_sources": {"default": "core"},
        "_interval_plugin_identifiers": {"default": "legacy"},
        "_interval_context_metadata": {"default": {"source": "test"}},
        "_explanation_plugin_overrides": {"factual": "core"},
        "_interval_plugin_override": "interval",
        "_fast_interval_plugin_override": "fast",
        "_plot_style_override": "plot",
        "_explanation_plugin_instances": {"core": object()},
    }

    for name, value in alias_values.items():
        setattr(expl, name, value)
        manager_attr = name[1:]
        assert getattr(pm, manager_attr) == value
        assert getattr(expl, name) == value

    for name in (
        "_interval_plugin_hints",
        "_interval_plugin_fallbacks",
        "_interval_preferred_identifier",
        "_telemetry_interval_sources",
        "_interval_plugin_identifiers",
        "_interval_context_metadata",
    ):
        delattr(expl, name)


def test_feature_names_internal_alias():
    expl, _ = make_stub_explainer_with_mock_pm()
    expl.feature_names = ["a", "b"]
    assert expl.feature_names_internal == ["a", "b"]
    expl.feature_names_internal = ["c"]
    assert expl.feature_names == ["c"]


def test_other_properties():
    expl, _ = make_stub_explainer_with_mock_pm()

    expl.last_explanation_mode = "factual"
    assert expl.last_explanation_mode == "factual"

    expl.fast = True
    assert expl.fast is True

    expl.feature_filter_per_instance_ignore = [1]
    assert expl.feature_filter_per_instance_ignore == [1]
    del expl.feature_filter_per_instance_ignore
    assert expl.feature_filter_per_instance_ignore is None

    expl.parallel_executor = "exec"
    assert expl.parallel_executor == "exec"

    expl.feature_filter_config = "conf"
    assert expl.feature_filter_config == "conf"

    expl.predict_bridge = "bridge"
    assert expl.predict_bridge == "bridge"

    expl.categorical_value_counts_cache = "count"
    assert expl.categorical_value_counts_cache == "count"

    expl.numeric_sorted_cache = "sort"
    assert expl.numeric_sorted_cache == "sort"

    expl.calibration_summary_shape = "shape"
    assert expl.calibration_summary_shape == "shape"

    expl.noise_type = "gaussian"
    assert expl.noise_type == "gaussian"

    expl.scale_factor = 1.0
    assert expl.scale_factor == 1.0

    expl.severity = 0.5
    assert expl.severity == 0.5
