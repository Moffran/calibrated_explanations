import copy
import pytest
from unittest.mock import MagicMock
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


def test_deepcopy_safe():
    expl, pm = make_stub_explainer_with_mock_pm()
    expl.some_attr = [1, 2]

    # Explicitly verify we are using picklable stubs for helpers mostly
    # But expl.learner is a MagicMock, which fails deepcopy usually unless patched?
    # Actually copy.deepcopy on MagicMock works, but pickle doesn't.

    expl_copy = copy.deepcopy(expl)

    assert expl_copy is not expl
    assert expl_copy.some_attr == expl.some_attr
    assert expl_copy.some_attr is not expl.some_attr


def test_getstate_logic():
    expl, pm = make_stub_explainer_with_mock_pm()
    expl.mode = "classification"
    expl.perf_cache = "dummy_cache"

    # Verify __getstate__ logic directly to avoid pickling mocks
    state = expl.__getstate__()

    assert state.get("mode") == "classification"
    # perf_cache should be None in state (excluded)
    assert state.get("perf_cache") is None
    # _perf_parallel should be None in state (excluded)
    assert state.get("_perf_parallel") is None


def test_property_delegators_to_plugin_manager():
    expl, pm = make_stub_explainer_with_mock_pm()

    # interval_plugin_hints
    _ = expl.interval_plugin_hints
    pm.interval_plugin_hints.__class__
    expl.interval_plugin_hints = {"a": ("b",)}
    assert pm.interval_plugin_hints == {"a": ("b",)}
    del expl.interval_plugin_hints

    # interval_plugin_fallbacks
    _ = expl.interval_plugin_fallbacks
    expl.interval_plugin_fallbacks = {}
    del expl.interval_plugin_fallbacks

    # explanation_plugin_overrides
    _ = expl.explanation_plugin_overrides
    expl.explanation_plugin_overrides = {}

    # interval_plugin_override
    _ = expl.interval_plugin_override
    expl.interval_plugin_override = "foo"

    # fast_interval_plugin_override
    _ = expl.fast_interval_plugin_override
    expl.fast_interval_plugin_override = "bar"

    # plot_style_override
    _ = expl.plot_style_override
    expl.plot_style_override = "style"

    # interval_preferred_identifier
    _ = expl.interval_preferred_identifier
    expl.interval_preferred_identifier = {}
    del expl.interval_preferred_identifier

    # telemetry_interval_sources
    _ = expl.telemetry_interval_sources
    expl.telemetry_interval_sources = {}
    del expl.telemetry_interval_sources

    # interval_plugin_identifiers
    _ = expl.interval_plugin_identifiers
    expl.interval_plugin_identifiers = {}
    del expl.interval_plugin_identifiers

    # interval_context_metadata
    _ = expl.interval_context_metadata
    expl.interval_context_metadata = {}
    del expl.interval_context_metadata

    # bridge_monitors
    _ = expl.bridge_monitors
    expl.bridge_monitors = {}

    # explanation_plugin_instances
    _ = expl.explanation_plugin_instances
    expl.explanation_plugin_instances = {}

    # pyproject_explanations/intervals/plots
    _ = expl.pyproject_explanations
    expl.pyproject_explanations = {}

    _ = expl.pyproject_intervals
    expl.pyproject_intervals = {}

    _ = expl.pyproject_plots
    expl.pyproject_plots = {}


def test_feature_names_internal_alias():
    expl, _ = make_stub_explainer_with_mock_pm()
    expl.feature_names = ["a", "b"]
    assert expl.feature_names_internal == ["a", "b"]
    expl.feature_names_internal = ["c"]
    assert expl.feature_names == ["c"]


def test_helper_properties():
    expl, _ = make_stub_explainer_with_mock_pm()

    # initialized to "lime_stub"
    assert expl.lime_helper == "lime_stub"
    # Deleter uses del self._lime_helper.
    del expl.lime_helper
    # Expect AttributeError
    with pytest.raises(AttributeError):
        _ = expl.lime_helper

    assert expl.shap_helper == "shap_stub"
    del expl.shap_helper
    with pytest.raises(AttributeError):
        _ = expl.shap_helper


def test_get_sigma_test_property():
    expl, _ = make_stub_explainer_with_mock_pm()
    expl.get_sigma_test = True
    assert expl.get_sigma_test is True


def test_other_properties():
    expl, _ = make_stub_explainer_with_mock_pm()

    expl.last_explanation_mode = "factual"
    assert expl.last_explanation_mode == "factual"

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
