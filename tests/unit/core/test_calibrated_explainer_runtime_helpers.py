"""Unit tests covering CalibratedExplainer runtime helper utilities."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from calibrated_explanations.core.prediction import orchestrator as prediction_orchestrator_module
from calibrated_explanations.core.calibrated_explainer import (
    CalibratedExplainer,
)
from calibrated_explanations.utils.exceptions import (
    ConfigurationError,
    DataShapeError,
    ValidationError,
)


def stub_explainer(explainer_factory, mode: str = "classification") -> CalibratedExplainer:
    """Construct a fully initialized explainer instance for unit tests."""

    explainer = explainer_factory(mode=mode)
    explainer.bins = None
    explainer._plot_style_override = None
    explainer.interval_plugin_override = None
    explainer._fast_interval_plugin_override = None
    explainer.interval_plugin_hints = {}
    explainer.interval_plugin_fallbacks = {"default": (), "fast": ()}
    explainer.interval_preferred_identifier = {"default": None, "fast": None}
    explainer.telemetry_interval_sources = {"default": None, "fast": None}
    explainer._interval_context_metadata = {"default": {}, "fast": {}}
    explainer.explanation_plugin_overrides = {
        key: None for key in ("factual", "alternative", "fast")
    }
    explainer._pyproject_explanations = {}
    explainer._pyproject_intervals = {}
    explainer._pyproject_plots = {}
    explainer._plot_style_override = None
    explainer._explanation_plugin_fallbacks = {}
    return explainer


def test_oob_exceptions_are_propagated(explainer_factory):
    class OOBFailingLearner:
        def __init__(self) -> None:
            self.fitted_ = True

        def fit(self, *_args, **_kwargs):  # pragma: no cover - unused
            return self

        def predict_proba(self, *_args, **_kwargs):  # pragma: no cover - unused
            return np.zeros((1, 2))

        @property
        def oob_decision_function_(self):
            raise RuntimeError("oob failure")

    with pytest.raises(RuntimeError, match="oob failure"):
        explainer_factory(learner=OOBFailingLearner(), oob=True)


def test_categorical_features_default_to_label_keys(explainer_factory):
    labels = {0: {0: "a"}, 2: {0: "b"}}
    explainer = explainer_factory(categorical_features=None, categorical_labels=labels)

    assert explainer.categorical_features == [0, 2]


def test_require_plugin_manager_raises_when_missing(explainer_factory):
    from calibrated_explanations.utils.exceptions import NotFittedError

    explainer = stub_explainer(explainer_factory)
    explainer.plugin_manager = None

    with pytest.raises(NotFittedError, match="PluginManager is not initialized"):
        explainer._require_plugin_manager()


def test_build_chains_fall_back_without_plugin_manager(explainer_factory):
    explainer = stub_explainer(explainer_factory)
    delattr(explainer, "plugin_manager")

    assert explainer._build_explanation_chain("factual") == ()
    assert explainer._build_interval_chain(fast=False) == ()
    assert explainer.build_plot_style_chain() == ()


def test_build_instance_telemetry_payload_delegates(explainer_factory):
    explainer = stub_explainer(explainer_factory)
    sentinel = object()

    class StubOrchestrator:
        def __init__(self) -> None:
            self.seen = None

        def _build_instance_telemetry_payload(self, explanations):
            self.seen = explanations
            return sentinel

    explainer.plugin_manager = types.SimpleNamespace(_explanation_orchestrator=StubOrchestrator())

    assert explainer._build_instance_telemetry_payload("payload") is sentinel
    assert explainer.plugin_manager._explanation_orchestrator.seen == "payload"


def test_oob_path_reraises_source_error(monkeypatch):
    class BareLearner:
        """Learner exposing only fitted_ flag to satisfy check_is_fitted."""

        def __init__(self) -> None:
            self.fitted_ = True

        def fit(self, *_args, **_kwargs):
            return self

    x_cal = np.asarray([[0.0], [1.0]])
    y_cal = np.asarray([0, 1])

    with pytest.raises(AttributeError):
        CalibratedExplainer(BareLearner(), x_cal, y_cal, oob=True)


def test_categorical_features_inferred_from_labels(explainer_factory):
    categorical_labels = {1: {0: "zero"}}
    explainer = explainer_factory(categorical_features=None, categorical_labels=categorical_labels)

    assert explainer.categorical_features == [1]


def test_instance_telemetry_payload_delegates(explainer_factory):
    explainer = stub_explainer(explainer_factory)
    sentinel = object()

    class StubOrchestrator:
        def _build_instance_telemetry_payload(self, explanations):
            return (explanations, sentinel)

    explainer.plugin_manager = types.SimpleNamespace(_explanation_orchestrator=StubOrchestrator())

    assert explainer._build_instance_telemetry_payload("payload") == ("payload", sentinel)

    assert explainer._pyproject_intervals is None
    assert explainer._pyproject_plots is None


def test_plugin_manager_deleters_forward_to_manager(explainer_factory):
    explainer = stub_explainer(explainer_factory)

    class DummyManager:
        def __init__(self) -> None:
            self._interval_plugin_hints = {"fast": ()}
            self._interval_plugin_fallbacks = {"fast": ()}
            self._interval_plugin_identifiers = {"fast": None}
            self._telemetry_interval_sources = {"fast": None}
            self._interval_preferred_identifier = {"fast": None}
            self._interval_context_metadata = {"fast": {}}

    explainer.plugin_manager = DummyManager()

    del explainer.interval_plugin_hints
    del explainer._interval_plugin_fallbacks
    del explainer._interval_plugin_identifiers
    del explainer._telemetry_interval_sources
    del explainer._interval_preferred_identifier
    del explainer._interval_context_metadata

    for attr in (
        "_interval_plugin_hints",
        "_interval_plugin_fallbacks",
        "_interval_plugin_identifiers",
        "_telemetry_interval_sources",
        "_interval_preferred_identifier",
        "_interval_context_metadata",
    ):
        assert not hasattr(explainer.plugin_manager, attr)


def test_plugin_manager_deleters_remove_backing_fields(explainer_factory):
    explainer = stub_explainer(explainer_factory)

    class StubManager:
        def __init__(self) -> None:
            self._interval_plugin_hints = {"default": ("a",)}
            self._interval_plugin_fallbacks = {"default": ("b",)}
            self._interval_plugin_identifiers = {"default": "id"}
            self._telemetry_interval_sources = {"default": "src"}
            self._interval_preferred_identifier = {"default": "pref"}
            self._interval_context_metadata = {"default": {"meta": 1}}
            self._plot_style_chain = ("style",)

    explainer.plugin_manager = StubManager()

    del explainer.interval_plugin_hints
    del explainer._interval_plugin_fallbacks
    del explainer._interval_plugin_identifiers
    del explainer._telemetry_interval_sources
    del explainer._interval_preferred_identifier
    del explainer._interval_context_metadata
    explainer._plot_style_chain = ("new_style",)

    manager = explainer.plugin_manager
    assert not hasattr(manager, "_interval_plugin_hints")
    assert not hasattr(manager, "_interval_plugin_fallbacks")
    assert not hasattr(manager, "_interval_plugin_identifiers")
    assert not hasattr(manager, "_telemetry_interval_sources")
    assert not hasattr(manager, "_interval_preferred_identifier")
    assert not hasattr(manager, "_interval_context_metadata")
    assert manager._plot_style_chain == ("new_style",)


def test_fast_interval_initializer_delegates_to_registry(monkeypatch, explainer_factory):
    explainer = explainer_factory(mode="regression")
    sentinel = object()

    class StubIntervalRegistry:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def initialize_for_fast_explainer(self) -> None:
            self.calls.append(sentinel)

    registry = StubIntervalRegistry()

    class StubPredictionOrchestrator:
        def __init__(self, interval_registry: StubIntervalRegistry) -> None:
            self._interval_registry = interval_registry
            self._prediction_orchestrator = self

    stub_manager = StubPredictionOrchestrator(registry)
    monkeypatch.setattr(explainer, "_require_plugin_manager", lambda: stub_manager)
    explainer._CalibratedExplainer__initialize_interval_learner_for_fast_explainer()

    assert registry.calls == [sentinel]


def test_call_delegates_to_explain(explainer_factory, monkeypatch):
    explainer = explainer_factory()
    captured = {}

    def fake_explain_mock(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "explained"

    monkeypatch.setattr(explainer, "_explain", fake_explain_mock)

    assert explainer("x", threshold=0.5) == "explained"
    assert captured["args"] == ("x", 0.5, (5, 95), None, None)
    assert captured["kwargs"] == {"_use_plugin": True, "_skip_instance_parallel": False}


def test_explain_uses_plugin_orchestrator(explainer_factory):
    explainer = explainer_factory()
    sentinel = object()

    class StubExplanationOrchestrator:
        def __init__(self) -> None:
            self.calls = []

        def invoke_factual(
            self, x, threshold, low_high_percentiles, bins, features_to_ignore, **kwargs
        ):
            self.calls.append(
                ((x, threshold, low_high_percentiles, bins, features_to_ignore), kwargs)
            )
            return sentinel

    orchestrator = StubExplanationOrchestrator()
    explainer._plugin_manager = types.SimpleNamespace(_explanation_orchestrator=orchestrator)
    explainer._infer_explanation_mode = lambda: "factual"

    result = explainer.explain_factual("x", threshold=0.2, bins="bins", features_to_ignore=[1])

    assert result is sentinel
    # The orchestrator should receive the factual invocation payload
    assert orchestrator.calls[0][0][0] == "x"
    assert orchestrator.calls[0][1].get("_use_plugin", True) is True


def test_reinitialize_validates_bins_length(explainer_factory):
    explainer = explainer_factory()
    bins = np.asarray([0, 1])
    explainer.bins = bins

    with pytest.raises(DataShapeError):
        explainer.reinitialize(explainer.learner, explainer.x_cal, explainer.y_cal, bins=bins[:1])


def test_explain_counterfactual_delegates(explainer_factory):
    explainer = explainer_factory()
    sentinel = object()
    explainer.explore_alternatives = lambda *args, **kwargs: (args, kwargs, sentinel)

    from tests.helpers.deprecation import deprecations_error_enabled

    if deprecations_error_enabled():
        with pytest.raises(DeprecationWarning):
            explainer.explain_counterfactual("x", threshold=1.0)
    else:
        with pytest.warns(DeprecationWarning):
            result = explainer.explain_counterfactual("x", threshold=1.0)
        assert result[2] is sentinel


def test_set_discretizer_defaults_feature_ignores(explainer_factory, monkeypatch):
    explainer = explainer_factory()
    explainer.categorical_features = np.asarray([0])
    explainer.features_to_ignore = np.asarray([1])

    called = {}

    class FakeDiscretizer:
        to_discretize: tuple[int, ...] = ()
        mins: dict[int, list[float]] = {}
        means: dict[int, list[float]] = {}

    def fake_instantiate_mock(
        discretizer,
        x_cal,
        not_to_discretize,
        feature_names,
        y_cal,
        seed,
        old_discretizer,
        **_kwargs,
    ):
        called["not_to_discretize"] = not_to_discretize
        return FakeDiscretizer()

    monkeypatch.setattr(
        "calibrated_explanations.core.discretizer_config.instantiate_discretizer",
        fake_instantiate_mock,
    )

    explainer.set_discretizer("entropy", features_to_ignore=np.asarray([0]))

    assert isinstance(explainer.discretizer, FakeDiscretizer)
    assert 1 in called["not_to_discretize"]


def test_set_discretizer_prediction_condition_source(explainer_factory, monkeypatch):
    class DummyTTLCache:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    monkeypatch.setitem(sys.modules, "cachetools", types.SimpleNamespace(TTLCache=DummyTTLCache))

    explainer = explainer_factory()
    explainer.condition_source = "prediction"

    captured: dict[str, Any] = {}

    def instantiate(
        discretizer, x_cal, not_to_discretize, feature_names, y_cal, seed, old, **kwargs
    ):
        captured["labels"] = kwargs.get("condition_labels")
        captured["source"] = kwargs.get("condition_source")
        return "disc"

    def setup(self, discretizer, x_cal, num_features):
        return {}, np.zeros_like(x_cal)

    monkeypatch.setattr(
        "calibrated_explanations.core.discretizer_config.instantiate_discretizer", instantiate
    )
    monkeypatch.setattr(
        "calibrated_explanations.core.discretizer_config.setup_discretized_data", setup
    )

    explainer.set_discretizer("entropy")

    expected = explainer.predict(explainer.x_cal, calibrated=True, uq_interval=False)
    if isinstance(expected, tuple):
        expected = expected[0]

    assert captured["source"] == "prediction"
    np.testing.assert_array_equal(captured["labels"], expected)


def test_predict_proba_uncalibrated_interval(explainer_factory):
    explainer = explainer_factory(mode="classification")

    proba, (low, high) = explainer.predict_proba(
        explainer.x_cal, uq_interval=True, calibrated=False
    )

    assert proba.shape[1] == 2
    np.testing.assert_array_equal(low, high)


def test_ensure_interval_runtime_state_populates_defaults(explainer_factory):
    explainer = explainer_factory()
    explainer._interval_plugin_hints = {}
    explainer._interval_plugin_fallbacks = {}
    explainer._interval_plugin_identifiers = {}
    explainer._telemetry_interval_sources = {}
    explainer._interval_preferred_identifier = {}
    explainer._interval_context_metadata = {}
    explainer._ensure_interval_runtime_state()

    assert explainer._interval_plugin_hints == {}
    assert explainer._interval_plugin_fallbacks == {}
    assert explainer._interval_plugin_identifiers == {"default": None, "fast": None}
    assert explainer._telemetry_interval_sources == {"default": None, "fast": None}
    assert explainer._interval_preferred_identifier == {"default": None, "fast": None}
    assert explainer._interval_context_metadata == {"default": {}, "fast": {}}


def test_verbose_repr_includes_metadata(explainer_factory):
    explainer = explainer_factory()
    explainer.verbose = True
    explainer._feature_names = ["f1", "f2"]  # noqa: SLF001 - testing repr content
    explainer.categorical_features = [0]
    explainer.categorical_labels = {0: {0: "no", 1: "yes"}}
    explainer.class_labels = {0: "neg", 1: "pos"}

    result = repr(explainer)

    assert "CalibratedExplainer" in result
    assert "feature_names=['f1', 'f2']" in result
    assert "categorical_features=[0]" in result
    assert "categorical_labels={0: {0: 'no', 1: 'yes'}}" in result
    assert "class_labels={0: 'neg', 1: 'pos'}" in result


def test_instantiate_plugin_handles_multiple_paths(monkeypatch, explainer_factory):
    explainer = stub_explainer(explainer_factory)

    class CallableWithMeta:
        plugin_meta = {}

        def __call__(self):  # pragma: no cover - guard against accidental call
            raise AssertionError("should not be invoked")

    callable_with_meta = CallableWithMeta()
    assert explainer._instantiate_plugin(callable_with_meta) is callable_with_meta

    class SimplePlugin:
        def __init__(self) -> None:
            self.token = object()

    prototype = SimplePlugin()
    clone = explainer._instantiate_plugin(prototype)
    assert isinstance(clone, SimplePlugin)
    assert clone is not prototype

    class BrokenPlugin:
        def __init__(self) -> None:
            raise RuntimeError("boom")

    broken = BrokenPlugin.__new__(BrokenPlugin)
    sentinel = object()
    # Patch copy.deepcopy in the orchestrator module where it's imported
    from calibrated_explanations.core.explain import orchestrator as explain_orch

    monkeypatch.setattr(explain_orch.copy, "deepcopy", lambda value: sentinel)
    assert explainer._instantiate_plugin(broken) is sentinel

    # Test fallback when deepcopy itself fails
    def raising_deepcopy(value):
        raise RuntimeError("fail")

    monkeypatch.setattr(explain_orch.copy, "deepcopy", raising_deepcopy)
    assert explainer._instantiate_plugin(broken) is broken


def test_reinitialize_bins_validation_and_updates(monkeypatch, explainer_factory):
    explainer = explainer_factory()
    explainer.bins = None

    with pytest.raises(ValidationError, match="Cannot mix calibration instances"):
        explainer.reinitialize(
            explainer.learner, xs=np.ones((1, 2)), ys=np.ones(1), bins=np.array([0])
        )

    explainer.bins = np.array([0, 1])

    with pytest.raises(DataShapeError, match="length of bins"):
        explainer.reinitialize(
            explainer.learner,
            xs=np.ones((1, 2)),
            ys=np.ones(1),
            bins=np.array([0, 1, 2]),
        )

    sentinel = object()

    def update_interval_mock(self, xs, ys, bins=None):
        self.marker = (xs.shape, ys.shape, None if bins is None else bins.shape)
        return sentinel

    monkeypatch.setattr(
        "calibrated_explanations.calibration.interval_learner.update_interval_learner",
        update_interval_mock,
    )

    explainer.reinitialize(
        explainer.learner,
        xs=np.ones((2, 2)),
        ys=np.ones(2),
        bins=np.array([2, 3]),
    )

    assert explainer.bins.tolist() == [0, 1, 2, 3]
    assert explainer.marker == ((2, 2), (2,), (2,))


def test_resolve_interval_plugin_handles_denied_and_success(monkeypatch, explainer_factory):
    explainer = stub_explainer(explainer_factory, mode="regression")
    explainer._interval_plugin_fallbacks = {"default": ("denied.plugin", "ok.plugin"), "fast": ()}
    explainer._instantiate_plugin = lambda plugin: plugin
    explainer._check_interval_runtime_metadata = lambda metadata, **_: None

    monkeypatch.setattr(prediction_orchestrator_module, "ensure_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        prediction_orchestrator_module,
        "is_identifier_denied",
        lambda identifier: identifier == "denied.plugin",
    )

    descriptor = types.SimpleNamespace(
        metadata={
            "schema_version": 1,
            "modes": ("regression",),
            "capabilities": ("interval:regression",),
            "fast_compatible": True,
        },
        trusted=True,
        plugin=types.SimpleNamespace(plugin_meta={"name": "ok.plugin"}),
    )

    monkeypatch.setattr(
        prediction_orchestrator_module,
        "find_interval_descriptor",
        lambda identifier: descriptor if identifier == "ok.plugin" else None,
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin", lambda identifier: None
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin_trusted", lambda identifier: None
    )

    plugin, identifier = explainer._resolve_interval_plugin(fast=False)

    assert identifier == "ok.plugin"
    assert plugin is descriptor.plugin


def test_resolve_interval_plugin_denied_override_raises(monkeypatch, explainer_factory):
    explainer = stub_explainer(explainer_factory, mode="regression")
    explainer._interval_plugin_override = "denied.plugin"
    explainer._interval_plugin_fallbacks = {"default": ("denied.plugin",), "fast": ()}

    monkeypatch.setattr(prediction_orchestrator_module, "ensure_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        prediction_orchestrator_module,
        "is_identifier_denied",
        lambda identifier: identifier == "denied.plugin",
    )

    with pytest.raises(ConfigurationError) as excinfo:
        explainer._resolve_interval_plugin(fast=False)

    assert "denied via CE_DENY_PLUGIN" in str(excinfo.value)


def test_build_interval_context_enriches_metadata(explainer_factory):
    explainer = stub_explainer(explainer_factory, mode="regression")
    explainer.x_cal = np.asarray([[1.0, 2.0]])
    explainer.y_cal = np.asarray([1.5])
    explainer._X_cal = explainer.x_cal
    explainer.bins = np.asarray([0])
    explainer.difficulty_estimator = "difficulty"
    explainer._interval_context_metadata = {"default": {"preexisting": {"value": 1}}, "fast": {}}
    explainer._CalibratedExplainer__noise_type = "gaussian"
    explainer.categorical_features = [1]
    explainer.learner = object()

    context = explainer._build_interval_context(fast=False, metadata={"extra": 2})

    assert context.learner is explainer.learner
    assert context.calibration_splits[0] == (explainer.x_cal, explainer.y_cal)
    assert context.bins["calibration"].shape == (1,)
    assert context.difficulty["estimator"] == "difficulty"
    assert context.fast_flags == {"fast": False}
    assert context.metadata["preexisting"] == {"value": 1}
    assert context.metadata["extra"] == 2
    assert context.metadata["task"] == "regression"
    assert context.metadata["mode"] == "regression"
    assert context.metadata["categorical_features"] == (1,)
    assert context.metadata["num_features"] == 2
    assert context.metadata["noise_config"]["noise_type"] == "gaussian"


def test_get_calibration_summaries_caches_results(explainer_factory):
    explainer = stub_explainer(explainer_factory)
    explainer.x_cal = np.asarray([[0, "a"], [1, "b"], [0, "a"]], dtype=object)
    explainer._X_cal = explainer.x_cal
    explainer.categorical_features = [1]
    explainer._categorical_value_counts_cache = None
    explainer._numeric_sorted_cache = None
    explainer._calibration_summary_shape = None

    counts, numeric = explainer._get_calibration_summaries()

    assert counts[1]["a"] == 2
    assert counts[1]["b"] == 1
    assert np.array_equal(numeric[0], np.array([0, 0, 1]))

    counts_cached, numeric_cached = explainer._get_calibration_summaries()

    assert counts_cached is counts
    assert numeric_cached is numeric


class RaisingInterval:
    def predict_uncertainty(self, *_, **__):
        raise RuntimeError("boom")

    def predict_probability(self, *_, **__):
        raise RuntimeError("boom")


def test_predict_impl_returns_degraded_arrays_when_suppressed(explainer_factory):
    explainer = stub_explainer(explainer_factory, mode="regression")
    explainer._CalibratedExplainer__initialized = True
    explainer._CalibratedExplainer__fast = False
    explainer.interval_learner = RaisingInterval()
    explainer.suppress_crepes_errors = True
    explainer._X_cal = np.zeros((1, 1))

    x = np.ones((2, 1))

    predict, low, high, classes = explainer._predict_impl(x)

    assert np.all(predict == 0)
    assert np.all(low == 0)
    assert np.all(high == 0)
    assert classes is None

    threshold = [0.5, 0.5]
    predict_t, low_t, high_t, classes_t = explainer._predict_impl(x, threshold=threshold)

    assert np.all(predict_t == 0)
    assert np.all(low_t == 0)
    assert np.all(high_t == 0)
    assert classes_t is None


def test_infer_explanation_mode_detects_entropy_discretizer(explainer_factory):
    """_infer_explanation_mode should detect alternative mode when EntropyDiscretizer is set."""
    from calibrated_explanations.utils import EntropyDiscretizer

    explainer = stub_explainer(explainer_factory)
    x_cal = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_cal = np.asarray([0, 1, 0])
    explainer.discretizer = EntropyDiscretizer(
        x_cal,
        categorical_features={},
        feature_names=["f0", "f1"],
        labels=y_cal,
        random_state=0,
    )

    assert explainer._infer_explanation_mode() == "alternative"


def test_infer_explanation_mode_detects_regressor_discretizer(explainer_factory):
    """_infer_explanation_mode should detect alternative mode when RegressorDiscretizer is set."""
    from calibrated_explanations.utils import RegressorDiscretizer

    explainer = stub_explainer(explainer_factory, mode="regression")
    x_cal = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_cal = np.asarray([1.0, 2.0, 3.0])
    explainer.discretizer = RegressorDiscretizer(
        x_cal,
        categorical_features={},
        feature_names=["f0", "f1"],
        labels=y_cal,
        random_state=0,
    )

    assert explainer._infer_explanation_mode() == "alternative"


def test_infer_explanation_mode_defaults_to_factual(explainer_factory):
    """_infer_explanation_mode should return 'factual' for None or other discretizers."""
    explainer = stub_explainer(explainer_factory)
    assert explainer._infer_explanation_mode() == "factual"


def test_preprocessor_metadata_round_trip(explainer_factory):
    """preprocessor_metadata property should preserve and return metadata dict."""
    explainer = stub_explainer(explainer_factory)

    metadata = {"scaling": "StandardScaler", "features": 10}
    explainer.set_preprocessor_metadata(metadata)

    assert explainer.preprocessor_metadata == metadata
    assert explainer.preprocessor_metadata is not metadata  # Should be a copy


def test_preprocessor_metadata_none_handling(explainer_factory):
    """set_preprocessor_metadata(None) should clear metadata."""
    explainer = stub_explainer(explainer_factory)
    explainer.set_preprocessor_metadata({"key": "value"})
    explainer.set_preprocessor_metadata(None)

    assert explainer.preprocessor_metadata is None


def test_prediction_orchestrator_raises_when_missing_attribute(explainer_factory):
    """_prediction_orchestrator should raise when manager lacks the attribute."""
    explainer = stub_explainer(explainer_factory)

    class BrokenManager:
        pass

    explainer._plugin_manager = BrokenManager()

    with pytest.raises(AttributeError, match="PluginManager has no '_prediction_orchestrator'"):
        _ = explainer._prediction_orchestrator


def test_explanation_orchestrator_raises_when_missing_attribute(explainer_factory):
    """_explanation_orchestrator should raise when manager lacks the attribute."""
    explainer = stub_explainer(explainer_factory)

    class BrokenManager:
        pass

    explainer._plugin_manager = BrokenManager()

    with pytest.raises(AttributeError, match="PluginManager has no '_explanation_orchestrator'"):
        _ = explainer._explanation_orchestrator


def test_reject_orchestrator_raises_when_missing_attribute(explainer_factory):
    """_reject_orchestrator should raise when manager lacks the attribute."""
    explainer = stub_explainer(explainer_factory)

    class BrokenManager:
        pass

    explainer._plugin_manager = BrokenManager()

    with pytest.raises(AttributeError, match="PluginManager has no '_reject_orchestrator'"):
        _ = explainer._reject_orchestrator


def test_explanation_plugin_instances_empty_without_manager(explainer_factory):
    """_explanation_plugin_instances should return empty dict without manager."""
    explainer = stub_explainer(explainer_factory)
    delattr(explainer, "_plugin_manager")

    assert explainer._explanation_plugin_instances == {}


def test_explanation_plugin_identifiers_empty_without_manager(explainer_factory):
    """_explanation_plugin_identifiers should return empty dict without manager."""
    explainer = stub_explainer(explainer_factory)
    delattr(explainer, "_plugin_manager")

    assert explainer._explanation_plugin_identifiers == {}


def test_interval_plugin_hints_property_operations(explainer_factory):
    """_interval_plugin_hints property should get/set via manager."""
    explainer = stub_explainer(explainer_factory)

    # Create a mock manager
    class MockManager:
        def __init__(self):
            self._interval_plugin_hints = {"default": ("hint1", "hint2")}

    explainer._plugin_manager = MockManager()

    assert explainer._interval_plugin_hints == {"default": ("hint1", "hint2")}

    explainer._interval_plugin_hints = {"fast": ("fast_hint",)}
    assert explainer._plugin_manager._interval_plugin_hints == {"fast": ("fast_hint",)}


def test_explanation_plugin_fallbacks_property_operations(explainer_factory):
    """_explanation_plugin_fallbacks property should get/set via manager."""
    explainer = stub_explainer(explainer_factory)

    class MockManager:
        def __init__(self):
            self._explanation_plugin_fallbacks = {"factual": ("a", "b")}

    explainer._plugin_manager = MockManager()

    assert explainer._explanation_plugin_fallbacks == {"factual": ("a", "b")}

    explainer._explanation_plugin_fallbacks = {"alternative": ("c",)}
    assert explainer._plugin_manager._explanation_plugin_fallbacks == {"alternative": ("c",)}


def test_plot_plugin_fallbacks_property_operations(explainer_factory):
    """_plot_plugin_fallbacks property should get/set via manager."""
    explainer = stub_explainer(explainer_factory)

    class MockManager:
        def __init__(self):
            self._plot_plugin_fallbacks = {"default": ("plot_a",)}

    explainer._plugin_manager = MockManager()

    assert explainer._plot_plugin_fallbacks == {"default": ("plot_a",)}

    explainer._plot_plugin_fallbacks = {"alt": ("plot_b",)}
    assert explainer._plugin_manager._plot_plugin_fallbacks == {"alt": ("plot_b",)}


def test_explain_counterfactual_deprecates_and_delegates(monkeypatch, explainer_factory):
    explainer = explainer_factory()
    sentinel = object()

    monkeypatch.setattr("calibrated_explanations.utils.deprecate", lambda *args, **kwargs: None)
    explainer.explore_alternatives = lambda *args, **kwargs: sentinel

    assert (
        explainer.explain_counterfactual(
            np.ones((1, 2)),
            threshold=None,
            low_high_percentiles=(5, 95),
            bins=None,
            features_to_ignore=None,
        )
        is sentinel
    )


def test_call_and_explain_delegate_with_plugin(monkeypatch, explainer_factory):
    explainer = explainer_factory()
    sentinel = object()
    recorded = {}

    explainer._infer_explanation_mode = lambda: "factual"

    class StubOrchestrator:
        def invoke(
            self, mode, x, threshold, low_high_percentiles, bins, features_to_ignore, *, extras
        ):
            recorded["extras"] = extras
            recorded["mode"] = mode
            recorded["payload"] = (x, threshold, low_high_percentiles, bins, features_to_ignore)
            return sentinel

    explainer._plugin_manager = types.SimpleNamespace(_explanation_orchestrator=StubOrchestrator())

    assert explainer(np.zeros((1, 2))) is sentinel
    assert recorded["extras"] == {"mode": "factual", "_skip_instance_parallel": False}
    assert recorded["mode"] == "factual"


def test_legacy_explain_path(monkeypatch, explainer_factory):
    explainer = explainer_factory()
    sentinel = object()

    def legacy_explain_mock(self, *args, **kwargs):
        return (self, args, kwargs, sentinel)

    monkeypatch.setattr(
        "calibrated_explanations.core.explain._legacy_explain.explain", legacy_explain_mock
    )

    result = explainer.explain_factual(np.zeros((1, 2)), _use_plugin=False)

    assert result[0] is explainer
    assert result[-1] is sentinel


def test_is_multiclass_and_set_mode_helpers(explainer_factory):
    explainer = explainer_factory()
    explainer._CalibratedExplainer__set_mode("classification", initialize=False)
    explainer.num_classes = 3

    assert explainer.is_multiclass() is True
    explainer._CalibratedExplainer__set_mode("regression", initialize=False)
    assert explainer.is_multiclass() is False

    with pytest.raises(ValidationError):
        explainer._CalibratedExplainer__set_mode("unknown", initialize=False)


def test_set_discretizer_defaults_and_populates(monkeypatch, explainer_factory):
    explainer = explainer_factory()
    explainer.categorical_features = [1]
    explainer.features_to_ignore = [0]
    explainer.discretizer = "existing"

    monkeypatch.setattr(
        "calibrated_explanations.core.discretizer_config.validate_discretizer_choice",
        lambda choice, mode: f"validated:{choice}:{mode}",
    )

    def instantiate(choice, x_cal, not_to_discretize, feature_names, y_cal, seed, old, **kwargs):
        assert kwargs.get("condition_source") == "observed"
        return f"disc:{choice}:{tuple(not_to_discretize)}:{old}"

    def setup(self, discretizer, x_cal, num_features):
        return {0: {"values": (1,), "frequencies": (2,)}}, np.ones_like(x_cal)

    monkeypatch.setattr(
        "calibrated_explanations.core.discretizer_config.instantiate_discretizer", instantiate
    )
    monkeypatch.setattr(
        "calibrated_explanations.core.discretizer_config.setup_discretized_data", setup
    )

    explainer.set_discretizer("auto")

    assert explainer.feature_values[0] == (1,)
    assert explainer.feature_frequencies[0] == (2,)
    assert "validated" in explainer.discretizer


class ListInterval:
    def __init__(self, proba=None, low=None, high=None):
        self.proba = proba
        self.low = low
        self.high = high

    def predict_probability(self, x, *_args, **_kwargs):
        return self.proba, self.low, self.high, None

    def predict_proba(self, x, *_args, **_kwargs):
        return self.proba, self.low, self.high, None


def test_predict_variants_cover_branches(explainer_factory):
    explainer = explainer_factory()
    explainer._CalibratedExplainer__initialized = True

    probs, intervals = explainer.predict_proba(np.zeros((2, 2)), calibrated=False, uq_interval=True)
    assert probs.shape[0] == 2
    assert intervals[0].shape == (2,)

    explainer._CalibratedExplainer__set_mode("regression", initialize=False)
    explainer._CalibratedExplainer__initialized = True
    explainer.interval_learner = [
        ListInterval(proba=np.array([0.2, 0.4]), low=np.zeros(2), high=np.ones(2))
    ]
    preds = explainer.predict_proba(np.zeros((2, 2)), calibrated=True, uq_interval=False)
    assert preds.shape == (2, 2)

    explainer._CalibratedExplainer__set_mode("classification", initialize=False)
    explainer._CalibratedExplainer__initialized = True
    explainer.num_classes = 3
    explainer.interval_learner = ListInterval(
        proba=np.full((2, 3), 1 / 3), low=np.zeros((2, 3)), high=np.ones((2, 3))
    )
    probs_multi = explainer.predict_proba(np.zeros((2, 2)), calibrated=True, uq_interval=True)
    assert probs_multi[0].shape == (2, 3)

    explainer.num_classes = 2
    explainer.interval_learner = [
        types.SimpleNamespace(
            predict_proba=lambda *_args, **_kwargs: (
                np.full((2, 2), 0.5),
                np.zeros((2, 2)),
                np.ones((2, 2)),
            )
        )
    ]
    probs_binary = explainer.predict_proba(np.zeros((2, 2)), calibrated=True, uq_interval=True)
    assert probs_binary[0].shape == (2, 2)


def test_calibrated_confusion_matrix_rejects_regression(explainer_factory):
    explainer = explainer_factory(mode="regression")

    with pytest.raises(ValidationError):
        explainer.calibrated_confusion_matrix()


def test_predict_calibration_uses_predict_function(monkeypatch, explainer_factory):
    explainer = explainer_factory()
    captured = {}

    def predict(x):
        captured["x"] = x
        return np.array([42])

    explainer.predict_function = predict

    assert explainer.predict_calibration().tolist() == [42]
    assert np.array_equal(captured["x"], explainer.x_cal)
