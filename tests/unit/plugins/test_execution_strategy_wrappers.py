"""Unit tests for execution strategy wrapper explanation plugins.

Tests verify that:
- Wrapper plugins are properly registered
- Metadata declares correct fallback chains
- Wrapper plugins delegate to execution plugins
- Fallback to legacy implementation works when execution fails
- Internal FAST-based feature filtering integrates without breaking behaviour
"""

from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.parallel import ParallelConfig, ParallelExecutor
from calibrated_explanations.plugins.explanations import ExplanationContext, ExplanationRequest
from calibrated_explanations.plugins import registry
from calibrated_explanations.plugins.builtins import (
    # FeatureParallelAlternativeExplanationPlugin,
    # FeatureParallelExplanationPlugin,
    InstanceParallelAlternativeExplanationPlugin,
    InstanceParallelExplanationPlugin,
    SequentialAlternativeExplanationPlugin,
    SequentialExplanationPlugin,
)
from calibrated_explanations.core.explain._feature_filter import FeatureFilterConfig


class TestExecutionStrategyPluginRegistration:
    """Test that execution strategy wrapper plugins are registered correctly."""

    def test_sequential_factual_plugin_registered(self):
        """Sequential factual plugin should be registered."""
        descriptor = registry.find_explanation_descriptor("core.explanation.factual.sequential")
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.factual.sequential"
        assert "factual" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted

    def test_feature_parallel_factual_plugin_registered(self):
        """Feature-parallel factual plugin should be registered."""
        descriptor = registry.find_explanation_descriptor(
            "core.explanation.factual.feature_parallel"
        )
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.factual.feature_parallel"
        assert "factual" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted

    def test_instance_parallel_factual_plugin_registered(self):
        """Instance-parallel factual plugin should be registered."""
        descriptor = registry.find_explanation_descriptor(
            "core.explanation.factual.instance_parallel"
        )
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.factual.instance_parallel"
        assert "factual" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted

    def test_sequential_alternative_plugin_registered(self):
        """Sequential alternative plugin should be registered."""
        descriptor = registry.find_explanation_descriptor("core.explanation.alternative.sequential")
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.alternative.sequential"
        assert "alternative" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted

    def test_feature_parallel_alternative_plugin_registered(self):
        """Feature-parallel alternative plugin should be registered."""
        descriptor = registry.find_explanation_descriptor(
            "core.explanation.alternative.feature_parallel"
        )
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.alternative.feature_parallel"
        assert "alternative" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted

    def test_instance_parallel_alternative_plugin_registered(self):
        """Instance-parallel alternative plugin should be registered."""
        descriptor = registry.find_explanation_descriptor(
            "core.explanation.alternative.instance_parallel"
        )
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.alternative.instance_parallel"
        assert "alternative" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted


class TestExecutionStrategyPluginMetadata:
    """Test that wrapper plugins declare correct metadata and fallback chains."""

    def test_sequential_factual_metadata_has_fallback(self):
        """Sequential factual plugin should declare fallback to legacy."""
        plugin = SequentialExplanationPlugin()
        assert "fallbacks" in plugin.plugin_meta
        assert "core.explanation.factual" in plugin.plugin_meta["fallbacks"]

    # def test_feature_parallel_factual_fallback_chain(self):
    #     """Feature-parallel factual should have sequential and legacy fallbacks."""
    #     plugin = FeatureParallelExplanationPlugin()
    #     fallbacks = plugin.plugin_meta["fallbacks"]
    #     assert "core.explanation.factual.sequential" in fallbacks
    #     assert "core.explanation.factual" in fallbacks

    def test_instance_parallel_factual_fallback_chain(self):
        """Instance-parallel factual should have sequential and legacy fallbacks."""
        plugin = InstanceParallelExplanationPlugin()
        fallbacks = plugin.plugin_meta["fallbacks"]
        # Feature-parallel is deprecated and removed from fallback chain
        assert "core.explanation.factual.sequential" in fallbacks
        assert "core.explanation.factual" in fallbacks

    def test_sequential_alternative_metadata_has_fallback(self):
        """Sequential alternative plugin should declare fallback to legacy."""
        plugin = SequentialAlternativeExplanationPlugin()
        assert "fallbacks" in plugin.plugin_meta
        assert "core.explanation.alternative" in plugin.plugin_meta["fallbacks"]

    # def test_feature_parallel_alternative_fallback_chain(self):
    #     """Feature-parallel alternative should have sequential and legacy fallbacks."""
    #     plugin = FeatureParallelAlternativeExplanationPlugin()
    #     fallbacks = plugin.plugin_meta["fallbacks"]
    #     assert "core.explanation.alternative.sequential" in fallbacks
    #     assert "core.explanation.alternative" in fallbacks

    def test_instance_parallel_alternative_fallback_chain(self):
        """Instance-parallel alternative should have sequential and legacy fallbacks."""
        plugin = InstanceParallelAlternativeExplanationPlugin()
        fallbacks = plugin.plugin_meta["fallbacks"]
        # Feature-parallel is deprecated and removed from fallback chain
        assert "core.explanation.alternative.sequential" in fallbacks
        assert "core.explanation.alternative" in fallbacks


class TestExecutionStrategyPluginAttributes:
    """Test that wrapper plugins have correct attributes and capabilities."""

    def test_sequential_factual_plugin_attributes(self):
        """Sequential factual plugin should have correct base attributes."""
        plugin = SequentialExplanationPlugin()
        assert plugin._mode == "factual"
        assert plugin._explanation_attr == "explain_factual"
        assert plugin.execution_plugin_class is not None

    # def test_feature_parallel_factual_plugin_attributes(self):
    #     """Feature-parallel factual plugin should have correct base attributes."""
    #     plugin = FeatureParallelExplanationPlugin()
    #     assert plugin._mode == "factual"
    #     assert plugin._explanation_attr == "explain_factual"
    #     assert plugin.execution_plugin_class is not None

    def test_instance_parallel_factual_plugin_attributes(self):
        """Instance-parallel factual plugin should have correct base attributes."""
        plugin = InstanceParallelExplanationPlugin()
        assert plugin._mode == "factual"
        assert plugin._explanation_attr == "explain_factual"
        assert plugin.execution_plugin_class is not None

    def test_sequential_alternative_plugin_attributes(self):
        """Sequential alternative plugin should have correct base attributes."""
        plugin = SequentialAlternativeExplanationPlugin()
        assert plugin._mode == "alternative"
        assert plugin._explanation_attr == "explore_alternatives"
        assert plugin.execution_plugin_class is not None

    # def test_feature_parallel_alternative_plugin_attributes(self):
    #     """Feature-parallel alternative plugin should have correct base attributes."""
    #     plugin = FeatureParallelAlternativeExplanationPlugin()
    #     assert plugin._mode == "alternative"
    #     assert plugin._explanation_attr == "explore_alternatives"
    #     assert plugin.execution_plugin_class is not None

    def test_instance_parallel_alternative_plugin_attributes(self):
        """Instance-parallel alternative plugin should have correct base attributes."""
        plugin = InstanceParallelAlternativeExplanationPlugin()
        assert plugin._mode == "alternative"
        assert plugin._explanation_attr == "explore_alternatives"
        assert plugin.execution_plugin_class is not None


class TestPluginSupportsMode:
    """Test that plugins support the correct modes and tasks."""

    def test_sequential_factual_supports_factual_mode(self):
        """Sequential factual plugin should support factual mode."""
        plugin = SequentialExplanationPlugin()
        assert plugin.supports_mode("factual", task="classification")
        assert plugin.supports_mode("factual", task="regression")
        assert not plugin.supports_mode("alternative", task="classification")

    def test_sequential_alternative_supports_alternative_mode(self):
        """Sequential alternative plugin should support alternative mode."""
        plugin = SequentialAlternativeExplanationPlugin()
        assert plugin.supports_mode("alternative", task="classification")
        assert plugin.supports_mode("alternative", task="regression")
        assert not plugin.supports_mode("factual", task="classification")

    # def test_feature_parallel_factual_supports_factual_mode(self):
    #     """Feature-parallel factual plugin should support factual mode."""
    #     plugin = FeatureParallelExplanationPlugin()
    #     assert plugin.supports_mode("factual", task="classification")
    #     assert not plugin.supports_mode("alternative", task="classification")

    def test_instance_parallel_factual_supports_factual_mode(self):
        """Instance-parallel factual plugin should support factual mode."""
        plugin = InstanceParallelExplanationPlugin()
        assert plugin.supports_mode("factual", task="classification")
        assert not plugin.supports_mode("alternative", task="classification")


class TestExecutionPluginClassConfiguration:
    """Test that execution plugin classes are correctly configured."""

    def test_sequential_loads_sequential_executor_class(self):
        """Sequential wrapper should load SequentialExplainExecutor."""
        plugin = SequentialExplanationPlugin()
        assert plugin.execution_plugin_class is not None
        # Verify the class name matches
        assert "Sequential" in plugin.execution_plugin_class.__name__

    # def test_feature_parallel_loads_feature_executor_class(self):
    #     """Feature-parallel wrapper should load FeatureParallelExplainExecutor."""
    #     plugin = FeatureParallelExplanationPlugin()
    #     assert plugin.execution_plugin_class is not None
    #     # Verify the class name matches
    #     assert "FeatureParallel" in plugin.execution_plugin_class.__name__

    def test_instance_parallel_loads_instance_executor_class(self):
        """Instance-parallel wrapper should load InstanceParallelExplainExecutor."""
        plugin = InstanceParallelExplanationPlugin()
        assert plugin.execution_plugin_class is not None
        # Verify the class name matches
        assert "InstanceParallel" in plugin.execution_plugin_class.__name__


def inc(x: int) -> int:
    return x + 1


def test_should_enter_parallel_executor_once_during_explain_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import calibrated_explanations.parallel.parallel as parallel_mod

    created_pools: list[int | None] = []

    class CountingThreadPool:
        def __init__(self, max_workers: int | None = None):
            created_pools.append(max_workers)
            self.max_workers_count = max_workers or 1

        def map(self, fn, items, chunksize: int = 1):  # noqa: ARG002
            return list(map(fn, items))

        def shutdown(self, wait: bool = True, cancel_futures: bool = False):  # noqa: ARG002
            return None

    monkeypatch.setattr(parallel_mod, "ThreadPoolExecutor", CountingThreadPool)

    cfg = ParallelConfig(enabled=True, strategy="threads", max_workers=2, min_batch_size=1)
    executor = ParallelExecutor(cfg)

    class DummyExecutionPlugin:
        def execute(self, explain_request, explain_config, explainer):  # noqa: ARG002
            explain_config.executor.map(inc, [1, 2], work_items=2)
            explain_config.executor.map(inc, [1, 2], work_items=2)

            class DummyCollection:
                mode = "factual"
                explanations = []

            return DummyCollection()

    def fake_build_explain_execution_plan(_explainer, _x, _request):  # noqa: ARG001
        from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest
        from unittest.mock import MagicMock

        runtime = MagicMock()

        def enter_runtime():
            # Simulate the runtime entering the executor
            if executor:
                executor.__enter__()
            return runtime

        runtime.__enter__.side_effect = enter_runtime
        runtime.__exit__.return_value = None

        return (
            ExplainRequest(
                x=None,
                threshold=None,
                low_high_percentiles=(0.05, 0.95),
                bins=None,
                features_to_ignore=(),
                use_plugin=False,
                skip_instance_parallel=False,
            ),
            ExplainConfig(
                executor=executor,
                granularity="feature",
                min_instances_for_parallel=1,
                chunk_size=1,
                num_features=1,
                features_to_ignore_default=(),
                categorical_features=(),
                feature_values={},
                mode="classification",
            ),
            runtime,
        )

    monkeypatch.setattr(
        "calibrated_explanations.core.explain.parallel_runtime.build_explain_execution_plan",
        fake_build_explain_execution_plan,
    )

    plugin = InstanceParallelExplanationPlugin()
    plugin.execution_plugin_class = DummyExecutionPlugin  # type: ignore[assignment]

    context = ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=("f0",),
        categorical_features=(),
        categorical_labels={},
        discretizer=object(),
        helper_handles={"explainer": object()},
        predict_bridge=object(),  # not used by execution wrapper
        interval_settings={},
        plot_settings={},
    )
    plugin.initialize(context)

    request = ExplanationRequest(
        threshold=None,
        low_high_percentiles=(0.05, 0.95),
        bins=None,
        features_to_ignore=(),
        extras={},
    )
    plugin.explain_batch("x", request)

    assert len(created_pools) == 1


def test_fast_feature_filter_updates_features_to_ignore(monkeypatch: pytest.MonkeyPatch) -> None:
    """FAST-based feature filter should narrow features_to_ignore per batch."""

    # Clear env-based overrides for deterministic behaviour
    monkeypatch.delenv("CE_FEATURE_FILTER", raising=False)

    # Build a minimal CalibratedExplanations container with dummy per-instance weights
    num_features = 4
    x_fast = np.zeros((2, num_features))

    class DummyExplainerForFrozen:
        def __init__(self, n_features: int):
            self.x_cal = np.zeros((1, n_features))
            self.y_cal = np.zeros(1)
            self.num_features = n_features

    fast_collection = CalibratedExplanations(
        DummyExplainerForFrozen(num_features), x_fast, None, None
    )

    class DummyFastExplanation:
        def __init__(self, weights: np.ndarray):
            self.feature_weights = {"predict": np.asarray(weights, dtype=float)}

    # Instance 0 strongly prefers feature 0; instance 1 prefers feature 1
    fast_collection.explanations = [
        DummyFastExplanation(np.array([10.0, 0.0, 0.0, 0.0])),
        DummyFastExplanation(np.array([0.0, 5.0, 0.3, 1.0])),
    ]

    # Dummy explainer exposing only the attributes used by the wrapper
    class DummyExplainer:
        def __init__(self):
            # No baseline ignore; all filtering comes from FAST + request.
            self.features_to_ignore = np.array([], dtype=int)
            self.num_features = num_features
            self._feature_filter_config = FeatureFilterConfig(enabled=True, per_instance_top_k=1)
            from unittest.mock import MagicMock

            self.plugin_manager = MagicMock()
            self.plugin_manager.explanation_orchestrator = self  # self-invoke for FAST

        @property
        def feature_filter_config(self):
            """Expose the feature filter configuration."""
            return self._feature_filter_config

        # Orchestrator-like interface used by the wrapper
        def invoke(
            self,
            mode: str,
            x: object,
            threshold: object,
            low_high_percentiles: object,
            bins: object,
            features_to_ignore: object,
            *,
            extras: object | None = None,
        ) -> CalibratedExplanations:
            assert mode == "fast"
            # For this unit test we ignore the arguments and return the prebuilt FAST collection
            return fast_collection

    recorded_request: dict[str, tuple[int, ...]] = {}

    def fake_build_explain_execution_plan(_explainer, _x, req: ExplanationRequest):  # noqa: ARG001
        """Capture the features_to_ignore after filtering and return a stub plan."""
        from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest
        from unittest.mock import MagicMock

        runtime = MagicMock()
        runtime.__enter__.return_value = runtime
        runtime.__exit__.return_value = None

        recorded_request["features_to_ignore"] = tuple(req.features_to_ignore or ())
        return (
            ExplainRequest(
                x=None,
                threshold=None,
                low_high_percentiles=(0.05, 0.95),
                bins=None,
                features_to_ignore=np.asarray(req.features_to_ignore or (), dtype=int),
                use_plugin=False,
                skip_instance_parallel=False,
            ),
            ExplainConfig(
                executor=None,
                granularity="feature",
                min_instances_for_parallel=1,
                chunk_size=1,
                num_features=num_features,
                features_to_ignore_default=(),
                categorical_features=(),
                feature_values={},
                mode="classification",
            ),
            runtime,
        )

    monkeypatch.setattr(
        "calibrated_explanations.core.explain.parallel_runtime.build_explain_execution_plan",
        fake_build_explain_execution_plan,
    )

    explainer = DummyExplainer()
    plugin = InstanceParallelExplanationPlugin()

    context = ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=("f0", "f1", "f2", "f3"),
        categorical_features=(),
        categorical_labels={},
        discretizer=None,
        helper_handles={"explainer": explainer},
        predict_bridge=object(),
        interval_settings={},
        plot_settings={},
    )
    plugin.initialize(context)

    # User has explicitly requested to ignore feature 2
    request = ExplanationRequest(
        threshold=None,
        low_high_percentiles=(0.05, 0.95),
        bins=None,
        features_to_ignore=(2,),
        extras={},
    )

    class NoopExecutionPlugin:
        def execute(self, *args, **kwargs):  # noqa: ARG002
            class DummyCollection:
                mode = "factual"
                explanations = []

            return DummyCollection()

    plugin.execution_plugin_class = NoopExecutionPlugin  # type: ignore[assignment]

    plugin.explain_batch("x", request)

    # With per_instance_top_k=1 and the chosen weights, instance 0 keeps
    # feature 0 and instance 1 keeps feature 1. Feature 2 is ignored by the
    # user request, and FAST decides that feature 3 is never among the
    # per-instance top-k set, so it can be globally ignored for this batch.
    assert recorded_request["features_to_ignore"] == (2, 3)


def test_should_warn_and_fallback_to_legacy_when_execution_plugin_raises(
    enable_fallbacks,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapper plugins should warn and use legacy path on execution failure.

    This test explicitly validates fallback behavior when the execution plugin fails.
    """

    class DummyLegacyCollection:
        mode = "factual"
        explanations: list[object] = []

    class DummyExplainer:
        def __init__(self) -> None:
            self.called_kwargs: dict[str, object] | None = None

        def explain_factual(self, x, **kwargs):  # noqa: ANN001
            self.called_kwargs = kwargs
            return DummyLegacyCollection()

    def fake_build_explain_execution_plan(_explainer, _x, _request):  # noqa: ARG001
        from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest
        from unittest.mock import MagicMock

        runtime = MagicMock()
        runtime.__enter__.return_value = runtime
        runtime.__exit__.return_value = None

        return (
            ExplainRequest(
                x=None,
                threshold=None,
                low_high_percentiles=(0.05, 0.95),
                bins=None,
                features_to_ignore=np.array([], dtype=int),
                use_plugin=False,
                skip_instance_parallel=False,
            ),
            ExplainConfig(
                executor=None,
                granularity="feature",
                min_instances_for_parallel=1,
                chunk_size=1,
                num_features=1,
                features_to_ignore_default=(),
                categorical_features=(),
                feature_values={},
                mode="classification",
            ),
            runtime,
        )

    monkeypatch.setattr(
        "calibrated_explanations.core.explain.parallel_runtime.build_explain_execution_plan",
        fake_build_explain_execution_plan,
    )

    class RaisingExecutionPlugin:
        def execute(self, *args, **kwargs):  # noqa: ARG002
            raise RuntimeError("boom")

    plugin = InstanceParallelExplanationPlugin()
    plugin.execution_plugin_class = RaisingExecutionPlugin  # type: ignore[assignment]

    explainer = DummyExplainer()
    context = ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=("f0",),
        categorical_features=(),
        categorical_labels={},
        discretizer=None,
        helper_handles={"explainer": explainer},
        predict_bridge=object(),
        interval_settings={},
        plot_settings={},
    )
    plugin.initialize(context)

    request = ExplanationRequest(
        threshold=None,
        low_high_percentiles=(0.05, 0.95),
        bins=None,
        features_to_ignore=(),
        extras={},
    )

    with pytest.warns(UserWarning, match="falling back to legacy"):
        plugin.explain_batch("x", request)

    assert explainer.called_kwargs is not None
    assert explainer.called_kwargs.get("_use_plugin") is False
