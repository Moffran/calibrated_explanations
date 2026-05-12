"""Memory retention lifecycle tests for CalibratedExplainer runtime state."""

from __future__ import annotations

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


class DummyHelper:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class DummyPluginManager:
    def __init__(self) -> None:
        self.clear_instances_calls = 0
        self.clear_identifiers_calls = 0
        self.clear_bridge_calls = 0
        self.explanation_contexts = {"factual": object()}

    def clear_explanation_plugin_instances(self) -> None:
        self.clear_instances_calls += 1

    def clear_explanation_plugin_identifiers(self) -> None:
        self.clear_identifiers_calls += 1

    def clear_bridge_monitors(self) -> None:
        self.clear_bridge_calls += 1


class DummyPerfExecutor:
    def __init__(self) -> None:
        self.exit_calls = 0

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.exit_calls += 1


def build_explainer_stub() -> tuple[CalibratedExplainer, DummyPluginManager]:
    explainer = object.__new__(CalibratedExplainer)
    explainer.latest_explanation = object()
    explainer.lime_helper = DummyHelper()
    explainer.shap_helper = DummyHelper()
    plugin_manager = DummyPluginManager()
    explainer.plugin_manager = plugin_manager
    explainer.perf_parallel = None
    return explainer, plugin_manager


def test_should_clear_latest_and_helper_plugin_state_when_reset_called() -> None:
    """should_clear_latest_explanation_and_helper_plugin_caches_when_reset."""
    explainer, plugin_manager = build_explainer_stub()

    explainer.reset()

    assert explainer.latest_explanation is None
    assert explainer.lime_helper.reset_calls == 1
    assert explainer.shap_helper.reset_calls == 1
    assert plugin_manager.clear_instances_calls == 1
    assert plugin_manager.clear_identifiers_calls == 1
    assert plugin_manager.clear_bridge_calls == 1
    assert plugin_manager.explanation_contexts == {}


def test_should_reset_runtime_state_and_close_pool_when_close_called() -> None:
    """should_call_reset_and_teardown_parallel_pool_when_close."""
    explainer, plugin_manager = build_explainer_stub()
    explainer.perf_parallel = DummyPerfExecutor()

    explainer.close()

    assert explainer.latest_explanation is None
    assert explainer.lime_helper.reset_calls == 1
    assert explainer.shap_helper.reset_calls == 1
    assert plugin_manager.clear_instances_calls == 1
    assert explainer.perf_parallel is None


def test_should_noop_close_when_parallel_pool_missing() -> None:
    """should_leave_state_consistent_when_close_without_pool."""
    explainer, _ = build_explainer_stub()

    explainer.close()

    assert explainer.latest_explanation is None
    assert explainer.perf_parallel is None
