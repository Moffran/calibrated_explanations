from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.core.exceptions import ConfigurationError, NotFittedError, ValidationError
from calibrated_explanations.plugins.intervals import IntervalCalibratorContext

sklearn = pytest.importorskip("sklearn")
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


def test_calibrated_explainer_rejects_invalid_condition_source() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = np.asarray([0, 1, 0, 1])
    learner = RandomForestClassifier(random_state=0, n_estimators=5).fit(x, y)

    with pytest.raises(ValidationError, match="condition_source"):
        CalibratedExplainer(
            learner,
            x,
            y,
            mode="classification",
            condition_source="invalid",
        )


def test_require_plugin_manager_raises_when_uninitialized() -> None:
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    setattr(explainer, "_plugin_manager", None)

    with pytest.raises(NotFittedError, match="PluginManager is not initialized"):
        CalibratedExplainer.require_plugin_manager(explainer)


def test_deepcopy_returns_existing_memo_entry() -> None:
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    existing = object()
    memo = {id(explainer): existing}
    assert CalibratedExplainer.__deepcopy__(explainer, memo) is existing


def test_initialize_pool_returns_when_executor_exists() -> None:
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    setattr(explainer, "_perf_parallel", object())
    assert CalibratedExplainer.initialize_pool(explainer) is None


def test_close_noop_when_pool_missing() -> None:
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    setattr(explainer, "_perf_parallel", None)
    assert CalibratedExplainer.close(explainer) is None


def test_context_manager_entry_and_exit_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    calls = []

    monkeypatch.setattr(
        explainer,
        "initialize_pool",
        lambda pool_at_init=False: calls.append(("init", pool_at_init)),
    )
    monkeypatch.setattr(explainer, "close", lambda: calls.append(("close", True)))

    entered = CalibratedExplainer.__enter__(explainer)
    assert entered is explainer
    CalibratedExplainer.__exit__(explainer, None, None, None)
    assert calls == [("init", True), ("close", True)]


def test_orchestrator_property_setters_delegate_to_manager() -> None:
    manager = SimpleNamespace(
        prediction_orchestrator="p0",
        explanation_orchestrator="e0",
        reject_orchestrator="r0",
    )
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    setattr(explainer, "_plugin_manager", manager)

    explainer.prediction_orchestrator = "p1"
    explainer.explanation_orchestrator = "e1"
    explainer.reject_orchestrator = "r1"
    assert manager.prediction_orchestrator == "p1"
    assert manager.explanation_orchestrator == "e1"
    assert manager.reject_orchestrator == "r1"

    del explainer.prediction_orchestrator
    del explainer.explanation_orchestrator
    del explainer.reject_orchestrator
    assert not hasattr(manager, "prediction_orchestrator")
    assert not hasattr(manager, "explanation_orchestrator")
    assert not hasattr(manager, "reject_orchestrator")


def test_build_instance_telemetry_payload_delegates_to_explanation_orchestrator() -> None:
    explanation_orchestrator = SimpleNamespace(
        build_instance_telemetry_payload=lambda explanations: {"count": len(explanations)}
    )
    manager = SimpleNamespace(explanation_orchestrator=explanation_orchestrator)
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    setattr(explainer, "_plugin_manager", manager)

    payload = explainer.build_instance_telemetry_payload([1, 2, 3])
    assert payload == {"count": 3}






def build_orchestrator_with_stub_explainer() -> PredictionOrchestrator:
    plugin_manager = SimpleNamespace(
        fast_interval_plugin_override=None,
        interval_plugin_override=None,
        interval_preferred_identifier={},
        interval_plugin_fallbacks={"default": (), "fast": ()},
        coerce_plugin_override=lambda override: None,
        interval_context_metadata={"default": {}, "fast": {}},
    )
    explainer = SimpleNamespace(
        initialized=True,
        interval_summary="regularized_mean",
        is_fast=lambda: False,
        mode="classification",
        is_multiclass=lambda: False,
        interval_learner=SimpleNamespace(
            predict_proba=lambda *_args, **_kwargs: (
                np.asarray([[0.2, 0.8]]),
                np.asarray([0.1]),
                np.asarray([0.9]),
            )
        ),
        plugin_manager=plugin_manager,
        instantiate_plugin=lambda p: p,
        bins=None,
        x_cal=np.asarray([[0.0], [1.0]]),
        y_cal=np.asarray([0, 1]),
        difficulty_estimator=None,
        categorical_features=(),
        num_features=1,
        noise_type="uniform",
        scale_factor=5,
        severity=1.0,
        seed=None,
        rng=None,
        predict_function=None,
    )
    return PredictionOrchestrator(explainer)


def test_prediction_orchestrator_predict_impl_requires_initialized() -> None:
    orch = build_orchestrator_with_stub_explainer()
    orch.explainer.initialized = False

    with pytest.raises(NotFittedError):
        orch.predict_impl(np.asarray([[0.0]]))


def test_prediction_orchestrator_resolve_interval_plugin_override_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch = build_orchestrator_with_stub_explainer()
    orch.explainer.plugin_manager.interval_plugin_override = "missing.plugin"
    orch.explainer.plugin_manager.interval_plugin_fallbacks["default"] = ("missing.plugin",)

    monkeypatch.setattr(
        "calibrated_explanations.core.prediction.orchestrator.ensure_builtin_plugins",
        lambda: None,
    )
    monkeypatch.setattr(
        "calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor",
        lambda _identifier: None,
    )
    monkeypatch.setattr(
        "calibrated_explanations.core.prediction.orchestrator.find_interval_plugin",
        lambda _identifier: None,
    )
    monkeypatch.setattr(
        "calibrated_explanations.core.prediction.orchestrator.find_interval_plugin_trusted",
        lambda _identifier: None,
    )
    monkeypatch.setattr(
        "calibrated_explanations.core.prediction.orchestrator.is_identifier_denied",
        lambda _identifier: False,
    )

    with pytest.raises(ConfigurationError, match="override failed"):
        orch.resolve_interval_plugin(fast=False)


def build_context(task: str = "classification") -> IntervalCalibratorContext:
    return IntervalCalibratorContext(
        learner=object(),
        calibration_splits=(),
        bins={},
        residuals={},
        difficulty={},
        metadata={"task": task},
        fast_flags={},
    )


def test_validate_interval_calibrator_fast_none_raises() -> None:
    orch = build_orchestrator_with_stub_explainer()
    with pytest.raises(ConfigurationError, match="returned None"):
        orch.validate_interval_calibrator(
            calibrator=None,
            context=build_context(),
            identifier="core.interval.fast",
            fast=True,
        )


def test_validate_interval_calibrator_fast_sequence_type_check() -> None:
    orch = build_orchestrator_with_stub_explainer()
    with pytest.raises(ConfigurationError, match="non-compliant for fast mode"):
        orch.validate_interval_calibrator(
            calibrator=[object()],
            context=build_context(),
            identifier="core.interval.fast",
            fast=True,
        )


def test_validate_interval_calibrator_untrusted_plugin_skips_protocol_checks() -> None:
    orch = build_orchestrator_with_stub_explainer()
    plugin = SimpleNamespace(plugin_meta={"trusted": False})

    # Untrusted plugins are allowed to return non-compliant payloads.
    orch.validate_interval_calibrator(
        calibrator=object(),
        context=build_context(),
        identifier="third.party.untrusted",
        fast=False,
        plugin=plugin,
    )
