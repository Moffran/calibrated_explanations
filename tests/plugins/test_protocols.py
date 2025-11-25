"""Structural checks for plugin protocols defined by ADR-013/ADR-014/ADR-015."""

from __future__ import annotations

import inspect
from typing import Any, Mapping

import pytest

from calibrated_explanations.plugins import (
    ClassificationIntervalCalibrator,
    ExplanationBatch,
    ExplanationContext,
    ExplanationPlugin,
    ExplanationRequest,
    IntervalCalibratorContext,
    IntervalCalibratorPlugin,
    PlotBuilder,
    PlotRenderContext,
    PlotRenderResult,
    PlotRenderer,
    PredictBridge,
    RegressionIntervalCalibrator,
)


class _DummyPredictBridge:
    def predict(self, x: Any, *, mode: str, task: str) -> Mapping[str, Any]:
        return {"mode": mode, "task": task, "x": x}

    def predict_interval(self, x: Any, *, task: str):  # pragma: no cover - protocol
        return (task, x)

    def predict_proba(self, x: Any):  # pragma: no cover - protocol
        return (x,)


def test_explanation_context_is_frozen() -> None:
    ctx = ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=("a", "b"),
        categorical_features=(1,),
        categorical_labels={1: {0: "no", 1: "yes"}},
        discretizer=object(),
        helper_handles={"grid": object()},
        predict_bridge=_DummyPredictBridge(),
        interval_settings={"plugin": "core.interval.legacy"},
        plot_settings={"style": "legacy"},
    )

    # Immutability contract: context must not allow field modification
    original_mode = ctx.mode
    with pytest.raises(Exception):
        ctx.mode = "alternative"  # type: ignore[misc]
    # Verify that the value did not change
    assert ctx.mode == original_mode


def test_explanation_request_is_frozen() -> None:
    req = ExplanationRequest(
        threshold=None,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=(0,),
        extras={"foo": "bar"},
    )
    # Immutability contract: request must not allow field modification
    original_extras = req.extras
    with pytest.raises(Exception):
        req.extras = {}  # type: ignore[misc]
    # Verify that the value did not change
    assert req.extras is original_extras


def test_explanation_batch_shape() -> None:
    batch = ExplanationBatch(
        container_cls=type("DummyContainer", (), {}),
        explanation_cls=type("DummyExplanation", (), {}),
        instances=({"payload": 1},),
        collection_metadata={"elapsed": 0.1},
    )
    assert batch.collection_metadata["elapsed"] == 0.1


def test_explanation_plugin_protocol_signatures() -> None:
    supports_sig = inspect.signature(ExplanationPlugin.supports_mode)
    initialize_sig = inspect.signature(ExplanationPlugin.initialize)
    batch_sig = inspect.signature(ExplanationPlugin.explain_batch)

    assert supports_sig.parameters["task"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tuple(supports_sig.parameters) == ("self", "mode", "task")
    assert tuple(initialize_sig.parameters) == ("self", "context")
    assert tuple(batch_sig.parameters) == ("self", "x", "request")


class _GoodExplanationPlugin:
    plugin_meta = {
        "name": "dummy",
        "schema_version": 1,
        "capabilities": ["explain"],
        "version": "0.0-test",
        "provider": "tests",
        "trusted": False,
        "trust": False,
    }

    def supports(self, model: Any) -> bool:
        return True

    def explain(self, model: Any, x: Any, **kwargs: Any) -> Any:
        return None

    def supports_mode(self, mode: str, *, task: str) -> bool:
        return True

    def initialize(self, context: ExplanationContext) -> None:  # pragma: no cover - protocol
        return None

    def explain_batch(self, x: Any, request: ExplanationRequest) -> ExplanationBatch:
        return ExplanationBatch(
            container_cls=type("Container", (), {}),
            explanation_cls=type("Explanation", (), {}),
            instances=(),
            collection_metadata={},
        )


class _BadExplanationPlugin:
    plugin_meta = {
        "name": "bad",
        "schema_version": 1,
        "capabilities": ["explain"],
        "version": "0.0-test",
        "provider": "tests",
        "trusted": False,
        "trust": False,
    }

    def supports_mode(self, mode: str, *, task: str) -> bool:  # pragma: no cover - protocol
        return True


def test_explanation_plugin_runtime_checks() -> None:
    assert isinstance(_GoodExplanationPlugin(), ExplanationPlugin)
    assert not isinstance(_BadExplanationPlugin(), ExplanationPlugin)


def test_predict_bridge_runtime_check() -> None:
    assert isinstance(_DummyPredictBridge(), PredictBridge)


def test_interval_context_is_frozen() -> None:
    ctx = IntervalCalibratorContext(
        learner=object(),
        calibration_splits=((),),
        bins={"values": ()},
        residuals={"values": ()},
        difficulty={"values": ()},
        metadata={"mode": "classification"},
        fast_flags={"fast": True},
    )
    # Immutability contract: context must not allow field modification
    original_learner = ctx.learner
    with pytest.raises(Exception):
        ctx.learner = None  # type: ignore[misc]
    # Verify that the value did not change
    assert ctx.learner is original_learner


class _GoodIntervalPlugin:
    plugin_meta = {
        "name": "interval",
        "schema_version": 1,
        "capabilities": ["interval"],
        "version": "0.0-test",
        "provider": "tests",
        "trusted": False,
        "trust": False,
    }

    def create(self, context: IntervalCalibratorContext, *, fast: bool = False):
        class _Calibrator:
            def predict_proba(
                self,
                x: Any,
                *,
                output_interval: bool = False,
                classes: Any | None = None,
                bins: Any | None = None,
            ) -> Any:
                return x

            def is_multiclass(self) -> bool:
                return False

            def is_mondrian(self) -> bool:
                return False

        return _Calibrator()


class _BadIntervalPlugin:
    plugin_meta = {
        "name": "interval",
        "schema_version": 1,
        "capabilities": ["interval"],
        "version": "0.0-test",
        "provider": "tests",
        "trusted": False,
        "trust": False,
    }


def test_interval_plugin_runtime_checks() -> None:
    good_calibrator = _GoodIntervalPlugin().create(
        IntervalCalibratorContext(
            learner=object(),
            calibration_splits=(),
            bins={},
            residuals={},
            difficulty={},
            metadata={},
            fast_flags={},
        )
    )
    assert isinstance(_GoodIntervalPlugin(), IntervalCalibratorPlugin)
    assert isinstance(good_calibrator, ClassificationIntervalCalibrator)
    assert not isinstance(_BadIntervalPlugin(), IntervalCalibratorPlugin)


class _RegressionCalibrator:
    def predict_proba(self, x: Any, *, output_interval: bool = False, classes=None, bins=None):
        return x

    def is_multiclass(self) -> bool:
        return False

    def is_mondrian(self) -> bool:
        return False

    def predict_probability(self, x: Any) -> Any:  # pragma: no cover - protocol
        return x

    def predict_uncertainty(self, x: Any) -> Any:  # pragma: no cover - protocol
        return x

    def pre_fit_for_probabilistic(self, x: Any, y: Any) -> None:  # pragma: no cover - protocol
        return None

    def compute_proba_cal(self, x: Any, y: Any, *, weights: Any | None = None) -> Any:
        return x

    def insert_calibration(self, x: Any, y: Any, *, warm_start: bool = False) -> None:
        return None


def test_regression_calibrator_protocol() -> None:
    assert isinstance(_RegressionCalibrator(), RegressionIntervalCalibrator)


def test_plot_context_is_frozen() -> None:
    ctx = PlotRenderContext(
        explanation={"id": 1},
        instance_metadata={"index": 0},
        style="legacy",
        intent={"kind": "bar"},
        show=False,
        path=None,
        save_ext="png",
        options={"dpi": 72},
    )
    # Immutability contract: context must not allow field modification
    original_style = ctx.style
    with pytest.raises(Exception):
        ctx.style = "alt"  # type: ignore[misc]
    # Verify that the value did not change
    assert ctx.style == original_style


class _GoodPlotBuilder:
    plugin_meta = {
        "name": "plot",
        "schema_version": 1,
        "capabilities": ["plot"],
        "version": "0.0-test",
        "provider": "tests",
        "trusted": False,
        "trust": False,
    }

    def build(self, context: PlotRenderContext) -> Mapping[str, Any]:
        return {"style": context.style}


class _GoodPlotRenderer:
    plugin_meta = {
        "name": "plot",
        "schema_version": 1,
        "capabilities": ["render"],
        "version": "0.0-test",
        "provider": "tests",
        "trusted": False,
        "trust": False,
    }

    def render(
        self, artifact: Mapping[str, Any], *, context: PlotRenderContext
    ) -> PlotRenderResult:
        return PlotRenderResult(artifact=artifact, saved_paths=("/tmp/out.png",))


class _BadPlotBuilder:
    plugin_meta = {
        "name": "plot",
        "schema_version": 1,
        "capabilities": ["plot"],
        "version": "0.0-test",
        "provider": "tests",
        "trusted": False,
        "trust": False,
    }


class _BadPlotRenderer:
    plugin_meta = {
        "name": "plot",
        "schema_version": 1,
        "capabilities": ["render"],
        "version": "0.0-test",
        "provider": "tests",
        "trusted": False,
        "trust": False,
    }


def test_plot_protocol_runtime_checks() -> None:
    assert isinstance(_GoodPlotBuilder(), PlotBuilder)
    assert isinstance(_GoodPlotRenderer(), PlotRenderer)
    assert not isinstance(_BadPlotBuilder(), PlotBuilder)
    assert not isinstance(_BadPlotRenderer(), PlotRenderer)


def test_validate_explanation_batch_invalid_batch_type() -> None:
    from calibrated_explanations.plugins.explanations import validate_explanation_batch

    with pytest.raises(
        TypeError, match="explanation plugins must return an ExplanationBatch instance"
    ):
        validate_explanation_batch("not a batch")


def test_validate_explanation_batch_invalid_container_cls() -> None:
    from calibrated_explanations.plugins.explanations import (
        ExplanationBatch,
        validate_explanation_batch,
    )
    from calibrated_explanations.explanations.explanation import CalibratedExplanation

    batch = ExplanationBatch(
        container_cls="not a class",
        explanation_cls=CalibratedExplanation,
        instances=[],
        collection_metadata={},
    )
    with pytest.raises(TypeError, match="batch.container_cls must be a class"):
        validate_explanation_batch(batch)


def test_validate_explanation_batch_invalid_container_inheritance() -> None:
    from calibrated_explanations.plugins.explanations import (
        ExplanationBatch,
        validate_explanation_batch,
    )
    from calibrated_explanations.explanations.explanation import CalibratedExplanation

    batch = ExplanationBatch(
        container_cls=object,  # doesn't inherit from CalibratedExplanations
        explanation_cls=CalibratedExplanation,
        instances=[],
        collection_metadata={},
    )
    with pytest.raises(
        TypeError, match="batch.container_cls must inherit from CalibratedExplanations"
    ):
        validate_explanation_batch(batch)


def test_validate_explanation_batch_invalid_explanation_cls() -> None:
    from calibrated_explanations.plugins.explanations import (
        ExplanationBatch,
        validate_explanation_batch,
    )
    from calibrated_explanations.explanations.explanations import CalibratedExplanations

    batch = ExplanationBatch(
        container_cls=CalibratedExplanations,
        explanation_cls="not a class",
        instances=[],
        collection_metadata={},
    )
    with pytest.raises(TypeError, match="batch.explanation_cls must be a class"):
        validate_explanation_batch(batch)


def test_validate_explanation_batch_invalid_instances_type() -> None:
    from calibrated_explanations.plugins.explanations import (
        ExplanationBatch,
        validate_explanation_batch,
    )
    from calibrated_explanations.explanations.explanations import CalibratedExplanations
    from calibrated_explanations.explanations.explanation import CalibratedExplanation

    batch = ExplanationBatch(
        container_cls=CalibratedExplanations,
        explanation_cls=CalibratedExplanation,
        instances="not a sequence",
        collection_metadata={},
    )
    with pytest.raises(TypeError, match="batch.instances must be a sequence of mappings"):
        validate_explanation_batch(batch)


def test_validate_explanation_batch_invalid_instance_mapping() -> None:
    from calibrated_explanations.plugins.explanations import (
        ExplanationBatch,
        validate_explanation_batch,
    )
    from calibrated_explanations.explanations.explanations import CalibratedExplanations
    from calibrated_explanations.explanations.explanation import CalibratedExplanation

    batch = ExplanationBatch(
        container_cls=CalibratedExplanations,
        explanation_cls=CalibratedExplanation,
        instances=["not a mapping"],
        collection_metadata={},
    )
    with pytest.raises(
        TypeError, match="batch.instances\\[0\\] must be a mapping describing the instance"
    ):
        validate_explanation_batch(batch)


def test_validate_explanation_batch_invalid_metadata_type() -> None:
    from calibrated_explanations.plugins.explanations import (
        ExplanationBatch,
        validate_explanation_batch,
    )
    from calibrated_explanations.explanations.explanations import CalibratedExplanations
    from calibrated_explanations.explanations.explanation import CalibratedExplanation

    batch = ExplanationBatch(
        container_cls=CalibratedExplanations,
        explanation_cls=CalibratedExplanation,
        instances=[],
        collection_metadata="not a mapping",
    )
    with pytest.raises(TypeError, match="batch.collection_metadata must be a mutable mapping"):
        validate_explanation_batch(batch)


def test_validate_explanation_batch_mode_mismatch() -> None:
    from calibrated_explanations.plugins.explanations import (
        ExplanationBatch,
        validate_explanation_batch,
    )
    from calibrated_explanations.explanations.explanations import CalibratedExplanations
    from calibrated_explanations.explanations.explanation import CalibratedExplanation

    batch = ExplanationBatch(
        container_cls=CalibratedExplanations,
        explanation_cls=CalibratedExplanation,
        instances=[],
        collection_metadata={"mode": "alternative"},
    )
    with pytest.raises(
        ValueError,
        match="ExplanationBatch metadata reports mode 'alternative' but runtime expected 'factual'",
    ):
        validate_explanation_batch(batch, expected_mode="factual")


def test_validate_explanation_batch_task_mismatch() -> None:
    from calibrated_explanations.plugins.explanations import (
        ExplanationBatch,
        validate_explanation_batch,
    )
    from calibrated_explanations.explanations.explanations import CalibratedExplanations
    from calibrated_explanations.explanations.explanation import CalibratedExplanation

    batch = ExplanationBatch(
        container_cls=CalibratedExplanations,
        explanation_cls=CalibratedExplanation,
        instances=[],
        collection_metadata={"task": "regression"},
    )
    with pytest.raises(
        ValueError,
        match="ExplanationBatch metadata reports task 'regression' but runtime expected 'classification'",
    ):
        validate_explanation_batch(batch, expected_task="classification")


def test_validate_explanation_batch_invalid_container_instance() -> None:
    from calibrated_explanations.plugins.explanations import (
        ExplanationBatch,
        validate_explanation_batch,
    )
    from calibrated_explanations.explanations.explanations import CalibratedExplanations
    from calibrated_explanations.explanations.explanation import CalibratedExplanation

    batch = ExplanationBatch(
        container_cls=CalibratedExplanations,
        explanation_cls=CalibratedExplanation,
        instances=[],
        collection_metadata={"container": "not a CalibratedExplanations instance"},
    )
    with pytest.raises(
        TypeError, match="ExplanationBatch metadata 'container' has unexpected type"
    ):
        validate_explanation_batch(batch)
