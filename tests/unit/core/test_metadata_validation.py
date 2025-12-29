import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.plugins import EXPLANATION_PROTOCOL_VERSION


# Stub for explainer
class ExplainerStub:
    def __init__(self, mode="classification"):
        self.mode = mode
        self.bins = None
        self.feature_names = []
        self.categorical_features = []
        self.categorical_labels = {}
        self.class_labels = {}
        self.discretizer = None
        self.learner = MagicMock()
        self.x_cal = []
        self.y_cal = []
        # Provide a minimal plugin manager so orchestrator helpers can be exercised
        orchestrator = ExplanationOrchestrator(self)
        self.plugin_manager = SimpleNamespace(explanation_orchestrator=orchestrator)


@pytest.fixture
def explanation_orchestrator():
    explainer = ExplainerStub()
    return ExplanationOrchestrator(explainer)


@pytest.fixture
def prediction_orchestrator():
    explainer = ExplainerStub()
    return PredictionOrchestrator(explainer)


@pytest.mark.parametrize(
    "metadata, expected_error",
    [
        (None, "plugin metadata unavailable"),
        ({"schema_version": 999}, "unsupported"),
        (
            {
                "schema_version": EXPLANATION_PROTOCOL_VERSION,
                "modes": ["factual"],
                "capabilities": {"requires_predict_proba": False},
            },
            "missing tasks",
        ),
        (
            {
                "schema_version": EXPLANATION_PROTOCOL_VERSION,
                "tasks": ["classification"],
                "modes": ["alternative"],
                "capabilities": {"requires_predict_proba": False},
            },
            "does not declare mode",
        ),
        (
            {
                "schema_version": EXPLANATION_PROTOCOL_VERSION,
                "tasks": ["classification"],
                "modes": ["factual"],
                "capabilities": ["explain"],  # Missing mode:factual
            },
            "missing required capabilities",
        ),
    ],
)
def test_check_explanation_runtime_metadata_errors(
    explanation_orchestrator, metadata, expected_error
):
    error = explanation_orchestrator.check_metadata(metadata, identifier="plugin", mode="factual")
    assert error is not None
    assert expected_error in error


def test_check_explanation_runtime_metadata_success(explanation_orchestrator):
    metadata = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "tasks": ["classification"],
        "modes": ["factual"],
        "capabilities": ["explain", "mode:factual", "task:classification"],
    }
    assert (
        explanation_orchestrator.check_metadata(metadata, identifier="plugin", mode="factual")
        is None
    )


@pytest.mark.parametrize(
    "metadata, fast, expected_error",
    [
        (None, False, "interval metadata unavailable"),
        ({"schema_version": 999}, False, "unsupported"),
        (
            {
                "schema_version": 1,
                "capabilities": ["interval:classification"],
            },
            False,
            "missing modes",
        ),
        (
            {
                "schema_version": 1,
                "modes": ["classification"],
                "capabilities": ["interval:regression"],
            },
            False,
            "missing capability",
        ),
        (
            {
                "schema_version": 1,
                "modes": ["classification"],
                "capabilities": ["interval:classification"],
                "fast_compatible": False,
            },
            True,
            "not marked fast_compatible",
        ),
        (
            {
                "schema_version": 1,
                "modes": ["classification"],
                "capabilities": ["interval:classification"],
                "fast_compatible": True,
                "requires_bins": True,
            },
            True,
            "requires bins",
        ),  # Assuming bins is None
    ],
)
def test_check_interval_runtime_metadata_errors(
    prediction_orchestrator, metadata, fast, expected_error
):
    error = prediction_orchestrator.check_interval_runtime_metadata(
        metadata, identifier="interval", fast=fast
    )
    assert error is not None
    assert expected_error in error


def test_check_interval_runtime_metadata_success(prediction_orchestrator):
    metadata = {
        "schema_version": 1,
        "modes": ["classification"],
        "capabilities": ["interval:classification"],
        "fast_compatible": True,
    }
    assert (
        prediction_orchestrator.check_interval_runtime_metadata(
            metadata, identifier="interval", fast=True
        )
        is None
    )
