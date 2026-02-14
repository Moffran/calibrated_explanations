import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator


@pytest.fixture
def mock_explainer():
    explainer = MagicMock()
    explainer.x_cal = np.array([[1, 2], [3, 4]])
    explainer.y_cal = np.array([0, 1])
    explainer.condition_source = "observed"
    explainer.mode = "classification"
    explainer.categorical_features = []
    explainer.features_to_ignore = []
    explainer.feature_names = ["f1", "f2"]
    explainer.seed = 42
    explainer.discretizer = None
    explainer.num_features = 2
    return explainer


@pytest.fixture
def orchestrator(mock_explainer):
    return ExplanationOrchestrator(mock_explainer)


def test_invoke_per_instance_ignore(orchestrator, mock_explainer):
    from calibrated_explanations.plugins import ExplanationBatch

    class CalibratedExplanations:
        pass

    class MockContainer(CalibratedExplanations):
        @classmethod
        def from_batch(cls, batch):
            return batch

    class CalibratedExplanation:
        pass

    class MockExplanation(CalibratedExplanation):
        pass

    mock_batch = MagicMock(spec=ExplanationBatch)
    mock_batch.container_cls = MockContainer
    mock_batch.explanation_cls = MockExplanation
    mock_batch.instances = []
    mock_batch.collection_metadata = {}

    mock_plugin = MagicMock()
    mock_plugin.explain_batch.return_value = mock_batch

    orchestrator.ensure_plugin = MagicMock(return_value=(mock_plugin, "mock_id"))
    mock_explainer.plugin_manager.bridge_monitors = {}

    features_to_ignore = [[0], [1]]

    orchestrator.invoke(
        mode="factual",
        x=[[1, 2], [3, 4]],
        threshold=None,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=features_to_ignore,
    )

    args, _ = mock_plugin.explain_batch.call_args
    request = args[1]
    assert request.feature_filter_per_instance_ignore == ((0,), (1,))
    # flat_ignore should be unique union: (0, 1)
    assert set(request.features_to_ignore) == {0, 1}


def test_check_metadata_unsupported_task(orchestrator, mock_explainer):
    from calibrated_explanations.plugins import EXPLANATION_PROTOCOL_VERSION

    mock_explainer.mode = "regression"
    metadata = {"schema_version": EXPLANATION_PROTOCOL_VERSION, "tasks": ("classification",)}
    error = orchestrator.check_metadata(metadata, identifier="test", mode="factual")
    assert "does not support task" in error


def test_check_metadata_unsupported_mode(orchestrator, mock_explainer):
    from calibrated_explanations.plugins import EXPLANATION_PROTOCOL_VERSION

    mock_explainer.mode = "classification"
    metadata = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "tasks": ("classification",),
        "modes": ("alternative",),
    }
    error = orchestrator.check_metadata(metadata, identifier="test", mode="factual")
    assert "does not declare mode" in error


def test_build_context(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.interval_plugin_hints = {"factual": ("dep1",)}
    mock_explainer.plugin_manager.bridge_monitors = {}
    mock_explainer.plugin_manager.plot_style_chain = ("legacy",)
    mock_explainer.categorical_labels = {"f1": {0: "a"}}

    with patch(
        "calibrated_explanations.core.explain.orchestrator.find_explanation_descriptor",
        return_value=None,
    ):
        context = orchestrator.build_context("factual", MagicMock(), "plugin1")

        assert context.mode == "factual"
        assert context.task == "classification"
        assert context.interval_settings["dependencies"] == ("dep1",)
        assert context.plot_settings["fallbacks"] == ("legacy",)
        assert context.categorical_labels == {"f1": {0: "a"}}
