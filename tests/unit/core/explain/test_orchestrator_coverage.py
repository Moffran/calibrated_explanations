import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.utils.exceptions import ValidationError, ConfigurationError


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




def test_infer_mode_alternative(orchestrator, mock_explainer):
    from calibrated_explanations.utils.discretizers import EntropyDiscretizer, RegressorDiscretizer

    # Use a mock that passes isinstance(obj, EntropyDiscretizer)
    mock_entropy_instance = MagicMock(spec=EntropyDiscretizer)
    mock_explainer.discretizer = mock_entropy_instance
    assert orchestrator.infer_mode() == "alternative"

    mock_regressor_instance = MagicMock(spec=RegressorDiscretizer)
    mock_explainer.discretizer = mock_regressor_instance
    assert orchestrator.infer_mode() == "alternative"


def test_infer_mode_factual(orchestrator, mock_explainer):
    mock_explainer.discretizer = "some_other_discretizer"
    assert orchestrator.infer_mode() == "factual"


def test_invoke_basic(orchestrator, mock_explainer):
    from calibrated_explanations.plugins import ExplanationRequest, ExplanationBatch

    # Create a dummy container class that satisfies the validation check
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
    # validate_explanation_batch checks container_cls
    mock_batch.container_cls = MockContainer
    mock_batch.explanation_cls = MockExplanation
    mock_batch.instances = []
    mock_batch.collection_metadata = {}

    mock_plugin = MagicMock()
    mock_plugin.explain_batch.return_value = mock_batch

    # Mock ensure_plugin
    orchestrator.ensure_plugin = MagicMock(return_value=(mock_plugin, "mock_id"))

    # Mock _bridge_monitors
    mock_explainer.plugin_manager.bridge_monitors = {}

    result = orchestrator.invoke(
        mode="factual",
        x=[[1, 2]],
        threshold=0.5,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=[0],
        extras={"foo": "bar"},
    )

    assert result == mock_batch
    orchestrator.ensure_plugin.assert_called_with("factual")
    mock_plugin.explain_batch.assert_called_once()

    args, kwargs = mock_plugin.explain_batch.call_args
    assert args[0] == [[1, 2]]
    request = args[1]
    assert isinstance(request, ExplanationRequest)
    assert request.threshold == 0.5
    assert request.features_to_ignore == (0,)
    assert request.extras == {"foo": "bar"}


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




def test_set_discretizer_prediction_source(orchestrator, mock_explainer):
    mock_explainer.predict.return_value = np.array([0, 1])

    with (
        patch(
            "calibrated_explanations.core.discretizer_config.instantiate_discretizer"
        ) as mock_instantiate,
        patch(
            "calibrated_explanations.core.discretizer_config.validate_discretizer_choice"
        ) as mock_validate,
        patch(
            "calibrated_explanations.core.discretizer_config.setup_discretized_data"
        ) as mock_setup,
    ):
        mock_validate.return_value = "binaryEntropy"
        mock_setup.return_value = ({"f1": {"values": [], "frequencies": []}}, np.array([[1, 2]]))

        orchestrator.set_discretizer("binaryEntropy", condition_source="prediction")

        mock_explainer.predict.assert_called_once()
        # Verify instantiate was called with condition_labels from predict
        args, kwargs = mock_instantiate.call_args
        assert kwargs["condition_source"] == "prediction"
        np.testing.assert_array_equal(kwargs["condition_labels"], np.array([0, 1]))


def test_invoke_wrappers(orchestrator, mock_explainer):
    # Mock invoke to verify delegation
    orchestrator.invoke = MagicMock()

    # Test invoke_factual
    orchestrator.invoke_factual(
        x=[[1]],
        threshold=0.5,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=None,
        discretizer="binaryEntropy",
    )
    orchestrator.invoke.assert_called_with(
        mode="factual",
        x=[[1]],
        threshold=0.5,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=None,
        extras={},
    )

    # Test invoke_alternative
    orchestrator.invoke_alternative(
        x=[[1]],
        threshold=0.5,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=None,
        discretizer="entropy",
    )
    orchestrator.invoke.assert_called_with(
        mode="alternative",
        x=[[1]],
        threshold=0.5,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=None,
        extras={},
    )


def test_resolve_plugin_success(orchestrator, mock_explainer):
    # Mock plugin manager and overrides
    mock_explainer.explanation_plugin_overrides = {}
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    mock_explainer.plugin_manager.explanation_plugin_fallbacks = {"factual": ["plugin1"]}

    with (
        patch(
            "calibrated_explanations.core.explain.orchestrator.find_explanation_descriptor"
        ) as mock_find_desc,
        patch("calibrated_explanations.core.explain.orchestrator.find_explanation_plugin"),
    ):
        # Setup mock descriptor and plugin
        mock_desc = MagicMock()
        mock_desc.trusted = True
        # We need to know the actual version string or mock it.
        # Let's import it to be safe.
        from calibrated_explanations.plugins import EXPLANATION_PROTOCOL_VERSION

        mock_desc.metadata = {
            "schema_version": EXPLANATION_PROTOCOL_VERSION,
            "tasks": ("classification",),
            "modes": ("factual",),
            "capabilities": ("explain", "explanation:factual", "task:classification"),
        }
        mock_plugin_cls = MagicMock()
        mock_desc.plugin = mock_plugin_cls
        mock_find_desc.return_value = mock_desc

        # Mock instantiated plugin
        mock_plugin_instance = MagicMock()
        mock_plugin_instance.supports_mode.return_value = True

        # Mock instantiate_plugin to return our instance
        orchestrator.instantiate_plugin = MagicMock(return_value=mock_plugin_instance)

        plugin, identifier = orchestrator.resolve_plugin("factual")

        assert plugin == mock_plugin_instance
        assert identifier == "plugin1"








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


def test_derive_plot_chain(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.plot_style_chain = ("base",)

    with patch(
        "calibrated_explanations.core.explain.orchestrator.find_explanation_descriptor"
    ) as mock_find:
        mock_desc = MagicMock()
        mock_desc.metadata = {"plot_dependency": ("dep1",)}
        mock_find.return_value = mock_desc

        chain = orchestrator.derive_plot_chain("factual", "plugin1")
        assert chain == ("dep1", "base")


def testbuild_instance_telemetry_payload(orchestrator):
    # Test with valid payload
    mock_explanation = MagicMock()
    mock_explanation.to_telemetry.return_value = {"key": "value"}
    payload = orchestrator.build_instance_telemetry_payload([mock_explanation])
    assert payload == {"key": "value"}

    # Test with no to_telemetry
    mock_explanation_no_telemetry = MagicMock()
    del mock_explanation_no_telemetry.to_telemetry
    payload = orchestrator.build_instance_telemetry_payload([mock_explanation_no_telemetry])
    assert payload == {}

    # Test with empty list
    payload = orchestrator.build_instance_telemetry_payload([])
    assert payload == {}
