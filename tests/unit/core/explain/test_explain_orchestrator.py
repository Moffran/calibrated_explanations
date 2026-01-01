import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.utils.exceptions import ConfigurationError, ValidationError


@pytest.fixture
def mock_explainer():
    explainer = MagicMock()
    explainer.mode = "classification"
    explainer.x_cal = np.array([[1, 2], [3, 4]])
    explainer.y_cal = np.array([0, 1])
    explainer.num_features = 2
    explainer.feature_names = ["f1", "f2"]
    explainer.categorical_features = []
    explainer.features_to_ignore = []
    explainer.seed = 42
    explainer.bins = None
    explainer.condition_source = "observed"
    explainer.discretizer = None
    explainer.plugin_manager = MagicMock()
    explainer._bridge_monitors = {}
    explainer.telemetry_interval_sources = {}
    explainer.interval_plugin_hints = {}
    explainer.plugin_manager.plot_plugin_fallbacks = {}
    explainer.preprocessor_metadata = None
    return explainer


@pytest.fixture
def orchestrator(mock_explainer):
    return ExplanationOrchestrator(mock_explainer)


def test_initialize_chains(orchestrator, mock_explainer):
    """Test delegation of initialize_chains."""
    orchestrator.initialize_chains()
    mock_explainer.plugin_manager.initialize_chains.assert_called_once()


def test_set_discretizer_invalid_condition_source(orchestrator):
    """Test set_discretizer with invalid condition_source."""
    with pytest.raises(ValidationError, match="condition_source must be either"):
        orchestrator.set_discretizer("entropy", condition_source="invalid")


def test_set_discretizer_prediction_source(orchestrator, mock_explainer):
    """Test set_discretizer with prediction condition source."""
    mock_explainer.predict.return_value = (np.array([0, 1]), None, None)

    with (
        patch(
            "calibrated_explanations.core.discretizer_config.validate_discretizer_choice",
            return_value="entropy",
        ),
        patch(
            "calibrated_explanations.core.discretizer_config.instantiate_discretizer"
        ) as mock_instantiate,
        patch(
            "calibrated_explanations.core.discretizer_config.setup_discretized_data",
            return_value=({"f1": {"values": [], "frequencies": []}}, None),
        ),
    ):
        orchestrator.set_discretizer("entropy", condition_source="prediction")

        mock_explainer.predict.assert_called_once()
        mock_instantiate.assert_called_once()
        # Check that condition_labels were passed
        _, kwargs = mock_instantiate.call_args
        assert "condition_labels" in kwargs
        assert np.array_equal(kwargs["condition_labels"], np.array([0, 1]))


def test_set_discretizer_success(orchestrator, mock_explainer):
    """Test successful set_discretizer execution."""
    mock_discretizer = MagicMock()

    with (
        patch(
            "calibrated_explanations.core.discretizer_config.validate_discretizer_choice",
            return_value="entropy",
        ),
        patch(
            "calibrated_explanations.core.discretizer_config.instantiate_discretizer",
            return_value=mock_discretizer,
        ),
        patch(
            "calibrated_explanations.core.discretizer_config.setup_discretized_data",
            return_value=({"f1": {"values": [1], "frequencies": [1]}}, "discretized_data"),
        ),
    ):
        orchestrator.set_discretizer("entropy")

        assert mock_explainer.discretizer == mock_discretizer
        assert mock_explainer.discretized_X_cal == "discretized_data"
        assert mock_explainer.feature_values == {"f1": [1]}
        assert mock_explainer.feature_frequencies == {"f1": [1]}


def test_infer_mode(orchestrator, mock_explainer):
    """Test infer_mode based on discretizer type."""
    from calibrated_explanations.utils import EntropyDiscretizer

    mock_explainer.discretizer = MagicMock(spec=EntropyDiscretizer)
    assert orchestrator.infer_mode() == "alternative"

    mock_explainer.discretizer = MagicMock()  # Not Entropy/Regressor
    assert orchestrator.infer_mode() == "factual"


def test_invoke_success(orchestrator, mock_explainer):
    """Test successful invoke execution."""
    mock_plugin = MagicMock()
    mock_batch = MagicMock()
    mock_container = MagicMock()
    mock_result = MagicMock()

    mock_plugin.explain_batch.return_value = mock_batch
    mock_batch.collection_metadata = {}
    mock_batch.container_cls = mock_container
    mock_container.from_batch.return_value = mock_result

    with (
        patch.object(orchestrator, "ensure_plugin", return_value=(mock_plugin, "test_plugin")),
        patch("calibrated_explanations.core.explain.orchestrator.validate_explanation_batch"),
    ):
        result = orchestrator.invoke(
            mode="factual",
            x=np.array([[1, 2]]),
            threshold=None,
            low_high_percentiles=None,
            bins=None,
            features_to_ignore=None,
        )

        assert result == mock_result
        mock_plugin.explain_batch.assert_called_once()
        mock_container.from_batch.assert_called_once()


def test_invoke_plugin_failure(orchestrator):
    """Test invoke when plugin execution fails."""
    mock_plugin = MagicMock()
    mock_plugin.explain_batch.side_effect = ValueError("Plugin error")

    with (
        patch.object(orchestrator, "ensure_plugin", return_value=(mock_plugin, "test_plugin")),
        pytest.raises(ConfigurationError, match="Explanation plugin execution failed"),
    ):
        orchestrator.invoke(
            mode="factual",
            x=np.array([[1, 2]]),
            threshold=None,
            low_high_percentiles=None,
            bins=None,
            features_to_ignore=None,
        )


def test_invoke_validation_failure(orchestrator):
    """Test invoke when validation fails."""
    mock_plugin = MagicMock()
    mock_batch = MagicMock()
    mock_plugin.explain_batch.return_value = mock_batch

    with (
        patch.object(orchestrator, "ensure_plugin", return_value=(mock_plugin, "test_plugin")),
        patch(
            "calibrated_explanations.core.explain.orchestrator.validate_explanation_batch",
            side_effect=ValueError("Validation error"),
        ),
        pytest.raises(ConfigurationError, match="returned an invalid batch"),
    ):
        orchestrator.invoke(
            mode="factual",
            x=np.array([[1, 2]]),
            threshold=None,
            low_high_percentiles=None,
            bins=None,
            features_to_ignore=None,
        )


def test_invoke_bridge_monitor_failure(orchestrator, mock_explainer):
    """Test invoke when bridge monitor is not used."""
    mock_plugin = MagicMock()
    mock_batch = MagicMock()
    mock_batch.collection_metadata = {}

    mock_monitor = MagicMock()
    mock_monitor.used = False
    mock_explainer.plugin_manager.get_bridge_monitor.return_value = mock_monitor

    with (
        patch.object(orchestrator, "ensure_plugin", return_value=(mock_plugin, "custom_plugin")),
        patch("calibrated_explanations.core.explain.orchestrator.validate_explanation_batch"),
        pytest.raises(ConfigurationError, match="did not use the calibrated predict bridge"),
    ):
        orchestrator.invoke(
            mode="factual",
            x=np.array([[1, 2]]),
            threshold=None,
            low_high_percentiles=None,
            bins=None,
            features_to_ignore=None,
        )


def testensure_plugin_cached(orchestrator, mock_explainer):
    """Test ensure_plugin returns cached instance."""
    mock_plugin = MagicMock()
    mock_explainer.plugin_manager.explanation_plugin_instances = {"factual": mock_plugin}
    mock_explainer.plugin_manager.explanation_plugin_identifiers = {"factual": "test_plugin"}

    plugin, identifier = orchestrator.ensure_plugin("factual")

    assert plugin == mock_plugin
    assert identifier == "test_plugin"


def testensure_plugin_new(orchestrator, mock_explainer):
    """Test ensure_plugin resolves and initializes new plugin."""
    mock_explainer.plugin_manager.explanation_plugin_instances = {}
    mock_plugin = MagicMock()
    mock_plugin.plugin_meta = {"name": "test_plugin"}

    with (
        patch.object(orchestrator, "resolve_plugin", return_value=(mock_plugin, "test_plugin")),
        patch.object(orchestrator, "check_metadata", return_value=None),
        patch.object(orchestrator, "build_context", return_value=MagicMock()),
    ):
        plugin, identifier = orchestrator.ensure_plugin("factual")

        assert plugin == mock_plugin
        assert identifier == "test_plugin"
        mock_plugin.initialize.assert_called_once()
        assert mock_explainer.plugin_manager.explanation_plugin_instances["factual"] == mock_plugin


def testensure_plugin_init_failure(orchestrator, mock_explainer):
    """Test ensure_plugin initialization failure."""
    mock_explainer.plugin_manager.explanation_plugin_instances = {}
    mock_plugin = MagicMock()
    mock_plugin.initialize.side_effect = ValueError("Init error")

    with (
        patch.object(orchestrator, "resolve_plugin", return_value=(mock_plugin, "test_plugin")),
        patch.object(orchestrator, "check_metadata", return_value=None),
        patch.object(orchestrator, "build_context", return_value=MagicMock()),
        pytest.raises(ConfigurationError, match="Explanation plugin initialisation failed"),
    ):
        orchestrator.ensure_plugin("factual")


def test_invoke_factual_delegation(orchestrator):
    """Test invoke_factual delegates to invoke."""
    with patch.object(orchestrator, "invoke") as mock_invoke:
        orchestrator.invoke_factual(
            x=np.array([[1, 2]]),
            threshold=None,
            low_high_percentiles=None,
            bins=None,
            features_to_ignore=None,
            extra_arg="value",
        )

        mock_invoke.assert_called_once()
        _, kwargs = mock_invoke.call_args
        assert kwargs["mode"] == "factual"
        assert kwargs["extras"] == {"extra_arg": "value"}


def test_invoke_alternative_delegation(orchestrator):
    """Test invoke_alternative delegates to invoke."""
    with patch.object(orchestrator, "invoke") as mock_invoke:
        orchestrator.invoke_alternative(
            x=np.array([[1, 2]]),
            threshold=None,
            low_high_percentiles=None,
            bins=None,
            features_to_ignore=None,
            extra_arg="value",
        )

        mock_invoke.assert_called_once()
        _, kwargs = mock_invoke.call_args
        assert kwargs["mode"] == "alternative"
        assert kwargs["extras"] == {"extra_arg": "value"}


def test_resolve_plugin_override_object(orchestrator, mock_explainer):
    """Test _resolve_plugin with object override."""
    mock_plugin = MagicMock()
    mock_plugin.plugin_meta = {"name": "custom_plugin"}
    mock_explainer.plugin_manager.explanation_plugin_overrides = {"factual": mock_plugin}
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = mock_plugin

    plugin, identifier = orchestrator.resolve_plugin("factual")

    assert plugin == mock_plugin
    assert identifier == "custom_plugin"


def test_resolve_plugin_fast_missing(orchestrator, mock_explainer):
    """Test _resolve_plugin fast mode missing error."""
    mock_explainer.plugin_manager.explanation_plugin_overrides = {}
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    mock_explainer.plugin_manager.explanation_plugin_fallbacks = {"fast": []}

    with pytest.raises(
        ConfigurationError,
        match="Fast explanation plugin 'core.explanation.fast' is not registered",
    ):
        orchestrator.resolve_plugin("fast")


def test_resolve_plugin_denied(orchestrator, mock_explainer):
    """Test _resolve_plugin with denied plugin."""
    mock_explainer.plugin_manager.explanation_plugin_overrides = {}
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    mock_explainer.plugin_manager.explanation_plugin_fallbacks = {"factual": ["denied_plugin"]}

    with (
        patch(
            "calibrated_explanations.core.explain.orchestrator.is_identifier_denied",
            return_value=True,
        ),
        pytest.raises(ConfigurationError, match="Unable to resolve explanation plugin"),
    ):
        orchestrator.resolve_plugin("factual")


def test_resolve_plugin_not_registered(orchestrator, mock_explainer):
    """Test _resolve_plugin with unregistered plugin."""
    mock_explainer.plugin_manager.explanation_plugin_overrides = {}
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    mock_explainer.plugin_manager.explanation_plugin_fallbacks = {"factual": ["missing_plugin"]}

    with (
        patch(
            "calibrated_explanations.core.explain.orchestrator.is_identifier_denied",
            return_value=False,
        ),
        patch(
            "calibrated_explanations.core.explain.orchestrator.find_explanation_descriptor",
            return_value=None,
        ),
        patch(
            "calibrated_explanations.core.explain.orchestrator.find_explanation_plugin",
            return_value=None,
        ),
        pytest.raises(ConfigurationError, match="Unable to resolve explanation plugin"),
    ):
        orchestrator.resolve_plugin("factual")


def test_resolve_plugin_metadata_error(orchestrator, mock_explainer):
    """Test _resolve_plugin with metadata error."""
    mock_explainer.plugin_manager.explanation_plugin_overrides = {}
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    mock_explainer.plugin_manager.explanation_plugin_fallbacks = {
        "factual": ["bad_metadata_plugin"]
    }

    mock_plugin = MagicMock()
    mock_plugin.plugin_meta = {}  # Empty metadata causes error

    with (
        patch(
            "calibrated_explanations.core.explain.orchestrator.is_identifier_denied",
            return_value=False,
        ),
        patch(
            "calibrated_explanations.core.explain.orchestrator.find_explanation_descriptor",
            return_value=None,
        ),
        patch(
            "calibrated_explanations.core.explain.orchestrator.find_explanation_plugin",
            return_value=mock_plugin,
        ),
        pytest.raises(ConfigurationError, match="Unable to resolve explanation plugin"),
    ):
        orchestrator.resolve_plugin("factual")


def test_resolve_plugin_supports_mode_failure(orchestrator, mock_explainer):
    """Test _resolve_plugin when supports_mode returns False."""
    mock_explainer.plugin_manager.explanation_plugin_overrides = {}
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    mock_explainer.plugin_manager.explanation_plugin_fallbacks = {"factual": ["unsupported_plugin"]}

    mock_plugin = MagicMock()
    mock_plugin.plugin_meta = None
    mock_plugin.supports_mode.return_value = False

    with (
        patch(
            "calibrated_explanations.core.explain.orchestrator.is_identifier_denied",
            return_value=False,
        ),
        patch(
            "calibrated_explanations.core.explain.orchestrator.find_explanation_descriptor",
            return_value=None,
        ),
        patch(
            "calibrated_explanations.core.explain.orchestrator.find_explanation_plugin",
            return_value=mock_plugin,
        ),
        patch.object(orchestrator, "check_metadata", return_value=None),
        pytest.raises(ConfigurationError, match="Unable to resolve explanation plugin"),
    ):
        orchestrator.resolve_plugin("factual")


def test_check_metadata_valid(orchestrator, mock_explainer):
    """Test _check_metadata with valid metadata."""
    from calibrated_explanations.plugins import EXPLANATION_PROTOCOL_VERSION

    metadata = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "tasks": ("classification",),
        "modes": ("factual",),
        "capabilities": ("explain", "explanation:factual", "task:classification"),
    }
    assert orchestrator.check_metadata(metadata, identifier="test", mode="factual") is None


def test_check_metadata_invalid_version(orchestrator):
    """Test _check_metadata with invalid version."""
    metadata = {"schema_version": -1}
    assert "unsupported" in orchestrator.check_metadata(metadata, identifier="test", mode="factual")


def test_check_metadata_missing_tasks(orchestrator):
    """Test _check_metadata with missing tasks."""
    from calibrated_explanations.plugins import EXPLANATION_PROTOCOL_VERSION

    metadata = {"schema_version": EXPLANATION_PROTOCOL_VERSION}
    assert "missing tasks declaration" in orchestrator.check_metadata(
        metadata, identifier="test", mode="factual"
    )


def test_check_metadata_missing_modes(orchestrator):
    """Test _check_metadata with missing modes."""
    from calibrated_explanations.plugins import EXPLANATION_PROTOCOL_VERSION

    metadata = {"schema_version": EXPLANATION_PROTOCOL_VERSION, "tasks": ("classification",)}
    assert "missing modes declaration" in orchestrator.check_metadata(
        metadata, identifier="test", mode="factual"
    )


def test_check_metadata_missing_capabilities(orchestrator):
    """Test _check_metadata with missing capabilities."""
    from calibrated_explanations.plugins import EXPLANATION_PROTOCOL_VERSION

    metadata = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "tasks": ("classification",),
        "modes": ("factual",),
        "capabilities": (),
    }
    assert "missing required capabilities" in orchestrator.check_metadata(
        metadata, identifier="test", mode="factual"
    )


def testinstantiate_plugin_callable(orchestrator):
    """Test instantiate_plugin with callable."""
    mock_callable = MagicMock()
    mock_callable.plugin_meta = {}
    assert orchestrator.instantiate_plugin(mock_callable) == mock_callable


def testinstantiate_plugin_class(orchestrator):
    """Test instantiate_plugin with class."""

    class MockPlugin:
        pass

    assert isinstance(orchestrator.instantiate_plugin(MockPlugin()), MockPlugin)


def test_build_context(orchestrator, mock_explainer):
    """Test _build_context."""
    mock_explainer.plugin_manager.interval_plugin_hints = {"factual": ("hint1",)}
    mock_explainer.plugin_manager.plot_style_chain = ("style1",)
    mock_explainer.categorical_labels = {0: {1: "cat"}}

    context = orchestrator.build_context("factual", MagicMock(), "test_plugin")

    assert context.mode == "factual"
    assert context.interval_settings["dependencies"] == ("hint1",)
    assert "style1" in context.plot_settings["fallbacks"]


def test_derive_plot_chain(orchestrator, mock_explainer):
    """Test _derive_plot_chain."""
    mock_explainer.plugin_manager.plot_style_chain = ("base",)

    mock_descriptor = MagicMock()
    mock_descriptor.metadata = {"plot_dependency": "dep"}

    with patch(
        "calibrated_explanations.core.explain.orchestrator.find_explanation_descriptor",
        return_value=mock_descriptor,
    ):
        chain = orchestrator._derive_plot_chain("factual", "test_plugin")
        assert chain == ("dep", "base")


def testbuild_instance_telemetry_payload(orchestrator):
    """Test build_instance_telemetry_payload."""
    mock_explanation = MagicMock()
    mock_explanation.to_telemetry.return_value = {"key": "value"}
    explanations = [mock_explanation]

    payload = orchestrator.build_instance_telemetry_payload(explanations)
    assert payload == {"key": "value"}
