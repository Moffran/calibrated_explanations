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
    explainer.bridge_monitors = {}
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


def test_infer_mode(orchestrator, mock_explainer):
    """Test infer_mode based on discretizer type."""
    from calibrated_explanations.utils import EntropyDiscretizer

    mock_explainer.discretizer = MagicMock(spec=EntropyDiscretizer)
    assert orchestrator.infer_mode() == "alternative"

    mock_explainer.discretizer = MagicMock()  # Not Entropy/Regressor
    assert orchestrator.infer_mode() == "factual"


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


def test_invoke_factual_multiclass_all_classes_enabled(
    orchestrator,
    mock_explainer,
):
    """Ensure multiclass all-class branch is reachable via multi_labels_enabled."""
    mock_explainer.class_labels = None
    expected = object()

    with (
        patch(
            "calibrated_explanations.core.explain._legacy_explain.explain",
            return_value=["legacy-explanation"],
        ) as legacy_explain,
        patch(
            "calibrated_explanations.explanations.explanations.MultiClassCalibratedExplanations",
            return_value=expected,
        ) as multi_cls,
    ):
        result = orchestrator.invoke_factual(
            x=np.array([[1, 2]]),
            threshold=None,
            low_high_percentiles=(5, 95),
            bins=None,
            features_to_ignore=None,
            multi_labels_enabled=True,
        )

    assert result is expected
    assert legacy_explain.call_count == len(np.unique(mock_explainer.y_cal))
    multi_cls.assert_called_once()


def test_invoke_factual_multiclass_all_classes_disabled(
    orchestrator,
    mock_explainer,
):
    """Ensure multi-label branch is not entered when multi_labels_enabled is False."""
    mock_explainer.class_labels = None

    with (
        patch(
            "calibrated_explanations.core.explain._legacy_explain.explain",
            return_value=["legacy-explanation"],
        ) as legacy_explain,
        patch(
            "calibrated_explanations.explanations.explanations.MultiClassCalibratedExplanations"
        ) as multi_cls,
        patch.object(orchestrator, "invoke", return_value="plugin-path") as invoke_mock,
    ):
        result = orchestrator.invoke_factual(
            x=np.array([[1, 2]]),
            threshold=None,
            low_high_percentiles=(5, 95),
            bins=None,
            features_to_ignore=None,
            multi_labels_enabled=False,
        )

    assert result == "plugin-path"
    legacy_explain.assert_not_called()
    multi_cls.assert_not_called()
    invoke_mock.assert_called_once()


def test_invoke_alternative_multiclass_all_classes_enabled(
    orchestrator,
    mock_explainer,
):
    """Ensure alternative mode supports multi-label aggregation."""
    mock_explainer.class_labels = None
    expected = object()

    with (
        patch(
            "calibrated_explanations.core.explain._legacy_explain.explain",
            return_value=["legacy-explanation"],
        ) as legacy_explain,
        patch(
            "calibrated_explanations.explanations.explanations.MultiClassCalibratedExplanations",
            return_value=expected,
        ) as multi_cls,
    ):
        result = orchestrator.invoke_alternative(
            x=np.array([[1, 2]]),
            threshold=None,
            low_high_percentiles=(5, 95),
            bins=None,
            features_to_ignore=None,
            multi_labels_enabled=True,
        )

    assert result is expected
    assert legacy_explain.call_count == len(np.unique(mock_explainer.y_cal))
    multi_cls.assert_called_once()


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


def test_resolve_plugin_metadata_error(orchestrator, mock_explainer):
    """Test _resolve_plugin with metadata error."""
    mock_explainer.plugin_manager.explanation_plugin_overrides = {}
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    mock_explainer.plugin_manager.explanation_plugin_fallbacks = {
        "factual": ["bad_metadata_plugin"]
    }

    mock_plugin = MagicMock()
    mock_plugin.plugin_meta = {}  # Empty metadata causes error
    mock_explainer.plugin_manager.resolve_explanation_plugin.return_value = (
        mock_plugin,
        None,
        None,
    )

    with (
        patch(
            "calibrated_explanations.core.explain.orchestrator.is_identifier_denied",
            return_value=False,
        ),
        pytest.raises(ConfigurationError, match="Unable to resolve explanation plugin"),
    ):
        orchestrator.resolve_plugin("factual")


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


def test_derive_plot_chain(orchestrator, mock_explainer):
    """Test _derive_plot_chain."""
    mock_explainer.plugin_manager.plot_style_chain = ("base",)

    mock_descriptor = MagicMock()
    mock_descriptor.metadata = {"plot_dependency": "dep"}

    with patch(
        "calibrated_explanations.core.explain.orchestrator.find_explanation_descriptor",
        return_value=mock_descriptor,
    ):
        chain = orchestrator.derive_plot_chain("factual", "test_plugin")
        assert chain == ("dep", "base")


@pytest.mark.parametrize("invoker_name", ("invoke_factual", "invoke_alternative"))
def test_should_freeze_bins_across_modes(orchestrator, mock_explainer, invoker_name):
    """Bins should be frozen to tuples for every mode delegate."""

    bins = np.array([[1, 2], [3, 4]])
    captured_request: dict[str, object] = {}

    mock_explainer.plugin_manager.get_bridge_monitor.return_value = None

    def capture_request_helper(_x, request):
        captured_request["bins"] = request.bins
        assert isinstance(request.bins, tuple)
        assert request.bins == tuple(tuple(row) for row in bins.tolist())
        with pytest.raises(TypeError):
            request.bins[0] = "mutate"  # type: ignore[index]
        return mock_batch

    mock_plugin = MagicMock()
    mock_plugin.explain_batch.side_effect = capture_request_helper
    mock_batch = MagicMock()
    mock_batch.collection_metadata = {}
    mock_container = MagicMock()
    mock_batch.container_cls = mock_container
    mock_container.from_batch.return_value = MagicMock()

    with (
        patch.object(orchestrator, "ensure_plugin", return_value=(mock_plugin, "core.test")),
        patch(
            "calibrated_explanations.core.explain.orchestrator.validate_explanation_batch",
        ),
        patch.object(ExplanationOrchestrator, "build_instance_telemetry_payload", return_value={}),
    ):
        getattr(orchestrator, invoker_name)(
            x=np.array([[1, 2]]),
            threshold=None,
            low_high_percentiles=None,
            bins=bins,
            features_to_ignore=None,
        )

    assert captured_request["bins"] == tuple(tuple(row) for row in bins.tolist())


@pytest.mark.parametrize("invoker_name", ("invoke_factual", "invoke_alternative"))
@pytest.mark.parametrize("x", (np.array([[1, 2]]), np.array([[1, 2], [3, 4]])))
def test_should_include_interval_dependency_telemetry_in_single_and_batch_paths(
    orchestrator,
    mock_explainer,
    invoker_name,
    x,
):
    """Interval dependency metadata should be present on emitted telemetry payloads."""
    mock_explainer.plugin_manager.get_bridge_monitor.return_value = None
    mock_explainer.plugin_manager.telemetry_interval_sources = {"default": "core.interval.legacy"}
    mock_explainer.plugin_manager.interval_plugin_hints = {
        "factual": ("core.interval.legacy",),
        "alternative": ("core.interval.legacy",),
    }
    mock_explainer.plugin_manager.plot_plugin_fallbacks = {}
    mock_explainer.plugin_manager.last_telemetry = {}

    mock_plugin = MagicMock()
    mock_batch = MagicMock()
    mock_batch.collection_metadata = {}
    mock_container = MagicMock()
    mock_batch.container_cls = mock_container
    result = MagicMock()
    mock_container.from_batch.return_value = result
    mock_plugin.explain_batch.return_value = mock_batch

    with (
        patch.object(orchestrator, "ensure_plugin", return_value=(mock_plugin, "core.test")),
        patch("calibrated_explanations.core.explain.orchestrator.validate_explanation_batch"),
        patch.object(ExplanationOrchestrator, "build_instance_telemetry_payload", return_value={}),
    ):
        getattr(orchestrator, invoker_name)(
            x=x,
            threshold=None,
            low_high_percentiles=None,
            bins=None,
            features_to_ignore=None,
        )

    assert result.telemetry["interval_dependencies"] == ("core.interval.legacy",)
    assert result.telemetry["interval_source"] == "core.interval.legacy"
