import pytest
from unittest.mock import MagicMock, patch
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.utils.exceptions import ConfigurationError


@pytest.fixture
def mock_explainer():
    explainer = MagicMock()
    explainer._plugin_manager = MagicMock()
    explainer._plugin_manager.coerce_plugin_override.return_value = None
    explainer.mode = "classification"
    explainer.is_multiclass.return_value = False
    explainer.is_fast.return_value = False
    explainer._CalibratedExplainer__initialized = True

    # Initialize interval state
    explainer._interval_plugin_identifiers = {"default": None, "fast": None}
    explainer._interval_plugin_fallbacks = {"default": ["default_plugin"], "fast": ["fast_plugin"]}
    explainer.interval_plugin_hints = {}
    explainer._interval_context_metadata = {"default": {}, "fast": {}}
    explainer._telemetry_interval_sources = {"default": None, "fast": None}
    explainer._interval_preferred_identifier = {"default": None, "fast": None}
    explainer._interval_plugin_override = None
    explainer._fast_interval_plugin_override = None
    explainer._instantiate_plugin = MagicMock(side_effect=lambda p: p)

    return explainer


@pytest.fixture
def orchestrator(mock_explainer):
    with patch("calibrated_explanations.core.prediction.interval_registry.IntervalRegistry"):
        return PredictionOrchestrator(mock_explainer)


@patch("calibrated_explanations.core.prediction.orchestrator.ensure_builtin_plugins")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_plugin")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_plugin_trusted")
@patch("calibrated_explanations.core.prediction.orchestrator.is_identifier_denied")
def test_resolve_interval_plugin_success(
    mock_is_denied,
    mock_find_trusted,
    mock_find,
    mock_find_desc,
    mock_ensure,
    orchestrator,
    mock_explainer,
):
    mock_is_denied.return_value = False

    # Setup descriptor
    mock_desc = MagicMock()
    mock_desc.metadata = {
        "name": "default_plugin",
        "modes": ("classification",),
        "capabilities": ("interval:classification",),
    }
    mock_desc.trusted = True
    mock_desc.plugin = MagicMock()
    mock_find_desc.return_value = mock_desc

    mock_explainer._instantiate_plugin.side_effect = lambda p: p

    plugin, identifier = orchestrator._resolve_interval_plugin(fast=False)

    assert identifier == "default_plugin"
    assert plugin == mock_desc.plugin
    mock_ensure.assert_called_once()


@patch("calibrated_explanations.core.prediction.orchestrator.ensure_builtin_plugins")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor")
@patch("calibrated_explanations.core.prediction.orchestrator.is_identifier_denied")
def test_resolve_interval_plugin_denied(mock_is_denied, mock_find_desc, mock_ensure, orchestrator):
    mock_is_denied.return_value = True

    with pytest.raises(ConfigurationError, match="Unable to resolve interval plugin"):
        orchestrator._resolve_interval_plugin(fast=False)


@patch("calibrated_explanations.core.prediction.orchestrator.ensure_builtin_plugins")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_plugin")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_plugin_trusted")
@patch("calibrated_explanations.core.prediction.orchestrator.is_identifier_denied")
def test_resolve_interval_plugin_metadata_error(
    mock_is_denied,
    mock_find_trusted,
    mock_find,
    mock_find_desc,
    mock_ensure,
    orchestrator,
    mock_explainer,
):
    mock_is_denied.return_value = False

    # Setup descriptor with incompatible metadata
    mock_desc = MagicMock()
    mock_desc.metadata = {
        "name": "default_plugin",
        "modes": ("regression",),
        "capabilities": ("interval:regression",),
    }  # Wrong mode
    mock_desc.trusted = True
    mock_desc.plugin = MagicMock()
    mock_find_desc.return_value = mock_desc

    with pytest.raises(ConfigurationError, match="Unable to resolve interval plugin"):
        orchestrator._resolve_interval_plugin(fast=False)


@patch("calibrated_explanations.core.prediction.orchestrator.ensure_builtin_plugins")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_plugin")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_plugin_trusted")
@patch("calibrated_explanations.core.prediction.orchestrator.is_identifier_denied")
def test_resolve_interval_plugin_override_object(
    mock_is_denied,
    mock_find_trusted,
    mock_find,
    mock_find_desc,
    mock_ensure,
    orchestrator,
    mock_explainer,
):
    # Setup override object
    mock_override = MagicMock()
    mock_override.plugin_meta = {"name": "override_plugin"}
    mock_explainer._plugin_manager.coerce_plugin_override.return_value = mock_override

    plugin, identifier = orchestrator._resolve_interval_plugin(fast=False)

    assert plugin == mock_override
    assert identifier == "override_plugin"


@patch("calibrated_explanations.core.prediction.orchestrator.ensure_builtin_plugins")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_plugin")
@patch("calibrated_explanations.core.prediction.orchestrator.find_interval_plugin_trusted")
@patch("calibrated_explanations.core.prediction.orchestrator.is_identifier_denied")
def test_resolve_interval_plugin_override_string(
    mock_is_denied,
    mock_find_trusted,
    mock_find,
    mock_find_desc,
    mock_ensure,
    orchestrator,
    mock_explainer,
):
    mock_is_denied.return_value = False
    mock_explainer._interval_plugin_override = "override_plugin"
    # The override plugin must be in the fallback chain to be considered
    mock_explainer._interval_plugin_fallbacks["default"] = ["override_plugin", "default_plugin"]

    # Setup descriptor for override
    mock_desc = MagicMock()
    mock_desc.metadata = {
        "name": "override_plugin",
        "modes": ("classification",),
        "capabilities": ("interval:classification",),
    }
    mock_desc.trusted = True
    mock_desc.plugin = MagicMock()

    def find_desc_side_effect(name):
        if name == "override_plugin":
            return mock_desc
        return None

    mock_find_desc.side_effect = find_desc_side_effect

    mock_explainer._instantiate_plugin.side_effect = lambda p: p

    plugin, identifier = orchestrator._resolve_interval_plugin(fast=False)

    assert identifier == "override_plugin"
    assert plugin == mock_desc.plugin


def test_obtain_interval_calibrator_success(orchestrator, mock_explainer):
    mock_plugin = MagicMock()
    mock_calibrator = MagicMock()
    mock_plugin.create.return_value = mock_calibrator

    with patch.object(
        orchestrator, "_resolve_interval_plugin", return_value=(mock_plugin, "test_plugin")
    ), patch.object(orchestrator, "_build_interval_context") as mock_build_context:
        mock_context = MagicMock()
        mock_context.metadata = {}
        mock_build_context.return_value = mock_context

        calibrator, identifier = orchestrator._obtain_interval_calibrator(fast=False, metadata={})

        assert calibrator == mock_calibrator
        assert identifier == "test_plugin"
        mock_plugin.create.assert_called_once()
        assert mock_explainer._interval_plugin_identifiers["default"] == "test_plugin"


def test_obtain_interval_calibrator_failure(orchestrator, mock_explainer):
    mock_plugin = MagicMock()
    mock_plugin.create.side_effect = ValueError("Creation failed")

    with patch.object(
        orchestrator, "_resolve_interval_plugin", return_value=(mock_plugin, "test_plugin")
    ), patch.object(orchestrator, "_build_interval_context"), pytest.raises(
        ConfigurationError, match="Interval plugin execution failed"
    ):
        orchestrator._obtain_interval_calibrator(fast=False, metadata={})


def test_check_interval_runtime_metadata_errors(orchestrator, mock_explainer):
    # Test various metadata validation errors

    # Missing metadata
    assert "unavailable" in orchestrator._check_interval_runtime_metadata(
        None, identifier="test", fast=False
    )

    # Wrong schema version
    assert "schema_version" in orchestrator._check_interval_runtime_metadata(
        {"schema_version": 2}, identifier="test", fast=False
    )

    # Missing modes
    assert "missing modes" in orchestrator._check_interval_runtime_metadata(
        {"schema_version": 1}, identifier="test", fast=False
    )

    # Wrong mode
    mock_explainer.mode = "classification"
    assert "does not support mode" in orchestrator._check_interval_runtime_metadata(
        {"schema_version": 1, "modes": ("regression",)}, identifier="test", fast=False
    )

    # Missing capability
    assert "missing capability" in orchestrator._check_interval_runtime_metadata(
        {"schema_version": 1, "modes": ("classification",), "capabilities": ()},
        identifier="test",
        fast=False,
    )

    # Not fast compatible
    assert "not marked fast_compatible" in orchestrator._check_interval_runtime_metadata(
        {
            "schema_version": 1,
            "modes": ("classification",),
            "capabilities": ("interval:classification",),
        },
        identifier="test",
        fast=True,
    )

    # Requires bins but none
    mock_explainer.bins = None
    assert "requires bins" in orchestrator._check_interval_runtime_metadata(
        {
            "schema_version": 1,
            "modes": ("classification",),
            "capabilities": ("interval:classification",),
            "requires_bins": True,
        },
        identifier="test",
        fast=False,
    )
