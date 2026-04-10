"""ADR-033 packaging smoke tests: entry-point discovery, modality contract, plugin_api_version."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Insert the fixture package into sys.path so MinimalModalityPlugin is importable.
_FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "ce_mock_modality_plugin"
if str(_FIXTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FIXTURE_ROOT))

from ce_mock_modality_plugin import MinimalModalityPlugin  # noqa: E402

from calibrated_explanations.plugins.registry import (  # noqa: E402
    DefaultPluginTrustPolicy,
    find_explanation_descriptor,
    get_last_discovery_report,
    load_entrypoint_plugins,
    set_trust_policy,
)
from tests.support.registry_helpers import (  # noqa: E402
    clear_explanation_plugins,
    clear_legacy_registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ep(name: str, plugin_class: type) -> MagicMock:
    """Return a minimal entrypoint mock that mimics importlib.metadata.EntryPoint."""
    ep = MagicMock()
    ep.name = name
    ep.group = "calibrated_explanations.plugins"
    ep.attr = None  # ensures identifier = entry_point.name
    ep.dist = None
    ep.load.return_value = plugin_class
    return ep


class _MockEntryPoints:
    """Minimal stand-in for importlib.metadata.EntryPoints."""

    def __init__(self, eps: list) -> None:
        self.eps = eps

    def select(self, *, group: str) -> list:
        return self.eps if group == "calibrated_explanations.plugins" else []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry_and_policy():
    clear_explanation_plugins()
    clear_legacy_registry()
    set_trust_policy(DefaultPluginTrustPolicy())
    yield
    clear_explanation_plugins()
    clear_legacy_registry()
    set_trust_policy(DefaultPluginTrustPolicy())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_should_register_valid_plugin_when_entry_point_is_discovered():
    """Entry-point discovery registers a MinimalModalityPlugin in the explanation catalog."""
    # Arrange
    ep = _make_ep("tests.mock.modality", MinimalModalityPlugin)
    mock_eps = _MockEntryPoints([ep])

    # Act
    with patch(
        "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
        return_value=mock_eps,
    ):
        load_entrypoint_plugins(include_untrusted=True)

    # Assert
    descriptor = find_explanation_descriptor("tests.mock.modality")
    assert descriptor is not None
    report = get_last_discovery_report()
    assert report is not None
    accepted_ids = {record.identifier for record in report.accepted}
    assert "tests.mock.modality" in accepted_ids


def test_should_preserve_data_modalities_when_entry_point_is_discovered():
    """Entry-point discovery preserves the declared data_modalities in the descriptor."""
    # Arrange
    ep = _make_ep("tests.mock.modality", MinimalModalityPlugin)
    mock_eps = _MockEntryPoints([ep])

    # Act
    with patch(
        "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
        return_value=mock_eps,
    ):
        load_entrypoint_plugins(include_untrusted=True)

    # Assert
    descriptor = find_explanation_descriptor("tests.mock.modality")
    assert descriptor is not None
    assert descriptor.metadata["data_modalities"] == ("vision",)


def test_should_normalise_modality_alias_when_plugin_declares_image():
    """ADR-033 §8.2: raw data_modalities alias 'image' is normalised to canonical 'vision'."""

    class AliasPlugin:
        plugin_meta = {
            "schema_version": 1,
            "name": "ce-alias-plugin",
            "version": "0.1.0",
            "provider": "tests.fixtures",
            "capabilities": ("explain",),
            "modes": ("factual",),
            "tasks": ("classification",),
            "dependencies": (),
            "data_modalities": ("image",),
            "plugin_api_version": "1.0",
            "trusted": False,
        }

        def supports(self, model):
            return True

        def explain(self, model, x, **kw):
            raise NotImplementedError

    # Arrange
    ep = _make_ep("tests.mock.modality.alias", AliasPlugin)
    mock_eps = _MockEntryPoints([ep])

    # Act
    with patch(
        "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
        return_value=mock_eps,
    ):
        load_entrypoint_plugins(include_untrusted=True)

    # Assert
    descriptor = find_explanation_descriptor("tests.mock.modality.alias")
    assert descriptor is not None
    assert descriptor.metadata["data_modalities"] == ("vision",)


def test_should_reject_plugin_when_plugin_api_version_major_mismatches():
    """ADR-033 §1.iv.a: major version mismatch emits UserWarning and skips registration."""

    class BadVersionPlugin:
        plugin_meta = {
            "schema_version": 1,
            "name": "ce-bad-version-plugin",
            "version": "0.1.0",
            "provider": "tests.fixtures",
            "capabilities": ("explain",),
            "modes": ("factual",),
            "tasks": ("classification",),
            "dependencies": (),
            "data_modalities": ("vision",),
            "plugin_api_version": "2.0",
            "trusted": False,
        }

        def supports(self, model):
            return True

        def explain(self, model, x, **kw):
            raise NotImplementedError

    # Arrange
    ep = _make_ep("tests.mock.modality.bad", BadVersionPlugin)
    mock_eps = _MockEntryPoints([ep])

    # Act & Assert
    with pytest.warns(UserWarning, match="plugin_api_version.*major is incompatible"):
        with patch(
            "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
            return_value=mock_eps,
        ):
            load_entrypoint_plugins(include_untrusted=True)

    assert find_explanation_descriptor("tests.mock.modality.bad") is None


def test_should_default_plugin_api_version_to_1_0_when_key_is_absent():
    """plugin_api_version defaults to '1.0' when the key is absent from plugin_meta."""

    class NoApiVersionPlugin:
        plugin_meta = {
            "schema_version": 1,
            "name": "ce-no-api-version-plugin",
            "version": "0.1.0",
            "provider": "tests.fixtures",
            "capabilities": ("explain",),
            "modes": ("factual",),
            "tasks": ("classification",),
            "dependencies": (),
            "data_modalities": ("vision",),  # explicit, avoids DeprecationWarning
            "trusted": False,
        }

        def supports(self, model):
            return True

        def explain(self, model, x, **kw):
            raise NotImplementedError

    # Arrange
    ep = _make_ep("tests.mock.modality.default_api", NoApiVersionPlugin)
    mock_eps = _MockEntryPoints([ep])

    # Act
    with patch(
        "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
        return_value=mock_eps,
    ):
        load_entrypoint_plugins(include_untrusted=True)

    # Assert
    descriptor = find_explanation_descriptor("tests.mock.modality.default_api")
    assert descriptor is not None
    assert descriptor.metadata["plugin_api_version"] == "1.0"


def test_should_warn_and_accept_plugin_when_plugin_api_minor_is_newer():
    """ADR-033 §1.iv.b: newer minor/patch is accepted with a UserWarning signal."""

    class MinorVersionPlugin:
        plugin_meta = {
            "schema_version": 1,
            "name": "ce-minor-version-plugin",
            "version": "0.1.0",
            "provider": "tests.fixtures",
            "capabilities": ("explain",),
            "modes": ("factual",),
            "tasks": ("classification",),
            "dependencies": (),
            "data_modalities": ("vision",),
            "plugin_api_version": "1.1",
            "trusted": False,
        }

        def supports(self, model):
            return True

        def explain(self, model, x, **kw):
            raise NotImplementedError

    # Arrange
    ep = _make_ep("tests.mock.modality.minor", MinorVersionPlugin)
    mock_eps = _MockEntryPoints([ep])

    # Act
    with pytest.warns(UserWarning, match="forward-compatibility risk"):
        with patch(
            "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
            return_value=mock_eps,
        ):
            load_entrypoint_plugins(include_untrusted=True)

    # Assert
    descriptor = find_explanation_descriptor("tests.mock.modality.minor")
    assert descriptor is not None
    assert descriptor.metadata["plugin_api_version"] == "1.1"
