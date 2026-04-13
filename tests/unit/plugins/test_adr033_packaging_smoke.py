"""ADR-033 packaging smoke tests: entry-point discovery, modality contract, plugin_api_version."""
from __future__ import annotations

import importlib.metadata as importlib_metadata
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Insert the fixture package into sys.path so MinimalModalityPlugin is importable.
_FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "ce_mock_modality_plugin"
if str(_FIXTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FIXTURE_ROOT))

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
    clear_trust_warnings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ep(name: str, module_name: str, class_name: str) -> importlib_metadata.EntryPoint:
    """Return a real EntryPoint object to keep smoke tests close to packaging reality."""
    return importlib_metadata.EntryPoint(
        name=name,
        value=f"{module_name}:{class_name}",
        group="calibrated_explanations.plugins",
    )


class MockEntryPoints:
    """Minimal stand-in for importlib.metadata.EntryPoints."""

    def __init__(self, eps: list[importlib_metadata.EntryPoint]) -> None:
        self.eps = eps

    def select(self, *, group: str) -> list:
        return self.eps if group == "calibrated_explanations.plugins" else []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry_and_policy():
    transient_modules = [name for name in sys.modules if name.startswith("tests_mock_")]
    for name in transient_modules:
        sys.modules.pop(name, None)
    clear_explanation_plugins()
    clear_legacy_registry()
    clear_trust_warnings()
    set_trust_policy(DefaultPluginTrustPolicy())
    yield
    transient_modules = [name for name in sys.modules if name.startswith("tests_mock_")]
    for name in transient_modules:
        sys.modules.pop(name, None)
    clear_explanation_plugins()
    clear_legacy_registry()
    clear_trust_warnings()
    set_trust_policy(DefaultPluginTrustPolicy())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_should_register_valid_plugin_when_entry_point_is_discovered():
    """Entry-point discovery registers a MinimalModalityPlugin in the explanation catalog."""
    # Arrange
    ep = make_ep("tests.mock.modality", "ce_mock_modality_plugin", "MinimalModalityPlugin")
    mock_eps = MockEntryPoints([ep])

    # Act
    with patch(
        "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
        return_value=mock_eps,
    ):
        load_entrypoint_plugins(include_untrusted=True)

    # Assert
    identifier = "ce_mock_modality_plugin:MinimalModalityPlugin"
    descriptor = find_explanation_descriptor(identifier)
    assert descriptor is not None
    report = get_last_discovery_report()
    assert report is not None
    accepted_ids = {record.identifier for record in report.accepted}
    assert identifier in accepted_ids


def test_should_preserve_data_modalities_when_entry_point_is_discovered():
    """Entry-point discovery preserves the declared data_modalities in the descriptor."""
    # Arrange
    ep = make_ep("tests.mock.modality", "ce_mock_modality_plugin", "MinimalModalityPlugin")
    mock_eps = MockEntryPoints([ep])

    # Act
    with patch(
        "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
        return_value=mock_eps,
    ):
        load_entrypoint_plugins(include_untrusted=True)

    # Assert
    descriptor = find_explanation_descriptor("ce_mock_modality_plugin:MinimalModalityPlugin")
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

    module = types.ModuleType("tests_mock_alias_plugin")
    module.AliasPlugin = AliasPlugin
    sys.modules[module.__name__] = module
    ep = make_ep("tests.mock.modality.alias", module.__name__, "AliasPlugin")
    mock_eps = MockEntryPoints([ep])

    # Act
    with patch(
        "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
        return_value=mock_eps,
    ):
        load_entrypoint_plugins(include_untrusted=True)

    # Assert
    descriptor = find_explanation_descriptor(f"{module.__name__}:AliasPlugin")
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

    module = types.ModuleType("tests_mock_bad_version_plugin")
    module.BadVersionPlugin = BadVersionPlugin
    sys.modules[module.__name__] = module
    ep = make_ep("tests.mock.modality.bad", module.__name__, "BadVersionPlugin")
    mock_eps = MockEntryPoints([ep])

    # Act & Assert
    with pytest.warns(UserWarning, match="plugin_api_version.*major is incompatible"):
        with patch(
            "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
            return_value=mock_eps,
        ):
            load_entrypoint_plugins(include_untrusted=True)

    assert find_explanation_descriptor(f"{module.__name__}:BadVersionPlugin") is None


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

    module = types.ModuleType("tests_mock_default_api_plugin")
    module.NoApiVersionPlugin = NoApiVersionPlugin
    sys.modules[module.__name__] = module
    ep = make_ep("tests.mock.modality.default_api", module.__name__, "NoApiVersionPlugin")
    mock_eps = MockEntryPoints([ep])

    # Act
    with patch(
        "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
        return_value=mock_eps,
    ):
        load_entrypoint_plugins(include_untrusted=True)

    # Assert
    descriptor = find_explanation_descriptor(f"{module.__name__}:NoApiVersionPlugin")
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

    module = types.ModuleType("tests_mock_minor_version_plugin")
    module.MinorVersionPlugin = MinorVersionPlugin
    sys.modules[module.__name__] = module
    ep = make_ep("tests.mock.modality.minor", module.__name__, "MinorVersionPlugin")
    mock_eps = MockEntryPoints([ep])

    # Act
    with pytest.warns(UserWarning, match="forward-compatibility risk"):
        with patch(
            "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
            return_value=mock_eps,
        ):
            load_entrypoint_plugins(include_untrusted=True)

    # Assert
    descriptor = find_explanation_descriptor(f"{module.__name__}:MinorVersionPlugin")
    assert descriptor is not None
    assert descriptor.metadata["plugin_api_version"] == "1.1"


def test_should_not_use_nonstandard_entrypoint_loader_fallbacks():
    """Discovery should fail closed when EntryPoint.load fails (no weak fallback paths)."""

    class FakeEntryPoint:
        name = "tests.mock.modality.fallback"
        module = "tests.mock.modality"
        attr = "FallbackPlugin"
        dist = None

        def load(self):
            raise RuntimeError("boom")

        # Legacy weak-fallback shapes that discovery must ignore.
        def loader(self):  # pragma: no cover - behavior asserted via registration outcome
            raise AssertionError("loader() should never be called")

        def _loader(self):  # pragma: no cover - behavior asserted via registration outcome
            raise AssertionError("_loader() should never be called")

    class EntryPoints:
        def select(self, *, group: str):
            return [FakeEntryPoint()] if group == "calibrated_explanations.plugins" else []

    with pytest.warns(UserWarning, match="Failed to load plugin entry point"):
        with patch(
            "calibrated_explanations.plugins.registry.importlib_metadata.entry_points",
            return_value=EntryPoints(),
        ):
            load_entrypoint_plugins(include_untrusted=True)

    assert find_explanation_descriptor("tests.mock.modality:FallbackPlugin") is None
