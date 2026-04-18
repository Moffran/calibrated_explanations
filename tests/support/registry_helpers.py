"""Registry helpers for tests.

This module exposes stable helper names for tests while delegating all
implementation to the internal plugin testing helper module.
"""

from __future__ import annotations

import importlib
from typing import Any, Iterable, Mapping

from calibrated_explanations.plugins import registry
from calibrated_explanations.plugins.base import ExplainerPlugin

registry_testing = importlib.import_module("calibrated_explanations.plugins._testing")


def clear_legacy_registry() -> None:
    """Clear the legacy _REGISTRY and _TRUSTED lists for test isolation."""
    registry_testing.clear_legacy_registry()


def clear_explanation_plugins() -> None:
    """Reset explanation plugin catalog for isolated tests."""
    registry.reset_plugin_catalog(kind="explanation")


def clear_interval_plugins() -> None:
    """Reset interval plugin catalog for isolated tests."""
    registry.reset_plugin_catalog(kind="interval")


def clear_plot_plugins() -> None:
    """Reset plot plugin catalog for isolated tests."""
    registry.reset_plugin_catalog(kind="plot")


def clear_env_trust_cache() -> None:
    """Clear cached trust identifiers used by registry trust checks."""
    registry_testing.clear_env_trust_cache()


def set_pyproject_trust_cache_for_testing(trusted: Iterable[str] | None) -> None:
    """Override pyproject trusted identifiers cache for deterministic tests."""
    registry_testing.set_pyproject_trust_cache_for_testing(trusted)


def clear_trust_warnings() -> None:
    """Clear one-time warning tracking for untrusted plugin messages."""
    registry_testing.clear_trust_warnings()


def normalise_trust(meta: Mapping[str, Any]) -> bool:
    """Normalize trust metadata from supported schema variants."""
    return registry_testing.normalise_trust(meta)


def env_trusted_names() -> set[str]:
    """Return trusted identifiers parsed from environment variables."""
    return registry_testing.env_trusted_names()


def should_trust(meta: Mapping[str, Any], *, identifier: str, source: str) -> bool:
    """Return whether plugin metadata resolves to trusted under current policy."""
    return registry_testing.should_trust(meta, identifier=identifier, source=source)


def propagate_trust_metadata(plugin: Any, meta: Mapping[str, Any]) -> None:
    """Propagate normalized trust flags to plugin metadata object."""
    registry_testing.propagate_trust_metadata(plugin, meta)


def update_trust_keys(meta: dict[str, Any], trusted: bool) -> None:
    """Update trust metadata keys to a consistent trusted/untrusted state."""
    registry_testing.update_trust_keys(meta, trusted)


def resolve_plugin_module_file(plugin: ExplainerPlugin):
    """Resolve plugin module file path for checksum-related tests."""
    return registry_testing.resolve_plugin_module_file(plugin)


def verify_plugin_checksum(plugin: ExplainerPlugin, meta: Mapping[str, Any]) -> None:
    """Verify plugin checksum metadata against plugin module source bytes."""
    registry_testing.verify_plugin_checksum(plugin, meta)


def get_entrypoint_group() -> str:
    """Return configured plugin discovery entry-point group name."""
    return registry_testing.get_entrypoint_group()


def plot_styles() -> dict[str, Any]:
    """Return a snapshot of registered plot style descriptors."""
    return registry_testing.plot_styles()


def set_plot_style(identifier: str, descriptor: Any) -> None:
    """Insert or replace a plot style descriptor for tests."""
    registry_testing.set_plot_style(identifier, descriptor)


def clear_plot_styles() -> None:
    """Clear all registered plot styles."""
    registry_testing.clear_plot_styles()


def plot_builders() -> dict[str, Any]:
    """Return a snapshot of registered plot builder descriptors."""
    return registry_testing.plot_builders()


def set_plot_builder(identifier: str, descriptor: Any, *, trusted: bool = False) -> None:
    """Insert or replace a plot builder descriptor for tests."""
    registry_testing.set_plot_builder(identifier, descriptor, trusted=trusted)


def clear_plot_builders() -> None:
    """Clear all registered plot builders."""
    registry_testing.clear_plot_builders()


def plot_renderers() -> dict[str, Any]:
    """Return a snapshot of registered plot renderer descriptors."""
    return registry_testing.plot_renderers()


def set_plot_renderer(identifier: str, descriptor: Any, *, trusted: bool = False) -> None:
    """Insert or replace a plot renderer descriptor for tests."""
    registry_testing.set_plot_renderer(identifier, descriptor, trusted=trusted)


def clear_plot_renderers() -> None:
    """Clear all registered plot renderers."""
    registry_testing.clear_plot_renderers()


def registry_snapshot():
    """Return immutable snapshot of legacy plugin registry list."""
    return registry_testing.registry_snapshot()


def append_to_registry(plugin: ExplainerPlugin) -> None:
    """Append plugin to legacy registry list if absent."""
    registry_testing.append_to_registry(plugin)


def remove_from_registry(plugin: ExplainerPlugin) -> None:
    """Remove plugin from legacy registry list when present."""
    registry_testing.remove_from_registry(plugin)


def resolve_plugin_from_name(name: str) -> ExplainerPlugin:
    """Resolve a plugin object by metadata name."""
    return registry_testing.resolve_plugin_from_name(name)


def safe_supports(plugin: ExplainerPlugin, model: Any) -> bool:
    """Return plugin support status while tolerating plugin exceptions."""
    return registry_testing.safe_supports(plugin, model)


def warn_untrusted_plugin(meta: Mapping[str, Any], *, source: str) -> None:
    """Emit one-time warning for an untrusted plugin metadata payload."""
    registry_testing.warn_untrusted_plugin(meta, source=source)


def ensure_sequence(
    meta: Mapping[str, Any],
    key: str,
    *,
    allowed: Iterable[str] | None = None,
    allow_empty: bool = False,
):
    """Validate metadata field as a sequence of strings."""
    return registry_testing.ensure_sequence(meta, key, allowed=allowed, allow_empty=allow_empty)


def coerce_string_collection(value: Any, *, key: str | None = None):
    """Coerce string or iterable of strings into canonical tuple."""
    return registry_testing.coerce_string_collection(value, key=key)


def normalise_dependency_field(
    value: Any,
    key: str,
    *,
    optional: bool = False,
    allow_empty: bool = False,
):
    """Normalize dependency-like metadata values into tuple form."""
    return registry_testing.normalise_dependency_field(
        value,
        key,
        optional=optional,
        allow_empty=allow_empty,
    )


def normalise_tasks(value: Any):
    """Normalize task declarations to supported canonical values."""
    return registry_testing.normalise_tasks(value)


def ensure_bool(value: Mapping[str, Any], key: str) -> bool:
    """Read and validate a boolean metadata key."""
    return registry_testing.ensure_bool(value, key)


def ensure_string(value: Mapping[str, Any], key: str) -> str:
    """Read and validate a string metadata key."""
    return registry_testing.ensure_string(value, key)
