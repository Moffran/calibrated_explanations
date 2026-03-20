"""Internal test support helpers for plugin registry behavior.

This module centralizes registry test scaffolding so tests can avoid direct
private-member access while production registry modules stay free of test
helper wrappers.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterable, Mapping

from . import registry
from ._trust import clear_trusted_identifiers, mutate_trust_atomic, update_trusted_identifier
from .base import ExplainerPlugin


def clear_env_trust_cache() -> None:
    """Clear cached trust identifiers loaded from environment and pyproject."""
    registry._ENV_TRUST_CACHE = None
    registry._PYPROJECT_TRUST_CACHE = None


def set_pyproject_trust_cache_for_testing(trusted: Iterable[str] | None) -> None:
    """Override cached pyproject trusted identifiers for deterministic tests."""
    registry._PYPROJECT_TRUST_CACHE = None if trusted is None else set(trusted)


def clear_trust_warnings() -> None:
    """Reset the set of untrusted plugin warnings emitted during discovery."""
    registry._WARNED_UNTRUSTED.clear()


def normalise_trust(meta: Mapping[str, Any]) -> bool:
    """Return normalized boolean trust state from plugin metadata."""
    return registry._normalise_trust(meta)


def env_trusted_names() -> set[str]:
    """Return trusted plugin identifiers from the environment cache."""
    return registry._env_trusted_names()


def should_trust(meta: Mapping[str, Any], *, identifier: str, source: str) -> bool:
    """Return whether a plugin should be trusted under active trust policy."""
    return registry._should_trust(meta, identifier=identifier, source=source)


def propagate_trust_metadata(plugin: Any, meta: Mapping[str, Any]) -> None:
    """Propagate trust keys from normalized metadata onto plugin metadata."""
    registry._propagate_trust_metadata(plugin, meta)


def update_trust_keys(meta: dict[str, Any], trusted: bool) -> None:
    """Update trust-related metadata keys to a consistent boolean state."""
    registry._update_trust_keys(meta, trusted)


def resolve_plugin_module_file(plugin: ExplainerPlugin):
    """Resolve a plugin module path used for checksum verification tests."""
    return registry._resolve_plugin_module_file(plugin)


def verify_plugin_checksum(plugin: ExplainerPlugin, meta: Mapping[str, Any]) -> None:
    """Verify plugin checksum metadata against its module source when possible."""
    registry._verify_plugin_checksum(plugin, meta)


def get_entrypoint_group() -> str:
    """Return the entry-point group used for plugin discovery."""
    return registry._ENTRYPOINT_GROUP


def plot_styles() -> dict[str, Any]:
    """Return a copy of registered plot style descriptors."""
    return dict(registry._PLOT_STYLES)


def set_plot_style(identifier: str, descriptor: Any) -> None:
    """Register or overwrite a plot style descriptor in test scaffolding."""
    registry._PLOT_STYLES[identifier] = descriptor


def clear_plot_styles() -> None:
    """Clear all registered plot style descriptors."""
    registry._PLOT_STYLES.clear()


def plot_builders() -> dict[str, Any]:
    """Return a copy of registered plot builder descriptors."""
    return dict(registry._PLOT_BUILDERS)


def set_plot_builder(identifier: str, descriptor: Any, *, trusted: bool = False) -> None:
    """Register a plot builder descriptor and synchronize trusted identifier state."""

    def _mutation() -> None:
        registry._PLOT_BUILDERS[identifier] = descriptor
        update_trusted_identifier(registry._TRUSTED_PLOT_BUILDERS, identifier, trusted)

    mutate_trust_atomic(
        identifier=identifier,
        trusted=bool(trusted),
        actor="testing.set_plot_builder",
        kind="plot_builder",
        source="tests",
        mutation=_mutation,
        verify=registry._verify_trust_invariants_if_enabled,
    )


def clear_plot_builders() -> None:
    """Clear registered plot builders and trusted-builder identifiers."""

    def _mutation() -> None:
        registry._PLOT_BUILDERS.clear()
        clear_trusted_identifiers(registry._TRUSTED_PLOT_BUILDERS)

    mutate_trust_atomic(
        identifier="testing.plot_builders.clear",
        trusted=False,
        actor="testing.clear_plot_builders",
        kind="plot_builder",
        source="tests",
        mutation=_mutation,
        verify=registry._verify_trust_invariants_if_enabled,
    )


def plot_renderers() -> dict[str, Any]:
    """Return a copy of registered plot renderer descriptors."""
    return dict(registry._PLOT_RENDERERS)


def set_plot_renderer(identifier: str, descriptor: Any, *, trusted: bool = False) -> None:
    """Register a plot renderer descriptor and synchronize trust state."""

    def _mutation() -> None:
        registry._PLOT_RENDERERS[identifier] = descriptor
        update_trusted_identifier(registry._TRUSTED_PLOT_RENDERERS, identifier, trusted)

    mutate_trust_atomic(
        identifier=identifier,
        trusted=bool(trusted),
        actor="testing.set_plot_renderer",
        kind="plot_renderer",
        source="tests",
        mutation=_mutation,
        verify=registry._verify_trust_invariants_if_enabled,
    )


def clear_plot_renderers() -> None:
    """Clear registered plot renderers and trusted-renderer identifiers."""

    def _mutation() -> None:
        registry._PLOT_RENDERERS.clear()
        clear_trusted_identifiers(registry._TRUSTED_PLOT_RENDERERS)

    mutate_trust_atomic(
        identifier="testing.plot_renderers.clear",
        trusted=False,
        actor="testing.clear_plot_renderers",
        kind="plot_renderer",
        source="tests",
        mutation=_mutation,
        verify=registry._verify_trust_invariants_if_enabled,
    )


def registry_snapshot():
    """Return an immutable snapshot of the legacy registry list."""
    return tuple(registry._REGISTRY)


def append_to_registry(plugin: ExplainerPlugin) -> None:
    """Add plugin to legacy registry list if absent."""
    if plugin not in registry._REGISTRY:
        registry._REGISTRY.append(plugin)


def remove_from_registry(plugin: ExplainerPlugin) -> None:
    """Remove plugin from legacy registry list when present."""
    with contextlib.suppress(ValueError):
        registry._REGISTRY.remove(plugin)


def resolve_plugin_from_name(name: str) -> ExplainerPlugin:
    """Resolve plugin object by human-readable name."""
    return registry._resolve_plugin_from_name(name)


def safe_supports(plugin: ExplainerPlugin, model: Any) -> bool:
    """Return plugin support check while suppressing plugin-raised exceptions."""
    return registry._safe_supports(plugin, model)


def warn_untrusted_plugin(meta: Mapping[str, Any], *, source: str) -> None:
    """Emit a one-time warning for an untrusted plugin metadata payload."""
    registry._warn_untrusted_plugin(meta, source=source)


def ensure_sequence(
    meta: Mapping[str, Any],
    key: str,
    *,
    allowed: Iterable[str] | None = None,
    allow_empty: bool = False,
):
    """Validate a metadata field is a sequence of strings."""
    return registry._ensure_sequence(meta, key, allowed=allowed, allow_empty=allow_empty)


def coerce_string_collection(value: Any, *, key: str | None = None):
    """Coerce metadata value into a tuple of strings."""
    return registry._coerce_string_collection(value, key=key)


def normalise_dependency_field(
    value: Any,
    key: str,
    *,
    optional: bool = False,
    allow_empty: bool = False,
):
    """Normalize dependency-like metadata field to canonical tuple form."""
    return registry._normalise_dependency_field(
        value,
        key,
        optional=optional,
        allow_empty=allow_empty,
    )


def normalise_tasks(value: Any):
    """Normalize plugin task metadata into canonical allowed values."""
    return registry._normalise_tasks(value)


def ensure_bool(value: Mapping[str, Any], key: str) -> bool:
    """Read metadata key and ensure it is a boolean."""
    return registry._ensure_bool(value, key)


def ensure_string(value: Mapping[str, Any], key: str) -> str:
    """Read metadata key and ensure it is a string."""
    return registry._ensure_string(value, key)
