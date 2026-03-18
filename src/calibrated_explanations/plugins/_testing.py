"""Internal test support helpers for plugin registry behavior.

This module centralizes registry test scaffolding so tests can avoid direct
private-member access while production registry modules stay free of test
helper wrappers.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterable, Mapping

from . import registry
from ._trust import clear_trusted_identifiers, update_trusted_identifier
from .base import ExplainerPlugin


def clear_env_trust_cache() -> None:
    registry._ENV_TRUST_CACHE = None
    registry._PYPROJECT_TRUST_CACHE = None


def set_pyproject_trust_cache_for_testing(trusted: Iterable[str] | None) -> None:
    registry._PYPROJECT_TRUST_CACHE = None if trusted is None else set(trusted)


def clear_trust_warnings() -> None:
    registry._WARNED_UNTRUSTED.clear()


def normalise_trust(meta: Mapping[str, Any]) -> bool:
    return registry._normalise_trust(meta)


def env_trusted_names() -> set[str]:
    return registry._env_trusted_names()


def should_trust(meta: Mapping[str, Any], *, identifier: str, source: str) -> bool:
    return registry._should_trust(meta, identifier=identifier, source=source)


def propagate_trust_metadata(plugin: Any, meta: Mapping[str, Any]) -> None:
    registry._propagate_trust_metadata(plugin, meta)


def update_trust_keys(meta: dict[str, Any], trusted: bool) -> None:
    registry._update_trust_keys(meta, trusted)


def resolve_plugin_module_file(plugin: ExplainerPlugin):
    return registry._resolve_plugin_module_file(plugin)


def verify_plugin_checksum(plugin: ExplainerPlugin, meta: Mapping[str, Any]) -> None:
    registry._verify_plugin_checksum(plugin, meta)


def get_entrypoint_group() -> str:
    return registry._ENTRYPOINT_GROUP


def plot_styles() -> dict[str, Any]:
    return dict(registry._PLOT_STYLES)


def set_plot_style(identifier: str, descriptor: Any) -> None:
    registry._PLOT_STYLES[identifier] = descriptor


def clear_plot_styles() -> None:
    registry._PLOT_STYLES.clear()


def plot_builders() -> dict[str, Any]:
    return dict(registry._PLOT_BUILDERS)


def set_plot_builder(identifier: str, descriptor: Any, *, trusted: bool = False) -> None:
    registry._PLOT_BUILDERS[identifier] = descriptor
    update_trusted_identifier(registry._TRUSTED_PLOT_BUILDERS, identifier, trusted)
    registry._verify_trust_invariants_if_enabled()


def clear_plot_builders() -> None:
    registry._PLOT_BUILDERS.clear()
    clear_trusted_identifiers(registry._TRUSTED_PLOT_BUILDERS)


def plot_renderers() -> dict[str, Any]:
    return dict(registry._PLOT_RENDERERS)


def set_plot_renderer(identifier: str, descriptor: Any, *, trusted: bool = False) -> None:
    registry._PLOT_RENDERERS[identifier] = descriptor
    update_trusted_identifier(registry._TRUSTED_PLOT_RENDERERS, identifier, trusted)
    registry._verify_trust_invariants_if_enabled()


def clear_plot_renderers() -> None:
    registry._PLOT_RENDERERS.clear()
    clear_trusted_identifiers(registry._TRUSTED_PLOT_RENDERERS)


def registry_snapshot():
    return tuple(registry._REGISTRY)


def append_to_registry(plugin: ExplainerPlugin) -> None:
    if plugin not in registry._REGISTRY:
        registry._REGISTRY.append(plugin)


def remove_from_registry(plugin: ExplainerPlugin) -> None:
    with contextlib.suppress(ValueError):
        registry._REGISTRY.remove(plugin)


def resolve_plugin_from_name(name: str) -> ExplainerPlugin:
    return registry._resolve_plugin_from_name(name)


def safe_supports(plugin: ExplainerPlugin, model: Any) -> bool:
    return registry._safe_supports(plugin, model)


def warn_untrusted_plugin(meta: Mapping[str, Any], *, source: str) -> None:
    registry._warn_untrusted_plugin(meta, source=source)


def ensure_sequence(
    meta: Mapping[str, Any],
    key: str,
    *,
    allowed: Iterable[str] | None = None,
    allow_empty: bool = False,
):
    return registry._ensure_sequence(meta, key, allowed=allowed, allow_empty=allow_empty)


def coerce_string_collection(value: Any, *, key: str | None = None):
    return registry._coerce_string_collection(value, key=key)


def normalise_dependency_field(
    value: Any,
    key: str,
    *,
    optional: bool = False,
    allow_empty: bool = False,
):
    return registry._normalise_dependency_field(
        value,
        key,
        optional=optional,
        allow_empty=allow_empty,
    )


def normalise_tasks(value: Any):
    return registry._normalise_tasks(value)


def ensure_bool(value: Mapping[str, Any], key: str) -> bool:
    return registry._ensure_bool(value, key)


def ensure_string(value: Mapping[str, Any], key: str) -> str:
    return registry._ensure_string(value, key)
