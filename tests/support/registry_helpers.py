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


def clear_explanation_plugins() -> None:
    registry.reset_plugin_catalog(kind="explanation")


def clear_interval_plugins() -> None:
    registry.reset_plugin_catalog(kind="interval")


def clear_plot_plugins() -> None:
    registry.reset_plugin_catalog(kind="plot")


def clear_env_trust_cache() -> None:
    registry_testing.clear_env_trust_cache()


def set_pyproject_trust_cache_for_testing(trusted: Iterable[str] | None) -> None:
    registry_testing.set_pyproject_trust_cache_for_testing(trusted)


def clear_trust_warnings() -> None:
    registry_testing.clear_trust_warnings()


def normalise_trust(meta: Mapping[str, Any]) -> bool:
    return registry_testing.normalise_trust(meta)


def env_trusted_names() -> set[str]:
    return registry_testing.env_trusted_names()


def should_trust(meta: Mapping[str, Any], *, identifier: str, source: str) -> bool:
    return registry_testing.should_trust(meta, identifier=identifier, source=source)


def propagate_trust_metadata(plugin: Any, meta: Mapping[str, Any]) -> None:
    registry_testing.propagate_trust_metadata(plugin, meta)


def update_trust_keys(meta: dict[str, Any], trusted: bool) -> None:
    registry_testing.update_trust_keys(meta, trusted)


def resolve_plugin_module_file(plugin: ExplainerPlugin):
    return registry_testing.resolve_plugin_module_file(plugin)


def verify_plugin_checksum(plugin: ExplainerPlugin, meta: Mapping[str, Any]) -> None:
    registry_testing.verify_plugin_checksum(plugin, meta)


def get_entrypoint_group() -> str:
    return registry_testing.get_entrypoint_group()


def plot_styles() -> dict[str, Any]:
    return registry_testing.plot_styles()


def set_plot_style(identifier: str, descriptor: Any) -> None:
    registry_testing.set_plot_style(identifier, descriptor)


def clear_plot_styles() -> None:
    registry_testing.clear_plot_styles()


def plot_builders() -> dict[str, Any]:
    return registry_testing.plot_builders()


def set_plot_builder(identifier: str, descriptor: Any, *, trusted: bool = False) -> None:
    registry_testing.set_plot_builder(identifier, descriptor, trusted=trusted)


def clear_plot_builders() -> None:
    registry_testing.clear_plot_builders()


def plot_renderers() -> dict[str, Any]:
    return registry_testing.plot_renderers()


def set_plot_renderer(identifier: str, descriptor: Any, *, trusted: bool = False) -> None:
    registry_testing.set_plot_renderer(identifier, descriptor, trusted=trusted)


def clear_plot_renderers() -> None:
    registry_testing.clear_plot_renderers()


def registry_snapshot():
    return registry_testing.registry_snapshot()


def append_to_registry(plugin: ExplainerPlugin) -> None:
    registry_testing.append_to_registry(plugin)


def remove_from_registry(plugin: ExplainerPlugin) -> None:
    registry_testing.remove_from_registry(plugin)


def resolve_plugin_from_name(name: str) -> ExplainerPlugin:
    return registry_testing.resolve_plugin_from_name(name)


def safe_supports(plugin: ExplainerPlugin, model: Any) -> bool:
    return registry_testing.safe_supports(plugin, model)


def warn_untrusted_plugin(meta: Mapping[str, Any], *, source: str) -> None:
    registry_testing.warn_untrusted_plugin(meta, source=source)


def ensure_sequence(
    meta: Mapping[str, Any],
    key: str,
    *,
    allowed: Iterable[str] | None = None,
    allow_empty: bool = False,
):
    return registry_testing.ensure_sequence(meta, key, allowed=allowed, allow_empty=allow_empty)


def coerce_string_collection(value: Any, *, key: str | None = None):
    return registry_testing.coerce_string_collection(value, key=key)


def normalise_dependency_field(
    value: Any,
    key: str,
    *,
    optional: bool = False,
    allow_empty: bool = False,
):
    return registry_testing.normalise_dependency_field(
        value,
        key,
        optional=optional,
        allow_empty=allow_empty,
    )


def normalise_tasks(value: Any):
    return registry_testing.normalise_tasks(value)


def ensure_bool(value: Mapping[str, Any], key: str) -> bool:
    return registry_testing.ensure_bool(value, key)


def ensure_string(value: Mapping[str, Any], key: str) -> str:
    return registry_testing.ensure_string(value, key)
