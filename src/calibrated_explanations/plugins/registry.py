"""Plugin registry (ADR-006 minimal, opt-in).

Explicit, in-process registry for explainer plugins. Users must call `register`
to add plugins. Discovery is local to this process; there is no I/O or import
side-effects here to reduce risk.
"""

from __future__ import annotations

import contextlib
from typing import Any, List, Tuple

from .base import ExplainerPlugin, validate_plugin_meta

_REGISTRY: List[ExplainerPlugin] = []

# Minimal trust store: only plugins explicitly trusted by the user are allowed
# to be returned by discovery helpers when trust is requested.
_TRUSTED: List[ExplainerPlugin] = []


def register(plugin: ExplainerPlugin) -> None:
    """Register a plugin after minimal metadata validation.

    Notes: Registering a plugin executes third-party code at import-time.
    Only register trusted plugins.
    """

    meta = getattr(plugin, "plugin_meta", None)
    validate_plugin_meta(meta)
    if plugin in _REGISTRY:
        return
    _REGISTRY.append(plugin)


def unregister(plugin: ExplainerPlugin) -> None:
    """Remove a plugin if present."""

    with contextlib.suppress(ValueError):
        _REGISTRY.remove(plugin)
    with contextlib.suppress(ValueError):
        _TRUSTED.remove(plugin)


def clear() -> None:
    """Clear all registered plugins (testing convenience)."""

    _REGISTRY.clear()


def list_plugins() -> Tuple[ExplainerPlugin, ...]:
    """Return a snapshot of registered plugins."""

    return tuple(_REGISTRY)


def trust_plugin(plugin: ExplainerPlugin) -> None:
    """Mark an already-registered plugin as trusted.

    Trust is an explicit, opt-in operation. The function validates metadata
    before adding to the trusted list. Only trusted plugins will be returned
    by :func:`find_for` when `trusted_only=True` is passed.
    """
    if plugin not in _REGISTRY:
        raise ValueError("Plugin must be registered before it can be trusted")
    meta = getattr(plugin, "plugin_meta", None)
    validate_plugin_meta(meta)
    if plugin in _TRUSTED:
        return
    _TRUSTED.append(plugin)


def untrust_plugin(plugin: ExplainerPlugin) -> None:
    """Remove a plugin from the trusted set if present."""
    with contextlib.suppress(ValueError):
        _TRUSTED.remove(plugin)


def find_for(model: Any) -> Tuple[ExplainerPlugin, ...]:
    """Find plugins that declare support for the given model."""

    return tuple(p for p in _REGISTRY if _safe_supports(p, model))


def find_for_trusted(model: Any) -> Tuple[ExplainerPlugin, ...]:
    """Find trusted plugins that declare support for the given model."""

    return tuple(p for p in _TRUSTED if _safe_supports(p, model))


def _safe_supports(plugin: ExplainerPlugin, model: Any) -> bool:
    try:
        return bool(plugin.supports(model))
    except Exception:
        return False


__all__ = [
    "register",
    "unregister",
    "clear",
    "list_plugins",
    "trust_plugin",
    "untrust_plugin",
    "find_for",
    "find_for_trusted",
]
