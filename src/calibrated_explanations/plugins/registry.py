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


def clear() -> None:
    """Clear all registered plugins (testing convenience)."""

    _REGISTRY.clear()


def list_plugins() -> Tuple[ExplainerPlugin, ...]:
    """Return a snapshot of registered plugins."""

    return tuple(_REGISTRY)


def find_for(model: Any) -> Tuple[ExplainerPlugin, ...]:
    """Find plugins that declare support for the given model."""

    return tuple(p for p in _REGISTRY if _safe_supports(p, model))


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
    "find_for",
]
