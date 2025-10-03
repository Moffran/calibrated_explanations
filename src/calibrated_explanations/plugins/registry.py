"""Plugin registry (ADR-006 minimal, opt-in).

Explicit, in-process registry for explainer plugins. Users must call
``register`` to add plugins. Discovery is local to this process; there is no
I/O or import side-effects here to reduce risk.

This module now also exposes identifier-based registries for explanation,
interval, and plot plugins as outlined by ADR-013/ADR-014/ADR-015. The legacy
list-based helpers remain available for the interim so callers can migrate
incrementally.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .base import ExplainerPlugin, validate_plugin_meta

_REGISTRY: List[ExplainerPlugin] = []

# Minimal trust store: only plugins explicitly trusted by the user are allowed
# to be returned by discovery helpers when trust is requested.
_TRUSTED: List[ExplainerPlugin] = []


def _freeze_meta(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable copy of plugin metadata."""

    return MappingProxyType(dict(meta))


def _normalise_trust(meta: Mapping[str, Any]) -> bool:
    """Extract the trusted-by-default flag from metadata."""

    trust = meta.get("trust", False)
    if isinstance(trust, Mapping):
        # Accept a couple of common patterns without committing to a schema
        if "trusted" in trust:
            return bool(trust["trusted"])
        if "default" in trust:
            return bool(trust["default"])
    return bool(trust)


def _ensure_sequence(
    meta: Mapping[str, Any],
    key: str,
    *,
    allowed: Iterable[str] | None = None,
    allow_empty: bool = False,
) -> Tuple[str, ...]:
    """Validate a metadata field is a sequence of strings."""

    if key not in meta:
        raise ValueError(f"plugin_meta missing required key: {key}")
    value = meta[key]
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(f"plugin_meta[{key!r}] must be a sequence of strings")

    collected: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"plugin_meta[{key!r}] must contain only string values"
            )
        collected.append(item)

    if not collected and not allow_empty:
        raise ValueError(f"plugin_meta[{key!r}] must not be empty")

    if allowed is not None:
        allowed_set = set(allowed)
        unknown = sorted(set(collected) - allowed_set)
        if unknown:
            raise ValueError(
                f"plugin_meta[{key!r}] has unsupported values: {', '.join(unknown)}"
            )

    return tuple(collected)


def _validate_dependencies(meta: Mapping[str, Any]) -> Tuple[str, ...]:
    """Validate dependency metadata as a sequence of identifiers."""

    return _ensure_sequence(meta, "dependencies", allow_empty=True)


def validate_explanation_metadata(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate ADR-015 metadata requirements for explanation plugins."""

    modes = _ensure_sequence(
        meta,
        "modes",
        allowed={
            "explanation:factual",
            "explanation:alternative",
            "explanation:fast",
        },
    )
    if not modes:
        raise ValueError("explanation plugin must declare at least one mode")

    _ensure_sequence(meta, "capabilities", allow_empty=False)
    _validate_dependencies(meta)
    # Trust flags can be bool or mapping; ensure the key exists for explicitness
    if "trust" not in meta:
        raise ValueError("plugin_meta missing required key: trust")
    return meta


def validate_interval_metadata(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate ADR-013 metadata requirements for interval plugins."""

    modes = _ensure_sequence(
        meta,
        "modes",
        allowed={"classification", "regression"},
    )
    if not modes:
        raise ValueError("interval plugin must declare at least one mode")

    _ensure_sequence(meta, "capabilities", allow_empty=False)
    _validate_dependencies(meta)
    if "trust" not in meta:
        raise ValueError("plugin_meta missing required key: trust")
    return meta


def validate_plot_metadata(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate ADR-014 metadata requirements for plot plugins."""

    if "style" not in meta or not isinstance(meta["style"], str):
        raise ValueError("plugin_meta missing required key: style (str)")
    _ensure_sequence(meta, "capabilities", allow_empty=False)
    _validate_dependencies(meta)
    if "trust" not in meta:
        raise ValueError("plugin_meta missing required key: trust")
    # ``output_formats`` and ``default_renderer`` are optional at this stage but
    # when present they must be strings/sequence of strings.
    if "output_formats" in meta:
        _ensure_sequence(meta, "output_formats", allow_empty=False)
    if "default_renderer" in meta and not isinstance(meta["default_renderer"], str):
        raise ValueError(
            "plugin_meta['default_renderer'] must be a string when provided"
        )
    return meta


@dataclass(frozen=True)
class ExplanationPluginDescriptor:
    """Descriptor for explanation plugins keyed by identifier."""

    identifier: str
    plugin: ExplainerPlugin
    metadata: Mapping[str, Any] = field(repr=False)
    trusted: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "metadata", _freeze_meta(self.metadata))


@dataclass(frozen=True)
class IntervalPluginDescriptor:
    """Descriptor for interval calibrator plugins."""

    identifier: str
    plugin: Any
    metadata: Mapping[str, Any] = field(repr=False)
    trusted: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "metadata", _freeze_meta(self.metadata))


@dataclass(frozen=True)
class PlotPluginDescriptor:
    """Descriptor for plot plugins (builder/renderer/style entries)."""

    identifier: str
    plugin: Any
    metadata: Mapping[str, Any] = field(repr=False)
    trusted: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "metadata", _freeze_meta(self.metadata))


_EXPLANATION_PLUGINS: Dict[str, ExplanationPluginDescriptor] = {}
_TRUSTED_EXPLANATIONS: set[str] = set()

_INTERVAL_PLUGINS: Dict[str, IntervalPluginDescriptor] = {}
_TRUSTED_INTERVALS: set[str] = set()

_PLOT_PLUGINS: Dict[str, PlotPluginDescriptor] = {}
_TRUSTED_PLOTS: set[str] = set()


def clear_explanation_plugins() -> None:
    """Clear explanation plugin descriptors (testing helper)."""

    _EXPLANATION_PLUGINS.clear()
    _TRUSTED_EXPLANATIONS.clear()


def clear_interval_plugins() -> None:
    """Clear interval plugin descriptors (testing helper)."""

    _INTERVAL_PLUGINS.clear()
    _TRUSTED_INTERVALS.clear()


def clear_plot_plugins() -> None:
    """Clear plot plugin descriptors (testing helper)."""

    _PLOT_PLUGINS.clear()
    _TRUSTED_PLOTS.clear()


def register_explanation_plugin(
    identifier: str,
    plugin: ExplainerPlugin,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ExplanationPluginDescriptor:
    """Register an explanation plugin under the given identifier."""

    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier must be a non-empty string")
    raw_meta = metadata or getattr(plugin, "plugin_meta", None)
    if raw_meta is None:
        raise ValueError("plugin must expose plugin_meta metadata")
    meta: Dict[str, Any] = dict(raw_meta)
    validate_plugin_meta(meta)
    validate_explanation_metadata(meta)
    trusted = _normalise_trust(meta)

    descriptor = ExplanationPluginDescriptor(
        identifier=identifier,
        plugin=plugin,
        metadata=meta,
        trusted=trusted,
    )
    _EXPLANATION_PLUGINS[identifier] = descriptor
    if trusted:
        _TRUSTED_EXPLANATIONS.add(identifier)
    else:
        _TRUSTED_EXPLANATIONS.discard(identifier)

    # Maintain backwards compatibility with the legacy list registry.
    register(plugin)
    if trusted:
        trust_plugin(plugin)

    return descriptor


def find_explanation_descriptor(identifier: str) -> ExplanationPluginDescriptor | None:
    """Return the explanation plugin descriptor for *identifier* if present."""

    return _EXPLANATION_PLUGINS.get(identifier)


def find_explanation_plugin(identifier: str) -> ExplainerPlugin | None:
    """Return the explanation plugin instance for *identifier* if present."""

    descriptor = find_explanation_descriptor(identifier)
    return descriptor.plugin if descriptor else None


def find_explanation_plugin_trusted(identifier: str) -> ExplainerPlugin | None:
    """Return the trusted explanation plugin instance for *identifier* if any."""

    descriptor = find_explanation_descriptor(identifier)
    if descriptor and descriptor.trusted:
        return descriptor.plugin
    return None


def register_interval_plugin(
    identifier: str,
    plugin: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> IntervalPluginDescriptor:
    """Register an interval plugin descriptor."""

    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier must be a non-empty string")
    raw_meta = metadata or getattr(plugin, "plugin_meta", None)
    if raw_meta is None:
        raise ValueError("plugin must expose plugin_meta metadata")
    meta: Dict[str, Any] = dict(raw_meta)
    validate_plugin_meta(meta)
    validate_interval_metadata(meta)
    trusted = _normalise_trust(meta)

    descriptor = IntervalPluginDescriptor(
        identifier=identifier,
        plugin=plugin,
        metadata=meta,
        trusted=trusted,
    )
    _INTERVAL_PLUGINS[identifier] = descriptor
    if trusted:
        _TRUSTED_INTERVALS.add(identifier)
    else:
        _TRUSTED_INTERVALS.discard(identifier)
    return descriptor


def find_interval_descriptor(identifier: str) -> IntervalPluginDescriptor | None:
    """Return the descriptor for an interval plugin by identifier."""

    return _INTERVAL_PLUGINS.get(identifier)


def find_interval_plugin(identifier: str) -> Any | None:
    """Return the interval plugin instance for *identifier* if registered."""

    descriptor = find_interval_descriptor(identifier)
    return descriptor.plugin if descriptor else None


def find_interval_plugin_trusted(identifier: str) -> Any | None:
    """Return the trusted interval plugin instance when available."""

    descriptor = find_interval_descriptor(identifier)
    if descriptor and descriptor.trusted:
        return descriptor.plugin
    return None


def register_plot_plugin(
    identifier: str,
    plugin: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PlotPluginDescriptor:
    """Register a plot plugin (builder/renderer/style) under an identifier."""

    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier must be a non-empty string")
    raw_meta = metadata or getattr(plugin, "plugin_meta", None)
    if raw_meta is None:
        raise ValueError("plugin must expose plugin_meta metadata")
    meta: Dict[str, Any] = dict(raw_meta)
    validate_plugin_meta(meta)
    validate_plot_metadata(meta)
    trusted = _normalise_trust(meta)

    descriptor = PlotPluginDescriptor(
        identifier=identifier,
        plugin=plugin,
        metadata=meta,
        trusted=trusted,
    )
    _PLOT_PLUGINS[identifier] = descriptor
    if trusted:
        _TRUSTED_PLOTS.add(identifier)
    else:
        _TRUSTED_PLOTS.discard(identifier)
    return descriptor


def find_plot_descriptor(identifier: str) -> PlotPluginDescriptor | None:
    """Return the plot plugin descriptor for *identifier* if present."""

    return _PLOT_PLUGINS.get(identifier)


def find_plot_plugin(identifier: str) -> Any | None:
    """Return the registered plot plugin instance for *identifier* if any."""

    descriptor = find_plot_descriptor(identifier)
    return descriptor.plugin if descriptor else None


def find_plot_plugin_trusted(identifier: str) -> Any | None:
    """Return the trusted plot plugin instance when available."""

    descriptor = find_plot_descriptor(identifier)
    if descriptor and descriptor.trusted:
        return descriptor.plugin
    return None


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
    "ExplanationPluginDescriptor",
    "IntervalPluginDescriptor",
    "PlotPluginDescriptor",
    "validate_explanation_metadata",
    "validate_interval_metadata",
    "validate_plot_metadata",
    "clear_explanation_plugins",
    "clear_interval_plugins",
    "clear_plot_plugins",
    "register_explanation_plugin",
    "register_interval_plugin",
    "register_plot_plugin",
    "find_explanation_descriptor",
    "find_interval_descriptor",
    "find_plot_descriptor",
    "find_explanation_plugin",
    "find_interval_plugin",
    "find_plot_plugin",
    "find_explanation_plugin_trusted",
    "find_interval_plugin_trusted",
    "find_plot_plugin_trusted",
    "register",
    "unregister",
    "clear",
    "list_plugins",
    "trust_plugin",
    "untrust_plugin",
    "find_for",
    "find_for_trusted",
]
