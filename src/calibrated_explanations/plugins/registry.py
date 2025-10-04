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
import warnings
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


_EXPLANATION_PROTOCOL_VERSION = 1
EXPLANATION_PROTOCOL_VERSION = _EXPLANATION_PROTOCOL_VERSION

_EXPLANATION_MODE_ALIASES = {
    "explanation:factual": "factual",
    "explanation:alternative": "alternative",
    "explanation:fast": "fast",
}

_EXPLANATION_VALID_MODES = {"factual", "alternative", "fast"}


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


def _coerce_string_collection(
    value: Any,
    *,
    key: str,
    allow_empty: bool = False,
) -> Tuple[str, ...]:
    """Coerce *value* to a tuple of strings."""

    if isinstance(value, str):
        result = (value,)
    elif isinstance(value, Iterable):
        collected: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    f"plugin_meta[{key!r}] must contain only string values"
                )
            collected.append(item)
        result = tuple(collected)
    else:
        raise ValueError(
            f"plugin_meta[{key!r}] must be a string or sequence of strings"
        )

    if not result and not allow_empty:
        raise ValueError(f"plugin_meta[{key!r}] must not be empty")
    return result


def _normalise_dependency_field(
    meta: Dict[str, Any],
    key: str,
    *,
    optional: bool = False,
    allow_empty: bool = False,
) -> Tuple[str, ...] | None:
    """Validate dependency style metadata fields."""

    if key not in meta:
        if optional:
            return None
        raise ValueError(f"plugin_meta missing required key: {key}")

    value = meta[key]
    normalised = _coerce_string_collection(value, key=key, allow_empty=allow_empty)
    meta[key] = normalised
    return normalised


def _normalise_tasks(meta: Dict[str, Any]) -> Tuple[str, ...]:
    """Validate the tasks field for explanation plugins."""

    allowed_tasks = {"classification", "regression", "both"}
    if "tasks" not in meta:
        raise ValueError("plugin_meta missing required key: tasks")
    tasks_value = meta["tasks"]
    tasks = _coerce_string_collection(tasks_value, key="tasks")
    unknown = sorted(set(tasks) - allowed_tasks)
    if unknown:
        raise ValueError(
            "plugin_meta['tasks'] has unsupported values: " + ", ".join(unknown)
        )
    meta["tasks"] = tasks
    return tasks


def validate_explanation_metadata(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate ADR-015 metadata requirements for explanation plugins."""

    if not isinstance(meta, dict):
        meta = dict(meta)
    schema_version = meta.get("schema_version")
    if isinstance(schema_version, int) and schema_version > _EXPLANATION_PROTOCOL_VERSION:
        raise ValueError(
            "explanation plugin declares unsupported schema_version "
            f"{schema_version}; runtime supports {_EXPLANATION_PROTOCOL_VERSION}"
        )

    allowed_modes = set(_EXPLANATION_VALID_MODES) | set(_EXPLANATION_MODE_ALIASES)
    raw_modes = _ensure_sequence(meta, "modes", allowed=allowed_modes)
    normalised_modes: List[str] = []
    seen: set[str] = set()
    for mode in raw_modes:
        canonical = _EXPLANATION_MODE_ALIASES.get(mode, mode)
        if mode in _EXPLANATION_MODE_ALIASES:
            warnings.warn(
                "explanation mode alias '" + mode + "' is deprecated; use '"
                + canonical
                + "'",
                DeprecationWarning,
                stacklevel=2,
            )
        if canonical not in _EXPLANATION_VALID_MODES:
            raise ValueError(
                f"plugin_meta['modes'] has unsupported values: {canonical}"
            )
        if canonical not in seen:
            seen.add(canonical)
            normalised_modes.append(canonical)
    if not normalised_modes:
        raise ValueError("explanation plugin must declare at least one mode")
    meta["modes"] = tuple(normalised_modes)

    meta["capabilities"] = _ensure_sequence(meta, "capabilities", allow_empty=False)
    meta["dependencies"] = _validate_dependencies(meta)
    _normalise_tasks(meta)
    _normalise_dependency_field(meta, "interval_dependency", optional=True)
    _normalise_dependency_field(meta, "plot_dependency", optional=True)
    _normalise_dependency_field(meta, "fallbacks", optional=True, allow_empty=True)
    # Trust flags can be bool or mapping; ensure the key exists for explicitness
    if "trust" not in meta:
        raise ValueError("plugin_meta missing required key: trust")
    return meta


def _ensure_bool(meta: Mapping[str, Any], key: str) -> bool:
    """Return *key* from *meta* ensuring it is a boolean."""

    if key not in meta:
        raise ValueError(f"plugin_meta missing required key: {key}")
    value = meta[key]
    if isinstance(value, bool):
        return value
    raise ValueError(f"plugin_meta[{key!r}] must be a boolean")


def _ensure_string(meta: Mapping[str, Any], key: str) -> str:
    """Return *key* from *meta* ensuring it is a string."""

    if key not in meta:
        raise ValueError(f"plugin_meta missing required key: {key}")
    value = meta[key]
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"plugin_meta[{key!r}] must be a non-empty string")


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
    _ensure_bool(meta, "fast_compatible")
    _ensure_bool(meta, "requires_bins")
    _ensure_string(meta, "confidence_source")
    if "legacy_compatible" in meta:
        _ensure_bool(meta, "legacy_compatible")
    if "trust" not in meta:
        raise ValueError("plugin_meta missing required key: trust")
    return meta


def validate_plot_builder_metadata(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate ADR-014 metadata requirements for plot builders."""

    _ensure_string(meta, "style")
    _ensure_sequence(meta, "capabilities", allow_empty=False)
    _validate_dependencies(meta)
    _ensure_bool(meta, "legacy_compatible")
    _ensure_sequence(meta, "output_formats", allow_empty=False)
    if "trust" not in meta:
        raise ValueError("plugin_meta missing required key: trust")
    return meta


def validate_plot_renderer_metadata(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate ADR-014 metadata requirements for plot renderers."""

    _ensure_sequence(meta, "capabilities", allow_empty=False)
    _validate_dependencies(meta)
    _ensure_sequence(meta, "output_formats", allow_empty=False)
    _ensure_bool(meta, "supports_interactive")
    if "trust" not in meta:
        raise ValueError("plugin_meta missing required key: trust")
    return meta


def validate_plot_style_metadata(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate metadata for plot style registrations."""

    _ensure_string(meta, "style")
    builder = _ensure_string(meta, "builder_id")
    renderer = _ensure_string(meta, "renderer_id")
    if builder == renderer:
        # no restriction; ensure they are non-empty strings only
        pass
    fallbacks = meta.get("fallbacks", ())
    if fallbacks:
        if isinstance(fallbacks, str):
            fallbacks = (fallbacks,)
        elif isinstance(fallbacks, Iterable):
            normalised: list[str] = []
            for item in fallbacks:
                if not isinstance(item, str) or not item:
                    raise ValueError(
                        "plugin_meta['fallbacks'] must contain non-empty strings"
                    )
                normalised.append(item)
            fallbacks = tuple(normalised)
        else:
            raise ValueError(
                "plugin_meta['fallbacks'] must be a string or sequence of strings"
            )
    else:
        fallbacks = tuple()
    meta = dict(meta)
    meta["fallbacks"] = fallbacks
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


_EXPLANATION_PLUGINS: Dict[str, ExplanationPluginDescriptor] = {}
_TRUSTED_EXPLANATIONS: set[str] = set()

_INTERVAL_PLUGINS: Dict[str, IntervalPluginDescriptor] = {}
_TRUSTED_INTERVALS: set[str] = set()

@dataclass(frozen=True)
class PlotBuilderDescriptor:
    """Descriptor for plot builders."""

    identifier: str
    builder: Any
    metadata: Mapping[str, Any] = field(repr=False)
    trusted: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "metadata", _freeze_meta(self.metadata))


@dataclass(frozen=True)
class PlotRendererDescriptor:
    """Descriptor for plot renderers."""

    identifier: str
    renderer: Any
    metadata: Mapping[str, Any] = field(repr=False)
    trusted: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "metadata", _freeze_meta(self.metadata))


@dataclass(frozen=True)
class PlotStyleDescriptor:
    """Descriptor mapping styles to builders and renderers."""

    identifier: str
    metadata: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "metadata", _freeze_meta(self.metadata))


_PLOT_BUILDERS: Dict[str, PlotBuilderDescriptor] = {}
_TRUSTED_PLOT_BUILDERS: set[str] = set()

_PLOT_RENDERERS: Dict[str, PlotRendererDescriptor] = {}
_TRUSTED_PLOT_RENDERERS: set[str] = set()

_PLOT_STYLES: Dict[str, PlotStyleDescriptor] = {}


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

    _PLOT_BUILDERS.clear()
    _TRUSTED_PLOT_BUILDERS.clear()
    _PLOT_RENDERERS.clear()
    _TRUSTED_PLOT_RENDERERS.clear()
    _PLOT_STYLES.clear()


def ensure_builtin_plugins() -> None:
    """Re-register in-tree plugins when registries have been cleared."""

    expected_explanations = {
        "core.explanation.factual",
        "core.explanation.alternative",
        "core.explanation.fast",
    }
    expected_intervals = {"core.interval.legacy", "core.interval.fast"}
    expected_plot_builders = {"core.plot.legacy"}
    expected_plot_renderers = {"core.plot.legacy"}
    expected_plot_styles = {"legacy"}

    missing = any(identifier not in _EXPLANATION_PLUGINS for identifier in expected_explanations)
    missing = missing or any(
        identifier not in _INTERVAL_PLUGINS for identifier in expected_intervals
    )
    missing = missing or any(
        identifier not in _PLOT_BUILDERS for identifier in expected_plot_builders
    )
    missing = missing or any(
        identifier not in _PLOT_RENDERERS for identifier in expected_plot_renderers
    )
    missing = missing or any(
        identifier not in _PLOT_STYLES for identifier in expected_plot_styles
    )

    if not missing:
        return

    from . import builtins as _builtins  # Local import avoids circular dependency

    _builtins._register_builtins()


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
    meta = validate_explanation_metadata(meta)
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


def register_plot_builder(
    identifier: str,
    builder: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PlotBuilderDescriptor:
    """Register a plot builder under *identifier*."""

    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier must be a non-empty string")
    raw_meta = metadata or getattr(builder, "plugin_meta", None)
    if raw_meta is None:
        raise ValueError("builder must expose plugin_meta metadata")
    meta: Dict[str, Any] = dict(raw_meta)
    validate_plugin_meta(meta)
    validate_plot_builder_metadata(meta)
    trusted = _normalise_trust(meta)

    descriptor = PlotBuilderDescriptor(
        identifier=identifier,
        builder=builder,
        metadata=meta,
        trusted=trusted,
    )
    _PLOT_BUILDERS[identifier] = descriptor
    if trusted:
        _TRUSTED_PLOT_BUILDERS.add(identifier)
    else:
        _TRUSTED_PLOT_BUILDERS.discard(identifier)
    return descriptor


def register_plot_renderer(
    identifier: str,
    renderer: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PlotRendererDescriptor:
    """Register a plot renderer under *identifier*."""

    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier must be a non-empty string")
    raw_meta = metadata or getattr(renderer, "plugin_meta", None)
    if raw_meta is None:
        raise ValueError("renderer must expose plugin_meta metadata")
    meta: Dict[str, Any] = dict(raw_meta)
    validate_plugin_meta(meta)
    validate_plot_renderer_metadata(meta)
    trusted = _normalise_trust(meta)

    descriptor = PlotRendererDescriptor(
        identifier=identifier,
        renderer=renderer,
        metadata=meta,
        trusted=trusted,
    )
    _PLOT_RENDERERS[identifier] = descriptor
    if trusted:
        _TRUSTED_PLOT_RENDERERS.add(identifier)
    else:
        _TRUSTED_PLOT_RENDERERS.discard(identifier)
    return descriptor


def register_plot_style(
    identifier: str,
    *,
    metadata: Mapping[str, Any],
) -> PlotStyleDescriptor:
    """Register a style entry that maps to builder and renderer identifiers."""

    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier must be a non-empty string")
    if metadata is None:
        raise ValueError("metadata is required for style registration")
    meta: Dict[str, Any] = dict(metadata)
    validate_plot_style_metadata(meta)
    if meta.get("style") != identifier:
        meta.setdefault("style", identifier)
    descriptor = PlotStyleDescriptor(identifier=identifier, metadata=meta)
    _PLOT_STYLES[identifier] = descriptor
    return descriptor


def find_plot_builder_descriptor(
    identifier: str,
) -> PlotBuilderDescriptor | None:
    """Return the builder descriptor for *identifier* if present."""

    return _PLOT_BUILDERS.get(identifier)


def find_plot_builder(identifier: str) -> Any | None:
    """Return the registered plot builder for *identifier* if any."""

    descriptor = find_plot_builder_descriptor(identifier)
    return descriptor.builder if descriptor else None


def find_plot_renderer_descriptor(
    identifier: str,
) -> PlotRendererDescriptor | None:
    """Return the renderer descriptor for *identifier* if present."""

    return _PLOT_RENDERERS.get(identifier)


def find_plot_renderer(identifier: str) -> Any | None:
    """Return the registered plot renderer for *identifier* if any."""

    descriptor = find_plot_renderer_descriptor(identifier)
    return descriptor.renderer if descriptor else None


def find_plot_style_descriptor(identifier: str) -> PlotStyleDescriptor | None:
    """Return the style descriptor for *identifier* if present."""

    return _PLOT_STYLES.get(identifier)


def find_plot_plugin(identifier: str) -> Any | None:
    """Return a combined plot plugin for the given style identifier."""
    
    style_descriptor = find_plot_style_descriptor(identifier)
    if style_descriptor is None:
        return None
    
    builder_id = style_descriptor.metadata.get("builder_id")
    renderer_id = style_descriptor.metadata.get("renderer_id")
    
    if not builder_id or not renderer_id:
        return None
    
    builder = find_plot_builder(builder_id)
    renderer = find_plot_renderer(renderer_id)
    
    if builder is None or renderer is None:
        return None
    
    # Create a combined plugin that has both build and render methods
    class CombinedPlotPlugin:
        def __init__(self, builder, renderer):
            self.builder = builder
            self.renderer = renderer
            self.plugin_meta = getattr(builder, "plugin_meta", {})
        
        def build(self, *args, **kwargs):
            return self.builder.build(*args, **kwargs)
        
        def render(self, *args, **kwargs):
            return self.renderer.render(*args, **kwargs)
    
    return CombinedPlotPlugin(builder, renderer)


def find_plot_plugin_trusted(identifier: str) -> Any | None:
    """Return a trusted combined plot plugin for the given style identifier."""
    
    style_descriptor = find_plot_style_descriptor(identifier)
    if style_descriptor is None:
        return None
    
    builder_id = style_descriptor.metadata.get("builder_id")
    renderer_id = style_descriptor.metadata.get("renderer_id")
    
    if not builder_id or not renderer_id:
        return None
    
    # Check if both builder and renderer are trusted
    builder_descriptor = find_plot_builder_descriptor(builder_id)
    renderer_descriptor = find_plot_renderer_descriptor(renderer_id)
    
    if (builder_descriptor is None or not builder_descriptor.trusted or
        renderer_descriptor is None or not renderer_descriptor.trusted):
        return None
    
    builder = builder_descriptor.builder
    renderer = renderer_descriptor.renderer
    
    # Create a combined plugin that has both build and render methods
    class CombinedPlotPlugin:
        def __init__(self, builder, renderer):
            self.builder = builder
            self.renderer = renderer
            self.plugin_meta = getattr(builder, "plugin_meta", {})
        
        def build(self, *args, **kwargs):
            return self.builder.build(*args, **kwargs)
        
        def render(self, *args, **kwargs):
            return self.renderer.render(*args, **kwargs)
    
    return CombinedPlotPlugin(builder, renderer)


def list_plot_builder_descriptors(
    *, trusted_only: bool = False
) -> Tuple[PlotBuilderDescriptor, ...]:
    """Return registered plot builder descriptors."""

    ensure_builtin_plugins()
    return _list_descriptors(
        _PLOT_BUILDERS,
        trusted_only,
        _TRUSTED_PLOT_BUILDERS,
    )


def list_plot_renderer_descriptors(
    *, trusted_only: bool = False
) -> Tuple[PlotRendererDescriptor, ...]:
    """Return registered plot renderer descriptors."""

    ensure_builtin_plugins()
    return _list_descriptors(
        _PLOT_RENDERERS,
        trusted_only,
        _TRUSTED_PLOT_RENDERERS,
    )


def list_plot_style_descriptors() -> Tuple[PlotStyleDescriptor, ...]:
    """Return registered plot style descriptors."""

    ensure_builtin_plugins()
    identifiers = sorted(_PLOT_STYLES.keys())
    return tuple(_PLOT_STYLES[identifier] for identifier in identifiers)


def register_plot_plugin(
    identifier: str,
    plugin: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PlotBuilderDescriptor:
    """Compatibility shim registering *plugin* as both builder and renderer."""

    warnings.warn(
        "register_plot_plugin is deprecated; use register_plot_builder/register_plot_renderer",
        DeprecationWarning,
        stacklevel=2,
    )
    descriptor = register_plot_builder(identifier, plugin, metadata=metadata)
    register_plot_renderer(identifier, plugin, metadata=metadata)
    register_plot_style(
        identifier,
        metadata={
            "style": identifier,
            "builder_id": identifier,
            "renderer_id": identifier,
            "fallbacks": (),
        },
    )
    return descriptor


def _list_descriptors(
    store: Dict[str, Any],
    trusted_only: bool,
    trusted_set: set[str],
) -> Tuple[Any, ...]:
    """Return descriptors from *store* with optional trust filtering."""

    if trusted_only:
        identifiers = sorted(identifier for identifier in trusted_set if identifier in store)
    else:
        identifiers = sorted(store.keys())
    return tuple(store[identifier] for identifier in identifiers)


def list_explanation_descriptors(*, trusted_only: bool = False) -> Tuple[ExplanationPluginDescriptor, ...]:
    """Return registered explanation plugin descriptors."""

    ensure_builtin_plugins()
    return _list_descriptors(
        _EXPLANATION_PLUGINS,
        trusted_only,
        _TRUSTED_EXPLANATIONS,
    )


def list_interval_descriptors(*, trusted_only: bool = False) -> Tuple[IntervalPluginDescriptor, ...]:
    """Return registered interval plugin descriptors."""

    ensure_builtin_plugins()
    return _list_descriptors(_INTERVAL_PLUGINS, trusted_only, _TRUSTED_INTERVALS)


def _refresh_descriptor_trust(identifier: str, *, trusted: bool) -> ExplanationPluginDescriptor:
    """Return descriptor with updated trust metadata stored in registries."""

    descriptor = find_explanation_descriptor(identifier)
    if descriptor is None:
        raise KeyError(f"Explanation plugin '{identifier}' is not registered")
    updated = ExplanationPluginDescriptor(
        identifier=descriptor.identifier,
        plugin=descriptor.plugin,
        metadata=descriptor.metadata,
        trusted=trusted,
    )
    _EXPLANATION_PLUGINS[identifier] = updated
    if trusted:
        _TRUSTED_EXPLANATIONS.add(identifier)
    else:
        _TRUSTED_EXPLANATIONS.discard(identifier)
    return updated


def mark_explanation_trusted(identifier: str) -> ExplanationPluginDescriptor:
    """Mark the explanation plugin *identifier* as trusted."""

    descriptor = _refresh_descriptor_trust(identifier, trusted=True)
    trust_plugin(descriptor.plugin)
    return descriptor


def mark_explanation_untrusted(identifier: str) -> ExplanationPluginDescriptor:
    """Remove the explanation plugin *identifier* from the trusted set."""

    descriptor = _refresh_descriptor_trust(identifier, trusted=False)
    untrust_plugin(descriptor.plugin)
    return descriptor


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
    "PlotBuilderDescriptor",
    "PlotRendererDescriptor",
    "PlotStyleDescriptor",
    "EXPLANATION_PROTOCOL_VERSION",
    "validate_explanation_metadata",
    "validate_interval_metadata",
    "validate_plot_builder_metadata",
    "validate_plot_renderer_metadata",
    "validate_plot_style_metadata",
    "clear_explanation_plugins",
    "clear_interval_plugins",
    "clear_plot_plugins",
    "ensure_builtin_plugins",
    "register_explanation_plugin",
    "register_interval_plugin",
    "register_plot_builder",
    "register_plot_renderer",
    "register_plot_style",
    "register_plot_plugin",
    "find_explanation_descriptor",
    "find_interval_descriptor",
    "find_plot_builder_descriptor",
    "find_plot_renderer_descriptor",
    "find_plot_style_descriptor",
    "find_explanation_plugin",
    "find_interval_plugin",
    "find_plot_builder",
    "find_plot_renderer",
    "find_plot_plugin",
    "find_plot_plugin_trusted",
    "list_explanation_descriptors",
    "list_interval_descriptors",
    "list_plot_builder_descriptors",
    "list_plot_renderer_descriptors",
    "list_plot_style_descriptors",
    "mark_explanation_trusted",
    "mark_explanation_untrusted",
    "register",
    "unregister",
    "clear",
    "list_plugins",
    "trust_plugin",
    "untrust_plugin",
    "find_for",
    "find_for_trusted",
]
