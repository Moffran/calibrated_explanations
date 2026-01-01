"""Plot builder and renderer protocols (ADR-014)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import (
    Any,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from ..viz.plotspec import PlotSpec


def _resolve_type_alias() -> Any:
    """Return ``typing.TypeAlias`` when available, otherwise fall back to ``object``."""
    try:  # pragma: no branch - helper used to exercise fallback in tests
        typing_mod = importlib.import_module("typing")
    except ImportError:  # pragma: no cover - stdlib module is always present on supported versions
        return object

    return getattr(typing_mod, "TypeAlias", object)


TypeAlias = _resolve_type_alias()


PlotArtifact: TypeAlias = Union[PlotSpec, Mapping[str, Any], Any]


@dataclass(frozen=True)
class PlotRenderContext:
    """Immutable context shared with plot plugins."""

    explanation: Any
    instance_metadata: Mapping[str, Any]
    style: str
    intent: Mapping[str, Any]
    show: bool
    path: str | None
    save_ext: str | Sequence[str] | None
    options: Mapping[str, Any]


@dataclass
class PlotRenderResult:
    """Return payload from :class:`PlotRenderer.render`."""

    artifact: PlotArtifact | None = None
    figure: Any | None = None
    saved_paths: Sequence[str] = field(default_factory=tuple)
    extras: MutableMapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PlotBuilder(Protocol):
    """Protocol for plot builders that emit :data:`PlotArtifact` payloads."""

    plugin_meta: Mapping[str, Any]

    def build(self, context: PlotRenderContext) -> PlotArtifact:
        """Return a serialisable artefact for *context*."""


@runtime_checkable
class PlotRenderer(Protocol):
    """Protocol for renderers that materialise plot artefacts."""

    plugin_meta: Mapping[str, Any]

    def render(self, artifact: PlotArtifact, *, context: PlotRenderContext) -> PlotRenderResult:
        """Render *artifact* using the runtime *context*."""


__all__ = [
    "PlotArtifact",
    "PlotBuilder",
    "PlotRenderContext",
    "PlotRenderResult",
    "PlotRenderer",
    "CombinedPlotPlugin",
]


class CombinedPlotPlugin:
    """Combine a `PlotBuilder` and `PlotRenderer` into a single plugin object.

    This wrapper exposes the surface expected by the registry:
    - `plugin_meta`: metadata sourced from the builder
    - `build(context)`: delegate to the configured builder
    - `render(artifact, *, context)`: delegate to the configured renderer

    Providing a named, documented class satisfies ADR-018 requirements for
    dynamically composed plugin classes (pydocstyle and docstring coverage).
    """

    def __init__(self, builder: PlotBuilder, renderer: PlotRenderer) -> None:
        self.builder = builder
        self.renderer = renderer
        self.plugin_meta = getattr(builder, "plugin_meta", {})

    def build(self, *args, **kwargs) -> PlotArtifact:
        """Delegate to the configured builder, preserving flexible signatures.

        Older plugins may implement `build(*args, **kwargs)` while newer
        implementations accept a single `context` argument. Forward all
        arguments to the builder to remain compatible.
        """
        return self.builder.build(*args, **kwargs)

    def render(self, *args, **kwargs) -> PlotRenderResult:
        """Delegate rendering to the configured renderer.

        Forward all args/kwargs so renderers that accept either
        `render(artifact, *, context=...)` or a flexible signature still
        work when wrapped.
        """
        return self.renderer.render(*args, **kwargs)
