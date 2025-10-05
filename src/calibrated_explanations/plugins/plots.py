"""Plot builder and renderer protocols (ADR-014)."""

from __future__ import annotations

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

try:  # Python < 3.10 compatibility
    from typing import TypeAlias
except ImportError:  # pragma: no cover - fallback when TypeAlias is unavailable
    TypeAlias = object  # type: ignore[assignment]

from ..viz.plotspec import PlotSpec

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
]
