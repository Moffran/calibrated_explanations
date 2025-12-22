"""Plot plugin helper base classes and light lifecycle hooks.

This module provides minimal base classes for plot builders and renderers so
that plugins can opt-in to validation and standardised initialisation.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..utils.exceptions import PlotPluginError
from .serializers import validate_plotspec


class BasePlotBuilder:
    """Base class for plot builders.

    Subclasses should implement `build(self, context)` and may call
    `validate_plotspec` when returning PlotSpec-shaped payloads.
    """

    plugin_meta: Mapping[str, Any] = {}

    def initialize(self, context: Any) -> None:
        """Optional initialisation hook called with a PlotRenderContext."""
        self._context = context

    def build(self, context: Any) -> Mapping[str, Any]:
        """Construct an artifact representing the plot request.

        Implementations should return a mapping. If the mapping contains a
        PlotSpec or PlotSpec envelope, callers are encouraged to call
        `validate_plotspec` to surface malformed payloads early.
        """
        raise PlotPluginError("Plot builder must implement build()")


class BasePlotRenderer:
    """Base class for plot renderers.

    Subclasses should implement `render(self, artifact, *, context)` and
    may rely on the artifact shape produced by a builder.
    """

    plugin_meta: Mapping[str, Any] = {}

    def initialize(self, context: Any) -> None:
        """Optional initialisation hook called with a PlotRenderContext."""
        self._context = context

    def render(self, artifact: Mapping[str, Any], *, context: Any) -> Any:
        """Render the artifact and return a PlotRenderResult-like object.

        If the artifact contains a PlotSpec-like dict, validate it before
        rendering to provide early, auditable failures.
        """
        # Best-effort validation when an apparent plotspec is supplied
        try:
            if isinstance(artifact, Mapping) and (
                "plot_spec" in artifact or "kind" in artifact or "plotspec_version" in artifact
            ):
                validate_plotspec(dict(artifact))
        except Exception as exc:  # pragma: no cover - surface errors to caller
            raise PlotPluginError("PlotSpec validation failed") from exc
        raise PlotPluginError("Plot renderer must implement render()")
