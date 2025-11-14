"""Plugin Manager for orchestrating plugin state and resolution (Phase 3).

This module centralizes plugin discovery, resolution, registry management, and
instance caching that was previously scattered across CalibratedExplainer.
Aligns with ADR-001 (boundary realignment) and ADR-006 (plugin trust model).

Responsibilities:
  - Plugin override configuration
  - Plugin descriptor lookup
  - Plugin resolution with fallback chains
  - Plugin instance caching
  - Bridge monitor management
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .predict_monitor import PredictBridgeMonitor


class PluginManager:
    """Centralized plugin state and resolution for CalibratedExplainer.

    This class encapsulates all plugin-related state that was previously
    embedded in CalibratedExplainer, including override configuration,
    plugin instances, fallback chains, and bridge monitors.

    Responsibilities:
      - Manage plugin override configuration (explanation, interval, plot)
      - Cache plugin instances and identifiers
      - Track fallback chains for explanation, interval, and plot plugins
      - Manage bridge monitors for instrumentation
      - Provide plugin override coercion for callable overrides
    """

    def __init__(self, explainer: Any) -> None:
        """Initialize plugin manager with back-reference to explainer.

        Parameters
        ----------
        explainer : CalibratedExplainer
            Parent explainer instance for accessing configuration and metadata.
        """
        self.explainer = explainer

        # Plugin override configuration
        self._explanation_plugin_overrides: Dict[str, Any] = {}
        self._interval_plugin_override: Any = None
        self._fast_interval_plugin_override: Any = None
        self._plot_style_override: Any = None

        # Plugin instance caching
        self._bridge_monitors: Dict[str, PredictBridgeMonitor] = {}
        self._explanation_plugin_instances: Dict[str, Any] = {}
        self._explanation_plugin_identifiers: Dict[str, str] = {}

        # Fallback chains for plugin resolution
        self._explanation_plugin_fallbacks: Dict[str, Tuple[str, ...]] = {}
        self._plot_plugin_fallbacks: Dict[str, Tuple[str, ...]] = {}
        self._interval_plugin_hints: Dict[str, Tuple[str, ...]] = {}
        self._interval_plugin_fallbacks: Dict[str, Tuple[str, ...]] = {}

        # Interval plugin state
        self._interval_plugin_identifiers: Dict[str, str | None] = {
            "default": None,
            "fast": None,
        }
        self._telemetry_interval_sources: Dict[str, str | None] = {
            "default": None,
            "fast": None,
        }
        self._interval_preferred_identifier: Dict[str, str | None] = {
            "default": None,
            "fast": None,
        }
        self._interval_context_metadata: Dict[str, Dict[str, Any]] = {
            "default": {},
            "fast": {},
        }

        # Plot style chain
        self._plot_style_chain: Tuple[str, ...] | None = None

    def initialize_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Initialize plugin overrides from keyword arguments.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Keyword arguments containing plugin overrides:
            - factual_plugin, alternative_plugin, fast_plugin for explanation plugins
            - interval_plugin, fast_interval_plugin for interval plugins
            - plot_style for plot style override
        """
        explanation_modes = ("factual", "alternative", "fast")

        self._explanation_plugin_overrides = {
            mode: kwargs.get(f"{mode}_plugin") for mode in explanation_modes
        }
        self._interval_plugin_override = kwargs.get("interval_plugin")
        self._fast_interval_plugin_override = kwargs.get("fast_interval_plugin")
        self._plot_style_override = kwargs.get("plot_style")

    def coerce_plugin_override(self, override: Any) -> Any:
        """Normalise a plugin override into an instance when possible.

        Parameters
        ----------
        override : Any
            Plugin override value (string identifier, instance, or callable).

        Returns
        -------
        Any
            Normalized plugin override: string identifier, callable, or instance.

        Raises
        ------
        ConfigurationError
            If callable override raises an exception.
        """
        if override is None:
            return None
        if isinstance(override, str):
            return override
        if callable(override) and not hasattr(override, "plugin_meta"):
            try:
                candidate = override()
            except Exception as exc:  # pragma: no cover - defensive
                # Lazy import to avoid circular dependency
                from ..core.exceptions import ConfigurationError

                raise ConfigurationError(
                    "Callable explanation plugin override raised an exception"
                ) from exc
            return candidate
        return override

    def get_bridge_monitor(self, identifier: str) -> PredictBridgeMonitor:
        """Get or create a bridge monitor for the given plugin identifier.

        Parameters
        ----------
        identifier : str
            Plugin identifier.

        Returns
        -------
        PredictBridgeMonitor
            Bridge monitor instance.
        """
        if identifier not in self._bridge_monitors:
            self._bridge_monitors[identifier] = PredictBridgeMonitor(identifier)
        return self._bridge_monitors[identifier]

    def clear_bridge_monitors(self) -> None:
        """Clear all cached bridge monitors."""
        self._bridge_monitors.clear()

    def get_explanation_plugin_instance(self, identifier: str) -> Any | None:
        """Get cached explanation plugin instance.

        Parameters
        ----------
        identifier : str
            Plugin identifier.

        Returns
        -------
        Any | None
            Cached plugin instance, or None if not cached.
        """
        return self._explanation_plugin_instances.get(identifier)

    def set_explanation_plugin_instance(self, identifier: str, instance: Any) -> None:
        """Cache explanation plugin instance.

        Parameters
        ----------
        identifier : str
            Plugin identifier.
        instance : Any
            Plugin instance to cache.
        """
        self._explanation_plugin_instances[identifier] = instance

    def clear_explanation_plugin_instances(self) -> None:
        """Clear all cached explanation plugin instances."""
        self._explanation_plugin_instances.clear()

    def get_explanation_plugin_identifier(self, mode: str) -> str | None:
        """Get cached explanation plugin identifier for mode.

        Parameters
        ----------
        mode : str
            Explanation mode (factual, alternative, fast).

        Returns
        -------
        str | None
            Cached plugin identifier, or None if not cached.
        """
        return self._explanation_plugin_identifiers.get(mode)

    def set_explanation_plugin_identifier(self, mode: str, identifier: str) -> None:
        """Cache explanation plugin identifier for mode.

        Parameters
        ----------
        mode : str
            Explanation mode (factual, alternative, fast).
        identifier : str
            Plugin identifier.
        """
        self._explanation_plugin_identifiers[mode] = identifier

    def clear_explanation_plugin_identifiers(self) -> None:
        """Clear all cached explanation plugin identifiers."""
        self._explanation_plugin_identifiers.clear()
