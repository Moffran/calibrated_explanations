"""Plugin Manager for orchestrating plugin state and resolution.

This module centralizes plugin discovery, resolution, registry management, and
instance caching that was previously scattered across CalibratedExplainer.
Aligns with ADR-001 (boundary realignment) and ADR-006 (plugin trust model).

SINGLE SOURCE OF TRUTH for all plugin defaults, chains, and fallbacks.

Responsibilities:
  - Define default plugin identifiers for all modes
  - Manage plugin override configuration (explanation, interval, plot)
  - Build and cache fallback chains for all plugin types
  - Cache plugin instances and identifiers
  - Track bridge monitors for instrumentation
  - Provide plugin override coercion for callable overrides
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from .predict_monitor import PredictBridgeMonitor
from .registry import (
    find_explanation_descriptor,
    find_interval_descriptor,
)

# DEFAULT PLUGIN IDENTIFIERS (Single Source of Truth)
# These are the fallback plugins used when no override/env/config is specified
DEFAULT_EXPLANATION_IDENTIFIERS: Dict[str, str] = {
    "factual": "core.explanation.factual.sequential",
    "alternative": "core.explanation.alternative.sequential",
    "fast": "core.explanation.fast",
}

DEFAULT_INTERVAL_IDENTIFIERS: Dict[str, str] = {
    "default": "core.interval.legacy",
    "fast": "core.interval.fast",
}

# External plugin identifiers to try as fallbacks (registered dynamically)
EXTERNAL_EXPLANATION_FAST_IDENTIFIER: str = "external.explanation.fast"
EXTERNAL_INTERVAL_FAST_IDENTIFIER: str = "external.interval.fast"

# Default plot style fallback chain
DEFAULT_PLOT_STYLE: str = "legacy"


class PluginManager:
    """Centralized plugin state and resolution for CalibratedExplainer.

    This class encapsulates all plugin-related state that was previously
    embedded in CalibratedExplainer, including override configuration,
    plugin instances, fallback chains, and bridge monitors.

    This is the SINGLE SOURCE OF TRUTH for:
    - Default plugin identifiers
    - Plugin override configuration
    - Plugin fallback chains
    - Plugin instance caching
    - Bridge monitor management

    All plugin defaults, chaining logic, and fallback resolution is defined
    here and ONLY here. Orchestrators delegate to this manager for all
    plugin-related decisions.

    Responsibilities:
      - Define default plugin identifiers for all modes
      - Manage plugin override configuration (explanation, interval, plot)
      - Build and cache fallback chains for all plugin types
      - Cache plugin instances and identifiers
      - Track bridge monitors for instrumentation
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

        # Default identifiers (can be patched in tests)
        self._default_explanation_identifiers = dict(DEFAULT_EXPLANATION_IDENTIFIERS)
        self._default_interval_identifiers = dict(DEFAULT_INTERVAL_IDENTIFIERS)

        # Plugin override configuration
        self._explanation_plugin_overrides: Dict[str, Any] = {}
        self._interval_plugin_override: Any = None
        self._fast_interval_plugin_override: Any = None
        self._plot_style_override: Any = None

        # Plugin instance caching
        self._bridge_monitors: Dict[str, PredictBridgeMonitor] = {}
        self._explanation_plugin_instances: Dict[str, Any] = {}
        self._explanation_plugin_identifiers: Dict[str, str] = {}

        # Fallback chains for plugin resolution (populated by initialize_chains)
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

        # Plot style chain (populated by initialize_chains)
        self._plot_style_chain: Tuple[str, ...] | None = None

        # Plugin-related state (moved from CalibratedExplainer)
        self._pyproject_explanations: Dict[str, Any] | None = None
        self._pyproject_intervals: Dict[str, Any] | None = None
        self._pyproject_plots: Dict[str, Any] | None = None
        self._explanation_contexts: Dict[str, Any] = {}
        self._last_explanation_mode: str | None = None
        self._last_telemetry: Dict[str, Any] = {}

        # Orchestrator instances (initialized by _initialize_orchestrators)
        self._explanation_orchestrator: Any = None
        self._prediction_orchestrator: Any = None
        self._reject_orchestrator: Any = None

    def initialize_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Initialize plugin overrides from keyword arguments.

        Also reads and caches pyproject.toml plugin configurations.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Keyword arguments containing plugin overrides:
            - factual_plugin, alternative_plugin, fast_plugin for explanation plugins
            - interval_plugin, fast_interval_plugin for interval plugins
            - plot_style for plot style override
        """
        # Lazy import to avoid circular dependency
        from ..core.config_helpers import (
            read_pyproject_section,  # pylint: disable=import-outside-toplevel
        )

        explanation_modes = ("factual", "alternative", "fast")

        self._explanation_plugin_overrides = {
            mode: kwargs.get(f"{mode}_plugin") for mode in explanation_modes
        }
        self._interval_plugin_override = kwargs.get("interval_plugin")
        self._fast_interval_plugin_override = kwargs.get("fast_interval_plugin")
        self._plot_style_override = kwargs.get("plot_style")

        # Cache pyproject.toml plugin configurations
        self._pyproject_explanations = read_pyproject_section(
            ("tool", "calibrated_explanations", "explanations")
        )
        self._pyproject_intervals = read_pyproject_section(
            ("tool", "calibrated_explanations", "intervals")
        )
        self._pyproject_plots = read_pyproject_section(("tool", "calibrated_explanations", "plots"))

    def initialize_chains(self) -> None:
        """Build and cache all plugin fallback chains.

        This method is called during explainer initialization to pre-compute the
        plugin resolution chains for all modes and types. It populates:
        - _explanation_plugin_fallbacks (for all modes)
        - _interval_plugin_fallbacks (for default and fast modes)
        - _plot_plugin_fallbacks
        - _plot_style_chain

        Notes
        -----
        Must be called after plugin overrides are initialized but before
        plugins are resolved. Assumes builtin plugins have been registered.
        """
        # Build explanation chains for all modes
        for mode in ("factual", "alternative", "fast"):
            default_id = self._default_explanation_identifiers.get(mode, "")
            self._explanation_plugin_fallbacks[mode] = self._build_explanation_chain(
                mode, default_id
            )

        # Build interval chains for default and fast modes
        self._interval_plugin_fallbacks["default"] = self._build_interval_chain(fast=False)
        self._interval_plugin_fallbacks["fast"] = self._build_interval_chain(fast=True)

        # Build plot style chain
        self._plot_plugin_fallbacks["default"] = self._build_plot_chain()

    def _build_explanation_chain(self, mode: str, default_identifier: str) -> Tuple[str, ...]:
        """Build the ordered explanation plugin fallback chain for a mode.

        Parameters
        ----------
        mode : str
            The explanation mode ("factual", "alternative", "fast").
        default_identifier : str
            The default/fallback identifier for this mode.

        Returns
        -------
        tuple of str
            Ordered list of plugin identifiers to try for this mode.
        """
        # Lazy import to avoid circular dependency
        from ..core.config_helpers import (  # pylint: disable=import-outside-toplevel
            coerce_string_tuple,
            split_csv,
        )

        entries: List[str] = []

        # 1. User override
        override = self._explanation_plugin_overrides.get(mode)
        if isinstance(override, str) and override:
            entries.append(override)

        # 2. Environment variables
        env_key = f"CE_EXPLANATION_PLUGIN_{mode.upper()}"
        env_value = os.environ.get(env_key)
        if env_value:
            entries.append(env_value.strip())
        entries.extend(split_csv(os.environ.get(f"{env_key}_FALLBACKS")))

        # 3. pyproject.toml settings
        py_settings = self._pyproject_explanations or {}
        py_value = py_settings.get(mode)
        if isinstance(py_value, str) and py_value:
            entries.append(py_value)
        entries.extend(coerce_string_tuple(py_settings.get(f"{mode}_fallbacks")))

        # 4. Expand with descriptor fallbacks and deduplicate
        seen: set[str] = set()
        expanded: List[str] = []
        for identifier in entries:
            if not identifier or identifier in seen:
                continue
            expanded.append(identifier)
            seen.add(identifier)
            descriptor = find_explanation_descriptor(identifier)
            if descriptor:
                for fallback in coerce_string_tuple(descriptor.metadata.get("fallbacks")):
                    if fallback and fallback not in seen:
                        expanded.append(fallback)
                        seen.add(fallback)

        # 5. Add default and mode-specific fallbacks
        if default_identifier and default_identifier not in seen:
            expanded.append(default_identifier)
            seen.add(default_identifier)

        if mode == "fast":
            # For fast mode, also try external fast plugin if registered
            ext_fast = EXTERNAL_EXPLANATION_FAST_IDENTIFIER
            if ext_fast and ext_fast not in seen:
                expanded.append(ext_fast)
                seen.add(ext_fast)

        return tuple(expanded)

    def _build_interval_chain(self, *, fast: bool) -> Tuple[str, ...]:
        """Build the ordered interval plugin fallback chain.

        Parameters
        ----------
        fast : bool
            Whether to build for fast mode (True) or default mode (False).

        Returns
        -------
        tuple of str
            Ordered list of interval plugin identifiers to try.
        """
        # Lazy import to avoid circular dependency
        from ..core.config_helpers import (  # pylint: disable=import-outside-toplevel
            coerce_string_tuple,
            split_csv,
        )

        entries: List[str] = []
        override = self._fast_interval_plugin_override if fast else self._interval_plugin_override
        preferred_identifier: str | None = None

        if isinstance(override, str) and override:
            entries.append(override)
            preferred_identifier = override

        env_key = "CE_INTERVAL_PLUGIN_FAST" if fast else "CE_INTERVAL_PLUGIN"
        env_value = os.environ.get(env_key)
        if env_value:
            entries.append(env_value.strip())
            if preferred_identifier is None:
                preferred_identifier = env_value.strip()
        entries.extend(split_csv(os.environ.get(f"{env_key}_FALLBACKS")))

        py_settings = self._pyproject_intervals or {}
        py_key = "fast" if fast else "default"
        py_value = py_settings.get(py_key)
        if isinstance(py_value, str) and py_value:
            entries.append(py_value)
        entries.extend(coerce_string_tuple(py_settings.get(f"{py_key}_fallbacks")))

        default_identifier = (
            self._default_interval_identifiers.get("fast")
            if fast
            else self._default_interval_identifiers.get("default")
        )

        seen: set[str] = set()
        ordered: List[str] = []
        for identifier in entries:
            if identifier and identifier not in seen:
                ordered.append(identifier)
                seen.add(identifier)
                descriptor = find_interval_descriptor(identifier)
                if descriptor:
                    for fallback in coerce_string_tuple(descriptor.metadata.get("fallbacks")):
                        if fallback and fallback not in seen:
                            ordered.append(fallback)
                            seen.add(fallback)

        if default_identifier and default_identifier not in seen:
            if fast:
                # Prefer the core fast identifier when available; otherwise
                # fall back to the external fast interval identifier if registered.
                if find_interval_descriptor(default_identifier) is not None:
                    ordered.append(default_identifier)
                else:
                    ext_fast = EXTERNAL_INTERVAL_FAST_IDENTIFIER
                    if find_interval_descriptor(ext_fast) is not None:
                        ordered.append(ext_fast)
            else:
                ordered.append(default_identifier)

        key = "fast" if fast else "default"
        self._interval_preferred_identifier[key] = preferred_identifier
        return tuple(ordered)

    def _build_plot_chain(self) -> Tuple[str, ...]:
        """Build the ordered plot style fallback chain.

        Returns
        -------
        tuple of str
            Ordered list of plot style identifiers to try.
        """
        # Lazy import to avoid circular dependency
        from ..core.config_helpers import (  # pylint: disable=import-outside-toplevel
            coerce_string_tuple,
            split_csv,
        )

        entries: List[str] = []

        # 1. User override
        override = self._plot_style_override
        if isinstance(override, str) and override:
            entries.append(override)

        # 2. Environment variables
        env_value = os.environ.get("CE_PLOT_STYLE")
        if env_value:
            entries.append(env_value.strip())
        entries.extend(split_csv(os.environ.get("CE_PLOT_STYLE_FALLBACKS")))

        # 3. pyproject.toml settings
        py_settings = self._pyproject_plots or {}
        py_value = py_settings.get("style")
        if isinstance(py_value, str) and py_value:
            entries.append(py_value)
        entries.extend(coerce_string_tuple(py_settings.get("style_fallbacks")))

        # 4. Default plot style
        entries.append(DEFAULT_PLOT_STYLE)

        # Deduplicate while preserving order
        seen: set[str] = set()
        result: List[str] = []
        for identifier in entries:
            if identifier and identifier not in seen:
                result.append(identifier)
                seen.add(identifier)

        return tuple(result)

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
                from ..core.exceptions import (
                    ConfigurationError,  # pylint: disable=import-outside-toplevel
                )

                raise ConfigurationError(
                    "Callable explanation plugin override raised an exception"
                ) from exc
            return candidate
        return override

    # =========================================================================
    # Plugin instance and identifier caching
    # =========================================================================

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

    # =========================================================================
    # Orchestrator initialization (moved from CalibratedExplainer)
    # =========================================================================

    def initialize_orchestrators(self) -> None:
        """Initialize ExplanationOrchestrator and PredictionOrchestrator.

        This method:
        1. Creates orchestrator instances
        2. Ensures builtin plugins are registered
        3. Builds all plugin fallback chains
        4. Initializes interval runtime state

        Called from CalibratedExplainer.__init__ after PluginManager is initialized.
        """
        # Lazy import to avoid circular dependency
        from ..calibration.interval_learner import (
            initialize_interval_learner,  # pylint: disable=import-outside-toplevel
        )
        from ..core.explain.orchestrator import (
            ExplanationOrchestrator,  # pylint: disable=import-outside-toplevel
        )
        from ..core.prediction.orchestrator import (
            PredictionOrchestrator,  # pylint: disable=import-outside-toplevel
        )
        from ..core.reject.orchestrator import (
            RejectOrchestrator,  # pylint: disable=import-outside-toplevel
        )
        from .registry import ensure_builtin_plugins  # pylint: disable=import-outside-toplevel

        # Ensure builtin plugins (including optional fast plugins) are registered
        # before we compute fallback chains. Without this, the initial chain
        # construction may miss identifiers that are subsequently required during
        # runtime resolution, causing ConfigurationError during explain_fast.
        ensure_builtin_plugins()

        # Initialize orchestrators
        self._explanation_orchestrator = ExplanationOrchestrator(self.explainer)
        self._prediction_orchestrator = PredictionOrchestrator(self.explainer)
        self._reject_orchestrator = RejectOrchestrator(self.explainer)

        # Build all plugin fallback chains
        self.initialize_chains()

        # Populate plot_style_chain from explanation orchestrator's chains
        self._plot_style_chain = self._plot_plugin_fallbacks.get("default")

        # Initialize interval runtime state
        self._prediction_orchestrator._ensure_interval_runtime_state()

        # Initialize interval learner
        initialize_interval_learner(self.explainer)

