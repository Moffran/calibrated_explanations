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
import sys
from typing import Any, Dict, List, Tuple
from types import MappingProxyType

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
        self._explanation_preferred_identifier: Dict[str, str | None] = {
            "factual": None,
            "alternative": None,
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

    @property
    def interval_plugin_hints(self) -> Dict[str, Tuple[str, ...]]:
        """Access the interval plugin fallback hints."""
        return getattr(self, "_interval_plugin_hints", {})

    @interval_plugin_hints.setter
    def interval_plugin_hints(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Update the interval plugin fallback hints."""
        self._interval_plugin_hints = value

    @interval_plugin_hints.deleter
    def interval_plugin_hints(self) -> None:
        """Delete the interval plugin fallback hints."""
        if hasattr(self, "_interval_plugin_hints"):
            del self._interval_plugin_hints

    @property
    def explanation_plugin_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Access the explanation plugin fallback chains."""
        return getattr(self, "_explanation_plugin_fallbacks", {})

    @explanation_plugin_fallbacks.setter
    def explanation_plugin_fallbacks(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Update the explanation plugin fallback chains."""
        self._explanation_plugin_fallbacks = value

    @explanation_plugin_fallbacks.deleter
    def explanation_plugin_fallbacks(self) -> None:
        """Delete the explanation plugin fallback chains."""
        if hasattr(self, "_explanation_plugin_fallbacks"):
            del self._explanation_plugin_fallbacks

    @property
    def plot_plugin_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Access the plot plugin fallback chains."""
        return getattr(self, "_plot_plugin_fallbacks", {})

    @plot_plugin_fallbacks.setter
    def plot_plugin_fallbacks(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Update the plot plugin fallback chains."""
        self._plot_plugin_fallbacks = value

    @plot_plugin_fallbacks.deleter
    def plot_plugin_fallbacks(self) -> None:
        """Delete the plot plugin fallback chains."""
        if hasattr(self, "_plot_plugin_fallbacks"):
            del self._plot_plugin_fallbacks

    @property
    def interval_plugin_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Access the interval plugin fallback chains."""
        return getattr(self, "_interval_plugin_fallbacks", {})

    @interval_plugin_fallbacks.setter
    def interval_plugin_fallbacks(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Update the interval plugin fallback chains."""
        self._interval_plugin_fallbacks = value

    @interval_plugin_fallbacks.deleter
    def interval_plugin_fallbacks(self) -> None:
        """Delete the interval plugin fallback chains."""
        if hasattr(self, "_interval_plugin_fallbacks"):
            del self._interval_plugin_fallbacks

    @property
    def interval_plugin_identifiers(self) -> Dict[str, str | None]:
        """Access the resolved interval plugin identifiers."""
        return getattr(self, "_interval_plugin_identifiers", {})

    @interval_plugin_identifiers.setter
    def interval_plugin_identifiers(self, value: Dict[str, str | None]) -> None:
        """Update the resolved interval plugin identifiers."""
        self._interval_plugin_identifiers = value

    @interval_plugin_identifiers.deleter
    def interval_plugin_identifiers(self) -> None:
        """Delete the resolved interval plugin identifiers."""
        if hasattr(self, "_interval_plugin_identifiers"):
            del self._interval_plugin_identifiers

    @property
    def telemetry_interval_sources(self) -> Dict[str, str | None]:
        """Access the telemetry metadata associated with interval sources."""
        return getattr(self, "_telemetry_interval_sources", {})

    @telemetry_interval_sources.setter
    def telemetry_interval_sources(self, value: Dict[str, str | None]) -> None:
        """Update the telemetry metadata associated with interval sources."""
        self._telemetry_interval_sources = value

    @telemetry_interval_sources.deleter
    def telemetry_interval_sources(self) -> None:
        """Delete the telemetry metadata associated with interval sources."""
        if hasattr(self, "_telemetry_interval_sources"):
            del self._telemetry_interval_sources

    @property
    def interval_preferred_identifier(self) -> Dict[str, str | None]:
        """Access the preferred interval identifiers."""
        return getattr(self, "_interval_preferred_identifier", {})

    @interval_preferred_identifier.setter
    def interval_preferred_identifier(self, value: Dict[str, str | None]) -> None:
        """Update the preferred interval identifiers."""
        self._interval_preferred_identifier = value

    @interval_preferred_identifier.deleter
    def interval_preferred_identifier(self) -> None:
        """Delete the preferred interval identifiers."""
        if hasattr(self, "_interval_preferred_identifier"):
            del self._interval_preferred_identifier

    @property
    def explanation_preferred_identifier(self) -> Dict[str, str | None]:
        """Access the preferred explanation identifiers."""
        return getattr(self, "_explanation_preferred_identifier", {})

    @explanation_preferred_identifier.setter
    def explanation_preferred_identifier(self, value: Dict[str, str | None]) -> None:
        """Update the preferred explanation identifiers."""
        self._explanation_preferred_identifier = value

    @explanation_preferred_identifier.deleter
    def explanation_preferred_identifier(self) -> None:
        """Delete the preferred explanation identifiers."""
        if hasattr(self, "_explanation_preferred_identifier"):
            del self._explanation_preferred_identifier

    @property
    def interval_context_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Access the context metadata captured for interval plugins."""
        return getattr(self, "_interval_context_metadata", {})

    @interval_context_metadata.setter
    def interval_context_metadata(self, value: Dict[str, Dict[str, Any]]) -> None:
        """Update the context metadata captured for interval plugins."""
        self._interval_context_metadata = value

    @interval_context_metadata.deleter
    def interval_context_metadata(self) -> None:
        """Delete the context metadata captured for interval plugins."""
        if hasattr(self, "_interval_context_metadata"):
            del self._interval_context_metadata

    @property
    def bridge_monitors(self) -> Dict[str, PredictBridgeMonitor]:
        """Access the bridge monitor registry."""
        return self._bridge_monitors

    @property
    def explanation_plugin_instances(self) -> Dict[str, Any]:
        """Access the cached explanation plugin instances."""
        return self._explanation_plugin_instances

    @explanation_plugin_instances.setter
    def explanation_plugin_instances(self, value: Dict[str, Any]) -> None:
        """Update the cached explanation plugin instances."""
        self._explanation_plugin_instances = value

    @property
    def explanation_plugin_identifiers(self) -> Dict[str, str]:
        """Access the resolved explanation plugin identifiers."""
        return self._explanation_plugin_identifiers

    @explanation_plugin_identifiers.setter
    def explanation_plugin_identifiers(self, value: Dict[str, str]) -> None:
        """Update the resolved explanation plugin identifiers."""
        self._explanation_plugin_identifiers = value

    @property
    def plot_style_chain(self) -> Tuple[str, ...] | None:
        """Access the resolved plot style chain."""
        return self._plot_style_chain

    @plot_style_chain.setter
    def plot_style_chain(self, value: Tuple[str, ...] | None) -> None:
        """Update the resolved plot style chain."""
        self._plot_style_chain = value

    @property
    def explanation_contexts(self) -> Dict[str, Any]:
        """Access the explanation contexts."""
        return self._explanation_contexts

    @explanation_contexts.setter
    def explanation_contexts(self, value: Dict[str, Any]) -> None:
        """Update the explanation contexts."""
        self._explanation_contexts = value

    @explanation_contexts.deleter
    def explanation_contexts(self) -> None:
        """Clear the explanation contexts."""
        self._explanation_contexts = {}

    @property
    def last_explanation_mode(self) -> str | None:
        """Access the last explanation mode."""
        return self._last_explanation_mode

    @last_explanation_mode.setter
    def last_explanation_mode(self, value: str | None) -> None:
        """Update the last explanation mode."""
        self._last_explanation_mode = value

    @property
    def last_telemetry(self) -> Dict[str, Any]:
        """Access the last telemetry payload."""
        return self._last_telemetry

    @last_telemetry.setter
    def last_telemetry(self, value: Dict[str, Any]) -> None:
        """Update the last telemetry payload."""
        self._last_telemetry = value

    @property
    def explanation_plugin_overrides(self) -> Dict[str, Any]:
        """Access the explanation plugin overrides."""
        return self._explanation_plugin_overrides

    @explanation_plugin_overrides.setter
    def explanation_plugin_overrides(self, value: Dict[str, Any]) -> None:
        """Update the explanation plugin overrides."""
        self._explanation_plugin_overrides = value

    @property
    def interval_plugin_override(self) -> Any:
        """Access the interval plugin override."""
        return self._interval_plugin_override

    @interval_plugin_override.setter
    def interval_plugin_override(self, value: Any) -> None:
        """Update the interval plugin override."""
        self._interval_plugin_override = value

    @property
    def fast_interval_plugin_override(self) -> Any:
        """Access the fast interval plugin override."""
        return self._fast_interval_plugin_override

    @fast_interval_plugin_override.setter
    def fast_interval_plugin_override(self, value: Any) -> None:
        """Update the fast interval plugin override."""
        self._fast_interval_plugin_override = value

    @property
    def plot_style_override(self) -> Any:
        """Access the plot style override."""
        return self._plot_style_override

    @plot_style_override.setter
    def plot_style_override(self, value: Any) -> None:
        """Update the plot style override."""
        self._plot_style_override = value

    @property
    def default_explanation_identifiers(self) -> Dict[str, str]:
        """Access the default explanation identifiers."""
        return self._default_explanation_identifiers

    def __deepcopy__(self, memo):
        """Deepcopy the plugin manager, handling circular references and unpicklable objects."""
        import copy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # Preserve MappingProxyType instances explicitly: deepcopy may
            # route through copyreg reducers (registered for pickling) which
            # would convert mappingproxy -> dict; to keep the proxy type, we
            # recreate a MappingProxyType with the same contents.
            if isinstance(v, MappingProxyType):
                try:
                    setattr(result, k, MappingProxyType(dict(v)))
                    continue
                except (TypeError, AttributeError) as exc:
                    # Fall back to original reference when recreation fails;
                    # log the reason at debug level and continue. Narrowing
                    # the exception types avoids masking unrelated errors
                    # per ADR-002.
                    self._logger.debug(
                        "__deepcopy__ preserve MappingProxyType failed for %s: %s",
                        k,
                        exc,
                    )
                    setattr(result, k, v)
                    continue

            try:
                # Special-case dicts to preserve MappingProxyType values
                if isinstance(v, dict):
                    try:
                        new_dict: Dict[Any, Any] = {}
                        for ik, iv in v.items():
                            if isinstance(iv, MappingProxyType):
                                # Recreate the mapping proxy to preserve immutability
                                new_dict[ik] = MappingProxyType(dict(iv))
                            else:
                                new_dict[ik] = copy.deepcopy(iv, memo)
                        setattr(result, k, new_dict)
                        continue
                    except (TypeError, AttributeError, RecursionError) as exc:
                        # Fallback to shallow copy of the dict on specific
                        # conversion errors; log at debug level to keep
                        # failures visible while avoiding broad catches.
                        self._logger.debug(
                            "__deepcopy__ dict-preserve failed for %s: %s", k, exc
                        )
                        setattr(result, k, v.copy())
                        continue

                setattr(result, k, copy.deepcopy(v, memo))
            except BaseException:
                exc_type = sys.exc_info()[0]
                if exc_type is not TypeError:
                    raise
                # Fallback for unpicklable objects (e.g., other mappingproxy-like)
                # We shallow copy containers to avoid sharing the container itself,
                # while sharing the unpicklable items (which are likely immutable).
                if isinstance(v, dict):
                    setattr(result, k, v.copy())
                elif isinstance(v, list):
                    setattr(result, k, v[:])
                else:
                    setattr(result, k, v)
        return result

    @property
    def explanation_orchestrator(self) -> Any:
        """Access the explanation orchestrator instance."""
        return self._explanation_orchestrator

    @property
    def prediction_orchestrator(self) -> Any:
        """Access the prediction orchestrator instance."""
        return self._prediction_orchestrator

    @property
    def reject_orchestrator(self) -> Any:
        """Access the reject orchestrator instance."""
        return self._reject_orchestrator

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
        self.plot_style_override = kwargs.get("plot_style")

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
            self._explanation_plugin_fallbacks[mode] = self.build_explanation_chain(
                mode, default_id
            )

        # Build interval chains for default and fast modes
        self._interval_plugin_fallbacks["default"] = self.build_interval_chain(fast=False)
        self._interval_plugin_fallbacks["fast"] = self.build_interval_chain(fast=True)

        # Build plot style chain
        self._plot_plugin_fallbacks["default"] = self.build_plot_chain()

    def build_explanation_chain(self, mode: str, default_identifier: str) -> Tuple[str, ...]:
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
        preferred_identifier: str | None = None

        # 1. User override
        override = self._explanation_plugin_overrides.get(mode)
        if isinstance(override, str) and override:
            entries.append(override)

        # 2. Environment variables
        default_env_key = (
            "CE_EXPLANATION_PLUGIN_FAST" if mode == "fast" else "CE_EXPLANATION_PLUGIN"
        )
        default_env_value = os.environ.get(default_env_key)
        if default_env_value:
            entries.append(default_env_value.strip())
            preferred_identifier = preferred_identifier or default_env_value.strip()

        env_key = f"CE_EXPLANATION_PLUGIN_{mode.upper()}"
        if env_key != default_env_key:
            env_value = os.environ.get(env_key)
            if env_value:
                entries.append(env_value.strip())
                preferred_identifier = preferred_identifier or env_value.strip()
        entries.extend(split_csv(os.environ.get(f"{env_key}_FALLBACKS")))

        # 3. pyproject.toml settings
        py_settings = self._pyproject_explanations or {}
        py_value = py_settings.get(mode)
        if isinstance(py_value, str) and py_value:
            entries.append(py_value)
            preferred_identifier = preferred_identifier or py_value
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

        # 5. Filter out any FAST-mode identifiers when building chains for
        # non-fast modes. FAST is opt-in and must never be implicitly added
        # to the default fallback chain for `factual`/`alternative` modes.
        if mode != "fast":

            def _is_fast_id(idt: str) -> bool:
                # Treat identifiers that explicitly reference 'fast' as FAST-mode
                # identifiers. This covers patterns like 'core.explanation.fast',
                # 'core.explanation.fast.sequential', and mode-suffixed forms.
                parts = idt.replace(":", ".").split(".")
                return "fast" in parts

            filtered: List[str] = []
            for ident in expanded:
                if _is_fast_id(ident):
                    continue
                filtered.append(ident)
            expanded = filtered

        # 6. Add default and mode-specific fallbacks
        if default_identifier and default_identifier not in seen:
            expanded.append(default_identifier)
            seen.add(default_identifier)

        if mode == "fast":
            # For fast mode, also try external fast plugin if registered
            ext_fast = EXTERNAL_EXPLANATION_FAST_IDENTIFIER
            if ext_fast and ext_fast not in seen:
                expanded.append(ext_fast)
                seen.add(ext_fast)

        self._explanation_preferred_identifier[mode] = preferred_identifier
        return tuple(expanded)

    def build_interval_chain(self, *, fast: bool) -> Tuple[str, ...]:
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

    def build_plot_chain(self) -> Tuple[str, ...]:
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
        override = self.plot_style_override
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
            except BaseException:  # pragma: no cover - defensive; ADR-002
                exc = sys.exc_info()[1]
                if not isinstance(exc, Exception):
                    raise
                # Lazy import to avoid circular dependency
                from ..utils.exceptions import (
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
            # Use the explainer's predict bridge as the target
            self._bridge_monitors[identifier] = PredictBridgeMonitor(self.explainer.predict_bridge)
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
        self._prediction_orchestrator.ensure_interval_runtime_state()

        # Initialize interval learner
        initialize_interval_learner(self.explainer)


# Public aliases for testing purposes (to avoid private member access in tests)
@property
def pyproject_explanations(self) -> Dict[str, Any] | None:
    """Access pyproject.toml explanations configuration."""
    return self._pyproject_explanations


@pyproject_explanations.setter
def pyproject_explanations(self, value: Dict[str, Any] | None) -> None:
    """Update pyproject.toml explanations configuration."""
    self._pyproject_explanations = value


@property
def pyproject_intervals(self) -> Dict[str, Any] | None:
    """Access pyproject.toml intervals configuration."""
    return self._pyproject_intervals


@pyproject_intervals.setter
def pyproject_intervals(self, value: Dict[str, Any] | None) -> None:
    """Update pyproject.toml intervals configuration."""
    self._pyproject_intervals = value


@property
def pyproject_plots(self) -> Dict[str, Any] | None:
    """Access pyproject.toml plots configuration."""
    return self._pyproject_plots


@pyproject_plots.setter
def pyproject_plots(self, value: Dict[str, Any] | None) -> None:
    """Update pyproject.toml plots configuration."""
    self._pyproject_plots = value
