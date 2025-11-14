"""Orchestration layer for explanation plugins.

This module provides the ExplanationOrchestrator class which coordinates
explanation pipeline execution, including plugin resolution, context building,
invocation, and result telemetry collection.

Part of Phase 1: Delegate Explanation Orchestration (ADR-001, ADR-004).
"""

# pylint: disable=protected-access, too-many-lines

from __future__ import annotations

import contextlib
import copy
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Tuple

from ..config_helpers import coerce_string_tuple, split_csv
from ..exceptions import ConfigurationError
from ...plugins import ExplanationContext, ExplanationRequest, validate_explanation_batch
from ...plugins.predict_monitor import PredictBridgeMonitor
from ...plugins.registry import (
    EXPLANATION_PROTOCOL_VERSION,
    ensure_builtin_plugins,
    find_explanation_descriptor,
    find_explanation_plugin,
    is_identifier_denied,
)
from ...utils.discretizers import EntropyDiscretizer, RegressorDiscretizer


if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer

# Default explanation plugin identifiers per mode (fallback chain terminus)
_DEFAULT_EXPLANATION_IDENTIFIERS = {
    "factual": "core.explanation.factual",
    "alternative": "core.explanation.alternative",
    "fast": "core.explanation.fast",
}


class ExplanationOrchestrator:
    """Orchestrate explanation pipeline execution and plugin coordination.

    This class handles the complete explanation workflow including:
    - Plugin resolution and instantiation
    - Context building for plugin execution
    - Plugin invocation and result validation
    - Telemetry collection and formatting

    Attributes
    ----------
    explainer : CalibratedExplainer
        Back-reference to the parent explainer instance.
    """

    def __init__(self, explainer: CalibratedExplainer) -> None:
        """Initialize the orchestrator with a back-reference to the explainer.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The parent explainer instance.

        Notes
        -----
        The orchestrator is a thin coordination layer that manages behavior
        using state stored on the parent explainer. It does not hold state itself.
        State is accessed through these explainer fields:
        - explainer._explanation_plugin_instances
        - explainer._explanation_plugin_identifiers
        - explainer._bridge_monitors
        - explainer._plot_plugin_fallbacks
        - explainer._interval_plugin_hints
        - explainer._explanation_contexts
        """
        self.explainer = explainer

    def initialize_chains(self) -> None:
        """Build and cache the explanation and plot plugin fallback chains.

        This method is called during explainer initialization to pre-compute the
        plugin resolution chains for all explanation modes. It populates:
        - explainer._explanation_plugin_fallbacks
        - explainer._plot_plugin_fallbacks

        Notes
        -----
        Must be called after explainer plugin configuration is set up but before
        plugins are resolved. Assumes builtin plugins have been registered.
        """
        # Use the default identifiers stored on the explainer instance
        # This allows tests to patch them by modifying the module-level dictionary
        # which is copied into each explainer instance
        for mode in ("factual", "alternative", "fast"):
            default_id = self.explainer._default_explanation_identifiers.get(mode, "")
            self.explainer._explanation_plugin_fallbacks[mode] = (
                self._build_explanation_chain(mode, default_id)
            )

        # Build plot style fallback chain
        self.explainer._plot_plugin_fallbacks["default"] = self._build_plot_chain()

    def _build_explanation_chain(
        self, mode: str, default_identifier: str
    ) -> Tuple[str, ...]:
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
        entries: List[str] = []

        # 1. User override
        override = self.explainer._explanation_plugin_overrides.get(mode)
        if isinstance(override, str) and override:
            entries.append(override)

        # 2. Environment variables
        env_key = f"CE_EXPLANATION_PLUGIN_{mode.upper()}"
        env_value = os.environ.get(env_key)
        if env_value:
            entries.append(env_value.strip())
        entries.extend(split_csv(os.environ.get(f"{env_key}_FALLBACKS")))

        # 3. pyproject.toml settings
        py_settings = self.explainer._pyproject_explanations or {}
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
            ext_fast = "external.explanation.fast"
            if ext_fast and ext_fast not in seen:
                expanded.append(ext_fast)
                seen.add(ext_fast)

        return tuple(expanded)

    def _build_plot_chain(self) -> Tuple[str, ...]:
        """Build the ordered plot style fallback chain.

        Returns
        -------
        tuple of str
            Ordered list of plot style identifiers to try.
        """
        entries: List[str] = []

        # 1. User override
        override = self.explainer._plot_style_override
        if isinstance(override, str) and override:
            entries.append(override)

        # 2. Environment variables
        env_value = os.environ.get("CE_PLOT_STYLE")
        if env_value:
            entries.append(env_value.strip())
        entries.extend(split_csv(os.environ.get("CE_PLOT_STYLE_FALLBACKS")))

        # 3. pyproject.toml settings
        py_settings = self.explainer._pyproject_plots or {}
        py_value = py_settings.get("style")
        if isinstance(py_value, str) and py_value:
            entries.append(py_value)
        entries.extend(coerce_string_tuple(py_settings.get("style_fallbacks")))

        entries.append("legacy")

        # 4. Deduplicate while maintaining order
        seen: set[str] = set()
        ordered: List[str] = []
        for identifier in entries:
            if identifier and identifier not in seen:
                ordered.append(identifier)
                seen.add(identifier)

        # 5. Ensure defaults are present
        if "plot_spec.default" not in seen:
            if "legacy" in ordered:
                legacy_index = ordered.index("legacy")
                ordered.insert(legacy_index, "plot_spec.default")
            else:
                ordered.append("plot_spec.default")

        if "legacy" not in ordered:
            ordered.append("legacy")

        return tuple(ordered)

    def infer_mode(self) -> str:
        """Infer the explanation mode based on the active discretizer.

        Returns
        -------
        str
            Either "factual" or "alternative" depending on the discretizer type.
        """
        discretizer = self.explainer.discretizer
        if isinstance(discretizer, (EntropyDiscretizer, RegressorDiscretizer)):
            return "alternative"
        return "factual"

    def invoke(  # pylint: disable=invalid-name
        self,
        mode: str,
        x: Any,  # pylint: disable=invalid-name
        threshold: Any,
        low_high_percentiles: Tuple[float, float] | None,
        bins: Any,
        features_to_ignore: Any,
        extras: Mapping[str, Any] | None = None,
    ) -> Any:
        """Execute the full explanation pipeline for the given mode.

        Parameters
        ----------
        mode : str
            The explanation mode ("factual", "alternative", "fast", etc.).
        x : array-like
            Test instances to explain.
        threshold : float, int, array-like, or None
            Threshold parameter for probabilistic explanations.
        low_high_percentiles : tuple of floats or None
            Low and high percentiles for interval calculation.
        bins : array-like or None
            Mondrian categories.
        features_to_ignore : sequence or None
            Feature indices to exclude from explanation.
        extras : dict or None
            Extra parameters to pass through to the plugin.

        Returns
        -------
        CalibratedExplanations
            The materialized explanation results.

        Raises
        ------
        ConfigurationError
            If plugin resolution, initialization, or invocation fails.
        """
        plugin, _identifier = self._ensure_plugin(mode)
        request = ExplanationRequest(
            threshold=threshold,
            low_high_percentiles=(
                tuple(low_high_percentiles) if low_high_percentiles is not None else None
            ),
            bins=bins,
            features_to_ignore=tuple(features_to_ignore or []),
            extras=dict(extras or {}),
        )
        monitor = self.explainer._bridge_monitors.get(mode)
        if monitor is not None:
            monitor.reset_usage()
        try:
            batch = plugin.explain_batch(x, request)
        except Exception as exc:
            raise ConfigurationError(
                f"Explanation plugin execution failed for mode '{mode}': {exc}"
            ) from exc
        try:
            validate_explanation_batch(
                batch,
                expected_mode=mode,
                expected_task=self.explainer.mode,
            )
        except Exception as exc:
            raise ConfigurationError(
                f"Explanation plugin for mode '{mode}' returned an invalid batch: {exc}"
            ) from exc

        metadata = batch.collection_metadata
        metadata.setdefault("task", self.explainer.mode)
        interval_key = "fast" if mode == "fast" else "default"
        interval_source = self.explainer._telemetry_interval_sources.get(interval_key)
        if interval_source:
            metadata["interval_source"] = interval_source
            metadata.setdefault("proba_source", interval_source)
        metadata.setdefault(
            "interval_dependencies",
            tuple(self.explainer._interval_plugin_hints.get(mode, ())),
        )
        preprocessor_meta = self.explainer.preprocessor_metadata
        if preprocessor_meta:
            metadata.setdefault("preprocessor", preprocessor_meta)
        plot_chain = self.explainer._plot_plugin_fallbacks.get(mode)
        if plot_chain:
            metadata.setdefault("plot_fallbacks", tuple(plot_chain))
            metadata.setdefault("plot_source", plot_chain[0])

        telemetry_payload = {
            "mode": mode,
            "task": self.explainer.mode,
            "interval_source": interval_source,
            "proba_source": metadata.get("proba_source"),
            "plot_source": metadata.get("plot_source"),
            "plot_fallbacks": tuple(plot_chain or ()),
        }
        if preprocessor_meta:
            telemetry_payload["preprocessor"] = preprocessor_meta

        self.explainer._last_telemetry = dict(telemetry_payload)
        if monitor is not None and not monitor.used:
            raise ConfigurationError(
                "Explanation plugin for mode '"
                + mode
                + "' did not use the calibrated predict bridge"
            )

        container_cls = batch.container_cls
        if hasattr(container_cls, "from_batch"):
            result = container_cls.from_batch(batch)
            instance_payload = self._build_instance_telemetry_payload(result)
            if instance_payload:
                telemetry_payload.update(instance_payload)
                self.explainer._last_telemetry.update(instance_payload)
            with contextlib.suppress(Exception):
                result.telemetry = dict(telemetry_payload)
            self.explainer.latest_explanation = result
            self.explainer._last_explanation_mode = mode
            return result

        raise ConfigurationError("Explanation plugin returned a batch that cannot be materialised")

    def _ensure_plugin(self, mode: str) -> Tuple[Any, str | None]:
        """Return the plugin instance for *mode*, initialising on demand.

        Parameters
        ----------
        mode : str
            The explanation mode.

        Returns
        -------
        tuple
            A tuple of (plugin_instance, plugin_identifier).
        """
        if mode in self.explainer._explanation_plugin_instances:
            return (
                self.explainer._explanation_plugin_instances[mode],
                self.explainer._explanation_plugin_identifiers.get(mode),
            )

        plugin, identifier = self._resolve_plugin(mode)
        metadata: Mapping[str, Any] | None = None
        if identifier:
            descriptor = find_explanation_descriptor(identifier)
            if descriptor:
                metadata = descriptor.metadata
                interval_dependency = metadata.get("interval_dependency")
                hints = coerce_string_tuple(interval_dependency)
                if hints:
                    self.explainer._interval_plugin_hints[mode] = hints
            else:
                metadata = getattr(plugin, "plugin_meta", None)
        else:
            metadata = getattr(plugin, "plugin_meta", None)

        error = self._check_metadata(
            metadata,
            identifier=identifier,
            mode=mode,
        )
        if error:
            raise ConfigurationError(error)

        if metadata is not None and not identifier:
            hints = coerce_string_tuple(metadata.get("interval_dependency"))
            if hints:
                self.explainer._interval_plugin_hints[mode] = hints

        context = self._build_context(mode, plugin, identifier)
        try:
            plugin.initialize(context)
        except Exception as exc:
            raise ConfigurationError(
                f"Explanation plugin initialisation failed for mode '{mode}': {exc}"
            ) from exc

        self.explainer._explanation_plugin_instances[mode] = plugin
        if identifier:
            self.explainer._explanation_plugin_identifiers[mode] = identifier
        self.explainer._explanation_contexts[mode] = context
        return plugin, identifier

    def _resolve_plugin(self, mode: str) -> Tuple[Any, str | None]:
        """Resolve or instantiate the plugin handling *mode*.

        Parameters
        ----------
        mode : str
            The explanation mode.

        Returns
        -------
        tuple
            A tuple of (plugin_instance, plugin_identifier).

        Raises
        ------
        ConfigurationError
            If no suitable plugin can be resolved.
        """
        ensure_builtin_plugins()

        raw_override = self.explainer._explanation_plugin_overrides.get(mode)
        override = self.explainer._coerce_plugin_override(raw_override)
        if override is not None and not isinstance(override, str):
            plugin = override
            identifier = getattr(plugin, "plugin_meta", {}).get("name")
            return plugin, identifier

        preferred_identifier = raw_override if isinstance(raw_override, str) else None
        chain = self.explainer._explanation_plugin_fallbacks.get(mode, ())
        if not chain and mode == "fast":
            msg = (
                "Fast explanation plugin 'core.explanation.fast' is not registered. "
                'Install the external plugins extra with '
                '``pip install "calibrated-explanations[external-plugins]"`` '
                "and call ``external_plugins.fast_explanations.register()`` or rerun "
                "``explain_fast(..., _use_plugin=False)`` to fall back to the legacy path."
            )
            raise ConfigurationError(msg)

        errors: List[str] = []
        for identifier in chain:
            is_preferred = (
                preferred_identifier is not None and identifier == preferred_identifier
            )
            if is_identifier_denied(identifier):
                message = f"{identifier}: denied via CE_DENY_PLUGIN"
                if is_preferred:
                    raise ConfigurationError(
                        "Explanation plugin override failed: " + message
                    ) from None
                errors.append(message)
                continue

            descriptor = find_explanation_descriptor(identifier)
            metadata: Mapping[str, Any] | None = None
            plugin = None
            if descriptor is not None:
                metadata = descriptor.metadata
                if descriptor.trusted:
                    plugin = descriptor.plugin
            if plugin is None:
                plugin = find_explanation_plugin(identifier)
            if plugin is None:
                message = f"{identifier}: not registered"
                if is_preferred:
                    raise ConfigurationError(
                        "Explanation plugin override failed: " + message
                    ) from None
                errors.append(message)
                continue

            meta_source = metadata or getattr(plugin, "plugin_meta", None)
            error = self._check_metadata(
                meta_source,
                identifier=identifier,
                mode=mode,
            )
            if error:
                if is_preferred:
                    raise ConfigurationError(error) from None
                errors.append(error)
                continue

            plugin = self._instantiate_plugin(plugin)
            try:
                supports = plugin.supports_mode
            except AttributeError as exc:
                errors.append(f"{identifier}: missing supports_mode ({exc})")
                continue
            try:
                if not supports(mode, task=self.explainer.mode):
                    errors.append(
                        f"{identifier}: mode '{mode}' "
                        f"unsupported for task {self.explainer.mode}"
                    )
                    continue
            except Exception as exc:  # pylint: disable=broad-except
                # pragma: no cover - defensive
                errors.append(f"{identifier}: error during supports_mode ({exc})")
                continue
            return plugin, identifier

        if mode == "fast" and "core.explanation.fast" in chain:
            msg = (
                "Fast explanation plugin 'core.explanation.fast' is not registered. "
                'Install the external plugins extra with '
                '``pip install "calibrated-explanations[external-plugins]"`` '
                "and call ``external_plugins.fast_explanations.register()`` or rerun "
                "``explain_fast(..., _use_plugin=False)`` to fall back to the legacy path."
            )
            raise ConfigurationError(msg)

        raise ConfigurationError(
            "Unable to resolve explanation plugin for mode '"
            + mode
            + "'. Tried: "
            + ", ".join(chain or ("<none>",))
            + ("; errors: " + "; ".join(errors) if errors else "")
        )

    def _check_metadata(
        self,
        metadata: Mapping[str, Any] | None,
        *,
        identifier: str | None,
        mode: str,
    ) -> str | None:
        """Return an error message if *metadata* is incompatible at runtime.

        Parameters
        ----------
        metadata : dict or None
            The plugin metadata to validate.
        identifier : str or None
            The plugin identifier for error messages.
        mode : str
            The explanation mode being validated.

        Returns
        -------
        str or None
            An error message if validation fails, None otherwise.
        """
        prefix = identifier or str((metadata or {}).get("name") or "<anonymous>")
        if metadata is None:
            return f"{prefix}: plugin metadata unavailable"

        schema_version = metadata.get("schema_version")
        if schema_version != EXPLANATION_PROTOCOL_VERSION:
            return (
                f"{prefix}: explanation schema_version {schema_version} unsupported; "
                f"expected {EXPLANATION_PROTOCOL_VERSION}"
            )

        tasks = coerce_string_tuple(metadata.get("tasks"))
        if not tasks:
            return f"{prefix}: plugin metadata missing tasks declaration"
        if "both" not in tasks and self.explainer.mode not in tasks:
            declared = ", ".join(tasks)
            return (
                f"{prefix}: does not support task '{self.explainer.mode}' "
                f"(declared: {declared})"
            )

        modes = coerce_string_tuple(metadata.get("modes"))
        if not modes:
            return f"{prefix}: plugin metadata missing modes declaration"
        if mode not in modes:
            declared = ", ".join(modes)
            return f"{prefix}: does not declare mode '{mode}' (modes: {declared})"

        capabilities = metadata.get("capabilities")
        cap_set: set[str] = set()
        if isinstance(capabilities, Iterable):
            for capability in capabilities:
                cap_set.add(str(capability))

        missing: List[str] = []
        if "explain" not in cap_set:
            missing.append("explain")
        mode_cap = f"explanation:{mode}"
        if mode_cap not in cap_set:
            alt_mode_cap = f"mode:{mode}"
            if alt_mode_cap not in cap_set:
                missing.append(mode_cap)
        task_cap = f"task:{self.explainer.mode}"
        if task_cap not in cap_set and "task:both" not in cap_set:
            missing.append(task_cap)

        if missing:
            return f"{prefix}: missing required capabilities {', '.join(sorted(missing))}"

        return None

    @staticmethod
    def _instantiate_plugin(prototype: Any) -> Any:
        """Best-effort instantiation that avoids sharing state across explainers.

        Parameters
        ----------
        prototype : Any
            The plugin class or instance to instantiate.

        Returns
        -------
        Any
            An instantiated plugin with its own state.
        """
        if prototype is None:
            return None
        if callable(prototype) and hasattr(prototype, "plugin_meta"):
            return prototype
        plugin_cls = type(prototype)
        try:
            return plugin_cls()
        except Exception:  # pylint: disable=broad-except
            try:
                return copy.deepcopy(prototype)
            except Exception:  # pylint: disable=broad-except
                # pragma: no cover - defensive
                return prototype

    def _build_context(
        self, mode: str, plugin: Any, identifier: str | None  # pylint: disable=unused-argument
    ) -> ExplanationContext:
        """Construct the immutable context passed to explanation plugins.

        Parameters
        ----------
        mode : str
            The explanation mode.
        plugin : Any
            The plugin instance.
        identifier : str or None
            The plugin identifier.

        Returns
        -------
        ExplanationContext
            The constructed context.
        """
        helper_handles = {"explainer": self.explainer}
        interval_settings = {
            "dependencies": self.explainer._interval_plugin_hints.get(mode, ()),
        }
        plot_chain = self._derive_plot_chain(mode, identifier)
        self.explainer._plot_plugin_fallbacks[mode] = plot_chain
        plot_settings = {"fallbacks": plot_chain}

        monitor = self.explainer._bridge_monitors.get(mode)
        if monitor is None:
            monitor = PredictBridgeMonitor(self.explainer._predict_bridge)
            self.explainer._bridge_monitors[mode] = monitor

        context = ExplanationContext(
            task=self.explainer.mode,
            mode=mode,
            feature_names=tuple(self.explainer.feature_names),
            categorical_features=tuple(self.explainer.categorical_features),
            categorical_labels=(
                {k: dict(v) for k, v in (self.explainer.categorical_labels or {}).items()}
                if self.explainer.categorical_labels
                else {}
            ),
            discretizer=self.explainer.discretizer,
            helper_handles=helper_handles,
            predict_bridge=monitor,
            interval_settings=interval_settings,
            plot_settings=plot_settings,
        )
        return context

    def _derive_plot_chain(  # pylint: disable=invalid-name
        self, mode: str, identifier: str | None  # pylint: disable=unused-argument
    ) -> Tuple[str, ...]:
        """Return plot fallback chain seeded by plugin metadata.

        Parameters
        ----------
        mode : str
            The explanation mode.
        identifier : str or None
            The plugin identifier.

        Returns
        -------
        tuple
            The plot fallback chain.
        """
        preferred: List[str] = []
        if identifier:
            descriptor = find_explanation_descriptor(identifier)
            if descriptor:
                plot_dependency = descriptor.metadata.get("plot_dependency")
                for hint in coerce_string_tuple(plot_dependency):
                    if hint:
                        preferred.append(hint)
        base_chain = self.explainer._plot_style_chain or ("legacy",)
        seen: set[str] = set()
        ordered: List[str] = []
        for item in tuple(preferred) + base_chain:
            if item and item not in seen:
                ordered.append(item)
                seen.add(item)
        return tuple(ordered)

    @staticmethod
    def _build_instance_telemetry_payload(explanations: Any) -> Dict[str, Any]:
        """Extract telemetry details from the first explanation instance.

        Parameters
        ----------
        explanations : Any
            The explanation container (should be indexable).

        Returns
        -------
        dict
            A telemetry payload extracted from the first explanation.
        """
        try:
            first_explanation = explanations[0]  # type: ignore[index]
        except Exception:  # pylint: disable=broad-except
            # pragma: no cover - defensive: empty or non-indexable containers
            return {}
        builder = getattr(first_explanation, "to_telemetry", None)
        if callable(builder):
            payload = builder()
            if isinstance(payload, dict):
                return payload
        return {}
