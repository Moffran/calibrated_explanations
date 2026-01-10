"""Orchestration layer for explanation plugins.

This module provides the ExplanationOrchestrator class which coordinates
explanation pipeline execution, including plugin resolution, context building,
invocation, and result telemetry collection.

Part of Phase 1: Delegate Explanation Orchestration (ADR-001, ADR-004).

Note: All plugin defaults, chaining, and fallback logic has been moved to
PluginManager. This orchestrator delegates all chain-building to PluginManager.
"""

# pylint: disable=protected-access, too-many-lines

from __future__ import annotations

import contextlib
import copy
import logging
import warnings
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from ...core.config_helpers import coerce_string_tuple
from ...logging import ensure_logging_context_filter, logging_context, telemetry_diagnostic_mode, update_logging_context
from ...plugins import (
    EXPLANATION_PROTOCOL_VERSION,
    ExplainerHandle,
    ExplanationContext,
    ExplanationRequest,
    ensure_builtin_plugins,
    find_explanation_descriptor,
    find_explanation_plugin,
    find_explanation_plugin_trusted,
    is_identifier_denied,
    validate_explanation_batch,
)
from ...utils import EntropyDiscretizer, RegressorDiscretizer
from ...utils.exceptions import ConfigurationError

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer

_EXPLANATION_MODES: Tuple[str, ...] = ("factual", "alternative", "fast")
_TELEMETRY_LOGGER = logging.getLogger("calibrated_explanations.telemetry.explanation")
ensure_logging_context_filter()


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
        """Delegate to PluginManager for chain initialization.

        PluginManager is now the single source of truth for all plugin
        defaults, chains, and fallbacks. This method delegates to it.

        Notes
        -----
        This method is called during explainer initialization to pre-compute the
        plugin resolution chains for all explanation modes.
        """
        self.explainer.plugin_manager.initialize_chains()

    def set_discretizer(
        self,
        discretizer: str | Any,
        x_cal: Any | None = None,
        y_cal: Any | None = None,
        features_to_ignore: List[int] | None = None,
        *,
        condition_source: str | None = None,
    ) -> None:
        """Assign the discretizer to be used.

        Parameters
        ----------
        discretizer : str or discretizer object
            The discretizer to be used.
        x_cal : array-like, optional
            The calibration data for the discretizer.
        y_cal : array-like, optional
            The calibration target data for the discretizer.
        features_to_ignore : list of int, optional
            Features to ignore during discretization.
        condition_source : str, optional
            Source for condition labels ('observed' or 'prediction').
        """
        import numpy as np

        from ...core.discretizer_config import (
            instantiate_discretizer,
            setup_discretized_data,
            validate_discretizer_choice,
        )
        from ...utils.exceptions import ValidationError

        if x_cal is None:
            x_cal = self.explainer.x_cal
        if y_cal is None:
            y_cal = self.explainer.y_cal

        selected_condition_source = condition_source or self.explainer.condition_source
        if selected_condition_source not in {"observed", "prediction"}:
            raise ValidationError(
                "condition_source must be either 'observed' or 'prediction'",
                details={
                    "param": "condition_source",
                    "value": selected_condition_source,
                    "allowed": ("observed", "prediction"),
                },
            )
        condition_labels = None
        if selected_condition_source == "prediction":
            predictions = self.explainer.predict(
                x_cal, calibrated=True, uq_interval=False, bins=self.explainer.bins
            )
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            condition_labels = np.asarray(predictions)

            # Filter out NaNs
            if (
                np.issubdtype(condition_labels.dtype, np.number)
                and np.isnan(condition_labels).any()
            ):
                mask = ~np.isnan(condition_labels)
                x_cal = x_cal[mask]
                condition_labels = condition_labels[mask]

        # Validate and potentially default the discretizer choice
        discretizer = validate_discretizer_choice(discretizer, self.explainer.mode)

        if features_to_ignore is None:
            features_to_ignore = []

        not_to_discretize = np.union1d(
            np.union1d(self.explainer.categorical_features, self.explainer.features_to_ignore),
            features_to_ignore,
        )

        # Store old discretizer to check if we can cache
        old_discretizer = self.explainer.discretizer

        # Instantiate the discretizer (may return cached instance if type matches)
        self.explainer.discretizer = instantiate_discretizer(
            discretizer,
            x_cal,
            not_to_discretize,
            self.explainer.feature_names,
            y_cal,
            self.explainer.seed,
            old_discretizer,
            condition_labels=condition_labels,
            condition_source=selected_condition_source,
        )

        # If discretizer is unchanged, skip recomputation
        if self.explainer.discretizer is old_discretizer and hasattr(
            self.explainer, "discretized_X_cal"
        ):
            return

        # Setup discretized data and build feature caches
        feature_data, self.explainer.discretized_X_cal = setup_discretized_data(
            self.explainer,
            self.explainer.discretizer,
            self.explainer.x_cal,
            self.explainer.num_features,
        )

        # Populate feature_values and feature_frequencies from the setup data
        self.explainer.feature_values = {}
        self.explainer.feature_frequencies = {}
        for feature, data in feature_data.items():
            self.explainer.feature_values[feature] = data["values"]
            self.explainer.feature_frequencies[feature] = data["frequencies"]

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
        reject_policy: Any | None = None,
        _ce_skip_reject: bool = False,
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
        # Reject orchestration is opt-in: only engage when reject_policy is
        # explicitly provided for this call. This preserves legacy behavior
        # and avoids changing mocked call args when reject_policy=None.
        from ...core.reject.policy import RejectPolicy

        if not _ce_skip_reject and reject_policy is not None:
            try:
                effective_policy = RejectPolicy(reject_policy)
            except Exception:
                effective_policy = RejectPolicy.NONE

            if effective_policy is not RejectPolicy.NONE:
                # Ensure reject orchestrator is available (implicit enable)
                try:
                    _ = self.explainer.reject_orchestrator
                except Exception:
                    try:
                        self.explainer.plugin_manager.initialize_orchestrators()
                    except Exception:
                        pass

                def _explain_fn(x_subset, **inner_kw):
                    return self.invoke(
                        mode,
                        x_subset,
                        threshold,
                        low_high_percentiles,
                        inner_kw.get("bins", bins),
                        features_to_ignore,
                        extras=extras,
                        _ce_skip_reject=True,
                    )

                return self.explainer.reject_orchestrator.apply_policy(
                    effective_policy, x, explain_fn=_explain_fn, bins=bins
                )

        plugin, _identifier = self.ensure_plugin(mode)
        explainer_identifier = getattr(self.explainer, "explainer_id", None) or str(
            id(self.explainer)
        )
        per_instance_ignore = None
        features_arg = features_to_ignore or []
        if (
            features_arg
            and isinstance(features_arg, (list, tuple))
            and isinstance(features_arg[0], (list, tuple, np.ndarray))
        ):
            # User supplied per-instance masks
            per_instance_ignore = tuple(tuple(int(f) for f in mask) for mask in features_arg)
            flat_ignore = np.unique(
                np.concatenate([np.asarray(mask, dtype=int) for mask in features_arg])
            )
            features_to_ignore_flat = tuple(int(f) for f in flat_ignore.tolist())
        else:
            features_to_ignore_flat = tuple(features_arg)

        with logging_context(
            mode=mode,
            plugin_identifier=_identifier or mode,
            explainer_id=explainer_identifier,
        ):
            request = ExplanationRequest(
                threshold=threshold,
                low_high_percentiles=(
                    tuple(low_high_percentiles) if low_high_percentiles is not None else None
                ),
                bins=tuple(bins) if bins is not None else None,
                features_to_ignore=features_to_ignore_flat,
                extras=dict(extras or {}),
                feature_filter_per_instance_ignore=per_instance_ignore,
            )
            monitor = self.explainer.plugin_manager.get_bridge_monitor(_identifier or mode)
            if monitor is not None:
                monitor.reset_usage()
            try:
                batch = plugin.explain_batch(x, request)
            except (
                Exception
            ) as exc:  # ADR002_ALLOW: wrap plugin failures in ConfigurationError.  # pragma: no cover
                raise ConfigurationError(
                    f"Explanation plugin execution failed for mode '{mode}': {exc}"
                ) from exc
            try:
                validate_explanation_batch(
                    batch,
                    expected_mode=mode,
                    expected_task=self.explainer.mode,
                )
            except (
                Exception
            ) as exc:  # ADR002_ALLOW: rewrap validation errors with context.  # pragma: no cover
                raise ConfigurationError(
                    f"Explanation plugin for mode '{mode}' returned an invalid batch: {exc}"
                ) from exc

        metadata = batch.collection_metadata
        metadata.setdefault("task", self.explainer.mode)
        interval_key = "fast" if mode == "fast" else "default"
        interval_source = self.explainer.plugin_manager.telemetry_interval_sources.get(interval_key)
        if interval_source:
            metadata["interval_source"] = interval_source
            metadata.setdefault("proba_source", interval_source)
        metadata.setdefault(
            "interval_dependencies",
            tuple(self.explainer.plugin_manager.interval_plugin_hints.get(mode, ())),
        )
        preprocessor_meta = self.explainer.preprocessor_metadata
        if preprocessor_meta:
            metadata.setdefault("preprocessor", preprocessor_meta)
        plot_chain = self.explainer.plugin_manager.plot_plugin_fallbacks.get(mode)
        if plot_chain:
            metadata.setdefault("plot_fallbacks", tuple(plot_chain))
            metadata.setdefault("plot_source", plot_chain[0])

        telemetry_payload = {
            "mode": mode,
            "task": self.explainer.mode,
            "interval_source": interval_source,
            "interval_dependencies": metadata.get("interval_dependencies"),
            "proba_source": metadata.get("proba_source"),
            "plot_source": metadata.get("plot_source"),
            "plot_fallbacks": tuple(plot_chain or ()),
        }
        if preprocessor_meta:
            telemetry_payload["preprocessor"] = preprocessor_meta

        self.explainer.plugin_manager.last_telemetry = dict(telemetry_payload)
        # Bridge monitor check: builtin plugins (starting with "core.") use internal
        # execution pipeline and don't need the bridge. Other plugins must use it.
        if (
            monitor is not None
            and not monitor.used
            and _identifier is not None
            and not _identifier.startswith("core.")
        ):
            raise ConfigurationError(
                "Explanation plugin for mode '"
                + mode
                + "' did not use the calibrated predict bridge"
            )

        container_cls = batch.container_cls
        if hasattr(container_cls, "from_batch"):
            result = container_cls.from_batch(batch)
            instance_payload = ExplanationOrchestrator.build_instance_telemetry_payload(result)
            if instance_payload:
                telemetry_payload.update(instance_payload)
                self.explainer.plugin_manager.last_telemetry.update(instance_payload)
            telemetry_snapshot_keys = (
                "interval_dependencies",
                "full_probabilities_shape",
                "full_probabilities_summary",
                "plot_source",
                "plot_fallbacks",
                "interval_source",
                "proba_source",
                "mode",
                "task",
            )
            telemetry_snapshot = {
                key: telemetry_payload.get(key)
                for key in telemetry_snapshot_keys
                if telemetry_payload.get(key) is not None
            }
            log_extra = {
                "mode": mode,
                "plugin_identifier": _identifier or mode,
                "explainer_id": explainer_identifier,
                "telemetry_snapshot": telemetry_snapshot,
                "telemetry_fields": tuple(sorted(telemetry_payload.keys())),
            }
            _TELEMETRY_LOGGER.info("explanation telemetry payload constructed", extra=log_extra)
            with contextlib.suppress(Exception):
                result.telemetry = dict(telemetry_payload)
            # parity instrumentation removed
            self.explainer.latest_explanation = result
            self.explainer.plugin_manager.last_explanation_mode = mode
            return result

        raise ConfigurationError("Explanation plugin returned a batch that cannot be materialised")

    def invoke_factual(  # pylint: disable=invalid-name
        self,
        x: Any,  # pylint: disable=invalid-name
        threshold: Any,
        low_high_percentiles: Tuple[float, float] | None,
        bins: Any,
        features_to_ignore: Any,
        discretizer: str | None = None,
        _use_plugin: bool = True,
        reject_policy: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute factual explanation with automatic discretizer setting.

        This is a convenience delegator that sets up the appropriate discretizer
        before invoking the explain pipeline.

        Parameters
        ----------
        x : array-like
            Test instances to explain.
        threshold : Any
            Threshold parameter for probabilistic explanations.
        low_high_percentiles : tuple or None
            Low and high percentiles for intervals.
        bins : array-like or None
            Mondrian categories.
        features_to_ignore : sequence or None
            Feature indices to exclude.
        discretizer : str or None
            Discretizer type to set (e.g., "binaryEntropy" for classification).
        _use_plugin : bool, default=True
            Whether to use the plugin system.
        **kwargs : Any
            Additional arguments passed to the explanation plugin.

        Returns
        -------
        CalibratedExplanations
            Factual explanations.
        """
        if discretizer is not None:
            self.explainer.set_discretizer(discretizer, features_to_ignore=features_to_ignore)

        # When _use_plugin=False, bypass plugin system and use legacy path directly
        if not _use_plugin:
            from ._legacy_explain import (
                explain as legacy_explain,  # pylint: disable=import-outside-toplevel
            )

            return legacy_explain(
                self.explainer,
                x,
                threshold=threshold,
                low_high_percentiles=low_high_percentiles,
                bins=bins,
                features_to_ignore=features_to_ignore,
            )

        return self.invoke(
            mode="factual",
            x=x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            bins=bins,
            features_to_ignore=features_to_ignore,
            extras=kwargs,
            **({"reject_policy": reject_policy} if reject_policy is not None else {}),
        )

    def invoke_alternative(  # pylint: disable=invalid-name
        self,
        x: Any,  # pylint: disable=invalid-name
        threshold: Any,
        low_high_percentiles: Tuple[float, float] | None,
        bins: Any,
        features_to_ignore: Any,
        discretizer: str | None = None,
        _use_plugin: bool = True,
        reject_policy: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute alternative explanation with automatic discretizer setting.

        This is a convenience delegator that sets up the appropriate discretizer
        before invoking the explain pipeline.

        Parameters
        ----------
        x : array-like
            Test instances to explain.
        threshold : Any
            Threshold parameter for probabilistic explanations.
        low_high_percentiles : tuple or None
            Low and high percentiles for intervals.
        bins : array-like or None
            Mondrian categories.
        features_to_ignore : sequence or None
            Feature indices to exclude.
        discretizer : str or None
            Discretizer type to set (e.g., "entropy" for classification).
        _use_plugin : bool, default=True
            Whether to use the plugin system.
        **kwargs : Any
            Additional arguments passed to the explanation plugin.

        Returns
        -------
        AlternativeExplanations
            Alternative explanations.
        """
        if discretizer is not None:
            self.explainer.set_discretizer(discretizer, features_to_ignore=features_to_ignore)

        # When _use_plugin=False, bypass plugin system and use legacy path directly
        if not _use_plugin:
            from ._legacy_explain import (
                explain as legacy_explain,  # pylint: disable=import-outside-toplevel
            )

            return legacy_explain(
                self.explainer,
                x,
                threshold=threshold,
                low_high_percentiles=low_high_percentiles,
                bins=bins,
                features_to_ignore=features_to_ignore,
            )

        return self.invoke(
            mode="alternative",
            x=x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            bins=bins,
            features_to_ignore=features_to_ignore,
            extras=kwargs,
            **({"reject_policy": reject_policy} if reject_policy is not None else {}),
        )

    def ensure_plugin(self, mode: str) -> Tuple[Any, str | None]:
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
        # Set logging context for plugin resolution
        with logging_context(
            explainer_id=getattr(self.explainer, "id", None),
            mode=mode,
        ):
            if mode in self.explainer.plugin_manager.explanation_plugin_instances:
                return (
                    self.explainer.plugin_manager.explanation_plugin_instances[mode],
                    self.explainer.plugin_manager.explanation_plugin_identifiers.get(mode),
                )

            plugin, identifier = self.resolve_plugin(mode)
            # Update context with resolved plugin identifier
            with logging_context(plugin_identifier=identifier):
                metadata: Mapping[str, Any] | None = None
                if identifier:
                    descriptor = find_explanation_descriptor(identifier)
                    if descriptor:
                        metadata = descriptor.metadata
                        interval_dependency = metadata.get("interval_dependency")
                        hints = coerce_string_tuple(interval_dependency)
                        if hints:
                            self.explainer.plugin_manager.interval_plugin_hints[mode] = hints
                    else:
                        metadata = getattr(plugin, "plugin_meta", None)
                else:
                    metadata = getattr(plugin, "plugin_meta", None)

                error = self.check_metadata(
                    metadata,
                    identifier=identifier,
                    mode=mode,
                )
                if error:
                    raise ConfigurationError(error)

                if metadata is not None and not identifier:
                    hints = coerce_string_tuple(metadata.get("interval_dependency"))
                    if hints:
                        self.explainer.plugin_manager.interval_plugin_hints[mode] = hints

                context = self.build_context(mode, plugin, identifier)
                try:
                    plugin.initialize(context)
                except (
                    Exception
                ) as exc:  # ADR002_ALLOW: wrap plugin initialization failure.  # pragma: no cover
                    raise ConfigurationError(
                        f"Explanation plugin initialisation failed for mode '{mode}': {exc}"
                    ) from exc

        self.explainer.plugin_manager.explanation_plugin_instances[mode] = plugin
        if identifier:
            self.explainer.plugin_manager.explanation_plugin_identifiers[mode] = identifier
        self.explainer.plugin_manager.explanation_contexts[mode] = context
        return plugin, identifier

    def resolve_plugin(self, mode: str) -> Tuple[Any, str | None]:
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
        # Ensure plugin fallback chains are initialized for this explainer so
        # explicit overrides, env vars, and pyproject settings are respected
        # during resolution. Only initialize when the chain for *mode* is
        # missing to avoid overwriting test-injected or precomputed chains.
        pm = getattr(self.explainer, "plugin_manager", None)
        if pm is not None:
            try:
                existing = pm.explanation_plugin_fallbacks.get(mode)
            except CalibratedError:
                existing = None
            if not existing:
                try:
                    pm.initialize_chains()
                except CalibratedError:
                    # Best-effort: resolution should continue even if chain init fails.
                    pass

        raw_override = self.explainer.plugin_manager.explanation_plugin_overrides.get(mode)
        override = self.explainer.plugin_manager.coerce_plugin_override(raw_override)
        if override is not None and not isinstance(override, str):
            plugin = override
            identifier = getattr(plugin, "plugin_meta", {}).get("name")
            return plugin, identifier

        explicit_override_identifier = raw_override if isinstance(raw_override, str) else None
        preferred_identifier = (
            explicit_override_identifier
            or self.explainer.plugin_manager.explanation_preferred_identifier.get(mode)
        )
        allow_untrusted = explicit_override_identifier is not None
        chain = self.explainer.plugin_manager.explanation_plugin_fallbacks.get(mode, ())
        if not chain and mode == "fast":
            msg = (
                "Fast explanation plugin 'core.explanation.fast' is not registered. "
                "Install the external plugins extra with "
                '``pip install "calibrated-explanations[external-plugins]"`` '
                "and call ``external_plugins.fast_explanations.register()`` or rerun "
                "``explain_fast(..., _use_plugin=False)`` to fall back to the legacy path."
            )
            raise ConfigurationError(msg)

        errors: List[str] = []
        for identifier in chain:
            is_preferred = preferred_identifier is not None and identifier == preferred_identifier
            is_explicit_override = (
                explicit_override_identifier is not None
                and identifier == explicit_override_identifier
            )
            if is_identifier_denied(identifier):
                message = f"{identifier}: denied via CE_DENY_PLUGIN"
                if is_preferred:
                    prefix = (
                        "Explanation plugin override failed: "
                        if is_explicit_override
                        else "Explanation plugin configuration failed: "
                    )
                    raise ConfigurationError(prefix + message) from None
                errors.append(message)
                continue

            descriptor = find_explanation_descriptor(identifier)
            metadata: Mapping[str, Any] | None = None
            plugin = None
            if descriptor is not None:
                metadata = descriptor.metadata
                if descriptor.trusted:
                    plugin = descriptor.plugin
                elif is_preferred and not allow_untrusted:
                    raise ConfigurationError(
                        "Explanation plugin configuration failed: "
                        + identifier
                        + " is untrusted; explicitly trust the plugin or pass an explicit override"
                    ) from None
                elif is_explicit_override:
                    warnings.warn(
                        f"Using untrusted explanation plugin '{identifier}' via explicit override. "
                        "Ensure you trust the source of this plugin.",
                        UserWarning,
                        stacklevel=2,
                    )
                    plugin = descriptor.plugin
            if plugin is None:
                if is_explicit_override:
                    plugin = find_explanation_plugin(identifier)
                else:
                    plugin = find_explanation_plugin_trusted(identifier)
            if plugin is None:
                message = f"{identifier}: not registered"
                if is_preferred:
                    prefix = (
                        "Explanation plugin override failed: "
                        if is_explicit_override
                        else "Explanation plugin configuration failed: "
                    )
                    raise ConfigurationError(prefix + message) from None
                errors.append(message)
                continue

            meta_source = metadata or getattr(plugin, "plugin_meta", None)
            error = self.check_metadata(
                meta_source,
                identifier=identifier,
                mode=mode,
            )
            if error:
                if is_preferred:
                    prefix = (
                        "Explanation plugin override failed: "
                        if is_explicit_override
                        else "Explanation plugin configuration failed: "
                    )
                    raise ConfigurationError(prefix + error) from None
                errors.append(error)
                continue

            plugin = self.instantiate_plugin(plugin)
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
            except (
                Exception
            ) as exc:  # ADR002_ALLOW: defensive catch for third-party plugins.  # pragma: no cover
                # pragma: no cover - defensive
                errors.append(f"{identifier}: error during supports_mode ({exc})")
                continue
            return plugin, identifier

        if mode == "fast" and "core.explanation.fast" in chain:
            msg = (
                "Fast explanation plugin 'core.explanation.fast' is not registered. "
                "Install the external plugins extra with "
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

    def check_metadata(
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
    def instantiate_plugin(prototype: Any) -> Any:
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
        except Exception:  # ADR002_ALLOW: fallback when plugins require args.  # pragma: no cover
            try:
                return copy.deepcopy(prototype)
            except (
                Exception
            ):  # ADR002_ALLOW: final fallback to reuse prototype instance.  # pragma: no cover
                # pragma: no cover - defensive
                return prototype

    def build_context(
        self,
        mode: str,
        plugin: Any,
        identifier: str | None,
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
        plot_chain = self._derive_plot_chain(mode, identifier)
        self.explainer.plugin_manager.plot_plugin_fallbacks[mode] = plot_chain
        interval_settings = {
            "dependencies": self.explainer.plugin_manager.interval_plugin_hints.get(mode, ()),
        }
        plot_settings = {"fallbacks": plot_chain}
        metadata = {
            "task": self.explainer.mode,
            "mode": mode,
            "interval_dependencies": tuple(interval_settings["dependencies"]),
            "plot_fallbacks": tuple(plot_chain),
            "plot_source": plot_chain[0] if plot_chain else None,
        }
        explainer_handle = ExplainerHandle(self.explainer, metadata)
        helper_handles = MappingProxyType({"explainer": explainer_handle})

        # Use the bridge monitor for this plugin to track usage.
        # We use the identifier if available, otherwise fall back to mode.
        monitor = self.explainer.plugin_manager.get_bridge_monitor(identifier or mode)

        context = ExplanationContext(
            task=self.explainer.mode,
            mode=mode,
            feature_names=tuple(self.explainer.feature_names),
            categorical_features=tuple(self.explainer.categorical_features),
            categorical_labels={
                k: dict(v) for k, v in (self.explainer.categorical_labels or {}).items()
            }
            if self.explainer.categorical_labels
            else {},
            discretizer=self.explainer.discretizer,
            helper_handles=helper_handles,
            predict_bridge=monitor,
            interval_settings=interval_settings,
            plot_settings=plot_settings,
        )
        return context

    def _derive_plot_chain(  # pylint: disable=invalid-name
        self,
        mode: str,
        identifier: str | None,  # pylint: disable=unused-argument
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
        base_chain = self.explainer.plugin_manager.plot_style_chain or ("legacy",)
        seen: set[str] = set()
        ordered: List[str] = []
        for item in tuple(preferred) + base_chain:
            if item and item not in seen:
                ordered.append(item)
                seen.add(item)
        return tuple(ordered)

    # Public alias for testing
    derive_plot_chain = _derive_plot_chain

    @staticmethod
    def build_instance_telemetry_payload(explanations: Any) -> Dict[str, Any]:
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
        except (
            Exception
        ):  # ADR002_ALLOW: telemetry is optional and best-effort.  # pragma: no cover
            # pragma: no cover - defensive: empty or non-indexable containers
            return {}

        # If explanation exposes `to_telemetry` and it returns a dict, preserve
        # that dict exactly (backwards-compatible behavior).
        builder = getattr(first_explanation, "to_telemetry", None)
        payload: Dict[str, Any] = {}
        if callable(builder):
            try:
                candidate = builder()
                if isinstance(candidate, dict):
                    payload.update(candidate)
            except Exception:
                # best-effort only; fall through to extract compact telemetry
                pass

        # Fallback: build compact telemetry from available prediction payloads
        try:
            full_probs = None
            if hasattr(first_explanation, "prediction"):
                full_probs = first_explanation.prediction.get("__full_probabilities__")
            diag_mode = telemetry_diagnostic_mode()
            if full_probs is not None:
                arr = np.asarray(full_probs)
                # Only expose telemetry for bona-fide arrays/collections
                if getattr(arr, "size", 0) > 0:
                    payload.setdefault("full_probabilities_shape", tuple(arr.shape))
                    with contextlib.suppress(Exception):
                        payload.setdefault(
                            "full_probabilities_summary",
                            {
                                "mean": float(np.mean(arr)),
                                "min": float(np.min(arr)),
                                "max": float(np.max(arr)),
                            },
                        )
                    if diag_mode:
                        payload.setdefault("full_probabilities", full_probs)
            # Also propagate interval dependency hints if present on the instance
            try:
                deps = getattr(first_explanation, "metadata", None) or {}
                if not deps and callable(builder):
                    # if builder didn't return dict, attempt to call a safe helper
                    with contextlib.suppress(Exception):
                        t = builder()
                        if isinstance(t, dict):
                            deps = t
                if isinstance(deps, dict):
                    interval_deps = (
                        deps.get("interval_dependencies")
                        or deps.get("metadata", {}).get("interval_dependencies")
                    )
                    if interval_deps:
                        payload.setdefault("interval_dependencies", interval_deps)
            except Exception:
                pass
        except Exception:
            # best-effort enrichment only
            pass

        return payload
