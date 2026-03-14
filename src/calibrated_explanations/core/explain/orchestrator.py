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
from ...logging import (
    ensure_logging_context_filter,
    logging_context,
    telemetry_diagnostic_mode,
)
from ...plugins import (
    EXPLANATION_PROTOCOL_VERSION,
    ExplainerHandle,
    ExplanationContext,
    ExplanationRequest,
    ensure_builtin_plugins,
    find_explanation_descriptor,
    is_identifier_denied,
    validate_explanation_batch,
)
from ...utils import EntropyDiscretizer, RegressorDiscretizer
from ...utils.exceptions import CalibratedError, ConfigurationError, DataShapeError
from ...utils.int_utils import as_int_array, coerce_to_int

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer

_EXPLANATION_MODES: Tuple[str, ...] = ("factual", "alternative", "fast")
_TELEMETRY_LOGGER = logging.getLogger("calibrated_explanations.telemetry.explanation")
ensure_logging_context_filter()


def _resolve_reject_policy_spec(candidate_policy: Any, explainer: Any) -> Any:
    """Resolve reject policy via lazy import and ensure orchestrators are initialized."""
    from ...core.reject.orchestrator import (
        resolve_policy_spec,  # pylint: disable=import-outside-toplevel
    )

    if getattr(explainer, "reject_orchestrator", None) is None:
        plugin_manager = getattr(explainer, "plugin_manager", None)
        if plugin_manager is not None:
            plugin_manager.initialize_orchestrators()
    return resolve_policy_spec(candidate_policy, explainer)


def _resolve_effective_reject_policy(
    candidate_policy: Any,
    explainer: Any,
    *,
    default_policy: Any,
) -> Any:
    """Resolve effective reject policy with shared fail-fast/fallback semantics."""
    from ...core.reject.orchestrator import (  # pylint: disable=import-outside-toplevel
        resolve_effective_reject_policy,
    )

    if getattr(explainer, "reject_orchestrator", None) is None:
        plugin_manager = getattr(explainer, "plugin_manager", None)
        if plugin_manager is not None:
            plugin_manager.initialize_orchestrators()
    return resolve_effective_reject_policy(
        candidate_policy,
        explainer,
        default_policy=default_policy,
        logger=logging.getLogger(__name__),
    )


def _warn_source_index_issue(message: str) -> None:
    """Emit visible warning and diagnostic log for source-index resolution issues."""
    logger = logging.getLogger(__name__)
    from ...explanations.reject import (
        RejectContractWarning,  # pylint: disable=import-outside-toplevel
    )

    logger.info(message)
    warnings.warn(message, RejectContractWarning, stacklevel=3)


def _resolve_source_indices_for_payload(
    *,
    policy: Any,
    metadata: Any,
    rejected_mask: Any,
    payload_count: int,
) -> list[int] | None:
    """Resolve source indices for reject-filtered payloads.

    Returns ``None`` when a safe mapping cannot be established.
    """
    source_indices_raw = None
    if isinstance(metadata, Mapping):
        source_indices_raw = metadata.get("source_indices")

    if source_indices_raw is not None:
        try:
            idx_arr = np.asarray(source_indices_raw)
            if idx_arr.ndim != 1 or not np.issubdtype(idx_arr.dtype, np.integer):
                raise ValueError("source_indices must be a 1D integer sequence")
            idxs = [int(v) for v in idx_arr.tolist()]
            if len(idxs) != payload_count:
                raise ValueError(
                    f"source_indices length {len(idxs)} does not match payload length {payload_count}"
                )
            if any(i < 0 for i in idxs):
                raise ValueError("source_indices must be non-negative")
            if len(set(idxs)) != len(idxs):
                raise ValueError("source_indices must be unique")
            if any(curr >= nxt for curr, nxt in zip(idxs, idxs[1:], strict=False)):
                raise ValueError("source_indices must preserve source order")
            if isinstance(metadata, Mapping) and metadata.get("original_count") is not None:
                original_count = int(metadata["original_count"])
                if any(i >= original_count for i in idxs):
                    raise ValueError(
                        f"source_indices must be < original_count={original_count}; got {idxs!r}"
                    )
            return idxs
        except (TypeError, ValueError) as exc:
            _warn_source_index_issue(
                f"Reject source_indices metadata is invalid ({exc!s}); attempting deterministic fallback."
            )

    if rejected_mask is None:
        return list(range(payload_count)) if payload_count == 0 else None

    try:
        rejected = np.asarray(rejected_mask, dtype=bool).flatten()
        policy_enum = policy
        from ...core.reject.policy import RejectPolicy  # pylint: disable=import-outside-toplevel

        if not isinstance(policy_enum, RejectPolicy):
            policy_enum = RejectPolicy(policy_enum)
        if policy_enum is RejectPolicy.ONLY_REJECTED:
            idxs = [i for i, v in enumerate(rejected) if v]
        elif policy_enum is RejectPolicy.ONLY_ACCEPTED:
            idxs = [i for i, v in enumerate(rejected) if not v]
        else:
            idxs = list(range(len(rejected)))
        if len(idxs) != payload_count:
            rejected_idxs = [i for i, v in enumerate(rejected) if v]
            accepted_idxs = [i for i, v in enumerate(rejected) if not v]
            if len(rejected_idxs) == payload_count and len(accepted_idxs) != payload_count:
                _warn_source_index_issue(
                    "Reject source mapping inferred from rejected subset cardinality."
                )
                return rejected_idxs
            if len(accepted_idxs) == payload_count and len(rejected_idxs) != payload_count:
                _warn_source_index_issue(
                    "Reject source mapping inferred from accepted subset cardinality."
                )
                return accepted_idxs
            _warn_source_index_issue(
                "Unable to map reject payload to source rows: derived fallback indices length "
                f"{len(idxs)} differs from payload length {payload_count}."
            )
            return None
        _warn_source_index_issue(
            "Reject result is missing source_indices metadata; using deterministic fallback mapping."
        )
        return idxs
    except (TypeError, ValueError) as exc:
        _warn_source_index_issue(
            f"Unable to derive reject source indices from policy/mask ({exc!s}); "
            "reject context attachment skipped."
        )
        return None


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

    def _coerce_ignore_items(self, items: Iterable[Any]) -> tuple[list[Any], list[str]]:
        """Coerce ignore entries to indices when possible.

        Supports feature names (strings) and numeric indices. Unknown feature
        names are collected for warning messages.
        """
        feature_names = getattr(self.explainer, "feature_names", None)
        name_to_index = None
        if feature_names:
            name_to_index = {str(name): idx for idx, name in enumerate(feature_names)}

        resolved: list[Any] = []
        unknown: list[str] = []
        for item in items:
            if isinstance(item, (str, bytes)):
                text = item.decode() if isinstance(item, bytes) else item
                if name_to_index and text in name_to_index:
                    resolved.append(name_to_index[text])
                    continue
                numeric = coerce_to_int(text)
                if numeric is not None:
                    resolved.append(numeric)
                    continue
                unknown.append(text)
                continue
            resolved.append(item)
        return resolved, unknown

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
                x_cal,
                calibrated=True,
                uq_interval=False,
                bins=self.explainer.bins,
                _ce_skip_reject=True,
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
                # If all prediction-derived condition labels are NaN, avoid
                # producing an empty calibration set — fall back to observed
                # labels so discretization can proceed.
                if mask.sum() == 0:
                    _TELEMETRY_LOGGER.info(
                        "All prediction-derived condition labels are NaN; falling back to observed y_cal"
                    )
                    warnings.warn(
                        "All prediction-derived condition labels are NaN; falling back to observed y_cal.",
                        UserWarning,
                        stacklevel=2,
                    )
                    condition_labels = None
                else:
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
        # Reject orchestration:
        # - default behavior remains "no reject" (ADR-029)
        # - if reject_policy is not provided (None), fall back to the explainer's
        #   default_reject_policy
        # - per-call policy overrides the explainer-level default
        from ...core.reject.policy import RejectPolicy

        if not _ce_skip_reject:
            confidence = extras.get("confidence", 0.95) if isinstance(extras, Mapping) else 0.95
            resolution = _resolve_effective_reject_policy(
                reject_policy,
                self.explainer,
                default_policy=getattr(self.explainer, "default_reject_policy", RejectPolicy.NONE),
            )
            effective_policy = resolution.policy

            if effective_policy is not RejectPolicy.NONE:
                # Ensure reject orchestrator is available (implicit enable)
                try:
                    _ = self.explainer.reject_orchestrator
                except Exception:  # adr002_allow
                    with contextlib.suppress(Exception):
                        self.explainer.plugin_manager.initialize_orchestrators()

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

                # Apply reject policy via the reject orchestrator. If a RejectResult
                # envelope with an `explanation` payload is returned, attach a
                # per-explanation `RejectContext` so downstream narrative/plot
                # code can render expertise-aware messages and badges.
                from ...explanations.reject import (
                    RejectAlternativeExplanations,
                    RejectCalibratedExplanations,
                    RejectContext,
                    RejectResult,
                )

                res = self.explainer.reject_orchestrator.apply_policy(
                    effective_policy,
                    x,
                    explain_fn=_explain_fn,
                    bins=bins,
                    confidence=confidence,
                    threshold=threshold,
                )

                # Attach RejectContext instances when possible. Be defensive
                # about shapes: explainers may return only rejected subset
                # explanations (ONLY_REJECTED) or full-length collections (FLAG).
                try:
                    if (
                        isinstance(res, RejectResult)
                        and getattr(res, "explanation", None) is not None
                    ):
                        explanation_obj = res.explanation
                        metadata = res.metadata or {}
                        rejected_mask = res.rejected
                        ambiguity_mask = metadata.get("ambiguity_mask")
                        novelty_mask = metadata.get("novelty_mask")
                        sizes = metadata.get("prediction_set_size")
                        pred_set_mask = metadata.get("prediction_set")
                        epsilon = metadata.get("epsilon")

                        # Normalize to list of target explanation objects
                        if hasattr(explanation_obj, "explanations"):
                            targets = explanation_obj.explanations
                        elif isinstance(explanation_obj, (list, tuple)):
                            targets = list(explanation_obj)
                        else:
                            targets = []

                        map_indices = _resolve_source_indices_for_payload(
                            policy=res.policy,
                            metadata=metadata,
                            rejected_mask=rejected_mask,
                            payload_count=len(targets),
                        )
                        if map_indices is None:
                            logging.getLogger(__name__).info(
                                "Skipping reject_context attachment due to unresolved source mapping."
                            )
                            map_indices = []

                        for local_idx, global_idx in enumerate(map_indices):
                            if local_idx >= len(targets):
                                continue
                            try:
                                is_rej = (
                                    bool(rejected_mask[global_idx])
                                    if rejected_mask is not None
                                    else False
                                )
                            except (IndexError, TypeError):
                                is_rej = False
                            rtype = None
                            try:
                                if ambiguity_mask is not None and bool(ambiguity_mask[global_idx]):
                                    rtype = "ambiguity"
                                elif novelty_mask is not None and bool(novelty_mask[global_idx]):
                                    rtype = "novelty"
                            except (IndexError, TypeError):
                                rtype = None
                            try:
                                psize = int(sizes[global_idx]) if sizes is not None else 1
                            except (IndexError, TypeError, ValueError):
                                psize = 1
                            try:
                                context_confidence = (
                                    None if epsilon is None else (1.0 - float(epsilon))
                                )
                            except (TypeError, ValueError):
                                context_confidence = None

                            prediction_set_ref = None
                            try:
                                if pred_set_mask is not None:
                                    row_mask = pred_set_mask[global_idx]
                                    if hasattr(row_mask, "flatten"):
                                        row_mask = row_mask.flatten()
                                    indices = np.flatnonzero(row_mask)
                                    prediction_set_ref = {
                                        "type": "indices",
                                        "indices": indices.tolist(),
                                    }
                            except (AttributeError, IndexError, TypeError, ValueError):
                                prediction_set_ref = None

                            rc = RejectContext(
                                rejected=is_rej,
                                reject_type=rtype,
                                prediction_set_size=psize,
                                confidence=context_confidence,
                                prediction_set_ref=prediction_set_ref,
                            )
                            # attach to the materialised explanation object when possible
                            try:
                                targets[local_idx].reject_context = rc
                            except Exception as exc:  # adr002_allow - best-effort attachment
                                logging.getLogger(__name__).debug(
                                    "failed to attach RejectContext to explanation: %s",
                                    exc,
                                    exc_info=True,
                                )
                except Exception as exc:  # adr002_allow - do not fail caller on propagation errors
                    logging.getLogger(__name__).debug(
                        "reject context propagation failed: %s", exc, exc_info=True
                    )

                # Upgrade to RejectCalibratedExplanations if possible to support
                # plotting and indexing directly on the result (Solution 1).
                if (
                    isinstance(res, RejectResult)
                    and getattr(res, "explanation", None) is not None
                    and hasattr(res.explanation, "explanations")
                ):
                    try:
                        try:
                            from ...explanations import (  # pylint: disable=import-outside-toplevel
                                AlternativeExplanations,
                            )

                            alt_cls = AlternativeExplanations
                        except Exception:  # adr002_allow - import environment variation
                            alt_cls = None
                            logging.getLogger(__name__).debug(
                                "AlternativeExplanations import failed during reject upgrade; falling back to attribute heuristic.",
                                exc_info=True,
                            )

                        if alt_cls is not None:
                            is_alternative = isinstance(res.explanation, alt_cls)
                        else:
                            # Deterministic fallback: when alternative class import
                            # is unavailable we avoid brittle duck-typing and keep
                            # generic reject wrapping.
                            is_alternative = False

                        if is_alternative:
                            return RejectAlternativeExplanations.from_collection(
                                res.explanation,
                                res.metadata or {},
                                res.policy,
                                rejected=res.rejected,
                            )
                        return RejectCalibratedExplanations.from_collection(
                            res.explanation,
                            res.metadata or {},
                            res.policy,
                            rejected=res.rejected,
                        )
                    except (
                        AttributeError,
                        CalibratedError,
                        DataShapeError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        warnings.warn(
                            "Reject wrapper upgrade skipped due to reject metadata/payload misalignment.",
                            UserWarning,
                            stacklevel=2,
                        )
                        logging.getLogger(__name__).debug(
                            "failed to upgrade to RejectCalibratedExplanations: %s",
                            exc,
                            exc_info=True,
                        )

                return res

        plugin, _identifier = self.ensure_plugin(mode)
        explainer_identifier = getattr(self.explainer, "explainer_id", None) or str(
            id(self.explainer)
        )
        per_instance_ignore = None
        features_arg = features_to_ignore or []
        if isinstance(features_arg, np.ndarray):
            features_arg = features_arg.tolist()
        if (
            features_arg
            and isinstance(features_arg, (list, tuple))
            and isinstance(features_arg[0], (list, tuple, np.ndarray))
        ):
            # User supplied per-instance masks
            per_instance_ignore_list: list[tuple[int, ...]] = []
            unknown_names: list[str] = []
            for mask in features_arg:
                mapped, unknown = self._coerce_ignore_items(mask)
                unknown_names.extend(unknown)
                per_instance_ignore_list.append(tuple(as_int_array(mapped).tolist()))
            if unknown_names:
                unknown_sorted = sorted(set(unknown_names))
                warnings.warn(
                    "Unknown feature names in features_to_ignore were ignored: "
                    + ", ".join(unknown_sorted[:5])
                    + ("..." if len(unknown_sorted) > 5 else ""),
                    UserWarning,
                    stacklevel=2,
                )
            per_instance_ignore = tuple(per_instance_ignore_list)
            flat_arrays = [
                np.asarray(mask, dtype=int) for mask in per_instance_ignore if len(mask) > 0
            ]
            if flat_arrays:
                flat_ignore = np.unique(np.concatenate(flat_arrays))
            else:
                flat_ignore = np.array([], dtype=int)
            features_to_ignore_flat = tuple(int(f) for f in flat_ignore.tolist())
        else:
            if isinstance(features_arg, (list, tuple)):
                mapped, unknown = self._coerce_ignore_items(features_arg)
                if unknown:
                    unknown_sorted = sorted(set(unknown))
                    warnings.warn(
                        "Unknown feature names in features_to_ignore were ignored: "
                        + ", ".join(unknown_sorted[:5])
                        + ("..." if len(unknown_sorted) > 5 else ""),
                        UserWarning,
                        stacklevel=2,
                    )
                features_to_ignore_flat = tuple(as_int_array(mapped).tolist())
            else:
                features_to_ignore_flat = tuple(as_int_array(features_arg).tolist())

        # Attempt FAST-based feature filtering if enabled and not already overridden by user
        feature_filter_config = getattr(self.explainer, "feature_filter_config", None)
        if (
            mode in {"factual", "alternative"}
            and per_instance_ignore is None
            and feature_filter_config is not None
            and getattr(feature_filter_config, "enabled", False) is True
        ):
            # Guardrail: warn when fast mode is auto-selected without explicit user intent
            warnings.warn(
                "Auto-selecting experimental 'fast' explanation mode for feature "
                "filtering. The 'fast' pathway is experimental and opt-in only. "
                "Set CE_FEATURE_FILTER=off or disable feature_filter_config to suppress.",
                UserWarning,
                stacklevel=2,
            )
            try:
                from ._feature_filter import compute_filtered_features_to_ignore

                # Run a lightweight FAST explanation on the same batch
                fast_results = self.invoke(
                    mode="fast",
                    x=x,
                    threshold=threshold,
                    low_high_percentiles=low_high_percentiles,
                    bins=bins,
                    # Pass the baseline ignore set so FAST respects it
                    features_to_ignore=features_to_ignore_flat,
                    extras=extras,
                    _ce_skip_reject=True,
                )

                filter_res = compute_filtered_features_to_ignore(
                    fast_results,
                    num_features=getattr(self.explainer, "num_features", None),
                    base_ignore=np.array(features_to_ignore_flat, dtype=int),
                    config=feature_filter_config,
                )

                # Update ignore sets with the filtered results
                features_to_ignore_flat = tuple(int(f) for f in filter_res.global_ignore)
                per_instance_ignore = tuple(
                    tuple(int(f) for f in row) for row in filter_res.per_instance_ignore
                )
            except Exception as exc:  # adr002_allow
                logging.getLogger(__name__).warning(
                    "FAST feature filtering failed; proceeding with baseline ignores: %s", exc
                )

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
                interval_summary=(extras or {}).get(
                    "interval_summary", self.explainer.interval_summary
                ),
                extras=dict(extras or {}),
                feature_filter_per_instance_ignore=per_instance_ignore,
            )
            monitor = self.explainer.plugin_manager.get_bridge_monitor(_identifier or mode)
            if monitor is not None:
                monitor.reset_usage()
            try:
                batch = plugin.explain_batch(x, request)
            except Exception as exc:  # ADR002_ALLOW: wrap plugin failures in ConfigurationError.  # pragma: no cover
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

        multi_labels_enabled = bool(kwargs.get("multi_labels_enabled", False))
        if multi_labels_enabled:
            from ...explanations.explanations import (  # pylint: disable=import-outside-toplevel
                MultiClassCalibratedExplanations,
            )
            from ._legacy_explain import (
                explain as legacy_explain,  # pylint: disable=import-outside-toplevel
            )

            multi_label_explanations = [{} for _ in range(len(x))]
            classes = np.unique(self.explainer.y_cal)
            if self.explainer.class_labels is not None and len(self.explainer.class_labels) > len(
                classes
            ):
                classes = np.arange(len(self.explainer.class_labels))
            # Warn when used on binary-labeled data: multi-label mode is intended for 3+ classes
            if len(classes) < 3:
                warnings.warn(
                    "multi_labels_enabled=True was requested but the problem appears to be binary; "
                    "this mode is intended for 3+ class problems.",
                    UserWarning,
                    stacklevel=2,
                )
            # Support per-class reject policies by delegating to the reject
            # orchestrator when a reject_policy is provided. For backward
            # compatibility, a None reject_policy performs the legacy loop.
            if reject_policy is None:
                for cls in classes:
                    labels = [cls for _ in range(len(x))]
                    explanations = legacy_explain(
                        self.explainer,
                        x,
                        threshold,
                        low_high_percentiles,
                        bins,
                        labels=labels,
                        features_to_ignore=features_to_ignore,
                        interval_summary=kwargs.get(
                            "interval_summary", self.explainer.interval_summary
                        ),
                    )
                    for i, explanation in enumerate(explanations):
                        multi_label_explanations[i][int(cls)] = explanation
            else:
                from ...core.reject.policy import RejectPolicy

                resolution = _resolve_effective_reject_policy(
                    reject_policy,
                    self.explainer,
                    default_policy=RejectPolicy.NONE,
                )
                effective_policy = resolution.policy
                confidence = kwargs.get("confidence", 0.95)

                # Prepare a per-class explain_fn closure that legacy_explain can call
                def make_explain_fn_for_class(cls_val):
                    def _explain_fn(x_subset, **inner_kw):
                        labels = [cls_val for _ in range(len(x_subset))]
                        return legacy_explain(
                            self.explainer,
                            x_subset,
                            threshold,
                            low_high_percentiles,
                            inner_kw.get("bins", bins),
                            labels=labels,
                            features_to_ignore=features_to_ignore,
                            interval_summary=inner_kw.get(
                                "interval_summary",
                                kwargs.get("interval_summary", self.explainer.interval_summary),
                            ),
                        )

                    return _explain_fn

                # Apply reject orchestration per-class and map results back
                for cls in classes:
                    explain_fn = make_explain_fn_for_class(int(cls))
                    try:
                        res = self.explainer.reject_orchestrator.apply_policy(
                            effective_policy,
                            x,
                            explain_fn=explain_fn,
                            bins=bins,
                            confidence=confidence,
                            threshold=threshold,
                        )
                    except Exception:  # adr002_allow
                        # Fallback to legacy explain if reject orchestration fails
                        labels = [cls for _ in range(len(x))]
                        explanations = legacy_explain(
                            self.explainer,
                            x,
                            threshold,
                            low_high_percentiles,
                            bins,
                            labels=labels,
                            features_to_ignore=features_to_ignore,
                            interval_summary=kwargs.get(
                                "interval_summary", self.explainer.interval_summary
                            ),
                        )
                        for i, explanation in enumerate(explanations):
                            multi_label_explanations[i][int(cls)] = explanation
                        continue

                    explanation_payload = getattr(res, "explanation", None)
                    rejected_mask = getattr(res, "rejected", None)

                    if explanation_payload is None:
                        continue

                    idxs = _resolve_source_indices_for_payload(
                        policy=getattr(res, "policy", effective_policy),
                        metadata=getattr(res, "metadata", {}),
                        rejected_mask=rejected_mask,
                        payload_count=len(explanation_payload),
                    )
                    if idxs is None:
                        continue
                    for j, inst_idx in enumerate(idxs):
                        multi_label_explanations[inst_idx][int(cls)] = explanation_payload[j]
            return MultiClassCalibratedExplanations(
                self.explainer, x, bins, len(classes), multi_label_explanations
            )

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
                interval_summary=kwargs.get("interval_summary", self.explainer.interval_summary),
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

        multi_labels_enabled = bool(kwargs.get("multi_labels_enabled", False))
        if multi_labels_enabled:
            from ...explanations.explanations import (  # pylint: disable=import-outside-toplevel
                MultiClassCalibratedExplanations,
            )
            from ._legacy_explain import (
                explain as legacy_explain,  # pylint: disable=import-outside-toplevel
            )

            multi_label_explanations = [{} for _ in range(len(x))]
            classes = np.unique(self.explainer.y_cal)
            if self.explainer.class_labels is not None and len(self.explainer.class_labels) > len(
                classes
            ):
                classes = np.arange(len(self.explainer.class_labels))
            # Warn when used on binary-labeled data: multi-label mode is intended for 3+ classes
            if len(classes) < 3:
                warnings.warn(
                    "multi_labels_enabled=True was requested but the problem appears to be binary; "
                    "this mode is intended for 3+ class problems.",
                    UserWarning,
                    stacklevel=2,
                )
            # Support per-class reject policies by delegating to the reject
            # orchestrator when a reject_policy is provided. For backward
            # compatibility, a None reject_policy performs the legacy loop.
            if reject_policy is None:
                for cls in classes:
                    labels = [cls for _ in range(len(x))]
                    explanations = legacy_explain(
                        self.explainer,
                        x,
                        threshold,
                        low_high_percentiles,
                        bins,
                        labels=labels,
                        features_to_ignore=features_to_ignore,
                        interval_summary=kwargs.get(
                            "interval_summary", self.explainer.interval_summary
                        ),
                    )
                    for i, explanation in enumerate(explanations):
                        multi_label_explanations[i][int(cls)] = explanation
            else:
                from ...core.reject.policy import RejectPolicy

                resolution = _resolve_effective_reject_policy(
                    reject_policy,
                    self.explainer,
                    default_policy=RejectPolicy.NONE,
                )
                effective_policy = resolution.policy
                confidence = kwargs.get("confidence", 0.95)

                def make_explain_fn_for_class(cls_val):
                    def _explain_fn(x_subset, **inner_kw):
                        labels = [cls_val for _ in range(len(x_subset))]
                        return legacy_explain(
                            self.explainer,
                            x_subset,
                            threshold,
                            low_high_percentiles,
                            inner_kw.get("bins", bins),
                            labels=labels,
                            features_to_ignore=features_to_ignore,
                            interval_summary=inner_kw.get(
                                "interval_summary",
                                kwargs.get("interval_summary", self.explainer.interval_summary),
                            ),
                        )

                    return _explain_fn

                for cls in classes:
                    explain_fn = make_explain_fn_for_class(int(cls))
                    try:
                        res = self.explainer.reject_orchestrator.apply_policy(
                            effective_policy,
                            x,
                            explain_fn=explain_fn,
                            bins=bins,
                            confidence=confidence,
                            threshold=threshold,
                        )
                    except Exception:  # adr002_allow
                        labels = [cls for _ in range(len(x))]
                        explanations = legacy_explain(
                            self.explainer,
                            x,
                            threshold,
                            low_high_percentiles,
                            bins,
                            labels=labels,
                            features_to_ignore=features_to_ignore,
                            interval_summary=kwargs.get(
                                "interval_summary", self.explainer.interval_summary
                            ),
                        )
                        for i, explanation in enumerate(explanations):
                            multi_label_explanations[i][int(cls)] = explanation
                        continue

                    explanation_payload = getattr(res, "explanation", None)
                    rejected_mask = getattr(res, "rejected", None)

                    if explanation_payload is None:
                        continue

                    idxs = _resolve_source_indices_for_payload(
                        policy=getattr(res, "policy", effective_policy),
                        metadata=getattr(res, "metadata", {}),
                        rejected_mask=rejected_mask,
                        payload_count=len(explanation_payload),
                    )
                    if idxs is None:
                        continue
                    for j, inst_idx in enumerate(idxs):
                        multi_label_explanations[inst_idx][int(cls)] = explanation_payload[j]
            return MultiClassCalibratedExplanations(
                self.explainer, x, bins, len(classes), multi_label_explanations
            )

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
                interval_summary=kwargs.get("interval_summary", self.explainer.interval_summary),
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

    def invoke_guarded_factual(  # pylint: disable=invalid-name
        self,
        x: Any,  # pylint: disable=invalid-name
        threshold: Any,
        low_high_percentiles: Tuple[float, float] | None,
        bins: Any,
        features_to_ignore: Any,
        per_instance_features_to_ignore: Any = None,
        reject_policy: Any | None = None,
        significance: float = 0.1,
        use_bonferroni: bool = False,
        merge_adjacent: bool = False,
        n_neighbors: int = 5,
        normalize_guard: bool = True,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute guarded factual explanation.

        Uses a multi-bin discretiser (``max_depth=3``) and prunes
        out-of-distribution leaves via KNN-based conformity testing.
        Returns a standard :class:`~calibrated_explanations.explanations.CalibratedExplanations`
        container whose per-instance explanations are
        :class:`~calibrated_explanations.explanations.guarded_explanation.GuardedFactualExplanation`.

        Parameters
        ----------
        x : array-like
            Test instances to explain.
        threshold : Any
            Threshold for probabilistic explanations.
        low_high_percentiles : tuple or None
            Low and high percentiles for calibrated intervals.
        bins : array-like or None
            Mondrian categories.
        features_to_ignore : sequence or None
            Feature indices to exclude.
        significance : float, default=0.1
            Conformity significance level.
        use_bonferroni : bool, default=False
            Whether to apply per-feature Bonferroni correction.
        merge_adjacent : bool, default=False
            Merge adjacent conforming bins into wider intervals.
        n_neighbors : int, default=5
            KNN neighbour count for the in-distribution guard.
        normalize_guard : bool, default=True
            Apply per-feature normalisation in the guard.
        verbose : bool, default=False
            When True, emit UserWarnings for guarded-explanation diagnostics.
        **kwargs : Any
            Currently unused; reserved for future parameters.

        Returns
        -------
        CalibratedExplanations
            Container with :class:`GuardedFactualExplanation` objects.
        """
        import numpy as np  # pylint: disable=import-outside-toplevel

        from ...core.reject.policy import RejectPolicy
        from ._guarded_explain import guarded_explain  # pylint: disable=import-outside-toplevel

        if not kwargs.pop("_ce_skip_reject", False):
            confidence = kwargs.get("confidence", 0.95)
            resolution = _resolve_effective_reject_policy(
                reject_policy,
                self.explainer,
                default_policy=getattr(self.explainer, "default_reject_policy", RejectPolicy.NONE),
            )
            effective_policy = resolution.policy
            if effective_policy is not RejectPolicy.NONE:
                with contextlib.suppress(Exception):
                    _ = self.explainer.reject_orchestrator
                return self.explainer.reject_orchestrator.apply_policy(
                    effective_policy,
                    x,
                    explain_fn=lambda x_subset, **inner_kw: self.invoke_guarded_factual(
                        x_subset,
                        threshold=threshold,
                        low_high_percentiles=low_high_percentiles,
                        bins=inner_kw.get("bins", bins),
                        features_to_ignore=features_to_ignore,
                        per_instance_features_to_ignore=per_instance_features_to_ignore,
                        reject_policy=RejectPolicy.NONE,
                        significance=significance,
                        use_bonferroni=use_bonferroni,
                        merge_adjacent=merge_adjacent,
                        n_neighbors=n_neighbors,
                        normalize_guard=normalize_guard,
                        verbose=verbose,
                        _ce_skip_reject=True,
                    ),
                    bins=bins,
                    confidence=confidence,
                    threshold=threshold,
                )

        per_instance_ignore = per_instance_features_to_ignore

        features_arg = features_to_ignore or []
        if isinstance(features_arg, np.ndarray):
            features_arg = features_arg.tolist()

        if (
            features_arg
            and isinstance(features_arg, (list, tuple))
            and isinstance(features_arg[0], (list, tuple, np.ndarray))
        ):
            per_instance_ignore_list: list[tuple[int, ...]] = []
            unknown_names: list[str] = []
            for mask in features_arg:
                mapped, unknown = self._coerce_ignore_items(mask)
                unknown_names.extend(unknown)
                per_instance_ignore_list.append(tuple(as_int_array(mapped).tolist()))
            if unknown_names and verbose:
                unknown_sorted = sorted(set(unknown_names))
                warnings.warn(
                    "Unknown feature names in features_to_ignore were ignored: "
                    + ", ".join(unknown_sorted[:5])
                    + ("..." if len(unknown_sorted) > 5 else ""),
                    UserWarning,
                    stacklevel=2,
                )
            per_instance_ignore = tuple(per_instance_ignore_list)
            flat_arrays = [
                np.asarray(mask, dtype=int) for mask in per_instance_ignore if len(mask) > 0
            ]
            if flat_arrays:
                flat_ignore = np.unique(np.concatenate(flat_arrays))
            else:
                flat_ignore = np.array([], dtype=int)
            features_to_ignore_flat = tuple(int(f) for f in flat_ignore.tolist())
        else:
            if isinstance(features_arg, (list, tuple)):
                mapped, unknown = self._coerce_ignore_items(features_arg)
                if unknown and verbose:
                    unknown_sorted = sorted(set(unknown))
                    warnings.warn(
                        "Unknown feature names in features_to_ignore were ignored: "
                        + ", ".join(unknown_sorted[:5])
                        + ("..." if len(unknown_sorted) > 5 else ""),
                        UserWarning,
                        stacklevel=2,
                    )
                features_to_ignore_flat = tuple(as_int_array(mapped).tolist())
            else:
                features_to_ignore_flat = tuple(as_int_array(features_arg).tolist())

        x_arr = np.atleast_2d(np.asarray(x))
        return guarded_explain(
            self.explainer,
            x_arr,
            mode="factual",
            threshold=threshold,
            low_high_percentiles=low_high_percentiles
            if low_high_percentiles is not None
            else (5, 95),
            mondrian_bins=bins,
            features_to_ignore=features_to_ignore_flat,
            per_instance_features_to_ignore=per_instance_ignore,
            significance=significance,
            use_bonferroni=use_bonferroni,
            merge_adjacent=merge_adjacent,
            n_neighbors=n_neighbors,
            normalize_guard=normalize_guard,
            verbose=verbose,
        )

    def invoke_guarded_alternative(  # pylint: disable=invalid-name
        self,
        x: Any,  # pylint: disable=invalid-name
        threshold: Any,
        low_high_percentiles: Tuple[float, float] | None,
        bins: Any,
        features_to_ignore: Any,
        per_instance_features_to_ignore: Any = None,
        reject_policy: Any | None = None,
        significance: float = 0.1,
        use_bonferroni: bool = False,
        merge_adjacent: bool = False,
        n_neighbors: int = 5,
        normalize_guard: bool = True,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute guarded alternative explanation.

        Uses a multi-bin discretiser (``max_depth=3``) and prunes
        out-of-distribution leaves via KNN-based conformity testing.
        Returns a standard :class:`~calibrated_explanations.explanations.AlternativeExplanations`
        container whose per-instance explanations are
        :class:`~calibrated_explanations.explanations.guarded_explanation.GuardedAlternativeExplanation`.
        Only conforming non-factual bins are exposed as alternatives.

        Parameters
        ----------
        x : array-like
            Test instances to explain.
        threshold : Any
            Threshold for probabilistic explanations.
        low_high_percentiles : tuple or None
            Low and high percentiles for calibrated intervals.
        bins : array-like or None
            Mondrian categories.
        features_to_ignore : sequence or None
            Feature indices to exclude.
        significance : float, default=0.1
            Conformity significance level.
        use_bonferroni : bool, default=False
            Whether to apply per-feature Bonferroni correction.
        merge_adjacent : bool, default=False
            Merge adjacent conforming bins into wider intervals.
        n_neighbors : int, default=5
            KNN neighbour count for the in-distribution guard.
        normalize_guard : bool, default=True
            Apply per-feature normalisation in the guard.
        verbose : bool, default=False
            When True, emit UserWarnings for guarded-explanation diagnostics.
        **kwargs : Any
            Currently unused; reserved for future parameters.

        Returns
        -------
        AlternativeExplanations
            Container with :class:`GuardedAlternativeExplanation` objects.
        """
        import numpy as np  # pylint: disable=import-outside-toplevel

        from ...core.reject.policy import RejectPolicy
        from ._guarded_explain import guarded_explain  # pylint: disable=import-outside-toplevel

        if not kwargs.pop("_ce_skip_reject", False):
            confidence = kwargs.get("confidence", 0.95)
            resolution = _resolve_effective_reject_policy(
                reject_policy,
                self.explainer,
                default_policy=getattr(self.explainer, "default_reject_policy", RejectPolicy.NONE),
            )
            effective_policy = resolution.policy
            if effective_policy is not RejectPolicy.NONE:
                with contextlib.suppress(Exception):
                    _ = self.explainer.reject_orchestrator
                return self.explainer.reject_orchestrator.apply_policy(
                    effective_policy,
                    x,
                    explain_fn=lambda x_subset, **inner_kw: self.invoke_guarded_alternative(
                        x_subset,
                        threshold=threshold,
                        low_high_percentiles=low_high_percentiles,
                        bins=inner_kw.get("bins", bins),
                        features_to_ignore=features_to_ignore,
                        per_instance_features_to_ignore=per_instance_features_to_ignore,
                        reject_policy=RejectPolicy.NONE,
                        significance=significance,
                        use_bonferroni=use_bonferroni,
                        merge_adjacent=merge_adjacent,
                        n_neighbors=n_neighbors,
                        normalize_guard=normalize_guard,
                        verbose=verbose,
                        _ce_skip_reject=True,
                    ),
                    bins=bins,
                    confidence=confidence,
                    threshold=threshold,
                )

        per_instance_ignore = per_instance_features_to_ignore

        features_arg = features_to_ignore or []
        if isinstance(features_arg, np.ndarray):
            features_arg = features_arg.tolist()

        if (
            features_arg
            and isinstance(features_arg, (list, tuple))
            and isinstance(features_arg[0], (list, tuple, np.ndarray))
        ):
            per_instance_ignore_list: list[tuple[int, ...]] = []
            unknown_names: list[str] = []
            for mask in features_arg:
                mapped, unknown = self._coerce_ignore_items(mask)
                unknown_names.extend(unknown)
                per_instance_ignore_list.append(tuple(as_int_array(mapped).tolist()))
            if unknown_names and verbose:
                unknown_sorted = sorted(set(unknown_names))
                warnings.warn(
                    "Unknown feature names in features_to_ignore were ignored: "
                    + ", ".join(unknown_sorted[:5])
                    + ("..." if len(unknown_sorted) > 5 else ""),
                    UserWarning,
                    stacklevel=2,
                )
            per_instance_ignore = tuple(per_instance_ignore_list)
            flat_arrays = [
                np.asarray(mask, dtype=int) for mask in per_instance_ignore if len(mask) > 0
            ]
            if flat_arrays:
                flat_ignore = np.unique(np.concatenate(flat_arrays))
            else:
                flat_ignore = np.array([], dtype=int)
            features_to_ignore_flat = tuple(int(f) for f in flat_ignore.tolist())
        else:
            if isinstance(features_arg, (list, tuple)):
                mapped, unknown = self._coerce_ignore_items(features_arg)
                if unknown and verbose:
                    unknown_sorted = sorted(set(unknown))
                    warnings.warn(
                        "Unknown feature names in features_to_ignore were ignored: "
                        + ", ".join(unknown_sorted[:5])
                        + ("..." if len(unknown_sorted) > 5 else ""),
                        UserWarning,
                        stacklevel=2,
                    )
                features_to_ignore_flat = tuple(as_int_array(mapped).tolist())
            else:
                features_to_ignore_flat = tuple(as_int_array(features_arg).tolist())

        x_arr = np.atleast_2d(np.asarray(x))
        return guarded_explain(
            self.explainer,
            x_arr,
            mode="alternative",
            threshold=threshold,
            low_high_percentiles=low_high_percentiles
            if low_high_percentiles is not None
            else (5, 95),
            mondrian_bins=bins,
            features_to_ignore=features_to_ignore_flat,
            per_instance_features_to_ignore=per_instance_ignore,
            significance=significance,
            use_bonferroni=use_bonferroni,
            merge_adjacent=merge_adjacent,
            n_neighbors=n_neighbors,
            normalize_guard=normalize_guard,
            verbose=verbose,
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
            existing = None
            with contextlib.suppress(CalibratedError):
                existing = pm.explanation_plugin_fallbacks.get(mode)
            if not existing:
                with contextlib.suppress(CalibratedError):
                    pm.initialize_chains()

        raw_override = self.explainer.plugin_manager.explanation_plugin_overrides.get(mode)
        override = self.explainer.plugin_manager.coerce_plugin_override(raw_override)
        if override is not None and not isinstance(override, str):
            plugin = override
            identifier = getattr(plugin, "plugin_meta", {}).get("name")
            meta = getattr(plugin, "plugin_meta", {})
            if isinstance(meta, Mapping):
                trusted = meta.get("trusted", meta.get("trust", True))
                if not bool(trusted):
                    warnings.warn(
                        f"Using untrusted explanation plugin '{identifier}' via explicit override. "
                        "Ensure you trust the source of this plugin.",
                        UserWarning,
                        stacklevel=2,
                    )
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

            plugin, metadata, reason = self.explainer.plugin_manager.resolve_explanation_plugin(
                identifier,
                allow_untrusted=allow_untrusted,
                is_preferred=is_preferred,
                is_explicit_override=is_explicit_override,
            )
            if reason == "untrusted":
                raise ConfigurationError(
                    "Explanation plugin configuration failed: "
                    + identifier
                    + " is untrusted; explicitly trust the plugin or pass an explicit override"
                ) from None
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
        builder_payload = None
        if callable(builder):
            with contextlib.suppress(Exception):
                builder_payload = builder()
                if isinstance(builder_payload, dict):
                    payload.update(builder_payload)

        # Fallback: build compact telemetry from available prediction payloads
        with contextlib.suppress(Exception):
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
            deps = getattr(first_explanation, "metadata", None) or {}
            if not deps and isinstance(builder_payload, dict):
                deps = builder_payload
            if isinstance(deps, dict):
                interval_deps = deps.get("interval_dependencies") or deps.get("metadata", {}).get(
                    "interval_dependencies"
                )
                if interval_deps:
                    payload.setdefault("interval_dependencies", interval_deps)

        return payload
