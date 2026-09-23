"""Explain black-box learners using calibrated prediction intervals.

This module implements the core :class:`CalibratedExplainer` which fits
interval calibrators on calibration data and exposes methods for generating
factual and alternative explanations augmented with uncertainty information.

The implementation follows the approach described in
"Calibrated Explanations: with Uncertainty Information and Counterfactuals"
by Helena Löfström et al.
"""

# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-lines, too-many-positional-arguments, too-many-public-methods
from __future__ import annotations

import copy
import logging
import sys
import warnings
import contextlib
from time import time
from typing import TYPE_CHECKING

import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from ..explanations import AlternativeExplanations, CalibratedExplanations
    from ..plugins.manager import PluginManager

from ..api.params import (
    reject_cross_surface_kwargs,
    reject_removed_aliases,
    reject_removed_guarded_kwargs,
    reject_removed_normalization_kwarg,
    reject_removed_reject_kwargs,
    reject_unknown_public_kwargs,
)

try:
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    try:  # pragma: no cover - optional dependency path
        import tomli as _tomllib  # type: ignore[assignment]
    except ModuleNotFoundError:  # pragma: no cover - tomllib unavailable
        _tomllib = None  # type: ignore[assignment]

# Core imports (no cross-sibling dependencies)
from ..calibration.interval_wrappers import is_fast_interval_collection
from ..utils import assert_threshold, check_is_fitted, convert_targets_to_numeric, safe_isinstance

from ..utils.exceptions import (
    DataShapeError,
    ValidationError,
)
from .validation import (
    normalize_mode,
    validate_bool_parameter,
    validate_classification_calibration_targets,
    validate_explainer_init_kwargs,
    validate_features_to_ignore,
    validate_inputs_matrix,
    validate_low_high_percentiles,
)
from .prediction.interval_summary import IntervalSummary, coerce_interval_summary
from .prediction_helpers import resolve_conditional_bins

# Lazy imports deferred to avoid cross-sibling coupling
# These are imported inside methods/properties where used
# - perf (CalibratorCache, ParallelExecutor) - lazy in __init__
# - plotting (_plot_global) - lazy in plotting methods
# - explanations (AlternativeExplanations, CalibratedExplanations) - lazy as needed
# - integrations (LimeHelper, ShapHelper) - lazy in __init__
# - api.params (canonicalize_kwargs, etc.) - lazy in param handling
# - plugins (IntervalCalibratorContext, PluginManager, LegacyPredictBridge) - lazy in __init__
# - utils.discretizers (EntropyDiscretizer, RegressorDiscretizer) - lazy in validation

# ADR-038 5C: extends the D3/5A/5B fail-fast policy (previously WrapCalibratedExplainer
# only) to CalibratedExplainer itself, so direct users of this class (bypassing the
# wrapper) get the same protection. __init__/predict/predict_proba are closed
# surfaces (any unrecognized name raises via reject_unknown_public_kwargs).
# explain_factual/explore_alternatives retain the ADR-038 §3 experimental
# plugin-forwarding exception: a name genuinely unknown anywhere is treated as
# plugin-defined and passed through; only names known on one of the *closed*
# surfaces below but not valid here are rejected (reject_cross_surface_kwargs).
# explain_fast has no **kwargs at all (already fully typed) and needs no set.

# These allow-lists are the single source of truth for kwargs accepted on each
# CalibratedExplainer surface. WrapCalibratedExplainer derives its own per-method
# gates from them (v0.11.6 Task 5D) -- never redefine the same surface in both
# modules, or the two gates will drift apart (a name accepted here must also be
# accepted by the wrapper).

# Explicit formal parameters of CalibratedExplainer.__init__ (besides
# learner/x_cal/y_cal). They never reach __init__'s **kwargs, but callers that
# forward a kwargs dict (WrapCalibratedExplainer.calibrate) accept them by name,
# and the cross-surface check needs them as "known elsewhere". Kept in sync with
# the signature by tests/unit/core/test_parameter_surface_contracts.py.
_INIT_EXPLICIT_PARAMS: frozenset[str] = frozenset(
    {
        "mode",
        "feature_names",
        "categorical_features",
        "categorical_labels",
        "class_labels",
        "bins",
        "difficulty_estimator",
    }
)

# used by: CalibratedExplainer.__init__ only.
_INIT_KWARGS: frozenset[str] = frozenset(
    {
        "perf_cache",
        "perf_parallel",
        "preprocessor_metadata",
        "predict_function",
        "suppress_crepes_errors",
        "oob",
        "seed",
        "sample_percentiles",
        "verbose",
        "interval_summary",
        "fast",
        "noise_type",
        "scale_factor",
        "severity",
        "condition_source",
        "features_to_ignore",
        "reject",
        "default_reject_policy",
        "factual_plugin",
        "alternative_plugin",
        "fast_plugin",
        "interval_plugin",
        "fast_interval_plugin",
        "plot_style",
    }
)

# used by: CalibratedExplainer.predict only. uq_interval/calibrated are already
# explicit named parameters and never reach **kwargs. "_ce_skip_reject" is an
# internal escape hatch used by core/explain/orchestrator.py; "show"/
# "style_override" are stripped defensively before validation (see predict()),
# not allow-listed, since plot()'s kwargs flow into predict() via plot_global().
_PREDICT_KWARGS: frozenset[str] = frozenset(
    {
        "threshold",
        "low_high_percentiles",
        "bins",
        "classes",
        "feature",
        "reject_policy",
        "reject_confidence",
        "interval_summary",
        "_ce_skip_reject",
    }
)

# used by: CalibratedExplainer.predict_proba only. uq_interval/calibrated/
# threshold are already explicit named parameters and never reach **kwargs.
_PREDICT_PROBA_KWARGS: frozenset[str] = frozenset(
    {
        "bins",
        "reject_policy",
        "reject_confidence",
        "interval_summary",
        "normalization",
        "_ce_skip_reject",
    }
)

# used by: explain_factual() and explore_alternatives() (identical surface).
# guarded_options/reject_policy/_use_plugin are already explicit named
# parameters and never reach **kwargs; listed here only for readers.
_EXPLAIN_KWARGS: frozenset[str] = frozenset(
    {
        "threshold",
        "low_high_percentiles",
        "bins",
        "features_to_ignore",
        "guarded_options",
        "reject_policy",
        "reject_confidence",  # active only when reject_policy is set
        "multi_labels_enabled",  # EXPERIMENTAL, ADR-038 §3
        "interval_summary",  # EXPERIMENTAL, ADR-038 §3
        "verbose",
    }
)

# Reference set for the explain_factual/explore_alternatives cross-surface check:
# names known on a *closed* CalibratedExplainer surface. A name in this set but
# not in _EXPLAIN_KWARGS is cross-method contamination and is rejected; a name
# in neither is treated as plugin-defined and passed through untouched (ADR-038
# §3 exception). Includes __init__/predict/predict_proba's own explicit formal
# parameter names (mode, feature_names, ...) even though those never reach
# **kwargs on their own methods -- they still need to be recognized as
# "known elsewhere" so e.g. explain_factual(x, mode="regression") is rejected
# instead of silently treated as a plugin-forwarded key.
_CLOSED_SURFACE_KWARGS: frozenset[str] = (
    _INIT_KWARGS
    | _PREDICT_KWARGS
    | _PREDICT_PROBA_KWARGS
    | _INIT_EXPLICIT_PARAMS
    | {
        "uq_interval",
        "calibrated",
    }
)


def _log_forwarded_explain_kwargs(
    logger: logging.Logger, surface: str, kwargs: dict[str, Any], *, allowed: frozenset[str]
) -> None:
    forwarded = sorted(set(kwargs) - allowed)
    if not forwarded:
        return
    logger.info("%s forwarding explanation keyword arguments to plugins: %s", surface, forwarded)


class CalibratedExplainer:
    """Explain a fitted learner using calibrated intervals and plugins.

    The explainer fits internal interval calibrators on provided calibration
    data and exposes high-level APIs for producing `CalibratedExplanations`.

    Recommended use is to use `WrapCalibratedExplainer`, which is a wrapper around the learner and this explainer.

    Examples
    --------
    >>> from calibrated_explanations import CalibratedExplainer
    >>> explainer = CalibratedExplainer(learner, X_cal, y_cal, mode="classification")
    >>> explanations = explainer.explain_factual(X_test)
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def __init__(
        self,
        learner,
        x_cal,
        y_cal,
        mode="classification",
        feature_names=None,
        categorical_features=None,
        categorical_labels=None,
        class_labels=None,
        bins=None,
        difficulty_estimator=None,
        **kwargs,
    ) -> None:
        """Initialize the explainer with calibration data and metadata.

        Parameters
        ----------
        learner : Any
            Predictive learner that must already expose ``fit``/``predict`` and,
            for classification, ``predict_proba``.
        x_cal : array-like of shape (n_calibration_samples, n_features)
            Calibration feature matrix used to fit interval calibrators.
        y_cal : array-like of shape (n_calibration_samples,)
            Calibration targets paired with ``x_cal``.
        mode : {"classification", "regression"}, default="classification"
            Operating mode controlling which calibrators/plugins are used.
        feature_names : Sequence[str] or None, optional
            Optional list of human-readable feature names.
        categorical_features : Sequence[int] or None, optional
            Indices describing which features should be treated as categorical.
        categorical_labels : Mapping[int, Mapping[int, str]] or None, optional
            Optional mapping translating categorical feature values to labels.
        class_labels : Mapping[int, str] or None, optional
            Optional mapping translating class indices to display labels.
        bins : array-like or None, optional
            Pre-computed Mondrian categories for fast explanations.
        difficulty_estimator : Any or None, optional
            Optional crepes ``DifficultyEstimator`` instance for regression tasks.
        **kwargs : Any
            Advanced configuration flags preserved for backward compatibility.
            Includes `condition_source` ("observed" or "prediction", default="prediction").

        Notes
        -----
        Minimal lifecycle logging is available at INFO level. To enable, run::

            import logging
            logging.getLogger("calibrated_explanations").setLevel(logging.INFO)
        """
        reject_removed_aliases(kwargs)
        reject_removed_guarded_kwargs(kwargs)
        reject_removed_reject_kwargs(kwargs)
        reject_removed_normalization_kwarg(kwargs)
        reject_unknown_public_kwargs(
            kwargs, allowed=_INIT_KWARGS, surface="CalibratedExplainer.__init__"
        )

        perf_cache = kwargs.pop("perf_cache", None)
        perf_parallel = kwargs.pop("perf_parallel", None)

        init_time = time()
        self._initialized = False
        preprocessor_metadata = kwargs.pop("preprocessor_metadata", None)
        if isinstance(preprocessor_metadata, Mapping):
            self._preprocessor_metadata: Dict[str, Any] | None = dict(preprocessor_metadata)
        else:
            self._preprocessor_metadata = None
        check_is_fitted(learner)
        self.learner = learner
        validate_inputs_matrix(x_cal, y_cal, require_y=True, allow_nan=False)
        mode, kwargs = validate_explainer_init_kwargs(
            kwargs,
            mode=mode,
            n_features=int(np.asarray(x_cal).shape[1]),
        )
        self.predict_function = kwargs.get("predict_function")
        if self.predict_function is None:
            self.predict_function = (
                learner.predict_proba if mode == "classification" else learner.predict
            )
        # Optionally suppress or convert low-level crepes errors into clearer messages.
        # Caller can pass suppress_crepes_errors=True via kwargs to avoid raising on
        # crepes broadcasting/shape errors (useful for synthetic tiny datasets).
        self.suppress_crepes_errors = kwargs.get("suppress_crepes_errors", False)
        self.oob = kwargs.get("oob", False)
        self._categorical_value_counts_cache: Dict[int, Dict[Any, int]] | None = None
        self._numeric_sorted_cache: Dict[int, np.ndarray] | None = None
        self._calibration_summary_shape: Tuple[int, int] | None = None
        if self.oob:
            if mode == "classification":
                y_oob_proba = self.learner.oob_decision_function_
                if (
                    len(y_oob_proba.shape) == 1 or y_oob_proba.shape[1] == 1
                ):  # Binary classification
                    y_oob_idx = (np.ravel(y_oob_proba) > 0.5).astype(int)
                else:  # Multiclass classification
                    y_oob_idx = np.argmax(y_oob_proba, axis=1)
                if safe_isinstance(y_cal, "pandas.core.arrays.categorical.Categorical"):
                    y_oob = y_cal.categories[y_oob_idx]
                else:
                    classes_ = np.asarray(getattr(self.learner, "classes_", None))
                    if classes_.ndim == 1 and classes_.size >= 2:
                        # Map OOB class indices back through the fitted
                        # learner's class labels instead of casting the raw
                        # integer index to y_cal's dtype, which silently
                        # produced wrong labels (e.g. "0"/"1" digit strings)
                        # for non-contiguous-numeric or string class spaces.
                        y_oob = classes_[y_oob_idx]
                    else:
                        y_oob = y_oob_idx.astype(np.dtype(y_cal.dtype))
            else:
                y_oob = self.learner.oob_prediction_
            if len(x_cal) != len(y_oob):
                raise DataShapeError(
                    "The length of the out-of-bag predictions does not match the length of x_cal."
                )
            y_cal = y_oob
        self.x_cal = x_cal
        self.y_cal = y_cal
        if mode == "classification":
            validate_classification_calibration_targets(self.y_cal, learner=self.learner)

        # Initialize RNG with seed
        from ..utils import set_rng_seed  # pylint: disable=import-outside-toplevel

        seed = kwargs.get("seed", 42)
        self.seed = seed
        self.rng = set_rng_seed(seed)

        self.sample_percentiles = kwargs.get("sample_percentiles", [25, 50, 75])
        self.verbose = kwargs.get("verbose", False)
        self.bins = bins
        self.interval_summary = coerce_interval_summary(
            kwargs.get("interval_summary", IntervalSummary.REGULARIZED_MEAN)
        )

        self._fast = kwargs.get("fast", False)
        self._noise_type = kwargs.get("noise_type", "uniform")
        self._scale_factor = kwargs.get("scale_factor", 5)
        self._severity = kwargs.get("severity", 1)
        # Prefer explicit caller value; otherwise default to 'prediction' as of v0.10.3
        if "condition_source" in kwargs:
            self.condition_source = kwargs.get("condition_source")
        else:
            self.condition_source = "prediction"
            logging.getLogger(__name__).info(
                "condition_source not provided; defaulting to 'prediction' (v0.10.3)"
            )
            if self.verbose:
                warnings.warn(
                    "condition_source not provided; defaulting to 'prediction' in v0.10.3. "
                    "Pass condition_source='observed' to retain previous behaviour.",
                    UserWarning,
                    stacklevel=2,
                )

        if self.condition_source not in {"observed", "prediction"}:
            raise ValidationError(
                "condition_source must be either 'observed' or 'prediction'",
                details={
                    "param": "condition_source",
                    "value": self.condition_source,
                    "allowed": ("observed", "prediction"),
                },
            )

        self.categorical_labels = categorical_labels
        self.class_labels = class_labels
        if categorical_features is None:
            if categorical_labels is not None:
                categorical_features = categorical_labels.keys()
            else:
                categorical_features = []
        self.categorical_features = list(categorical_features)
        self._invalidate_calibration_summaries()
        self.features_to_ignore = kwargs.get("features_to_ignore", [])

        # Identify constant calibration features that can be ignored downstream
        from .calibration_helpers import identify_constant_features  # pylint: disable=import-outside-toplevel

        constant_ignore = identify_constant_features(self.x_cal)
        try:
            self.features_to_ignore = (
                np.union1d(self.features_to_ignore, constant_ignore).astype(int).tolist()
            )
        except (TypeError, ValueError):
            # Be defensive: if union fails due to incompatible types, fall back to constants.
            self.features_to_ignore = list(constant_ignore)

        if feature_names is None:
            feature_names = (
                self._X_cal[0].keys()
                if isinstance(self._X_cal[0], dict)
                else [str(i) for i in range(self.num_features)]
            )
        self._feature_names = list(feature_names)

        if mode == "classification":
            original_class_values = np.unique(np.asarray(self.y_cal))
            self.y_cal_numeric, self.label_map = convert_targets_to_numeric(self.y_cal)
            self.y_cal = self.y_cal_numeric  # save to _y_cal to avoid append
            self.original_class_values = np.asarray(original_class_values)
            if self.label_map is not None:
                if self.class_labels is None:
                    self.class_labels = {
                        int(encoded_label): str(original_label)
                        for original_label, encoded_label in self.label_map.items()
                    }
                elif isinstance(self.class_labels, Mapping):
                    normalized_class_labels = {}
                    for original_label, encoded_label in self.label_map.items():
                        if original_label in self.class_labels:
                            normalized_class_labels[int(encoded_label)] = self.class_labels[
                                original_label
                            ]
                        elif str(original_label) in self.class_labels:
                            normalized_class_labels[int(encoded_label)] = self.class_labels[
                                str(original_label)
                            ]
                        elif int(encoded_label) in self.class_labels:
                            normalized_class_labels[int(encoded_label)] = self.class_labels[
                                int(encoded_label)
                            ]
                    if len(normalized_class_labels) == len(self.label_map):
                        self.class_labels = normalized_class_labels
            elif self.class_labels is None:
                self.class_labels = {int(label): str(label) for label in np.unique(self.y_cal)}
        else:
            self.label_map = None
            self.class_labels = None
            self.original_class_values = None

        self.discretizer: Any = None
        self.discretized_X_cal: Optional[np.ndarray] = None
        # Predeclare attributes for fast mode to satisfy type checkers
        self.fast_x_cal: Optional[np.ndarray] = None
        self.scaled_x_cal: Optional[np.ndarray] = None
        self.scaled_y_cal: Optional[np.ndarray] = None

        self.feature_values: Dict[int, List[Any]] = {}
        self.feature_frequencies: Dict[int, np.ndarray] = {}

        # Lazy import helper integrations (deferred from module level)
        from ..integrations import LimeHelper, ShapHelper

        self.latest_explanation: Optional[CalibratedExplanations] = None
        self._lime_helper = LimeHelper(self)
        self._shap_helper = ShapHelper(self)
        self.reject = kwargs.get("reject", False)
        # Optional default reject policy for explainer-level defaults
        from .reject.policy import RejectPolicy as _RejectPolicy

        self.default_reject_policy = kwargs.get("default_reject_policy", _RejectPolicy.NONE)

        self.set_difficulty_estimator(difficulty_estimator, initialize=False)
        self.set_mode(str.lower(mode), initialize=False)

        # Lazy import orchestrator and plugin management (deferred from module level)
        from ..plugins.manager import PluginManager
        from ..plugins.builtins import LegacyPredictBridge
        from ..cache import CalibratorCache

        # Initialize plugin manager (SINGLE SOURCE OF TRUTH for plugin management)
        # PluginManager handles ALL plugin initialization including:
        # - Reading pyproject.toml configurations
        # - Setting up plugin overrides from kwargs
        # - Creating and initializing orchestrators
        # - Building plugin fallback chains
        self.plugin_manager = PluginManager(self)
        self.plugin_manager.initialize_from_kwargs(kwargs)
        self.plugin_manager.initialize_orchestrators()

        # Initialize interval learner after orchestrators are ready
        self.prediction_orchestrator.interval_registry.initialize()

        self.perf_cache: CalibratorCache[Any] | None = perf_cache

        # Initialize parallel executor (ADR-004: Honor CE_PARALLEL overrides)
        self._perf_parallel: Any | None = self._resolve_parallel_executor(perf_parallel)

        # Orchestrator references are now accessed via properties that delegate to PluginManager
        # No direct assignment needed - properties handle the delegation

        # Reject learner initialization
        if kwargs.get("reject", False):
            self.plugin_manager.initialize_orchestrators()
            self.reject_learner = self.reject_orchestrator.initialize_reject_learner(
                calibration_set=None, threshold=None, ncf=None, w=0.5
            )
        else:
            self.reject_learner = None

        self._predict_bridge = LegacyPredictBridge(self)

        self.init_time = time() - init_time

    def __deepcopy__(self, memo):
        """Safely deepcopy the explainer, handling circular references.

        ``_perf_parallel`` (a worker pool/executor) and ``perf_cache`` (a
        thread-safe cache guarded by a lock) are shared by reference with the
        original instance: both hold unpicklable OS-level resources and
        neither carries explanation-affecting state, so sharing them cannot
        leak mutation between a copy (for example a
        :class:`~calibrated_explanations.explanations.explanations.FrozenCalibratedExplainer`
        snapshot) and the live explainer.

        ``latest_explanation`` is deliberately *not* carried over: the copy
        starts with ``None`` (the same state :meth:`reset` produces). At
        deepcopy time the live explainer's ``latest_explanation`` still
        points at the *previous* explanation collection, and every collection
        embeds a ``FrozenCalibratedExplainer`` snapshot (a full explainer
        copy) of its own. Sharing the pointer by reference therefore links
        each new snapshot to the previous explanation, forming an unbounded
        retention chain (explanation -> snapshot -> previous explanation ->
        older snapshot -> ...) that keeps every historical snapshot reachable
        for the life of the explainer -- the repeated reject-flow RSS leak
        closed by Task 56 / decision D-56. Deep-copying the pointer instead
        would walk and duplicate that same chain on every copy, which grows
        without bound and makes ``FrozenCalibratedExplainer`` construction
        arbitrarily slow. Starting from ``None`` is safe: each explanation
        object carries its own snapshot independently of this bookkeeping
        pointer, so a *snapshot's* pointer to "the most recent explanation"
        has no consumer.

        Every other attribute -- including ``learner``, ``rng``,
        ``_plugin_manager`` (and, through it, the interval learner and
        prediction orchestrator), and the LIME/SHAP integration helpers -- is
        deep-copied so that mutating the copy can never affect the original.
        """
        if id(self) in memo:
            return memo[id(self)]
        # Create a shallow copy without calling __init__
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # These attributes are deliberately shared by reference rather than
        # deep-copied; see the class-level rationale in the docstring above.
        deliberately_shared_keys = {
            "_perf_parallel",
            "perf_cache",
        }

        for k, v in self.__dict__.items():
            if k in deliberately_shared_keys:
                # ADR002_ALLOW: sharing is deliberate, not a fallback.
                with contextlib.suppress(Exception):
                    setattr(result, k, v)
                continue
            if k == "latest_explanation":
                with contextlib.suppress(Exception):
                    setattr(result, k, None)
                continue

            try:
                setattr(result, k, copy.deepcopy(v, memo))
            except Exception as exc:  # ADR002_ALLOW: governed, visible fallback below.
                # Fallback-visibility policy (CONTRIBUTOR_INSTRUCTIONS.md Sec.
                # 5): a deepcopy failure must never be silently downgraded to
                # sharing the original object -- emit a UserWarning and an
                # INFO log so the reduced isolation guarantee is observable.
                message = (
                    f"CalibratedExplainer.__deepcopy__ could not deep-copy attribute "
                    f"'{k}' ({exc!r}); falling back to sharing the original object. "
                    "Mutating this attribute on the copy may affect the original."
                )
                warnings.warn(message, UserWarning, stacklevel=2)
                logging.getLogger(__name__).info(message)
                with contextlib.suppress(Exception):
                    setattr(result, k, v)

        return result

    def __getstate__(self):
        """Exclude runtime helpers and caches when pickling."""
        state = self.__dict__.copy()
        state["perf_cache"] = None
        state["_perf_parallel"] = None
        state["_lime_helper"] = None
        state["_shap_helper"] = None
        state["latest_explanation"] = None
        return state

    def __setstate__(self, state):
        """Restore state after pickling without restoring helpers."""
        self.__dict__.update(state)

    def require_plugin_manager(self) -> PluginManager:
        """Return the plugin manager or raise if the explainer is not initialized.

        Returns
        -------
        PluginManager
            The active plugin manager instance.

        Raises
        ------
        NotFittedError
            If the plugin manager is not initialized.
        """
        from ..utils.exceptions import NotFittedError

        manager = getattr(self, "_plugin_manager", None)
        if manager is None:
            raise NotFittedError(
                "PluginManager is not initialized. Instantiate CalibratedExplainer via __init__.",
                details={
                    "state": "uninitialized",
                    "reason": "plugin_manager_missing",
                    "required_method": "__init__",
                },
            )
        return manager

    def get_plugin_manager(self) -> PluginManager:
        """Return the active plugin manager, applying any derived defaults.

        Wrapper layers must not mutate plugin manager state directly. Any
        runtime-derived plugin preferences (for example, feature filter
        execution requirements) are enforced here so orchestration remains
        centralized in the explainer/manager layers.
        """
        manager = self.require_plugin_manager()
        self._enforce_feature_filter_plugin_preferences(manager)
        return manager

    def _enforce_feature_filter_plugin_preferences(self, manager: PluginManager) -> None:
        cfg = getattr(self, "_feature_filter_config", None)
        enabled = getattr(cfg, "enabled", False)
        if enabled is not True:
            return

        override_id = "core.explanation.factual.sequential"
        logger = logging.getLogger(__name__)

        try:
            chain = manager.explanation_plugin_fallbacks.get("factual", ())
        except Exception as exc:  # adr002_allow
            logger.warning(
                "Failed to read explanation plugin fallback chain; feature filter enforcement skipped: %s",
                exc,
                exc_info=True,
            )
            return

        if chain and chain[0] == override_id:
            return

        if not chain:
            try:
                manager.initialize_chains()
                chain = manager.explanation_plugin_fallbacks.get("factual", ())
            except Exception as exc:  # adr002_allow
                logger.warning(
                    "Failed to initialize plugin chains; feature filter enforcement skipped: %s",
                    exc,
                    exc_info=True,
                )
                return

            if chain and chain[0] == override_id:
                return

        previous = chain[0] if chain else None
        logger.warning(
            "Feature filter enabled; forcing factual explanation plugin to '%s' (was '%s')",
            override_id,
            previous,
            extra={"mode": "factual", "plugin_identifier": override_id},
        )

        try:
            manager.explanation_plugin_overrides["factual"] = override_id
            manager.clear_explanation_plugin_instances()
            manager.clear_explanation_plugin_identifiers()
            manager.initialize_chains()
        except Exception as exc:  # adr002_allow
            logger.warning(
                "Failed to enforce factual explanation plugin for feature filter: %s",
                exc,
                exc_info=True,
            )

    def _resolve_parallel_executor(self, explicit_executor: Any | None) -> Any | None:
        """Resolve the parallel executor honoring overrides and environment config."""
        return self.resolve_parallel_executor(explicit_executor)

    def resolve_parallel_executor(self, explicit_executor: Any | None) -> Any | None:
        """Resolve the parallel executor honoring overrides and environment config.

        Raises
        ------
        ConfigurationError
            If ``CE_PARALLEL`` enables parallel execution without naming an
            explicit strategy (ADR-004 forbids the removed ``auto`` strategy).
        """
        from ..parallel import ParallelConfig, ParallelExecutor
        from ..parallel.parallel import _require_explicit_strategy

        if explicit_executor is not None:
            return explicit_executor

        env_config = ParallelConfig.from_env()
        if env_config.enabled:
            _require_explicit_strategy(env_config, source="CE_PARALLEL")
            return ParallelExecutor(env_config)

        return None

    # ------------------------------------------------------------------
    # Parallel pool lifecycle helpers
    # ------------------------------------------------------------------
    def initialize_pool(self, n_workers: int | None = None, *, pool_at_init: bool = False) -> None:
        """Create a `ParallelExecutor` for this explainer.

        Parameters
        ----------
        n_workers: int | None
            Optional maximum worker count to enforce.
        pool_at_init: bool
            If True, enter the pool immediately so worker processes are
            spawned at initialization time (useful for warm-up and
            initializer-based harness installation).

        Notes
        -----
        The strategy comes from ``CE_PARALLEL``. When it names no strategy the
        pool runs ``"sequential"``; the removed ``"auto"`` strategy is never
        used (ADR-004).
        """
        from ..parallel import ParallelConfig, ParallelExecutor

        if getattr(self, "_perf_parallel", None) is not None:
            return

        cfg = ParallelConfig.from_env()
        cfg.enabled = True
        # ADR-004: never rely on the removed "auto" strategy. Without an explicit
        # CE_PARALLEL strategy the explainer-managed pool stays sequential.
        if cfg.strategy == "auto":
            cfg.strategy = "sequential"
        if n_workers is not None:
            cfg.max_workers = n_workers

        # If requested, set up a worker initializer that will receive a
        # compact explainer spec. Keep the spec deliberately small and
        # picklable.
        if pool_at_init:
            # ADR002_ALLOW: optional initializer wiring should not block.
            with contextlib.suppress(Exception):
                import calibrated_explanations.core.explain.parallel_runtime as pr_mod

                # Build a picklable compact spec containing only the data
                # required to rehydrate an explainer in worker processes.
                # Attempt to include a picklable learner payload. If the
                # learner is not picklable, fall back to omitting it so the
                # worker initializer must handle a missing learner case.
                learner_bytes = None
                try:
                    import pickle  # nosec B403

                    learner_bytes = pickle.dumps(getattr(self, "learner", None))
                except (
                    Exception
                ):  # ADR002_ALLOW: learner pickling best-effort fallback.  # pragma: no cover
                    learner_bytes = None

                spec = {
                    "learner_bytes": learner_bytes,
                    "x_cal": getattr(self, "x_cal", None),
                    "y_cal": getattr(self, "y_cal", None),
                    "mode": getattr(self, "mode", None),
                    "num_features": getattr(self, "num_features", None),
                    "bins": getattr(self, "bins", None),
                    "sample_percentiles": getattr(self, "sample_percentiles", None),
                }
                cfg.worker_initializer = pr_mod.worker_init_from_explainer_spec
                cfg.worker_init_args = (spec,)

        self._perf_parallel = ParallelExecutor(cfg)
        if pool_at_init:
            # Enter context to spawn worker pool now
            self._perf_parallel.__enter__()

    def close(self) -> None:
        """Reset runtime state, then shutdown any provisioned parallel pool."""
        self.reset()
        perf = getattr(self, "_perf_parallel", None)
        if perf is None:
            return
        try:
            perf.__exit__(None, None, None)
        finally:
            self._perf_parallel = None

    def reset(self) -> None:
        """Clear transient runtime state retained between explanation calls."""
        self.latest_explanation = None

        for helper_name in ("_lime_helper", "_shap_helper"):
            helper = getattr(self, helper_name, None)
            if helper is not None and hasattr(helper, "reset"):
                helper.reset()

        plugin_manager = getattr(self, "_plugin_manager", None)
        if plugin_manager is None:
            return

        with contextlib.suppress(Exception):
            plugin_manager.clear_explanation_plugin_instances()
        with contextlib.suppress(Exception):
            plugin_manager.clear_explanation_plugin_identifiers()
        with contextlib.suppress(Exception):
            plugin_manager.clear_bridge_monitors()
        contexts = getattr(plugin_manager, "explanation_contexts", None)
        if isinstance(contexts, dict):
            contexts.clear()

    def __enter__(self) -> "CalibratedExplainer":
        """Context manager entry; create and enter a worker pool."""
        self.initialize_pool(pool_at_init=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit; close any provisioned pool."""
        self.close()

    def infer_explanation_mode(self) -> str:
        """Infer the explanation mode from runtime state."""
        # Lazy import discretizers (deferred from module level)
        from ..utils import EntropyDiscretizer, RegressorDiscretizer

        # Check discretizer type to infer mode
        discretizer = self.discretizer if hasattr(self, "discretizer") else None
        if discretizer is not None and isinstance(
            discretizer, (EntropyDiscretizer, RegressorDiscretizer)
        ):
            return "alternative"
        # All other discretizers (Binary*, or None) indicate factual
        return "factual"

    # ===================================================================
    # Delegation methods for orchestrator operations
    # ===================================================================
    # These methods delegate to PluginManager and orchestrators.
    # PluginManager is the single source of truth for plugin defaults and chains.
    # Tests that call these directly MUST initialize PluginManager properly.

    @property
    def prediction_orchestrator(self) -> Any:
        """Return the PredictionOrchestrator provisioned by the PluginManager."""
        return self.require_plugin_manager().prediction_orchestrator

    @prediction_orchestrator.setter
    def prediction_orchestrator(self, value: Any) -> None:
        """Set the PredictionOrchestrator."""
        self.require_plugin_manager().prediction_orchestrator = value

    @prediction_orchestrator.deleter
    def prediction_orchestrator(self) -> None:
        """Delete the PredictionOrchestrator."""
        del self.require_plugin_manager().prediction_orchestrator

    @property
    def explanation_orchestrator(self) -> Any:
        """Return the ExplanationOrchestrator provisioned by the PluginManager."""
        return self.require_plugin_manager().explanation_orchestrator

    @explanation_orchestrator.setter
    def explanation_orchestrator(self, value: Any) -> None:
        """Set the ExplanationOrchestrator."""
        self.require_plugin_manager().explanation_orchestrator = value

    @explanation_orchestrator.deleter
    def explanation_orchestrator(self) -> None:
        """Delete the ExplanationOrchestrator."""
        del self.require_plugin_manager().explanation_orchestrator

    @property
    def reject_orchestrator(self) -> Any:
        """Return the RejectOrchestrator provisioned by the PluginManager."""
        return self.require_plugin_manager().reject_orchestrator

    @reject_orchestrator.setter
    def reject_orchestrator(self, value: Any) -> None:
        """Set the RejectOrchestrator."""
        self.require_plugin_manager().reject_orchestrator = value

    @reject_orchestrator.deleter
    def reject_orchestrator(self) -> None:
        """Delete the RejectOrchestrator."""
        del self.require_plugin_manager().reject_orchestrator

    def build_instance_telemetry_payload(self, explanations: Any) -> Dict[str, Any]:
        """Delegate to ExplanationOrchestrator."""
        return self.explanation_orchestrator.build_instance_telemetry_payload(explanations)

    # ===================================================================
    # Backward-compatibility properties for plugin state (via PluginManager)
    # ===================================================================
    # These properties delegate to PluginManager for backward compatibility
    # with code that accesses plugin state directly from explainer.

    @property
    def _interval_plugin_hints(self) -> Dict[str, Tuple[str, ...]]:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_plugin_hints

    @_interval_plugin_hints.setter
    def _interval_plugin_hints(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_plugin_hints = value

    @_interval_plugin_hints.deleter
    def _interval_plugin_hints(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.interval_plugin_hints

    @property
    def _interval_plugin_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_plugin_fallbacks

    @_interval_plugin_fallbacks.setter
    def _interval_plugin_fallbacks(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_plugin_fallbacks = value

    @_interval_plugin_fallbacks.deleter
    def _interval_plugin_fallbacks(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.interval_plugin_fallbacks

    @property
    def _interval_preferred_identifier(self) -> Dict[str, str | None]:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_preferred_identifier

    @_interval_preferred_identifier.setter
    def _interval_preferred_identifier(self, value: Dict[str, str | None]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_preferred_identifier = value

    @_interval_preferred_identifier.deleter
    def _interval_preferred_identifier(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.interval_preferred_identifier

    @property
    def _telemetry_interval_sources(self) -> Dict[str, str | None]:
        """Delegate to PluginManager."""
        return self.plugin_manager.telemetry_interval_sources

    @_telemetry_interval_sources.setter
    def _telemetry_interval_sources(self, value: Dict[str, str | None]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.telemetry_interval_sources = value

    @_telemetry_interval_sources.deleter
    def _telemetry_interval_sources(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.telemetry_interval_sources

    @property
    def _interval_plugin_identifiers(self) -> Dict[str, str | None]:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_plugin_identifiers

    @_interval_plugin_identifiers.setter
    def _interval_plugin_identifiers(self, value: Dict[str, str | None]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_plugin_identifiers = value

    @_interval_plugin_identifiers.deleter
    def _interval_plugin_identifiers(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.interval_plugin_identifiers

    @property
    def _interval_context_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_context_metadata

    @_interval_context_metadata.setter
    def _interval_context_metadata(self, value: Dict[str, Dict[str, Any]]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_context_metadata = value

    @_interval_context_metadata.deleter
    def _interval_context_metadata(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.interval_context_metadata

    @property
    def plot_plugin_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Return the plot plugin fallback configuration.

        Returns
        -------
        Dict[str, Tuple[str, ...]]
            Mapping of mode to fallback identifiers.
        """
        return self.plugin_manager.plot_plugin_fallbacks

    @plot_plugin_fallbacks.setter
    def plot_plugin_fallbacks(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Set the plot plugin fallback configuration."""
        self.plugin_manager.plot_plugin_fallbacks = value

    @property
    def _explanation_plugin_overrides(self) -> Dict[str, Any]:
        """Delegate to PluginManager."""
        return self.plugin_manager.explanation_plugin_overrides

    @_explanation_plugin_overrides.setter
    def _explanation_plugin_overrides(self, value: Dict[str, Any]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.explanation_plugin_overrides = value

    @property
    def _interval_plugin_override(self) -> Any:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_plugin_override

    @_interval_plugin_override.setter
    def _interval_plugin_override(self, value: Any) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_plugin_override = value

    @property
    def _fast_interval_plugin_override(self) -> Any:
        """Delegate to PluginManager."""
        return self.plugin_manager.fast_interval_plugin_override

    @_fast_interval_plugin_override.setter
    def _fast_interval_plugin_override(self, value: Any) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.fast_interval_plugin_override = value

    @property
    def _plot_style_override(self) -> Any:
        """Delegate to PluginManager."""
        return self.plugin_manager.plot_style_override

    @_plot_style_override.setter
    def _plot_style_override(self, value: Any) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.plot_style_override = value

    @property
    def _explanation_plugin_instances(self) -> Dict[str, Any]:
        """Delegate to PluginManager."""
        return self.plugin_manager.explanation_plugin_instances

    @_explanation_plugin_instances.setter
    def _explanation_plugin_instances(self, value: Dict[str, Any]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.explanation_plugin_instances = value

    # Public aliases to replace test access of private members (safe one-line delegations)
    @property
    def plugin_manager(self) -> PluginManager:
        """Public accessor for the active PluginManager."""
        return self.get_plugin_manager()

    @plugin_manager.setter
    def plugin_manager(self, value: Any) -> None:
        """Set the plugin manager for this explainer."""
        self._plugin_manager = value

    @plugin_manager.deleter
    def plugin_manager(self) -> None:
        """Delete the plugin manager."""
        if hasattr(self, "_plugin_manager"):
            del self._plugin_manager

    @property
    def preprocessor_metadata(self) -> Any:
        """Public alias for `_preprocessor_metadata`."""
        return self._preprocessor_metadata

    @preprocessor_metadata.setter
    def preprocessor_metadata(self, value: Any) -> None:
        self._preprocessor_metadata = value

    @property
    def perf_parallel(self) -> bool:
        """Public alias for `_perf_parallel`."""
        return self._perf_parallel

    @perf_parallel.setter
    def perf_parallel(self, value: bool) -> None:
        self._perf_parallel = value

    @property
    def initialized(self) -> bool:
        """Return True if the explainer is initialized."""
        return getattr(self, "_initialized", False)

    @initialized.setter
    def initialized(self, value: bool) -> None:
        """Set the initialization state of the explainer."""
        self._initialized = value

    @property
    def last_explanation_mode(self) -> str | None:
        """Return the mode of the last generated explanation."""
        return self._last_explanation_mode

    @last_explanation_mode.setter
    def last_explanation_mode(self, value: str | None) -> None:
        """Set the mode of the last generated explanation."""
        self._last_explanation_mode = value

    @property
    def feature_filter_per_instance_ignore(self) -> Any:
        """Return the per-instance feature filter ignore list."""
        return getattr(self, "_feature_filter_per_instance_ignore", None)

    @feature_filter_per_instance_ignore.setter
    def feature_filter_per_instance_ignore(self, value: Any) -> None:
        """Set the per-instance feature filter ignore list."""
        self._feature_filter_per_instance_ignore = value

    @feature_filter_per_instance_ignore.deleter
    def feature_filter_per_instance_ignore(self) -> None:
        """Delete the per-instance feature filter ignore list."""
        if hasattr(self, "_feature_filter_per_instance_ignore"):
            delattr(self, "_feature_filter_per_instance_ignore")

    @property
    def parallel_executor(self) -> Any:
        """Return the active parallel executor."""
        return getattr(self, "_perf_parallel", None)

    @parallel_executor.setter
    def parallel_executor(self, value: Any) -> None:
        """Set the active parallel executor."""
        self._perf_parallel = value

    @property
    def feature_filter_config(self) -> Any:
        """Return the feature filter configuration."""
        return getattr(self, "_feature_filter_config", None)

    @feature_filter_config.setter
    def feature_filter_config(self, value: Any) -> None:
        """Set the feature filter configuration."""
        self._feature_filter_config = value

    @property
    def predict_bridge(self) -> Any:
        """Return the prediction bridge."""
        return getattr(self, "_predict_bridge", None)

    @predict_bridge.setter
    def predict_bridge(self, value: Any) -> None:
        """Set the prediction bridge."""
        self._predict_bridge = value

    @property
    def categorical_value_counts_cache(self) -> Any:
        """Return the categorical value counts cache."""
        return getattr(self, "_categorical_value_counts_cache", None)

    @categorical_value_counts_cache.setter
    def categorical_value_counts_cache(self, value: Any) -> None:
        """Set the categorical value counts cache."""
        self._categorical_value_counts_cache = value

    @property
    def numeric_sorted_cache(self) -> Any:
        """Return the numeric sorted cache."""
        return getattr(self, "_numeric_sorted_cache", None)

    @numeric_sorted_cache.setter
    def numeric_sorted_cache(self, value: Any) -> None:
        """Set the numeric sorted cache."""
        self._numeric_sorted_cache = value

    @property
    def calibration_summary_shape(self) -> Any:
        """Return the calibration summary shape."""
        return getattr(self, "_calibration_summary_shape", None)

    @calibration_summary_shape.setter
    def calibration_summary_shape(self, value: Any) -> None:
        """Set the calibration summary shape."""
        self._calibration_summary_shape = value

    def enable_fast_mode(self) -> None:
        """Enable fast explanation mode.

        This initializes the interval learner for fast explanations if not already done.
        """
        if not self.is_fast():
            try:
                self._fast = True
                self._initialize_interval_learner_for_fast_explainer()
            except Exception:  # adr002_allow
                self._fast = False
                raise

    @property
    def _bridge_monitors(self) -> Dict[str, Any]:
        """Expose bridge monitor registry managed by PluginManager."""
        return self.require_plugin_manager().bridge_monitors

    @property
    def _pyproject_explanations(self) -> Dict[str, Any] | None:
        """Delegate to PluginManager."""
        return self.plugin_manager.pyproject_explanations

    @_pyproject_explanations.setter
    def _pyproject_explanations(self, value: Dict[str, Any] | None) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.pyproject_explanations = value

    @property
    def _pyproject_intervals(self) -> Dict[str, Any] | None:
        """Delegate to PluginManager."""
        return self.plugin_manager.pyproject_intervals

    @_pyproject_intervals.setter
    def _pyproject_intervals(self, value: Dict[str, Any] | None) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.pyproject_intervals = value

    @property
    def _pyproject_plots(self) -> Dict[str, Any] | None:
        """Delegate to PluginManager."""
        return self.plugin_manager.pyproject_plots

    @_pyproject_plots.setter
    def _pyproject_plots(self, value: Dict[str, Any] | None) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.pyproject_plots = value

    @property
    def runtime_telemetry(self) -> Mapping[str, Any]:
        """Return the most recent telemetry payload reported by the explainer."""
        return dict(self.plugin_manager.last_telemetry)

    @property
    def preprocessor_metadata(self) -> Dict[str, Any] | None:
        """Return the telemetry-safe preprocessing snapshot if available."""
        if self._preprocessor_metadata is None:
            return None
        return dict(self._preprocessor_metadata)

    def set_preprocessor_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        """Update the stored preprocessing metadata snapshot."""
        if metadata is None:
            self._preprocessor_metadata = None
        else:
            self._preprocessor_metadata = dict(metadata)

    @property
    def x_cal(self):
        """Get the calibration input data.

        Returns
        -------
        array-like
            The calibration input data.
        """
        from ..calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        return CalibrationState.get_x_cal(self)

    @x_cal.setter
    def x_cal(self, value):
        """Set the calibration input data.

        Parameters
        ----------
        value : array-like of shape (n_samples, n_features)
            The new calibration input data.
        """
        from ..calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        CalibrationState.set_x_cal(self, value)

    @property
    def y_cal(self):
        """Get the calibration target data.

        Returns
        -------
        array-like
            The calibration target data.
        """
        from ..calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        return CalibrationState.get_y_cal(self)

    @y_cal.setter
    def y_cal(self, value):
        """Set the calibration target data.

        Parameters
        ----------
        value : array-like of shape (n_samples,)
            The new calibration target data.
        """
        from ..calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        CalibrationState.set_y_cal(self, value)

    def append_cal(self, x, y):
        """Append new calibration data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The new calibration input data to append.
        y : array-like of shape (n_samples,)
            The new calibration target data to append.
        """
        from ..calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        CalibrationState.append_calibration(self, x, y)

    def _invalidate_calibration_summaries(self) -> None:
        """Drop cached calibration summaries used during explanation.

        Delegates to the calibration.summaries module which manages the cache.
        """
        from ..calibration.summaries import (  # pylint: disable=import-outside-toplevel
            invalidate_calibration_summaries as _invalidate,
        )

        _invalidate(self)

    def get_calibration_summaries(
        self, x_cal_np: Optional[np.ndarray] = None
    ) -> Tuple[Dict[int, Dict[Any, int]], Dict[int, np.ndarray]]:
        """Return cached categorical counts and sorted numeric calibration values.

        Delegates to the calibration.summaries module which manages caching of
        statistical summaries used during explanation generation.
        """
        from ..calibration.summaries import (  # pylint: disable=import-outside-toplevel
            get_calibration_summaries as _get,
        )

        return _get(self, x_cal_np)

    @property
    def num_features(self):
        """Get the number of features in the calibration data.

        Returns
        -------
        int
            The number of features in the calibration data. For dictionary input,
            returns the number of keys. For array input, returns the number of columns.
        """
        return (
            len(self._X_cal[0].keys())
            if isinstance(self._X_cal[0], dict)
            else len(self._X_cal[0, :])
        )

    @property
    def feature_names(self):
        """Get the feature names.

        Returns
        -------
        list
            The list of feature names. If no feature names were provided during initialization,
            returns None.
        """
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        """Set the feature names.

        Parameters
        ----------
        value : list
            The list of feature names.
        """
        self._feature_names = list(value) if value is not None else None

    @property
    def interval_learner(self) -> Any:
        """Access the interval learner managed by the prediction orchestrator.

        Returns
        -------
        Any
            The interval calibrator (e.g., VennAbers, IntervalRegressor, or list for fast mode).

        Notes
        -----
        This is a backward-compatible property that delegates to the interval registry
        managed by the PredictionOrchestrator. See ADR-001.
        """
        return self.prediction_orchestrator.interval_registry.interval_learner

    @interval_learner.setter
    def interval_learner(self, value: Any) -> None:
        """Set the interval learner through the prediction orchestrator's registry.

        Parameters
        ----------
        value : Any
            The interval calibrator to set (e.g., VennAbers, IntervalRegressor).

        Notes
        -----
        This is a backward-compatible setter that delegates to the interval registry
        managed by the PredictionOrchestrator.
        """
        self.prediction_orchestrator.interval_registry.interval_learner = value

    def _get_sigma_test(self, x: np.ndarray) -> np.ndarray:
        """Return the difficulty (sigma) of the test instances.

        Parameters
        ----------
        x : np.ndarray
            Test instances for which to estimate difficulty.

        Returns
        -------
        np.ndarray
            Difficulty estimates (sigma values) for each test instance.

        Notes
        -----
        This is a backward-compatible method that delegates to the interval registry
        managed by the PredictionOrchestrator. See ADR-001.
        """
        return self.prediction_orchestrator.interval_registry.get_sigma_test(x)

    def get_sigma_test(self, x: np.ndarray) -> np.ndarray:
        """Return the difficulty (sigma) of the test instances.

        Parameters
        ----------
        x : np.ndarray
            Test instances for which to estimate difficulty.

        Returns
        -------
        np.ndarray
            Difficulty estimates (sigma values) for each test instance.
        """
        return self._get_sigma_test(x)

    def _initialize_interval_learner_for_fast_explainer(self) -> None:  # noqa: N802
        """Backward-compatible wrapper for fast-mode interval learner initialization.

        Notes
        -----
        This method delegates to the interval registry. It is kept for backward
        compatibility with the external fast_explanations plugin and other
        production code that calls this private method.

        See ADR-001.
        """
        self.prediction_orchestrator.interval_registry.initialize_for_fast_explainer()

    def reinitialize(self, learner, xs=None, ys=None, bins=None):
        """Reinitialize the explainer with a new learner.

        This is useful when the learner is updated or retrained and the explainer needs to be reinitialized.

        Parameters
        ----------
        learner : predictive learner
            A predictive learner that can be used to predict the target variable. The learner must be fitted and have a predict_proba method (for classification) or a predict method (for regression).
        xs : array-like, optional
            New calibration input data to append
        ys : array-like, optional
            New calibration target data to append

        Returns
        -------
        :class:`.CalibratedExplainer`
            A :class:`.CalibratedExplainer` object that can be used to explain predictions from a predictive learner.
        """
        self._initialized = False
        check_is_fitted(learner)
        self.learner = learner
        if xs is not None and ys is not None:
            self.append_cal(xs, ys)
            if bins is not None:
                if self.bins is None:
                    raise ValidationError("Cannot mix calibration instances with and without bins.")
                if len(bins) != len(ys):
                    raise DataShapeError(
                        "The length of bins must match the number of added instances."
                    )
                self.bins = np.concatenate((self.bins, bins)) if self.bins is not None else bins
            # update interval learner via helper
            from ..calibration.interval_learner import update_interval_learner as _upd_il

            _upd_il(self, xs, ys, bins=bins)
        else:
            from ..calibration.interval_learner import initialize_interval_learner as _init_il

            _init_il(self)
        self._initialized = True

    def __repr__(self):
        """Return the string representation of the CalibratedExplainer."""
        # pylint: disable=line-too-long
        disp_str = f"CalibratedExplainer(mode={self.mode}{', conditional=True' if self.bins is not None else ''}{f', discretizer={self.discretizer}' if self.discretizer is not None else ''}, learner={self.learner}{f', difficulty_estimator={self.difficulty_estimator})' if self.mode == 'regression' else ')'}"
        if self.verbose:
            disp_str += f"\n\tinit_time={self.init_time}"
            if self.latest_explanation is not None:
                disp_str += f"\n\ttotal_explain_time={self.latest_explanation.total_explain_time}"
            disp_str += f"\n\tsample_percentiles={self.sample_percentiles}\
                        \n\tseed={self.seed}\
                        \n\tverbose={self.verbose}"
            if self.feature_names is not None:
                disp_str += f"\n\tfeature_names={self.feature_names}"
            if self.categorical_features is not None:
                disp_str += f"\n\tcategorical_features={self.categorical_features}"
            if self.categorical_labels is not None:
                disp_str += f"\n\tcategorical_labels={self.categorical_labels}"
            if self.class_labels is not None:
                disp_str += f"\n\tclass_labels={self.class_labels}"
        return disp_str

    def obtain_interval_calibrator(
        self,
        *,
        fast: bool,
        metadata: Mapping[str, Any],
    ) -> Tuple[Any, str | None]:
        """Return the interval calibrator from the prediction orchestrator."""
        return self.prediction_orchestrator.obtain_interval_calibrator(fast=fast, metadata=metadata)

    def explain_factual(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        *,
        guarded_options=None,
        _use_plugin: bool = True,
        **kwargs,
    ) -> CalibratedExplanations:
        """Create a :class:`.CalibratedExplanations` object for the test data with the discretizer automatically assigned for factual explanations.

        This is a thin delegator that sets up the appropriate discretizer and delegates to the orchestrator.

        Parameters
        ----------
        x : array-like
            A set with n_samples of test objects to predict.
        threshold : float, int or array-like, default=None
            Target value for probabilistic regression; the explainer returns the calibrated
            probability P(y ≤ threshold) for each instance. Ignored for classification.
            Mutually exclusive with ``confidence_level`` (per ``EXCLUSIVE_PARAM_GROUPS``).
            See ``docs/foundations/concepts/parameter-reference.md`` for the full
            disambiguation of ``threshold``, ``confidence_level``, ``reject_confidence``, and
            ``GuardedOptions.confidence``.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        guarded_options : GuardedOptions or None, default=None
            **[EXPERIMENTAL]** Per-call tuning for the KNN-based in-distribution guard
            (ADR-038). When provided, the guarded path is activated automatically.
            Use :class:`~calibrated_explanations.GuardedOptions` to bundle guard tuning
            parameters (``confidence``, ``n_neighbors``, ``normalize``, ``merge_adjacent``,
            ``verbose``).
        reject_policy : RejectPolicySpec | None, default=None
            When non-``None``, activates reject orchestration.  Pass a
            :class:`.RejectPolicySpec` constructed via
            ``RejectPolicySpec.flag()``, ``RejectPolicySpec.only_accepted()``,
            or ``RejectPolicySpec.only_rejected()``.  When active, the return
            type is :class:`~calibrated_explanations.explanations.reject.RejectCalibratedExplanations`
            rather than :class:`.CalibratedExplanations`.
        multi_labels_enabled : bool, default=False
            **[EXPERIMENTAL]** When ``True``, generates one explanation per class for
            multi-class problems (3+ classes). Passed via ``**kwargs``. This parameter
            surface is under active development; its signature will be promoted to an
            explicit typed argument before graduation out of experimental status.
            Unknown additional kwargs are forwarded to the explanation plugin and
            silently ignored if not recognised (ADR-038 §3 experimental exception).
        interval_summary : str or None, default=None
            **[EXPERIMENTAL]** Controls the interval summary mode forwarded to the
            explanation plugin. Passed via ``**kwargs``. Subject to the same
            experimental-graduation constraint as ``multi_labels_enabled``.
        **kwargs : dict
            **[EXPERIMENTAL]** Additional keyword arguments forwarded to the explanation plugin.
            Silently ignored if not recognised (ADR-038 §3 experimental exception).

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A `CalibratedExplanations` containing one :class:`.FactualExplanation` for each instance.
            When ``guarded_options`` is non-``None``, per-instance explanations are
            :class:`~calibrated_explanations.explanations.guarded_explanation.GuardedFactualExplanation`.
            When ``reject_policy`` is non-``None``, returns
            :class:`~calibrated_explanations.explanations.reject.RejectCalibratedExplanations`.
        """
        reject_removed_aliases(kwargs)
        reject_removed_guarded_kwargs(kwargs)
        reject_removed_reject_kwargs(kwargs)
        reject_removed_normalization_kwarg(kwargs)
        reject_cross_surface_kwargs(
            kwargs,
            allowed=_EXPLAIN_KWARGS,
            closed_surface_names=_CLOSED_SURFACE_KWARGS,
            surface="CalibratedExplainer.explain_factual",
        )
        _log_forwarded_explain_kwargs(
            logging.getLogger(__name__),
            "CalibratedExplainer.explain_factual",
            kwargs,
            allowed=_EXPLAIN_KWARGS,
        )
        if "multi_labels_enabled" in kwargs:
            kwargs["multi_labels_enabled"] = validate_bool_parameter(
                kwargs["multi_labels_enabled"],
                param="multi_labels_enabled",
            )
        if features_to_ignore is not None:
            features_to_ignore = validate_features_to_ignore(
                features_to_ignore,
                n_features=int(self.num_features),
            )
        if threshold is not None and "regression" in self.mode:
            assert_threshold(threshold, x)
        elif "regression" in self.mode:
            low_high_percentiles = validate_low_high_percentiles(low_high_percentiles)
        bins = resolve_conditional_bins(x, bins, calibration_bins=self.bins)
        if guarded_options is not None:
            if not _use_plugin and kwargs.get("verbose", False):
                warnings.warn(
                    "_use_plugin has no effect on guarded explanation methods",
                    UserWarning,
                    stacklevel=2,
                )
            ctx = (
                self._perf_parallel if self._perf_parallel is not None else contextlib.nullcontext()
            )
            with ctx:
                reject_policy = kwargs.pop("reject_policy", None)
                return self.explanation_orchestrator.invoke_guarded_factual(
                    x=x,
                    threshold=threshold,
                    low_high_percentiles=low_high_percentiles,
                    bins=bins,
                    features_to_ignore=features_to_ignore,
                    reject_policy=reject_policy,
                    guarded_options=guarded_options,
                    **kwargs,
                )
        # Thin delegator that sets discretizer and delegates to orchestrator
        discretizer = "binaryRegressor" if "regression" in self.mode else "binaryEntropy"
        ctx = self._perf_parallel if self._perf_parallel is not None else contextlib.nullcontext()
        with ctx:
            reject_policy = kwargs.pop("reject_policy", None)
            invoke_kwargs = {
                "x": x,
                "threshold": threshold,
                "low_high_percentiles": low_high_percentiles,
                "bins": bins,
                "features_to_ignore": features_to_ignore,
                "discretizer": discretizer,
                "_use_plugin": _use_plugin,
                **kwargs,
            }
            if reject_policy is not None:
                invoke_kwargs["reject_policy"] = reject_policy
            return self.explanation_orchestrator.invoke_factual(**invoke_kwargs)

    def explore_alternatives(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        *,
        guarded_options=None,
        _use_plugin: bool = True,
        **kwargs,
    ) -> AlternativeExplanations:
        """Create a :class:`.AlternativeExplanations` object for the test data with the discretizer automatically assigned for alternative explanations.

        This is a thin delegator that sets up the appropriate discretizer and delegates to the orchestrator.

        Parameters
        ----------
        x : array-like
            A set with n_samples of test objects to predict.
        threshold : float, int or array-like, default=None
            Target value for probabilistic regression; the explainer returns the calibrated
            probability P(y ≤ threshold) for each instance. Ignored for classification.
            Mutually exclusive with ``confidence_level`` (per ``EXCLUSIVE_PARAM_GROUPS``).
            See ``docs/foundations/concepts/parameter-reference.md`` for the full
            disambiguation of ``threshold``, ``confidence_level``, ``reject_confidence``, and
            ``GuardedOptions.confidence``.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        guarded_options : GuardedOptions or None, default=None
            **[EXPERIMENTAL]** Per-call tuning for the KNN-based in-distribution guard
            (ADR-038). When provided, the guarded path is activated automatically.
            Use :class:`~calibrated_explanations.GuardedOptions` to bundle guard tuning
            parameters.
        reject_policy : RejectPolicySpec | None, default=None
            When non-``None``, activates reject orchestration.  Pass a
            :class:`.RejectPolicySpec` constructed via
            ``RejectPolicySpec.flag()``, ``RejectPolicySpec.only_accepted()``,
            or ``RejectPolicySpec.only_rejected()``.  When active, the return
            type is :class:`~calibrated_explanations.explanations.reject.RejectAlternativeExplanations`
            rather than :class:`.AlternativeExplanations`.
        multi_labels_enabled : bool, default=False
            **[EXPERIMENTAL]** When ``True``, generates one explanation per class for
            multi-class problems (3+ classes). Passed via ``**kwargs``. This parameter
            surface is under active development; its signature will be promoted to an
            explicit typed argument before graduation out of experimental status.
            Unknown additional kwargs are forwarded to the explanation plugin and
            silently ignored if not recognised (ADR-038 §3 experimental exception).
        interval_summary : str or None, default=None
            **[EXPERIMENTAL]** Controls the interval summary mode forwarded to the
            explanation plugin. Passed via ``**kwargs``. Subject to the same
            experimental-graduation constraint as ``multi_labels_enabled``.
        **kwargs : dict
            **[EXPERIMENTAL]** Additional keyword arguments forwarded to the explanation plugin.
            Silently ignored if not recognised (ADR-038 §3 experimental exception).

        Returns
        -------
        AlternativeExplanations : :class:`.AlternativeExplanations`
            When ``reject_policy`` is non-``None``, returns
            :class:`~calibrated_explanations.explanations.reject.RejectAlternativeExplanations`.

        Notes
        -----
        The `explore_alternatives` will eventually be used instead of the `explain_counterfactual` method.
        When ``guarded_options`` is non-``None``, per-instance explanations are
        :class:`~calibrated_explanations.explanations.guarded_explanation.GuardedAlternativeExplanation`.
        """
        reject_removed_aliases(kwargs)
        reject_removed_guarded_kwargs(kwargs)
        reject_removed_reject_kwargs(kwargs)
        reject_removed_normalization_kwarg(kwargs)
        reject_cross_surface_kwargs(
            kwargs,
            allowed=_EXPLAIN_KWARGS,
            closed_surface_names=_CLOSED_SURFACE_KWARGS,
            surface="CalibratedExplainer.explore_alternatives",
        )
        _log_forwarded_explain_kwargs(
            logging.getLogger(__name__),
            "CalibratedExplainer.explore_alternatives",
            kwargs,
            allowed=_EXPLAIN_KWARGS,
        )
        if "multi_labels_enabled" in kwargs:
            kwargs["multi_labels_enabled"] = validate_bool_parameter(
                kwargs["multi_labels_enabled"],
                param="multi_labels_enabled",
            )
        if features_to_ignore is not None:
            features_to_ignore = validate_features_to_ignore(
                features_to_ignore,
                n_features=int(self.num_features),
            )
        if threshold is not None and "regression" in self.mode:
            assert_threshold(threshold, x)
        elif "regression" in self.mode:
            low_high_percentiles = validate_low_high_percentiles(low_high_percentiles)
        bins = resolve_conditional_bins(x, bins, calibration_bins=self.bins)
        if guarded_options is not None:
            if not _use_plugin and kwargs.get("verbose", False):
                warnings.warn(
                    "_use_plugin has no effect on guarded explanation methods",
                    UserWarning,
                    stacklevel=2,
                )
            ctx = (
                self._perf_parallel if self._perf_parallel is not None else contextlib.nullcontext()
            )
            with ctx:
                reject_policy = kwargs.pop("reject_policy", None)
                return self.explanation_orchestrator.invoke_guarded_alternative(
                    x=x,
                    threshold=threshold,
                    low_high_percentiles=low_high_percentiles,
                    bins=bins,
                    features_to_ignore=features_to_ignore,
                    reject_policy=reject_policy,
                    guarded_options=guarded_options,
                    **kwargs,
                )  # type: ignore[return-value]
        # Thin delegator that sets discretizer and delegates to orchestrator
        discretizer = "regressor" if "regression" in self.mode else "entropy"
        ctx = self._perf_parallel if self._perf_parallel is not None else contextlib.nullcontext()
        with ctx:
            reject_policy = kwargs.pop("reject_policy", None)
            invoke_kwargs = {
                "x": x,
                "threshold": threshold,
                "low_high_percentiles": low_high_percentiles,
                "bins": bins,
                "features_to_ignore": features_to_ignore,
                "discretizer": discretizer,
                "_use_plugin": _use_plugin,
                **kwargs,
            }
            if reject_policy is not None:
                invoke_kwargs["reject_policy"] = reject_policy
            return self.explanation_orchestrator.invoke_alternative(**invoke_kwargs)  # type: ignore[return-value]

    def __call__(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        *,
        reject_policy: Any | None = None,
        _use_plugin: bool = True,
        _skip_instance_parallel: bool = False,
    ) -> CalibratedExplanations:
        """Call self as a function to create a :class:`.CalibratedExplanations` object for the test data with the already assigned discretizer.

        Since v0.4.0, this method is equivalent to the `_explain` method.
        """
        call_kwargs: dict[str, Any] = {
            "_use_plugin": _use_plugin,
            "_skip_instance_parallel": _skip_instance_parallel,
        }
        if reject_policy is not None:
            call_kwargs["reject_policy"] = reject_policy

        return self._explain(
            x,
            threshold,
            low_high_percentiles,
            bins,
            features_to_ignore,
            **call_kwargs,
        )

    def _explain(self, *args, **kwargs) -> CalibratedExplanations:
        """Generate explanations for test instances by analyzing feature effects.

        This is an internal orchestration primitive that delegates to the explanation orchestrator.
        It is NOT part of the public API and should not be called directly.

        This method:
        1. Makes predictions on original test instances
        2. Creates perturbed versions by varying feature values
        3. Analyzes how predictions change with feature perturbations
        4. Generates feature importance weights and prediction intervals

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A :class:`.CalibratedExplanations` containing one :class:`.CalibratedExplanation` for each instance.

        See Also
        --------
        :meth:`.CalibratedExplainer.explain_factual` : Refer to the documentation for `explain_factual` for more details.
        :meth:`.CalibratedExplainer.explore_alternatives` : Refer to the documentation for `explore_alternatives` for more details.
        """
        # Delegate the args to the actual implementation
        return self._explain_impl(*args, **kwargs)

    def _explain_impl(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        *,
        reject_policy: Any | None = None,
        _use_plugin: bool = True,
        _skip_instance_parallel: bool = False,
    ) -> CalibratedExplanations:
        bins = resolve_conditional_bins(x, bins, calibration_bins=self.bins)
        # Thin delegator to orchestrator
        if _use_plugin:
            mode = self.infer_explanation_mode()
            invoke_kwargs: dict[str, Any] = {
                "extras": {"mode": mode, "_skip_instance_parallel": _skip_instance_parallel}
            }
            if reject_policy is not None:
                invoke_kwargs["reject_policy"] = reject_policy
            return self.explanation_orchestrator.invoke(
                mode,
                x,
                threshold,
                low_high_percentiles,
                bins,
                features_to_ignore,
                **invoke_kwargs,
            )

        # Legacy path for backward compatibility and testing
        from .explain import legacy_explain  # pylint: disable=import-outside-toplevel

        return legacy_explain(
            self,
            x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            bins=bins,
            features_to_ignore=features_to_ignore,
        )

    # NOTE: Instance- and feature-parallel helpers have been moved into the
    # plugin-based implementation under `core.explain.*`. The legacy helper
    # methods were intentionally removed to centralize parallel execution in
    # the plugin modules. Tests should exercise the plugin classes
    # (e.g. InstanceParallelExplainExecutor, FeatureParallelExplainExecutor,
    # SequentialExplainExecutor) rather than calling these private helpers.

    # NOTE: merge_feature_result functionality has been moved to
    # `calibrated_explanations.core.explain._helpers.merge_feature_result`.
    # Plugins and explain code should call that free-function directly.

    # NOTE: Thin wrapper methods (_slice_threshold, _slice_bins, _validate_and_prepare_input,
    # _initialize_explanation, _compute_weight_delta, _discretize) have been removed.
    # Callers should import these directly from core.explain submodules:
    # - core.explain._helpers: slice_threshold, slice_bins, validate_and_prepare_input
    # - core.explain._computation: initialize_explanation, discretize
    # - core.explain._helpers: compute_weight_delta

    def explain_fast(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        *,
        reject_policy: Any | None = None,
        _use_plugin: bool = True,
    ) -> CalibratedExplanations:
        """Create a :class:`.CalibratedExplanations` object for the test data.

        Parameters
        ----------
        x : array-like
            A set with n_samples of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories

        Raises
        ------
        ConfigurationError
            If plugin resolution, initialization, or invocation fails for the
            fast-explanation plugin (for example, an unsupported model, a
            feature-count mismatch, an invalid ``threshold`` value, or an
            invalid batch returned by the plugin).

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A `CalibratedExplanations` containing one :class:`.FastExplanation` for each instance.
        """
        bins = resolve_conditional_bins(x, bins, calibration_bins=self.bins)
        if _use_plugin:
            return self.explanation_orchestrator.invoke(
                "fast",
                x,
                threshold,
                low_high_percentiles,
                bins,
                tuple(self.features_to_ignore),
                extras={"mode": "fast"},
                reject_policy=reject_policy,
            )

        # Delegate to external plugin pipeline for non-plugin path
        # pylint: disable-next=import-outside-toplevel
        from pathlib import Path

        # Ensure the repository root is in the path
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from external_plugins.fast_explanations.pipeline import FastExplanationPipeline

        pipeline = FastExplanationPipeline(self)
        return pipeline.explain(x, threshold, low_high_percentiles, bins)

    # feature-merge and feature-parallel logic moved to plugin helpers

    def is_multiclass(self) -> bool:
        """Test if it is a multiclass problem.

        Returns
        -------
        bool
            True if multiclass (num_classes > 2).
        """
        return self.num_classes > 2

    def is_fast(self) -> bool:
        """Test if the explainer uses fast mode.

        Returns
        -------
        bool
            True if fast mode is enabled.
        """
        return self._fast

    def is_mondrian(self) -> bool:
        """Test if Mondrian (per-bin) calibration is enabled.

        Returns
        -------
        bool
            True if bins are configured, indicating Mondrian calibration.
        """
        return self.bins is not None

    def discretize(self, data: np.ndarray) -> np.ndarray:
        """Apply the discretizer to input data.

        Parameters
        ----------
        data : np.ndarray
            The data to discretize.

        Returns
        -------
        np.ndarray
            The discretized data.
        """
        from .explain import discretize as _discretize_func  # pylint: disable=import-outside-toplevel

        return _discretize_func(self, data)

    def rule_boundaries(self, instances, perturbed_instances=None):
        """Extract the rule boundaries for a set of instances.

        Parameters
        ----------
        instances : array-like
            The instances to extract boundaries for.
        perturbed_instances : array-like, optional
            Discretized versions of instances. Defaults to None.

        Returns
        -------
        array-like
            Min and max values for each feature for each instance.
        """
        from .explain import rule_boundaries as _rule_boundaries  # pylint: disable=import-outside-toplevel

        return _rule_boundaries(self, instances, perturbed_instances)

    def set_difficulty_estimator(self, difficulty_estimator, initialize=True) -> None:
        """Assign or update the difficulty estimator.

        If initialized to a difficulty estimator, the explainer can be used to reject explanations that are deemed too difficult.

        Parameters
        ----------
        difficulty_estimator : :class:`crepes.extras.DifficultyEstimator` or None):
            A :class:`crepes.extras.DifficultyEstimator` object from the crepes package. To remove the :class:`crepes.extras.DifficultyEstimator`, set to None.
        initialize (bool, optional):
            If true, then the interval learner is initialized once done. Defaults to True.
        """
        from .difficulty_estimator_helpers import (  # pylint: disable=import-outside-toplevel
            validate_difficulty_estimator,
        )

        validate_difficulty_estimator(difficulty_estimator)
        if initialize:
            self._initialized = False
        self.difficulty_estimator = difficulty_estimator

        # Invalidate cached interval plugin metadata.
        # Interval resolution persists context metadata (including a cached calibrator)
        # across invocations for performance. When the difficulty estimator changes,
        # we must drop that cache so the regression backend (IntervalRegressor)
        # re-fits crepes' ConformalPredictiveSystem with the updated `sigmas`.
        plugin_manager = getattr(self, "_plugin_manager", None)
        if initialize and plugin_manager is not None:
            meta = getattr(plugin_manager, "interval_context_metadata", None)
            if isinstance(meta, dict):
                for key in ("default", "fast"):
                    bucket = meta.get(key)
                    if isinstance(bucket, dict):
                        bucket.pop("calibrator", None)
                        bucket.pop("fast_calibrators", None)
                        bucket.pop("existing_fast_calibrators", None)
                        bucket.pop("difficulty_estimator", None)

        # Clear the active interval learner only when reinitializing. With
        # initialize=False, callers intentionally update metadata without
        # changing calibrated prediction internals.
        orchestrator_ready = (
            plugin_manager is not None
            and getattr(plugin_manager, "_prediction_orchestrator", None) is not None
        )
        if initialize and orchestrator_ready:
            self.interval_learner = None
        if initialize:
            self.prediction_orchestrator.interval_registry.initialize()  # type: ignore[attr-defined]

    def set_mode(self, mode, initialize=True) -> None:
        """Assign the mode of the explainer. The mode can be either 'classification' or 'regression'.

        Parameters
        ----------
            mode (str): The mode can be either 'classification' or 'regression'.
            initialize (bool, optional): If true, then the interval learner is initialized once done. Defaults to True.

        Raises
        ------
            ValidationError: If ``mode`` is not 'classification' or 'regression'.
        """
        mode = normalize_mode(mode)
        self._initialized = False
        if mode == "classification":
            # assert 'predict_proba' in dir(self.learner), "The learner must have a predict_proba method."
            self.num_classes = len(np.unique(self.y_cal))
        elif mode == "regression":
            # assert 'predict' in dir(self.learner), "The learner must have a predict method."
            self.num_classes = 0
        self.mode = mode
        if initialize:
            self.prediction_orchestrator.interval_registry.initialize()  # type: ignore[attr-defined]

    # pylint: disable=too-many-branches
    def set_discretizer(
        self,
        discretizer,
        x_cal=None,
        y_cal=None,
        features_to_ignore=None,
        *,
        condition_source: Optional[str] = None,
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
        """
        self.explanation_orchestrator.set_discretizer(
            discretizer,
            x_cal=x_cal,
            y_cal=y_cal,
            features_to_ignore=features_to_ignore,
            condition_source=condition_source,
        )

    # pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
    def predict(self, x, uq_interval=False, calibrated=True, **kwargs):
        """Generate predictions for the test data.

        Parameters
        ----------
        x : array-like
            The test data.
        uq_interval : bool, default=False
            Whether to return uncertainty intervals.
        calibrated : bool, default=True
            If True, the calibrator is used for prediction. If False, the underlying learner is used for prediction.
        **kwargs : Various types, optional
            Additional parameters to customize the explanation process. Supported parameters include:

            - threshold : float, int, or array-like of shape (n_samples,), optional, default=None
                Specifies the threshold for probabilistic regression. Returns calibrated probabilities
                P(y <= threshold) for regression tasks. Classification calls with this
                parameter raise ``ValidationError``.

            - low_high_percentiles : tuple of two floats, optional, default=(5, 95)
                The lower and upper percentiles used to calculate the prediction interval for regression tasks.
                Determines the breadth of the interval based on the distribution of the predictions.
                This parameter is only used for regression tasks without ``threshold=``.

        Raises
        ------
        NotFittedError
            If the explainer has not been fitted/calibrated prior to calling
            ``predict``.

        ConfigurationError
            If unsupported, removed, or conflicting keyword arguments are
            supplied.

        ValidationError
            If ``threshold`` or ``low_high_percentiles`` is invalid for the
            configured mode (for example, a classification call with
            ``threshold=``, or a threshold whose length does not match the
            number of instances in ``x``).

        Returns
        -------
        calibrated_prediction : float or array-like, or str
            The calibrated prediction. For regression tasks without threshold, this is the median of the
            conformal predictive system. For probabilistic regression (with threshold), this is a probability
            P(y <= threshold). For classification tasks, it is the class label with the highest calibrated probability.
        interval : tuple of floats, optional
            A tuple (low, high) representing the lower and upper bounds of the uncertainty interval. This is returned only if ``uq_interval=True``.

        Examples
        --------
        For a prediction without prediction intervals:

        .. code-block:: python

            w.predict(x)

        For a prediction with uncertainty quantification intervals:

        .. code-block:: python

            w.predict(x, uq_interval=True)

        Notes
        -----
        Classification calls with `threshold=` raise `ValidationError`. `low_high_percentiles`
        is only used for regression tasks without `threshold=`.
        """
        # strip plotting-only keys that callers may pass (plot()'s kwargs flow into
        # predict() via plotting.plot_global(); see ADR-038 5C)
        kwargs.pop("show", None)
        kwargs.pop("style_override", None)

        from .prediction_helpers import (  # pylint: disable=import-outside-toplevel
            handle_uncalibrated_regression_prediction,
            handle_uncalibrated_classification_prediction,
            format_regression_prediction,
            format_classification_prediction,
        )

        # Lazy import API params functions (deferred from module level)
        from ..api.params import (
            canonicalize_kwargs,
            reject_removed_aliases,
            reject_removed_reject_kwargs,
            validate_param_combination,
        )

        # reject removed aliases and normalize kwargs
        reject_removed_aliases(kwargs)
        reject_removed_guarded_kwargs(kwargs)
        reject_removed_reject_kwargs(kwargs)
        reject_removed_normalization_kwarg(kwargs)
        reject_unknown_public_kwargs(
            kwargs, allowed=_PREDICT_KWARGS, surface="CalibratedExplainer.predict"
        )
        kwargs = canonicalize_kwargs(kwargs)
        validate_param_combination(kwargs)
        if "interval_summary" not in kwargs or kwargs["interval_summary"] is None:
            kwargs["interval_summary"] = self.interval_summary
        else:
            kwargs["interval_summary"] = coerce_interval_summary(kwargs["interval_summary"])
        if kwargs.get("threshold") is not None and "regression" in self.mode:
            assert_threshold(kwargs["threshold"], x)
        elif "regression" in self.mode:
            validated_percentiles = validate_low_high_percentiles(
                kwargs.get("low_high_percentiles", (5, 95))
            )
            if "low_high_percentiles" in kwargs:
                kwargs["low_high_percentiles"] = validated_percentiles

        if not calibrated:
            if self.mode == "regression":
                return handle_uncalibrated_regression_prediction(
                    self.learner, x, threshold=kwargs.get("threshold"), uq_interval=uq_interval
                )
            return handle_uncalibrated_classification_prediction(
                self.learner, x, threshold=kwargs.get("threshold"), uq_interval=uq_interval
            )

        kwargs["bins"] = resolve_conditional_bins(
            x,
            kwargs.get("bins"),
            calibration_bins=self.bins,
        )

        # Resolve reject policy (per-call overrides explainer default)
        from .reject.policy import RejectPolicy as _RejectPolicy
        from .reject.orchestrator import (  # pylint: disable=import-outside-toplevel
            resolve_effective_reject_policy,
        )

        # Internal callers may skip reject orchestration by setting this flag
        if kwargs.pop("_ce_skip_reject", False):
            skip_reject_for_internal = True
            resolution = None
        else:
            skip_reject_for_internal = False
            resolution = resolve_effective_reject_policy(
                kwargs.pop("reject_policy", None),
                self,
                default_policy=getattr(self, "default_reject_policy", _RejectPolicy.NONE),
                logger=logging.getLogger(__name__),
            )
        policy = _RejectPolicy.NONE if skip_reject_for_internal else resolution.policy

        implicit_default_used = (
            (not skip_reject_for_internal)
            and resolution is not None
            and resolution.used_default
            and policy is not _RejectPolicy.NONE
        )

        # If no reject orchestration requested, proceed with legacy behavior
        if policy is _RejectPolicy.NONE or skip_reject_for_internal:
            # Calibrated predictions
            if self.mode == "regression":
                predict, low, high, _ = self.prediction_orchestrator.predict(x, **kwargs)
                return format_regression_prediction(
                    predict, low, high, threshold=kwargs.get("threshold"), uq_interval=uq_interval
                )

            # Classification
            predict, low, high, new_classes = self.prediction_orchestrator.predict(x, **kwargs)
            return format_classification_prediction(
                predict,
                low,
                high,
                new_classes,
                self.is_multiclass(),
                original_class_values=self.original_class_values,
                label_map=self.label_map,
                class_labels=self.class_labels,
                uq_interval=uq_interval,
            )

        # Reject policy active: use orchestrator to apply policy and return RejectResult envelope
        bins_arg = kwargs.pop("bins", None)
        confidence_arg = kwargs.pop("reject_confidence", 0.95)
        rr = self.reject_orchestrator.apply_policy(
            policy,
            x,
            explain_fn=None,
            bins=bins_arg,
            reject_confidence=confidence_arg,
            result_schema="v2",
            **kwargs,
        )
        try:
            from ..explanations.reject import (
                RejectResultV2,  # pylint: disable=import-outside-toplevel
                reject_result_v2_to_legacy,
            )

            if isinstance(rr, RejectResultV2):
                rr = reject_result_v2_to_legacy(rr, emit_deprecation_warning=False)
        except Exception as exc:  # adr002_allow
            logging.getLogger(__name__).debug(
                "RejectResultV2 compatibility conversion failed in predict: %s",
                exc,
                exc_info=True,
            )

        # Format the legacy payload into rr.prediction for consumer ergonomics
        try:
            if rr.prediction is not None:
                if self.mode == "regression":
                    # prediction is expected as (predict, low, high, _)
                    predict, low, high, _ = rr.prediction
                    rr.prediction = format_regression_prediction(
                        predict,
                        low,
                        high,
                        threshold=kwargs.get("threshold"),
                        uq_interval=uq_interval,
                    )
                else:
                    predict, low, high, new_classes = rr.prediction
                    rr.prediction = format_classification_prediction(
                        predict,
                        low,
                        high,
                        new_classes,
                        self.is_multiclass(),
                        original_class_values=self.original_class_values,
                        label_map=self.label_map,
                        class_labels=self.class_labels,
                        uq_interval=uq_interval,
                    )
        except Exception as exc:  # adr002_allow
            # If formatting fails, leave rr.prediction as-is but warn
            logging.getLogger(__name__).info(
                "Failed to format RejectResult.prediction; leaving raw.", exc_info=True
            )
            warnings.warn(
                f"Failed to format RejectResult.prediction: {exc!s}", UserWarning, stacklevel=2
            )

        # Log once-per-call when an implicit default caused an envelope return
        if implicit_default_used:
            logging.getLogger(__name__).info(
                "Default reject policy %s applied implicitly; returning RejectResult envelope for this call.",
                str(policy),
            )

        return rr

    def predict_proba(self, x, uq_interval=False, calibrated=True, threshold=None, **kwargs):
        """Generate probability predictions for the test data.

        This is a wrapper around the predict_proba method which is more similar to the scikit-learn predict_proba method for classification.
        As opposed to predict_proba, this method may output uncertainty intervals.

        Parameters
        ----------
        x : array-like
            The test data for which predictions are to be made. This should be in a format compatible with sklearn (e.g., numpy arrays, pandas DataFrames).
        uq_interval : bool, default=False
            If true, then the prediction interval is returned as well.
        calibrated : bool, default=True
            If True, the calibrator is used for prediction. If False, the underlying learner is used for prediction.
        threshold : float, int or array-like of shape (n_samples,), optional, default=None
            Threshold values used with regression to get probability of being below the
            threshold. Classification calls with this parameter raise
            ``ValidationError``.

        Raises
        ------
        NotFittedError
            If the explainer has not been fitted/calibrated prior to calling
            ``predict_proba``.

        ConfigurationError
            If unsupported, removed, or conflicting keyword arguments are
            supplied.

        ValidationError
            If ``threshold`` is invalid for the configured mode (for example,
            a classification call with ``threshold=``, or a threshold whose
            length does not match the number of instances in ``x``).

        Returns
        -------
        calibrated probability :
            The calibrated probability of the positive class (or the predicted class for multiclass).
        (low, high) : tuple of float lists, corresponding to the lower and upper bound of each prediction interval.

        Examples
        --------
        For a prediction without uncertainty quantification intervals:

        .. code-block:: python

            w.predict_proba(x)

        For a prediction with uncertainty quantification intervals:

        .. code-block:: python

            w.predict_proba(x, uq_interval=True)

        Notes
        -----
        Classification calls with `threshold=` raise `ValidationError`.
        """
        # strip plotting-only keys that callers may pass
        kwargs.pop("show", None)
        kwargs.pop("style_override", None)
        # Lazy import API params functions (deferred from module level)
        from ..api.params import (
            canonicalize_kwargs,
            reject_removed_aliases,
            reject_removed_reject_kwargs,
            validate_param_combination,
        )

        # reject removed aliases and normalize kwargs
        reject_removed_aliases(kwargs)
        reject_removed_guarded_kwargs(kwargs)
        reject_removed_reject_kwargs(kwargs)
        reject_removed_normalization_kwarg(kwargs)
        reject_unknown_public_kwargs(
            kwargs, allowed=_PREDICT_PROBA_KWARGS, surface="CalibratedExplainer.predict_proba"
        )
        kwargs = canonicalize_kwargs(kwargs)
        validate_param_combination(kwargs)
        if threshold is not None and "regression" in self.mode:
            assert_threshold(threshold, x)

        # Inject default interval_summary if not provided
        kwargs.setdefault("interval_summary", self.interval_summary)
        confidence_arg = kwargs.pop("reject_confidence", 0.95)

        # Resolve reject policy (per-call override else explainer default)
        from .reject.policy import RejectPolicy as _RejectPolicy
        from .reject.orchestrator import (  # pylint: disable=import-outside-toplevel
            resolve_effective_reject_policy,
        )

        # Internal callers may skip reject orchestration by setting this flag
        if kwargs.pop("_ce_skip_reject", False):
            skip_reject_for_internal = True
            resolution = None
        else:
            skip_reject_for_internal = False
            resolution = resolve_effective_reject_policy(
                kwargs.pop("reject_policy", None),
                self,
                default_policy=getattr(self, "default_reject_policy", _RejectPolicy.NONE),
                logger=logging.getLogger(__name__),
            )

        policy = _RejectPolicy.NONE if skip_reject_for_internal else resolution.policy

        implicit_default_used = (
            (not skip_reject_for_internal)
            and resolution is not None
            and resolution.used_default
            and policy is not _RejectPolicy.NONE
        )
        if (
            not skip_reject_for_internal
            and policy is not _RejectPolicy.NONE
            and self.mode == "regression"
            and threshold is None
        ):
            raise ValidationError("reject learner unavailable for regression without threshold")

        # Helper: compute legacy proba payload for this call
        proba_payload = None

        if not calibrated:
            if threshold is not None:
                raise ValidationError(
                    "A thresholded prediction is not possible for uncalibrated learners."
                )
            if uq_interval:
                proba = self.learner.predict_proba(x)
                if proba.shape[1] > 2:
                    proba_payload = (proba, (proba, proba))
                else:
                    proba_payload = (proba, (proba[:, 1], proba[:, 1]))
            else:
                proba_payload = self.learner.predict_proba(x)
        else:
            kwargs["bins"] = resolve_conditional_bins(
                x,
                kwargs.get("bins"),
                calibration_bins=self.bins,
            )
            # Calibrated predictions
            if self.mode == "regression":
                # y_threshold is the internal alias for the user-facing `threshold` parameter (matches crepes API convention)
                if is_fast_interval_collection(self.interval_learner):
                    proba_1, low, high, _ = self.interval_learner[-1].predict_probability(
                        x, y_threshold=threshold, **kwargs
                    )
                else:
                    proba_1, low, high, _ = self.interval_learner.predict_probability(
                        x, y_threshold=threshold, **kwargs
                    )
                proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
                proba_payload = (proba, (low, high)) if uq_interval else proba

            # Classification - multiclass
            elif self.is_multiclass():
                if threshold is not None:
                    raise ValidationError(
                        "The threshold parameter is only supported for mode='regression'.",
                        details={
                            "param": "threshold",
                            "mode": self.mode,
                            "surface": "CalibratedExplainer.predict_proba",
                        },
                    )
                if is_fast_interval_collection(self.interval_learner):
                    proba, low, high, _ = self.interval_learner[-1].predict_proba(
                        x, output_interval=True, **kwargs
                    )
                else:
                    proba, low, high, _ = self.interval_learner.predict_proba(
                        x, output_interval=True, **kwargs
                    )
                proba_payload = (proba, (low, high)) if uq_interval else proba

            # Classification - binary
            else:
                if threshold is not None:
                    raise ValidationError(
                        "The threshold parameter is only supported for mode='regression'.",
                        details={
                            "param": "threshold",
                            "mode": self.mode,
                            "surface": "CalibratedExplainer.predict_proba",
                        },
                    )
                if is_fast_interval_collection(self.interval_learner):
                    proba, low, high = self.interval_learner[-1].predict_proba(
                        x, output_interval=True, **kwargs
                    )
                else:
                    proba, low, high = self.interval_learner.predict_proba(
                        x, output_interval=True, **kwargs
                    )
                proba_payload = (proba, (low, high)) if uq_interval else proba

        # If no reject orchestration requested, return legacy payload
        if policy is _RejectPolicy.NONE or skip_reject_for_internal:
            return proba_payload

        # Reject policy active: compute envelope via orchestrator and attach legacy payload
        bins_arg = kwargs.pop("bins", None)
        rr = self.reject_orchestrator.apply_policy(
            policy,
            x,
            explain_fn=None,
            bins=bins_arg,
            reject_confidence=confidence_arg,
            threshold=threshold,
            result_schema="v2",
            **kwargs,
        )
        try:
            from ..explanations.reject import (
                RejectResultV2,  # pylint: disable=import-outside-toplevel
                reject_result_v2_to_legacy,
            )

            if isinstance(rr, RejectResultV2):
                rr = reject_result_v2_to_legacy(rr, emit_deprecation_warning=False)
        except Exception as exc:  # adr002_allow
            logging.getLogger(__name__).debug(
                "RejectResultV2 compatibility conversion failed in predict_proba: %s",
                exc,
                exc_info=True,
            )
        rr.prediction = proba_payload

        # Log once-per-call when an implicit default caused an envelope return
        if implicit_default_used:
            logging.getLogger(__name__).info(
                "Default reject policy %s applied implicitly; returning RejectResult envelope for this call.",
                str(policy),
            )

        return rr

    # pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
    def plot(self, x, y=None, threshold=None, **kwargs):
        """Generate plots for the test data."""
        # Pass any style overrides along to the plotting function
        style_override = kwargs.pop("style_override", None)
        kwargs["style_override"] = style_override
        # Lazy import plotting function (deferred from module level)
        from ..plotting import plot_global

        return plot_global(self, x, y=y, threshold=threshold, **kwargs)

    def calibrated_confusion_matrix(self):
        """Generate a calibrated confusion matrix.

        Generates a confusion matrix for the calibration set to provide insights about model behavior.
        The confusion matrix is only available for classification tasks. Stratified cross-validation is
        used on the calibration set to generate the confusion matrix while avoiding quadratic
        recalibration overhead.

        Returns
        -------
        array-like
            The calibrated confusion matrix.
        """
        if self.mode != "classification":
            raise ValidationError(
                "The confusion matrix is only available for classification tasks."
            )
        from .calibration_metrics import (  # pylint: disable=import-outside-toplevel
            compute_calibrated_confusion_matrix,
        )

        return compute_calibrated_confusion_matrix(
            self.x_cal, self.y_cal, self.learner, bins=self.bins
        )

    def predict_calibration(self):
        """Predict the target values for the calibration data.

        Returns
        -------
        array-like
            Predicted values for the calibration data. For models that expose a hat matrix,
            this returns updated predictions using that matrix; otherwise it uses the
            predict_function on the calibration data.
        """
        return self.predict_function(self.x_cal)

    # Public alias for testing purposes (to avoid private member access in tests)
    @property
    def fast(self) -> bool:
        """Whether to use fast mode.

        Returns
        -------
        bool
            True if fast mode is enabled.
        """
        return self._fast

    @fast.setter
    def fast(self, value: bool) -> None:
        self._fast = value

    @property
    def noise_type(self) -> str:
        """The type of noise to use.

        Returns
        -------
        str
            The noise type.
        """
        return self._noise_type

    @noise_type.setter
    def noise_type(self, value: str) -> None:
        self._noise_type = value

    @property
    def scale_factor(self) -> float | None:
        """The scale factor for perturbations.

        Returns
        -------
        float | None
            The scale factor.
        """
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value: float | None) -> None:
        self._scale_factor = value

    @property
    def severity(self) -> float | None:
        """The severity of perturbations.

        Returns
        -------
        float | None
            The severity.
        """
        return self._severity

    @severity.setter
    def severity(self, value: float | None) -> None:
        self._severity = value


__all__ = ["CalibratedExplainer"]
