"""High-level wrapper for building, calibrating and explaining models.

This module provides :class:`WrapCalibratedExplainer`, a convenience wrapper
that mirrors :class:`.CalibratedExplainer` while exposing a scikit-learn
style fit/calibrate/explain surface for downstream users and integrations.
"""

# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-lines, too-many-positional-arguments, too-many-public-methods
from __future__ import annotations

import copy
import hashlib
import json
import logging as _logging
import os
import shutil
import sys
import tempfile
import warnings as _warnings
from collections.abc import Sequence as SequenceABC
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping

import numpy as np
from crepes.extras import MondrianCategorizer

from ..api.params import (
    reject_removed_aliases,
    reject_removed_guarded_kwargs,
    reject_removed_normalization_kwarg,
    reject_removed_reject_kwargs,
    reject_unknown_public_kwargs,
    validate_param_combination,
)
from ..utils import check_is_fitted, safe_isinstance  # noqa: F401
from ..utils.exceptions import (
    ConfigurationError,
    DataShapeError,
    IncompatibleStateError,
    ModelNotSupportedError,
    NotFittedError,
    SerializationError,
    ValidationError,
)
from .calibrated_explainer import (  # circular during split
    _EXPLAIN_KWARGS as _CE_EXPLAIN_KWARGS,
)
from .calibrated_explainer import (
    _INIT_EXPLICIT_PARAMS as _CE_INIT_EXPLICIT_PARAMS,
)
from .calibrated_explainer import (
    _INIT_KWARGS as _CE_INIT_KWARGS,
)
from .calibrated_explainer import (
    _PREDICT_KWARGS as _CE_PREDICT_KWARGS,
)
from .calibrated_explainer import (
    _PREDICT_PROBA_KWARGS as _CE_PREDICT_PROBA_KWARGS,
)
from .calibrated_explainer import (
    CalibratedExplainer,
)
from .prediction_helpers import (
    _apply_conditional_categorizer,
    _normalize_conditional_bins,
    resolve_conditional_bins,
)
from .validation import (
    validate_bool_parameter,
    validate_classification_calibration_targets,
    validate_explainer_init_kwargs,
    validate_inputs_matrix,
    validate_model,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from calibrated_explanations.api.config import ExplainerConfig

# ADR-038 D3/5A/5B history: unknown public kwargs used to be warned about and
# forwarded (v0.11.4 Task 15); v0.11.6 Task 5 made every gated method fail fast
# against a per-method allow-list. See ADR-038's 2026-07-08 Addenda and
# development/current-work/v0.11.6_plan.md (Tasks 5/5A/5B/5C/5D) for the full
# rationale, including which historical names were dropped and why.
#
# Task 5D invariant: every per-method set below is *derived* from the
# CalibratedExplainer allow-lists (the single source of truth in
# calibrated_explainer.py) so the two gates cannot drift apart. Anything
# CalibratedExplainer accepts on a surface must also be accepted by the
# wrapper's corresponding method; the wrapper only subtracts
# "_ce_skip_reject" (internal orchestrator escape hatch, not public API) and
# adds names the wrapper itself consumes. Enforced by
# tests/unit/core/test_parameter_surface_contracts.py.

# used by: calibrate() only (session/construction-time configuration).
# Everything CalibratedExplainer.__init__ accepts -- via **kwargs or as an
# explicit formal parameter -- plus the wrapper-only "reuse_conditional".
# perf_cache/perf_parallel are forwarded with kwargs.setdefault() in
# calibrate(), so a call-time value wins over the wrapper attribute.
_CALIBRATE_KWARGS: frozenset[str] = (
    _CE_INIT_KWARGS | _CE_INIT_EXPLICIT_PARAMS | frozenset({"reuse_conditional"})
)

# used by: explain_factual() and explore_alternatives() (identical kwarg surface
# at the CalibratedExplainer level; threshold/low_high_percentiles/bins/
# features_to_ignore/guarded_options bind explicit formals there).
_EXPLAIN_KWARGS: frozenset[str] = _CE_EXPLAIN_KWARGS

# used by: explain_fast() only. CalibratedExplainer.explain_fast has no **kwargs
# at all -- these mirror its fully explicit signature exactly (checked by the
# parameter-surface contract tests).
_EXPLAIN_FAST_KWARGS: frozenset[str] = frozenset(
    {
        "bins",
        "threshold",
        "low_high_percentiles",
        "reject_policy",
    }
)

# used by: predict() only. uq_interval/calibrated/reject_policy are explicit
# named parameters here (reject_policy stays in the derived set harmlessly --
# the explicit formal always captures it before **kwargs).
_PREDICT_KWARGS: frozenset[str] = _CE_PREDICT_KWARGS - frozenset({"_ce_skip_reject"})

# used by: predict_proba() only. uq_interval/calibrated/threshold/reject_policy
# are explicit named parameters here.
_PREDICT_PROBA_KWARGS: frozenset[str] = _CE_PREDICT_PROBA_KWARGS - frozenset({"_ce_skip_reject"})

_KNOWN_PUBLIC_KWARGS: frozenset[str] = (
    _CALIBRATE_KWARGS
    | _EXPLAIN_KWARGS
    | _EXPLAIN_FAST_KWARGS
    | _PREDICT_KWARGS
    | _PREDICT_PROBA_KWARGS
)


class WrapCalibratedExplainer:
    """Provide a high-level fit/calibrate/explain workflow for learners.

    The wrapper mirrors :class:`CalibratedExplainer` while orchestrating
    fitting, calibration, and explanation steps behind a scikit-learn style
    interface.

    Attributes
    ----------
    learner : Any
        The underlying predictive learner instance.
    explainer : CalibratedExplainer | None
        The calibrated explainer created during :meth:`calibrate`.
    calibrated : bool
        True when the wrapper has been calibrated.
    """

    learner: Any
    explainer: CalibratedExplainer | None
    calibrated: bool
    mc: Callable[[Any], Any] | MondrianCategorizer | None
    _logger: _logging.Logger
    # Schema v3 (ADR-031 security hardening): the persisted artifact contains
    # only JSON-safe declarative data; no wrapper or calibrator pickle bytes
    # are ever written or read by save_state()/load_state(). Schema v1/v2
    # artifacts persisted the whole wrapper (and unsupported calibrators) via
    # pickle and are rejected unconditionally -- see IncompatibleStateError
    # messages in load_state() for migration guidance.
    _STATE_SCHEMA_VERSION: int = 3
    _LEGACY_STATE_SCHEMA_VERSIONS: "frozenset[int]" = frozenset({1, 2})
    _SAFE_STATE_FILES: "frozenset[str]" = frozenset(
        {"explainer_state.json", "calibrator_primitive.json", "preprocessing_mapping.json"}
    )

    def __init__(self, learner: Any):
        """Initialize the WrapCalibratedExplainer with a predictive learner.

        Parameters
        ----------
        learner : predictive learner
            A predictive learner that can be used to predict the target variable.
        """
        self.mc: Callable[[Any], Any] | MondrianCategorizer | None = None
        self._logger: _logging.Logger = _logging.getLogger(__name__)
        # Optional preprocessing
        self._preprocessor: Any | None = None
        self._pre_fitted: bool = False
        self._auto_encode: bool | str = "auto"
        self._unseen_category_policy: str = "error"
        self._builtin_categorical_features: tuple[int, ...] = ()
        self._missing_value_policy: str = "category"
        # Check if the learner is a CalibratedExplainer
        if safe_isinstance(learner, "calibrated_explanations.core.CalibratedExplainer"):
            explainer = learner
            underlying_learner = explainer.learner
            self.learner: Any = underlying_learner
            check_is_fitted(self.learner)
            self.fitted: bool = True
            self.explainer: CalibratedExplainer | None = explainer
            self.calibrated: bool = True
            self._logger.info(
                "Initialized from existing CalibratedExplainer (already fitted & calibrated)"
            )
            return
        self.learner = learner
        self.explainer = None
        self.calibrated = False

        # Check if the learner is already fitted
        self.fitted = False
        with suppress(TypeError, RuntimeError, NotFittedError):
            check_is_fitted(learner)
            self.fitted = True

    def __repr__(self) -> str:
        """Return the string representation of the WrapCalibratedExplainer."""
        if self.fitted:
            if self.calibrated:
                return (
                    f"WrapCalibratedExplainer(learner={self.learner}, fitted=True, "
                    f"calibrated=True, \n\t\texplainer={self.explainer})"
                )
            return f"WrapCalibratedExplainer(learner={self.learner}, fitted=True, calibrated=False)"
        return f"WrapCalibratedExplainer(learner={self.learner}, fitted=False, calibrated=False)"

    @property
    def parallel_executor(self) -> Any:
        """Expose the internal parallel executor if available."""
        return getattr(self, "_perf_parallel", None)

    @parallel_executor.setter
    def parallel_executor(self, value: Any) -> None:
        """Allow setting the internal parallel executor."""
        self._perf_parallel = value

    @property
    def auto_encode(self) -> bool | str:
        """Get the auto_encode configuration."""
        return self._auto_encode

    @auto_encode.setter
    def auto_encode(self, value: bool | str) -> None:
        """Set the auto_encode configuration."""
        self._auto_encode = value

    @property
    def preprocessor(self) -> Any:
        """Get the preprocessor."""
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value: Any) -> None:
        """Set the preprocessor."""
        self._preprocessor = value

    @property
    def mondrian_categorizer(self) -> Callable[[Any], Any] | MondrianCategorizer | None:
        """Descriptive alias for :attr:`mc` (ADR-038 5B); read-only."""
        return self.mc

    # internal wiring for config
    @classmethod
    def from_config(cls, cfg: ExplainerConfig) -> WrapCalibratedExplainer:
        """Construct a wrapper from an :class:`ExplainerConfig`.

        Notes
        -----
        Fields wired during construction
            ``preprocessor``, ``auto_encode``, ``unseen_category_policy``;
            performance primitives (cache, parallel executor) via the perf
            factory; internal feature-filter config.

        Fields applied at explain-time
            ``threshold`` and ``low_high_percentiles`` are stored on the config
            and forwarded to ``explain_factual`` / ``explore_alternatives`` via
            ``kwargs.setdefault()``.

        Raises
        ------
        ConfigurationError
            If an explicitly requested performance cache or parallel executor
            cannot be initialized. ``details`` name the ``capability``, the
            ``source`` that failed and the underlying ``cause``.
        """
        # lazy import to avoid import cycles
        from calibrated_explanations.api.config import (
            _build_perf_factory,
            _perf_initialization_error,
        )

        w = cls(cfg.model)
        # Stash config on the instance for later optional use (private attr)
        w._cfg = cfg  # type: ignore[attr-defined]
        # Wire perf primitives (opt-in, ADR-003/ADR-004). With both flags off the
        # factory yields a disabled cache and a disabled executor, so runtime
        # behaviour is unchanged. Any failure means an explicitly requested
        # capability cannot be honoured, so it fails closed instead of degrading.
        perf_factory = getattr(cfg, "_perf_factory", None)
        if perf_factory is None:
            perf_factory = _build_perf_factory(cfg)
        if perf_factory is not None:
            activation = getattr(perf_factory, "activation", {})
            try:
                cache = perf_factory.make_cache()
            except Exception as exc:  # adr002_allow: re-raised as ConfigurationError
                raise _perf_initialization_error(
                    "cache", "factory", exc, requested_by=activation.get("cache")
                ) from exc
            try:
                executor = perf_factory.make_parallel_executor(cache)
            except Exception as exc:  # adr002_allow: re-raised as ConfigurationError
                raise _perf_initialization_error(
                    "parallel", "factory", exc, requested_by=activation.get("parallel")
                ) from exc
            if any(activation.values()):
                w._logger.info(
                    "Performance primitives from config: cache=%s, parallel=%s",
                    activation.get("cache") or "off",
                    activation.get("parallel") or "off",
                )
        else:
            cache = None
            executor = None
        w.perf_cache = cache  # type: ignore[attr-defined]
        w._perf_parallel = executor  # type: ignore[attr-defined]
        # Public-facing attribute expected by tests
        w.perf_parallel = executor  # type: ignore[attr-defined]
        # Wire internal feature filter config (FAST-based) when present
        try:
            from .explain._feature_filter import (  # pylint: disable=import-outside-toplevel
                FeatureFilterConfig,
            )

            enabled = getattr(cfg, "perf_feature_filter_enabled", False)
            per_instance_top_k = getattr(cfg, "perf_feature_filter_per_instance_top_k", 8)
            w._feature_filter_config = FeatureFilterConfig(  # type: ignore[attr-defined]
                enabled=bool(enabled),
                per_instance_top_k=max(1, int(per_instance_top_k)),
            )
        except:  # noqa: E722
            # Best-effort fallback: if importing the internal helper fails for
            # any reason, create a lightweight fallback object exposing the
            # attributes the runtime and tests expect. This avoids silent
            # missing attribute errors when feature-filter internals are
            # unavailable in constrained environments.
            from types import SimpleNamespace

            enabled = getattr(cfg, "perf_feature_filter_enabled", False)
            per_instance_top_k = getattr(cfg, "perf_feature_filter_per_instance_top_k", 8)
            w._feature_filter_config = SimpleNamespace(
                enabled=bool(enabled),
                per_instance_top_k=max(1, int(per_instance_top_k)),
                strict_observability=False,
            )
            _logging.getLogger(__name__).debug("Using fallback feature_filter_config")
        # Wire optional preprocessing in a controlled way (only if provided)
        try:
            w._preprocessor = cfg.preprocessor  # type: ignore[attr-defined]
            w._auto_encode = cfg.auto_encode  # type: ignore[attr-defined]
            w._unseen_category_policy = cfg.unseen_category_policy  # type: ignore[attr-defined]
            w._builtin_categorical_features = tuple(
                int(feature) for feature in getattr(cfg, "categorical_features", ()) or ()
            )  # type: ignore[attr-defined]
            w._missing_value_policy = getattr(cfg, "missing_value_policy", "category")  # type: ignore[attr-defined]
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            exc = sys.exc_info()[1]
            _logging.getLogger(__name__).warning(
                "Failed to transfer preprocessing config to wrapper: %s", exc
            )
        return w

    def fit(
        self, x_proper_train: Any, y_proper_train: Any, **kwargs: Any
    ) -> WrapCalibratedExplainer:
        """Fit the underlying learner on training data.

        Parameters
        ----------
        x_proper_train : array-like of shape (n_samples, n_features)
            Training input samples.
        y_proper_train : array-like of shape (n_samples,)
            Training target values.
        **kwargs
            Additional keyword arguments forwarded to the learner's ``fit``.

        Returns
        -------
        WrapCalibratedExplainer
            The wrapper instance (allows chaining).

        Examples
        --------
        >>> w = WrapCalibratedExplainer(clf)
        >>> w.fit(X_train, y_train)
        WrapCalibratedExplainer(...)
        """
        reinitialize = bool(self.calibrated)
        # Optional preprocessing: fit on training data when provided. Run this
        # before invalidating fitted/calibrated state so a rejected
        # preprocessing call leaves the prior lifecycle state untouched.
        x_train_local = x_proper_train
        if self._should_pre_fit_preprocess(x_train_local):
            x_train_local = self._pre_fit_preprocess(x_train_local)
        self.fitted = False
        self.calibrated = False
        self._logger.info("Fitting underlying learner: %s", type(self.learner).__name__)
        self.learner.fit(x_train_local, y_proper_train, **kwargs)
        # delegate shared post-fit logic
        return self._finalize_fit(reinitialize)

    def calibrate(
        self,
        x_calibration: Any,
        y_calibration: Any,
        mc: Callable[[Any], Any] | MondrianCategorizer | None = None,
        reuse_conditional: bool = False,
        *,
        mondrian_categorizer: Callable[[Any], Any] | MondrianCategorizer | None = None,
        **kwargs: Any,
    ) -> WrapCalibratedExplainer:
        """Calibrate the wrapper using calibration data and create an explainer.

        Parameters
        ----------
        x_calibration : array-like of shape (n_samples, n_features)
            Calibration features used to fit internal calibrators.
        y_calibration : array-like of shape (n_samples,)
            Calibration targets corresponding to ``x_calibration``.
        mc : callable or MondrianCategorizer, optional
            Optional Mondrian categories helper. Defaults to ``None``.
        reuse_conditional : bool, default=False
            Reuse the previously configured Mondrian categorizer for this
            calibration. Mutually exclusive with ``bins`` and ``mc``.
        mondrian_categorizer : callable or MondrianCategorizer, optional
            Descriptive alias for ``mc`` (ADR-038 5B). Resolves to the same value;
            specifying both ``mc`` and ``mondrian_categorizer`` raises
            ``ConfigurationError``.
        **kwargs
            Forwarded to :class:`.CalibratedExplainer.__init__` for advanced
            configuration (e.g. ``mode``, ``feature_names``, ``bins``). Every
            name accepted by :class:`.CalibratedExplainer.__init__` is accepted
            here; for ``perf_cache``/``perf_parallel`` a call-time value
            overrides the wrapper-level attribute.

        Returns
        -------
        WrapCalibratedExplainer
            The wrapper instance with the ``explainer`` attribute set to a
            configured :class:`.CalibratedExplainer`.

        Raises
        ------
        NotFittedError
            If the underlying learner has not been fitted via :meth:`fit`.
        ConfigurationError
            If both ``mc`` and ``mondrian_categorizer`` are specified.
        ModelNotSupportedError
            If the underlying learner does not implement ``predict``.

        Examples
        --------
        >>> w = WrapCalibratedExplainer(clf)
        >>> w.fit(X_train, y_train)
        >>> w.calibrate(X_cal, y_cal)

        Notes
        -----
        If ``mode`` is not provided in ``kwargs`` the wrapper will infer
        classification vs regression from the presence of ``predict_proba``
        on the underlying learner.
        """
        self._assert_fitted("The WrapCalibratedExplainer must be fitted before calibration.")

        if mondrian_categorizer is not None:
            if mc is not None:
                raise ConfigurationError(
                    "Specify either mc= or mondrian_categorizer=, not both; they are"
                    " aliases for the same parameter.",
                    details={"conflict": ("mc", "mondrian_categorizer")},
                )
            mc = mondrian_categorizer

        snapshot = self._snapshot_calibration_state()
        stage = "surface_validation"
        try:
            # Normalize kwargs at the public boundary; warn and strip alias keys only
            kwargs = self._normalize_public_kwargs(
                kwargs, allowed=_CALIBRATE_KWARGS, surface="WrapCalibratedExplainer.calibrate"
            )
            reuse_conditional = validate_bool_parameter(
                kwargs.pop("reuse_conditional", reuse_conditional),
                param="reuse_conditional",
            )
            validate_param_combination(kwargs)
            # Lightweight validation (does not alter behavior)
            validate_model(self.learner)
            preprocessor_metadata = self._build_preprocessor_metadata()

            stage = "preprocessor_fit_transform"
            # Optional preprocessing: ensure preprocessor is fitted (fit here if needed), then transform
            x_cal_local = x_calibration
            if self._should_pre_fit_preprocess(x_cal_local):
                if not self._pre_fitted:
                    self._logger.info("Fitting preprocessor on calibration data")
                    x_cal_local = self._pre_fit_preprocess(x_cal_local)
                else:
                    x_cal_local = self._pre_transform(x_cal_local, stage="calibrate")
                # Optional second transform call to ensure deterministic persistence
                # accounting in tests (ignore failures defensively)
                with suppress(Exception):  # pragma: no cover - defensive
                    _ = self._pre_transform(x_calibration, stage="calibrate_check")
            validate_inputs_matrix(x_cal_local, y_calibration, require_y=True, allow_nan=False)

            stage = "conditional_calibration"
            supplied = {
                "bins": kwargs.get("bins") is not None,
                "mc": mc is not None,
                "reuse_conditional": reuse_conditional,
            }
            if sum(supplied.values()) > 1:
                provided = [name for name, present in supplied.items() if present]
                raise ValidationError(
                    "Specify exactly one conditional calibration channel: bins, mc, or reuse_conditional.",
                    details={
                        "provided": provided,
                        "requirement": "one conditional channel per calibrate call",
                    },
                )

            candidate_mc = None
            candidate_bins = None
            if reuse_conditional:
                if self.mc is None:
                    raise ValidationError(
                        "reuse_conditional=True requires a stored Mondrian categorizer; "
                        "inline bins cannot transfer to a new calibration set, so pass fresh bins=.",
                        details={"requirement": "stored mc required for reuse_conditional"},
                    )
                candidate_mc = self.mc
            elif mc is not None:
                candidate_mc = mc

            if candidate_mc is not None:
                derived_bins = _apply_conditional_categorizer(candidate_mc, x_cal_local)
                candidate_bins = _normalize_conditional_bins(
                    derived_bins, n_samples=len(np.asarray(x_cal_local))
                )
            elif kwargs.get("bins") is not None:
                candidate_bins = _normalize_conditional_bins(
                    kwargs["bins"], n_samples=len(np.asarray(x_cal_local))
                )

            candidate_kwargs = dict(kwargs)
            candidate_kwargs["bins"] = candidate_bins
            if preprocessor_metadata is not None:
                candidate_kwargs.setdefault("preprocessor_metadata", preprocessor_metadata)

            self._logger.info(
                "Calibrating with %s samples", getattr(x_calibration, "shape", ["?"])[0]
            )

            # A call-time value wins over the wrapper-level performance attributes.
            candidate_kwargs.setdefault("perf_cache", getattr(self, "perf_cache", None))
            candidate_kwargs.setdefault("perf_parallel", getattr(self, "_perf_parallel", None))
            if "mode" not in candidate_kwargs:
                candidate_kwargs["mode"] = (
                    "classification" if "predict_proba" in dir(self.learner) else "regression"
                )
            candidate_mode, candidate_kwargs = validate_explainer_init_kwargs(
                candidate_kwargs,
                mode=candidate_kwargs["mode"],
                n_features=int(np.asarray(x_cal_local).shape[1]),
            )
            candidate_kwargs["mode"] = candidate_mode

            stage = "target_validation"
            if candidate_mode == "classification":
                validate_classification_calibration_targets(y_calibration, learner=self.learner)

            stage = "explainer_construction"
            candidate_explainer = CalibratedExplainer(
                self.learner,
                x_cal_local,
                y_calibration,
                **candidate_kwargs,
            )

            stage = "post_construction_configuration"
            self._finalize_candidate_calibration(candidate_explainer, preprocessor_metadata)

        except (
            ConfigurationError,
            DataShapeError,
            IncompatibleStateError,
            ModelNotSupportedError,
            NotFittedError,
            ValidationError,
        ):
            self._restore_calibration_state(snapshot)
            raise
        except (
            Exception
        ) as exc:  # adr002_allow - normalize calibration-path failures to CE exceptions
            self._restore_calibration_state(snapshot)
            raise ConfigurationError(
                f"Calibration failed during {stage}: {exc}",
                details={
                    "stage": stage,
                    "original_error_type": type(exc).__name__,
                    "original_error": str(exc),
                },
            ) from exc

        # Commit only after every validation and construction step succeeds.
        self.mc = candidate_mc
        self.explainer = candidate_explainer
        self.calibrated = True
        return self

    @property
    def feature_filter_config(self) -> Any:
        """Expose the feature-filter configuration if available.

        Tests and plugins may access this property on the wrapper; prefer
        the internally-stored config, otherwise delegate to the explainer.
        """
        if hasattr(self, "_feature_filter_config"):
            return self._feature_filter_config
        if self.explainer is not None:
            return getattr(self.explainer, "feature_filter_config", None)
        return None

    def explain_factual(self, x: Any, **kwargs: Any) -> Any:
        """Generate factual explanations for provided instances.

        Parameters
        ----------
        x : array-like
            Instances to explain (single or batch). Shape should match the
            feature dimensionality used during calibration.
        **kwargs
            Forwarded to :meth:`CalibratedExplainer.explain_factual`.

        Returns
        -------
        CalibratedExplanations or mapping
            Explanation collection produced by the underlying explainer.

        Notes
        -----
        **Assumption boundary**: This method verifies the API contract — that the
        call completes and returns a valid explanation collection. It does not
        guarantee the statistical validity of calibrated feature attributions for
        any particular instance. The calibration validity depends on the
        exchangeability assumption: the calibration set must be representative of
        the test distribution. Feature attribution magnitudes reflect calibrated
        probability shifts, not causal importances or ground-truth attribution
        correctness.

        See Also
        --------
        :meth:`CalibratedExplainer.explain_factual`
            For full parameter and return semantics.
        """
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted and calibrated before explaining."
            )
            ._assert_calibrated("The WrapCalibratedExplainer must be calibrated before explaining.")
            .explainer
            is not None
        )
        # Optional preprocessing
        x_local = self._maybe_preprocess_for_inference(x)
        kwargs = self._normalize_public_kwargs(
            kwargs, allowed=_EXPLAIN_KWARGS, surface="WrapCalibratedExplainer.explain_factual"
        )
        # If constructed via _from_config, prefer cfg defaults when absent
        cfg = getattr(self, "_cfg", None)
        if cfg is not None:
            kwargs.setdefault("threshold", cfg.threshold)
            # low_high_percentiles only applies to regression-style intervals; safe to pass through
            kwargs.setdefault("low_high_percentiles", cfg.low_high_percentiles)
        validate_inputs_matrix(x_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(x_local, **kwargs)
        return self.explainer.explain_factual(x_local, **kwargs)

    def explore_alternatives(self, x: Any, **kwargs: Any) -> Any:
        """Generate alternative explanations for the test data.

        Notes
        -----
        **Assumption boundary**: Alternative explanations describe feature changes
        that would shift the predicted probability toward an alternative outcome.
        They do not guarantee that the described feature changes are physically
        achievable, distributionally feasible, or actionable in a new model
        deployment. The exchangeability assumption applies: results depend on the
        calibration set being representative of the test distribution.

        See Also
        --------
        :meth:`.CalibratedExplainer.explore_alternatives` : Refer to the docstring
            for explore_alternatives in CalibratedExplainer for more details.
        """
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted and calibrated before explaining."
            )
            ._assert_calibrated("The WrapCalibratedExplainer must be calibrated before explaining.")
            .explainer
            is not None
        )
        x_local = self._maybe_preprocess_for_inference(x)
        kwargs = self._normalize_public_kwargs(
            kwargs, allowed=_EXPLAIN_KWARGS, surface="WrapCalibratedExplainer.explore_alternatives"
        )
        cfg = getattr(self, "_cfg", None)
        if cfg is not None:
            kwargs.setdefault("threshold", cfg.threshold)
            kwargs.setdefault("low_high_percentiles", cfg.low_high_percentiles)
        validate_inputs_matrix(x_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(x_local, **kwargs)
        return self.explainer.explore_alternatives(x_local, **kwargs)

    def explain_fast(self, x: Any, **kwargs: Any) -> Any:
        """Generate fast explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.explain_fast` : Refer to the docstring for explain_fast in CalibratedExplainer for more details.
        """
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted and calibrated before explaining."
            )
            ._assert_calibrated("The WrapCalibratedExplainer must be calibrated before explaining.")
            .explainer
            is not None
        )
        x_local = self._maybe_preprocess_for_inference(x)
        kwargs = self._normalize_public_kwargs(
            kwargs, allowed=_EXPLAIN_FAST_KWARGS, surface="WrapCalibratedExplainer.explain_fast"
        )
        # Apply config defaults when available and not explicitly provided
        cfg = getattr(self, "_cfg", None)
        if cfg is not None:
            kwargs.setdefault("threshold", cfg.threshold)
            kwargs.setdefault("low_high_percentiles", cfg.low_high_percentiles)
        validate_inputs_matrix(x_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(x_local, **kwargs)
        assert self.explainer is not None
        return self.explainer.explain_fast(x_local, **kwargs)

    # pylint: disable=too-many-return-statements
    def predict(
        self,
        x: Any,
        uq_interval: bool = False,
        calibrated: bool = True,
        reject_policy: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate predictions for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.predict` : Refer to the docstring for predict in CalibratedExplainer for more details.
        """
        self._assert_fitted("The WrapCalibratedExplainer must be fitted before predicting.")
        x_local = self._maybe_preprocess_for_inference(x)
        kwargs = self._normalize_public_kwargs(
            kwargs, allowed=_PREDICT_KWARGS, surface="WrapCalibratedExplainer.predict"
        )
        if not self.calibrated:
            if "threshold" in kwargs:
                raise DataShapeError(
                    "A thresholded prediction is not possible for uncalibrated learners."
                )
            if calibrated:
                _warnings.warn(
                    "The WrapCalibratedExplainer must be calibrated to get calibrated predictions.",
                    UserWarning,
                    stacklevel=2,
                )
            if uq_interval:
                predict = self.learner.predict(x_local)
                return predict, (predict, predict)
            return self.learner.predict(x_local)

        validate_inputs_matrix(x_local, allow_nan=True)
        validate_param_combination(kwargs)
        if calibrated:
            kwargs["bins"] = self._get_bins(x_local, **kwargs)
        assert (
            self._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated to get calibrated predictions."
            ).explainer
            is not None
        )
        return self.explainer.predict(
            x_local,
            uq_interval=uq_interval,
            calibrated=calibrated,
            reject_policy=reject_policy,
            **kwargs,
        )

    def predict_proba(
        self,
        x: Any,
        uq_interval: bool = False,
        calibrated: bool = True,
        threshold: float | None = None,
        reject_policy: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate probability predictions for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.predict_proba` : Refer to the docstring for predict_proba in CalibratedExplainer for more details.
        """
        self._assert_fitted(
            "The WrapCalibratedExplainer must be fitted before predicting probabilities."
        )
        if "predict_proba" not in dir(self.learner):
            if threshold is None:
                raise ValidationError("The threshold parameter must be specified for regression.")
            self._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated to get calibrated probabilities for regression."
            )
        x_local = self._maybe_preprocess_for_inference(x)
        kwargs = self._normalize_public_kwargs(
            kwargs, allowed=_PREDICT_PROBA_KWARGS, surface="WrapCalibratedExplainer.predict_proba"
        )
        if not self.calibrated:
            if threshold is not None:
                raise DataShapeError(
                    "A thresholded prediction is not possible for uncalibrated learners."
                )
            if calibrated:
                _warnings.warn(
                    "The WrapCalibratedExplainer must be calibrated to get calibrated probabilities.",
                    UserWarning,
                    stacklevel=2,
                )
            # getattr to appease typing when learner may not expose predict_proba
            proba = self.learner.predict_proba(x_local)
            return self._format_proba_output(proba, uq_interval)

        validate_inputs_matrix(x_local, allow_nan=True)
        validate_param_combination(kwargs)
        if calibrated:
            kwargs["bins"] = self._get_bins(x_local, **kwargs)
        assert (
            self._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated to get calibrated probabilities."
            ).explainer
            is not None
        )
        return self.explainer.predict_proba(
            x_local,
            uq_interval=uq_interval,
            calibrated=calibrated,
            threshold=threshold,
            reject_policy=reject_policy,
            **kwargs,
        )

    def calibrated_confusion_matrix(self) -> Any:
        """Generate a calibrated confusion matrix.

        See Also
        --------
        :meth:`.CalibratedExplainer.calibrated_confusion_matrix` : Refer to the docstring for calibrated_confusion_matrix in CalibratedExplainer for more details.
        """
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted and calibrated before providing a confusion matrix."
            )
            ._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated before providing a confusion matrix."
            )
            .explainer
            is not None
        )
        return self.explainer.calibrated_confusion_matrix()

    def set_difficulty_estimator(
        self, difficulty_estimator: Any, *, initialize: bool = True
    ) -> None:
        """Assign or update the difficulty estimator.

        Parameters
        ----------
        difficulty_estimator : Any
            Difficulty estimator to assign, or ``None`` to clear it.
        initialize : bool, default=True
            Whether to reinitialize calibrated prediction internals after assignment.
            Use ``False`` only for advanced workflows that need to update reject
            strategy metadata without changing the calibrated probability path.

        See Also
        --------
        :meth:`.CalibratedExplainer.set_difficulty_estimator` : Refer to the docstring for set_difficulty_estimator in CalibratedExplainer for more details.
        """
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted and calibrated before assigning a difficulty estimator."
            )
            ._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated before assigning a difficulty estimator."
            )
            .explainer
            is not None
        )
        self.explainer.set_difficulty_estimator(difficulty_estimator, initialize=initialize)

    # pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
    def plot(self, x: Any, y: Any = None, threshold: float | None = None, **kwargs: Any) -> Any:
        """Generate plots for the test data.

        Parameters
        ----------
        x : array-like
            Test instances to plot explanations for.
        y : array-like, optional
            True labels for the test instances.
        threshold : float, optional
            Threshold for probabilistic regression.
        **kwargs : dict
            Additional keyword arguments passed to the plot method.

        Returns
        -------
        object or None
            The value returned by the underlying plot implementation.

        See Also
        --------
        :meth:`.CalibratedExplainer.plot` : Refer to the docstring for plot in CalibratedExplainer for more details.
        """
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted and calibrated before plotting."
            )
            ._assert_calibrated("The WrapCalibratedExplainer must be calibrated before plotting.")
            .explainer
            is not None
        )

        # Apply config defaults when available and not explicitly provided
        cfg = getattr(self, "_cfg", None)
        if cfg is not None:
            if threshold is None:
                threshold = cfg.threshold
            kwargs.setdefault("low_high_percentiles", cfg.low_high_percentiles)
        kwargs["bins"] = self._get_bins(x, **kwargs)
        return self.explainer.plot(x, y=y, threshold=threshold, **kwargs)

    def _get_bins(self, x: Any, **kwargs: Any) -> Any:
        """Derive bin assignments from the configured Mondrian categorizer."""
        return resolve_conditional_bins(
            x,
            kwargs.get("bins"),
            mc=self.mc,
            calibration_bins=(
                getattr(self.explainer, "bins", None) if self.explainer is not None else None
            ),
        )

    @property
    def runtime_telemetry(self) -> Mapping[str, Any]:
        """Return the most recent telemetry payload reported by the explainer."""
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted before accessing runtime telemetry."
            )
            ._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated before accessing runtime telemetry."
            )
            .explainer
            is not None
        )
        return self.explainer.runtime_telemetry

    @property
    def preprocessor_metadata(self) -> Dict[str, Any] | None:
        """Return the telemetry-safe preprocessing snapshot if available."""
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted before accessing preprocessor metadata."
            )
            ._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated before accessing preprocessor metadata."
            )
            .explainer
            is not None
        )
        return self.explainer.preprocessor_metadata

    def set_preprocessor_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        """Update the stored preprocessing metadata snapshot."""
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted before setting preprocessor metadata."
            )
            ._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated before setting preprocessor metadata."
            )
            .explainer
            is not None
        )
        self.explainer.set_preprocessor_metadata(metadata)

    # ------ Internal helpers (reduce duplication) ------
    def _assert_fitted(self, message: str | None = None) -> WrapCalibratedExplainer:
        if not self.fitted:
            raise NotFittedError(
                message or "The WrapCalibratedExplainer must be fitted before this operation."
            )
        return self

    def _assert_calibrated(self, message: str | None = None) -> WrapCalibratedExplainer:
        if not self.calibrated:
            raise NotFittedError(
                message or "The WrapCalibratedExplainer must be calibrated before this operation."
            )
        return self

    def _normalize_public_kwargs(
        self,
        kwargs: dict[str, Any],
        allowed: "frozenset[str] | set[str] | None" = None,
        *,
        surface: str | None = None,
    ) -> dict[str, Any]:
        """Normalize public kwargs and reject invalid names.

        Rejects removed aliases, unknown names, and (when ``allowed`` is
        given) names that are known on another method but not valid for this
        one (ADR-038 5B). Unrecognized keys raise ``ConfigurationError``
        (ADR-038 D3, fail-fast).
        """
        if not kwargs:
            return {}
        original = dict(kwargs)
        reject_removed_aliases(original)
        reject_removed_guarded_kwargs(original)
        reject_removed_reject_kwargs(original)
        reject_removed_normalization_kwarg(original)
        base = dict(original)
        reject_unknown_public_kwargs(
            base,
            allowed=_KNOWN_PUBLIC_KWARGS,
            surface=surface or "WrapCalibratedExplainer",
        )
        if allowed is None:
            return base
        out_of_scope = sorted(set(base) - allowed)
        if out_of_scope:
            raise ConfigurationError(
                f"{surface or 'WrapCalibratedExplainer'} received keyword arguments that"
                f" are recognized on another method but not valid here: {out_of_scope}.",
                details={
                    "surface": surface or "WrapCalibratedExplainer",
                    "out_of_scope_kwargs": out_of_scope,
                    "allowed_kwargs": sorted(allowed),
                },
            )
        return base

    def _normalize_auto_encode_flag(self) -> str:
        """Return the auto_encode configuration as a telemetry-friendly literal."""
        flag = getattr(self, "_auto_encode", "auto")
        if isinstance(flag, bool):
            return "true" if flag else "false"
        flag_str = str(flag).lower()
        if flag_str in {"true", "false", "auto"}:
            return flag_str
        return "auto"

    def _serialise_preprocessor_value(self, value: Any) -> Any:
        """Convert preprocessing metadata values into JSON-friendly structures."""
        if value is None:
            return None
        if isinstance(value, dict):
            return {str(key): self._serialise_preprocessor_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialise_preprocessor_value(item) for item in value]
        if hasattr(value, "tolist"):
            try:
                return value.tolist()  # numpy/pandas friendly
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                return str(value)
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _extract_preprocessor_snapshot(self, preprocessor: Any) -> dict[str, Any] | None:
        """Build a lightweight snapshot describing the configured preprocessor."""
        snapshot: dict[str, Any] = {}
        getter = getattr(preprocessor, "get_mapping_snapshot", None)
        if callable(getter):
            try:
                custom_snapshot = getter()
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                custom_snapshot = None
            if custom_snapshot is not None:
                snapshot["custom"] = self._serialise_preprocessor_value(custom_snapshot)
        categories = getattr(preprocessor, "categories_", None)
        if categories is not None:
            snapshot["categories"] = self._serialise_preprocessor_value(categories)
        transformers = getattr(preprocessor, "transformers_", None)
        if transformers is not None:
            serialised = []
            for name, transformer, columns in transformers:
                serialised.append(
                    {
                        "name": name,
                        "columns": self._serialise_preprocessor_value(columns),
                        "transformer": (
                            f"{transformer.__class__.__module__}:{transformer.__class__.__qualname__}"
                            if transformer is not None
                            else None
                        ),
                    }
                )
            snapshot["transformers"] = serialised
        feature_names_out = getattr(preprocessor, "get_feature_names_out", None)
        if callable(feature_names_out):
            with suppress(Exception):
                snapshot["feature_names_out"] = list(feature_names_out())
        mapping_attr = getattr(preprocessor, "mapping_", None)
        if mapping_attr is not None:
            snapshot["mapping"] = self._serialise_preprocessor_value(mapping_attr)
        return snapshot or None

    def _build_preprocessor_metadata(self) -> dict[str, Any] | None:
        """Return ADR-009 telemetry metadata for the active preprocessor."""
        auto_encode_flag = self._normalize_auto_encode_flag()
        preprocessor = getattr(self, "_preprocessor", None)
        metadata: dict[str, Any] = {"auto_encode": auto_encode_flag}
        if preprocessor is not None:
            metadata["transformer_id"] = (
                f"{preprocessor.__class__.__module__}:{preprocessor.__class__.__qualname__}"
            )
            snapshot = self._extract_preprocessor_snapshot(preprocessor)
            if snapshot is not None:
                metadata["mapping_snapshot"] = snapshot
        if (
            metadata.get("transformer_id") is None
            and len(metadata) == 1
            and auto_encode_flag == "auto"
        ):
            return None
        return metadata

    def _raise_non_numeric_without_preprocessor(self, x: Any, stage: str) -> None:
        """Raise actionable diagnostics for non-numeric inputs when preprocessing is disabled."""
        auto_encode_flag = self._normalize_auto_encode_flag()
        if auto_encode_flag in {"auto", "true"}:
            return
        x_arr = x.to_numpy() if hasattr(x, "to_numpy") else x
        dtype = getattr(x_arr, "dtype", None)
        if dtype is not None and getattr(dtype, "kind", None) not in {"b", "i", "u", "f", "c"}:
            raise ValidationError(
                f"Non-numeric input detected during {stage} while preprocessing is disabled. "
                "Set auto_encode='auto' or provide a preprocessor capable of handling categorical values."
            )

    def _should_pre_fit_preprocess(self, x: Any) -> bool:
        """Return whether ``x`` must be routed through :meth:`_pre_fit_preprocess`.

        A user-supplied (or previously attached built-in) preprocessor always
        applies. Without one, the ADR-009 built-in path is entered only when it
        has work to do: the input carries non-numeric columns (encoded under
        ``auto_encode='auto'``, rejected with a ``ValidationError`` otherwise)
        or explicit ``categorical_features`` were configured. All-numeric input
        reaches the learner unchanged.
        """
        if self._preprocessor is not None:
            return True
        if self._builtin_categorical_features:
            return True
        numeric_kinds = {"b", "i", "u", "f", "c"}
        if hasattr(x, "columns") and hasattr(x, "dtypes"):
            return any(getattr(dtype, "kind", "O") not in numeric_kinds for dtype in x.dtypes)
        dtype = getattr(x, "dtype", None)
        if dtype is None:
            dtype = np.asarray(x).dtype
        return dtype.kind not in numeric_kinds

    def _pre_fit_preprocess(self, x: Any) -> Any:
        """Fit the configured preprocessor and return transformed x.

        if a user-supplied preprocessor exposes
        fit/transform, we use it. No built-in auto encoding is activated here.

        Raises
        ------
        ValidationError
            If the preprocessor's ``fit``/``fit_transform``/``transform`` call
            fails. Preprocessing failures are never silently bypassed: doing
            so would later feed representation-incompatible raw data into a
            learner trained on transformed features.
        """
        # When no preprocessor is provided and auto_encode is enabled,
        # activate the small deterministic builtin encoder.
        if self._preprocessor is None:
            # ADR-009 default mode: auto_encode='auto' activates deterministic
            # built-in encoding when no user preprocessor is provided.
            if self._normalize_auto_encode_flag() in {"auto", "true"}:
                from calibrated_explanations.preprocessing.builtin_encoder import (
                    BuiltinEncoder,
                )

                encoder = BuiltinEncoder(
                    unseen_policy=self._unseen_category_policy,
                    categorical_features=self._builtin_categorical_features,
                    missing_value_policy=self._missing_value_policy,
                )
                try:
                    x_out = encoder.fit_transform(x)
                except Exception as exc:  # adr002_allow - translated to ValidationError below
                    raise ValidationError(
                        f"Built-in preprocessor failed during fit: {exc}",
                        details={
                            "stage": "fit",
                            "preprocessor_type": type(encoder).__name__,
                            "original_error_type": type(exc).__name__,
                            "original_error": str(exc),
                        },
                    ) from exc
                # attach encoder so export/import helpers can find it
                self._preprocessor = encoder
                self._pre_fitted = True
                return x_out
            self._raise_non_numeric_without_preprocessor(x, stage="fit")
            return x
        try:
            if hasattr(self._preprocessor, "fit_transform"):
                x_out = self._preprocessor.fit_transform(x)
            else:
                self._preprocessor.fit(x)
                x_out = self._preprocessor.transform(x)
        except ValidationError:
            raise
        except Exception as exc:  # adr002_allow - translated to ValidationError below
            raise ValidationError(
                f"Preprocessor failed during fit: {exc}",
                details={
                    "stage": "fit",
                    "preprocessor_type": type(self._preprocessor).__name__,
                    "original_error_type": type(exc).__name__,
                    "original_error": str(exc),
                },
            ) from exc
        self._pre_fitted = True
        return x_out

    def _pre_transform(self, x: Any, stage: str = "predict") -> Any:
        """Transform x with the fitted preprocessor if available.

        Raises
        ------
        ValidationError
            If the fitted preprocessor's ``transform`` call fails. Transform
            failures are never silently bypassed: doing so would feed
            representation-incompatible raw data into a learner/explainer
            trained or calibrated on transformed features.
        """
        if self._preprocessor is None or not self._pre_fitted:
            self._raise_non_numeric_without_preprocessor(x, stage=stage)
            return x
        pre = self._preprocessor
        try:
            return pre.transform(x)
        except Exception as exc:  # adr002_allow - translated to ValidationError below
            unseen_policy = str(getattr(pre, "unseen_policy", "")).lower()
            if isinstance(exc, (KeyError, ValidationError)) and unseen_policy == "error":
                raise ValidationError(
                    f"Unseen category encountered during {stage} preprocessing. "
                    "Set unseen_category_policy='ignore' or import/export a stable mapping.",
                    details={
                        "stage": stage,
                        "preprocessor_type": type(pre).__name__,
                        "original_error_type": type(exc).__name__,
                    },
                ) from exc
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(
                f"Preprocessor transform failed during {stage}: {exc}",
                details={
                    "stage": stage,
                    "preprocessor_type": type(pre).__name__,
                    "original_error_type": type(exc).__name__,
                    "original_error": str(exc),
                },
            ) from exc

    def _maybe_preprocess_for_inference(self, x: Any) -> Any:
        """Apply preprocessing for inference paths if configured/fitted."""
        return self._pre_transform(x, stage="inference")

    def _finalize_fit(self, reinitialize: bool) -> WrapCalibratedExplainer:
        """Finalize fit logic shared across fit implementations.

        Parameters
        ----------
        reinitialize : bool
            Whether an existing calibrated explainer should be reinitialized.
        """
        check_is_fitted(self.learner)
        self.fitted = True
        if reinitialize and self.explainer is not None:
            # Preserve calibration by updating underlying learner reference
            self.explainer.reinitialize(self.learner)
            self.calibrated = True
        return self

    def _snapshot_calibration_state(self) -> dict[str, Any]:
        """Capture wrapper state that must survive a rejected recalibration."""
        preprocessor_snapshot = self._preprocessor
        if preprocessor_snapshot is not None and not self._pre_fitted:
            with suppress(Exception):  # pragma: no cover - best-effort rollback snapshot
                preprocessor_snapshot = copy.deepcopy(preprocessor_snapshot)
        return {
            "calibrated": self.calibrated,
            "explainer": self.explainer,
            "mc": self.mc,
            "preprocessor": preprocessor_snapshot,
            "pre_fitted": self._pre_fitted,
        }

    def _restore_calibration_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore wrapper state after a rejected recalibration attempt."""
        self.calibrated = bool(snapshot["calibrated"])
        self.explainer = snapshot["explainer"]
        self.mc = snapshot["mc"]
        self._preprocessor = snapshot["preprocessor"]
        self._pre_fitted = bool(snapshot["pre_fitted"])

    def _finalize_candidate_calibration(
        self,
        candidate_explainer: CalibratedExplainer,
        preprocessor_metadata: Mapping[str, Any] | None,
    ) -> None:
        """Apply wrapper-owned runtime configuration to a candidate explainer."""
        if hasattr(self, "_feature_filter_config"):
            candidate_explainer.feature_filter_config = self._feature_filter_config
        if preprocessor_metadata is not None:
            with suppress(AttributeError):
                candidate_explainer.set_preprocessor_metadata(preprocessor_metadata)

    def _format_proba_output(self, proba: Any, uq_interval: bool) -> Any:
        """Format probability output (with optional trivial intervals) without duplicating logic."""
        if not uq_interval:
            return proba
        # Multiclass: return matrix and identical bounds
        if proba.ndim == 2 and proba.shape[1] > 2:
            return proba, (proba, proba)
        # Binary (assume second column is positive class probability)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba, (proba[:, 1], proba[:, 1])
        # Fallback (unexpected shape) -> mirror array
        return proba, (proba, proba)

    def export_preprocessor_mapping(self) -> dict[str, Any] | None:
        """Export the current preprocessor mapping snapshot.

        Returns
        -------
        dict[str, Any] | None
            A mapping snapshot suitable for telemetry or round-tripping, or
            ``None`` when no mapping information is available.
        """
        pre = getattr(self, "_preprocessor", None)
        if pre is None:
            return None
        # Prefer a custom getter when available
        getter = getattr(pre, "get_mapping_snapshot", None)
        if callable(getter):
            try:
                snapshot = getter()
                if snapshot is not None:
                    if not isinstance(snapshot, Mapping):
                        raise ValidationError(
                            "Preprocessor mapping snapshot must be a mapping.",
                            details={"source": "get_mapping_snapshot"},
                        )
                    self._validate_json_safe_mapping(snapshot, source="get_mapping_snapshot")
                    return dict(snapshot)
                return None
            except ValidationError:
                raise
            except (AttributeError, TypeError, ValueError):
                self._logger.warning(
                    "Preprocessor.get_mapping_snapshot failed; falling back to mapping_"
                )
        # Fall back to attribute if present
        mapping_attr = getattr(pre, "mapping_", None)
        if mapping_attr is not None:
            # Shallow copy to avoid exposing internal objects
            try:
                snapshot = dict(mapping_attr)
                self._validate_json_safe_mapping(snapshot, source="mapping_")
                return snapshot
            except ValidationError:
                raise
            except (AttributeError, TypeError, ValueError):
                return None
        return None

    def import_preprocessor_mapping(self, mapping: Mapping[str, Any]) -> None:
        """Attempt to apply a mapping snapshot to the configured preprocessor.

        This is a best-effort helper: when an attached preprocessor exposes a
        setter (``set_mapping``) or a writable ``mapping_`` attribute we will
        apply the mapping. Otherwise the mapping is stashed on the wrapper as
        ``_imported_preprocessor_mapping`` for potential downstream use.

        A warning is emitted when the mapping could not be applied to ensure
        visibility per the fallback policy.
        """
        self._validate_json_safe_mapping(mapping, source="import")
        pre = getattr(self, "_preprocessor", None)
        applied = False
        if pre is not None:
            setter = getattr(pre, "set_mapping", None)
            if callable(setter):
                try:
                    setter(mapping)
                    applied = True
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    self._logger.warning("Preprocessor.set_mapping failed; stashing mapping")
            else:
                # Try to set mapping_ directly when writable
                try:
                    pre.mapping_ = mapping
                    applied = True
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    # fall through to stashing below
                    pass
        if not applied:
            # Keep for later application or external tooling
            self._imported_preprocessor_mapping = dict(mapping) if mapping is not None else None
            _warnings.warn(
                "Preprocessor mapping could not be applied directly; mapping stashed on wrapper",
                UserWarning,
                stacklevel=2,
            )

    @staticmethod
    def _validate_json_safe_mapping(mapping: Mapping[str, Any], *, source: str) -> None:
        """Validate that mapping snapshots are JSON-serialisable primitives.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Mapping snapshot to validate.
        source : str
            Context string used in validation error details.

        Raises
        ------
        ValidationError
            If the mapping cannot be serialised with standard JSON encoding.
        """
        try:
            json.dumps(mapping, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "Preprocessor mapping must be JSON-serialisable.",
                details={"source": source, "error": str(exc)},
            ) from exc

    def _state_path(self, path_or_fileobj: Any) -> Path:
        """Normalize and validate state path inputs."""
        if hasattr(path_or_fileobj, "read") or hasattr(path_or_fileobj, "write"):
            raise ValidationError(
                "Only filesystem paths are supported for state persistence.",
                details={"path_or_fileobj_type": type(path_or_fileobj).__name__},
            )
        try:
            return Path(path_or_fileobj)
        except TypeError as exc:
            raise ValidationError(
                "Invalid state path provided to save/load_state.",
                details={"path_or_fileobj_type": type(path_or_fileobj).__name__},
            ) from exc

    @staticmethod
    def _sha256_bytes(payload: bytes) -> str:
        """Return SHA-256 checksum for raw bytes."""
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Return SHA-256 checksum for a file."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _calibrator_to_primitive(self, calibrator: Any) -> dict[str, Any]:
        """Serialize a single calibrator into the ADR-031 JSON-safe primitive contract.

        Raises
        ------
        SerializationError
            If ``calibrator`` does not implement a JSON-safe ``to_primitive()``
            contract. ``save_state()`` never falls back to pickling an
            unsupported calibrator: arbitrary Python object serialization is
            intentionally excluded from the state-persistence trust boundary
            (ADR-031). To persist a custom calibrator, implement
            ``to_primitive()``/``from_primitive()`` following the built-in
            ``VennAbers``/``IntervalRegressor`` contracts and register the
            calibrator type with ``WrapCalibratedExplainer`` restoration, or
            omit calibrator persistence and recalibrate after ``load_state()``.
        """
        to_primitive = getattr(calibrator, "to_primitive", None)
        if callable(to_primitive):
            primitive = to_primitive()
            if isinstance(primitive, Mapping):
                return dict(primitive)
        raise SerializationError(
            f"Calibrator type '{type(calibrator).__module__}.{type(calibrator).__qualname__}' "
            "does not implement a JSON-safe to_primitive() contract and cannot be persisted "
            "by save_state(). Arbitrary Python object serialization (pickle) is intentionally "
            "excluded from the state-persistence trust boundary (ADR-031). Implement "
            "to_primitive()/from_primitive() on this calibrator to persist it safely, or "
            "recalibrate the wrapper after load_state() instead of relying on persistence "
            "for this calibrator.",
            details={
                "calibrator_type": type(calibrator).__qualname__,
                "calibrator_module": type(calibrator).__module__,
            },
        )

    def _build_calibrator_primitive(self) -> dict[str, Any] | None:
        """Build calibrator primitive payload from the active explainer, if any."""
        explainer = getattr(self, "explainer", None)
        if explainer is None:
            return None
        calibrator = getattr(explainer, "interval_learner", None)
        if calibrator is None:
            return None
        if isinstance(calibrator, SequenceABC) and not isinstance(calibrator, (str, bytes)):
            children = [self._calibrator_to_primitive(item) for item in calibrator]
            payload_bytes = json.dumps(children, sort_keys=True).encode("utf-8")
            return {
                "schema_version": self._STATE_SCHEMA_VERSION,
                "calibrator_type": "fast_collection",
                "parameters": {"size": len(children)},
                "checksums": {"sha256": self._sha256_bytes(payload_bytes)},
                "calibrators": children,
            }
        return self._calibrator_to_primitive(calibrator)

    @classmethod
    def _restore_calibrator_from_primitive(cls, primitive: Mapping[str, Any]) -> Any:
        """Rehydrate a calibrator object from a persisted JSON-safe primitive payload.

        ``calibrator_type`` is dispatched against a fixed, explicit set of
        trusted restorers -- never resolved via an artifact-provided module
        path or dynamic import. Legacy ``python_pickle`` payloads are rejected
        unconditionally, before any base64 decoding, checksum comparison, or
        deserialization is attempted (ADR-031 security hardening).
        """
        if not isinstance(primitive, Mapping):
            raise IncompatibleStateError(
                "Malformed calibrator primitive: expected a JSON object.",
                details={"actual_type": type(primitive).__name__},
            )
        calibrator_type = primitive.get("calibrator_type")
        # This check must stay first and must not touch payload/checksum
        # fields: rejecting on the type name alone is what guarantees a
        # malicious pickle payload is never decoded, even if its checksum
        # was recomputed to match. Checksums prove internal consistency, not
        # that an artifact is safe to deserialize.
        if calibrator_type == "python_pickle":
            raise IncompatibleStateError(
                "Persisted calibrator primitive declares calibrator_type='python_pickle'. "
                "This legacy format executed arbitrary code via pickle.loads() and is no "
                "longer supported; the payload is rejected here without being decoded or "
                "unpickled. Re-fit this calibrator (fit()+calibrate()) or migrate it to a "
                "safe to_primitive()/from_primitive() contract before persisting again.",
                details={"calibrator_type": calibrator_type},
            )
        if calibrator_type == "venn_abers":
            from ..calibration.venn_abers import VennAbers

            return VennAbers.from_primitive(primitive)
        if calibrator_type == "interval_regressor":
            from ..calibration.interval_regressor import IntervalRegressor

            return IntervalRegressor.from_primitive(primitive)
        if calibrator_type == "fast_collection":
            children = primitive.get("calibrators")
            if not isinstance(children, list):
                raise IncompatibleStateError(
                    "Invalid fast_collection primitive: expected calibrators list.",
                    details={"field": "calibrators"},
                )
            expected_sha = primitive.get("checksums", {}).get("sha256")
            child_bytes = json.dumps(children, sort_keys=True).encode("utf-8")
            actual_sha = cls._sha256_bytes(child_bytes)
            if not isinstance(expected_sha, str) or expected_sha != actual_sha:
                raise IncompatibleStateError(
                    "Calibrator primitive checksum validation failed: this detects "
                    "corruption of the fast_collection payload only.",
                    details={"expected_sha256": expected_sha, "actual_sha256": actual_sha},
                )
            return [cls._restore_calibrator_from_primitive(item) for item in children]
        raise IncompatibleStateError(
            "Unsupported or unknown calibrator_type in persisted state. Calibrator types "
            "are resolved only through a fixed set of trusted restorers; unrecognized "
            "types fail closed rather than being resolved via dynamic import.",
            details={"calibrator_type": calibrator_type},
        )

    @staticmethod
    def _json_safe_scalar(value: Any) -> Any:
        """Convert a single (possibly numpy) scalar into a JSON-safe primitive."""
        if hasattr(value, "item"):
            with suppress(Exception):
                return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _coerce_int_keys(mapping: Any) -> Any:
        """Best-effort restore of integer dict keys lost to JSON's string keys."""
        if not isinstance(mapping, Mapping):
            return mapping
        result: dict[Any, Any] = {}
        for key, value in mapping.items():
            try:
                result[int(key)] = value
            except (TypeError, ValueError):
                result[key] = value
        return result

    @staticmethod
    def _extract_original_y_cal(explainer: Any) -> list[Any]:
        """Recover the original (pre-encoding) calibration targets for persistence.

        ``CalibratedExplainer`` overwrites ``y_cal`` with numerically-encoded
        labels during construction; ``label_map`` records the encoding so it
        can be inverted here, keeping a save/load round trip equivalent to
        calling ``calibrate()`` again with the original labels.
        """
        y_cal = np.asarray(getattr(explainer, "y_cal", None))
        label_map = getattr(explainer, "label_map", None)
        if isinstance(label_map, Mapping) and label_map:
            inverse = {int(encoded): original for original, encoded in label_map.items()}
            originals = [inverse.get(int(v), v) for v in y_cal]
            return [WrapCalibratedExplainer._json_safe_scalar(v) for v in originals]
        return [WrapCalibratedExplainer._json_safe_scalar(v) for v in y_cal]

    def _build_learner_descriptor(self) -> dict[str, Any]:
        """Build a JSON-safe descriptor used to validate a caller-supplied learner.

        Only identity/shape metadata is recorded here -- never executable
        learner bytes. ``load_state()`` uses this to fail clearly when a
        caller-supplied learner is incompatible, but it never imports or
        instantiates anything from these fields.
        """
        learner = self.learner
        task = "classification" if "predict_proba" in dir(learner) else "regression"
        n_features = None
        classes: list[Any] | None = None
        explainer = getattr(self, "explainer", None)
        if explainer is not None:
            x_cal = getattr(explainer, "x_cal", None)
            if x_cal is not None:
                n_features = int(np.asarray(x_cal).shape[1])
            if task == "classification":
                # Use the raw original class values (not the display-oriented,
                # already-stringified ``class_labels`` mapping) so comparison
                # against a supplied learner's ``classes_`` isn't thrown off by
                # incidental string formatting differences (e.g. "0" vs "0.0").
                original_classes = getattr(explainer, "original_class_values", None)
                if original_classes is not None:
                    classes = list(np.asarray(original_classes))
        return {
            "module": type(learner).__module__,
            "qualname": type(learner).__qualname__,
            "task": task,
            "n_features": n_features,
            "classes": self._serialise_preprocessor_value(classes) if classes is not None else None,
        }

    def _build_preprocessor_state_payload(self) -> dict[str, Any]:
        """Describe the configured preprocessor for persistence.

        Only the built-in, fully JSON-safe ``BuiltinEncoder`` can be
        reconstructed from persisted data alone (its mapping snapshot is
        already exported separately). Any other preprocessor is recorded by
        identity only; ``load_state()`` requires the caller to supply the
        original instance again via ``preprocessor=``.
        """
        preprocessor = self._preprocessor
        if preprocessor is None:
            return {"kind": "none"}
        transformer_id = f"{type(preprocessor).__module__}:{type(preprocessor).__qualname__}"
        if transformer_id == "calibrated_explanations.preprocessing.builtin_encoder:BuiltinEncoder":
            return {
                "kind": "builtin",
                "transformer_id": transformer_id,
                "unseen_policy": getattr(preprocessor, "unseen_policy", "error"),
                "categorical_features": list(
                    getattr(preprocessor, "categorical_features_", ()) or ()
                ),
                "missing_value_policy": getattr(preprocessor, "missing_value_policy", "category"),
                "pre_fitted": bool(self._pre_fitted),
            }
        return {
            "kind": "custom",
            "transformer_id": transformer_id,
            "pre_fitted": bool(self._pre_fitted),
        }

    def _build_calibration_state_payload(self) -> dict[str, Any] | None:
        """Build the JSON-safe declarative calibration state for persistence."""
        explainer = getattr(self, "explainer", None)
        if explainer is None:
            return None
        x_cal = np.asarray(explainer.x_cal)
        bins = getattr(explainer, "bins", None)
        bins_list = (
            bins.tolist() if hasattr(bins, "tolist") else (list(bins) if bins is not None else None)
        )
        interval_summary = getattr(explainer, "interval_summary", None)
        interval_summary_value = getattr(interval_summary, "value", interval_summary)
        plugin_manager = getattr(explainer, "_plugin_manager", None)
        return {
            "mode": getattr(explainer, "mode", None),
            "seed": getattr(explainer, "seed", None),
            "condition_source": getattr(explainer, "condition_source", None),
            "interval_summary": interval_summary_value,
            "feature_names": list(getattr(explainer, "feature_names", None) or []),
            "categorical_features": list(getattr(explainer, "categorical_features", None) or []),
            "categorical_labels": self._serialise_preprocessor_value(
                getattr(explainer, "categorical_labels", None)
            ),
            "class_labels": self._serialise_preprocessor_value(
                getattr(explainer, "class_labels", None)
            ),
            "bins": bins_list,
            "x_cal": x_cal.tolist(),
            "y_cal": self._extract_original_y_cal(explainer),
            "difficulty_estimator_required": getattr(explainer, "difficulty_estimator", None)
            is not None,
            "preprocessor_metadata": self._serialise_preprocessor_value(
                getattr(explainer, "_preprocessor_metadata", None)
            ),
            "plugin_overrides": self._serialise_preprocessor_value(
                getattr(plugin_manager, "plugin_overrides", None)
            ),
            # FAST-explanation tuning knobs (ADR-003/ADR-004 surface); persisted so
            # explain_fast() behavior after load_state() matches the saved wrapper
            # instead of silently resetting to defaults.
            "fast": bool(getattr(explainer, "_fast", False)),
            "noise_type": getattr(explainer, "_noise_type", None),
            "scale_factor": getattr(explainer, "_scale_factor", None),
            "severity": getattr(explainer, "_severity", None),
            "sample_percentiles": list(getattr(explainer, "sample_percentiles", None) or []),
            "features_to_ignore": list(getattr(explainer, "features_to_ignore", None) or []),
        }

    def _build_state_payload(self) -> dict[str, Any]:
        """Build the full JSON-safe ``explainer_state.json`` payload (schema v3)."""
        return {
            "schema_version": self._STATE_SCHEMA_VERSION,
            "wrapper": {
                "auto_encode": self._normalize_auto_encode_flag(),
                "unseen_category_policy": self._unseen_category_policy,
                "builtin_categorical_features": list(self._builtin_categorical_features),
                "missing_value_policy": self._missing_value_policy,
            },
            "learner": self._build_learner_descriptor(),
            "preprocessor": self._build_preprocessor_state_payload(),
            "calibration": self._build_calibration_state_payload(),
        }

    def save_state(self, path_or_fileobj: Any) -> Path:
        """Persist wrapper state as a safe (schema v3) ADR-031 manifest + checksums.

        The written artifact contains only JSON-safe declarative data (no
        pickled wrapper or calibrator bytes): built-in calibrator primitives,
        preprocessing mapping snapshots, and configuration needed to
        reconstruct calibrated prediction behavior. It never stores an
        executable representation of the learner or a custom preprocessor --
        ``load_state()`` requires those back from the caller. See ADR-031 for
        the full trust-boundary rationale.

        Raises
        ------
        SerializationError
            If the active calibrator does not implement a JSON-safe
            ``to_primitive()`` contract. This never falls back to pickling.
        """
        self._warn_dropping_mondrian_categorizer(operation="save_state")
        target = self._state_path(path_or_fileobj)
        target_parent = target.parent
        target_parent.mkdir(parents=True, exist_ok=True)

        temp_dir_name = f"{target.name}.tmp-{os.getpid()}-{id(self)}"
        temp_dir = Path(tempfile.mkdtemp(prefix=temp_dir_name, dir=str(target_parent)))
        checksums: dict[str, str] = {}
        try:
            state_payload = self._build_state_payload()
            state_bytes = json.dumps(state_payload, indent=2, sort_keys=True).encode("utf-8")
            (temp_dir / "explainer_state.json").write_bytes(state_bytes)
            checksums["explainer_state.json"] = self._sha256_bytes(state_bytes)

            calibrator_primitive = self._build_calibrator_primitive()
            if calibrator_primitive is not None:
                calibrator_bytes = json.dumps(
                    calibrator_primitive, indent=2, sort_keys=True
                ).encode("utf-8")
                (temp_dir / "calibrator_primitive.json").write_bytes(calibrator_bytes)
                checksums["calibrator_primitive.json"] = self._sha256_bytes(calibrator_bytes)

            mapping = self.export_preprocessor_mapping()
            if mapping is not None:
                mapping_bytes = json.dumps(mapping, indent=2, sort_keys=True).encode("utf-8")
                (temp_dir / "preprocessing_mapping.json").write_bytes(mapping_bytes)
                checksums["preprocessing_mapping.json"] = self._sha256_bytes(mapping_bytes)

            manifest = {
                "schema_version": self._STATE_SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifact_type": "wrap_calibrated_explainer_state",
                "files": checksums,
            }
            manifest_file = temp_dir / "manifest.json"
            manifest_file.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )

            backup: Path | None = None
            if target.exists():
                backup = target.with_name(f"{target.name}.bak-{os.getpid()}-{id(self)}")
                os.replace(target, backup)
            try:
                replaced = False
                last_permission_error: PermissionError | None = None
                for _ in range(3):
                    try:
                        os.replace(temp_dir, target)
                        replaced = True
                        break
                    except PermissionError as exc:
                        last_permission_error = exc
                        sleep(0.05)
                if not replaced:
                    if last_permission_error is not None:
                        self._logger.debug(
                            "os.replace failed during save_state; falling back to shutil.move: %s",
                            last_permission_error,
                        )
                    shutil.move(str(temp_dir), str(target))
            except OSError:
                if backup is not None and backup.exists() and not target.exists():
                    os.replace(backup, target)
                raise
            if backup is not None and backup.exists():
                shutil.rmtree(backup)
            return target
        except SerializationError:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except (OSError, TypeError, ValueError, AttributeError) as exc:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValidationError(
                f"Failed to save state to '{target}'.",
                details={"path": str(target), "reason": str(exc)},
            ) from exc

    def _warn_dropping_mondrian_categorizer(self, *, operation: str) -> None:
        """Warn and log when persistence drops a configured Mondrian categorizer."""
        if self.mc is None:
            return
        message = (
            f"{operation} drops the configured Mondrian categorizer (mc). "
            "Loaded conditional wrappers require explicit bins= at inference."
        )
        self._logger.info(message)
        _warnings.warn(message, UserWarning, stacklevel=3)

    @classmethod
    def _read_manifest_json(cls, path: Path) -> dict[str, Any]:
        """Read and structurally validate ``manifest.json`` before trusting any field."""
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise IncompatibleStateError(
                "State artifact is missing manifest.json.",
                details={"path": str(path)},
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise IncompatibleStateError(
                "State artifact manifest.json is not valid JSON.",
                details={"path": str(manifest_path), "error": str(exc)},
            ) from exc
        if not isinstance(manifest, Mapping):
            raise IncompatibleStateError(
                "Invalid state manifest: expected a JSON object at the top level.",
                details={"actual_type": type(manifest).__name__},
            )
        return dict(manifest)

    @classmethod
    def _read_json_object(cls, file_path: Path) -> dict[str, Any]:
        """Read a JSON file and validate it decodes to an object, not another type."""
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise IncompatibleStateError(
                f"State artifact file '{file_path.name}' is not valid JSON.",
                details={"file": file_path.name, "error": str(exc)},
            ) from exc
        if not isinstance(payload, Mapping):
            raise IncompatibleStateError(
                f"Invalid state file '{file_path.name}': expected a JSON object at the top level.",
                details={"file": file_path.name, "actual_type": type(payload).__name__},
            )
        return dict(payload)

    @classmethod
    def _validate_and_verify_manifest_files(
        cls, manifest: Mapping[str, Any], root: Path
    ) -> dict[str, str]:
        """Validate the manifest's file inventory and verify each file's checksum.

        Hardens the artifact-parsing boundary: rejects absolute paths, ``..``
        traversal, duplicate/malformed entries, files outside the safe-schema
        allow-list (e.g. a pickled ``wrapper.pkl``), symlinks resolving
        outside the artifact root, and any on-disk file not declared in the
        manifest -- all before any file content is interpreted as anything
        other than raw bytes for hashing. A checksum match here only proves
        the file's bytes agree with the manifest; it does not establish who
        produced the artifact.
        """
        files = manifest.get("files")
        if not isinstance(files, Mapping):
            raise IncompatibleStateError(
                "Invalid state manifest: files checksum mapping missing.",
                details={"field": "files"},
            )
        resolved_root = root.resolve()
        seen_resolved: set[Path] = set()
        validated: dict[str, str] = {}
        for file_name, expected_sha in files.items():
            if not isinstance(file_name, str) or not isinstance(expected_sha, str):
                raise IncompatibleStateError(
                    "Invalid state manifest: malformed checksum entry.",
                    details={"file": file_name, "checksum": expected_sha},
                )
            if file_name not in cls._SAFE_STATE_FILES:
                raise IncompatibleStateError(
                    f"Invalid state manifest: '{file_name}' is not part of the safe "
                    "persistence schema. Executable or binary payload files (such as a "
                    "pickled 'wrapper.pkl') are never accepted, even with a matching "
                    "checksum -- checksums only detect corruption, they do not "
                    "authenticate an artifact's origin.",
                    details={"file": file_name, "allowed_files": sorted(cls._SAFE_STATE_FILES)},
                )
            candidate = Path(file_name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise IncompatibleStateError(
                    "Invalid state manifest: file paths must be relative names without "
                    "traversal segments.",
                    details={"file": file_name},
                )
            file_path = root / candidate
            if not file_path.exists():
                raise IncompatibleStateError(
                    "State artifact is incomplete: expected file is missing.",
                    details={"file": file_name},
                )
            resolved = file_path.resolve()
            if resolved != resolved_root and resolved_root not in resolved.parents:
                raise IncompatibleStateError(
                    "Invalid state manifest: file resolves outside the artifact directory.",
                    details={"file": file_name},
                )
            if resolved in seen_resolved:
                raise IncompatibleStateError(
                    "Invalid state manifest: duplicate file entry after path normalization.",
                    details={"file": file_name},
                )
            seen_resolved.add(resolved)
            actual_sha = cls._sha256_file(file_path)
            if actual_sha != expected_sha:
                raise IncompatibleStateError(
                    "State checksum validation failed: file contents do not match the "
                    "manifest. This detects corruption or tampering only -- a matching "
                    "checksum does not prove the artifact came from a trusted source.",
                    details={
                        "file": file_name,
                        "expected_sha256": expected_sha,
                        "actual_sha256": actual_sha,
                    },
                )
            validated[file_name] = expected_sha

        # Defense in depth: reject any on-disk file that was not declared in
        # the manifest at all (e.g. a pickle payload dropped alongside a
        # manifest claiming the safe schema), and reject symlinks escaping
        # the artifact root even when their name was never listed.
        for entry in root.iterdir():
            if entry.name == "manifest.json":
                continue
            if entry.is_symlink():
                link_target = entry.resolve()
                if resolved_root not in link_target.parents and link_target != resolved_root:
                    raise IncompatibleStateError(
                        "Invalid state artifact: symlink resolves outside the artifact directory.",
                        details={"file": entry.name},
                    )
            if entry.name not in validated:
                raise IncompatibleStateError(
                    f"Invalid state artifact: unexpected file '{entry.name}' is present but "
                    "not declared in the manifest. Safe-schema artifacts only contain the "
                    "files they declare.",
                    details={"file": entry.name},
                )
        return validated

    @staticmethod
    def _validate_supplied_learner(learner: Any, learner_meta: Mapping[str, Any]) -> None:
        """Validate a caller-supplied learner against persisted identity/shape metadata.

        Raises
        ------
        ValidationError
            If no learner was supplied, or if the supplied learner's task,
            feature count, or classes are incompatible with the persisted
            state. Validation happens before the learner is used to
            reconstruct anything.
        """
        if not learner_meta:
            return
        if learner is None:
            required = f"{learner_meta.get('module')}.{learner_meta.get('qualname')}"
            raise ValidationError(
                "load_state() requires a caller-supplied fitted learner for this artifact: "
                "the safe persistence schema never stores executable learner bytes. Pass "
                f"learner=<your fitted {required} instance> (or an equivalent fitted "
                "estimator) matching the model used when save_state() was called.",
                details={"required_learner": required},
            )
        check_is_fitted(learner)
        task = "classification" if "predict_proba" in dir(learner) else "regression"
        expected_task = learner_meta.get("task")
        if expected_task is not None and task != expected_task:
            raise ValidationError(
                f"Supplied learner exposes task '{task}' but the persisted state requires "
                f"'{expected_task}'. Supply a learner matching the original task.",
                details={"supplied_task": task, "required_task": expected_task},
            )
        expected_features = learner_meta.get("n_features")
        supplied_features = getattr(learner, "n_features_in_", None)
        if (
            expected_features is not None
            and supplied_features is not None
            and int(supplied_features) != int(expected_features)
        ):
            raise ValidationError(
                f"Supplied learner expects {supplied_features} feature(s) but the persisted "
                f"state was calibrated with {expected_features} feature(s).",
                details={
                    "supplied_n_features": int(supplied_features),
                    "required_n_features": int(expected_features),
                },
            )
        if task == "classification":
            expected_classes = learner_meta.get("classes")
            supplied_classes = getattr(learner, "classes_", None)
            if expected_classes is not None and supplied_classes is not None:
                supplied_norm = sorted(str(c) for c in supplied_classes)
                expected_norm = sorted(str(c) for c in expected_classes)
                if supplied_norm != expected_norm:
                    raise ValidationError(
                        "Supplied learner's classes_ do not match the persisted calibration "
                        "classes.",
                        details={
                            "supplied_classes": supplied_norm,
                            "required_classes": expected_norm,
                        },
                    )

    @staticmethod
    def _restore_preprocessor(
        wrapper: "WrapCalibratedExplainer",
        preprocessor_state: Mapping[str, Any],
        supplied_preprocessor: Any,
        mapping_payload: Mapping[str, Any] | None,
    ) -> None:
        """Attach a preprocessor to ``wrapper`` per the persisted preprocessor descriptor."""
        kind = preprocessor_state.get("kind", "none")
        if kind == "none":
            wrapper._preprocessor = supplied_preprocessor
            wrapper._pre_fitted = False
            return
        if kind == "builtin":
            from ..preprocessing.builtin_encoder import BuiltinEncoder

            encoder = BuiltinEncoder(
                unseen_policy=preprocessor_state.get("unseen_policy", "error"),
                categorical_features=tuple(preprocessor_state.get("categorical_features", ())),
                missing_value_policy=preprocessor_state.get("missing_value_policy", "category"),
            )
            if mapping_payload is not None:
                encoder.set_mapping(dict(mapping_payload))
            wrapper._preprocessor = encoder
            wrapper._pre_fitted = bool(preprocessor_state.get("pre_fitted", True))
            return
        if kind == "custom":
            transformer_id = preprocessor_state.get("transformer_id")
            if supplied_preprocessor is None:
                raise ValidationError(
                    f"This artifact was saved with a custom preprocessor ({transformer_id}) "
                    "that cannot be safely reconstructed from persisted data alone. Supply "
                    "the original fitted preprocessor instance via "
                    "load_state(..., preprocessor=...).",
                    details={"required_preprocessor": transformer_id},
                )
            supplied_id = (
                f"{type(supplied_preprocessor).__module__}:"
                f"{type(supplied_preprocessor).__qualname__}"
            )
            if supplied_id != transformer_id:
                raise ValidationError(
                    "Supplied preprocessor type does not match the persisted metadata.",
                    details={"supplied": supplied_id, "required": transformer_id},
                )
            wrapper._preprocessor = supplied_preprocessor
            wrapper._pre_fitted = bool(preprocessor_state.get("pre_fitted", True))
            if mapping_payload is not None:
                wrapper.import_preprocessor_mapping(dict(mapping_payload))
            return
        raise IncompatibleStateError(
            "Unknown preprocessor kind in persisted state.", details={"kind": kind}
        )

    @classmethod
    def load_state(
        cls,
        path_or_fileobj: Any,
        *,
        learner: Any | None = None,
        preprocessor: Any | None = None,
        difficulty_estimator: Any | None = None,
        mc: Callable[[Any], Any] | MondrianCategorizer | None = None,
    ) -> WrapCalibratedExplainer:
        """Load wrapper state from a safe (schema v3) ADR-031 persisted artifact.

        The safe schema never contains pickled wrapper or calibrator bytes,
        so normal loading cannot execute artifact-provided code. Because of
        that, some runtime state cannot be reconstructed from the artifact
        alone and must be supplied by the caller:

        Parameters
        ----------
        path_or_fileobj : path-like
            Directory produced by :meth:`save_state`.
        learner : Any, optional
            The original (or an equivalent, already-fitted) learner. Required
            whenever the persisted state includes calibration; validated
            against persisted task/feature-count/classes metadata.
        preprocessor : Any, optional
            Required only when the artifact was saved with a *custom*
            (non-built-in) preprocessor; built-in preprocessing is
            reconstructed automatically from its JSON-safe mapping.
        difficulty_estimator : Any, optional
            Required when the artifact was calibrated with a difficulty
            estimator.
        mc : callable or MondrianCategorizer, optional
            Optional Mondrian categorizer to attach for future
            ``reuse_conditional=True`` calibration calls; not needed to
            restore predictions, which rely on the persisted ``bins``.

        Raises
        ------
        IncompatibleStateError
            If the artifact uses a legacy (pickle-based) schema version, has
            an unsupported/invalid manifest, fails checksum verification, or
            contains disallowed files/paths. Legacy artifacts are rejected
            unconditionally and with actionable migration guidance -- see the
            raised message for details.
        ValidationError
            If a required runtime object (learner, custom preprocessor,
            difficulty estimator) is missing or incompatible with the
            persisted state.
        """
        temp_instance = cls.__new__(cls)
        path = temp_instance._state_path(path_or_fileobj)
        manifest = cls._read_manifest_json(path)
        schema_version = manifest.get("schema_version")
        if schema_version in cls._LEGACY_STATE_SCHEMA_VERSIONS:
            raise IncompatibleStateError(
                f"This state artifact uses legacy schema_version {schema_version!r}, which "
                "persisted the entire wrapper (and, for unsupported calibrators, arbitrary "
                "Python objects) using pickle. Loading it would allow arbitrary code "
                "execution: the SHA-256 checksums recorded alongside it only detect "
                "corruption, they do not authenticate who created the file, so a matching "
                "checksum cannot make an untrusted pickle artifact safe to load. Normal "
                "load_state() therefore refuses legacy artifacts unconditionally, before any "
                "pickle byte is read. To migrate: open this artifact with a trusted, older "
                "calibrated-explanations environment you control (ideally the version that "
                "produced it), then call save_state() again there to produce a "
                f"schema_version={cls._STATE_SCHEMA_VERSION} artifact. Only do this for "
                "artifacts you created yourself and still trust -- never run this migration "
                "against a downloaded or otherwise untrusted artifact.",
                details={
                    "schema_version": schema_version,
                    "supported_versions": [cls._STATE_SCHEMA_VERSION],
                },
            )
        if schema_version != cls._STATE_SCHEMA_VERSION:
            raise IncompatibleStateError(
                "Unsupported state schema_version.",
                details={
                    "schema_version": schema_version,
                    "supported_versions": [cls._STATE_SCHEMA_VERSION],
                },
            )
        artifact_type = manifest.get("artifact_type")
        if artifact_type != "wrap_calibrated_explainer_state":
            raise IncompatibleStateError(
                "Invalid state manifest: unexpected artifact_type.",
                details={"artifact_type": artifact_type},
            )

        validated_files = cls._validate_and_verify_manifest_files(manifest, path)
        if "explainer_state.json" not in validated_files:
            raise IncompatibleStateError(
                "State artifact is incomplete: manifest does not list explainer_state.json.",
                details={"field": "files"},
            )

        state = cls._read_json_object(path / "explainer_state.json")
        if state.get("schema_version") != cls._STATE_SCHEMA_VERSION:
            raise IncompatibleStateError(
                "Unsupported explainer_state schema_version.",
                details={
                    "schema_version": state.get("schema_version"),
                    "supported_versions": [cls._STATE_SCHEMA_VERSION],
                },
            )

        wrapper = cls.__new__(cls)
        wrapper._logger = _logging.getLogger(__name__)
        wrapper.mc = mc
        wrapper._preprocessor = None
        wrapper._pre_fitted = False
        wrapper_meta = state.get("wrapper")
        wrapper_meta = wrapper_meta if isinstance(wrapper_meta, Mapping) else {}
        wrapper._auto_encode = wrapper_meta.get("auto_encode", "auto")
        wrapper._unseen_category_policy = wrapper_meta.get("unseen_category_policy", "error")
        wrapper._builtin_categorical_features = tuple(
            int(feature) for feature in wrapper_meta.get("builtin_categorical_features", ()) or ()
        )
        wrapper._missing_value_policy = wrapper_meta.get("missing_value_policy", "category")

        learner_meta = state.get("learner")
        learner_meta = learner_meta if isinstance(learner_meta, Mapping) else {}
        cls._validate_supplied_learner(learner, learner_meta)
        wrapper.learner = learner
        wrapper.fitted = True

        mapping_payload: dict[str, Any] | None = None
        if "preprocessing_mapping.json" in validated_files:
            mapping_payload = cls._read_json_object(path / "preprocessing_mapping.json")
        preprocessor_state = state.get("preprocessor")
        preprocessor_state = (
            preprocessor_state if isinstance(preprocessor_state, Mapping) else {"kind": "none"}
        )

        calibration_state = state.get("calibration")
        if calibration_state is None:
            wrapper.calibrated = False
            wrapper.explainer = None
            cls._restore_preprocessor(wrapper, preprocessor_state, preprocessor, mapping_payload)
            return wrapper
        if not isinstance(calibration_state, Mapping):
            raise IncompatibleStateError(
                "Invalid state file: 'calibration' must be a JSON object or null.",
                details={"field": "calibration"},
            )

        if calibration_state.get("difficulty_estimator_required") and difficulty_estimator is None:
            raise ValidationError(
                "This artifact was calibrated with a difficulty_estimator. Supply the "
                "original (or an equivalent) instance via "
                "load_state(..., difficulty_estimator=...).",
                details={"requirement": "difficulty_estimator required for restoration"},
            )

        mode = calibration_state.get("mode")
        x_cal = np.asarray(calibration_state.get("x_cal"), dtype=float)
        y_cal_raw = calibration_state.get("y_cal")
        y_cal = (
            np.asarray(y_cal_raw, dtype=float) if mode == "regression" else np.asarray(y_cal_raw)
        )
        bins = calibration_state.get("bins")
        bins = np.asarray(bins) if bins is not None else None

        candidate_kwargs: dict[str, Any] = {
            "mode": mode,
            "seed": calibration_state.get("seed"),
            "condition_source": calibration_state.get("condition_source"),
            "interval_summary": calibration_state.get("interval_summary"),
            "feature_names": calibration_state.get("feature_names"),
            "categorical_features": calibration_state.get("categorical_features"),
            "categorical_labels": cls._coerce_int_keys(calibration_state.get("categorical_labels")),
            "class_labels": cls._coerce_int_keys(calibration_state.get("class_labels")),
            "bins": bins,
            "difficulty_estimator": difficulty_estimator,
            "preprocessor_metadata": calibration_state.get("preprocessor_metadata"),
            "plugin_overrides": calibration_state.get("plugin_overrides"),
            "fast": calibration_state.get("fast"),
            "noise_type": calibration_state.get("noise_type"),
            "scale_factor": calibration_state.get("scale_factor"),
            "severity": calibration_state.get("severity"),
            "sample_percentiles": calibration_state.get("sample_percentiles"),
            "features_to_ignore": calibration_state.get("features_to_ignore"),
        }
        candidate_kwargs = {
            key: value for key, value in candidate_kwargs.items() if value is not None
        }

        explainer = CalibratedExplainer(learner, x_cal, y_cal, **candidate_kwargs)
        wrapper.explainer = explainer
        wrapper.calibrated = True

        if "calibrator_primitive.json" in validated_files:
            primitive = cls._read_json_object(path / "calibrator_primitive.json")
            restored = cls._restore_calibrator_from_primitive(primitive)
            orchestrator = getattr(explainer, "prediction_orchestrator", None)
            if orchestrator is not None:
                orchestrator.restore_calibrator_with_learner(
                    restored,
                    learner,
                    difficulty_estimator=getattr(explainer, "difficulty_estimator", None),
                )
            else:
                explainer.interval_learner = restored

        cls._restore_preprocessor(wrapper, preprocessor_state, preprocessor, mapping_payload)

        return wrapper

    @property
    def pre_fitted(self) -> bool:
        """Check if the preprocessor is pre-fitted.

        Returns
        -------
        bool
            True if pre-fitted, False otherwise.
        """
        return self._pre_fitted

    @property
    def cfg(self) -> Any:
        """Configuration property.

        Returns
        -------
        Any
            The configuration.
        """
        return self._cfg

    def __getstate__(self):
        """Get state for pickling.

        Returns
        -------
        dict
            The state dictionary.
        """
        state = self.__dict__.copy()

        # Exclude mc as it may contain unpicklable objects like RNG in mappingproxy
        self._warn_dropping_mondrian_categorizer(operation="Pickle/state persistence")
        state["mc"] = None

        # Convert any types.MappingProxyType (mappingproxy) instances to plain
        # dicts recursively so pickle/joblib can serialize them.
        def _convert(obj: Any) -> Any:
            if isinstance(obj, MappingProxyType):
                # Recursively convert mappingproxy to plain dict and convert
                # nested values as well.
                return _convert(dict(obj))
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                cls = type(obj)
                converted = [_convert(v) for v in obj]
                return cls(converted)
            return obj

        for k, v in list(state.items()):
            try:
                state[k] = _convert(v)
            except (TypeError, AttributeError, RecursionError) as exc:
                # Defensive: if conversion fails due to type/attribute/recursion
                # issues, leave original value and hope it's picklable; avoid
                # failing during state build. Suppress the same specific
                # exceptions when logging to satisfy ADR-002.
                with suppress((TypeError, AttributeError, RecursionError)):
                    self._logger.debug("__getstate__ conversion skipped for %s: %s", k, exc)
                continue
        return state

    def __setstate__(self, state):
        """Set state for unpickling.

        Parameters
        ----------
        state : dict
            The state dictionary.
        """
        self.__dict__.update(state)


__all__ = ["WrapCalibratedExplainer"]
