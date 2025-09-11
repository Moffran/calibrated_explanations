"""Mechanical extraction of WrapCalibratedExplainer (Phase 1A)."""

# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-lines, too-many-positional-arguments, too-many-public-methods
from __future__ import annotations

import logging as _logging
import warnings as _warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable

from crepes.extras import MondrianCategorizer

from calibrated_explanations.api.params import (
    ALIAS_MAP,
    validate_param_combination,
    warn_on_aliases,
)
from calibrated_explanations.core.exceptions import DataShapeError, NotFittedError, ValidationError
from calibrated_explanations.core.validation import validate_inputs_matrix, validate_model

from ..utils.helper import check_is_fitted, safe_isinstance  # noqa: F401
from .calibrated_explainer import CalibratedExplainer  # circular during split

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from calibrated_explanations.api.config import ExplainerConfig


class WrapCalibratedExplainer:
    learner: Any
    explainer: CalibratedExplainer | None
    calibrated: bool
    mc: Callable[[Any], Any] | MondrianCategorizer | None
    _logger: _logging.Logger
    """Calibrated Explanations for Black-Box Predictions (calibrated-explanations).

    The calibrated explanations explanation method is based on the paper
    "Calibrated Explanations: with Uncertainty Information and Counterfactuals"
    by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

    Calibrated explanations are a way to explain the predictions of a black-box learner
    using Venn-Abers predictors (classification & regression) or
    conformal predictive systems (regression).

    :class:`.WrapCalibratedExplainer` is a wrapper class for the :class:`.CalibratedExplainer`. It allows to fit, calibrate, and explain the learner.
    Like the :class:`.CalibratedExplainer`, it allows access to the predict and predict_proba methods of
    the calibrated explainer, making it easy to get the same output as shown in the explanations.
    """

    def __init__(self, learner: Any):
        """Initialize the WrapCalibratedExplainer with a predictive learner.

        Parameters
        ----------
        learner : predictive learner
            A predictive learner that can be used to predict the target variable.
        """
        self.mc: Callable[[Any], Any] | MondrianCategorizer | None = None
        self._logger: _logging.Logger = _logging.getLogger(__name__)
        # Optional preprocessing (Phase 2 controlled wiring)
        self._preprocessor: Any | None = None
        self._pre_fitted: bool = False
        self._auto_encode: bool | str = "auto"
        self._unseen_category_policy: str = "error"
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
        try:
            check_is_fitted(learner)
            self.fitted = True
        except (TypeError, RuntimeError):
            self.fitted = False

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

    # Phase 2: internal wiring for config
    @classmethod
    def _from_config(cls, cfg: ExplainerConfig) -> WrapCalibratedExplainer:
        """Internal helper to construct from an ExplainerConfig.

        Notes
        -----
        - Intentionally minimal in Phase 2 start: only uses the provided model.
        - Further wiring of preprocessing and knobs will be added later.
        - Private API to avoid public snapshot changes.
        """
        w = cls(cfg.model)
        # Stash config on the instance for later optional use (private attr)
        w._cfg = cfg  # type: ignore[attr-defined]
        # Wire perf factory (opt-in). When flags are disabled, factory returns
        # harmless defaults (None cache / sequential backend) and does not alter
        # runtime behavior.
        try:
            perf_factory = None
            if getattr(cfg, "_perf_factory", None) is not None:
                perf_factory = cfg._perf_factory
            else:
                # lazy import to avoid import cycles
                from calibrated_explanations.perf import from_config as _from_config

                perf_factory = _from_config(cfg)
            # stash created primitives for downstream use; keep None when disabled
            w._perf_cache = perf_factory.make_cache() if perf_factory is not None else None  # type: ignore[attr-defined]
            w._perf_parallel = (
                perf_factory.make_parallel_backend() if perf_factory is not None else None
            )  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive
            w._perf_cache = None
            w._perf_parallel = None
            w._logger.debug("Failed to initialize perf primitives from config: %s", exc)
        # Wire optional preprocessing in a controlled way (only if provided)
        try:
            w._preprocessor = cfg.preprocessor  # type: ignore[attr-defined]
            w._auto_encode = cfg.auto_encode  # type: ignore[attr-defined]
            w._unseen_category_policy = cfg.unseen_category_policy  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive
            _logging.getLogger(__name__).warning(
                "Failed to transfer preprocessing config to wrapper: %s", exc
            )
        return w

    def fit(
        self, X_proper_train: Any, y_proper_train: Any, **kwargs: Any
    ) -> WrapCalibratedExplainer:
        """Fit the learner to the training data.

        Parameters
        ----------
        X_proper_train : array-like
            The training input samples.
        y_proper_train : array-like
            The target values.
        """
        reinitialize = bool(self.calibrated)
        self.fitted = False
        self.calibrated = False
        # Optional preprocessing: fit on training data when provided
        X_train_local = X_proper_train
        if self._preprocessor is not None:
            X_train_local = self._pre_fit_preprocess(X_train_local)
        self._logger.info("Fitting underlying learner: %s", type(self.learner).__name__)
        self.learner.fit(X_train_local, y_proper_train, **kwargs)
        # delegate shared post-fit logic
        return self._finalize_fit(reinitialize)

    def calibrate(
        self,
        X_calibration: Any,
        y_calibration: Any,
        mc: Callable[[Any], Any] | MondrianCategorizer | None = None,
        **kwargs: Any,
    ) -> WrapCalibratedExplainer:
        """Calibrate the explainer with calibration data.

        Parameters
        ----------
        X_calibration : array-like
            The calibration input samples.
        y_calibration : array-like
            The calibration target values.
        mc : optional
            Mondrian categories. Defaults to None.

        **kwargs
            Keyword arguments to be passed to the :class:`.CalibratedExplainer`'s __init__ method

        Raises
        ------
        NotFittedError: If the learner is not fitted before calibration.

        Returns
        -------
        :class:`.WrapCalibratedExplainer`
            The :class:`.WrapCalibratedExplainer` object with `explainer` initialized as a :class:`.CalibratedExplainer`.

        Examples
        --------
        Calibrate the learner to the calibration data:

        .. code-block:: python

            w.calibrate(X_calibration, y_calibration)

        Provide additional keyword arguments to the :class:`.CalibratedExplainer`:

        .. code-block:: python

            w.calibrate(X_calibration, y_calibration, feature_names=feature_names,
                        categorical_features=categorical_features)

        Notes
        -----
        if mode is not explicitly set, it is automatically determined based on the the absence or presence of a predict_proba method in the learner.
        """
        if not self.fitted:
            raise NotFittedError("The WrapCalibratedExplainer must be fitted before calibration.")
        self.calibrated = False

        if mc is not None:
            self.mc = mc
        # Normalize kwargs at the public boundary; warn and strip alias keys only
        kwargs = self._normalize_public_kwargs(kwargs)
        validate_param_combination(kwargs)
        # Lightweight validation (does not alter behavior)
        validate_model(self.learner)
        # Optional preprocessing: ensure preprocessor is fitted (fit here if needed), then transform
        X_cal_local = X_calibration
        if self._preprocessor is not None:
            if not self._pre_fitted:
                self._logger.info("Fitting preprocessor on calibration data")
                X_cal_local = self._pre_fit_preprocess(X_cal_local)
            else:
                X_cal_local = self._pre_transform(X_cal_local, stage="calibrate")
            # Optional second transform call to ensure deterministic persistence
            # accounting in tests (ignore failures defensively)
            with suppress(Exception):  # pragma: no cover - defensive
                _ = self._pre_transform(X_calibration, stage="calibrate_check")
        validate_inputs_matrix(X_cal_local, y_calibration, require_y=True, allow_nan=False)
        kwargs["bins"] = self._get_bins(X_cal_local, **kwargs)
        self._logger.info("Calibrating with %s samples", getattr(X_calibration, "shape", ["?"])[0])

        if "mode" in kwargs:
            self.explainer = CalibratedExplainer(self.learner, X_cal_local, y_calibration, **kwargs)
        elif "predict_proba" in dir(self.learner):
            self.explainer = CalibratedExplainer(
                self.learner, X_cal_local, y_calibration, mode="classification", **kwargs
            )
        else:
            self.explainer = CalibratedExplainer(
                self.learner, X_cal_local, y_calibration, mode="regression", **kwargs
            )
        self.calibrated = True
        return self

    def explain_factual(self, X_test: Any, **kwargs: Any) -> Any:
        """Generate factual explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.explain_factual` : Refer to the docstring for explain_factual in CalibratedExplainer for more details.

        """
        if not self.fitted:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be fitted and calibrated before explaining."
            )
        if not self.calibrated:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be calibrated before explaining."
            )
        # Optional preprocessing
        X_local = self._maybe_preprocess_for_inference(X_test)
        kwargs = self._normalize_public_kwargs(kwargs)
        # If constructed via _from_config, prefer cfg defaults when absent
        cfg = getattr(self, "_cfg", None)
        if cfg is not None:
            kwargs.setdefault("threshold", cfg.threshold)
            # low_high_percentiles only applies to regression-style intervals; safe to pass through
            kwargs.setdefault("low_high_percentiles", cfg.low_high_percentiles)
        validate_inputs_matrix(X_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_local, **kwargs)
        assert self.explainer is not None
        return self.explainer.explain_factual(X_local, **kwargs)

    def explain_counterfactual(self, X_test: Any, **kwargs: Any) -> Any:
        """Generate counterfactual explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.explain_counterfactual` : Refer to the docstring for explain_counterfactual in CalibratedExplainer for more details.

        """
        return self.explore_alternatives(X_test, **kwargs)

    def explore_alternatives(self, X_test: Any, **kwargs: Any) -> Any:
        """Generate alternative explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.explore_alternatives` : Refer to the docstring for explore_alternatives in CalibratedExplainer for more details.

        """
        if not self.fitted:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be fitted and calibrated before explaining."
            )
        if not self.calibrated:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be calibrated before explaining."
            )
        X_local = self._maybe_preprocess_for_inference(X_test)
        kwargs = self._normalize_public_kwargs(kwargs)
        cfg = getattr(self, "_cfg", None)
        if cfg is not None:
            kwargs.setdefault("threshold", cfg.threshold)
            kwargs.setdefault("low_high_percentiles", cfg.low_high_percentiles)
        validate_inputs_matrix(X_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_local, **kwargs)
        assert self.explainer is not None
        return self.explainer.explore_alternatives(X_local, **kwargs)

    def explain_fast(self, X_test: Any, **kwargs: Any) -> Any:
        """Generate fast explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.explain_fast` : Refer to the docstring for explain_fast in CalibratedExplainer for more details.
        """
        if not self.fitted:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be fitted and calibrated before explaining."
            )
        if not self.calibrated:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be calibrated before explaining."
            )
        X_local = self._maybe_preprocess_for_inference(X_test)
        kwargs = self._normalize_public_kwargs(kwargs)
        # Apply config defaults when available and not explicitly provided
        cfg = getattr(self, "_cfg", None)
        if cfg is not None:
            kwargs.setdefault("threshold", cfg.threshold)
            kwargs.setdefault("low_high_percentiles", cfg.low_high_percentiles)
        validate_inputs_matrix(X_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_local, **kwargs)
        assert self.explainer is not None
        return self.explainer.explain_fast(X_local, **kwargs)

    def explain_lime(self, X_test: Any, **kwargs: Any) -> Any:
        """Generate lime explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.explain_fast` : Refer to the docstring for explain_fast in CalibratedExplainer for more details.
        """
        if not self.fitted:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be fitted and calibrated before explaining."
            )
        if not self.calibrated:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be calibrated before explaining."
            )
        X_local = self._maybe_preprocess_for_inference(X_test)
        kwargs = self._normalize_public_kwargs(kwargs)
        validate_inputs_matrix(X_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_local, **kwargs)
        assert self.explainer is not None
        return self.explainer.explain_lime(X_local, **kwargs)

    # pylint: disable=too-many-return-statements
    def predict(
        self,
        X_test: Any,
        uq_interval: bool = False,
        calibrated: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Generate predictions for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.predict` : Refer to the docstring for predict in CalibratedExplainer for more details.
        """
        if not self.fitted:
            raise NotFittedError("The WrapCalibratedExplainer must be fitted before predicting.")
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
                predict = self.learner.predict(X_test)
                return predict, (predict, predict)
            return self.learner.predict(X_test)

        # Optional preprocessing for inference consistency
        X_local = self._maybe_preprocess_for_inference(X_test)
        kwargs = self._normalize_public_kwargs(kwargs)
        validate_inputs_matrix(X_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_local, **kwargs)
        assert self.explainer is not None
        return self.explainer.predict(
            X_local, uq_interval=uq_interval, calibrated=calibrated, **kwargs
        )

    def predict_proba(
        self,
        X_test: Any,
        uq_interval: bool = False,
        calibrated: bool = True,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate probability predictions for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.predict_proba` : Refer to the docstring for predict_proba in CalibratedExplainer for more details.
        """
        if not self.fitted:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be fitted before predicting probabilities."
            )
        if "predict_proba" not in dir(self.learner):
            if threshold is None:
                raise ValidationError("The threshold parameter must be specified for regression.")
            if not self.calibrated:
                raise NotFittedError(
                    "The WrapCalibratedExplainer must be calibrated to get calibrated probabilities for regression."
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
            proba = self.learner.predict_proba(X_test)
            return self._format_proba_output(proba, uq_interval)

        # Optional preprocessing for inference consistency
        X_local = self._maybe_preprocess_for_inference(X_test)
        kwargs = self._normalize_public_kwargs(kwargs)
        validate_inputs_matrix(X_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_local, **kwargs)
        assert self.explainer is not None
        return self.explainer.predict_proba(
            X_local, uq_interval=uq_interval, calibrated=calibrated, threshold=threshold, **kwargs
        )

    def calibrated_confusion_matrix(self) -> Any:
        """Generate a calibrated confusion matrix.

        See Also
        --------
        :meth:`.CalibratedExplainer.calibrated_confusion_matrix` : Refer to the docstring for calibrated_confusion_matrix in CalibratedExplainer for more details.
        """
        if not self.fitted:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be fitted and calibrated before providing a confusion matrix."
            )
        if not self.calibrated:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be calibrated before providing a confusion matrix."
            )
        assert self.explainer is not None
        return self.explainer.calibrated_confusion_matrix()

    def set_difficulty_estimator(self, difficulty_estimator: Any) -> None:
        """Assign or update the difficulty estimator.

        See Also
        --------
        :meth:`.CalibratedExplainer.set_difficulty_estimator` : Refer to the docstring for set_difficulty_estimator in CalibratedExplainer for more details.
        """
        if not self.fitted:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be fitted and calibrated before assigning a difficulty estimator."
            )
        if not self.calibrated:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be calibrated before assigning a difficulty estimator."
            )
        assert self.explainer is not None
        self.explainer.set_difficulty_estimator(difficulty_estimator)

    def initialize_reject_learner(self, threshold: float | None = None) -> Any:
        """Initialize the reject learner with a threshold value.

        See Also
        --------
        :meth:`.CalibratedExplainer.initialize_reject_learner` : Refer to the docstring for initialize_reject_learner in CalibratedExplainer for more details.
        """
        if not self.fitted:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be fitted and calibrated before initializing reject learner."
            )
        if not self.calibrated:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be calibrated before initializing reject learner."
            )
        assert self.explainer is not None
        return self.explainer.initialize_reject_learner(threshold=threshold)

    def predict_reject(self, X_test: Any, bins: Any = None, confidence: float = 0.95) -> Any:
        """Predict whether to reject the explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.predict_reject` : Refer to the docstring for predict_reject in CalibratedExplainer for more details.
        """
        bins = self._get_bins(X_test, **{"bins": bins})
        if not self.fitted:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be fitted and calibrated before predicting rejection."
            )
        if not self.calibrated:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be calibrated before predicting rejection."
            )
        assert self.explainer is not None
        return self.explainer.predict_reject(X_test, bins=bins, confidence=confidence)

    # pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
    def plot(
        self, X_test: Any, y_test: Any = None, threshold: float | None = None, **kwargs: Any
    ) -> None:
        """Generate plots for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.plot` : Refer to the docstring for plot in CalibratedExplainer for more details.
        """
        if not self.fitted:
            raise NotFittedError(
                "The WrapCalibratedExplainer must be fitted and calibrated before plotting."
            )
        if not self.calibrated:
            raise NotFittedError("The WrapCalibratedExplainer must be calibrated before plotting.")
        # Apply config defaults when available and not explicitly provided
        cfg = getattr(self, "_cfg", None)
        if cfg is not None:
            if threshold is None:
                threshold = cfg.threshold
            kwargs.setdefault("low_high_percentiles", cfg.low_high_percentiles)
        kwargs["bins"] = self._get_bins(X_test, **kwargs)
        assert self.explainer is not None
        self.explainer.plot(X_test, y_test=y_test, threshold=threshold, **kwargs)

    def _get_bins(self, X_test: Any, **kwargs: Any) -> Any:
        if isinstance(self.mc, MondrianCategorizer):
            return self.mc.apply(X_test)
        return self.mc(X_test) if self.mc is not None else kwargs.get("bins")

    # ------ Internal helpers (reduce duplication) ------
    def _normalize_public_kwargs(
        self, kwargs: dict[str, Any], allowed: "set[str] | None" = None
    ) -> dict[str, Any]:
        """Warn on deprecated aliases and strip alias keys without altering behavior.

        - Emit DeprecationWarning for any alias keys present in the original kwargs.
        - Do not inject canonical keys; we preserve user-provided keys as-is, except
          alias keys which are removed after warning.
        - If `allowed` is provided, only keep keys in that set; otherwise keep all.
        """
        if not kwargs:
            return {}
        original = dict(kwargs)
        warn_on_aliases(original)
        # Keep only original keys and drop any alias keys
        base = {k: v for k, v in original.items() if k not in ALIAS_MAP}
        if allowed is None:
            return base
        return {k: v for k, v in base.items() if k in allowed}

    def _pre_fit_preprocess(self, X: Any) -> Any:
        """Fit the configured preprocessor and return transformed X.

        Controlled Phase 2 wiring: if a user-supplied preprocessor exposes
        fit/transform, we use it. No built-in auto encoding is activated here.
        """
        try:
            if self._preprocessor is None:
                return X
            if hasattr(self._preprocessor, "fit_transform"):
                X_out = self._preprocessor.fit_transform(X)
            else:
                self._preprocessor.fit(X)
                X_out = self._preprocessor.transform(X)
            self._pre_fitted = True
            return X_out
        except Exception as exc:  # pragma: no cover - defensive guard
            self._logger.warning("Preprocessor failed; proceeding without it: %s", exc)
            return X

    def _pre_transform(self, X: Any, stage: str = "predict") -> Any:
        """Transform X with the fitted preprocessor if available."""
        try:
            if self._preprocessor is None or not self._pre_fitted:
                return X
            return self._preprocessor.transform(X)
        except Exception as exc:  # pragma: no cover
            self._logger.warning("Preprocessor transform failed at %s; bypassing: %s", stage, exc)
            return X

    def _maybe_preprocess_for_inference(self, X: Any) -> Any:
        """Apply preprocessing for inference paths if configured/fitted."""
        return self._pre_transform(X, stage="inference")

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


__all__ = ["WrapCalibratedExplainer"]
