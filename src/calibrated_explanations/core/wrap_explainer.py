"""Mechanical extraction of WrapCalibratedExplainer."""

# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-lines, too-many-positional-arguments, too-many-public-methods
from __future__ import annotations

import logging as _logging
import sys
import warnings as _warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping

from crepes.extras import MondrianCategorizer

from calibrated_explanations.api.params import (
    ALIAS_MAP,
    validate_param_combination,
    warn_on_aliases,
)
from calibrated_explanations.core.validation import validate_inputs_matrix, validate_model
from calibrated_explanations.utils.exceptions import DataShapeError, NotFittedError, ValidationError

from ..utils import check_is_fitted, safe_isinstance  # noqa: F401
from .calibrated_explainer import CalibratedExplainer  # circular during split

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from calibrated_explanations.api.config import ExplainerConfig


class WrapCalibratedExplainer:
    """Provide a high-level fit/calibrate/explain workflow for learners.

    The wrapper mirrors :class:`CalibratedExplainer` while orchestrating
    fitting, calibration, and explanation steps behind a scikit-learn style
    interface.
    """

    learner: Any
    explainer: CalibratedExplainer | None
    calibrated: bool
    mc: Callable[[Any], Any] | MondrianCategorizer | None
    _logger: _logging.Logger

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

    # internal wiring for config
    @classmethod
    def _from_config(cls, cfg: ExplainerConfig) -> WrapCalibratedExplainer:
        """Construct a wrapper from an :class:`ExplainerConfig`.

        Notes
        -----
        - Intentionally minimal and only uses the provided model.
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
            if perf_factory is not None:
                cache = perf_factory.make_cache()
                w._perf_cache = cache  # type: ignore[attr-defined]
                w._perf_parallel = perf_factory.make_parallel_executor(cache)  # type: ignore[attr-defined]
            else:
                w._perf_cache = None
                w._perf_parallel = None
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            exc = sys.exc_info()[1]
            w._perf_cache = None
            w._perf_parallel = None
            w._logger.debug("Failed to initialize perf primitives from config: %s", exc)
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
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            exc = sys.exc_info()[1]
            _logging.getLogger(__name__).debug(
                "Failed to initialize feature filter config from ExplainerConfig: %s", exc
            )
        # Wire optional preprocessing in a controlled way (only if provided)
        try:
            w._preprocessor = cfg.preprocessor  # type: ignore[attr-defined]
            w._auto_encode = cfg.auto_encode  # type: ignore[attr-defined]
            w._unseen_category_policy = cfg.unseen_category_policy  # type: ignore[attr-defined]
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
        """Fit the learner to the training data.

        Parameters
        ----------
        x_proper_train : array-like
            The training input samples.
        y_proper_train : array-like
            The target values.
        """
        reinitialize = bool(self.calibrated)
        self.fitted = False
        self.calibrated = False
        # Optional preprocessing: fit on training data when provided
        x_train_local = x_proper_train
        if self._preprocessor is not None:
            x_train_local = self._pre_fit_preprocess(x_train_local)
        self._logger.info("Fitting underlying learner: %s", type(self.learner).__name__)
        self.learner.fit(x_train_local, y_proper_train, **kwargs)
        # delegate shared post-fit logic
        return self._finalize_fit(reinitialize)

    def calibrate(
        self,
        x_calibration: Any,
        y_calibration: Any,
        mc: Callable[[Any], Any] | MondrianCategorizer | None = None,
        **kwargs: Any,
    ) -> WrapCalibratedExplainer:
        """Calibrate the explainer with calibration data.

        Parameters
        ----------
        x_calibration : array-like
            The calibration input samples.
        y_calibration : array-like
            The calibration target values.
        mc : callable or MondrianCategorizer, optional
            Optional Mondrian categories helper. Defaults to ``None``.

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

            w.calibrate(x_calibration, y_calibration)

        Provide additional keyword arguments to the :class:`.CalibratedExplainer`:

        .. code-block:: python

            w.calibrate(x_calibration, y_calibration, feature_names=feature_names,
                        categorical_features=categorical_features)

        Notes
        -----
        if mode is not explicitly set, it is automatically determined based on the the absence or presence of a predict_proba method in the learner.
        """
        self._assert_fitted("The WrapCalibratedExplainer must be fitted before calibration.")
        self.calibrated = False

        if mc is not None:
            self.mc = mc
        # Normalize kwargs at the public boundary; warn and strip alias keys only
        kwargs = self._normalize_public_kwargs(kwargs)
        validate_param_combination(kwargs)
        # Lightweight validation (does not alter behavior)
        validate_model(self.learner)
        preprocessor_metadata = self._build_preprocessor_metadata()
        # Optional preprocessing: ensure preprocessor is fitted (fit here if needed), then transform
        x_cal_local = x_calibration
        if self._preprocessor is not None:
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
        kwargs["bins"] = self._get_bins(x_cal_local, **kwargs)
        if preprocessor_metadata is not None:
            kwargs.setdefault("preprocessor_metadata", preprocessor_metadata)
        self._logger.info("Calibrating with %s samples", getattr(x_calibration, "shape", ["?"])[0])

        if "mode" in kwargs:
            self.explainer = CalibratedExplainer(
                self.learner,
                x_cal_local,
                y_calibration,
                perf_cache=getattr(self, "_perf_cache", None),
                perf_parallel=getattr(self, "_perf_parallel", None),
                **kwargs,
            )
        elif "predict_proba" in dir(self.learner):
            self.explainer = CalibratedExplainer(
                self.learner,
                x_cal_local,
                y_calibration,
                mode="classification",
                perf_cache=getattr(self, "_perf_cache", None),
                perf_parallel=getattr(self, "_perf_parallel", None),
                **kwargs,
            )
        else:
            self.explainer = CalibratedExplainer(
                self.learner,
                x_cal_local,
                y_calibration,
                mode="regression",
                perf_cache=getattr(self, "_perf_cache", None),
                perf_parallel=getattr(self, "_perf_parallel", None),
                **kwargs,
            )
        # Propagate internal feature filter config to explainer when available
        if self.explainer is not None and hasattr(self, "_feature_filter_config"):
            try:
                self.explainer._feature_filter_config = self._feature_filter_config
            except AttributeError:  # pragma: no cover - defensive
                self._logger.debug(
                    "Failed to attach feature filter config to explainer", exc_info=True
                )
        self.calibrated = True
        if preprocessor_metadata is not None and self.explainer is not None:
            with suppress(AttributeError):
                self.explainer.set_preprocessor_metadata(preprocessor_metadata)
        return self

    def explain_factual(self, x: Any, **kwargs: Any) -> Any:
        """Generate factual explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.explain_factual` : Refer to the docstring for explain_factual in CalibratedExplainer for more details.

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
        kwargs = self._normalize_public_kwargs(kwargs)
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

    def explain_counterfactual(self, x: Any, **kwargs: Any) -> Any:
        """Generate counterfactual explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.explain_counterfactual` : Refer to the docstring for explain_counterfactual in CalibratedExplainer for more details.

        """
        return self.explore_alternatives(x, **kwargs)

    def explore_alternatives(self, x: Any, **kwargs: Any) -> Any:
        """Generate alternative explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.explore_alternatives` : Refer to the docstring for explore_alternatives in CalibratedExplainer for more details.

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
        kwargs = self._normalize_public_kwargs(kwargs)
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
        kwargs = self._normalize_public_kwargs(kwargs)
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

    def explain_lime(self, x: Any, **kwargs: Any) -> Any:
        """Generate lime explanations for the test data.

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
        kwargs = self._normalize_public_kwargs(kwargs)
        validate_inputs_matrix(x_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(x_local, **kwargs)
        return self.explainer.explain_lime(x_local, **kwargs)

    def explain_shap(self, x: Any, **kwargs: Any) -> Any:
        """Generate SHAP explanations for the test data."""
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted and calibrated before explaining."
            )
            ._assert_calibrated("The WrapCalibratedExplainer must be calibrated before explaining.")
            .explainer
            is not None
        )
        x_local = self._maybe_preprocess_for_inference(x)
        kwargs = self._normalize_public_kwargs(kwargs)
        validate_inputs_matrix(x_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(x_local, **kwargs)
        return self.explainer.explain_shap(x_local, **kwargs)

    # pylint: disable=too-many-return-statements
    def predict(
        self,
        x: Any,
        uq_interval: bool = False,
        calibrated: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Generate predictions for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.predict` : Refer to the docstring for predict in CalibratedExplainer for more details.
        """
        self._assert_fitted("The WrapCalibratedExplainer must be fitted before predicting.")
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
                predict = self.learner.predict(x)
                return predict, (predict, predict)
            return self.learner.predict(x)

        # Optional preprocessing for inference consistency
        x_local = self._maybe_preprocess_for_inference(x)
        kwargs = self._normalize_public_kwargs(kwargs)
        validate_inputs_matrix(x_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(x_local, **kwargs)
        assert (
            self._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated to get calibrated predictions."
            ).explainer
            is not None
        )
        return self.explainer.predict(
            x_local, uq_interval=uq_interval, calibrated=calibrated, **kwargs
        )

    def predict_proba(
        self,
        x: Any,
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
        self._assert_fitted(
            "The WrapCalibratedExplainer must be fitted before predicting probabilities."
        )
        if "predict_proba" not in dir(self.learner):
            if threshold is None:
                raise ValidationError("The threshold parameter must be specified for regression.")
            self._assert_calibrated(
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
            proba = self.learner.predict_proba(x)
            return self._format_proba_output(proba, uq_interval)

        # Optional preprocessing for inference consistency
        x_local = self._maybe_preprocess_for_inference(x)
        kwargs = self._normalize_public_kwargs(kwargs)
        validate_inputs_matrix(x_local, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(x_local, **kwargs)
        assert (
            self._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated to get calibrated probabilities."
            ).explainer
            is not None
        )
        return self.explainer.predict_proba(
            x_local, uq_interval=uq_interval, calibrated=calibrated, threshold=threshold, **kwargs
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

    def set_difficulty_estimator(self, difficulty_estimator: Any) -> None:
        """Assign or update the difficulty estimator.

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
        self.explainer.set_difficulty_estimator(difficulty_estimator)

    def initialize_reject_learner(self, threshold: float | None = None) -> Any:
        """Initialize the reject learner with a threshold value.

        See Also
        --------
        :meth:`.CalibratedExplainer.initialize_reject_learner` : Refer to the docstring for initialize_reject_learner in CalibratedExplainer for more details.
        """
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted before initializing the reject learner."
            )
            ._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated before initializing the reject learner."
            )
            .explainer
            is not None
        )
        return self.explainer.initialize_reject_learner(threshold=threshold)

    def predict_reject(self, x: Any, bins: Any = None, confidence: float = 0.95) -> Any:
        """Predict whether to reject the explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.predict_reject` : Refer to the docstring for predict_reject in CalibratedExplainer for more details.
        """
        bins = self._get_bins(x, **{"bins": bins})
        assert (
            self._assert_fitted(
                "The WrapCalibratedExplainer must be fitted and calibrated before predicting rejection."
            )
            ._assert_calibrated(
                "The WrapCalibratedExplainer must be calibrated before predicting rejection."
            )
            .explainer
            is not None
        )
        return self.explainer.predict_reject(x, bins=bins, confidence=confidence)

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
        None

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
        self.explainer.plot(x, y=y, threshold=threshold, **kwargs)

    def _get_bins(self, x: Any, **kwargs: Any) -> Any:
        """Derive bin assignments from the configured Mondrian categorizer."""
        if isinstance(self.mc, MondrianCategorizer):
            return self.mc.apply(x)
        return self.mc(x) if self.mc is not None else kwargs.get("bins")

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
        return self.explainer._preprocessor_metadata

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

    def _pre_fit_preprocess(self, x: Any) -> Any:
        """Fit the configured preprocessor and return transformed x.

        if a user-supplied preprocessor exposes
        fit/transform, we use it. No built-in auto encoding is activated here.
        """
        try:
            if self._preprocessor is None:
                return x
            if hasattr(self._preprocessor, "fit_transform"):
                x_out = self._preprocessor.fit_transform(x)
            else:
                self._preprocessor.fit(x)
                x_out = self._preprocessor.transform(x)
            self._pre_fitted = True
            return x_out
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            exc = sys.exc_info()[1]
            self._logger.warning("Preprocessor failed; proceeding without it: %s", exc)
            return x

    def _pre_transform(self, x: Any, stage: str = "predict") -> Any:
        """Transform x with the fitted preprocessor if available."""
        try:
            if self._preprocessor is None or not self._pre_fitted:
                return x
            return self._preprocessor.transform(x)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            exc = sys.exc_info()[1]
            self._logger.warning("Preprocessor transform failed at %s; bypassing: %s", stage, exc)
            return x

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
