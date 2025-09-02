"""Mechanical extraction of WrapCalibratedExplainer (Phase 1A)."""

# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-lines, too-many-positional-arguments, too-many-public-methods
from __future__ import annotations

import logging as _logging
import warnings as _warnings

from crepes.extras import MondrianCategorizer  # type: ignore

from calibrated_explanations.api.params import canonicalize_kwargs, validate_param_combination
from calibrated_explanations.core.exceptions import DataShapeError, NotFittedError, ValidationError
from calibrated_explanations.core.validation import validate_inputs_matrix, validate_model

from ..utils.helper import check_is_fitted, safe_isinstance  # noqa: F401
from .calibrated_explainer import CalibratedExplainer  # type: ignore  # circular during split


class WrapCalibratedExplainer:
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

    def __init__(self, learner):
        """Initialize the WrapCalibratedExplainer with a predictive learner.

        Parameters
        ----------
        learner : predictive learner
            A predictive learner that can be used to predict the target variable.
        """
        self.mc = None
        self._logger = _logging.getLogger(__name__)
        # Check if the learner is a CalibratedExplainer
        if safe_isinstance(learner, "calibrated_explanations.core.CalibratedExplainer"):
            explainer = learner
            learner = explainer.learner
            self.learner = learner
            check_is_fitted(self.learner)
            self.fitted = True
            self.explainer = explainer
            self.calibrated = True
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

    def __repr__(self):
        """Return the string representation of the WrapCalibratedExplainer."""
        if self.fitted:
            if self.calibrated:
                return (
                    f"WrapCalibratedExplainer(learner={self.learner}, fitted=True, "
                    f"calibrated=True, \n\t\texplainer={self.explainer})"
                )
            return f"WrapCalibratedExplainer(learner={self.learner}, fitted=True, calibrated=False)"
        return f"WrapCalibratedExplainer(learner={self.learner}, fitted=False, calibrated=False)"

    def fit(self, X_proper_train, y_proper_train, **kwargs):
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
        self._logger.info("Fitting underlying learner: %s", type(self.learner).__name__)
        self.learner.fit(X_proper_train, y_proper_train, **kwargs)
        # delegate shared post-fit logic (also used by OnlineCalibratedExplainer)
        return self._finalize_fit(reinitialize)

    def calibrate(self, X_calibration, y_calibration, mc=None, **kwargs):
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
        # Canonicalize parameters before passing along
        kwargs = canonicalize_kwargs(kwargs)
        validate_param_combination(kwargs)
        # Lightweight validation (does not alter behavior)
        validate_model(self.learner)
        validate_inputs_matrix(X_calibration, y_calibration, require_y=True, allow_nan=False)
        kwargs["bins"] = self._get_bins(X_calibration, **kwargs)
        self._logger.info("Calibrating with %s samples", getattr(X_calibration, "shape", ["?"])[0])

        if "mode" in kwargs:
            self.explainer = CalibratedExplainer(
                self.learner, X_calibration, y_calibration, **kwargs
            )
        elif "predict_proba" in dir(self.learner):
            self.explainer = CalibratedExplainer(
                self.learner, X_calibration, y_calibration, mode="classification", **kwargs
            )
        else:
            self.explainer = CalibratedExplainer(
                self.learner, X_calibration, y_calibration, mode="regression", **kwargs
            )
        self.calibrated = True
        return self

    def explain_factual(self, X_test, **kwargs):
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

        kwargs = canonicalize_kwargs(kwargs)
        validate_inputs_matrix(X_test, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_test, **kwargs)
        return self.explainer.explain_factual(X_test, **kwargs)

    def explain_counterfactual(self, X_test, **kwargs):
        """Generate counterfactual explanations for the test data.

        See Also
        --------
        :meth:`.CalibratedExplainer.explain_counterfactual` : Refer to the docstring for explain_counterfactual in CalibratedExplainer for more details.

        """
        return self.explore_alternatives(X_test, **kwargs)

    def explore_alternatives(self, X_test, **kwargs):
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

        kwargs = canonicalize_kwargs(kwargs)
        validate_inputs_matrix(X_test, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_test, **kwargs)
        return self.explainer.explore_alternatives(X_test, **kwargs)

    def explain_fast(self, X_test, **kwargs):
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

        kwargs = canonicalize_kwargs(kwargs)
        validate_inputs_matrix(X_test, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_test, **kwargs)
        return self.explainer.explain_fast(X_test, **kwargs)

    def explain_lime(self, X_test, **kwargs):
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

        kwargs = canonicalize_kwargs(kwargs)
        validate_inputs_matrix(X_test, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_test, **kwargs)
        return self.explainer.explain_lime(X_test, **kwargs)

    # pylint: disable=too-many-return-statements
    def predict(self, X_test, uq_interval=False, calibrated=True, **kwargs):
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

        kwargs = canonicalize_kwargs(kwargs)
        validate_inputs_matrix(X_test, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_test, **kwargs)
        return self.explainer.predict(
            X_test, uq_interval=uq_interval, calibrated=calibrated, **kwargs
        )

    def predict_proba(self, X_test, uq_interval=False, calibrated=True, threshold=None, **kwargs):
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
            proba = self.learner.predict_proba(X_test)
            return self._format_proba_output(proba, uq_interval)

        kwargs = canonicalize_kwargs(kwargs)
        validate_inputs_matrix(X_test, allow_nan=True)
        validate_param_combination(kwargs)
        kwargs["bins"] = self._get_bins(X_test, **kwargs)
        return self.explainer.predict_proba(
            X_test, uq_interval=uq_interval, calibrated=calibrated, threshold=threshold, **kwargs
        )

    def calibrated_confusion_matrix(self):
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
        return self.explainer.calibrated_confusion_matrix()

    def set_difficulty_estimator(self, difficulty_estimator) -> None:
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
        self.explainer.set_difficulty_estimator(difficulty_estimator)

    def initialize_reject_learner(self, threshold=None):
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
        return self.explainer.initialize_reject_learner(threshold=threshold)

    def predict_reject(self, X_test, bins=None, confidence=0.95):
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
        return self.explainer.predict_reject(X_test, bins=bins, confidence=confidence)

    # pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
    def plot(self, X_test, y_test=None, threshold=None, **kwargs):
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
        kwargs["bins"] = self._get_bins(X_test, **kwargs)
        self.explainer.plot(X_test, y_test=y_test, threshold=threshold, **kwargs)

    def _get_bins(self, X_test, **kwargs):
        if isinstance(self.mc, MondrianCategorizer):
            return self.mc.apply(X_test)
        return self.mc(X_test) if self.mc is not None else kwargs.get("bins")

    # ------ Internal helpers (reduce duplication) ------
    def _finalize_fit(self, reinitialize: bool):
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

    def _format_proba_output(self, proba, uq_interval: bool):
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
