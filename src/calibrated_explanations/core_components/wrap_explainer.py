"""Mechanical extraction of WrapCalibratedExplainer (Phase 1A)."""
from __future__ import annotations
import warnings
import numpy as np
from ..utils.helper import check_is_fitted, safe_isinstance  # noqa: F401
from ..core import CalibratedExplainer  # type: ignore  # circular during split
from crepes.extras import MondrianCategorizer  # type: ignore

class WrapCalibratedExplainer:  # noqa: D401
    """Wrapper around CalibratedExplainer (mechanically moved)."""
    def __init__(self, learner):
        self.mc = None
        if safe_isinstance(learner, "calibrated_explanations.core.CalibratedExplainer"):
            explainer = learner
            learner = explainer.learner
            self.learner = learner
            check_is_fitted(self.learner)
            self.fitted = True
            self.explainer = explainer
            self.calibrated = True
            return
        self.learner = learner
        self.explainer = None
        self.calibrated = False
        try:
            check_is_fitted(learner)
            self.fitted = True
        except (TypeError, RuntimeError):
            self.fitted = False

    def __repr__(self):  # noqa: D401
        if self.fitted:
            if self.calibrated:
                return (f"WrapCalibratedExplainer(learner={self.learner}, fitted=True, "
                        f"calibrated=True, \n\t\texplainer={self.explainer})")
            return f"WrapCalibratedExplainer(learner={self.learner}, fitted=True, calibrated=False)"
        return f"WrapCalibratedExplainer(learner={self.learner}, fitted=False, calibrated=False)"

    def fit(self, X_proper_train, y_proper_train, **kwargs):  # noqa: D401
        reinitialize = bool(self.calibrated)
        self.fitted = False
        self.calibrated = False
        self.learner.fit(X_proper_train, y_proper_train, **kwargs)
        check_is_fitted(self.learner)
        self.fitted = True
        if reinitialize:
            self.explainer.reinitialize(self.learner)
            self.calibrated = True
        return self

    def calibrate(self, X_calibration, y_calibration, mc=None, **kwargs):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before calibration.")
        self.calibrated = False
        if mc is not None:
            self.mc = mc
        kwargs['bins'] = self._get_bins(X_calibration, **kwargs)
        if 'mode' in kwargs:
            self.explainer = CalibratedExplainer(self.learner, X_calibration, y_calibration, **kwargs)
        elif 'predict_proba' in dir(self.learner):
            self.explainer = CalibratedExplainer(self.learner, X_calibration, y_calibration, mode='classification', **kwargs)
        else:
            self.explainer = CalibratedExplainer(self.learner, X_calibration, y_calibration, mode='regression', **kwargs)
        self.calibrated = True
        return self

    def explain_factual(self, X_test, **kwargs):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted and calibrated before explaining.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before explaining.")
        kwargs['bins'] = self._get_bins(X_test, **kwargs)
        return self.explainer.explain_factual(X_test, **kwargs)

    def explain_counterfactual(self, X_test, **kwargs):  # noqa: D401
        return self.explore_alternatives(X_test, **kwargs)

    def explore_alternatives(self, X_test, **kwargs):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted and calibrated before explaining.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before explaining.")
        kwargs['bins'] = self._get_bins(X_test, **kwargs)
        return self.explainer.explore_alternatives(X_test, **kwargs)

    def explain_fast(self, X_test, **kwargs):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted and calibrated before explaining.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before explaining.")
        kwargs['bins'] = self._get_bins(X_test, **kwargs)
        return self.explainer.explain_fast(X_test, **kwargs)

    def explain_lime(self, X_test, **kwargs):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted and calibrated before explaining.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before explaining.")
        kwargs['bins'] = self._get_bins(X_test, **kwargs)
        return self.explainer.explain_lime(X_test, **kwargs)

    def predict(self, X_test, uq_interval=False, calibrated=True, **kwargs):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before predicting.")
        if X_test is None:
            raise ValueError("X_test is None.")
        X_arr = np.asarray(X_test)
        if X_arr.size == 0:
            raise ValueError("X_test is empty.")
        if np.isnan(X_arr).any() or np.isinf(X_arr).any():
            raise ValueError("Input contains NaN or infinite values.")
        if not self.calibrated:
            if 'threshold' in kwargs:
                raise ValueError("A thresholded prediction is not possible for uncalibrated learners.")
            if calibrated:
                warnings.warn("The WrapCalibratedExplainer must be calibrated to get calibrated predictions.", UserWarning)
            if uq_interval:
                predict = self.learner.predict(X_test)
                return predict, (predict, predict)
            return self.learner.predict(X_test)
        # calibrated wrapper present
        if calibrated is False:
            # Return raw learner predictions (uncalibrated) per test expectations
            raw = self.learner.predict(X_test)
            if uq_interval:
                return raw, (raw, raw)
            return raw
        kwargs['bins'] = self._get_bins(X_test, **kwargs)
        return self.explainer.predict(X_test, uq_interval=uq_interval, calibrated=calibrated, **kwargs)

    def predict_proba(self, X_test, uq_interval=False, calibrated=True, threshold=None, **kwargs):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before predicting probabilities.")
        if X_test is None:
            raise ValueError("X_test is None.")
        X_arr = np.asarray(X_test)
        if X_arr.size == 0:
            raise ValueError("X_test is empty.")
        if np.isnan(X_arr).any() or np.isinf(X_arr).any():
            raise ValueError("Input contains NaN or infinite values.")
        if 'predict_proba' not in dir(self.learner):
            if threshold is None:
                raise ValueError("The threshold parameter must be specified for regression.")
            if not self.calibrated:
                raise RuntimeError("The WrapCalibratedExplainer must be calibrated to get calibrated probabilities for regression.")
        if not self.calibrated:
            if threshold is not None:
                raise ValueError("A thresholded prediction is not possible for uncalibrated learners.")
            if calibrated:
                warnings.warn("The WrapCalibratedExplainer must be calibrated to get calibrated probabilities.", UserWarning)
            if uq_interval:
                proba = self.learner.predict_proba(X_test)
                if proba.shape[1] > 2:
                    return proba, (proba, proba)
                return proba, (proba[:,1], proba[:,1])
            return self.learner.predict_proba(X_test)
        if calibrated is False:
            # Return raw learner probabilities (or thresholded regression) after calibration
            if 'predict_proba' in dir(self.learner):
                proba = self.learner.predict_proba(X_test)
                if uq_interval:
                    if proba.shape[1] > 2:
                        return proba, (proba, proba)
                    return proba, (proba[:,1], proba[:,1])
                return proba
            # regression case with threshold(s)
            if threshold is None:
                raise ValueError("Threshold must be specified for regression probabilities.")
            # delegate to calibrated explainer for probability of threshold event
            return self.explainer.predict_proba(X_test, uq_interval=uq_interval, calibrated=True, threshold=threshold, **kwargs)
        kwargs['bins'] = self._get_bins(X_test, **kwargs)
        return self.explainer.predict_proba(X_test, uq_interval=uq_interval, calibrated=calibrated, threshold=threshold, **kwargs)

    def calibrated_confusion_matrix(self):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted and calibrated before providing a confusion matrix.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before providing a confusion matrix.")
        return self.explainer.calibrated_confusion_matrix()

    def set_difficulty_estimator(self, difficulty_estimator) -> None:  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted and calibrated before assigning a difficulty estimator.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before assigning a difficulty estimator.")
        self.explainer.set_difficulty_estimator(difficulty_estimator)

    def initialize_reject_learner(self, threshold=None):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted and calibrated before initializing reject learner.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before initializing reject learner.")
        return self.explainer.initialize_reject_learner(threshold=threshold)

    def predict_reject(self, X_test, bins=None, confidence=0.95):  # noqa: D401
        bins = self._get_bins(X_test, **{'bins': bins})
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted and calibrated before predicting rejection.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before predicting rejection.")
        return self.explainer.predict_reject(X_test, bins=bins, confidence=confidence)

    def plot(self, X_test, y_test=None, threshold=None, **kwargs):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted and calibrated before plotting.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before plotting.")
        kwargs['bins'] = self._get_bins(X_test, **kwargs)
        self.explainer.plot(X_test, y_test=y_test, threshold=threshold, **kwargs)

    def _get_bins(self, X_test, **kwargs):  # noqa: D401
        if isinstance(self.mc, MondrianCategorizer):
            return self.mc.apply(X_test)
        return self.mc(X_test) if self.mc is not None else kwargs.get('bins', None)

__all__ = ["WrapCalibratedExplainer"]
