"""Mechanical extraction of OnlineCalibratedExplainer (Phase 1A).

Verbatim move from legacy core.py (no semantic changes)."""
from __future__ import annotations
import numpy as np
from .wrap_explainer import WrapCalibratedExplainer  # fixed import
from ..utils.helper import check_is_fitted  # noqa: F401


class OnlineCalibratedExplainer(WrapCalibratedExplainer):  # noqa: D401
    """Calibrated Explanations for Online Learning (mechanically moved).

    Extends WrapCalibratedExplainer to support incremental learning & calibration.
    """
    def fit(self, X_proper_train, y_proper_train, **kwargs):  # noqa: D401
        reinitialize = bool(self.calibrated)
        self.fitted = False
        self.calibrated = False
        if hasattr(self.learner, 'fit'):
            self.learner.fit(X_proper_train, y_proper_train, **kwargs)
        else:
            if 'classes' not in kwargs:
                kwargs['classes'] = np.unique(y_proper_train)
            try:
                self.learner.partial_fit(X_proper_train, y_proper_train, **kwargs)
            except TypeError:
                kwargs.pop('classes', None)
                self.learner.partial_fit(X_proper_train, y_proper_train, **kwargs)
        check_is_fitted(self.learner)
        self.fitted = True
        if reinitialize:
            self.explainer.reinitialize(self.learner)
            self.calibrated = True
        return self

    def partial_fit(self, X, y, **kwargs):  # noqa: D401
        if not hasattr(self.learner, 'partial_fit'):
            raise AttributeError("The learner must implement partial_fit for incremental learning")
        if np.isscalar(y):
            X = np.asarray(X).reshape(1, -1)
            y = np.asarray(y).reshape(1)
        self.learner.partial_fit(X, y, **kwargs)
        self.fitted = True
        return self

    def online_fit_and_calibrate(self, X, y, **kwargs):  # noqa: D401
        pre_pred = self.explainer.predict_function(self.explainer.X_cal) if self.calibrated else None
        try:
            self.partial_fit(X, y, **kwargs)
        except AttributeError:
            self.partial_fit(X, y)
        post_pred = self.explainer.predict_function(self.explainer.X_cal) if self.calibrated else None
        if self.calibrated and np.all(pre_pred == post_pred):
            if np.isscalar(y):
                self.calibrate_one(X, y, **kwargs)
            else:
                self.calibrate_many(X, y, **kwargs)
        else:
            if np.isscalar(y):
                X = X.reshape(1, -1)
                y = y.reshape(-1)
            if self.calibrated:
                X = np.concatenate((self.explainer.X_cal, X), axis=0)
                y = np.concatenate((self.explainer.y_cal, y), axis=0)
            if 'mode' not in kwargs:
                if hasattr(self.learner, 'predict_proba'):
                    kwargs['mode'] = 'classification'
                else:
                    kwargs['mode'] = 'regression'
            self.calibrate(X, y, **kwargs)

    def calibrate_one(self, x, y, **kwargs):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The OnlineCalibratedExplainer must be fitted before calibration.")
        if self.calibrated:
            if np.isscalar(y):
                self.explainer.append_cal(np.asarray(x).reshape(1, -1), np.asarray([y]))
            else:
                self.explainer.append_cal(np.asarray(x).reshape(1, -1), np.asarray(y).reshape(1))
            self.explainer._preprocess()  # pylint: disable=protected-access
            self.explainer._predict(np.asarray(x).reshape(1, -1))  # pylint: disable=protected-access
            self.explainer._discretize(np.asarray(x).reshape(1, -1))  # pylint: disable=protected-access
            self.explainer.interval_learner.insert_calibration(np.asarray(x).reshape(1, -1), np.asarray([y]), bins=None)
        else:
            if 'mode' not in kwargs:
                if hasattr(self.learner, 'predict_proba'):
                    kwargs['mode'] = 'classification'
                else:
                    kwargs['mode'] = 'regression'
            self.calibrate(np.asarray(x).reshape(1, -1), np.asarray([y]), **kwargs)
        self.calibrated = True
        return self

    def calibrate_many(self, X, y, **kwargs):  # noqa: D401
        if not self.fitted:
            raise RuntimeError("The OnlineCalibratedExplainer must be fitted before calibration.")
        if self.calibrated:
            self.explainer.reinitialize(self.learner, X, y, **kwargs)
        else:
            if 'mode' not in kwargs:
                if hasattr(self.learner, 'predict_proba'):
                    kwargs['mode'] = 'classification'
                else:
                    kwargs['mode'] = 'regression'
            self.calibrate(X, y, **kwargs)
        self.calibrated = True
        return self

    def predict_one(self, x, **kwargs):  # noqa: D401
        x = np.asarray(x).reshape(1, -1)
        return self.predict(x, **kwargs)

    def predict_proba_one(self, x, **kwargs):  # noqa: D401
        x = np.asarray(x).reshape(1, -1)
        return self.predict_proba(x, **kwargs)

__all__ = ["OnlineCalibratedExplainer"]
