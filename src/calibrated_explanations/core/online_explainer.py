"""Mechanical extraction of OnlineCalibratedExplainer (Phase 1A).

Verbatim move from legacy core.py (no semantic changes)."""

# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-lines, too-many-positional-arguments, too-many-public-methods
from __future__ import annotations

import numpy as np

from .wrap_explainer import WrapCalibratedExplainer  # fixed import


class OnlineCalibratedExplainer(WrapCalibratedExplainer):
    """Calibrated Explanations for Online Learning.

    This class extends WrapCalibratedExplainer to support online/incremental learning.
    It maintains compatibility with scikit-learn style interfaces while allowing
    incremental updates to both the model and calibration.

    The calibrated explanations are updated incrementally as new data arrives, making it suitable for streaming
    data scenarios where the model needs to continuously learn and adapt.
    """

    def fit(self, X_proper_train, y_proper_train, **kwargs):
        """Fit the learner to the training data.

        Parameters
        ----------
        X_proper_train : array-like of shape (n_samples, n_features)
            The training input samples in sklearn-compatible format.
        y_proper_train : array-like of shape (n_samples,)
            The target values.
        **kwargs : dict
            Additional arguments passed to the underlying learner's fit method.

        Returns
        -------
        self
            The fitted explainer.
        """
        reinitialize = bool(self.calibrated)
        self.fitted = False
        self.calibrated = False

        if hasattr(self.learner, "fit"):
            self.learner.fit(X_proper_train, y_proper_train, **kwargs)
        else:
            if "classes" not in kwargs:
                kwargs["classes"] = np.unique(y_proper_train)
            try:
                self.learner.partial_fit(X_proper_train, y_proper_train, **kwargs)
            except TypeError:
                kwargs.pop("classes", None)
                self.learner.partial_fit(X_proper_train, y_proper_train, **kwargs)

        return self._finalize_fit(reinitialize)

    def partial_fit(self, X, y, **kwargs):
        """Incrementally fit the model with samples X and y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data in sklearn-compatible format.
        y : array-like of shape (n_samples,)
            Target values.
        **kwargs : dict
            Additional arguments passed to the learner's partial_fit method.

        Returns
        -------
        self
            The updated explainer.

        Raises
        ------
        AttributeError
            If the underlying learner does not support incremental learning.
        """
        if not hasattr(self.learner, "partial_fit"):
            raise AttributeError("The learner must implement partial_fit for incremental learning")
        if np.isscalar(y):
            X = np.asarray(X).reshape(1, -1)
            y = np.asarray(y).reshape(1)
        self.learner.partial_fit(X, y, **kwargs)
        self.fitted = True
        return self

    def online_fit_and_calibrate(self, X, y, **kwargs):
        """Incrementally fit and calibrate the model with samples X and y. Calls partial_fit and calibrate_many.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data in sklearn-compatible format.
        y : array-like of shape (n_samples,)
            Target values.
        **kwargs : dict
            Additional arguments passed to the learner's partial_fit method.

        Returns
        -------
        self
            The updated explainer.

        Raises
        ------
        AttributeError
            If the underlying learner does not support incremental learning.
        """
        pre_pred = (
            self.explainer.predict_function(self.explainer.X_cal) if self.calibrated else None
        )
        try:
            self.partial_fit(X, y, **kwargs)
        except AttributeError:
            self.partial_fit(X, y)
        post_pred = (
            self.explainer.predict_function(self.explainer.X_cal) if self.calibrated else None
        )
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
            if "mode" not in kwargs:
                if hasattr(self.learner, "predict_proba"):
                    kwargs["mode"] = "classification"
                else:
                    kwargs["mode"] = "regression"
            self.calibrate(X, y, **kwargs)

    def calibrate_one(self, x, y, **kwargs):
        """Update the calibration set with a single instance.

        Parameters
        ----------
        x : array-like of shape (1, n_features)
            Single instance to calibrate with in sklearn-compatible format.
        y : array-like of shape (1,)
            The target value for the instance.
        **kwargs : dict
            Additional arguments passed to calibrate_many.

        Returns
        -------
        self
            The updated explainer.
        """
        x = np.asarray(x).reshape(1, -1)
        y = np.asarray(y).reshape(1)
        return self.calibrate_many(x, y, **kwargs)

    def calibrate_many(self, X, y, **kwargs):
        """Update the calibration set with multiple instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Multiple instances to calibrate with in sklearn-compatible format.
        y : array-like of shape (n_samples,)
            The target values for the instances.
        **kwargs : dict
            Additional arguments passed to the calibrate method.

        Returns
        -------
        self
            The updated explainer.

        Raises
        ------
        RuntimeError
            If the explainer has not been fitted before calling this method.
        """
        if not self.fitted:
            raise RuntimeError("The OnlineCalibratedExplainer must be fitted before calibration.")

        if self.calibrated:
            self.explainer.reinitialize(self.learner, X, y, **kwargs)
        else:
            if "mode" not in kwargs:
                if hasattr(self.learner, "predict_proba"):
                    kwargs["mode"] = "classification"
                else:
                    kwargs["mode"] = "regression"
            self.calibrate(X, y, **kwargs)

        self.calibrated = True
        return self

    def predict_one(self, x, **kwargs):
        """Predict target for a single instance.

        Parameters
        ----------
        x : array-like of shape (1, n_features)
            Single instance in sklearn-compatible format.
        **kwargs : dict
            Additional arguments passed to predict.

        Returns
        -------
        array-like
            Predicted value(s).
        """
        x = np.asarray(x).reshape(1, -1)
        return self.predict(x, **kwargs)

    def predict_proba_one(self, x, **kwargs):
        """Predict class probabilities for a single instance.

        Parameters
        ----------
        x : array-like of shape (1, n_features)
            Single instance in sklearn-compatible format.
        **kwargs : dict
            Additional arguments passed to predict_proba.

        Returns
        -------
        array-like
            Predicted probabilities.
        """
        x = np.asarray(x).reshape(1, -1)
        return self.predict_proba(x, **kwargs)


__all__ = ["OnlineCalibratedExplainer"]
