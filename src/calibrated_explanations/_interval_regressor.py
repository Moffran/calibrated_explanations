# pylint: disable=invalid-name, line-too-long, too-many-instance-attributes
# flake8: noqa: E501
"""
This module contains the class for the interval regressors.
"""
import crepes
import numpy as np
from .VennAbers import VennAbers


class IntervalRegressor:
    """
    Regressor
    """
    def __init__(self, calibrated_explainer, model, cal_X, cal_y):
        """
        Parameters
        ----------
        model : object
            A fitted regression model object that has a predict method.
        cal_X : numpy.ndarray
            The instance objects used for calibration.
        cal_y : numpy.ndarray
            The instance targets used for calibration.
        """
        self.calibrated_explainer = calibrated_explainer
        self.model = self
        self.predictor = model
        self.cal_X = cal_X
        self.cal_y = cal_y
        self.cal_y_hat = self.predictor.predict(cal_X)
        self.residual_cal = cal_y - self.cal_y_hat
        cps = crepes.ConformalPredictiveSystem()
        if self.calibrated_explainer.difficulty_estimator is not None:
            sigma_cal = self.calibrated_explainer.difficulty_estimator.apply(X=cal_X)
            cps.fit(residuals=self.residual_cal, sigmas=sigma_cal)
        else:
            cps.fit(residuals=self.residual_cal)
        self.cps = cps
        self.venn_abers = None
        self.proba_cal = None
        self.y_threshold = None

    def predict_probability(self, test_X, y_threshold):
        """
        Parameters
        ----------
        X : numpy.ndarray
            The instance objects for which to predict the probability.
        """
        self.assign_threshold(y_threshold)
        # proba = self.predict_proba(test_X)[:,1]
        if np.isscalar(self.y_threshold):
            proba, low, high = self.venn_abers.predict_proba(test_X, output_interval=True)
            return proba[:, 1], low, high, None

        interval = np.array([np.array([0.0, 0.0]) for i in range(test_X.shape[0])])
        proba = np.zeros(test_X.shape[0])
        for i, _ in enumerate(proba):
            self.compute_proba_cal(self.y_threshold[i])
            p, low, high = self.venn_abers.predict_proba(test_X[i, :].reshape(-1, 1), output_interval=True)
            proba[i] = p[1]
            interval[i, :] = np.array([low, high])
        return proba, interval[:, 0], interval[:, 1], None

    def predict_uncertainty(self, test_X, low_high_percentiles):
        """
        Parameters
        ----------
        X : numpy.ndarray
            The instance objects for which to predict the uncertainty.
        """
        predict = self.predictor.predict(test_X)

        sigma_test = self.calibrated_explainer.get_sigma_test(X=test_X)
        low = [low_high_percentiles[0], 50] if low_high_percentiles[0] != -np.inf else [50, 50]
        high = [low_high_percentiles[1], 50] if low_high_percentiles[1] != np.inf else [50, 50]

        interval = self.cps.predict(y_hat=predict, sigmas=sigma_test,
                                    lower_percentiles=low,
                                    higher_percentiles=high)
        predict = (interval[:, 1] + interval[:, 3]) / 2  # The median
        return predict, \
            interval[:, 0] if low_high_percentiles[0] != -np.inf else np.array([min(self.cal_y)]), \
            interval[:, 2] if low_high_percentiles[1] != np.inf else np.array([max(self.cal_y)]), \
            None

    def predict_proba(self, test_X):
        """_summary_

        Parameters
        ----------
        X : numpy.ndarray
            The instance objects for which to predict the probability.

        Returns
        -------
        proba : numpy.ndarray
            The predicted probabilities of being above y.
        """
        predict = self.predictor.predict(test_X)

        sigma_test = self.calibrated_explainer.get_sigma_test(X=test_X)
        proba = self.cps.predict(y_hat=predict, sigmas=sigma_test, y=self.y_threshold)
        return np.array([[1-proba[i], proba[i]] for i in range(len(proba))])


    def assign_threshold(self, y_threshold):
        """
        Parameters
        ----------
        y_threshold : float or numpy.ndarray
            The threshold for the probability.
        """
        self.y_threshold = y_threshold
        if np.isscalar(self.y_threshold):
            self.compute_proba_cal(y_threshold)

    def compute_proba_cal(self, y_threshold: float):
        """_summary_

        Parameters
        ----------
        y_threshold : float
            The threshold for the probability.
        """
        cps = crepes.ConformalPredictiveSystem()
        self.proba_cal = np.zeros((len(self.residual_cal),2))
        for i, _ in enumerate(self.residual_cal):
            idx = np.setdiff1d(np.arange(len(self.residual_cal)), i)
            if self.calibrated_explainer.difficulty_estimator is not None:
                sigma_cal = self.calibrated_explainer.difficulty_estimator.apply(X=self.cal_X[idx, :])
                cps.fit(residuals=self.residual_cal[idx], sigmas=sigma_cal)
            else:
                cps.fit(residuals=self.residual_cal[idx])
            sigma_i = self.calibrated_explainer.get_sigma_test(self.cal_X[i, :].reshape(1, -1))
            self.proba_cal[i, 1] = cps.predict(y_hat=[self.cal_y_hat[i]],
                                            y=y_threshold,
                                            sigmas=sigma_i)
            self.proba_cal[i, 0] = 1 - self.proba_cal[i, 1]
        self.venn_abers = VennAbers(self.proba_cal, (self.cal_y <= self.y_threshold).astype(int), self)
