# pylint: disable=invalid-name, line-too-long, too-many-instance-attributes
"""This module contains the class for the interval regressors.

Classes
-------
IntervalRegressor
    The main class for performing regression analysis on intervals of data.

Methods
-------
__init__(self, calibrated_explainer)
    Initialize an object with various attributes used for calibration and explanation extraction.
predict_probability(self, X_test, y_threshold, bins=None)
    Predict the probabilities for each instance in the dataset being above the threshold(s), along with confidence intervals.
predict_uncertainty(self, X_test, low_high_percentiles, bins=None)
    Predict the uncertainty of a given set of instances using a `ConformalPredictiveSystem`.
predict_proba(self, X_test, bins=None)
    Predict the probabilities for being above the y_threshold.
pre_fit_for_probabilistic(self)
    Split the calibration set into two parts.
compute_proba_cal(self, y_threshold: float)
    Calculate the probability calibration for a given threshold.
"""
from functools import singledispatchmethod
import crepes
import numpy as np
from ._VennAbers import VennAbers

class IntervalRegressor:
    """The IntervalRegressor class is used for regression analysis on intervals of data."""

    def __init__(self, calibrated_explainer):
        """Initialize an object with various attributes used for calibration and explanation extraction.

        Parameters
        ----------
        calibrated_explainer
            An object of the class `CalibratedExplainer` which is used to extract explanations. It is
            assumed that this object has been initialized and contains the necessary methods and attributes.
        model
            The `model` parameter is an object that represents a fitted regression model. It should have a
            `predict` method that can be used to make predictions on new instances.
        X_cal
            A numpy array containing the instance objects used for calibration. These are the input
            features for the model.
        y_cal
            The instance targets used for calibration. It is a numpy array that contains the true target
            values for the instances in the calibration set.
        """
        self.ce = calibrated_explainer
        self.bins = calibrated_explainer.bins
        self.model = self
        self.y_cal_hat = self.ce.predict_function(self.ce.X_cal)  # can be calculated through calibrated_explainer
        self.residual_cal = self.ce.y_cal - self.y_cal_hat  # can be calculated through calibrated_explainer
        self.sigma_cal = self.ce._get_sigma_test(X=self.ce.X_cal)  # pylint: disable=protected-access
        cps = crepes.ConformalPredictiveSystem()
        if self.ce.difficulty_estimator is not None:
            cps.fit(residuals=self.residual_cal, sigmas=self.sigma_cal, bins=self.bins, seed=self.ce.seed)
        else:
            cps.fit(residuals=self.residual_cal, bins=self.bins, seed=self.ce.seed)
        self.cps = cps
        self.venn_abers = None
        self.proba_cal = None
        self.y_threshold = None
        self.current_y_threshold = None
        self.split = {}
        self.pre_fit_for_probabilistic()

    # pylint: disable=too-many-locals
    def predict_probability(self, X_test, y_threshold, bins=None):
        """Predict the probabilities for each instance in the dataset being above the threshold(s), along with confidence intervals.

        Parameters
        ----------
        X_test
            X_test is a numpy.ndarray containing the instance objects for which we want to predict the
            probability.
        y_threshold
            The `y_threshold` parameter is used to determine the probability of the true value being 
            below the threshold value. If the predicted probability of the positive class is smaller than or 
            equal to `y_threshold`, the instance is classified as positive; otherwise, it is classified as negative.
        bins 
            array-like of shape (n_samples,), default=None
            Mondrian categories

        Returns
        -------
            four values: proba (y <= y_threshold), lower bound, upper bound, and None.
        """
        if bins is not None:
            assert self.bins is not None, 'Calibration bins must be assigned when test bins are submitted.'
        self.y_threshold = y_threshold
        if np.isscalar(self.y_threshold) or isinstance(self.y_threshold, tuple):
            self.current_y_threshold = self.y_threshold
            self.compute_proba_cal(self.y_threshold)
            proba, low, high = self.split['va'].predict_proba(X_test, output_interval=True, bins=bins)
            return proba[:, 1], low, high, None

        bins = bins if bins is not None else [None]*X_test.shape[0]
        interval = np.zeros((X_test.shape[0],2))
        proba = np.zeros(X_test.shape[0])
        for i, _ in enumerate(proba):
            self.current_y_threshold = self.y_threshold[i]
            self.compute_proba_cal(self.y_threshold[i])
            p, low, high = self.split['va'].predict_proba(X_test[i, :].reshape(1, -1), output_interval=True, bins=[bins[i]])
            p = p[0,1]
            low = low[0]
            high = high[0]
            proba[i] = p
            interval[i, :] = np.array([low, high])
        return proba, interval[:, 0], interval[:, 1], None

    def _predict_tuple_interval(self, X_test, threshold, bins):
        h_threshold = np.max(threshold)
        self.current_y_threshold = h_threshold
        self.compute_proba_cal(h_threshold)
        _, low_h, high_h = self.split['va'].predict_proba(X_test, output_interval=True, bins=bins)
        l_threshold = np.min(threshold)
        self.current_y_threshold = l_threshold
        self.compute_proba_cal(l_threshold)
        _, low_l, high_l = self.split['va'].predict_proba(X_test, output_interval=True, bins=bins)
        low_ = low_h-low_l
        high_ = high_h-high_l
        low = np.min(np.array([low_, high_]), axis=0)
        high = np.max(np.array([low_, high_]), axis=0)
        proba = high / (1-low + high)
        assert np.all([low[i] <= proba[i] <= high[i] for i in range(len(low))]), 'Lower bound must be less than or equal to upper bound, with proba in the middle.'
        return proba, low, high, None

    def predict_uncertainty(self, X_test, low_high_percentiles, bins=None):
        """Predict the uncertainty of a given set of instances using a `ConformalPredictiveSystem`.

        Parameters
        ----------
        X_test
            X_test is a numpy array containing the instance objects for which we want to predict the
            uncertainty.
        low_high_percentiles
            The `low_high_percentiles` parameter is a list containing two values. The first value
            represents the lower percentile and the second value represents the higher percentile. These
            percentiles are used to calculate the prediction interval for the uncertainty estimation. If the
            first value is set to -np.inf (negative infinity), the interval will be one-sided and upper-bounded 
            and if the second value is np.inf (infinity), the interval will be one-sided and lower-bounded.        
        bins 
            array-like of shape (n_samples,), default=None
            Mondrian categories

        Returns
        -------
            four values: median, lower bound, upper bound, and None.
        """
        y_test_hat = self.ce.predict_function(X_test)

        sigma_test = self.ce._get_sigma_test(X=X_test)  # pylint: disable=protected-access
        low = [low_high_percentiles[0], 50] if low_high_percentiles[0] != -np.inf else [50, 50]
        high = [low_high_percentiles[1], 50] if low_high_percentiles[1] != np.inf else [50, 50]

        interval = self.cps.predict(y_hat=y_test_hat, sigmas=sigma_test,
                                    lower_percentiles=low,
                                    higher_percentiles=high,
                                    bins=bins)
        y_test_hat = (interval[:, 1] + interval[:, 3]) / 2  # The median
        return y_test_hat, \
            interval[:, 0] if low_high_percentiles[0] != -np.inf else np.tile(np.array([np.min(self.ce.y_cal)]), len(interval)), \
            interval[:, 2] if low_high_percentiles[1] != np.inf else np.tile(np.array([np.max(self.ce.y_cal)]), len(interval)), \
            None

    def predict_proba(self, X_test, bins=None):
        """Predict the probabilities for being below the y_threshold (for float threshold) or below the lower bound and above the upper bound (for tuple threshold).

        Parameters
        ----------
        X_test
            The X_test parameter is the input data for which you want to predict the probabilities. It
            should be a numpy array or a pandas DataFrame containing the features of the test data.     
        bins 
            array-like of shape (n_samples,), default=None
            Mondrian categories

        Returns
        -------
            a numpy array of shape (n_samples, 2), where each row represents the predicted probabilities
            for being above or below the y_threshold. The first column represents the probability of the 
            negative class (1-proba) and the second column represents the probability of the positive class (proba).
        """
        y_test_hat = self.ce.predict_function(X_test)

        sigma_test = self.ce._get_sigma_test(X=X_test)  # pylint: disable=protected-access
        if isinstance(self.current_y_threshold, tuple):
            proba_lower = self.cps.predict(y_hat=y_test_hat, sigmas=sigma_test, y=self.current_y_threshold[0], bins=bins)
            proba_upper = self.cps.predict(y_hat=y_test_hat, sigmas=sigma_test, y=self.current_y_threshold[1], bins=bins)
            proba = proba_upper - proba_lower
        else:
            proba = self.cps.predict(y_hat=y_test_hat, sigmas=sigma_test, y=self.current_y_threshold, bins=bins)
        return np.array([[1-proba[i], proba[i]] for i in range(len(proba))])

    def pre_fit_for_probabilistic(self):
        """Split the calibration set into two parts.

        The first part is used to fit the `ConformalPredictiveSystem` and the second part is used to
        calculate the probability calibration for a given threshold (at prediction time).
        """
        n = len(self.ce.y_cal)
        cal_parts = np.random.permutation(n).tolist()
        self.split['parts'] = [cal_parts[:n//2], cal_parts[n//2:]]
        cal_cps = self.split['parts'][0]
        self.split['cps'] = crepes.ConformalPredictiveSystem()
        if self.bins is None:
            self.split['cps'].fit(residuals=self.residual_cal[cal_cps],
                            sigmas=self.sigma_cal[cal_cps], seed=self.ce.seed)
        else:
            self.split['cps'].fit(residuals=self.residual_cal[cal_cps],
                            sigmas=self.sigma_cal[cal_cps],
                            bins=self.bins[cal_cps], seed=self.ce.seed)

    @singledispatchmethod
    def compute_proba_cal(self, y_threshold):
        """Base method for computing the probability calibration.

        Parameters
        ----------
        y_threshold : float or tuple
            The `y_threshold` parameter is a float or tuple value that represents the threshold for the probability.
            It is used in the `compute_proba_cal` method to determine the predicted probabilities of the 
            calibration set for a given threshold value.     
        """
        raise TypeError('y_threshold must be a float or a tuple.')

    @compute_proba_cal.register(float)
    def _(self, y_threshold: float):
        """Calculate the probability calibration for a given threshold.

        Parameters
        ----------
        y_threshold : float
            The `y_threshold` parameter is a float value that represents the threshold for the probability.
            It is used in the `compute_proba_cal` method to determine the predicted probabilities of the 
            calibration set for a given threshold value.     
        """
        cal_va = self.split['parts'][1]
        bins = None if self.bins is None else self.bins[cal_va]
        proba = self.split['cps'].predict(y_hat=self.y_cal_hat[cal_va],
                                y=y_threshold,
                                sigmas=self.sigma_cal[cal_va],
                                bins=bins)
        self.split['proba'] = np.array([[1-proba[i], proba[i]] for i in range(len(proba))])
        self.split['va'] = VennAbers(None,
                                        (self.ce.y_cal[cal_va] <= y_threshold).astype(int),
                                        self,
                                        bins=bins,
                                        cprobs=self.split['proba'])

    @compute_proba_cal.register(tuple)
    def _(self, y_threshold: tuple):
        """Calculate the probability calibration for a given interval threshold.

        Parameters
        ----------
        y_threshold : tuple
            The `y_threshold` parameter is a tuple that represents the interval threshold for the probability.
            It is used in the `compute_proba_cal` method to determine the predicted probabilities of the 
            calibration set for a given threshold value.     
        """
        cal_va = self.split['parts'][1]
        bins = None if self.bins is None else self.bins[cal_va]
        proba_lower = self.split['cps'].predict(y_hat=self.y_cal_hat[cal_va],
                                y=y_threshold[0],
                                sigmas=self.sigma_cal[cal_va],
                                bins=bins)
        proba_upper = self.split['cps'].predict(y_hat=self.y_cal_hat[cal_va],
                                y=y_threshold[1],
                                sigmas=self.sigma_cal[cal_va],
                                bins=bins)
        proba = proba_upper - proba_lower
        self.split['proba'] = np.array([[1-proba[i], proba[i]] for i in range(len(proba))])
        self.split['va'] = VennAbers(None,
                                        (y_threshold[0] < self.ce.y_cal[cal_va]) & (self.ce.y_cal[cal_va] <= y_threshold[1]).astype(int),
                                        self,
                                        bins=bins,
                                        cprobs=self.split['proba'])
