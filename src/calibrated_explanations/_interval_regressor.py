# pylint: disable=invalid-name, line-too-long, too-many-instance-attributes
# flake8: noqa: E501
"""
This module contains the class for the interval regressors.
"""
import crepes
import numpy as np
from .VennAbers import VennAbers

class IntervalRegressor:
    """The IntervalRegressor class is used for regression analysis on intervals of data.
    """
    def __init__(self, calibrated_explainer):
        '''This function initializes an object with various attributes used for calibration and explanation
        extraction.
        
        Parameters
        ----------
        calibrated_explainer
            An object of the class `CalibratedExplainer` which is used to extract explanations. It is
        assumed that this object has been initialized and contains the necessary methods and attributes.
        model
            The `model` parameter is an object that represents a fitted regression model. It should have a
        `predict` method that can be used to make predictions on new instances.
        cal_X
            A numpy array containing the instance objects used for calibration. These are the input
        features for the model.
        cal_y
            The instance targets used for calibration. It is a numpy array that contains the true target
        values for the instances in the calibration set.
        
        '''
        self.ce = calibrated_explainer
        self.model = self
        self.cal_y_hat = self.ce.model.predict(self.ce.cal_X)  # can be calculated through calibrated_explainer
        self.residual_cal = self.ce.cal_y - self.cal_y_hat  # can be calculated through calibrated_explainer
        self.sigma_cal = self.ce._get_sigma_test(X=self.ce.cal_X)  # pylint: disable=protected-access
        cps = crepes.ConformalPredictiveSystem()
        if self.ce.difficulty_estimator is not None:
            cps.fit(residuals=self.residual_cal, sigmas=self.sigma_cal, bins=self.ce.bins)
        else:
            cps.fit(residuals=self.residual_cal, bins=self.ce.bins)
        self.cps = cps
        self.venn_abers = None
        self.proba_cal = None
        self.y_threshold = None
        self.current_y_threshold = None
        self.split = {}
        self.pre_fit_for_probabilistic()

    def predict_probability(self, test_X, y_threshold, bins=None):
        '''The `predict_probability` function takes in a test dataset and a threshold value, and returns
        the predicted probabilities for each instance in the dataset being above the threshold(s), along 
        with confidence intervals.
        
        Parameters
        ----------
        test_X
            test_X is a numpy.ndarray containing the instance objects for which we want to predict the
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
        
        '''
        self.y_threshold = y_threshold
        if np.isscalar(self.y_threshold):
            self.current_y_threshold = self.y_threshold
            if bins is not None:
                assert self.ce.bins is not None, 'Calibration bins must be assigned when test bins are submitted.'
            self.compute_proba_cal(self.y_threshold)
            proba, low, high = self.split['va'].predict_proba(test_X, output_interval=True, bins=bins)
            return proba[:, 1], low, high, None

        interval = np.zeros((test_X.shape[0],2))
        proba = np.zeros(test_X.shape[0])
        for i, _ in enumerate(proba):
            self.current_y_threshold = self.y_threshold[i]
            self.compute_proba_cal(self.y_threshold[i])
            p, low, high = self.split['va'].predict_proba(test_X[i, :].reshape(1, -1), output_interval=True, bins=bins)
            proba[i] = p[0,1]
            interval[i, :] = np.array([low[0], high[0]])
        return proba, interval[:, 0], interval[:, 1], None

    def predict_uncertainty(self, test_X, low_high_percentiles, bins=None):
        '''The function `predict_uncertainty` predicts the uncertainty of a given set of instances using a
        `ConformalPredictiveSystem` and returns the predicted values along with the lower and upper bounds of
        the uncertainty interval.
        
        Parameters
        ----------
        test_X
            test_X is a numpy array containing the instance objects for which we want to predict the
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
            four values: median, lower bound, upper bound, and None
                    
        '''
        test_y_hat = self.ce.model.predict(test_X)

        sigma_test = self.ce._get_sigma_test(X=test_X)  # pylint: disable=protected-access
        low = [low_high_percentiles[0], 50] if low_high_percentiles[0] != -np.inf else [50, 50]
        high = [low_high_percentiles[1], 50] if low_high_percentiles[1] != np.inf else [50, 50]

        interval = self.cps.predict(y_hat=test_y_hat, sigmas=sigma_test,
                                    lower_percentiles=low,
                                    higher_percentiles=high,
                                    bins=bins)
        test_y_hat = (interval[:, 1] + interval[:, 3]) / 2  # The median
        return test_y_hat, \
            interval[:, 0] if low_high_percentiles[0] != -np.inf else np.array([np.min(self.ce.cal_y)]), \
            interval[:, 2] if low_high_percentiles[1] != np.inf else np.array([np.max(self.ce.cal_y)]), \
            None

    def predict_proba(self, test_X, bins=None):
        '''The function `predict_proba` takes in a set of test data and returns the predicted probabilities
        for being above the y_threshold.
        
        Parameters
        ----------
        test_X
            The test_X parameter is the input data for which you want to predict the probabilities. It
        should be a numpy array or a pandas DataFrame containing the features of the test data.     
        bins 
            array-like of shape (n_samples,), default=None
            Mondrian categories
        
        Returns
        -------
            a numpy array of shape (n_samples, 2), where each row represents the predicted probabilities
        for being above or below the y_threshold. The first column represents the probability of the 
        negative class (1-proba) and the second column represents the probability of the positive class (proba).
        
        '''
        test_y_hat = self.ce.model.predict(test_X)

        sigma_test = self.ce._get_sigma_test(X=test_X)  # pylint: disable=protected-access
        proba = self.cps.predict(y_hat=test_y_hat, sigmas=sigma_test, y=self.current_y_threshold, bins=bins)
        return np.array([[1-proba[i], proba[i]] for i in range(len(proba))])

    def pre_fit_for_probabilistic(self):
        '''
        The `pre_fit_for_probabilistic` function is used to split the calibration set into two parts. 
        The first part is used to fit the `ConformalPredictiveSystem` and the second part is used to
        calculate the probability calibration for a given threshold (at prediction time).
        '''
        n = len(self.ce.cal_y)
        cal_parts = np.random.permutation(n).tolist()
        self.split['parts'] = [cal_parts[:n//2], cal_parts[n//2:]]
        cal_cps = self.split['parts'][0]
        self.split['cps'] = crepes.ConformalPredictiveSystem()
        if self.ce.bins is None:
            self.split['cps'].fit(residuals=self.residual_cal[cal_cps],
                            sigmas=self.sigma_cal[cal_cps])
        else:
            self.split['cps'].fit(residuals=self.residual_cal[cal_cps],
                            sigmas=self.sigma_cal[cal_cps],
                            bins=self.ce.bins[cal_cps])

    def compute_proba_cal(self, y_threshold: float):
        '''The `compute_proba_cal` function calculates the probability calibration for a given threshold.
        
        Parameters
        ----------
        y_threshold : float
            The `y_threshold` parameter is a float value that represents the threshold for the probability.
        It is used in the `compute_proba_cal` method to determine the predicted probabilities of the 
        calibration set for a given threshold value.     
        bins 
            array-like of shape (n_samples,), default=None
            Mondrian categories
        
        '''
        cal_va = self.split['parts'][1]
        if self.ce.bins is None:
            bins = None
        else:
            bins = self.ce.bins[cal_va]
        proba = self.split['cps'].predict(y_hat=self.cal_y_hat[cal_va],
                                y=y_threshold,
                                sigmas=self.sigma_cal[cal_va],
                                bins=bins)
        self.split['proba'] = np.array([[1-proba[i], proba[i]] for i in range(len(proba))])
        self.split['va'] = VennAbers(self.split['proba'],
                                        (self.ce.cal_y[cal_va] <= y_threshold).astype(int),
                                        self,
                                        bins=bins)
