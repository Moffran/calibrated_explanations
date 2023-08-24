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
        cps = crepes.ConformalPredictiveSystem()
        if self.ce.difficulty_estimator is not None:
            sigma_cal = self.ce.get_sigma_test(X=self.ce.cal_X)
            cps.fit(residuals=self.residual_cal, sigmas=sigma_cal)
        else:
            cps.fit(residuals=self.residual_cal)
        self.cps = cps
        self.venn_abers = None
        self.proba_cal = None
        self.y_threshold = None

    def predict_probability(self, test_X, y_threshold):
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
        
        Returns
        -------
            four values: proba (y <= y_threshold), lower bound, upper bound, and None.
        
        '''
        self.y_threshold = y_threshold
        if np.isscalar(self.y_threshold):
            self.compute_proba_cal(self.y_threshold)
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
        
        Returns
        -------
            four values: median, lower bound, upper bound, and None
                    
        '''
        test_y_hat = self.ce.model.predict(test_X)

        sigma_test = self.ce.get_sigma_test(X=test_X)
        low = [low_high_percentiles[0], 50] if low_high_percentiles[0] != -np.inf else [50, 50]
        high = [low_high_percentiles[1], 50] if low_high_percentiles[1] != np.inf else [50, 50]

        interval = self.cps.predict(y_hat=test_y_hat, sigmas=sigma_test,
                                    lower_percentiles=low,
                                    higher_percentiles=high)
        test_y_hat = (interval[:, 1] + interval[:, 3]) / 2  # The median
        return test_y_hat, \
            interval[:, 0] if low_high_percentiles[0] != -np.inf else np.array([min(self.ce.cal_y)]), \
            interval[:, 2] if low_high_percentiles[1] != np.inf else np.array([max(self.ce.cal_y)]), \
            None

    def predict_proba(self, test_X):
        '''The function `predict_proba` takes in a set of test data and returns the predicted probabilities
        for being above the y_threshold.
        
        Parameters
        ----------
        test_X
            The test_X parameter is the input data for which you want to predict the probabilities. It
        should be a numpy array or a pandas DataFrame containing the features of the test data.
        
        Returns
        -------
            a numpy array of shape (n_samples, 2), where each row represents the predicted probabilities
        for being above or below the y_threshold. The first column represents the probability of the 
        negative class (1-proba) and the second column represents the probability of the positive class (proba).
        
        '''
        test_y_hat = self.ce.model.predict(test_X)

        sigma_test = self.ce.get_sigma_test(X=test_X)
        proba = self.cps.predict(y_hat=test_y_hat, sigmas=sigma_test, y=self.y_threshold)
        return np.array([[1-proba[i], proba[i]] for i in range(len(proba))])

    def compute_proba_cal(self, y_threshold: float):
        '''The `compute_proba_cal` function calculates the probability calibration for a given threshold.
        
        Parameters
        ----------
        y_threshold : float
            The `y_threshold` parameter is a float value that represents the threshold for the probability.
        It is used in the `compute_proba_cal` method to determine the predicted probabilities of the 
        calibration set for a given threshold value.
        
        '''
        # A less exact but faster solution, suitable when difficulty_estimator is assigned.
        # Activated temporarily
        if self.ce.difficulty_estimator is not None:  
            sigmas = self.ce.get_sigma_test(self.ce.cal_X)
            proba = self.cps.predict(y_hat=self.cal_y_hat,
                                                y=y_threshold,
                                                sigmas=sigmas)
            self.proba_cal = np.array([[1-proba[i], proba[i]] for i in range(len(proba))])
        else:
            cps = crepes.ConformalPredictiveSystem()
            self.proba_cal = np.zeros((len(self.residual_cal),2))
            for i, _ in enumerate(self.residual_cal):
                idx = np.setdiff1d(np.arange(len(self.residual_cal)), i)
                sigma_cal = self.ce.get_sigma_test(self.ce.cal_X[idx, :])
                cps.fit(residuals=self.residual_cal[idx], sigmas=sigma_cal)
                sigma_i = self.ce.get_sigma_test(self.ce.cal_X[i, :].reshape(1, -1))
                self.proba_cal[i, 1] = cps.predict(y_hat=[self.cal_y_hat[i]],
                                                y=y_threshold,
                                                sigmas=sigma_i)
                self.proba_cal[i, 0] = 1 - self.proba_cal[i, 1]
        self.venn_abers = VennAbers(self.proba_cal, (self.ce.cal_y <= self.y_threshold).astype(int), self)
