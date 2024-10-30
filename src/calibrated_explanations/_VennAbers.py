"""contains the VennAbers class which is used to calibrate the predictions of a model

"""
# pylint: disable=invalid-name, line-too-long, too-many-instance-attributes, too-many-arguments, too-many-positional-arguments
# flake8: noqa: E501
import warnings
import numpy as np
import venn_abers as va
from scipy.special import logit, expit

class VennAbers:
    """a class to calibrate the predictions of a model using the VennABERS method
    """
    def __init__(self, X_cal, y_cal, learner, bins=None, cprobs=None, difficulty_estimator=None):
        self.de = difficulty_estimator
        self.learner = learner
        self.X_cal = X_cal
        self.ctargets = y_cal
        self.__is_multiclass = len(np.unique(y_cal)) > 2
        cprobs = self.__predict_proba_with_difficulty(X_cal) if cprobs is None else cprobs
        self.cprobs = cprobs
        self.bins = bins
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.is_mondrian():
            self.va = {}
            if self.is_multiclass():
                tmp_probs = np.zeros((cprobs.shape[0],2))
                for c in np.unique(self.ctargets):
                    self.va[c] = {}
                    tmp_probs[:,0] = 1 - cprobs[:,c]
                    tmp_probs[:,1] = cprobs[:,c]
                    for b in np.unique(self.bins):
                        va_class_bin = va.VennAbers()
                        va_class_bin.fit(tmp_probs[self.bins == b,:], np.multiply(c == self.ctargets[self.bins == b], 1), precision=4)
                        self.va[c][b] = va_class_bin
            else:
                for b in np.unique(self.bins):
                    va_bin = va.VennAbers()
                    va_bin.fit(cprobs[self.bins == b,:], self.ctargets[self.bins == b], precision=4)
                    self.va[b] = va_bin
        elif self.is_multiclass():
            self.va = {}
            tmp_probs = np.zeros((cprobs.shape[0],2))
            for c in np.unique(self.ctargets):
                tmp_probs[:,0] = 1 - cprobs[:,c]
                tmp_probs[:,1] = cprobs[:,c]
                va_class = va.VennAbers()
                va_class.fit(tmp_probs, np.multiply(c == self.ctargets, 1), precision=4)
                self.va[c] = va_class
        else:
            self.va = va.VennAbers()
            self.va.fit(cprobs, self.ctargets, precision=4)
        warnings.filterwarnings("default", category=RuntimeWarning)

    def __predict_proba_with_difficulty(self, X, bins=None):
        if 'bins' in self.learner.predict_proba.__code__.co_varnames:
            probs = self.learner.predict_proba(X, bins=bins)
        else:
            probs = self.learner.predict_proba(X)
        if self.de is not None:
            difficulty = self.de.apply(X)
            # method = logit_based_scaling_list
            method = exponent_scaling_list
            # method = sigmoid_scaling_list
            if self.is_multiclass():
                probs_tmp = method(probs, difficulty)
            else:
                probs_tmp = method(probs, np.repeat(difficulty, 2).reshape(-1,2))
            probs = np.array([np.asarray(tmp) for tmp in probs_tmp])
        return probs

    def predict(self, X_test, bins=None):
        """a function to predict the class of the test samples

        Args:
            X_test (n_test_samples, n_features): test samples
            bins (array-like of shape (n_samples,), optional): Mondrian categories

        Returns:
            predicted classes (n_test_samples,): predicted classes based on the regularized VennABERS probabilities. If multiclass, the predicted class is 1 if the prediction from the underlying model is the same after calibration and 0 otherwise.
        """
        if self.is_multiclass():
            tmp, _ = self.predict_proba(X_test, bins=bins)
            return np.asarray(np.round(tmp[:,1]))
        tmp = self.predict_proba(X_test, bins=bins)[:,1]
        return np.asarray(np.round(tmp))

    # pylint: disable=too-many-locals, too-many-branches
    def predict_proba(self, X_test, output_interval=False, classes=None, bins=None):
        """a function to predict the probabilities of the test samples, optionally outputting the VennABERS interval

        Args:
            X_test (n_test_samples, n_features): test samples
            output_interval (bool, optional): if true, the VennAbers intervals are outputted. Defaults to False.
            classes ((n_test_samples,), optional): a list of predicted classes. Defaults to None.
            bins (array-like of shape (n_samples,), optional): Mondrian categories

        Returns:
            proba (n_test_samples,2): regularized VennABERS probabilities for the test samples. 
            if output_interval is true, the VennABERS intervals are also returned:
                low (n_test_samples,): lower bounds of the VennABERS interval for each test sample
                high (n_test_samples,): upper bounds of the VennABERS interval for each test sample
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        tprobs = self.__predict_proba_with_difficulty(X_test, bins=bins)
        p0p1 = np.zeros((tprobs.shape[0],2))
        va_proba = np.zeros(tprobs.shape)

        if self.is_multiclass():
            low, high = np.zeros(tprobs.shape), np.zeros(tprobs.shape)
            tmp_probs = np.zeros((tprobs.shape[0],2))
            for c, va_class in self.va.items():
                tmp_probs[:,0] = 1 - tprobs[:,c]
                tmp_probs[:,1] = tprobs[:,c]
                if self.is_mondrian():
                    assert bins is not None, "bins must be provided if Mondrian"
                    for b, va_class_bin in va_class.items():
                        p0p1[bins == b,:] = va_class_bin.predict_proba(tmp_probs[bins == b,:])[1]
                else:
                    p0p1 = va_class.predict_proba(tmp_probs)[1]
                low[:,c], high[:,c] = p0p1[:,0], p0p1[:,1]
                tmp = high[:,c] / (1-low[:,c] + high[:,c])
                va_proba[:,c] = tmp
            if classes is not None:
                if type(classes) not in (list, np.ndarray):
                    classes = [classes]
                if output_interval:
                    return np.asarray(va_proba), [low[i,c] for i,c in enumerate(classes)], [high[i,c] for i,c in enumerate(classes)], classes
                return np.asarray(va_proba), classes
            classes = np.argmax(va_proba, axis=1)
            if output_interval:
                return np.asarray(va_proba), low, high, classes
            return np.asarray(va_proba), classes

        if self.is_mondrian():
            assert bins is not None, "bins must be provided if Mondrian"
            for b, va_bin in self.va.items():
                p0p1[bins == b,:] = va_bin.predict_proba(tprobs[bins == b,:])[1]
        else:
            _, p0p1 = self.va.predict_proba(tprobs)
        low, high = p0p1[:,0], p0p1[:,1]
        tmp = high / (1-low + high)
        va_proba[:,0] = 1-tmp
        va_proba[:,1] = tmp
        # binary
        warnings.filterwarnings("default", category=RuntimeWarning)
        if output_interval:
            return np.asarray(va_proba), low, high
        return np.asarray(va_proba)

    def is_multiclass(self) -> bool:
        """returns true if more than two classes

        Returns:
            bool: true if more than two classes
        """
        return self.__is_multiclass

    def is_mondrian(self) -> bool:
        """returns true if Mondrian

        Returns:
            bool: true if Mondrian
        """
        return self.bins is not None

def mixture_of_experts_scaling(p, difficulty):
    """
    Use a mixture of experts approach to scale the probability.
    
    Args:
        p (float): Original predicted probability (between 0 and 1).
        difficulty (float): Difficulty of the instance (0 = easy, 1 = hard).
        
    Returns:
        float: Scaled probability.
    """
    # Weighting function based on difficulty
    weight = 1 - difficulty

    # Scale the probability as a mixture of the original and 0.5
    return weight * p + (1 - weight) * 0.5

def linear_interpolation_with_half(p, difficulty):
    """
    Linearly interpolate the probability with 0.5 based on difficulty.
    
    Args:
        p (float): Original predicted probability (between 0 and 1).
        difficulty (float): Difficulty of the instance (0 = easy, 1 = hard).
        
    Returns:
        float: Scaled probability.
    """
    # Interpolate between p and 0.5 based on difficulty
    return (1 - difficulty) * p + difficulty * 0.5

def temperature_scaling(p, difficulty, base_temperature=1.0):
    """
    Scale the probability using temperature scaling.
    
    Args:
        p (float): Original predicted probability (between 0 and 1).
        difficulty (float): Difficulty of the instance (0 = easy, 1 = hard).
        base_temperature (float): Base temperature value (default is 1.0).
        
    Returns:
        float: Scaled probability.
    """
    # Higher difficulty should increase temperature
    T = base_temperature + difficulty

    # Apply temperature scaling
    return p ** (1 / T) / (p ** (1 / T) + (1 - p) ** (1 / T))

def regularized_probability(p, difficulty, lambda_reg=1.0):
    """
    Regularize the probability based on difficulty.
    
    Args:
        p (float): Original predicted probability (between 0 and 1).
        difficulty (float): Difficulty of the instance (0 = easy, 1 = hard).
        lambda_reg (float): Regularization strength.
        
    Returns:
        float: Scaled probability.
    """
    # Regularize the probability towards 0.5
    return p - lambda_reg * difficulty * (p - 0.5)

def sigmoid_scaling_list(probs, difficulties, alpha=10):
    """
    Scale a list of probabilities using sigmoid-based scaling.
    
    Args:
        probs (list of float): List of predicted probabilities (between 0 and 1).
        difficulties (list of float): List of difficulties (0 = easy, 1 = hard).
        alpha (float): Controls the steepness of the sigmoid scaling (default is 10).
        
    Returns:
        list of float: Scaled probabilities.
    """
    scaled_probs = []
    for p, difficulty in zip(probs, difficulties):
        if p[0] < 0.5:
            scaled_p = p ** (1 + alpha * (1 - difficulty))
        else:
            scaled_p = 1 - (1 - p) ** (1 + alpha * (1 - difficulty))

        final_scaled_p = (1 - difficulty) * scaled_p + difficulty * 0.5
        final_scaled_p = final_scaled_p / np.sum(final_scaled_p)
        scaled_probs.append(final_scaled_p)

    return scaled_probs

def logit_based_scaling_list(probs, difficulties, gamma=2):
    """
    Use logit-based scaling to adjust a list of probabilities based on difficulties.
    
    Args:
        probs (list of float): List of predicted probabilities (between 0 and 1).
        difficulties (list of float): List of difficulties (0 = easy, 1 = hard).
        gamma (float): Scaling factor to control the effect of difficulty.
        
    Returns:
        list of float: Scaled probabilities.
    """
    scaled_probs = []
    for p, difficulty in zip(probs, difficulties):
        logit_p = logit(p)
        scaled_logit = logit_p * (1 - difficulty) * gamma
        scaled_p = expit(scaled_logit)
        final_scaled_p = (1 - difficulty) * scaled_p + difficulty * 0.5
        final_scaled_p = final_scaled_p / np.sum(final_scaled_p)
        scaled_probs.append(final_scaled_p)

    return scaled_probs

def exponent_scaling_list(probs, difficulties, beta=5):
    """
    Exponentially scale a list of probabilities towards 0/1 for low difficulty, and towards 0.5 for high difficulty.
    
    Args:
        probs (list of float): List of predicted probabilities (between 0 and 1).
        difficulties (list of float): List of difficulties (0 = easy, 1 = hard).
        beta (float): Scaling factor to control the effect of difficulty (default is 5).
        
    Returns:
        list of float: Scaled probabilities.
    """
    scaled_probs = []
    for p, difficulty in zip(probs, difficulties):
        if p[0] < 0.5:
            scaled_p = p ** (1 + beta * (1 - difficulty))
        else:
            scaled_p = 1 - (1 - p) ** (1 + beta * (1 - difficulty))

        final_scaled_p = (1 - difficulty) * scaled_p + difficulty * 0.5
        final_scaled_p = final_scaled_p / np.sum(final_scaled_p)
        scaled_probs.append(final_scaled_p)

    return scaled_probs
