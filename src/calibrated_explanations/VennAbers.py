"""contains the VennAbers class which is used to calibrate the predictions of a model

"""
# pylint: disable=invalid-name, line-too-long
# flake8: noqa: E501
import numpy as np
from sklearn.isotonic import IsotonicRegression



def VennABERS_by_def(calibration, test):
# Function copied from https://github.com/ptocca/VennABERS/blob/master/test/VennABERS_test.ipynb
    """a function to compute the VennABERS score

    Args:
        calibration (n_calibration_samples,): the probabilities of the positive class for the calibration samples
        test (n_test_samples,): the probabilities of the positive class for the test samples

    Returns:
        lower_bounds (n_test_samples,): lower bounds of the VennABERS interval for each test sample
        upper_bounds (n_test_samples,): upper bounds of the VennABERS interval for each test sample
    """
    p0,p1 = [],[]
    for x in test:
        ds0 = calibration+[(x,0)]
        iso0 = IsotonicRegression().fit(*zip(*ds0))
        p0.append(iso0.predict([x]))

        ds1 = calibration+[(x,1)]
        iso1 = IsotonicRegression().fit(*zip(*ds1))
        p1.append(iso1.predict([x]))
    return np.array(p0).flatten(),np.array(p1).flatten()

class VennAbers:
    """a class to calibrate the predictions of a model using the VennABERS method
    """
    iso = IsotonicRegression(out_of_bounds="clip")

    def __init__(self, cal_X, cal_y, model):
        self.cprobs = model.predict_proba(cal_X)
        self.ctargets = cal_y
        self.model = model

    def predict(self, test_X):
        """a function to predict the class of the test samples

        Args:
            test_X (n_test_samples, n_features): test samples

        Returns:
            predicted classes (n_test_samples,): predicted classes based on the regularized VennABERS probabilities
        """
        cprobs, predict = self.get_p_value(self.cprobs)
        targets = np.multiply(predict == self.ctargets, 1) if self.is_multiclass() else self.ctargets
        tprobs, _ = self.get_p_value(self.model.predict_proba(test_X))
        low,high = VennABERS_by_def(list(zip(cprobs,targets)),tprobs)
        tmp = high / (1-low + high)
        return np.asarray(np.round(tmp))

    def predict_proba(self, test_X, output_interval=False, classes=None):
        """a function to predict the probabilities of the test samples, optionally outputting the VennABERS interval

        Args:
            testX (n_test_samples, n_features): test samples
            output_interval (bool, optional): if true, the VennAbers intervals are outputted. Defaults to False.
            classes ((n_test_samples,), optional): a list of predicted classes. Defaults to None.

        Returns:
            proba (n_test_samples,2): regularized VennABERS probabilities for the test samples. 
            if output_interval is true, the VennABERS intervals are also returned:
                low (n_test_samples,): lower bounds of the VennABERS interval for each test sample
                high (n_test_samples,): upper bounds of the VennABERS interval for each test sample
        """
        va_proba = self.model.predict_proba(test_X)
        cprobs, predict = self.get_p_value(self.cprobs)
        targets = np.multiply(predict == self.ctargets, 1) if self.is_multiclass() else self.ctargets
        tprobs, classes = self.get_p_value(va_proba, classes)
        low,high = VennABERS_by_def(list(zip(cprobs,targets)),tprobs)
        tmp = high / (1-low + high)
        va_proba[:,0] = 1-tmp
        va_proba[:,1] = tmp
        if self.is_multiclass():
            va_proba = va_proba[:,:2]
            if output_interval:
                return np.asarray(va_proba), low, high, classes
            return np.asarray(va_proba), classes
        else: # binary
            if output_interval:
                return np.asarray(va_proba), low, high
            return np.asarray(va_proba)

    def get_p_value(self, proba, classes=None):
        """return probability for the positive class when binary classification and for the most 
        probable class otherwise
        """
        if classes is None:
            return np.max(proba, axis=1) if self.is_multiclass() else proba[:,1], np.argmax(proba, axis=1)
        return proba[:,classes], classes

    def is_multiclass(self) -> bool:
        """returns true if more than two classes

        Returns:
            bool: true if more than two classes
        """
        return len(self.cprobs[0,:]) > 2
