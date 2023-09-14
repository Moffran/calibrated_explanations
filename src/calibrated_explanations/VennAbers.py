"""contains the VennAbers class which is used to calibrate the predictions of a model

"""
# pylint: disable=invalid-name, line-too-long
# flake8: noqa: E501
import numpy as np
import venn_abers as va

class VennAbers:
    """a class to calibrate the predictions of a model using the VennABERS method
    """
    def __init__(self, cal_probs, cal_y, model):
        self.cprobs = cal_probs
        self.ctargets = cal_y
        self.model = model
        self.va = va.VennAbers()
        cprobs, predict = self.get_p_value(self.cprobs)
        self.va.fit(cprobs, np.multiply(predict == self.ctargets, 1) if self.is_multiclass() else self.ctargets)

    def predict(self, test_X):
        """a function to predict the class of the test samples

        Args:
            test_X (n_test_samples, n_features): test samples

        Returns:
            predicted classes (n_test_samples,): predicted classes based on the regularized VennABERS probabilities
        """
        tprobs, _ = self.get_p_value(self.model.predict_proba(test_X))
        _, p0p1 = self.va.predict_proba(tprobs)
        low, high = p0p1[:,0], p0p1[:,1]
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
        tprobs, classes = self.get_p_value(va_proba, classes)
        _,p0p1 = self.va.predict_proba(tprobs)
        low, high = p0p1[:,0], p0p1[:,1]
        tmp = high / (1-low + high)
        va_proba[:,0] = 1-tmp
        va_proba[:,1] = tmp
        if self.is_multiclass():
            va_proba = va_proba[:,:2]
            if output_interval:
                return np.asarray(va_proba), low, high, classes
            return np.asarray(va_proba), classes
        # binary
        if output_interval:
            return np.asarray(va_proba), low, high
        return np.asarray(va_proba)

    def get_p_value(self, proba, classes=None):
        """return probability for the positive class when binary classification and for the most 
        probable class otherwise
        """
        if classes is None:
            return proba, np.argmax(proba, axis=1)
        proba_2 = np.zeros((proba.shape[0], 2))
        proba_2[:,1] = proba[:,classes]
        proba_2[:,0] = 1 - proba[:,classes]
        return proba_2, classes

    def is_multiclass(self) -> bool:
        """returns true if more than two classes

        Returns:
            bool: true if more than two classes
        """
        return len(self.cprobs[0,:]) > 2
