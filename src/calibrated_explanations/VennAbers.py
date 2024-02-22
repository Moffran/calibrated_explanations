"""contains the VennAbers class which is used to calibrate the predictions of a model

"""
# pylint: disable=invalid-name, line-too-long
# flake8: noqa: E501
import numpy as np
import venn_abers as va

class VennAbers:
    """a class to calibrate the predictions of a model using the VennABERS method
    """
    def __init__(self, cal_probs, cal_y, model, bins=None):
        self.cprobs = cal_probs
        self.ctargets = cal_y
        self.model = model
        self.bins = bins
        cprobs, predict = self.get_p_value(self.cprobs)
        if self.is_mondrian():
            self.va = []
            for b in np.unique(self.bins):
                va_bin = va.VennAbers()
                va_bin.fit(cprobs[self.bins == b,:], np.multiply(predict[self.bins == b] == self.ctargets[self.bins == b], 1) if self.is_multiclass() else self.ctargets[self.bins == b], precision=4)
                self.va.append((va_bin, b))
        else:
            self.va = va.VennAbers()
            self.va.fit(cprobs, np.multiply(predict == self.ctargets, 1) if self.is_multiclass() else self.ctargets, precision=4)

    def predict(self, test_X, bins=None):
        """a function to predict the class of the test samples

        Args:
            test_X (n_test_samples, n_features): test samples
            bins (array-like of shape (n_samples,), optional): Mondrian categories

        Returns:
            predicted classes (n_test_samples,): predicted classes based on the regularized VennABERS probabilities. If multiclass, the predicted class is 1 if the prediction from the underlying model is the same after calibration and 0 otherwise.
        """
        # tprobs, _ = self.get_p_value(self.model.predict_proba(test_X))
        # if self.is_mondrian():
        #     p0p1 = np.zeros((tprobs.shape[0],2))
        #     for va_bin, b in self.va:
        #         p0p1[bins == b,:] = va_bin.predict_proba(tprobs[bins == b,:])[1]
        # else:
        #     _, p0p1 = self.va.predict_proba(tprobs)
        # low, high = p0p1[:,0], p0p1[:,1]
        # tmp = high / (1-low + high)
        if self.is_multiclass():
            tmp, _ = self.predict_proba(test_X, bins=bins)
            return np.asarray(np.round(tmp[:,1]))
        tmp = self.predict_proba(test_X, bins=bins)[:,1]
        return np.asarray(np.round(tmp))

    def predict_proba(self, test_X, output_interval=False, classes=None, bins=None):
        """a function to predict the probabilities of the test samples, optionally outputting the VennABERS interval

        Args:
            testX (n_test_samples, n_features): test samples
            output_interval (bool, optional): if true, the VennAbers intervals are outputted. Defaults to False.
            classes ((n_test_samples,), optional): a list of predicted classes. Defaults to None.
            bins (array-like of shape (n_samples,), optional): Mondrian categories

        Returns:
            proba (n_test_samples,2): regularized VennABERS probabilities for the test samples. 
            if output_interval is true, the VennABERS intervals are also returned:
                low (n_test_samples,): lower bounds of the VennABERS interval for each test sample
                high (n_test_samples,): upper bounds of the VennABERS interval for each test sample
        """
        if 'bins' in self.model.predict_proba.__code__.co_varnames:
            va_proba = self.model.predict_proba(test_X, bins=bins)
        else:
            va_proba = self.model.predict_proba(test_X)
        tprobs, classes = self.get_p_value(va_proba, classes)
        if self.is_mondrian():
            assert bins is not None, "bins must be provided if Mondrian"
            p0p1 = np.zeros((tprobs.shape[0],2))
            for va_bin, b in self.va:
                p0p1[bins == b,:] = va_bin.predict_proba(tprobs[bins == b,:])[1]
        else:
            _, p0p1 = self.va.predict_proba(tprobs)
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

    def is_mondrian(self) -> bool:
        """returns true if Mondrian

        Returns:
            bool: true if Mondrian
        """
        return self.bins is not None
