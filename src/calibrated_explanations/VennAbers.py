"""contains the VennAbers class which is used to calibrate the predictions of a model

"""
# pylint: disable=invalid-name, line-too-long
# flake8: noqa: E501
import warnings
import numpy as np
import venn_abers as va

class VennAbers:
    """a class to calibrate the predictions of a model using the VennABERS method
    """
    def __init__(self, cprobs, cal_y, model, bins=None):
        self.cprobs = cprobs
        self.ctargets = cal_y
        self.model = model
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
        else:
            if self.is_multiclass():
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

    # pylint: disable=too-many-locals, too-many-branches
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
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if 'bins' in self.model.predict_proba.__code__.co_varnames:
            tprobs = self.model.predict_proba(test_X, bins=bins)
        else:
            tprobs = self.model.predict_proba(test_X)
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
        return len(self.cprobs[0,:]) > 2

    def is_mondrian(self) -> bool:
        """returns true if Mondrian

        Returns:
            bool: true if Mondrian
        """
        return self.bins is not None
