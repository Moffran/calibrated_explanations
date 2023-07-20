import numpy as np
from sklearn.isotonic import IsotonicRegression

# Function copied from https://github.com/ptocca/VennABERS/blob/master/test/VennABERS_test.ipynb
def VennABERS_by_def(ds,test):
    p0,p1 = [],[]
    for x in test:
        ds0 = ds+[(x,0)]
        iso0 = IsotonicRegression().fit(*zip(*ds0))
        p0.append(iso0.predict([x]))
        
        ds1 = ds+[(x,1)]
        iso1 = IsotonicRegression().fit(*zip(*ds1))
        p1.append(iso1.predict([x]))
    return np.array(p0).flatten(),np.array(p1).flatten()

class VennAbers:
    iso = IsotonicRegression(out_of_bounds="clip")

    def __init__(self, calX, calY, model):
        self.cprobs = model.predict_proba(calX)
        self.ctargets = calY
        self.model = model

    def predict(self, testX):
        cprobs, predict = self.get_p_value(self.cprobs)
        targets = np.multiply(predict == self.ctargets, 1) if self.is_multiclass() else self.ctargets
        tprobs, classes = self.get_p_value(self.model.predict_proba(testX))
        low,high = VennABERS_by_def(list(zip(cprobs,targets)),tprobs)
        tmp = high / (1-low + high)
        return np.asarray(np.round(tmp))

    def predict_proba(self, testX, output_interval=False, classes=None):
        va_proba = self.model.predict_proba(testX)        
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
        # return probability for the positive class when binary classification and for the most 
        # probable class otherwise
        if classes is None:
            return np.max(proba, axis=1) if self.is_multiclass() else proba[:,1], np.argmax(proba, axis=1)
        return proba[:,classes], classes
    
    def is_multiclass(self):
        return len(self.cprobs[0,:]) > 2