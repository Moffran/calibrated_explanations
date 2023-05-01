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
        cprobs = self.cprobs[:,1]
        tprobs = self.model.predict_proba(testX)[:,1]
        low,high = VennABERS_by_def(list(zip(cprobs,self.ctargets)),tprobs)
        tmp = high / (1-low + high)
        return np.asarray(np.round(tmp))

    def predict_proba(self, testX, output_interval=False):        
        cprobs = self.cprobs[:,1]
        va_proba = self.model.predict_proba(testX)
        tprobs = va_proba[:,1]
        low,high = VennABERS_by_def(list(zip(cprobs,self.ctargets)),tprobs)
        tmp = high / (1-low + high)
        va_proba[:,0] = 1-tmp
        va_proba[:,1] = tmp
        if output_interval:
            return np.asarray(va_proba), low, high
        return np.asarray(va_proba)
