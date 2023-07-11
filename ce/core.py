'''Calibrated Explanations for Black-Box Predictions (ce)
This file contains the code for the calibrated explanations.

The calibrated explanations are based on the paper 
"Calibrated Explanations for Black-Box Predictions" by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

Calibrated explanations are a way to explain the predictions of a black-box model 
using Venn Abers predictors (classification) or conformal predictive systems (regression) and perturbations.
'''

import inspect
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple, Dict, Callable, Any
import sklearn.neighbors as nn
import copy
from tqdm import tqdm
import random
import crepes

from lime.lime_tabular import LimeTabularExplainer
from shap import Explainer

from ._explanations import CalibratedExplanation
from ._discretizers import BinaryDiscretizer, BinaryEntropyDiscretizer, DecileDiscretizer, QuartileDiscretizer, EntropyDiscretizer 
from .VennAbers import VennAbers

__version__ = '0.0.1'

class CalibratedExplainer:
    """
    Calibrated Explainer for black-box models.

    """
    def __init__(self, 
                 model, 
                 calX, 
                 calY, 
                 mode: str = 'classification',
                 feature_names: Optional[List[str]] = None, 
                 discretizer: Optional[str] = None, 
                 categorical_features: Optional[List[int]] = None, 
                 categorical_labels: Optional[Dict[int, Dict[int, str]]] = None,
                 class_labels: Optional[Dict[int, str]] = None,
                 difficultyEstimator: Optional[Callable] = None,
                 sample_percentiles: List[int] = [25, 50, 75], 
                 num_neighbors: Union[float, int] = 1.0, 
                 random_state: int = 42, 
                 preload_LIME: bool=False, 
                 preload_SHAP: bool=False,
                 verbose: bool = False,
                 ) -> None:        
        """
        CalibratedExplainer is a class that can be used to explain the predictions of a black-box 
        model.

        Args:
            model (a predictive model): 
                A predictive model that can be used to predict the target variable.
            calX (same format as training and test input): 
                The calibration input data for the model.
            calY (same format as training and test target): 
                The calibration target data for the model.
            mode (str, optional): 
                Possible modes are 'classificaiton' or 'regression'. Defaults to 'classification'.
            feature_names (Optional[List[str]], optional): 
                A list of feature names. Must be None or include one str per feature. 
                Defaults to None.
            discretizer (Optional[str], optional): 
                The strategy used for numerical features. Possible discretizers include:
                'binary': split using the median value, 
                'binaryEntropy': use a one-layered decision tree to find the best split,
                'quartile': split using the quartiles, 
                'decile': split using the deciles, 
                'entropy': use a three-layered decision tree to find the best splits. 
                Defaults to 'binary' for regression and 'binaryEntropy' for classification.
            categorical_features (Optional[List[int]], optional): 
                A list of indeces for categorical features. Defaults to None.
            categorical_labels (Optional[Dict[int, Dict[int, str]]], optional): 
                A dictionary with the feature index as key and a feature dictionary mapping each feature 
                value (keys) to a feature label (values). Only applicable for classification. 
                Defaults to None.
            class_labels (Optional[Dict[int, str]], optional): 
                A dictionary mapping numerical target values to class names. 
                Defaults to None.
            sigma (Optional[Callable], optional):

            sample_percentiles (List[int], optional): 
                Percentiles used to sample values for evvaluation of numerical features. 
                Defaults to [25, 50, 75].
            num_neighbors (Union[float, int], optional): 
                Enables a local discretizer to be defined using the nearest neighbors in the calibration set. 
                Values (int) above 1 are interpreted as number of neighbors and float values in the range 
                (0,1] are interpreted as fraction of calibration instances. 
                Defaults to 1.0, meaning 100% of the calibration set is always used.
            random_state (int, optional): 
                Parameter to adjust the random state. 
                Defaults to 42.
            preload_LIME (bool, optional): 
                If the LIME wrapper is known to be used, it can be preloaded at initialization. 
                Defaults to False.
            preload_SHAP (bool, optional): 
                If the SHAP wrapper is known to be used, it can be preloaded at initialization. 
                Defaults to False.
            verbose (bool, optional): 
                Enable additional printouts during operation. 
                Defaults to False.
        """
        self.__initialized = False
        if isinstance(calX, pd.DataFrame):
            calX = calX.values
        else: 
            self.calX = calX
        if isinstance(calY, pd.DataFrame):
            calY = calY.values
        else:
            self.calY = calY

        self.model = model
        self.num_features = len(self.calX[0,:])  
        self.set_random_state(random_state)
        self.sample_percentiles = sample_percentiles
        self.set_num_neighbors(num_neighbors) 
        self.verbose = verbose  
        
        self.set_difficultyEstimator(difficultyEstimator, initialize=False)   
        self.set_mode(str.lower(mode), initialize=False)
        self.__initialize_interval_model()
        
        self.categorical_labels = categorical_labels
        self.class_labels = class_labels
        if categorical_features is None:
            if categorical_labels is not None:
                categorical_features = categorical_labels.keys()
            else:
                categorical_features = []
        self.categorical_features = list(categorical_features)        
        
        if feature_names is None:
            feature_names = [str(i) for i in range(self.num_features)]            
        self.feature_names = list(feature_names)

        self.set_discretizer(discretizer)  

        self.__LIME_enabled = False
        if preload_LIME:
            self.preload_LIME()
        
        self.__SHAP_enabled = False
        if preload_SHAP:
            self.preload_SHAP()
    
            
            
    def __repr__(self):
        return f"CalibratedExplainer:\n\t\
                mode={self.mode}\n\t\
                discretizer={self.discretizer.__class__}\n\t\
                model={self.model}"
    
                # feature_names={self.feature_names}\n\t\
                # categorical_features={self.categorical_features}\n\t\
                # categorical_labels={self.categorical_labels}\n\t\
                # class_labels={self.class_labels}\n\t\
                # sample_percentiles={self.sample_percentiles}\n\t\
                # num_neighbors={self.num_neighbors}\n\t\
                # random_state={self.random_state}\n\t\
                # verbose={self.verbose}\n\t\



    def predict(self, 
                testX: Any,
                y: Optional[List or float] = None, # The same meaning as y has for cps in crepes.
                low_high_percentiles: Tuple[float,float] = (5, 95),
                classes: Optional[List[int]] = None,
                ):
        """Predicts the target variable for the test data.

        Args:
            testX: A set of test objects to predict
            y (Optional[List or float], optional): 
                The y parameter has the same meaning as y has for ConformalPredictiveSystem in crepes. 
                Defaults to None.
            low_high_percentiles (Tuple[float,float], optional): 
                The low and high percentile used to calculate the interval. 
                Defaults to (5, 95).
            classes (Optional[List[int]], optional):
                The classes predicted for the original instance. None if not multiclass.

        Raises:
            ValueError: The length of the y parameter must be either a constant or the same as the number of 
                instances in testX.

        Returns:
            predict: The prediction for the test data. For classification, this is the regularized probability 
                of the positive class, derived using the intervals from VennAbers. For regression, this is the 
                median prediction from the ConformalPredictiveSystem.
            low: The lower bound of the prediction interval. For classification, this is derived using 
                VennAbers. For regression, this is the lower percentile given as parameter, derived from the 
                ConformalPredictiveSystem.
            high: The upper bound of the prediction interval. For classification, this is derived using 
                VennAbers. For regression, this is the upper percentile given as parameter, derived from the 
                ConformalPredictiveSystem.
        """
        assert self.__initialized, "The model must be initialized before calling predict."
        if self.mode == 'classification':
            if self.is_multiclass():
                # raise ValueError("Calibrated explanations does not currently support multiclass classification.")
                predict, low, high, new_classes = self.interval_model.predict_proba(testX, output_interval=True, classes=classes)
                if classes is None:
                    return predict[:,1], low, high, new_classes
                return predict[:,1], low, high, None                
            else:
                predict, low, high = self.interval_model.predict_proba(testX, output_interval=True)
                return predict[:,1], low, high, None
        elif 'regression' in self.mode:
            predict = self.model.predict(testX)
            assert low_high_percentiles[0] <= low_high_percentiles[1], "The low percentile must be smaller than (or equal to) the high percentile."
            assert ((low_high_percentiles[0] > 0 and low_high_percentiles[0] <= 50) and \
                    (low_high_percentiles[1] >= 50 and low_high_percentiles[1] < 100)) or \
                    low_high_percentiles[0] == -np.inf or low_high_percentiles[1] == np.inf and \
                    not (low_high_percentiles[0] == -np.inf and low_high_percentiles[1] == np.inf), \
                        "The percentiles must be between 0 and 100 (exclusive). The lower percentile can be -np.inf and the higher percentile can be np.inf (but not at the same time) to allow one-sided intervals."
            low = [low_high_percentiles[0], 50] if low_high_percentiles[0] != -np.inf else [50, 50]
            high = [low_high_percentiles[1], 50] if low_high_percentiles[1] != np.inf else [50, 50]
                        
            sigma_test = self.__get_sigma_test(X=testX)
            if y is None:     
                interval = self.interval_model.predict(y_hat=predict, sigmas=sigma_test,
                            lower_percentiles=low,
                            higher_percentiles=high)
                predict = (interval[:,1] + interval[:,3]) / 2 # The median
                return predict, \
                    interval[:,0] if low_high_percentiles[0] != -np.inf else np.array([min(self.calY)]), \
                    interval[:,2] if low_high_percentiles[1] != np.inf else np.array([max(self.calY)]), \
                    None
            else:
                if not np.isscalar(y) and len(y) != len(testX):
                    raise ValueError("The length of the y parameter must be either a scalar or the same as the number of instances in testX.") 
                y_prob = self.interval_model.predict(y_hat=predict, sigmas=sigma_test, y = float(y) if np.isscalar(y) else y,)
                # Use the width of the interval from prediction to determine which interval values to use as low and high thresholds for the interval.
                interval_ = self.interval_model.predict(y_hat=predict, sigmas=sigma_test,
                            lower_percentiles=low,
                            higher_percentiles=high)
                median = (interval_[:,1] + interval_[:,3]) / 2 # The median
                interval = np.array([np.array([0.0,0.0]) for i in range(testX.shape[0])])
                for i,x in enumerate(testX):                    
                    interval[i,0] = self.interval_model.predict(y_hat=[predict[i]], sigmas=sigma_test, y=float(interval_[i,0] - median[i] + y))
                    interval[i,1] = self.interval_model.predict(y_hat=[predict[i]], sigmas=sigma_test, y=float(interval_[i,2] - median[i] + y))
                predict = y_prob
                # Changed to 1-p so that high probability means high prediction and vice versa
                return [1-predict[0]], 1-interval[:,1], 1-interval[:,0], None



    def __call__(self, 
                 testX: Any,
                 y: Optional[List or float] = None, # The same meaning as y has for cps in crepes.
                 low_high_percentiles: Tuple[float,float] = (5, 95),
                 ) -> CalibratedExplanation:
        """Creates a CalibratedExplanation object for the test data.

        Args:
            testX: A set of test objects to predict
            y (Optional[List or float], optional): 
                The y parameter has the same meaning as y has for ConformalPredictiveSystems in crepes. 
                Defaults to None.
            low_high_percentiles (Tuple[float,float], optional): 
                The low and high percentile used to calculate the interval. 
                Defaults to (5, 95).

        Raises:
            ValueError: The number of features in the test data must be the same as in the calibration data.
            Warning: The y parameter is only supported for mode='regression'.
            ValueError: The length of the y parameter must be either a constant or the same as the number of 
                instances in testX.

        Returns:
            CalibratedExplanations: A CalibratedExplanations object containing the predictions and the 
                intervals. 
        """
        if len(testX.shape) == 1:
            testX = testX.reshape(1, -1)        
        if testX.shape[1] != self.calX.shape[1]:
            raise ValueError("The number of features in the test data must be the same as in the calibration data.")
        explanation = CalibratedExplanation(self, testX)
        discretizer = self.__get_discretizer()
        
        if y is not None:
            if not ('regression' in self.mode):
                raise Warning("The y parameter is only supported for mode='regression'.")
            if not np.isscalar(y) and len(y) != len(testX):
                raise ValueError("The length of the y parameter must be either a constant or the same as the number of instances in testX.")
            explanation.y = y
            explanation.low_high_percentiles = low_high_percentiles
        elif 'regression' in self.mode:
            explanation.low_high_percentiles = low_high_percentiles

        calX = self.calX
        calY = self.calY
        
        feature_weights =  {'predict':[],'low':[],'high':[],}
        feature_predict =  {'predict':[],'low':[],'high':[],}
        prediction =  {'predict':[],'low':[],'high':[], 'classes':[]}
        binned_predict =  {'predict':[],'low':[],'high':[],'current_bin':[],'rule_values':[]}
        
        for i,x in tqdm(enumerate(testX)) if self.verbose else enumerate(testX):#
            if y is not None and not np.isscalar(explanation.y):
                y = float(explanation.y[i])
            predict, low, high, predicted_class = self.predict(x.reshape(1,-1), y=y, low_high_percentiles=low_high_percentiles)
            prediction['predict'].append(predict[0])
            prediction['low'].append(low[0])
            prediction['high'].append(high[0])
            if self.is_multiclass():
                prediction['classes'].append(predicted_class[0])
            else:
                prediction['classes'].append(1)
                
            if not self.num_neighbors == len(self.calY):                
                calX, calY = self.find_local_calibration_data(x)
                self.set_discretizer(discretizer, calX, calY)

            rule_values = {}
            instance_weights = {'predict':np.zeros(x.shape[0]),'low':np.zeros(x.shape[0]),'high':np.zeros(x.shape[0])}
            instance_predict = {'predict':np.zeros(x.shape[0]),'low':np.zeros(x.shape[0]),'high':np.zeros(x.shape[0])}
            instance_binned = {'predict':[],'low':[],'high':[],'current_bin':[],'rule_values':[]} 
            # Get the perturbations
            x_original = copy.deepcopy(x)
            perturbed_original = self.discretize(copy.deepcopy(x).reshape(1,-1))
            rule_boundaries = self.rule_boundaries(x_original, perturbed_original)
            for f in range(x.shape[0]): # For each feature
                perturbed = copy.deepcopy(x)

                current_bin = -1
                if f in self.categorical_features:
                    values = self.feature_values[f]
                    rule_value = values
                    average_predict, low_predict, high_predict = np.zeros(len(values)),np.zeros(len(values)),np.zeros(len(values))
                    for bin, value in enumerate(values):  # For each bin (i.e. discretized value) in the values array...
                        perturbed[f] = perturbed_original[0,f] # Assign the original discretized value to ensure similarity to value
                        if perturbed[f] == value:
                            current_bin = bin  # If the discretized value is the same as the original, skip it                            

                        perturbed[f] = value             
                        predict, low, high, _ = self.predict(perturbed.reshape(1,-1), y=y, low_high_percentiles=low_high_percentiles, classes=predicted_class)
                        average_predict[bin] = predict[0] 
                        low_predict[bin] = low[0]  
                        high_predict[bin] = high[0]
                else:
                    rule_value = []
                    values = np.array(calX[:,f])
                    lesser = rule_boundaries[f][0]
                    greater = rule_boundaries[f][1]
                    num_bins = 1
                    num_bins += int(np.any(values > greater))
                    num_bins += int(np.any(values < lesser))
                    average_predict, low_predict, high_predict = np.zeros(num_bins),np.zeros(num_bins),np.zeros(num_bins)
                    
                    bin = 0
                    if np.any(values < lesser): 
                        lesser_values = np.unique(self.__get_lesser_values(f, lesser))
                        rule_value.append(lesser_values)
                        for value in lesser_values:
                            perturbed[f] = value                  
                            predict, low, high, _ = self.predict(perturbed.reshape(1,-1), y=y, low_high_percentiles=low_high_percentiles, classes=predicted_class)
                            average_predict[bin] += predict[0] 
                            low_predict[bin] += low[0]  
                            high_predict[bin] += high[0]
                        average_predict[bin] = average_predict[bin]/len(lesser_values)
                        low_predict[bin] = low_predict[bin]/len(lesser_values)
                        high_predict[bin] = high_predict[bin]/len(lesser_values)                 
                        bin += 1   
                    if np.any(values > greater):
                        greater_values = self.__get_greater_values(f, greater)
                        rule_value.append(greater_values)
                        for value in greater_values:
                            perturbed[f] = value             
                            predict, low, high, _ = self.predict(perturbed.reshape(1,-1), y=y, low_high_percentiles=low_high_percentiles, classes=predicted_class)
                            average_predict[bin] += predict[0] 
                            low_predict[bin] += low[0]  
                            high_predict[bin] += high[0]
                        average_predict[bin] = average_predict[bin]/len(greater_values)
                        low_predict[bin] = low_predict[bin]/len(greater_values)
                        high_predict[bin] = high_predict[bin]/len(greater_values)                 
                        bin += 1                   

                    covered_values = self.__get_covered_values(f, lesser, greater)
                    rule_value.append(covered_values)
                    for value in covered_values:
                        perturbed[f] = value              
                        predict, low, high, _ = self.predict(perturbed.reshape(1,-1), y=y, low_high_percentiles=low_high_percentiles, classes=predicted_class)
                        average_predict[bin] += predict[0] 
                        low_predict[bin] += low[0]  
                        high_predict[bin] += high[0]
                    average_predict[bin] = average_predict[bin]/len(covered_values)
                    low_predict[bin] = low_predict[bin]/len(covered_values)
                    high_predict[bin] = high_predict[bin]/len(covered_values)                 
                    current_bin = bin

                rule_values[f] = (rule_value, x_original[f], perturbed_original[0,f])
                uncovered = np.setdiff1d(np.arange(len(average_predict)), current_bin)

                instance_binned['predict'].append(average_predict)
                instance_binned['low'].append(low_predict)
                instance_binned['high'].append(high_predict)
                instance_binned['current_bin'].append(current_bin)

                # Handle the situation where the current bin is the only bin
                if len(uncovered) == 0:
                    instance_weights['predict'][f] = 0
                    instance_weights['low'][f] = 0
                    instance_weights['high'][f] = 0

                    instance_predict['predict'][f] = 0
                    instance_predict['low'][f] = 0
                    instance_predict['high'][f] = 0
                else:
                    instance_weights['predict'][f] = np.mean(average_predict[uncovered]) - prediction['predict'][-1]
                    instance_weights['low'][f] = np.mean(low_predict[uncovered]) - prediction['predict'][-1]
                    instance_weights['high'][f] = np.mean(high_predict[uncovered]) - prediction['predict'][-1]
                    
                    instance_predict['predict'][f] = np.mean(average_predict[uncovered])
                    instance_predict['low'][f] = np.mean(low_predict[uncovered])
                    instance_predict['high'][f] = np.mean(high_predict[uncovered])

            binned_predict['predict'].append(instance_binned['predict'])
            binned_predict['low'].append(instance_binned['low'])
            binned_predict['high'].append(instance_binned['high'])
            binned_predict['current_bin'].append(instance_binned['current_bin'])
            binned_predict['rule_values'].append(rule_values)

            feature_weights['predict'].append(instance_weights['predict'])
            feature_weights['low'].append(instance_weights['low'])
            feature_weights['high'].append(instance_weights['high'])

            feature_predict['predict'].append(instance_predict['predict'])
            feature_predict['low'].append(instance_predict['low'])
            feature_predict['high'].append(instance_predict['high'])

        explanation.finalize(binned_predict, feature_weights, feature_predict, prediction)
        return explanation
    
    
    
    def is_multiclass(self):
        return True if self.num_classes > 2 else False
    
    
    
    def rule_boundaries(self, instance, perturbed_instance=None):
        min_max = []
        if perturbed_instance is None:
            perturbed_instance = self.discretize(instance.reshape(1,-1))
        for f in range(self.num_features):
            if f in self.categorical_features:
                min_max.append([instance[f], instance[f]])
            else:
                values = np.array(self.discretizer.means[f])
                min_max.append([self.discretizer.mins[f][np.where(perturbed_instance[0,f] == values)[0][0]], \
                            self.discretizer.maxs[f][np.where(perturbed_instance[0,f] == values)[0][0]]])
        return min_max
    
    
    
    def __get_greater_values(self, f: int, greater: float):
        greater_values = np.percentile(self.calX[self.calX[:,f] > greater,f], self.sample_percentiles)
        return greater_values
    
    
    
    def __get_lesser_values(self, f: int, lesser: float):
        lesser_values = np.percentile(self.calX[self.calX[:,f] < lesser,f], self.sample_percentiles)
        return lesser_values 
    
    
    
    def __get_covered_values(self, f: int, lesser: float, greater: float):
        covered = np.where((self.calX[:,f] >= lesser) & (self.calX[:,f] <= greater))[0]
        covered_values = np.percentile(self.calX[covered,f], self.sample_percentiles)
        return covered_values 
    
    
    
    def set_random_state(self, random_state: int) -> None:
        self.random_state = random_state        
        random.seed(self.random_state)
    
    
    
    def set_difficultyEstimator(self, difficultyEstimator, initialize=True) -> None:
        self.__initialized = False
        self.difficultyEstimator = difficultyEstimator   
        # initialize the model with the new sigma
        if initialize:
            self.__initialize_interval_model()
            
            
        
    def __constant_sigma(self, X: np.ndarray, learner=None, beta=None) -> np.ndarray:    
        return np.ones(X.shape[0])
    
    
    def __get_sigma_test(self, X: np.ndarray) -> np.ndarray:
        if self.difficultyEstimator is None:
            return self.__constant_sigma(X)
        else:
            return self.difficultyEstimator.apply(X)
    
    
        
    def set_mode(self, mode: str, initialize=True) -> None:
        self.__initialized = False
        if mode == 'classification':
            assert 'predict_proba' in dir(self.model), "The model must have a predict_proba method."
            self.num_classes = len(np.unique(self.calY))
        elif 'regression' in mode:
            assert 'predict' in dir(self.model), "The model must have a predict method."
            self.num_classes = 0
        else:
            raise ValueError("The mode must be either 'classification' or 'regression'.")
        self.mode = mode
        if initialize:
            self.__initialize_interval_model()
        



    def __initialize_interval_model(self) -> None:
        if self.mode == 'classification':
            va = VennAbers(self.calX, self.calY, self.model)
            self.interval_model = va
        elif 'regression' in self.mode:
            calY_hat = self.model.predict(self.calX)
            self.residual_cal = self.calY - calY_hat
            cps = crepes.ConformalPredictiveSystem()
            if self.difficultyEstimator is not None:
                sigma_cal = self.difficultyEstimator.apply(X=self.calX)
                cps.fit(residuals=self.residual_cal, sigmas=sigma_cal)
            else:
                cps.fit(residuals=self.residual_cal)
            self.interval_model = cps
        self.__initialized = True

    def set_num_neighbors(self, num_neighbors: Union[float, int]) -> None:        
        if num_neighbors < 0:
            raise ValueError("num_neighbors must be positive")
        if num_neighbors <= 1.0:
            num_neighbors = int(len(self.calX) * num_neighbors)
        self.num_neighbors = num_neighbors
    
    
    
    def find_local_calibration_data(self, x) -> Tuple[Any, Any]:
        nn_model = nn.NearestNeighbors(n_neighbors=self.num_neighbors, algorithm='ball_tree').fit(self.calX)
        _, indices = nn_model.kneighbors(x.reshape(1,-1))
        return self.calX[indices[0]], self.calY[indices[0]]
    
    
    
    def discretize(self, x):
        if len(np.shape(x)) == 1:
            x = np.array(x)
        x.dtype = float
        tmp = self.discretizer.discretize(x)
        for f in self.discretizer.to_discretize:
            x[:,f] = [self.discretizer.means[f][int(tmp[i,f])] for i in range(len(x[:,0]))]
        return x
    
    
    
    def __get_discretizer(self) -> str:
        if isinstance(self.discretizer, QuartileDiscretizer):
            return 'quartile'
        elif isinstance(self.discretizer, DecileDiscretizer):
            return 'decile'
        elif isinstance(self.discretizer, EntropyDiscretizer):
            return 'entropy'
        elif isinstance(self.discretizer, BinaryEntropyDiscretizer):
            return 'binaryEntropy'
        elif isinstance(self.discretizer, BinaryDiscretizer):
            return 'binary'



    def set_discretizer(self, discretizer: str, calX=None, calY=None) -> None:
        if calX is None:
            calX = self.calX
        if calY is None:
            calY = self.calY
            
        if discretizer is None:
            if 'regression' in self.mode:
                discretizer = 'binary'
            else:
                discretizer = 'binaryEntropy'
        else:
            if 'regression'in self.mode:
                assert discretizer is None or discretizer in ['binary', 'quartile', 'decile'], "The discretizer must be 'binary' (default), 'quartile', or 'decile' for regression."
        
        if discretizer == 'quartile':
            self.discretizer = QuartileDiscretizer(
                    calX, self.categorical_features,
                    self.feature_names, labels=calY,
                    random_state=self.random_state)
        elif discretizer == 'decile':
            self.discretizer = DecileDiscretizer(
                    calX, self.categorical_features,
                    self.feature_names, labels=calY,
                    random_state=self.random_state)
        elif discretizer == 'entropy':
            self.discretizer = EntropyDiscretizer(
                    calX, self.categorical_features,
                    self.feature_names, labels=calY,
                    random_state=self.random_state)
        elif discretizer == 'binary':
            self.discretizer = BinaryDiscretizer(
                    calX, self.categorical_features,
                    self.feature_names, labels=calY,
                    random_state=self.random_state)
        elif discretizer == 'binaryEntropy':
            self.discretizer = BinaryEntropyDiscretizer(
                    calX, self.categorical_features,
                    self.feature_names, labels=calY,
                    random_state=self.random_state)
            
        self.discretized_calX = self.discretize(copy.deepcopy(self.calX))

        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in range(self.calX.shape[1]):
            column = self.discretized_calX[:, feature]
            feature_count = {}
            for item in column:
                feature_count[item] = feature_count.get(item, 0) + 1
            values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            
            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
    
    
    
    def set_latest_explanation(self, explanation) -> None:
        self.latest_explanation = explanation
    
    
    
    def is_LIME_enabled(self, is_enabled=None) -> bool:
        if is_enabled is not None:
            self.__LIME_enabled = is_enabled
        return self.__LIME_enabled
    
    
    
    def is_SHAP_enabled(self, is_enabled=None) -> bool:
        if is_enabled is not None:
            self.__SHAP_enabled = is_enabled
        return self.__SHAP_enabled



    def preload_LIME(self) -> None:
        if not self.is_LIME_enabled():
            if self.mode == 'classification':
                self.lime = LimeTabularExplainer(self.calX[:1,:], feature_names=self.feature_names, class_names=['0','1'], mode=self.mode)
                self.lime_exp = self.lime.explain_instance(self.calX[0,:], self.model.predict_proba, num_features=self.num_features)
            elif 'regression' in self.mode:
                self.lime = LimeTabularExplainer(self.calX[:1,:], feature_names=self.feature_names, mode='regression')
                self.lime_exp = self.lime.explain_instance(self.calX[0,:], self.model.predict, num_features=self.num_features)                
            self.is_LIME_enabled(True)
        return self.lime, self.lime_exp
        
        
        
    def preload_SHAP(self, num_test=None) -> None:
        if not self.is_SHAP_enabled() or num_test is not None and self.shap_exp.shape[0] != num_test:
            f = lambda x: self.predict(x)[0]
            self.shap = Explainer(f, self.calX[:1,:], feature_names=self.feature_names)
            self.shap_exp = self.shap(self.calX[0,:].reshape(1,-1)) if num_test is None else self.shap(self.calX[:num_test,:]) 
            self.is_SHAP_enabled(True)
        return self.shap, self.shap_exp


# class CalibratedAsLimeTabularExplainer(LimeTabularExplainer):
#     '''
#     Wrapper for LimeTabularExplainer to use calibrated explainer
#     '''
#     def __init__(self,
#                  training_data,
#                  mode="classification",
#                  training_labels=None,
#                  feature_names=None,
#                  categorical_features=None,
#                  discretizer='binaryEntropy',
#                  **kwargs):
#         self.calibrated_explainer = None
#         self.training_data = training_data
#         self.training_labels = training_labels
#         self.discretizer = discretizer
#         self.categorical_features = categorical_features
#         self.feature_names = feature_names
#         assert training_labels is not None, "Calibrated Explanations requires training labels"



#     def explain_instance(self, data_row, classifier, **kwargs):
#         if self.calibrated_explainer is None:
#             assert 'predict_proba' in dir(classifier), "The classifier must have a predict_proba method."
#             self.calibrated_explainer = CalibratedExplainer(classifier, self.training_data, self.training_labels, self.feature_names, self.discretizer, self.categorical_features,)
#             self.discretizer = self.calibrated_explainer.discretizer
#         return self.calibrated_explainer(data_row).as_lime()[0]
    
    
    
# class CalibratedAsShapExplainer(Explainer):
#     '''
#     Wrapper for the CalibratedExplainer to be used as a shap explainer.
#     The masker must contain a data and a target field for the calibration set.
#     The model must have a predict_proba method.
#     '''
#     def __init__(self, model, masker, feature_names=None, **kwargs):
#         assert ['data','target'] in dir(masker), "The masker must contain a data and a target field for the calibration set."
#         assert 'predict_proba' in dir(model), "The model must have a predict_proba method."
#         self.calibrated_explainer = CalibratedExplainer(model, masker.data, masker.labels, feature_names)



#     def __call__(self, *args, **kwargs):
#         return self.calibrated_explainer(*args, **kwargs).as_shap()
