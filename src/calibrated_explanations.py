'''
This file contains the code for the calibrated explanations.

The calibrated explanations are based on the paper 
"Calibrated Explanations for Black-Box Predictions" by Helena Löfström, Tuwe Löfström, and Ulf Johansson.

Calibrated explanations are a way to explain the predictions of a black-box model 
using Venn Abers predictors and perturbations.
'''

import collections
from typing import List, Optional, Union, Tuple, Dict, Callable, Any
from numpy.typing import ArrayLike
from lime.lime_tabular import LimeTabularExplainer
from shap import Explainer, links
from lime.discretize import BaseDiscretizer, DecileDiscretizer, QuartileDiscretizer, \
    EntropyDiscretizer#, BinaryEntropyDiscretizer, BinaryDiscretizer
import numpy as np
from VennAbers import VennAbers
import sklearn.neighbors as nn
import copy
import matplotlib.pyplot as plt
import sklearn
from tqdm import tqdm
import random

class BinaryDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):

        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature], [50]))
            bins.append(qts)
        return bins

class BinaryEntropyDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        if(labels is None):
            raise ValueError('Labels must be not None when using \
                             EntropyDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            # Entropy splitting / at most 2 bins so max_depth=1
            dt = sklearn.tree.DecisionTreeClassifier(criterion='entropy',
                                                     max_depth=1,
                                                     random_state=self.random_state)
            x = np.reshape(data[:, feature], (-1, 1))
            dt.fit(x, labels)
            qts = dt.tree_.threshold[np.where(dt.tree_.children_left > -1)]

            if qts.shape[0] == 0:
                qts = np.array([np.median(data[:, feature])])
            else:
                qts = np.sort(qts)

            bins.append(qts)

        return bins

class CalibratedExplainer:
    def __init__(self, 
                 model, 
                 calX: ArrayLike, 
                 calY: ArrayLike, 
                 feature_names: Optional[List[str]] = None, 
                 discretizer: Optional[str] = None, 
                 categorical_features: Optional[List[int]] = None, 
                 num_neighbors: Union[float, int] = 1.0, 
                 random_state: int = 42, 
                 sample_percentiles: List[int] = [25, 50, 75], 
                 preload_LIME: bool=False, 
                 preload_SHAP: bool=False,
                 verbose: bool = False
                 ) -> None:
        assert 'predict_proba' in dir(model), "The model must have a predict_proba method."
        self.model = model
        self.calX = calX
        self.calY = calY
        self.set_random_state(random_state)
        self.sample_percentiles = sample_percentiles
        self.set_num_neighbors(num_neighbors)
        self.num_features = len(self.calX[0,:])   
        self.verbose = verbose         

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(calX.shape[1])]
        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        va = VennAbers(self.calX, self.calY, self.model)
        self.va_model = va

        self.set_discretizer(discretizer)        
        self.mode = 'classification'

        self.__LIME_enabled = False
        if preload_LIME:
            self.preload_LIME()
        
        self.__SHAP_enabled = False
        if preload_SHAP:
            self.preload_SHAP()

    def __call__(self, testX: ArrayLike):
        if len(testX.shape) == 1:
            testX = testX.reshape(1, -1)        
        if testX.shape[1] != self.calX.shape[1]:
            raise ValueError("The number of features in the test data must be the same as in the calibration data.")

        explanation = CalibratedExplanation(self, testX)
        discretizer = self.__get_discretizer()

        va = self.va_model
        calX = self.calX
        calY = self.calY
        
        feature_weights =  {'regularized':[],'low':[],'high':[]}
        feature_proba =  {'regularized':[],'low':[],'high':[]}
        proba =  {'regularized':[],'low':[],'high':[]}
        binned_proba =  {'regularized':[],'low':[],'high':[],'current_bin':[]}

        for x in tqdm(testX) if self.verbose else testX:#
            va_proba, va_low, va_high = va.predict_proba([x], output_interval=True)
            proba['regularized'].append(va_proba[:,1])
            proba['low'].append(va_low)
            proba['high'].append(va_high)
            if self.num_neighbors == len(self.calY):
                va_local = va
            else:
                calX, calY = self.find_local_calibration_data(x)
                # va_local = VennAbers(calX, calY, self.model)
                self.set_discretizer(discretizer, calX, calY)


            instance_weights = {'regularized':np.zeros(x.shape[0]),'low':np.zeros(x.shape[0]),'high':np.zeros(x.shape[0])}
            instance_proba = {'regularized':np.zeros(x.shape[0]),'low':np.zeros(x.shape[0]),'high':np.zeros(x.shape[0])}
            instance_binned = {'regularized':[],'low':[],'high':[],'current_bin':[]} 
            # Get the perturbations
            x_original = copy.deepcopy(x)
            perturbed_original = self.discretize(copy.deepcopy(x).reshape(1,-1))
            rule_boundaries = self.rule_boundaries(x_original, perturbed_original)
            for f in range(x.shape[0]): # For each feature
                perturbed = copy.deepcopy(x)

                current_bin = -1
                if f in self.categorical_features:
                    values = self.feature_values[f]
                    average_proba, low_proba, high_proba = np.zeros(len(values)),np.zeros(len(values)),np.zeros(len(values))
                    for bin, value in enumerate(values):  # For each bin (i.e. discretized value) in the values array...
                        perturbed[f] = perturbed_original[0,f] # Assign the original discretized value to ensure similarity to value
                        if perturbed[f] == value:
                            current_bin = bin  # If the discretized value is the same as the original, skip it
                            

                        perturbed[f] = value
                        pert_proba, pert_low, pert_high = va.predict_proba([perturbed], output_interval=True) # Predict the probability of the new discretized value
                        average_proba[bin] = pert_proba[:,1] # Store the predicted probability of the new discretized value
                        low_proba[bin] = pert_low  # Store the lower bound of the predicted probability of the new discretized value
                        high_proba[bin] = pert_high # Store the higher bound of the predicted probability of the new discretized value

                else:
                    values = np.array(calX[:,f])
                    lesser = rule_boundaries[f][0]
                    greater = rule_boundaries[f][1]
                    num_bins = 1
                    num_bins += int(np.any(values > greater))
                    num_bins += int(np.any(values < lesser))
                    average_proba, low_proba, high_proba = np.zeros(num_bins),np.zeros(num_bins),np.zeros(num_bins)
                    
                    bin = 0
                    if np.any(values < lesser): 
                        lesser_values = np.unique(self.__get_lesser_values(f, lesser))
                        for value in lesser_values:
                            perturbed[f] = value
                            pert_proba, pert_low, pert_high = va.predict_proba([perturbed], output_interval=True)
                            average_proba[bin] += pert_proba[:,1]
                            low_proba[bin] += pert_low 
                            high_proba[bin] += pert_high
                        average_proba[bin] = average_proba[bin].sum()/len(lesser_values)
                        low_proba[bin] = low_proba[bin].sum()/len(lesser_values) 
                        high_proba[bin] = high_proba[bin].sum()/len(lesser_values)                    
                        bin += 1   
                    if np.any(values > greater):
                        greater_values = np.unique(self.__get_greater_values(f, greater))
                        for value in greater_values:
                            perturbed[f] = value
                            pert_proba, pert_low, pert_high = va.predict_proba([perturbed], output_interval=True)
                            average_proba[bin] += pert_proba[:,1]
                            low_proba[bin] += pert_low 
                            high_proba[bin] += pert_high
                        average_proba[bin] = average_proba[bin].sum()/len(greater_values)
                        low_proba[bin] = low_proba[bin].sum()/len(greater_values) 
                        high_proba[bin] = high_proba[bin].sum()/len(greater_values)                       
                        bin += 1                   

                    covered_values = np.unique(self.__get_covered_values(f, lesser, greater))
                    for value in covered_values:
                        perturbed[f] = value
                        pert_proba, pert_low, pert_high = va.predict_proba([perturbed], output_interval=True)
                        average_proba[bin] += pert_proba[:,1]
                        low_proba[bin] += pert_low 
                        high_proba[bin] += pert_high
                    average_proba[bin] = average_proba[bin].sum()/len(covered_values)
                    low_proba[bin] = low_proba[bin].sum()/len(covered_values) 
                    high_proba[bin] = high_proba[bin].sum()/len(covered_values)  
                    current_bin = bin

                uncovered = np.setdiff1d(np.arange(len(average_proba)), current_bin)

                instance_binned['regularized'].append(average_proba)
                instance_binned['low'].append(low_proba)
                instance_binned['high'].append(high_proba)
                instance_binned['current_bin'].append(current_bin)

                if len(uncovered) == 0:
                    instance_weights['regularized'][f] = 0
                    instance_weights['low'][f] = 0
                    instance_weights['high'][f] = 0

                    instance_proba['regularized'][f] = 0
                    instance_proba['low'][f] = 0
                    instance_proba['high'][f] = 0
                else:
                    instance_weights['regularized'][f] = average_proba[current_bin] - np.mean(average_proba[uncovered])
                    instance_weights['low'][f] = average_proba[current_bin] - np.mean(low_proba[uncovered])
                    instance_weights['high'][f] = average_proba[current_bin] - np.mean(high_proba[uncovered])

                    instance_proba['regularized'][f] = np.mean(average_proba[uncovered])
                    instance_proba['low'][f] = np.mean(low_proba[uncovered])
                    instance_proba['high'][f] = np.mean(high_proba[uncovered])

            binned_proba['regularized'].append(instance_binned['regularized'])
            binned_proba['low'].append(instance_binned['low'])
            binned_proba['high'].append(instance_binned['high'])

            feature_weights['regularized'].append(instance_weights['regularized'])
            feature_weights['low'].append(instance_weights['low'])
            feature_weights['high'].append(instance_weights['high'])

            feature_proba['regularized'].append(instance_proba['regularized'])
            feature_proba['low'].append(instance_proba['low'])
            feature_proba['high'].append(instance_proba['high'])

        explanation.finalize(binned_proba, feature_weights, feature_proba, proba)
        return explanation
    
    def rule_boundaries(self, instance, perturbed_instance=None):
        min_max = []
        if perturbed_instance is None:
            perturbed_instance = self.discretize(instance.reshape(1,-1))
        for f in range(self.num_features):
            if f in self.categorical_features:
                min_max.append((instance[f], instance[f]))
            else:
                values = np.array(self.discretizer.means[f])
                min_max.append((self.discretizer.mins[f][np.where(perturbed_instance[0,f] == values)[0][0]], \
                            self.discretizer.maxs[f][np.where(perturbed_instance[0,f] == values)[0][0]]))
        return min_max
    
    def __get_greater_values(self, f: int, greater: float) -> ArrayLike:
        greater_values = np.percentile(self.calX[self.calX[:,f] > greater,f], self.sample_percentiles)
        return greater_values
    
    def __get_lesser_values(self, f: int, lesser: float) -> ArrayLike:
        lesser_values = np.percentile(self.calX[self.calX[:,f] < lesser,f], self.sample_percentiles)
        return lesser_values 
    
    def __get_covered_values(self, f: int, lesser: float, greater: float) -> ArrayLike:
        covered = np.where((self.calX[:,f] >= lesser) & (self.calX[:,f] <= greater))[0]
        covered_values = np.percentile(self.calX[covered,f], self.sample_percentiles)
        return covered_values 
    
    def set_random_state(self, random_state: int) -> None:
        self.random_state = random_state        
        random.seed(self.random_state)

    def set_num_neighbors(self, num_neighbors: int) -> None:        
        if num_neighbors < 0:
            raise ValueError("num_neighbors must be positive")
        if num_neighbors <= 1.0:
            num_neighbors = int(len(self.calX) * num_neighbors)
        self.num_neighbors = num_neighbors
    
    def find_local_calibration_data(self, x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        nn_model = nn.NearestNeighbors(n_neighbors=self.num_neighbors, algorithm='ball_tree').fit(self.calX)
        _, indices = nn_model.kneighbors(x.reshape(1,-1))
        return self.calX[indices[0]], self.calY[indices[0]]
    
    def discretize(self, x: ArrayLike) -> ArrayLike:
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

    def set_discretizer(self, discretizer: str, calX: Optional[ArrayLike]=None, calY: Optional[ArrayLike]=None) -> None:
        if calX is None:
            calX = self.calX
        if calY is None:
            calY = self.calY
            
        if discretizer is None:
            discretizer = 'binaryEntropy'

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

            feature_count = collections.Counter(column)
            values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            
            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
    
    def set_latest_explanation(self, explanation) -> None:
        self.latest_explanation = explanation
    
    def preload_LIME(self) -> None:
        if not self.__LIME_enabled:
            self.lime = LimeTabularExplainer(self.calX[:1,:], feature_names=self.feature_names, class_names=['0','1'], mode=self.mode, categorical_features=self.categorical_features)
            self.lime.discretizer = self.discretizer
            self.lime_exp = self.lime.explain_instance(self.calX[0,:], self.va_model.predict_proba, num_features=self.num_features)
            self.__LIME_enabled = True
    
    def preload_SHAP(self) -> None:
        if not self.__SHAP_enabled:
            f = lambda x: self.va_model.predict_proba(x)[:,1]
            self.shap = Explainer(f, self.calX[:1,:], feature_names=self.feature_names)
            self.shap_exp = self.shap(self.calX[0,:].reshape(1,-1))  
            self.__SHAP_enabled = True
    
    
class CalibratedExplanation:
    def __init__(self, CE: CalibratedExplainer, x: List[float]) -> None:
        self.CE = CE
        self.x = copy.deepcopy(x)
        
    def finalize(self, binned_proba, feature_weights, feature_proba, proba) -> None:
        self.binned_proba = binned_proba
        self.feature_weights = feature_weights
        self.feature_proba = feature_proba
        self.proba = proba
        self.CE.set_latest_explanation(self)
        
    def define_rules(self, instance: ArrayLike) -> List[str]:
        self.rules = []
        x = self.CE.discretizer.discretize(instance)
        for f in range(self.CE.num_features):
            if f in self.CE.categorical_features:    
                name = x[f]
                rule = '%s = %g' % (self.CE.feature_names[f], name)
            else:
                rule = self.CE.discretizer.names[f][int(x[f])]
            self.rules.append(rule)
        return self.rules   
    
    def __color_brew(self, n):
        color_list = []

        # Initialize saturation & value; calculate chroma & value shift
        s, v = 0.75, 0.9
        c = s * v
        m = v - c

        # for h in np.arange(25, 385, 360. / n).astype(int):
        for h in np.arange(5, 385, 490. / n).astype(int):
            # Calculate some intermediate values
            h_bar = h / 60.
            x = c * (1 - abs((h_bar % 2) - 1))
            # Initialize RGB with same hue & chroma as our color
            rgb = [(c, x, 0),
                (x, c, 0),
                (0, c, x),
                (0, x, c),
                (x, 0, c),
                (c, 0, x),
                (c, x, 0)]
            r, g, b = rgb[int(h_bar)]
            # Shift the initial RGB values to match value and store
            rgb = [(int(255 * (r + m))),
                (int(255 * (g + m))),
                (int(255 * (b + m)))]
            color_list.append(rgb)
        color_list.reverse()
        return color_list

    def __get_fill_color(self, venn_abers, reduction=1):
        colors = self.__color_brew(2)
        winner_class = int(venn_abers['reg_proba']>= 0.5)
        color = colors[winner_class]
        
        alpha = venn_abers['reg_proba'] if winner_class==1 else 1-venn_abers['reg_proba']
        alpha = ((alpha - 0.5) / (1 - 0.5)) * (1-.25) + .25 # normalize values to the range [.25,1]
        if reduction != 1:
            alpha = reduction
        
        # unpack numpy scalars
        alpha = float(alpha)
        # compute the color as alpha against white
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return '#%2x%2x%2x' % tuple(color)
    
    def __rank_features(self, feature_weights: ArrayLike, num_to_show: Optional[int]=None) -> ArrayLike:
        if num_to_show is None:
            num_to_show = len(feature_weights)
        sorted_indices = np.argsort(np.abs(feature_weights))
        return sorted_indices[-num_to_show:]
    
    def __make_directory(self, path: str) -> None:
        # create directory if it does not exist
        import os
        if not os.path.isdir(path):
            os.mkdir(path)

    def get_counterfactuals(self) -> List[Dict[str, List]]:
        self.counterfactuals = []
        for i in range(len(self.x)):
            instance = self.x[i,:]
            discretized = self.CE.discretize(copy.deepcopy(instance).reshape(1,-1))[0]
            instance_regularized = self.binned_proba['regularized'][i]
            instance_low = self.binned_proba['low'][i]
            instance_high = self.binned_proba['high'][i]
            counterfactual = {'weight':[],'weight_low':[],'weight_high':[],'proba':[],'proba_low':[],'proba_high':[],'value':[],'rule':[]}
            
            rule_boundaries = self.CE.rule_boundaries(instance)
            for f in range(len(instance)):
                if f in self.CE.categorical_features:
                    values = np.array(self.CE.feature_values[f])
                    values = np.delete(values, values == discretized[f])
                    for bin, v in enumerate(values):
                        counterfactual['proba'].append(instance_regularized[f][bin])
                        counterfactual['proba_low'].append(instance_low[f][bin])
                        counterfactual['proba_high'].append(instance_high[f][bin])
                        counterfactual['weight'].append(self.proba['regularized'][i][0] - instance_regularized[f][bin])
                        counterfactual['weight_low'].append(self.proba['low'][i][0] - instance_low[f][bin])
                        counterfactual['weight_high'].append(self.proba['high'][i][0] - instance_high[f][bin])
                        counterfactual['value'].append(instance[f])
                        counterfactual['rule'].append('%s = %g' % (self.CE.feature_names[f], v))
                else:                   
                    values = np.array(self.CE.calX[:,f])
                    lesser = rule_boundaries[f][0]
                    greater = rule_boundaries[f][1]
                    
                    bin = 0
                    if np.any(values < lesser):
                        counterfactual['proba'].append(np.mean(instance_regularized[f][bin]))
                        counterfactual['proba_low'].append(np.mean(instance_low[f][bin]))
                        counterfactual['proba_high'].append(np.mean(instance_high[f][bin]))
                        counterfactual['weight'].append(self.proba['regularized'][i][0] - np.mean(instance_regularized[f][bin]))
                        counterfactual['weight_low'].append(self.proba['low'][i][0] - np.mean(instance_low[f][bin]))
                        counterfactual['weight_high'].append(self.proba['high'][i][0] - np.mean(instance_high[f][bin]))
                        counterfactual['value'].append(instance[f]) 
                        counterfactual['rule'].append('%s < %g' % (self.CE.feature_names[f], lesser)) 
                        bin = 1
                    
                    if np.any(values > greater):
                        counterfactual['proba'].append(np.mean(instance_regularized[f][bin]))
                        counterfactual['proba_low'].append(np.mean(instance_low[f][bin]))
                        counterfactual['proba_high'].append(np.mean(instance_high[f][bin]))
                        counterfactual['weight'].append(self.proba['regularized'][i][0] - np.mean(instance_regularized[f][bin]))
                        counterfactual['weight_low'].append(self.proba['low'][i][0] - np.mean(instance_low[f][bin]))
                        counterfactual['weight_high'].append(self.proba['high'][i][0] - np.mean(instance_high[f][bin]))
                        counterfactual['value'].append(instance[f]) 
                        counterfactual['rule'].append('%s > %g' % (self.CE.feature_names[f], greater)) 

            self.counterfactuals.append(counterfactual)
        return self.counterfactuals

    def as_lime(self):
        self.CE.preload_LIME()
        lime_exp = self.CE.lime_exp
        exp = []
        for i in range(len(self.x[:,0])):
            tmp = copy.deepcopy(lime_exp)
            tmp.intercept[1] = 0
            tmp.local_pred = self.proba['regularized'][i]
            tmp.predict_proba[0], tmp.predict_proba[1] = 1-self.proba['regularized'][i], self.proba['regularized'][i]
            
            feature_weights = self.feature_weights['regularized'][i]
            features_to_plot = self.__rank_features(feature_weights, self.CE.num_features)
            rules = self.define_rules(self.x[i,:])
            for j,f in enumerate(features_to_plot[::-1]):
                tmp.local_exp[1][j] = (f, feature_weights[f])
            tmp.domain_mapper.discretized_feature_names = rules
            tmp.domain_mapper.feature_values = self.x[i,:]
            exp.append(tmp)
                # raise NotImplementedError('Not implemented yet')
        self.lime = self.CE.lime
        return exp
    
    def as_shap_values(self):
        self.CE.preload_SHAP()
        f = lambda x: self.CE.va_model.predict_proba(x)[:,1]
        shap = Explainer(f, self.CE.calX[:1,:], feature_names=self.CE.feature_names)
        shap_exp = shap(self.x)   
        for i in range(len(self.x[:,0])):
            shap_exp.base_values[i] = self.proba['regularized'][i]
            for f in range(len(self.x[0,:])):
                shap_exp.values[i][f] = -self.feature_weights['regularized'][i][f]
        return shap_exp
        # raise NotImplementedError('Not implemented yet')

    
    def plot_regular(self, title="", num_to_show=10, postfix="", show=False, path='../plots/'):
        feature_proba = self.feature_proba
        feature_weights = self.feature_weights['regularized']
        proba = self.proba 
        (num_instances, num_features) = np.shape(feature_proba['low'])
        num_to_show = np.min([num_features, num_to_show])

        self.__make_directory(path+title)

        for i in tqdm(range(num_instances)):
            instance = self.x[i,:]
            features_to_plot = self.__rank_features(feature_weights[i], num_to_show)
            column_names = self.define_rules(instance)
            # try:
            self.__plot_weight(instance, proba['regularized'][i], feature_weights[i], features_to_plot, num_to_show, column_names, title, str(i), path, show)
            # except:
            #     print('Error plotting instance ' + str(i))
        

    def plot_uncertainty(self, title="", num_to_show=None, postfix="", show=False, path='../plots/'):
        feature_proba = self.feature_proba
        feature_weights = self.feature_weights['regularized']
        proba = self.proba 
        (num_instances, num_features) = np.shape(feature_proba['low'])
        num_to_show = np.min([num_features, num_to_show])

        self.__make_directory(path+title)

        for i in tqdm(range(num_instances)):
            instance = self.x[i,:]
            features_to_plot = self.__rank_features(feature_weights[i], num_to_show)
            column_names = self.define_rules(instance)
            # try:
            self.__plot_weight(instance, proba, self.feature_weights, features_to_plot, num_to_show, column_names, title, str(i), path, show,interval=True, idx=i)
            # except:
            #     print('Error plotting instance ' + str(i))
    
    def plot_counterfactuals(self, title="", num_to_show=None, postfix="", show=False, path='../plots/',show_base_explanation=False):  
        proba = self.proba 
        counterfactuals = self.get_counterfactuals()

        self.__make_directory(path+title)

        for i, instance in tqdm(enumerate(self.x)):
            counterfactual = counterfactuals[i]
            feature_proba = {'regularized': counterfactual['proba'], 'low': counterfactual['proba_low'], 'high': counterfactual['proba_high']}
            feature_weights = counterfactual['weight']
            # feature_proba = counterfactual['proba'] 
            num_rules = len(counterfactual['rule'])
            num_to_show_ = np.min([num_rules, num_to_show])
            features_to_plot = self.__rank_features(feature_weights, num_to_show_)
            column_names = counterfactual['rule']
            # try:
            self.__plot_counterfactual(counterfactual['value'], proba, feature_proba, \
                                       features_to_plot, num_to_show=num_to_show_, \
                                        column_names=column_names, title=title, postfix=str(i), \
                                        path=path, show=show, idx=i)
            # except:
            #     print('Error plotting instance ' + str(i))
            # 

    def __plot_counterfactual(self, instance, proba, feature_proba, features_to_plot, \
                              num_to_show, column_names, title, postfix, path, show, idx=None):        
        fig = plt.figure(figsize=(10,num_to_show*.5)) 
        ax1 = fig.add_subplot(111)
        ax1.set_xlim(0,1)
        
        x = np.linspace(0, num_to_show-1, num_to_show)
        p_l,p_h = proba['low'][idx][0], proba['high'][idx][0]
        p = proba['regularized'][idx][0]
        venn_abers={'low_high':[p_l,p_h],'reg_proba':p}
        # Fill original Venn Abers interval
        xl = np.linspace(-0.5, x[0], 2)
        xh = np.linspace(x[-1], x[-1]+0.5, 2)
        if (p_l < 0.5 and p_h < 0.5) or (p_l > 0.5 and p_h > 0.5):
            color = self.__get_fill_color(venn_abers,0.15)
            ax1.fill_betweenx(x, [p_l]*(num_to_show), [p_h]*(num_to_show),color=color)
            # Fill up to the edges
            ax1.fill_betweenx(xl, [p_l]*(2), [p_h]*(2),color=color)
            ax1.fill_betweenx(xh, [p_l]*(2), [p_h]*(2),color=color)
        else:
            venn_abers['reg_proba'] = p_l
            color = self.__get_fill_color(venn_abers, 0.15)
            ax1.fill_betweenx(x, [p_l]*(num_to_show), [0.5]*(num_to_show),color=color)
            # Fill up to the edges
            ax1.fill_betweenx(xl, [p_l]*(2), [0.5]*(2),color=color)
            ax1.fill_betweenx(xh, [p_l]*(2), [0.5]*(2),color=color)
            venn_abers['reg_proba'] = p_h
            color = self.__get_fill_color(venn_abers, 0.15)
            ax1.fill_betweenx(x, [0.5]*(num_to_show), [p_h]*(num_to_show),color=color)
            # Fill up to the edges
            ax1.fill_betweenx(xl, [0.5]*(2), [p_h]*(2),color=color)
            ax1.fill_betweenx(xh, [0.5]*(2), [p_h]*(2),color=color)

        for jx, j in enumerate(features_to_plot):
            p_l,p_h = feature_proba['low'][j], feature_proba['high'][j]
            p = feature_proba['regularized'][j]
            xj = np.linspace(x[jx]-0.2, x[jx]+0.2,2)
            venn_abers={'low_high':[p_l,p_h],'reg_proba':p}
            # Fill each feature impact
            if (p_l < 0.5 and p_h < 0.5) or (p_l > 0.5 and p_h > 0.5):
                ax1.fill_betweenx(xj, p_l,p_h,color=self.__get_fill_color(venn_abers, 0.99))
            else:
                venn_abers['reg_proba'] = p_l
                ax1.fill_betweenx(xj, p_l,0.5,color=self.__get_fill_color(venn_abers, 0.99))
                venn_abers['reg_proba'] = p_h
                ax1.fill_betweenx(xj, 0.5,p_h,color=self.__get_fill_color(venn_abers, 0.99))
        
        ax1.set_yticks(range(num_to_show))
        ax1.set_yticklabels(labels=[column_names[i] for i in features_to_plot]) \
            if column_names is not None else ax1.set_yticks(range(num_to_show))
        ax1.set_ylim(-0.5,x[-1]+0.5)
        ax1.set_ylabel('Counterfactual rules') 
        ax2 = ax1.twinx()
        ax2.set_yticks(range(num_to_show))
        ax2.set_yticklabels([np.around(instance[i],decimals=2) for i in features_to_plot], )
        ax2.set_ylim(-0.5,x[-1]+0.5)
        ax2.set_ylabel('Instance values')
        ax1.set_xlabel('Probability for positive class')
        ax1.set_xticks(np.linspace(0, 1, 11))
        fig.tight_layout()
        fig.savefig(path + title + '/' + title + '_' + postfix +'.svg', bbox_inches='tight')
        fig.savefig(path + title + '/' + title + '_' + postfix +'.pdf', bbox_inches='tight')
        fig.savefig(path + title + '/' + title + '_' + postfix +'.png', bbox_inches='tight')
        if show:
            fig.show() 

    def __plot_weight(self, instance, proba, feature_weights, features_to_plot, num_to_show, column_names, title, postfix, path, show, interval=False, idx=None):        
        if interval is True:
            assert idx is not None
        fig = plt.figure(figsize=(10,num_to_show*.5+2))
        subfigs = fig.subfigures(4, 1, height_ratios=[1, 1, 1, num_to_show+2])

        ax00 = subfigs[0].add_subplot(111)
        ax01 = subfigs[1].add_subplot(111)
        
        ax1 = subfigs[3].add_subplot(111)

        # plot the probabilities
        x = np.linspace(0, 1, 2)
        xj = np.linspace(x[0]-0.2, x[0]+0.2,2)
        if interval:
            p = proba['regularized'][idx]
            pl = proba['low'][idx]
            ph = proba['high'][idx]
            ax00.fill_betweenx(xj, 0, 1-ph, color='b')
            ax00.fill_betweenx(xj, 1-pl, 1-ph, color='b', alpha=0.2)
            ax01.fill_betweenx(xj, 0, pl, color='r')
            ax01.fill_betweenx(xj, pl, ph, color='r', alpha=0.2)
        else:   
            p = proba
            ax00.fill_betweenx(xj, 0, 1-p, color='b')
            ax01.fill_betweenx(xj, 0, p, color='r')
        ax00.set_xlim([0,1])
        ax01.set_xlim([0,1])
        ax00.set_yticks(range(1))
        ax01.set_yticks(range(1))
        ax00.set_yticklabels(labels=['P(y=0)'])
        ax01.set_yticklabels(labels=['P(y=1)'])
        ax00.set_xticks([])
        ax01.set_xlabel('Probability')      
        
        x = np.linspace(0, num_to_show-1, num_to_show)
        xl = np.linspace(-0.5, x[0], 2)
        xh = np.linspace(x[-1], x[-1]+0.5, 2)
        ax1.plot([0]*(num_to_show), x, color='k')
        ax1.plot([0]*(2), xl, color='k')
        ax1.plot([0]*(2), xh, color='k')
        if interval:
            p = proba['regularized'][idx]
            if p > 0.5:
                gwl = proba['low'][idx] - p
                gwh = proba['high'][idx] - p
            else:
                gwl = p - proba['low'][idx]
                gwh = p - proba['high'][idx]

            gwh, gwl = np.max([gwh, gwl]), np.min([gwh, gwl])
            ax1.fill_betweenx([-0.5,num_to_show-0.5], gwl, gwh, color='k', alpha=0.2)

        for jx, j in enumerate(features_to_plot):
            xj = np.linspace(x[jx]-0.2, x[jx]+0.2,2)
            if interval:
                w = feature_weights['regularized'][idx][j]
                wl = feature_weights['low'][idx][j] 
                wh = feature_weights['high'][idx][j]
                wh, wl = np.max([wh, wl]), np.min([wh, wl]) 
                mn = wh if w < 0 else 0
                mx = wl if w > 0 else 0
                if wh > 0 and wl < 0:
                    mn = 0
                    mx = 0
            else:
                w = feature_weights[j]
                mn = w if w < 0 else 0
                mx = w if w > 0 else 0
            color = 'r' if w > 0 else 'b'
            ax1.fill_betweenx(xj, mn,mx,color=color)
            if interval:
                if wh > 0 and wl < 0:
                    ax1.fill_betweenx(xj, 0, wl, color='b', alpha=0.2)
                    ax1.fill_betweenx(xj, wh, 0, color='r', alpha=0.2)                    
                else:
                    ax1.fill_betweenx(xj, wl, wh, color=color, alpha=0.2)
        
        ax1.set_yticks(range(num_to_show))
        ax1.set_yticklabels(labels=[column_names[i] for i in features_to_plot]) \
            if column_names is not None else ax1.set_yticks(range(num_to_show))
        ax1.set_ylim(-0.5,x[-1]+0.5)
        ax1.set_ylabel('Rules') 
        ax1.set_xlabel('Feature weights')
        ax12 = ax1.twinx()
        ax12.set_yticks(range(num_to_show))
        ax12.set_yticklabels([np.around(instance[i],decimals=2) for i in features_to_plot])
        ax12.set_ylim(-0.5,x[-1]+0.5)
        ax12.set_ylabel('Instance values')
        fig.savefig(path + title + '/' + title + '_' + postfix +'.svg', bbox_inches='tight')
        fig.savefig(path + title + '/' + title + '_' + postfix +'.pdf', bbox_inches='tight')
        fig.savefig(path + title + '/' + title + '_' + postfix +'.png', bbox_inches='tight')
        if show:
            fig.show() 


class CalibratedAsLimeTabularExplainer(LimeTabularExplainer):
    '''
    Wrapper for LimeTabularExplainer to use calibrated explainer
    '''
    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 discretizer='binaryEntropy',
                 **kwargs):
        self.calibrated_explainer = None
        self.training_data = training_data
        self.training_labels = training_labels
        self.discretizer = discretizer
        self.categorical_features = categorical_features
        self.feature_names = feature_names
        assert training_labels is not None, "Calibrated Explanations requires training labels"

    def explain_instance(self, data_row, classifier, **kwargs):
        if self.calibrated_explainer is None:
            assert 'predict_proba' in dir(classifier), "The classifier must have a predict_proba method."
            self.calibrated_explainer = CalibratedExplainer(classifier, self.training_data, self.training_labels, self.feature_names, self.discretizer, self.categorical_features,)
            self.discretizer = self.calibrated_explainer.discretizer
        return self.calibrated_explainer(data_row).as_lime()[0]
    
class CalibratedAsShapExplainer(Explainer):
    '''
    Wrapper for the CalibratedExplainer to be used as a shap explainer.
    The masker must contain a data and a target field for the calibration set.
    The model must have a predict_proba method.
    '''
    def __init__(self, model, masker, feature_names=None, **kwargs):
        assert ['data','target'] in dir(masker), "The masker must contain a data and a target field for the calibration set."
        assert 'predict_proba' in dir(model), "The model must have a predict_proba method."
        self.calibrated_explainer = CalibratedExplainer(model, masker.data, masker.labels, feature_names)

    def __call__(self, *args, **kwargs):
        return self.calibrated_explainer(*args, **kwargs).as_shap()