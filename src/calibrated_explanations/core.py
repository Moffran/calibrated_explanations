"""Calibrated Explanations for Black-Box Predictions (calibrated-explanationse)
This file contains the code for the calibrated explanations.

The calibrated explanations are based on the paper 
"Calibrated Explanations for Black-Box Predictions" 
by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

Calibrated explanations are a way to explain the predictions of a black-box model 
using Venn Abers predictors (classification) or 
conformal predictive systems (regression) and perturbations.
"""
# pylint: disable=invalid-name, line-too-long
# flake8: noqa: E501
import copy
import numpy as np
import pandas as pd
import sklearn.neighbors as nn
import crepes

from shap import Explainer
from lime.lime_tabular import LimeTabularExplainer

from ._explanations import CalibratedExplanation
from ._discretizers import BinaryDiscretizer, BinaryEntropyDiscretizer, \
                DecileDiscretizer, QuartileDiscretizer, EntropyDiscretizer
from .VennAbers import VennAbers

__version__ = 'v0.0.12'


class CalibratedExplainer:
    """
    The class CalibratedExplainer for black-box models.

    """
     # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-branches, too-many-statements
     # pylint: disable=dangerous-default-value
    def __init__(self,
                 model,
                 cal_X,
                 cal_y,
                 mode = 'classification',
                 feature_names = None,
                 discretizer = None,
                 categorical_features = None,
                 categorical_labels = None,
                 class_labels = None,
                 difficulty_estimator = None,
                 sample_percentiles = [25, 50, 75],
                 n_neighbors = 1.0,
                 random_state = 42,
                 preload_lime=False,
                 preload_shap=False,
                 verbose = False,
                 ) -> None:
         # pylint: disable=line-too-long
        """
        CalibratedExplainer is a class that can be used to explain the predictions of a black-box 
        model.

        Parameters
        ----------
        model : a sklearn predictive model
            A predictive model that can be used to predict the target variable.
        cal_X : array-like of shape (n_calibrations_samples, n_features) 
            The calibration input data for the model.
        cal_y : array-like of shape (n_calibrations_samples,)  
            The calibration target data for the model.
        mode : a string, default="classification" 
            Possible modes are 'classificaiton' or 'regression'.
        feature_names : None or a list of feature names in the shape (n_features,), default=None 
            A list of feature names. Must be None or include one str per feature. 
        discretizer : a string, default='binary' if mode="regression" else 'binaryEntropy'  
            The strategy used for numerical features. Possible discretizers include:
            'binary': split using the median value, suitable for regular 
                                explanations of regression models, 
            'binaryEntropy': use a one-layered decision tree to find the best split suitable 
                                for regular explanations of classification models, 
            'quartile': split using the quartiles suitable for counterfactual explanations of 
                                regression models, 
            'decile': split using the deciles, 
            'entropy': use a three-layered decision tree to find the best splits suitable for 
                                counterfactual explanations of classification models. 
        categorical_features : None or an array-like of shape (n_categorical_features,), default=None 
            A list of indeces for categorical features. 
        categorical_labels : None or a nested dictionary of the shape (n_categorical_features : (n_values : string)), default=None 
            A dictionary with the feature index as key and a feature dictionary mapping each feature 
            value (keys) to a feature label (values). 
        class_labels : None or a dictionary of the shape (n_classes : string), default=None 
            A dictionary mapping numerical target values to class names. 
            Only applicable for classification.
        difficulty_estimator : None or a difficulty_estimator object from the crepes package, default=None 
            A difficulty_estimator object from the crepes package. 
            If None, a normalization is not used. Only used by regression models.
        sample_percentiles : an array-like of shape (n_percentiles) default=[25, 50, 75].
            Percentiles used to sample values for evaluation of numerical features.             
        n_neighbors : either a fraction in the range [0,1) or a fixed integer in the range [1,ncalibration_samples), default=1.0
            Enables a local discretizer to be defined using the nearest neighbors in the calibration set. 
            Values (int) above 1 are interpreted as number of neighbors and float values in the range 
            (0,1] are interpreted as fraction of calibration instances. 
            Defaults to 1.0, meaning 100% of the calibration set is always used.
        random_state : an integer, default=42
            Parameter to adjust the random state. 
        preload_LIME : bool, default=False
            If the LIME wrapper is known to be used, it can be preloaded at initialization. 
        preload_SHAP : bool, default=False 
            If the SHAP wrapper is known to be used, it can be preloaded at initialization. 
        verbose : bool, default=False 
            Enable additional printouts during operation. 
        """
        self.__initialized = False
        if isinstance(cal_X, pd.DataFrame):
            self.cal_X = cal_X.values  # pylint: disable=invalid-name
        else:
            self.cal_X = cal_X
        if isinstance(cal_y, pd.DataFrame):
            self.cal_y = cal_y.values  # pylint: disable=invalid-name
        else:
            self.cal_y = cal_y

        self.model = model
        self.num_features = len(self.cal_X[0, :])
        self.set_random_state(random_state)
        self.sample_percentiles = sample_percentiles
        self.set_num_neighbors(n_neighbors)
        self.verbose = verbose

        self.set_difficulty_estimator(difficulty_estimator, initialize=False)
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

        self.__lime_enabled = False
        if preload_lime:
            self.preload_lime()

        self.__shap_enabled = False
        if preload_shap:
            self.preload_shap()

        self.latest_explanation = None



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


     # pylint: disable=invalid-name
    def predict(self,
                test_X,
                y = None, # The same meaning as y has for cps in crepes.
                low_high_percentiles = (5, 95),
                classes = None,
                ):
        """
        Predicts the target variable for the test data.

        Parameters
        ----------
        testX : A set of test objects to predict
        y : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        classes : None or array-like of shape (n_samples,), default=None
            The classes predicted for the original instance. None if not multiclass or regression.

        Raises
        ------
        ValueError: The length of the y-parameter must be either a constant or the same as the number of 
            instances in testX.

        Returns
        -------
        predict : ndarray of shape (n_samples,)
            The prediction for the test data. For classification, this is the regularized probability 
            of the positive class, derived using the intervals from VennAbers. For regression, this is the 
            median prediction from the ConformalPredictiveSystem.
        low : ndarray of shape (n_samples,)
            The lower bound of the prediction interval. For classification, this is derived using 
            VennAbers. For regression, this is the lower percentile given as parameter, derived from the 
            ConformalPredictiveSystem.
        high : ndarray of shape (n_samples,)
            The upper bound of the prediction interval. For classification, this is derived using 
            VennAbers. For regression, this is the upper percentile given as parameter, derived from the 
            ConformalPredictiveSystem.
        """
        assert self.__initialized, "The model must be initialized before calling predict."
        if self.mode == 'classification':
            if self.is_multiclass():
                predict, low, high, new_classes = self.interval_model.predict_proba(test_X,
                                                                                    output_interval=True,
                                                                                    classes=classes)
                if classes is None:
                    return predict[:,1], low, high, new_classes
                return predict[:,1], low, high, None

            predict, low, high = self.interval_model.predict_proba(test_X, output_interval=True)
            return predict[:,1], low, high, None
        if 'regression' in self.mode:
            predict = self.model.predict(test_X)
            assert low_high_percentiles[0] <= low_high_percentiles[1], \
                        "The low percentile must be smaller than (or equal to) the high percentile."
            assert ((low_high_percentiles[0] > 0 and low_high_percentiles[0] <= 50) and \
                    (low_high_percentiles[1] >= 50 and low_high_percentiles[1] < 100)) or \
                    low_high_percentiles[0] == -np.inf or low_high_percentiles[1] == np.inf and \
                    not (low_high_percentiles[0] == -np.inf and low_high_percentiles[1] == np.inf), \
                        "The percentiles must be between 0 and 100 (exclusive). \
                        The lower percentile can be -np.inf and the higher percentile can \
                        be np.inf (but not at the same time) to allow one-sided intervals."
            low = [low_high_percentiles[0], 50] if low_high_percentiles[0] != -np.inf else [50, 50]
            high = [low_high_percentiles[1], 50] if low_high_percentiles[1] != np.inf else [50, 50]

            sigma_test = self.__get_sigma_test(X=test_X)
             # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            if y is None:
                interval = self.interval_model.predict(y_hat=predict, sigmas=sigma_test,
                            lower_percentiles=low,
                            higher_percentiles=high)
                predict = (interval[:,1] + interval[:,3]) / 2 # The median
                return predict, \
                    interval[:,0] if low_high_percentiles[0] != -np.inf else np.array([min(self.cal_y)]), \
                    interval[:,2] if low_high_percentiles[1] != np.inf else np.array([max(self.cal_y)]), \
                    None
            if not np.isscalar(y) and len(y) != len(test_X):
                raise ValueError("The length of the y parameter must be either a scalar or \
                    the same as the number of instances in testX.")
             # pylint: disable=unexpected-keyword-arg
            y_prob = self.interval_model.predict(y_hat=predict, sigmas=sigma_test,
                                                    y = float(y) if np.isscalar(y) else y,)
            # Use the width of the interval from prediction to determine which interval
            # values to use as low and high thresholds for the interval.
            interval_ = self.interval_model.predict(y_hat=predict, sigmas=sigma_test,
                        lower_percentiles=low,
                        higher_percentiles=high)
            median = (interval_[:,1] + interval_[:,3]) / 2 # The median
            interval = np.array([np.array([0.0,0.0]) for i in range(test_X.shape[0])])
            for i,_ in enumerate(test_X):
                interval[i,0] = self.interval_model.predict(y_hat=[predict[i]],
                                                            sigmas=sigma_test,
                                                            y=float(interval_[i,0] - median[i] + y))
                interval[i,1] = self.interval_model.predict(y_hat=[predict[i]],
                                                            sigmas=sigma_test,
                                                            y=float(interval_[i,2] - median[i] + y))
            predict = y_prob
            # Changed to 1-p so that high probability means high prediction and vice versa
            return [1-predict[0]], 1-interval[:,1], 1-interval[:,0], None
        return None, None, None, None # Should never happen

    def get_factuals(self,
                 test_X,
                 y = None,
                 low_high_percentiles = (5, 95),
                 ) -> CalibratedExplanation:
        """
        Creates a CalibratedExplanation object for the test data with the discretizer automatically assigned for factual explanations.

        Parameters
        ----------
        testX : A set of test objects to predict
        y : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The y-parameter is only supported for mode='regression'.
        ValueError: The length of the y parameter must be either a constant or the same as the number of 
            instances in testX.

        Returns
        -------
        CalibratedExplanations : A CalibratedExplanations object containing the predictions and the 
            intervals. 
        """
        if 'regression' in self.mode:
            discretizer = 'binary'
        else:
            discretizer = 'binaryEntropy'
        self.set_discretizer(discretizer)
        return self(test_X, y, low_high_percentiles)

    def get_counterfactuals(self,
                 test_X,
                 y = None,
                 low_high_percentiles = (5, 95),
                 ) -> CalibratedExplanation:
        """
        Creates a CalibratedExplanation object for the test data with the discretizer automatically assigned for counterfactual explanations.

        Parameters
        ----------
        testX : A set of test objects to predict
        y : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The y-parameter is only supported for mode='regression'.
        ValueError: The length of the y parameter must be either a constant or the same as the number of 
            instances in testX.

        Returns
        -------
        CalibratedExplanations : A CalibratedExplanations object containing the predictions and the 
            intervals. 
        """
        if 'regression' in self.mode:
            discretizer = 'decile'
        else:
            discretizer = 'entropy'
        self.set_discretizer(discretizer)
        return self(test_X, y, low_high_percentiles)

    def __call__(self,
                 testX,
                 y = None,
                 low_high_percentiles = (5, 95),
                 ) -> CalibratedExplanation:
        """
        Creates a CalibratedExplanation object for the test data with the already assigned discretizer.

        Parameters
        ----------
        testX : A set of test objects to predict
        y : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The y-parameter is only supported for mode='regression'.
        ValueError: The length of the y parameter must be either a constant or the same as the number of 
            instances in testX.

        Returns
        -------
        CalibratedExplanations : A CalibratedExplanations object containing the predictions and the 
            intervals. 
        """
        if len(testX.shape) == 1:
            testX = testX.reshape(1, -1)
        if testX.shape[1] != self.cal_X.shape[1]:
            raise ValueError("The number of features in the test data must be the same as in the \
                            calibration data.")
        explanation = CalibratedExplanation(self, testX)
        discretizer = self.__get_discretizer()

        if y is not None:
            if not 'regression' in self.mode:
                raise Warning("The y parameter is only supported for mode='regression'.")
            if not np.isscalar(y) and len(y) != len(testX):
                raise ValueError("The length of the y parameter must be either a constant or the same \
                                as the number of instances in testX.")
            explanation.y = y
            explanation.low_high_percentiles = low_high_percentiles
        elif 'regression' in self.mode:
            explanation.low_high_percentiles = low_high_percentiles

        cal_X = self.cal_X
        cal_y = self.cal_y

        feature_weights =  {'predict': [],'low': [],'high': [],}
        feature_predict =  {'predict': [],'low': [],'high': [],}
        prediction =  {'predict': [],'low': [],'high': [], 'classes': []}
        binned_predict =  {'predict': [],'low': [],'high': [],'current_bin': [],'rule_values': []}

        for i,x in enumerate(testX):
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

            if not self.num_neighbors == len(self.cal_y):
                cal_X, cal_y = self.find_local_calibration_data(x)
                self.set_discretizer(discretizer, cal_X, cal_y)

            rule_values = {}
            instance_weights = {'predict':np.zeros(x.shape[0]),'low':np.zeros(x.shape[0]),'high':np.zeros(x.shape[0])}
            instance_predict = {'predict':np.zeros(x.shape[0]),'low':np.zeros(x.shape[0]),'high':np.zeros(x.shape[0])}
            instance_binned = {'predict': [],'low': [],'high': [],'current_bin': [],'rule_values': []}
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
                    for bin_value, value in enumerate(values):  # For each bin (i.e. discretized value) in the values array...
                        perturbed[f] = perturbed_original[0,f] # Assign the original discretized value to ensure similarity to value
                        if perturbed[f] == value:
                            current_bin = bin_value  # If the discretized value is the same as the original, skip it

                        perturbed[f] = value
                        predict, low, high, _ = self.predict(perturbed.reshape(1,-1), y=y, low_high_percentiles=low_high_percentiles, classes=predicted_class)
                        average_predict[bin_value] = predict[0]
                        low_predict[bin_value] = low[0]
                        high_predict[bin_value] = high[0]
                else:
                    rule_value = []
                    values = np.array(cal_X[:,f])
                    lesser = rule_boundaries[f][0]
                    greater = rule_boundaries[f][1]
                    num_bins = 1
                    num_bins += int(np.any(values > greater))
                    num_bins += int(np.any(values < lesser))
                    average_predict, low_predict, high_predict = np.zeros(num_bins),np.zeros(num_bins),np.zeros(num_bins)

                    bin_value = 0
                    if np.any(values < lesser):
                        lesser_values = np.unique(self.__get_lesser_values(f, lesser))
                        rule_value.append(lesser_values)
                        for value in lesser_values:
                            perturbed[f] = value
                            predict, low, high, _ = self.predict(perturbed.reshape(1,-1), y=y, low_high_percentiles=low_high_percentiles, classes=predicted_class)
                            average_predict[bin_value] += predict[0]
                            low_predict[bin_value] += low[0]
                            high_predict[bin_value] += high[0]
                        average_predict[bin_value] = average_predict[bin_value]/len(lesser_values)
                        low_predict[bin_value] = low_predict[bin_value]/len(lesser_values)
                        high_predict[bin_value] = high_predict[bin_value]/len(lesser_values)
                        bin_value += 1
                    if np.any(values > greater):
                        greater_values = self.__get_greater_values(f, greater)
                        rule_value.append(greater_values)
                        for value in greater_values:
                            perturbed[f] = value
                            predict, low, high, _ = self.predict(perturbed.reshape(1,-1), y=y, low_high_percentiles=low_high_percentiles, classes=predicted_class)
                            average_predict[bin_value] += predict[0]
                            low_predict[bin_value] += low[0]
                            high_predict[bin_value] += high[0]
                        average_predict[bin_value] = average_predict[bin_value]/len(greater_values)
                        low_predict[bin_value] = low_predict[bin_value]/len(greater_values)
                        high_predict[bin_value] = high_predict[bin_value]/len(greater_values)
                        bin_value += 1

                    covered_values = self.__get_covered_values(f, lesser, greater)
                    rule_value.append(covered_values)
                    for value in covered_values:
                        perturbed[f] = value
                        predict, low, high, _ = self.predict(perturbed.reshape(1,-1), y=y, low_high_percentiles=low_high_percentiles, classes=predicted_class)
                        average_predict[bin_value] += predict[0]
                        low_predict[bin_value] += low[0]
                        high_predict[bin_value] += high[0]
                    average_predict[bin_value] = average_predict[bin_value]/len(covered_values)
                    low_predict[bin_value] = low_predict[bin_value]/len(covered_values)
                    high_predict[bin_value] = high_predict[bin_value]/len(covered_values)
                    current_bin = bin_value

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
        """test if it is a multiclass problem

        Returns:
            bool: True if multiclass
        """
        return self.num_classes > 2



    def rule_boundaries(self, instance, perturbed_instance=None):
        """extracts the rule boundaries for the instance

        Args:
            instance (n_features,): the instance to extract boundaries for
            perturbed_instance ((n_features,), optional): a discretized version of instance. Defaults to None.

        Returns:
            (n_features, 2): min and max values for each feature
        """
        min_max = []
        if perturbed_instance is None:
            perturbed_instance = self.discretize(instance.reshape(1,-1))
        for f in range(self.num_features):
            if f in self.categorical_features:
                min_max.append([instance[f], instance[f]])
            else:
                values = np.array(self.discretizer.means[f])
                min_max.append([self.discretizer.mins[f][np.where(
                                    perturbed_instance[0,f] == values)[0][0]], \
                                self.discretizer.maxs[f][np.where(
                                    perturbed_instance[0,f] == values)[0][0]]])
        return min_max



    def __get_greater_values(self, f: int, greater: float):
        greater_values = np.percentile(self.cal_X[self.cal_X[:,f] > greater,f],
                                       self.sample_percentiles)
        return greater_values



    def __get_lesser_values(self, f: int, lesser: float):
        lesser_values = np.percentile(self.cal_X[self.cal_X[:,f] < lesser,f],
                                      self.sample_percentiles)
        return lesser_values



    def __get_covered_values(self, f: int, lesser: float, greater: float):
        covered = np.where((self.cal_X[:,f] >= lesser) & (self.cal_X[:,f] <= greater))[0]
        covered_values = np.percentile(self.cal_X[covered,f], self.sample_percentiles)
        return covered_values



    def set_random_state(self, random_state: int) -> None:
        """changes the random seed

        Args:
            random_state (int): a seed to the random number generator
        """
        self.random_state = random_state
        np.random.seed(self.random_state)



    def set_difficulty_estimator(self, difficulty_estimator, initialize=True) -> None:
        """assigns a difficulty estimator for regression. For further information, 
        see the documentation for the difficulty estimator or refer to the crepes package 
        for further information.

        Args:
            difficulty_estimator (crepes.extras.DifficultyEstimator): A DifficultyEstimator object from the crepes package
            initialize (bool, optional): If true, then the interval model is initialized once done. Defaults to True.
        """
        self.__initialized = False
        self.difficulty_estimator = difficulty_estimator
        # initialize the model with the new sigma
        if initialize:
            self.__initialize_interval_model()



    def __constant_sigma(self, X: np.ndarray, learner=None, beta=None) -> np.ndarray:  # pylint: disable=unused-argument
        return np.ones(X.shape[0])



    def __get_sigma_test(self, X: np.ndarray) -> np.ndarray:
        if self.difficulty_estimator is None:
            return self.__constant_sigma(X)
        return self.difficulty_estimator.apply(X)



    def set_mode(self, mode, initialize=True) -> None:
        """assign the mode of the explainer. The mode can be either 'classification' or 'regression'.

        Args:
            mode (str): The mode can be either 'classification' or 'regression'.
            initialize (bool, optional): If true, then the interval model is initialized once done. Defaults to True.

        Raises:
            ValueError: The mode can be either 'classification' or 'regression'.
        """
        self.__initialized = False
        if mode == 'classification':
            assert 'predict_proba' in dir(self.model), "The model must have a predict_proba method."
            self.num_classes = len(np.unique(self.cal_y))
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
            va = VennAbers(self.cal_X, self.cal_y, self.model)
            self.interval_model = va
        elif 'regression' in self.mode:
            cal_y_hat = self.model.predict(self.cal_X)
            self.residual_cal = self.cal_y - cal_y_hat
            cps = crepes.ConformalPredictiveSystem()
            if self.difficulty_estimator is not None:
                sigma_cal = self.difficulty_estimator.apply(X=self.cal_X)
                cps.fit(residuals=self.residual_cal, sigmas=sigma_cal)
            else:
                cps.fit(residuals=self.residual_cal)
            self.interval_model = cps
        self.__initialized = True



    def set_num_neighbors(self, n_neighbors) -> None:
        """Enables a local discretizer to be defined using the nearest neighbors in the calibration set. 
        Values (int) above 1 are interpreted as number of neighbors and float values in the range 
        (0,1] are interpreted as fraction of calibration instances. 
        Defaults to 1.0, meaning 100% of the calibration set is always used.

        Args:
            n_neighbors (int or float): either a fraction in the range [0,1) or a fixed integer in the range [1,ncalibration_samples), default=1.0

        Raises:
            ValueError: if n_neighbors < 0
        """
        if n_neighbors < 0:
            raise ValueError("num_neighbors must be positive")
        if n_neighbors <= 1.0:
            n_neighbors = int(len(self.cal_X) * n_neighbors)
        self.num_neighbors = n_neighbors



    def find_local_calibration_data(self, x):
        """applies nearest neighbor to find the local calibration data closest to the instance x

        Args:
            x (n_features,): the test instance to find calibration data for

        Returns:
            cal_X (n_neighbors,), cal_y (n_neighbors,): returns input and output for the n_neighbors closest calibration instances
        """
        nn_model = nn.NearestNeighbors(n_neighbors=self.num_neighbors,
                                       algorithm='ball_tree').fit(self.cal_X)
        _, indices = nn_model.kneighbors(x.reshape(1,-1))
        return self.cal_X[indices[0]], self.cal_y[indices[0]]



    def discretize(self, x):
        """applies the discretizer to the test instance x

        Args:
            x (n_features,): the test instance to discretize

        Returns:
            (n_features,): a perturbed test instance
        """
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
        if isinstance(self.discretizer, DecileDiscretizer):
            return 'decile'
        if isinstance(self.discretizer, EntropyDiscretizer):
            return 'entropy'
        if isinstance(self.discretizer, BinaryEntropyDiscretizer):
            return 'binaryEntropy'
        if isinstance(self.discretizer, BinaryDiscretizer):
            return 'binary'
        raise ValueError("The discretizer is not supported.")


     # pylint: disable=too-many-branches
    def set_discretizer(self, discretizer: str, cal_X=None, cal_y=None) -> None:
        """assign discretizer to the explainer. 
        The discretizer can be either 'quartile', 'decile', 'entropy', 'binary', or 'binaryEntropy'. 
        Once the discretizer is assigned, the calibration data is discretized.

        Args:
            discretizer (str): _description_
            cal_X ((n_calibration_samples,n_features), optional): calibration inputs. Defaults to None.
            cal_y ((n_calibrations_samples), optional): calibration targets. Defaults to None.
        """
        if cal_X is None:
            cal_X = self.cal_X
        if cal_y is None:
            cal_y = self.cal_y

        if discretizer is None:
            if 'regression' in self.mode:
                discretizer = 'binary'
            else:
                discretizer = 'binaryEntropy'
        else:
            if 'regression'in self.mode:
                assert discretizer is None or discretizer in ['binary', 'quartile', 'decile'], \
                    "The discretizer must be 'binary' (default), 'quartile', or \
                    'decile' for regression."

        if discretizer == 'quartile':
            self.discretizer = QuartileDiscretizer(
                    cal_X, self.categorical_features,
                    self.feature_names, labels=cal_y,
                    random_state=self.random_state)
        elif discretizer == 'decile':
            self.discretizer = DecileDiscretizer(
                    cal_X, self.categorical_features,
                    self.feature_names, labels=cal_y,
                    random_state=self.random_state)
        elif discretizer == 'entropy':
            self.discretizer = EntropyDiscretizer(
                    cal_X, self.categorical_features,
                    self.feature_names, labels=cal_y,
                    random_state=self.random_state)
        elif discretizer == 'binary':
            self.discretizer = BinaryDiscretizer(
                    cal_X, self.categorical_features,
                    self.feature_names, labels=cal_y,
                    random_state=self.random_state)
        elif discretizer == 'binaryEntropy':
            self.discretizer = BinaryEntropyDiscretizer(
                    cal_X, self.categorical_features,
                    self.feature_names, labels=cal_y,
                    random_state=self.random_state)

        self.discretized_cal_X = self.discretize(copy.deepcopy(self.cal_X))

        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in range(self.cal_X.shape[1]):
            column = self.discretized_cal_X[:, feature]
            feature_count = {}
            for item in column:
                feature_count[item] = feature_count.get(item, 0) + 1
            values, frequencies = map(list, zip(*(sorted(feature_count.items()))))

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))



    def set_latest_explanation(self, explanation) -> None:
        """assigns the latest explanation to the explainer

        Args:
            explanation (CalibratedExplanation): the latest created explanation
        """
        self.latest_explanation = explanation



    def is_lime_enabled(self, is_enabled=None) -> bool:
        """returns whether lime export is enabled. 
        If is_enabled is not None, then the lime export is enabled/disabled according to the value of is_enabled.

        Args:
            is_enabled (bool, optional): is used to assign whether lime export is enabled or not. Defaults to None.

        Returns:
            bool: returns whether lime export is enabled
        """
        if is_enabled is not None:
            self.__lime_enabled = is_enabled
        return self.__lime_enabled



    def is_shap_enabled(self, is_enabled=None) -> bool:
        """returns whether shap export is enabled. 
        If is_enabled is not None, then the shap export is enabled/disabled according to the value of is_enabled.

        Args:
            is_enabled (bool, optional): is used to assign whether shap export is enabled or not. Defaults to None.

        Returns:
            bool: returns whether shap export is enabled
        """
        if is_enabled is not None:
            self.__shap_enabled = is_enabled
        return self.__shap_enabled



    def preload_lime(self) -> None:
        """creates a lime structure for the explainer

        Returns:
            LimeTabularExplainer: a LimeTabularExplainer object defined for the problem
            lime_exp: a template lime explanation achieved through the explain_instance method
        """
        if not self.is_lime_enabled():
            if self.mode == 'classification':
                self.lime = LimeTabularExplainer(self.cal_X[:1, :],
                                                 feature_names=self.feature_names,
                                                 class_names=['0','1'],
                                                 mode=self.mode)
                self.lime_exp = self.lime.explain_instance(self.cal_X[0, :],
                                                           self.model.predict_proba,
                                                           num_features=self.num_features)
            elif 'regression' in self.mode:
                self.lime = LimeTabularExplainer(self.cal_X[:1, :],
                                                 feature_names=self.feature_names,
                                                 mode='regression')
                self.lime_exp = self.lime.explain_instance(self.cal_X[0, :],
                                                           self.model.predict,
                                                           num_features=self.num_features)
            self.is_lime_enabled(True)
        return self.lime, self.lime_exp



    def preload_shap(self, num_test=None) -> None:
        """creates a shap structure for the explainer

        Returns:
            shap.Explainer: a Explainer object defined for the problem
            shap_exp: a template shap explanation achieved through the __call__ method
        """
         # pylint: disable=access-member-before-definition
        if not self.is_shap_enabled() or \
            num_test is not None and self.shap_exp.shape[0] != num_test:
            f = lambda x: self.predict(x)[0]  # pylint: disable=unnecessary-lambda-assignment
            self.shap = Explainer(f, self.cal_X[:1, :], feature_names=self.feature_names)
            self.shap_exp = self.shap(self.cal_X[0, :].reshape(1,-1)) \
                                    if num_test is None else self.shap(self.cal_X[:num_test, :])
            self.is_shap_enabled(True)
        return self.shap, self.shap_exp
