"""Calibrated Explanations for Black-Box Predictions (calibrated-explanations)

The calibrated explanations are based on the paper 
"Calibrated Explanations for Black-Box Predictions" 
by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

Calibrated explanations are a way to explain the predictions of a black-box model 
using Venn-Abers predictors (classification) or 
conformal predictive systems (regression).
"""
# pylint: disable=invalid-name, line-too-long, too-many-lines
# flake8: noqa: E501
import copy
import warnings
import numpy as np

from lime.lime_tabular import LimeTabularExplainer

from ._explanations import CalibratedExplanations
from ._discretizers import BinaryDiscretizer, BinaryEntropyDiscretizer, \
                DecileDiscretizer, QuartileDiscretizer, EntropyDiscretizer
from .VennAbers import VennAbers
from ._interval_regressor import IntervalRegressor
from .utils import safe_isinstance, safe_import, check_is_fitted

__version__ = 'v0.2.1'



class CalibratedExplainer:
    """The CalibratedExplainer class is used for explaining machine learning models with calibrated
    predictions.

    The calibrated explanations are based on the paper 
    "Calibrated Explanations for Black-Box Predictions" 
    by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

    Calibrated explanations provides a way to explain the predictions of a black-box model 
    using Venn-Abers predictors (classification) or 
    conformal predictive systems (regression).
    """
    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    # pylint: disable=dangerous-default-value
    def __init__(self,
                model,
                cal_X,
                cal_y,
                mode = 'classification',
                feature_names = None,
                categorical_features = None,
                categorical_labels = None,
                class_labels = None,
                difficulty_estimator = None,
                sample_percentiles = [25, 50, 75],
                random_state = 42,
                verbose = False,
                ) -> None:
        # pylint: disable=line-too-long
        '''Constructor for the CalibratedExplainer object for explaining the predictions of a
        black-box model.
        
        Parameters
        ----------
        model : predictive model
            A predictive model that can be used to predict the target variable. The model must be fitted and have a predict_proba method (for classification) or a predict method (for regression).
        cal_X : array-like of shape (n_calibrations_samples, n_features)
            The calibration input data for the model.
        cal_y : array-like of shape (n_calibrations_samples,)
            The calibration target data for the model.
        mode : str equal to "classification" or "regression", default="classification"
            The mode parameter specifies the type of problem being solved.
        feature_names : list of str, default=None
            A list of feature names for the input data. Each feature name should be a string. If not
            provided, the feature names will be assigned as "0", "1", "2", etc.
        categorical_features : list of int, default=None
            A list of indices for categorical features. These are the features that have discrete values
            and are not continuous.
        categorical_labels : dict(int, dict(int, str)), default=None
            A nested dictionary that maps the index of categorical features to another dictionary. The
            inner dictionary maps each feature value to a feature label. This is used for categorical
            feature encoding in the explanations. If None, the feature values will be used as labels.
        class_labels : dict(int, str), default=None
            A dictionary mapping numerical target values to class names. This parameter is only applicable
            for classification models. If None, the numerical target values will be used as labels.
        difficulty_estimator : DifficultyEstimator, default=None
            A `DifficultyEstimator` object from the `crepes` package. It is used to estimate the difficulty of
            explaining a prediction. If None, no difficulty estimation is used. This parameter is only used
            for regression models.
        sample_percentiles : list of int, default=[25, 50, 75]
            An array-like object that specifies the percentiles used to sample values for evaluation of
            numerical features. For example, if `sample_percentiles = [25, 50, 75]`, then the values at the
            25th, 50th, and 75th percentiles within each discretized group will be sampled from the calibration 
            data for each numerical feature.
        random_state : int, default=42
            The random_state parameter is an integer that is used to set the random state for
            reproducibility. It is used in various parts of the code where randomization is involved, such
            as sampling values for evaluation of numerical features or initializing the random state for
            certain operations.
        verbose : bool, default=False
            A boolean parameter that determines whether additional printouts should be enabled during the
            operation of the class. If set to True, it will print out additional information during the
            execution of the code. If set to False, it will not print out any additional information.
        
        Return
        ------
        CalibratedExplainer : A CalibratedExplainer object that can be used to explain predictions from a predictive model.
        
        '''
        self.__initialized = False
        if safe_isinstance(cal_X, "pandas.core.frame.DataFrame"):
            self.cal_X = cal_X.values  # pylint: disable=invalid-name
        else:
            self.cal_X = cal_X
        if safe_isinstance(cal_y, "pandas.core.frame.DataFrame"):
            self.cal_y = cal_y.values  # pylint: disable=invalid-name
        else:
            self.cal_y = cal_y

        check_is_fitted(model)
        self.model = model
        self.num_features = len(self.cal_X[0, :])
        self.set_random_state(random_state)
        self.sample_percentiles = sample_percentiles
        self.verbose = verbose

        self.set_difficulty_estimator(difficulty_estimator, initialize=False)
        self.__set_mode(str.lower(mode), initialize=False)
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

        self.discretizer = None
        self.discretized_cal_X = None
        self.feature_values = {}
        self.feature_frequencies = {}
        self.latest_explanation = None
        self.__shap_enabled = False
        self.__lime_enabled = False
        self.lime = None
        self.lime_exp = None
        self.shap = None
        self.shap_exp = None



    def __repr__(self):
        return f"CalibratedExplainer:\n\t\
                mode={self.mode}\n\t\
                discretizer={self.discretizer.__class__}\n\t\
                model={self.model}\n\t\
                {f'difficulty_estimator={self.difficulty_estimator}' if self.mode == 'regression' else ''}"

                # feature_names={self.feature_names}\n\t\
                # categorical_features={self.categorical_features}\n\t\
                # categorical_labels={self.categorical_labels}\n\t\
                # class_labels={self.class_labels}\n\t\
                # sample_percentiles={self.sample_percentiles}\n\t\
                # num_neighbors={self.num_neighbors}\n\t\
                # random_state={self.random_state}\n\t\
                # verbose={self.verbose}\n\t\


    # pylint: disable=invalid-name
    def _predict(self,
                test_X,
                threshold = None, # The same meaning as threshold has for cps in crepes.
                low_high_percentiles = (5, 95),
                classes = None,
                ):
        # """
        # Predicts the target variable for the test data.

        # Parameters
        # ----------
        # testX : A set of test objects to predict
        # threshold : float, int or array-like of shape (n_samples,), default=None
        #     values for which p-values should be returned. Only used for probabilistic explanations for regression.
        # low_high_percentiles : a tuple of floats, default=(5, 95)
        #     The low and high percentile used to calculate the interval. Applicable to regression.
        # classes : None or array-like of shape (n_samples,), default=None
        #     The classes predicted for the original instance. None if not multiclass or regression.

        # Raises
        # ------
        # ValueError: The length of the threshold-parameter must be either a constant or the same as the number of
        #     instances in testX.

        # Returns
        # -------
        # predict : ndarray of shape (n_samples,)
        #     The prediction for the test data. For classification, this is the regularized probability
        #     of the positive class, derived using the intervals from VennAbers. For regression, this is the
        #     median prediction from the ConformalPredictiveSystem.
        # low : ndarray of shape (n_samples,)
        #     The lower bound of the prediction interval. For classification, this is derived using
        #     VennAbers. For regression, this is the lower percentile given as parameter, derived from the
        #     ConformalPredictiveSystem.
        # high : ndarray of shape (n_samples,)
        #     The upper bound of the prediction interval. For classification, this is derived using
        #     VennAbers. For regression, this is the upper percentile given as parameter, derived from the
        #     ConformalPredictiveSystem.
        # classes : ndarray of shape (n_samples,)
        #     The classes predicted for the original instance. None if not multiclass or regression.
        # """
        assert self.__initialized, "The model must be initialized before calling predict."
        if self.mode == 'classification':
            if self._is_multiclass():
                predict, low, high, new_classes = self.interval_model.predict_proba(test_X,
                                                                                    output_interval=True,
                                                                                    classes=classes)
                if classes is None:
                    return predict[:,1], low, high, new_classes
                return predict[:,1], low, high, None

            predict, low, high = self.interval_model.predict_proba(test_X, output_interval=True)
            return predict[:,1], low, high, None
        if 'regression' in self.mode:
            # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            if threshold is None: # normal regression
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

                return self.interval_model.predict_uncertainty(test_X, low_high_percentiles)

            # regression with threshold condition
            if not np.isscalar(threshold) and len(threshold) != len(test_X):
                raise ValueError("The length of the threshold parameter must be either a scalar or \
                    the same as the number of instances in testX.")
            # pylint: disable=unexpected-keyword-arg
            return self.interval_model.predict_probability(test_X, threshold)

        return None, None, None, None # Should never happen

    def explain_factual(self,
                        test_X,
                        threshold = None,
                        low_high_percentiles = (5, 95),
                        ) -> CalibratedExplanations:
        """
        Creates a CalibratedExplanations object for the test data with the discretizer automatically assigned for factual explanations.

        Parameters
        ----------
        testX : A set of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of 
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
        return self(test_X, threshold, low_high_percentiles)

    def explain_counterfactual(self,
                                test_X,
                                threshold = None,
                                low_high_percentiles = (5, 95),
                                ) -> CalibratedExplanations:
        """
        Creates a CalibratedExplanations object for the test data with the discretizer automatically assigned for counterfactual explanations.

        Parameters
        ----------
        testX : A set of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of 
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
        return self(test_X, threshold, low_high_percentiles)

    def __call__(self,
                testX,
                threshold = None,
                low_high_percentiles = (5, 95),
                ) -> CalibratedExplanations:
        """
        Calling self as a function creates a CalibratedExplanations object for the test data with the 
        already assigned discretizer. Called by the `explain_factual` and `explain_counterfactual` methods. 
        See their documentation for further information.
        """
        if len(testX.shape) == 1:
            testX = testX.reshape(1, -1)
        if testX.shape[1] != self.cal_X.shape[1]:
            raise ValueError("The number of features in the test data must be the same as in the \
                            calibration data.")
        explanation = CalibratedExplanations(self, testX, threshold)

        is_probabilistic = True # classification or when threshold is used for regression
        if threshold is not None:
            if not 'regression' in self.mode:
                raise Warning("The threshold parameter is only supported for mode='regression'.")
            if not np.isscalar(threshold) and len(threshold) != len(testX):
                raise ValueError("The length of the threshold parameter must be either a constant or the same \
                                as the number of instances in testX.")
            # explanation.low_high_percentiles = low_high_percentiles
        elif 'regression' in self.mode:
            explanation.low_high_percentiles = low_high_percentiles
            is_probabilistic = False

        cal_X = self.cal_X

        feature_weights =  {'predict': [],'low': [],'high': [],}
        feature_predict =  {'predict': [],'low': [],'high': [],}
        prediction =  {'predict': [],'low': [],'high': [], 'classes': []}
        binned_predict =  {'predict': [],'low': [],'high': [],'current_bin': [],'rule_values': []}

        for i, x in enumerate(testX):
            if threshold is not None and not np.isscalar(explanation.y_threshold):
                threshold = float(explanation.y_threshold[i])
            predict, low, high, predicted_class = self._predict(x.reshape(1,-1), threshold=threshold, low_high_percentiles=low_high_percentiles)
            prediction['predict'].append(predict[0])
            prediction['low'].append(low[0])
            prediction['high'].append(high[0])
            if self._is_multiclass():
                prediction['classes'].append(predicted_class[0])
            else:
                prediction['classes'].append(1)

            rule_values = {}
            instance_weights = {'predict':np.zeros(x.shape[0]),'low':np.zeros(x.shape[0]),'high':np.zeros(x.shape[0])}
            instance_predict = {'predict':np.zeros(x.shape[0]),'low':np.zeros(x.shape[0]),'high':np.zeros(x.shape[0])}
            instance_binned = {'predict': [],'low': [],'high': [],'current_bin': [],'rule_values': []}
            # Get the perturbations
            x_original = copy.deepcopy(x)
            perturbed_original = self._discretize(copy.deepcopy(x).reshape(1,-1))
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
                        predict, low, high, _ = self._predict(perturbed.reshape(1,-1), threshold=threshold, low_high_percentiles=low_high_percentiles, classes=predicted_class)
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
                            predict, low, high, _ = self._predict(perturbed.reshape(1,-1), threshold=threshold, low_high_percentiles=low_high_percentiles, classes=predicted_class)
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
                            predict, low, high, _ = self._predict(perturbed.reshape(1,-1), threshold=threshold, low_high_percentiles=low_high_percentiles, classes=predicted_class)
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
                        predict, low, high, _ = self._predict(perturbed.reshape(1,-1), threshold=threshold, low_high_percentiles=low_high_percentiles, classes=predicted_class)
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
                    instance_predict['predict'][f] = 0
                    instance_predict['low'][f] = 0
                    instance_predict['high'][f] = 0

                    instance_weights['predict'][f] = 0
                    instance_weights['low'][f] = 0
                    instance_weights['high'][f] = 0
                else:
                    instance_predict['predict'][f] = np.mean(average_predict[uncovered])
                    instance_predict['low'][f] = np.mean(low_predict[uncovered])
                    instance_predict['high'][f] = np.mean(high_predict[uncovered])

                    instance_weights['predict'][f] = self._assign_weight(instance_predict['predict'][f], prediction['predict'][-1], is_probabilistic)
                    tmp_low = self._assign_weight(instance_predict['low'][f], prediction['predict'][-1], is_probabilistic)
                    tmp_high = self._assign_weight(instance_predict['high'][f], prediction['predict'][-1], is_probabilistic)
                    instance_weights['low'][f] = np.min([tmp_low, tmp_high])
                    instance_weights['high'][f] = np.max([tmp_low, tmp_high])

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

        explanation._finalize(binned_predict, feature_weights, feature_predict, prediction)
        return explanation



    def _assign_weight(self, instance_predict, prediction, is_probabilistic):
        if is_probabilistic:
            return prediction - instance_predict # probabilistic regression
        return prediction - instance_predict # standard regression



    def _is_multiclass(self):
        # """test if it is a multiclass problem

        # Returns:
        #     bool: True if multiclass
        # """
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
            perturbed_instance = self._discretize(instance.reshape(1,-1))
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
        if difficulty_estimator is not None:
            sklearn = safe_import('sklearn')
            if sklearn:
                try:
                    if not difficulty_estimator.fitted:
                        raise sklearn.utils.validation.NotFittedError("The difficulty estimator is not fitted. Please fit the estimator first.")
                except AttributeError as e:
                    raise sklearn.utils.validation.NotFittedError("The difficulty estimator is not fitted. Please fit the estimator first.") from e
            else:
                try:
                    if not difficulty_estimator.fitted:
                        raise RuntimeError("The difficulty estimator is not fitted. Please fit the estimator first.")
                except AttributeError as e:
                    raise RuntimeError("The difficulty estimator is not fitted. Please fit the estimator first.") from e
        if initialize:
            self.__initialize_interval_model()



    def __constant_sigma(self, X: np.ndarray, learner=None, beta=None) -> np.ndarray:  # pylint: disable=unused-argument
        return np.ones(X.shape[0])



    def _get_sigma_test(self, X: np.ndarray) -> np.ndarray:
        # """returns the difficulty (sigma) of the test instances

        # """
        if self.difficulty_estimator is None:
            return self.__constant_sigma(X)
        return self.difficulty_estimator.apply(X)



    def __set_mode(self, mode, initialize=True) -> None:
        # """assign the mode of the explainer. The mode can be either 'classification' or 'regression'.

        # Args:
        #     mode (str): The mode can be either 'classification' or 'regression'.
        #     initialize (bool, optional): If true, then the interval model is initialized once done. Defaults to True.

        # Raises:
        #     ValueError: The mode can be either 'classification' or 'regression'.
        # """
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
            self.interval_model = VennAbers(self.model.predict_proba(self.cal_X), self.cal_y, self.model)
        elif 'regression' in self.mode:
            self.interval_model = IntervalRegressor(self)
        self.__initialized = True



    def _discretize(self, x):
        # """applies the discretizer to the test instance x

        # Args:
        #     x (n_features,): the test instance to discretize

        # Returns:
        #     (n_features,): a perturbed test instance
        # """
        if len(np.shape(x)) == 1:
            x = np.array(x)
        x.dtype = float
        tmp = self.discretizer.discretize(x)
        for f in self.discretizer.to_discretize:
            x[:,f] = [self.discretizer.means[f][int(tmp[i,f])] for i in range(len(x[:,0]))]
        return x


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

        self.discretized_cal_X = self._discretize(copy.deepcopy(self.cal_X))

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



    def _set_latest_explanation(self, explanation) -> None:
        # """assigns the latest explanation to the explainer

        # Args:
        #     explanation (CalibratedExplanations): the latest created explanation
        # """
        self.latest_explanation = explanation



    def _is_lime_enabled(self, is_enabled=None) -> bool:
        # """returns whether lime export is enabled.
        # If is_enabled is not None, then the lime export is enabled/disabled according to the value of is_enabled.

        # Args:
        #     is_enabled (bool, optional): is used to assign whether lime export is enabled or not. Defaults to None.

        # Returns:
        #     bool: returns whether lime export is enabled
        # """
        if is_enabled is not None:
            self.__lime_enabled = is_enabled
        return self.__lime_enabled



    def _is_shap_enabled(self, is_enabled=None) -> bool:
        # """returns whether shap export is enabled.
        # If is_enabled is not None, then the shap export is enabled/disabled according to the value of is_enabled.

        # Args:
        #     is_enabled (bool, optional): is used to assign whether shap export is enabled or not. Defaults to None.

        # Returns:
        #     bool: returns whether shap export is enabled
        # """
        if is_enabled is not None:
            self.__shap_enabled = is_enabled
        return self.__shap_enabled



    def _preload_lime(self):
        # """creates a lime structure for the explainer

        # Returns:
        #     LimeTabularExplainer: a LimeTabularExplainer object defined for the problem
        #     lime_exp: a template lime explanation achieved through the explain_instance method
        # """
        if not self._is_lime_enabled():
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
            self._is_lime_enabled(True)
        return self.lime, self.lime_exp



    def _preload_shap(self, num_test=None):
        # """creates a shap structure for the explainer

        # Returns:
        #     shap.Explainer: a Explainer object defined for the problem
        #     shap_exp: a template shap explanation achieved through the __call__ method
        # """
        # pylint: disable=access-member-before-definition
        shap = safe_import("shap")
        if shap:
            if not self._is_shap_enabled() or \
                num_test is not None and self.shap_exp.shape[0] != num_test:
                f = lambda x: self._predict(x)[0]  # pylint: disable=unnecessary-lambda-assignment
                self.shap = shap.Explainer(f, self.cal_X[:1, :], feature_names=self.feature_names)
                self.shap_exp = self.shap(self.cal_X[0, :].reshape(1,-1)) \
                                        if num_test is None else self.shap(self.cal_X[:num_test, :])
                self._is_shap_enabled(True)
            return self.shap, self.shap_exp
        return None, None

class WrapCalibratedExplainer():
    """A wrapper class for the CalibratedExplainer. It allows to fit, calibrate, and explain the model.
    """
    def __init__(self, learner):
        self.learner = learner
        self.explainer = None
        self.calibrated = False
        self.fitted = False

    def __repr__(self):
        if self.fitted:
            if self.calibrated:
                return (f"WrapCalibratedExplainer(learner={self.learner}, fitted=True, "
                    f"calibrated=True, explainer={self.explainer})")
            return f"WrapCalibratedExplainer(learner={self.learner}, fitted=True, calibrated=False)"
        return f"WrapCalibratedExplainer(learner={self.learner}, fitted=False, calibrated=False)"

    def fit(self, X_proper_train, y_proper_train, **kwargs):
        '''
        Fits the learner to the proper training data.
        
        Parameters
        ----------
        X_proper_train : A set of proper training objects to fit the learner to
        y_proper_train : The true labels of the proper training objects
        **kwargs : Keyword arguments to be passed to the learner's fit method
        
        Returns
        -------
        WrapCalibratedExplainer : The WrapCalibratedExplainer object with the fitted learner.
        '''
        self.learner.fit(X_proper_train, y_proper_train, **kwargs)
        self.fitted = True
        return self

    def calibrate(self, X_calibration, y_calibration, **kwargs):
        '''
        Calibrates the learner to the calibration data.
        
        Parameters
        ----------
        X_calibration : A set of calibration objects to predict
        y_calibration : The true labels of the calibration objects
        **kwargs : Keyword arguments to be passed to the CalibratedExplainer's __init__ method
        
        Raises
        ------
        RuntimeError: If the learner is not fitted before calibration.
        
        Returns
        -------
        WrapCalibratedExplainer : The WrapCalibratedExplainer object with the calibrated explainer.
        '''
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before calibration.")
        if 'predict_proba' in dir(self.learner):
            self.explainer = CalibratedExplainer(self.learner, X_calibration, y_calibration, mode='classification', **kwargs)
        else:
            self.explainer = CalibratedExplainer(self.learner, X_calibration, y_calibration, mode='regression', **kwargs)
        self.calibrated = True
        return self

    def explain_factual(self, X_test, **kwargs):
        """
        Creates a CalibratedExplanations object for the test data with the discretizer automatically assigned for factual explanations.

        Parameters
        ----------
        testX : A set of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of 
            instances in testX.

        Returns
        -------
        CalibratedExplanations : A CalibratedExplanations object containing the predictions and the 
            intervals. 
        """
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before explaining.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before explaining.")
        return self.explainer.explain_factual(X_test, **kwargs)

    def explain_counterfactual(self, X_test, **kwargs):
        """
        Creates a CalibratedExplanations object for the test data with the discretizer automatically assigned for counterfactual explanations.

        Parameters
        ----------
        testX : A set of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of 
            instances in testX.

        Returns
        -------
        CalibratedExplanations : A CalibratedExplanations object containing the predictions and the 
            intervals. 
        """
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before explaining.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before explaining.")
        return self.explainer.explain_counterfactual(X_test, **kwargs)

    # pylint: disable=too-many-return-statements
    def predict(self, X_test, uq_interval=False, **kwargs):
        """
        A predict function that outputs a calibrated prediction. If the explainer is not calibrated, then the
        prediction is not calibrated either.

        Parameters
        ----------
        X_test : A set of test objects to predict
        uq_interval : bool, default=False
            If true, then the prediction interval is returned as well. 
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to standard regression.

        Raises
        ------
        RuntimeError: If the learner is not fitted before predicting.

        Returns
        -------
        calibrated prediction : 
            If regression, then the calibrated prediction is the median of the conformal predictive system.
            If classification, then the calibrated prediction is the class with the highest calibrated probability.
        (low, high) : tuple of floats, corresponding to the lower and upper bound of the prediction interval.
        """
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before predicting.")
        if not self.calibrated:
            warnings.warn("The WrapCalibratedExplainer must be calibrated to get calibrated predictions.", Warning)
            return self.learner.predict(X_test)
        if self.explainer.mode in 'regression':
            predict, low, high, _ = self.explainer._predict(X_test, **kwargs) # pylint: disable=protected-access
            if 'threshold' in kwargs.keys(): # pylint: disable=consider-iterating-dictionary
                threshold = kwargs['threshold']
                if np.isscalar(threshold):
                    new_classes = [f'y_hat <= {threshold}' if predict[i] >= 0.5 else f'y_hat > {threshold}' for i in range(len(predict))]
                else:
                    new_classes = [f'y_hat <= {threshold[i]}' if predict[i] >= 0.5 else f'y_hat > {threshold[i]}' for i in range(len(predict))]
                if uq_interval:
                    return new_classes, (low, high)
                return new_classes
            if uq_interval:
                return predict, (low, high)
            return predict
        _, low, high, new_classes = self.explainer._predict(X_test, **kwargs) # pylint: disable=protected-access
        if uq_interval:
            return new_classes, (low, high)
        return new_classes

    def predict_proba(self, X_test, uq_interval=False, threshold=None):
        """
        A predict_proba function that outputs a calibrated prediction. If the explainer is not calibrated, then the
        prediction is not calibrated either.
        
        Parameters
        ----------
        X_test : A set of test objects to predict
        uq_interval : bool, default=False
            If true, then the prediction interval is returned as well.
        threshold : float, int or array-like of shape (n_samples,), default=None
            Threshold values used with regression to get probability of being below the threshold. Only applicable to regression.

        Raises
        ------
        RuntimeError: If the learner is not fitted before predicting.

        Returns
        -------
        calibrated probability : 
            The calibrated probability of the positive class (or the predicted class for multiclass).
        (low, high) : tuple of floats, corresponding to the lower and upper bound of the prediction interval.
        """
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before predicting probabilities.")
        if 'predict_proba' not in dir(self.learner):
            if threshold is None:
                raise ValueError("The threshold parameter must be specified for regression.")
            if not self.calibrated:
                raise RuntimeError("The WrapCalibratedExplainer must be calibrated to get calibrated probabilities for regression.")
        if not self.calibrated:
            warnings.warn("The WrapCalibratedExplainer must be calibrated to get calibrated probabilities.", Warning)
            return self.learner.predict_proba(X_test)
        predict, low, high, _ = self.explainer._predict(X_test, threshold=threshold) # pylint: disable=protected-access
        proba = np.zeros((predict.shape[0],2))
        proba[:,1] = predict
        proba[:,0] = 1 - predict
        if uq_interval:
            return proba, (low, high)
        return proba
