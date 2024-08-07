"""Calibrated Explanations for Black-Box Predictions (calibrated-explanations)

The calibrated explanations explanation method is based on the paper 
"Calibrated Explanations: with Uncertainty Information and Counterfactuals" 
by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

Calibrated explanations are a way to explain the predictions of a black-box model 
using Venn-Abers predictors (classification & regression) or 
conformal predictive systems (regression).
"""
# pylint: disable=invalid-name, line-too-long, too-many-lines
# flake8: noqa: E501
import copy
import math
import warnings
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# from mpl_toolkits.axes_grid1 import make_axes_locatable

from crepes import ConformalClassifier
from crepes.extras import hinge

from ._explanations import CalibratedExplanations
from .VennAbers import VennAbers
from ._interval_regressor import IntervalRegressor
from .utils.discretizers import BinaryEntropyDiscretizer, EntropyDiscretizer, \
                RegressorDiscretizer, BinaryRegressorDiscretizer
from .utils.helper import safe_isinstance, safe_import, check_is_fitted
from .utils.perturbation import perturb_dataset

__version__ = 'v0.3.5'



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
                bins = None,
                difficulty_estimator = None,
                sample_percentiles = [25, 50, 75],
                random_state = 42,
                verbose = False,
                perturb = False,
                reject=False,
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
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
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
        perturb : bool, default=False
            A boolean parameter that determines whether the explainer should perturb the calibration set to 
            enable perturbed explanations.
        reject : bool, default=False
            A boolean parameter that determines whether the explainer should reject explanations that are
            deemed too difficult to explain. If set to True, the explainer will reject explanations that are
            deemed too difficult to explain. If set to False, the explainer will not reject any explanations.
        
        Return
        ------
        CalibratedExplainer : A CalibratedExplainer object that can be used to explain predictions from a predictive model.
        
        '''
        init_time = time()
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
        self.bins = bins

        self.__perturb = perturb

        self.categorical_labels = categorical_labels
        self.class_labels = class_labels
        if categorical_features is None:
            if categorical_labels is not None:
                categorical_features = categorical_labels.keys()
            else:
                categorical_features = []
        self.categorical_features = list(categorical_features)
        self.features_to_ignore = []
        self._preprocess()

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
        self.reject = reject

        self.set_difficulty_estimator(difficulty_estimator, initialize=False)
        self.__set_mode(str.lower(mode), initialize=False)

        self.__initialize_interval_model()
        self.reject_model = self.initialize_reject_model() if reject else None

        self.init_time = time() - init_time

    def reinitialize(self, model):
        """
        Reinitializes the explainer with a new model. This is useful when the model is updated or retrained and the
        explainer needs to be reinitialized.
        
        Parameters
        ----------
        model : predictive model
            A predictive model that can be used to predict the target variable. The model must be fitted and have a predict_proba method (for classification) or a predict method (for regression).
        
        Return
        ------
        CalibratedExplainer : A CalibratedExplainer object that can be used to explain predictions from a predictive model.
        
        """
        self.__initialized = False
        check_is_fitted(model)
        self.model = model
        self.__initialize_interval_model()
        self.__initialized = True


    def __repr__(self):
        # pylint: disable=line-too-long
        disp_str = f"CalibratedExplainer(mode={self.mode}{', conditional=True' if self.bins is not None else ''}{f', discretizer={self.discretizer}' if self.discretizer is not None else ''}, model={self.model}{f', difficulty_estimator={self.difficulty_estimator})' if self.mode == 'regression' else ')'}"
        if self.verbose:
            disp_str += f"\n\tinit_time={self.init_time}"
            if self.latest_explanation is not None:
                disp_str += f"\n\ttotal_explain_time={self.latest_explanation.total_explain_time}"
            disp_str += f"\n\tsample_percentiles={self.sample_percentiles}\
                        \n\trandom_state={self.random_state}\
                        \n\tverbose={self.verbose}"
            if self.feature_names is not None:
                disp_str += f"\n\tfeature_names={self.feature_names}"
            if self.categorical_features is not None:
                disp_str += f"\n\tcategorical_features={self.categorical_features}"
            if self.categorical_labels is not None:
                disp_str += f"\n\tcategorical_labels={self.categorical_labels}"
            if self.class_labels is not None:
                disp_str += f"\n\tclass_labels={self.class_labels}"
        return disp_str


    # pylint: disable=invalid-name, too-many-return-statements
    def _predict(self,
                test_X,
                threshold = None, # The same meaning as threshold has for cps in crepes.
                low_high_percentiles = (5, 95),
                classes = None,
                bins = None,
                feature = None,
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
        # bins : array-like of shape (n_samples,), default=None
        #     Mondrian categories
        # """
        assert self.__initialized, "The model must be initialized before calling predict."
        if feature is None and self.is_perturbed():
            feature = self.num_features # Use the calibrator defined using cal_X
        if self.mode == 'classification':
            if self.is_multiclass():
                if self.is_perturbed():
                    predict, low, high, new_classes = self.interval_model[feature].predict_proba(test_X,
                                                                                    output_interval=True,
                                                                                    classes=classes,
                                                                                    bins=bins)
                else:
                    predict, low, high, new_classes = self.interval_model.predict_proba(test_X,
                                                                                    output_interval=True,
                                                                                    classes=classes,
                                                                                    bins=bins)
                if classes is None:
                    return [predict[i,c] for i,c in enumerate(new_classes)], [low[i,c] for i,c in enumerate(new_classes)], [high[i,c] for i,c in enumerate(new_classes)], new_classes
                if type(classes) not in (list, np.ndarray):
                    classes = [classes]
                return [predict[i,c] for i,c in enumerate(classes)], low, high, None

            if self.is_perturbed():
                predict, low, high = self.interval_model[feature].predict_proba(test_X, output_interval=True, bins=bins)
            else:
                predict, low, high = self.interval_model.predict_proba(test_X, output_interval=True, bins=bins)
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

                if self.is_perturbed():
                    return self.interval_model[feature].predict_uncertainty(test_X, low_high_percentiles, bins=bins)
                return self.interval_model.predict_uncertainty(test_X, low_high_percentiles, bins=bins)

            # regression with threshold condition
            if not np.isscalar(threshold) and len(threshold) != len(test_X):
                raise ValueError("The length of the threshold parameter must be either a scalar or \
                    the same as the number of instances in testX.")
            if self.is_perturbed():
                return self.interval_model[feature].predict_probability(test_X, threshold, bins=bins)
            # pylint: disable=unexpected-keyword-arg
            return self.interval_model.predict_probability(test_X, threshold, bins=bins)

        return None, None, None, None # Should never happen

    def explain_factual(self,
                        test_X,
                        threshold = None,
                        low_high_percentiles = (5, 95),
                        bins = None,
                        ) -> CalibratedExplanations:
        """
        Creates a CalibratedExplanations object for the test data with the discretizer automatically assigned for factual explanations.

        Parameters
        ----------
        testX : A set with n_samples of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories

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
            discretizer = 'binaryRegressor'
        else:
            discretizer = 'binaryEntropy'
        self.set_discretizer(discretizer)
        return self(test_X, threshold, low_high_percentiles, bins)

    def explain_counterfactual(self,
                                test_X,
                                threshold = None,
                                low_high_percentiles = (5, 95),
                                bins = None,
                                ) -> CalibratedExplanations:
        """
        Creates a CalibratedExplanations object for the test data with the discretizer automatically assigned for counterfactual explanations.

        Parameters
        ----------
        testX : A set with n_samples of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories

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
            discretizer = 'regressor'
        else:
            discretizer = 'entropy'
        self.set_discretizer(discretizer)
        return self(test_X, threshold, low_high_percentiles, bins)

    def __call__(self,
                testX,
                threshold = None,
                low_high_percentiles = (5, 95),
                bins = None,
                ) -> CalibratedExplanations:
        """
        Calling self as a function creates a CalibratedExplanations object for the test data with the 
        already assigned discretizer. Called by the `explain_factual` and `explain_counterfactual` methods. 
        See their documentation for further information.
        """
        total_time = time()
        instance_time = []
        if safe_isinstance(testX, "pandas.core.frame.DataFrame"):
            testX = testX.values  # pylint: disable=invalid-name
        if len(testX.shape) == 1:
            testX = testX.reshape(1, -1)
        if testX.shape[1] != self.cal_X.shape[1]:
            raise ValueError("The number of features in the test data must be the same as in the \
                            calibration data.")
        if self._is_mondrian():
            assert bins is not None, "The bins parameter must be specified for Mondrian explanations."
            assert len(bins) == len(testX), "The length of the bins parameter must be the same as the number of instances in testX."
        explanation = CalibratedExplanations(self, testX, threshold, bins)

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
        binned_predict =  {'predict': [],'low': [],'high': [],'current_bin': [],'rule_values': [], 'counts': [], 'fractions': []}

        for i, x in enumerate(testX):
            instance_time.append(time())

            bin_x = [bins[i]] if bins is not None else None

            if threshold is not None and not np.isscalar(explanation.y_threshold):
                threshold = float(explanation.y_threshold[i])
            predict, low, high, predicted_class = self._predict(x.reshape(1,-1), threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bin_x)
            prediction['predict'].append(predict[0])
            prediction['low'].append(low[0])
            prediction['high'].append(high[0])
            if self.is_multiclass():
                prediction['classes'].append(predicted_class[0])
            else:
                prediction['classes'].append(1)

            rule_values = {}
            instance_weights = {'predict':np.zeros(x.shape[0]),'low':np.zeros(x.shape[0]),'high':np.zeros(x.shape[0])}
            instance_predict = {'predict':np.zeros(x.shape[0]),'low':np.zeros(x.shape[0]),'high':np.zeros(x.shape[0])}
            instance_binned = {'predict': [],'low': [],'high': [],'current_bin': [],'rule_values': [], 'counts': [], 'fractions': []}
            # Get the perturbations
            x_original = copy.deepcopy(x)
            perturbed_original = self._discretize(copy.deepcopy(x).reshape(1,-1))
            rule_boundaries = self.rule_boundaries(x_original, perturbed_original)
            for f in range(x.shape[0]): # For each feature
                if f in self.features_to_ignore:
                    continue
                perturbed = copy.deepcopy(x)

                current_bin = -1
                if f in self.categorical_features:
                    values = self.feature_values[f]
                    rule_value = values
                    average_predict, low_predict, high_predict, counts = np.zeros(len(values)),np.zeros(len(values)),np.zeros(len(values)),np.zeros(len(values))
                    for bin_value, value in enumerate(values):  # For each bin (i.e. discretized value) in the values array...
                        perturbed[f] = perturbed_original[0,f] # Assign the original discretized value to ensure similarity to value
                        if perturbed[f] == value:
                            current_bin = bin_value  # If the discretized value is the same as the original, skip it

                        perturbed[f] = value
                        predict, low, high, _ = self._predict(perturbed.reshape(1,-1), threshold=threshold, low_high_percentiles=low_high_percentiles, classes=predicted_class, bins=bin_x)
                        average_predict[bin_value] = predict[0]
                        low_predict[bin_value] = low[0]
                        high_predict[bin_value] = high[0]
                        counts[bin_value] = len(np.where(cal_X[:,f] == value)[0])
                else:
                    rule_value = []
                    values = np.array(cal_X[:,f])
                    lesser = rule_boundaries[f][0]
                    greater = rule_boundaries[f][1]
                    lesser = -np.inf if not np.any(values < lesser) else lesser
                    greater = np.inf if not np.any(values > greater) else greater
                    num_bins = 1
                    num_bins += 1 if lesser != -np.inf else 0
                    num_bins += 1 if greater != np.inf else 0
                    average_predict, low_predict, high_predict, counts = np.zeros(num_bins),np.zeros(num_bins),np.zeros(num_bins),np.zeros(num_bins)

                    bin_value = 0
                    if np.any(values < lesser):
                        lesser_values = np.unique(self.__get_lesser_values(f, lesser))
                        rule_value.append(lesser_values)
                        for value in lesser_values:
                            perturbed[f] = value
                            predict, low, high, _ = self._predict(perturbed.reshape(1,-1), threshold=threshold, low_high_percentiles=low_high_percentiles, classes=predicted_class, bins=bin_x)
                            average_predict[bin_value] += predict[0]
                            low_predict[bin_value] += low[0]
                            high_predict[bin_value] += high[0]
                        average_predict[bin_value] = average_predict[bin_value]/len(lesser_values)
                        low_predict[bin_value] = low_predict[bin_value]/len(lesser_values)
                        high_predict[bin_value] = high_predict[bin_value]/len(lesser_values)
                        counts[bin_value] = len(np.where(cal_X[:,f] < lesser)[0])
                        bin_value += 1
                    if np.any(values > greater):
                        greater_values = np.unique(self.__get_greater_values(f, greater))
                        rule_value.append(greater_values)
                        for value in greater_values:
                            perturbed[f] = value
                            predict, low, high, _ = self._predict(perturbed.reshape(1,-1), threshold=threshold, low_high_percentiles=low_high_percentiles, classes=predicted_class, bins=bin_x)
                            average_predict[bin_value] += predict[0]
                            low_predict[bin_value] += low[0]
                            high_predict[bin_value] += high[0]
                        average_predict[bin_value] = average_predict[bin_value]/len(greater_values)
                        low_predict[bin_value] = low_predict[bin_value]/len(greater_values)
                        high_predict[bin_value] = high_predict[bin_value]/len(greater_values)
                        counts[bin_value] = len(np.where(cal_X[:,f] > greater)[0])
                        bin_value += 1

                    covered_values = self.__get_covered_values(f, lesser, greater)
                    rule_value.append(covered_values)
                    for value in covered_values:
                        perturbed[f] = value
                        predict, low, high, _ = self._predict(perturbed.reshape(1,-1), threshold=threshold, low_high_percentiles=low_high_percentiles, classes=predicted_class, bins=bin_x)
                        average_predict[bin_value] += predict[0]
                        low_predict[bin_value] += low[0]
                        high_predict[bin_value] += high[0]
                    average_predict[bin_value] = average_predict[bin_value]/len(covered_values)
                    low_predict[bin_value] = low_predict[bin_value]/len(covered_values)
                    high_predict[bin_value] = high_predict[bin_value]/len(covered_values)
                    counts[bin_value] = len(np.where((cal_X[:,f] >= lesser) & (cal_X[:,f] <= greater))[0])
                    current_bin = bin_value

                rule_values[f] = (rule_value, x_original[f], perturbed_original[0,f])
                uncovered = np.setdiff1d(np.arange(len(average_predict)), current_bin)

                fractions = counts[uncovered]/np.sum(counts[uncovered])

                instance_binned['predict'].append(average_predict)
                instance_binned['low'].append(low_predict)
                instance_binned['high'].append(high_predict)
                instance_binned['current_bin'].append(current_bin)
                instance_binned['counts'].append(counts)
                instance_binned['fractions'].append(fractions)

                # Handle the situation where the current bin is the only bin
                if len(uncovered) == 0:
                    instance_predict['predict'][f] = 0
                    instance_predict['low'][f] = 0
                    instance_predict['high'][f] = 0

                    instance_weights['predict'][f] = 0
                    instance_weights['low'][f] = 0
                    instance_weights['high'][f] = 0
                else:
                    # Calculate the weighted average (only makes a difference for categorical features)
                    # instance_predict['predict'][f] = np.sum(average_predict[uncovered]*fractions[uncovered])
                    # instance_predict['low'][f] = np.sum(low_predict[uncovered]*fractions[uncovered])
                    # instance_predict['high'][f] = np.sum(high_predict[uncovered]*fractions[uncovered])
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
            binned_predict['counts'].append(instance_binned['counts'])
            binned_predict['fractions'].append(instance_binned['fractions'])

            feature_weights['predict'].append(instance_weights['predict'])
            feature_weights['low'].append(instance_weights['low'])
            feature_weights['high'].append(instance_weights['high'])

            feature_predict['predict'].append(instance_predict['predict'])
            feature_predict['low'].append(instance_predict['low'])
            feature_predict['high'].append(instance_predict['high'])
            instance_time[-1] = time() - instance_time[-1]

        explanation._finalize(binned_predict, feature_weights, feature_predict, prediction, instance_time=instance_time, total_time=total_time)
        self.latest_explanation = explanation
        return explanation


    def explain_perturbed(self,
                                test_X,
                                threshold = None,
                                low_high_percentiles = (5, 95),
                                bins = None,
                                ) -> CalibratedExplanations:
        """
        Creates a CalibratedExplanations object for the test data.

        Parameters
        ----------
        testX : A set with n_samples of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression. 
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of 
            instances in testX.
        RuntimeError: Perturbed explanations are only possible if the explainer is perturbed.

        Returns
        -------
        CalibratedExplanations : A CalibratedExplanations object containing the predictions and the 
            intervals. 
        """
        if not self.is_perturbed:
            raise RuntimeError("Perturbed explanations are only possible if the explainer is perturbed.")
        total_time = time()
        instance_time = []
        if safe_isinstance(test_X, "pandas.core.frame.DataFrame"):
            test_X = test_X.values  # pylint: disable=invalid-name
        if len(test_X.shape) == 1:
            test_X = test_X.reshape(1, -1)
        if test_X.shape[1] != self.cal_X.shape[1]:
            raise ValueError("The number of features in the test data must be the same as in the \
                            calibration data.")
        if self._is_mondrian():
            assert bins is not None, "The bins parameter must be specified for Mondrian explanations."
            assert len(bins) == len(test_X), "The length of the bins parameter must be the same as the number of instances in testX."
        explanation = CalibratedExplanations(self, test_X, threshold, bins)

        is_probabilistic = True # classification or when threshold is used for regression
        if threshold is not None:
            if not 'regression' in self.mode:
                raise Warning("The threshold parameter is only supported for mode='regression'.")
            if not np.isscalar(threshold) and len(threshold) != len(test_X):
                raise ValueError("The length of the threshold parameter must be either a constant or the same \
                                as the number of instances in testX.")
            # explanation.low_high_percentiles = low_high_percentiles
        elif 'regression' in self.mode:
            explanation.low_high_percentiles = low_high_percentiles
            is_probabilistic = False

        feature_weights =  {'predict': [],'low': [],'high': [],}
        feature_predict =  {'predict': [],'low': [],'high': [],}
        prediction =  {'predict': [],'low': [],'high': [], 'classes': []}

        instance_weights = [{'predict':np.zeros(self.num_features),'low':np.zeros(self.num_features),'high':np.zeros(self.num_features)} for _ in range(len(test_X))]
        instance_predict = [{'predict':np.zeros(self.num_features),'low':np.zeros(self.num_features),'high':np.zeros(self.num_features)} for _ in range(len(test_X))]

        feature_time = time()

        predict, low, high, predicted_class = self._predict(test_X, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins)
        prediction['predict'] = predict
        prediction['low'] = low
        prediction['high'] = high
        if self.is_multiclass():
            prediction['classes'] = predicted_class
        else:
            prediction['classes'] = np.ones(test_X.shape[0])
        cal_y = self.cal_y
        self.cal_y = self.scaled_cal_y
        for f in range(self.num_features):
            if f in self.features_to_ignore:
                continue

            predict, low, high, predicted_class = self._predict(test_X, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins, feature=f)

            for i in range(len(test_X)):
                instance_weights[i]['predict'][f] = self._assign_weight(predict[i], prediction['predict'][i], is_probabilistic)
                tmp_low = self._assign_weight(low[i], prediction['predict'][i], is_probabilistic)
                tmp_high = self._assign_weight(high[i], prediction['predict'][i], is_probabilistic)
                instance_weights[i]['low'][f] = np.min([tmp_low, tmp_high])
                instance_weights[i]['high'][f] = np.max([tmp_low, tmp_high])

                instance_predict[i]['predict'][f] = predict[i]
                instance_predict[i]['low'][f] = low[i]
                instance_predict[i]['high'][f] = high[i]
        self.cal_y = cal_y

        for i in range(len(test_X)):
            feature_weights['predict'].append(instance_weights[i]['predict'])
            feature_weights['low'].append(instance_weights[i]['low'])
            feature_weights['high'].append(instance_weights[i]['high'])

            feature_predict['predict'].append(instance_predict[i]['predict'])
            feature_predict['low'].append(instance_predict[i]['low'])
            feature_predict['high'].append(instance_predict[i]['high'])
        feature_time = time() - feature_time
        instance_time = [feature_time / test_X.shape[0]]*test_X.shape[0]


        explanation.finalize_perturbed(feature_weights, feature_predict, prediction, instance_time=instance_time, total_time=total_time)
        self.latest_explanation = explanation
        return explanation


    def _assign_weight(self, instance_predict, prediction, is_probabilistic):
        if is_probabilistic:
            return prediction - instance_predict if np.isscalar(prediction) \
                else [prediction[i]-ip for i,ip in enumerate(instance_predict)] # probabilistic regression
        return prediction - instance_predict  if np.isscalar(prediction) \
            else [prediction[i]-ip for i,ip in enumerate(instance_predict)] # standard regression



    def is_multiclass(self):
        """test if it is a multiclass problem

        Returns:
            bool: True if multiclass
        """
        return self.num_classes > 2


    def is_perturbed(self):
        """test if the explainer is perturbed

        Returns:
            bool: True if perturbed
        """
        return self.__perturb


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
            if f not in self.discretizer.to_discretize:
                min_max.append([instance[f], instance[f]])
            else:
                bins = np.concatenate(([-np.inf], self.discretizer.mins[f][1:], [np.inf]))
                min_max.append([self.discretizer.mins[f][np.digitize(perturbed_instance[0,f], bins, right=True)-1], \
                                self.discretizer.maxs[f][np.digitize(perturbed_instance[0,f], bins, right=True)-1]])
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
        np.random.seed = self.random_state



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
        if self.is_perturbed():
            self.interval_model = []
            cal_X, cal_y, bins = self.cal_X, self.cal_y, self.bins
            self.perturbed_cal_X, self.scaled_cal_X, self.scaled_cal_y, scale_factor = \
                perturb_dataset(self.cal_X, self.cal_y, self.categorical_features, noise_type='uniform', scale_factor=5, severity=1)
            self.bins = np.tile(self.bins.copy(), scale_factor) if self.bins is not None else None
            for f in range(self.num_features):
                perturbed_cal_X = self.scaled_cal_X.copy()
                perturbed_cal_X[:,f] = self.perturbed_cal_X[:,f]
                if self.mode == 'classification':
                    self.interval_model.append(VennAbers(self.model.predict_proba(perturbed_cal_X), self.scaled_cal_y, self.model, self.bins))
                elif 'regression' in self.mode:
                    self.cal_X = perturbed_cal_X
                    self.cal_y = self.scaled_cal_y
                    self.interval_model.append(IntervalRegressor(self))

            self.cal_X, self.cal_y, self.bins = cal_X, cal_y, bins
            if self.mode == 'classification':
                self.interval_model.append(VennAbers(self.model.predict_proba(self.cal_X), self.cal_y, self.model, self.bins))
            elif 'regression' in self.mode:
                # Add a reference model using the original calibration data last
                self.interval_model.append(IntervalRegressor(self))
        else:
            if self.mode == 'classification':
                self.interval_model = VennAbers(self.model.predict_proba(self.cal_X), self.cal_y, self.model, self.bins)
            elif 'regression' in self.mode:
                self.interval_model = IntervalRegressor(self)
        self.__initialized = True

    def initialize_reject_model(self, calibration_set=None, threshold=None):
        '''
        Initializes the reject model for the explainer. The reject model is a ConformalClassifier
        that is trained on the calibration data. The reject model is used to determine whether a test
        instance is within the calibration data distribution. The reject model is only available for
        classification, unless a threshold is assigned.
        
        Parameters
        ----------
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression.
        '''
        if calibration_set is not None:
            if calibration_set is tuple:
                cal_X, cal_y = calibration_set
            else:
                cal_X, cal_y = calibration_set[0], calibration_set[1]
        else:
            cal_X, cal_y = self.cal_X, self.cal_y
        self.reject_threshold = None
        if self.mode in 'regression':
            proba_1, _, _, _ = self.interval_model.predict_probability(cal_X, y_threshold=threshold, bins=self.bins)
            proba = np.array([[1-proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            classes = (cal_y < threshold).astype(int)
            self.reject_threshold = threshold
        elif self.is_multiclass(): # pylint: disable=protected-access
            proba, classes = self.interval_model.predict_proba(cal_X, bins=self.bins)
            proba = np.array([[1-proba[i,c], proba[i,c]] for i,c in enumerate(classes)])
            classes = (classes == cal_y).astype(int)
        else:
            proba = self.interval_model.predict_proba(cal_X, bins=self.bins)
            classes = cal_y
        alphas_cal = hinge(proba, np.unique(classes), classes)
        self.reject_model = ConformalClassifier().fit(alphas=alphas_cal, bins=classes)
        return self.reject_model

    def predict_reject(self, test_X, bins=None, confidence=0.95):
        '''
        Predicts whether a test instance is within the calibration data distribution.

        Parameters
        ----------
        test_X : A set with n_samples of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression.

        Returns
        -------
        np.ndarray : A boolean array of shape (n_samples,) indicating whether the test instances are within the calibration data distribution.
        '''
        if self.mode in 'regression':
            assert self.reject_threshold is not None, "The reject model is only available for regression with a threshold."
            proba_1, _, _, _ = self.interval_model.predict_probability(test_X, y_threshold=self.reject_threshold, bins=bins)
            proba = np.array([[1-proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            classes = [0,1]
        elif self.is_multiclass(): # pylint: disable=protected-access
            proba, classes = self.interval_model.predict_proba(test_X, bins=bins)
            proba = np.array([[1-proba[i,c], proba[i,c]] for i,c in enumerate(classes)])
            classes = [0,1]
        else:
            proba = self.interval_model.predict_proba(test_X, bins=bins)
            classes = np.unique(self.cal_y)
        alphas_test = hinge(proba)

        prediction_set = np.array([
                self.reject_model.predict_set(alphas_test,
                                            np.full(len(alphas_test), classes[c]),
                                            confidence=confidence)[:, c]
                for c in range(len(classes))
            ]).T
        singelton = np.sum(np.sum(prediction_set, axis=1) == 1)
        empty = np.sum(np.sum(prediction_set, axis=1) == 0)
        n = len(test_X)

        epsilon = 1 - confidence
        error_rate = (n*epsilon - empty) / singelton
        reject_rate = 1 - singelton/n

        rejected = np.sum(prediction_set, axis=1) != 1
        return rejected, error_rate, reject_rate


    def _preprocess(self):
        # preprocesses the calibration data by identifying constant value columns to ignore
        constant_columns = [np.where(np.all(self.cal_X[:,f] == self.cal_X[0,f], axis=0) for f in range(self.cal_X.shape[1]))]
        self.features_to_ignore = constant_columns


    def _discretize(self, x):
        # """applies the discretizer to the test instance x

        # Args:
        #     x (n_features,): the test instance to discretize

        # Returns:
        #     (n_features,): a perturbed test instance
        # """
        if len(np.shape(x)) == 1:
            x = np.array(x)
        for f in self.discretizer.to_discretize:
            bins = np.concatenate(([-np.inf], self.discretizer.mins[f][1:], [np.inf]))
            x[:,f] = [self.discretizer.means[f][np.digitize(x[i,f], bins, right=True)-1]  for i in range(len(x[:,f]))]
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
                discretizer = 'binaryRegressor'
            else:
                discretizer = 'binaryEntropy'
        else:
            if 'regression'in self.mode:
                assert discretizer is None or discretizer in ['regressor', 'binaryRegressor'], \
                    "The discretizer must be 'binaryRegressor' (default for factuals) or 'regressor' (default for counterfactuals) for regression."
            else:
                assert discretizer is None or discretizer in ['entropy', 'binaryEntropy'], \
                    "The discretizer must be 'binaryEntropy' (default for factuals) or 'entropy' (default for counterfactuals) for classification."

        not_to_discretize = self.categorical_features #np.union1d(self.categorical_features, self.features_to_ignore)
        if discretizer == 'entropy':
            self.discretizer = EntropyDiscretizer(
                    cal_X, not_to_discretize,
                    self.feature_names, labels=cal_y,
                    random_state=self.random_state)
        elif discretizer == 'binaryEntropy':
            self.discretizer = BinaryEntropyDiscretizer(
                    cal_X, not_to_discretize,
                    self.feature_names, labels=cal_y,
                    random_state=self.random_state)
        elif discretizer == 'regressor':
            self.discretizer = RegressorDiscretizer(
                    cal_X, not_to_discretize,
                    self.feature_names, labels=cal_y,
                    random_state=self.random_state)
        elif discretizer == 'binaryRegressor':
            self.discretizer = BinaryRegressorDiscretizer(
                    cal_X, not_to_discretize,
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


    def _is_mondrian(self):
        # """returns whether the explainer is a Mondrian explainer

        # Returns:
        #     bool: True if Mondrian
        # """
        return self.bins is not None



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
        lime = safe_import("lime.lime_tabular","LimeTabularExplainer")
        if lime:
            if not self._is_lime_enabled():
                if self.mode == 'classification':
                    self.lime = lime(self.cal_X[:1, :],
                                                    feature_names=self.feature_names,
                                                    class_names=['0','1'],
                                                    mode=self.mode)
                    self.lime_exp = self.lime.explain_instance(self.cal_X[0, :],
                                                                self.model.predict_proba,
                                                                num_features=self.num_features)
                elif 'regression' in self.mode:
                    self.lime = lime(self.cal_X[:1, :],
                                                    feature_names=self.feature_names,
                                                    mode='regression')
                    self.lime_exp = self.lime.explain_instance(self.cal_X[0, :],
                                                                self.model.predict,
                                                                num_features=self.num_features)
                self._is_lime_enabled(True)
            return self.lime, self.lime_exp
        return None, None



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
    """Calibrated Explanations for Black-Box Predictions (calibrated-explanations)

    The calibrated explanations explanation method is based on the paper 
    "Calibrated Explanations: with Uncertainty Information and Counterfactuals" 
    by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

    Calibrated explanations are a way to explain the predictions of a black-box model 
    using Venn-Abers predictors (classification & regression) or 
    conformal predictive systems (regression).

    WrapCalibratedExplainer is a wrapper class for the CalibratedExplainer. It allows to fit, calibrate, and explain the model.
    Compared to the CalibratedExplainer, it allow access to the predict and predict_proba methods of
    the calibrated explainer, making it easy to get the same output as shown in the explanations.
    """
    def __init__(self, learner):
        # Check if the learner is a CalibratedExplainer
        if safe_isinstance(learner, "calibrated_explanations.core.CalibratedExplainer"):
            explainer = learner
            learner = explainer.model
            self.calibrated = True
            self.explainer = explainer
            self.learner = learner
            check_is_fitted(self.learner)
            self.fitted = True
            return

        self.learner = learner
        self.explainer = None
        self.calibrated = False

        # Check if the learner is already fitted
        try:
            check_is_fitted(learner)
            self.fitted = True
        except (TypeError, RuntimeError):
            self.fitted = False

    def __repr__(self):
        if self.fitted:
            if self.calibrated:
                return (f"WrapCalibratedExplainer(learner={self.learner}, fitted=True, "
                    f"calibrated=True, \n\t\texplainer={self.explainer})")
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
        reinitialize = bool(self.calibrated)
        self.fitted = False
        self.calibrated = False
        self.learner.fit(X_proper_train, y_proper_train, **kwargs)
        check_is_fitted(self.learner)
        self.fitted = True

        if reinitialize:
            self.explainer.reinitialize(self.learner)
            self.calibrated = True

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
        
        Examples
        --------
        Calibrate the learner to the calibration data:
        >>> calibrate(X_calibration, y_calibration)
        
        Provide additional keyword arguments to the CalibratedExplainer:
        >>> calibrate(X_calibration, y_calibration, feature_names=feature_names, categorical_features=categorical_features)
        
        Note: if mode is not explicitly set, it is automatically determined based on the the absence or presence of a predict_proba method in the learner.
        '''
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before calibration.")
        self.calibrated = False
        if 'mode' in kwargs:
            self.explainer = CalibratedExplainer(self.learner, X_calibration, y_calibration, **kwargs)
        elif 'predict_proba' in dir(self.learner):
            self.explainer = CalibratedExplainer(self.learner, X_calibration, y_calibration, mode='classification', **kwargs)
        else:
            self.explainer = CalibratedExplainer(self.learner, X_calibration, y_calibration, mode='regression', **kwargs)
        self.calibrated = True
        return self

    def explain_factual(self, X_test, **kwargs):
        """
        Generates a CalibratedExplanations object for the provided test data, automatically selecting an appropriate discretizer for factual explanations.

        Parameters
        ----------
        X_test : array-like
            The test data for which predictions and explanations are to be generated. This should be in a format compatible with sklearn (e.g., numpy arrays, pandas DataFrames).

        **kwargs : Various types, optional
            Additional parameters to customize the explanation process. Supported parameters include:

            - threshold (float, int, or array-like of shape (n_samples,), default=None): Specifies the p-value thresholds for probabilistic explanations in regression tasks. This parameter is ignored for classification tasks.

            - low_high_percentiles (tuple of two floats, default=(5, 95)): Defines the lower and upper percentiles for calculating prediction intervals in regression tasks. This is used to adjust the breadth of the intervals based on the distribution of the predictions.

            Additional keyword arguments can be passed to further customize the behavior of the explanation generation. These arguments are dynamically processed based on the specific requirements of the explanation task.

        Raises
        ------
        ValueError
            If the number of features in `X_test` does not match the number of features in the calibration data used to initialize the CalibratedExplanations object.

        Warning
            If the `threshold` parameter is provided for a task other than regression, a warning is issued indicating that this parameter is only applicable to regression tasks.

        ValueError
            If the `threshold` parameter's length does not match the number of instances in `X_test`, or if it is not a single constant value applicable to all instances.

        Returns
        -------
        CalibratedExplanations
            An object containing the generated predictions and their corresponding intervals or explanations. This object provides methods to further analyze and visualize the explanations.

        Examples
        --------
        Generate explanations with a specific threshold for regression:
        
        .. code-block:: python
        
            w.explain_factual(X_test, threshold=0.05)

        Generate explanations using custom percentile values for interval calculation:
        
        .. code-block:: python
        
            w.explain_factual(X_test, low_high_percentiles=(10, 90))
        """
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before explaining.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before explaining.")
        return self.explainer.explain_factual(X_test, **kwargs)

    def explain_counterfactual(self, X_test, **kwargs):
        """
        Generates a CalibratedExplanations object for the provided test data, automatically selecting an appropriate discretizer for counterfactual explanations.

        Parameters
        ----------
        X_test : array-like
            The test data for which predictions and explanations are to be generated. This should be in a format compatible with sklearn (e.g., numpy arrays, pandas DataFrames).

        **kwargs : Various types, optional
            Additional parameters to customize the explanation process. Supported parameters include:

            - threshold (float, int, or array-like of shape (n_samples,), default=None): Specifies the p-value thresholds for probabilistic explanations in regression tasks. This parameter is ignored for classification tasks.

            - low_high_percentiles (tuple of two floats, default=(5, 95)): Defines the lower and upper percentiles for calculating prediction intervals in regression tasks. This is used to adjust the breadth of the intervals based on the distribution of the predictions.

            Additional keyword arguments can be passed to further customize the behavior of the explanation generation. These arguments are dynamically processed based on the specific requirements of the explanation task.

        Raises
        ------
        ValueError
            If the number of features in `X_test` does not match the number of features in the calibration data used to initialize the CalibratedExplanations object.

        Warning
            If the `threshold` parameter is provided for a task other than regression, a warning is issued indicating that this parameter is only applicable to regression tasks.

        ValueError
            If the `threshold` parameter's length does not match the number of instances in `X_test`, or if it is not a single constant value applicable to all instances.

        Returns
        -------
        CalibratedExplanations
            An object containing the generated predictions and their corresponding intervals or explanations. This object provides methods to further analyze and visualize the explanations.

        Examples
        --------
        Generate explanations with a specific threshold for regression:
        
        .. code-block:: python
        
            w.explain_counterfactual(X_test, threshold=0.05)

        Generate explanations using custom percentile values for interval calculation:
        
        .. code-block:: python
        
            w.explain_counterfactual(X_test, low_high_percentiles=(10, 90))
        """
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before explaining.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before explaining.")
        return self.explainer.explain_counterfactual(X_test, **kwargs)

    def explain_perturbed(self, X_test, **kwargs):
        """
        Generates a CalibratedExplanations object for the provided test data, provided that the CalibratedExplainer has been perturbed (using the parameter perturb=True).

        Parameters
        ----------
        X_test : array-like
            The test data for which predictions and explanations are to be generated. This should be in a format compatible with sklearn (e.g., numpy arrays, pandas DataFrames).

        **kwargs : Various types, optional
            Additional parameters to customize the explanation process. Supported parameters include:

            - threshold (float, int, or array-like of shape (n_samples,), default=None): Specifies the p-value thresholds for probabilistic explanations in regression tasks. This parameter is ignored for classification tasks.

            - low_high_percentiles (tuple of two floats, default=(5, 95)): Defines the lower and upper percentiles for calculating prediction intervals in regression tasks. This is used to adjust the breadth of the intervals based on the distribution of the predictions.

            Additional keyword arguments can be passed to further customize the behavior of the explanation generation. These arguments are dynamically processed based on the specific requirements of the explanation task.

        Raises
        ------
        ValueError
            If the number of features in `X_test` does not match the number of features in the calibration data used to initialize the CalibratedExplanations object.

        Warning
            If the `threshold` parameter is provided for a task other than regression, a warning is issued indicating that this parameter is only applicable to regression tasks.

        ValueError
            If the `threshold` parameter's length does not match the number of instances in `X_test`, or if it is not a single constant value applicable to all instances.

        Returns
        -------
        CalibratedExplanations
            An object containing the generated predictions and their corresponding intervals or explanations. This object provides methods to further analyze and visualize the explanations.

        Examples
        --------
        Generate explanations with a specific threshold for regression:
        
        .. code-block:: python
        
            w.explain_perturbed(X_test, threshold=0.05)

        Generate explanations using custom percentile values for interval calculation:
        
        .. code-block:: python
        
            w.explain_perturbed(X_test, low_high_percentiles=(10, 90))
        """
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before explaining.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before explaining.")
        return self.explainer.explain_perturbed(X_test, **kwargs)


    # pylint: disable=too-many-return-statements
    def predict(self, X_test, uq_interval=False, **kwargs):
        """
        Generates a calibrated prediction for the given test data. If the model is not calibrated, the prediction remains uncalibrated.

        Parameters
        ----------
        X_test : array-like
            The test data for which predictions are to be made. This should be in a format compatible with sklearn (e.g., numpy arrays, pandas DataFrames).
        uq_interval : bool, default=False
            If True, returns the uncertainty quantification interval along with the calibrated prediction. 
        **kwargs : Various types, optional
            Additional parameters to customize the explanation process. Supported parameters include:

            - threshold : float, int, or array-like of shape (n_samples,), optional, default=None
                Specifies the threshold(s) to get a thresholded prediction for regression tasks (prediction labels: `y_hat<=threshold-value` | `y_hat>threshold-value`). This parameter is ignored for classification tasks.

            - low_high_percentiles : tuple of two floats, optional, default=(5, 95)
                The lower and upper percentiles used to calculate the prediction interval for regression tasks. Determines the breadth of the interval based on the distribution of the predictions. This parameter is ignored for classification tasks.

        Raises
        ------
        RuntimeError
            If the model has not been fitted prior to making predictions.

        Warning
            If the model is not calibrated.

        Returns
        -------
        calibrated_prediction : float or array-like, or str
            The calibrated prediction. For regression tasks, this is the median of the conformal predictive system or a thresholded prediction if `threshold`is set. For classification tasks, it is the class label with the highest calibrated probability.
        interval : tuple of floats, optional
            A tuple (low, high) representing the lower and upper bounds of the uncertainty interval. This is returned only if `uq_interval=True`.

        Examples
        --------
        For a prediction without prediction intervals:
        
        .. code-block:: python
        
            w.predict(X_test)

        For a prediction with uncertainty quantification intervals:
        
        .. code-block:: python
        
            w.predict(X_test, uq_interval=True)

        Note
        ----
        The `threshold` and `low_high_percentiles` parameters are only used for regression tasks.
        """
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before predicting.")
        if not self.calibrated:
            if 'threshold' in kwargs:
                raise ValueError("A thresholded prediction is not possible for uncalibrated models.")
            warnings.warn("The WrapCalibratedExplainer must be calibrated to get calibrated predictions.", Warning)
            if uq_interval:
                predict = self.learner.predict(X_test)
                return predict, (predict, predict)
            return self.learner.predict(X_test)
        if self.explainer.mode in 'regression':
            predict, low, high, _ = self.explainer._predict(X_test, **kwargs) # pylint: disable=protected-access
            if 'threshold' in kwargs:
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
        predict, low, high, new_classes = self.explainer._predict(X_test, **kwargs) # pylint: disable=protected-access
        if new_classes is None:
            new_classes = (predict >= 0.5).astype(int)
        if uq_interval:
            return new_classes, (low, high)
        return new_classes

    def predict_proba(self, X_test, uq_interval=False, threshold=None):
        """
        A predict_proba function that outputs a calibrated prediction. If the explainer is not calibrated, then the
        prediction is not calibrated either.
        
        Parameters
        ----------
        X_test : array-like
            The test data for which predictions are to be made. This should be in a format compatible with sklearn (e.g., numpy arrays, pandas DataFrames).
        uq_interval : bool, default=False
            If true, then the prediction interval is returned as well.
        threshold : float, int or array-like of shape (n_samples,), optional, default=None
            Threshold values used with regression to get probability of being below the threshold. Only applicable to regression.

        Raises
        ------
        RuntimeError
            If the learner is not fitted before predicting.

        ValueError
            If the `threshold` parameter's length does not match the number of instances in `X_test`, or if it is not a single constant value applicable to all instances.

        RuntimeError
            If the learner is not fitted before predicting.
            
        Warning
            If the model is not calibrated.
            
        Returns
        -------
        calibrated probability : 
            The calibrated probability of the positive class (or the predicted class for multiclass).
        (low, high) : tuple of floats, corresponding to the lower and upper bound of the prediction interval.
        
        Examples
        --------
        For a prediction without uncertainty quantification intervals:
        
        .. code-block:: python
        
            w.predict_proba(X_test)

        For a prediction with uncertainty quantification intervals:
        
        .. code-block:: python
        
            w.predict_proba(X_test, uq_interval=True)

        Note
        ----
        The `threshold` parameter is only used for regression tasks.
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
            if uq_interval:
                proba = self.learner.predict_proba(X_test)
                if proba.shape[1] > 2:
                    return proba, (proba, proba)
                return proba, (proba[:,1], proba[:,1])
            return self.learner.predict_proba(X_test)
        if self.explainer.mode in 'regression':
            proba_1, low, high, _ = self.explainer.interval_model.predict_probability(X_test, y_threshold=threshold)
            proba = np.array([[1-proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            if uq_interval:
                return proba, (low, high)
            return proba
        if self.explainer.is_multiclass(): # pylint: disable=protected-access
            proba, low, high, _ = self.explainer.interval_model.predict_proba(X_test, output_interval=True)
            if uq_interval:
                return proba, (low, high)
            return proba
        proba, low, high = self.explainer.interval_model.predict_proba(X_test, output_interval=True)
        if uq_interval:
            return proba, (low, high)
        return proba

    # pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
    def plot_global(self, X_test, y_test=None, threshold=None, **kwargs):
        """
        Generates a global explanation plot for the given test data. This plot is based on the probability distribution and the uncertainty quantification intervals.
        The plot is only available for calibrated probabilistic models (both classification and thresholded regression).
        
        Parameters
        ----------
        X_test : array-like
            The test data for which predictions are to be made. This should be in a format compatible with sklearn (e.g., numpy arrays, pandas DataFrames).
        y_test : array-like, optional
            The true labels of the test data. 
        threshold : float, int, optional
            The threshold value used with regression to get probability of being below the threshold. Only applicable to regression.
        """
        if not self.fitted:
            raise RuntimeError("The WrapCalibratedExplainer must be fitted before plotting.")
        if not self.calibrated:
            raise RuntimeError("The WrapCalibratedExplainer must be calibrated before plotting.")
        is_regularized = True
        if 'predict_proba' not in dir(self.learner):
            if threshold is None:
                predict, (low, high) = self.predict(X_test, uq_interval=True, **kwargs)
                is_regularized = False
            else:
                proba, (low, high) = self.predict_proba(X_test, uq_interval=True, threshold=threshold)
        else:
            proba, (low, high) = self.predict_proba(X_test, uq_interval=True, threshold=threshold)
        uncertainty = np.array(high - low)

        marker_size = 50
        min_x, min_y = 0,0
        max_x, max_y = 1,1
        ax = None
        if is_regularized:
            plt.figure()
            x = np.arange(0, 1, 0.01)
            plt.plot((x / (1 + x)), x, color='black')
            plt.plot(x, ((1 - x) / x), color='black')
            x = np.arange(0.5, 1, 0.005)
            plt.plot((0.5 + x - 0.5)/(1 + x - 0.5), x - 0.5, color='black')
            x = np.arange(0, 0.5, 0.005)
            plt.plot((x + 0.5 - x)/(1 + x), x, color='black')
        else:
            _, ax = plt.subplots()
            # draw a line from (0,0) to (0.5,1) and from (1,0) to (0.5,1)
            min_x = np.min(self.explainer.cal_y)
            max_x = np.max(self.explainer.cal_y)
            min_y = np.min(uncertainty)
            max_y = np.max(uncertainty)
            if math.isclose(min_x, max_x, rel_tol=1e-9):
                warnings.warn("All uncertainties are (almost) identical.", Warning)

            min_x = min_x - min(0.1 * (max_x - min_x), 0) if min_x > 0 else min_x - 0.1 * (max_x - min_x)
            max_x = max_x + 0.1 * (max_x - min_x)
            # min_y = min_y - max(0.1 * (max_y - min_y), 0) # uncertainty is always positive
            max_y = max_y + 0.1 * (max_y - min_y)
            # mid_x = (min_x + max_x) / 2
            # mid_y = (min_y + max_y) / 2
            # ax.plot([min_x, mid_x], [min_y, max_y], color='black')
            # ax.plot([max_x, mid_x], [min_y, max_y], color='black')
            # # draw a line from (0.5,0) to halfway between (0.5,0) and (0,1)
            # ax.plot([mid_x, mid_x / 2], [min_y, mid_y], color='black')
            # # draw a line from (0.5,0) to halfway between (0.5,0) and (1,1)
            # ax.plot([mid_x, mid_x + mid_x / 2], [min_y, mid_y], color='black')

        if y_test is not None:
            if 'predict_proba' not in dir(self.learner) and threshold is None: # not probabilistic
                norm = mcolors.Normalize(vmin=y_test.min(), vmax=y_test.max())
                # Choose a colormap
                colormap = plt.cm.viridis  # pylint: disable=no-member
                # Map the normalized values to colors
                colors = colormap(norm(y_test))
                ax.scatter(predict, uncertainty, label='Predictions', color=colors, marker='.', s=marker_size)
                # # Create a new axes for the colorbar
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.05)
                # # Add the colorbar to the new axes
                # plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax, label='Target Values')
            else:
                if 'predict_proba' not in dir(self.learner):
                    assert np.isscalar(threshold), "The threshold parameter must be a single constant value for all instances when used in plot_global."
                    y_test = np.array([0 if y_test[i] >= threshold else 1 for i in range(len(y_test))])
                    labels = [f'Y >= {threshold}', f'Y < {threshold}']
                else:
                    if self.explainer.class_labels is not None:
                        labels = [f'Y = {i}' for i in self.explainer.class_labels.values()]
                    else:
                        labels = [f'Y = {i}' for i in np.unique(y_test)]
                marker_size = 25
                if len(labels) == 2:
                    colors = ['blue', 'red']
                    markers = ['o', 'x']
                    proba = proba[:,1]
                else:
                    colormap = plt.get_cmap('tab10', len(labels))
                    colors = [colormap(i) for i in range(len(labels))]
                    markers = ['o', 'x', 's', '^', 'v', 'D', 'P', '*', 'h', 'H','o', 'x', 's', '^', 'v', 'D', 'P', '*', 'h', 'H'][:len(labels)]
                    proba = proba[np.arange(len(proba)), y_test]
                    uncertainty = uncertainty[np.arange(len(uncertainty)), y_test]
                for i, c in enumerate(np.unique(y_test)):
                    plt.scatter(proba[y_test == c], uncertainty[y_test == c], color=colors[i], label=labels[i], marker=markers[i], s=marker_size)
                plt.legend()
        else:
            if 'predict_proba' not in dir(self.learner) and threshold is None: # not probabilistic
                plt.scatter(predict, uncertainty, label='Predictions', marker='.', s=marker_size)
            else:
                if self.explainer.is_multiclass(): # pylint: disable=protected-access
                    predicted = np.argmax(proba, axis=1)
                    proba = proba[np.arange(len(proba)), predicted]
                    uncertainty = uncertainty[np.arange(len(uncertainty)), predicted]
                else:
                    proba = proba[:,1]
                plt.scatter(proba, uncertainty, label='Predictions', marker='.', s=marker_size)

        if 'predict_proba' not in dir(self.learner) and threshold is None: # not probabilistic
            plt.xlabel('Predictions',loc='center')
            plt.ylabel('Uncertainty',loc='center')
        else:
            plt.ylabel('Uncertainty')
            if 'predict_proba' not in dir(self.learner):
                plt.xlabel(f'Probability of Y < {threshold}')
            else:
                if self.explainer.is_multiclass(): # pylint: disable=protected-access
                    if y_test is not None:
                        plt.xlabel('Probability of Y = actual class')
                    else:
                        plt.xlabel('Probability of Y = predicted class')
                else:
                    plt.xlabel('Probability of Y = 1')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.show()
