# pylint: disable=too-many-lines, line-too-long, too-many-public-methods, invalid-name, too-many-positional-arguments
"""contains the :class:`.CalibratedExplanations` class created by :class:`.CalibratedExplainer`
"""

import contextlib
import os
import warnings
from copy import deepcopy
from abc import ABC, abstractmethod
from time import time
import numpy as np
from pandas import Categorical
from .utils.discretizers import BinaryEntropyDiscretizer, EntropyDiscretizer, RegressorDiscretizer, BinaryRegressorDiscretizer
from .utils.helper import make_directory, calculate_metrics
from ._plots import _plot_alternative, _plot_probabilistic, _plot_regression, _plot_triangular

class CalibratedExplanations: # pylint: disable=too-many-instance-attributes
    """
    A class for storing and visualizing calibrated explanations.
    """
    def __init__(self, calibrated_explainer, X_test, y_threshold, bins) -> None:
        self.calibrated_explainer = deepcopy(calibrated_explainer)
        self.X_test = X_test
        self.y_threshold = y_threshold
        self.low_high_percentiles = None
        self.explanations = []
        self.start_index = 0
        self.current_index = self.start_index
        self.end_index = len(X_test[:,0])
        self.bins = bins
        self.total_explain_time = None

    def __iter__(self):
        self.current_index = self.start_index
        return self

    def __next__(self):
        if self.current_index >= self.end_index:
            raise StopIteration
        result = self.get_explanation(self.current_index)
        self.current_index += 1
        return result

    def __len__(self):
        return len(self.X_test[:, 0])

    def __getitem__(self, key):
        '''
        The function `__getitem__` returns the explanation(s) corresponding to the index key. In case the 
        index key is an integer (or results in a single result), the function returns the explanation 
        corresponding to the index. If the key is a slice or an integer or boolean list (or numpy array) 
        resulting in more than one explanation, the function returns a new `CalibratedExplanations` 
        object with the indexed explanations.
        '''
        if isinstance(key, int):
            # Handle single item access
            return self.explanations[key]
        if isinstance(key, (slice, list, np.ndarray)):
            new_ = deepcopy(self)
            if isinstance(key, slice):
                # Handle slicing
                new_.explanations = self.explanations[key]
            if isinstance(key, (list, np.ndarray)):
                if isinstance(key[0], (bool, np.bool_)):
                    # Handle boolean indexing
                    new_.explanations = [exp for exp, include in zip(self.explanations, key) if include]
                elif isinstance(key[0], int):
                    # Handle integer list indexing
                    new_.explanations = [self.explanations[i] for i in key]
            if len(new_.explanations) == 1:
                return new_.explanations[0]
            new_.start_index = 0
            new_.current_index = new_.start_index
            new_.end_index = len(new_.explanations)
            new_.bins = None if self.bins is None else [self.bins[e.index] for e in new_]
            new_.X_test = np.array([self.X_test[e.index,:] for e in new_])
            new_.y_threshold = None if self.y_threshold is None else self.y_threshold \
                        if np.isscalar(self.y_threshold) else [self.y_threshold[e.index] for e in new_]
            for i,e in enumerate(new_):
                e.index = i
            return new_
        raise TypeError("Invalid argument type.")

    def __repr__(self) -> str:
        explanations_str = "\n".join([str(e) for e in self.explanations])
        return f"CalibratedExplanations({len(self)} explanations):\n{explanations_str}"

    def __str__(self) -> str:
        return self.__repr__()

    def _is_thresholded(self) -> bool:
        # """test if the explanation is thresholded

        # Returns:
        #     bool: True if the y_threshold is not None
        # """
        return self.y_threshold is not None



    def _is_one_sided(self) -> bool:
        # """test if a regression explanation is one-sided

        # Returns:
        #     bool: True if one of the low or high percentiles is infinite
        # """
        if self.low_high_percentiles is None:
            return False
        return np.isinf(self.get_low_percentile()) or np.isinf(self.get_high_percentile())



    def get_confidence(self) -> float:
        """get the user assigned confidence of the explanation
        
        Returns: 
            returns the difference between the low and high percentiles
        """
        if np.isinf(self.get_high_percentile()):
            return 100-self.get_low_percentile()
        if np.isinf(self.get_low_percentile()):
            return self.get_high_percentile()
        return self.get_high_percentile() - self.get_low_percentile()



    def get_low_percentile(self) -> float:
        """get the low percentile of the explanation
        """
        return self.low_high_percentiles[0] # pylint: disable=unsubscriptable-object



    def get_high_percentile(self) -> float:
        """get the high percentile of the explanation
        """
        return self.low_high_percentiles[1] # pylint: disable=unsubscriptable-object



    # pylint: disable=too-many-arguments
    def finalize(self, binned, feature_weights, feature_predict, prediction, instance_time=None, total_time=None) -> None:
        """finalize the explanation by adding the binned data and the feature weights
        """
        for i, instance in enumerate(self.X_test):
            instance_bin = self.bins[i] if self.bins is not None else None
            if self._is_alternative():
                explanation = AlternativeExplanation(self, i, instance, binned, feature_weights, feature_predict, prediction, self.y_threshold, instance_bin=instance_bin)
            else:
                explanation = FactualExplanation(self, i, instance, binned, feature_weights, feature_predict, prediction, self.y_threshold, instance_bin=instance_bin)
            explanation.explain_time = instance_time[i] if instance_time is not None else None
            self.explanations.append(explanation)
        self.total_explain_time = time() - total_time if total_time is not None else None
        if self._is_alternative():
            return self.__convert_to_AlternativeExplanations()
        return self

    def __convert_to_AlternativeExplanations(self):
        alternative_explanations = AlternativeExplanations.__new__(AlternativeExplanations)
        alternative_explanations.__dict__.update(self.__dict__)
        return alternative_explanations


    # pylint: disable=too-many-arguments
    def finalize_fast(self, feature_weights, feature_predict, prediction, instance_time=None, total_time=None) -> None:
        """finalize the explanation by adding the binned data and the feature weights
        """
        for i, instance in enumerate(self.X_test):
            instance_bin = self.bins[i] if self.bins is not None else None
            explanation = FastExplanation(self, i, instance, feature_weights, feature_predict, prediction, self.y_threshold, instance_bin=instance_bin)
            explanation.explain_time = instance_time[i] if instance_time is not None else None
            self.explanations.append(explanation)
        self.total_explain_time = time() - total_time if total_time is not None else None



    def _get_explainer(self):
        # """get the explainer object
        # """
        return self.calibrated_explainer



    def _get_rules(self):
        return [explanation._get_rules() for explanation in self.explanations] # pylint: disable=protected-access



    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        """
        Adds conjunctive rules to the factual or alternative explanations. The conjunctive rules are added to the `conjunctive_rules` 
        attribute of the :class:`.CalibratedExplanations` object.

        Args:
            n_top_features (int, optional): the number of most important factual rules to try to combine into conjunctive rules. Defaults to 5.
            max_rule_size (int, optional): the maximum size of the conjunctions. Defaults to 2 (meaning `rule_one and rule_two`).

        Returns:
            :class:`.CalibratedExplanations`: Returns a self reference, to allow for method chaining
        """
        for explanation in self.explanations:
            explanation.remove_conjunctions()
            explanation.add_conjunctions(n_top_features, max_rule_size)
        return self



    def remove_conjunctions(self):
        """removes any conjunctive rules"""
        for explanation in self.explanations:
            explanation.remove_conjunctions()
        return self



    def get_explanation(self, index):
        '''The function `get_explanation` returns the explanation corresponding to the index.

        Parameters
        ----------
        index
            The `index` parameter is an integer that represents the index of the explanation
            instance that you want to retrieve. It is used to specify which explanation instance you want to
            get from either the alternative rules or the factual rules.

        Returns
        -------
            The method `get_explanation` returns either a :class:`.AlternativeExplanation` or a :class:`.FactualExplanation`, depending
            on the condition `self._is_alternative()`. 
        '''
        assert isinstance(index, int), "index must be an integer"
        assert index >= 0, "index must be greater than or equal to 0"
        assert index < len(self.X_test), "index must be less than the number of test instances"
        return self.explanations[index]


    def _is_alternative(self):
        # '''The function checks if the explanations are alternatives by checking if the `discretizer` attribute of the `calibrated_explainer` object is an
        # instance of either `DecileDiscretizer` or `EntropyDiscretizer`.

        # Returns
        # -------
        #     a boolean value indicating whether the explanations are alternatives.
        # '''
        return isinstance(self.calibrated_explainer.discretizer, (RegressorDiscretizer, EntropyDiscretizer))

    # pylint: disable=too-many-arguments, too-many-locals, unused-argument
    def plot(self,
                index=None,
                filter_top=10,
                show=False,
                filename='',
                uncertainty=False,
                style='regular',
                rnk_metric='feature_weight',
                rnk_weight=0.5,
                sort_on_uncertainty=False,
                interactive=False):
        '''The function `plot` plots either alternative or factual explanations for a given
        instance, with the option to show or save the plots.

        Parameters
        ----------
        index : int or None, default=None
            The index of the instance for which you want to plot the explanation. If None, the function will plot all the explanations.
        filter_top :  int or None, default=10
            The parameter `filter_top` determines the number of top features to display in the
            plot. It specifies how many of the most important features should be shown in the plot. If set to
            `None`, all the features will be shown. 
        show : bool, default=False
            The `show` parameter determines whether the plots should be displayed immediately after they
            are generated. If set to True, the plots will be shown; if set to False, the plots will not be
            shown. Plots will be shown in jupyter notebooks.
        filename : str, default=''
            The filename parameter is a string that represents the full path and filename of the plot
            image file that will be saved. If this parameter is not provided or is an empty string, the plot
            will not be saved as an image file. The index of each explanation will be appended to the
            filename (e.g. filename0.png, filename1.png, etc.).
        uncertainty : bool, default=False
            The `uncertainty` parameter is a boolean flag that determines whether to include uncertainty
            information in the plots. If set to True, the plots will show uncertainty measures, if
            available, along with the explanations. If set to False, the plots will only show the
            explanations without uncertainty information. Only applicable to factual explanations.
        style : str, default='regular'
            The `style` parameter is a string that determines the style of the plot. The following styles are supported: 
            - 'regular': The plot will show the feature weights as bars with the uncertainty intervals as lighter bars.
            - 'triangular': Experimental.
        rnk_metric : str, default='feature_weight'
            The `rnk_metric` parameter is a string that determines the metric used to rank the features. 
            The following metrics are supported:
            - 'ensured': The weighted sum of the feature weights and the uncertainty intervals. The `rnk_weight` 
            parameter can be used to balance the importance of the prediction and the uncertainty.
            - 'feature_weight': The feature weights only. High feature weights are better.
            - 'uncertainty': The uncertainty intervals only. Low uncertainty is better. Is the same as 
            'ensured' with rnk_weight=0.
        rnk_weight : float, default=0.5
            The `rnk_weight` parameter is a float that determines the weight of the uncertainty in the 
            ranking. Used with the 'ensured' ranking metric. 
        '''
        # Check for deprecated parameters and issue warnings
        if sort_on_uncertainty is not None:
            warnings.warn(
                "The 'sort_on_uncertainty' parameter is deprecated and will be removed in future versions.",
                DeprecationWarning
            )
        if len(filename) > 0:
            path = f'{os.path.dirname(filename)}/'
            filename = os.path.basename(filename)
            title, ext = os.path.splitext(filename)
            make_directory(path, save_ext=np.array([ext]))
        if index is not None:
            if len(filename) > 0:
                filename = path + title + str(index) + ext
            self.get_explanation(index).plot(filter_top=filter_top,
                                        show=show, filename=filename, uncertainty=uncertainty, style=style,
                                        rnk_metric=rnk_metric, rnk_weight=rnk_weight,
                                        )
        else:
            for i, explanation in enumerate(self.explanations):
                if len(filename) > 0:
                    filename = path + title + str(i) + ext
                explanation.plot(filter_top=filter_top,
                                        show=show, filename=filename, uncertainty=uncertainty, style=style,
                                        rnk_metric=rnk_metric, rnk_weight=rnk_weight,
                                        )



    # pylint: disable=protected-access
    def as_lime(self, num_features_to_show=None):
        """transforms the explanation into a lime explanation object

        Returns:
            list of lime.Explanation : list of lime explanation objects with the same values as the :class:`.CalibratedExplanations`
        """
        _, lime_exp = self.calibrated_explainer._preload_lime() # pylint: disable=protected-access
        exp = []
        for explanation in self.explanations: #range(len(self.X_test[:,0])):
            tmp = deepcopy(lime_exp)
            tmp.intercept[1] = 0
            tmp.local_pred = explanation.prediction['predict']
            if 'regression' in self.calibrated_explainer.mode:
                tmp.predicted_value = explanation.prediction['predict']
                tmp.min_value = np.min(self.calibrated_explainer.y_cal)
                tmp.max_value = np.max(self.calibrated_explainer.y_cal)
            else:
                tmp.predict_proba[0], tmp.predict_proba[1] = \
                        1-explanation.prediction['predict'], explanation.prediction['predict']

            feature_weights = explanation.feature_weights['predict']
            num_to_show = num_features_to_show if num_features_to_show is not None else \
                self.calibrated_explainer.num_features
            features_to_plot = explanation._rank_features(feature_weights,
                        num_to_show=num_to_show)
            rules = explanation._define_conditions()
            for j,f in enumerate(features_to_plot[::-1]): # pylint: disable=invalid-name
                tmp.local_exp[1][j] = (f, feature_weights[f])
            del tmp.local_exp[1][num_to_show:]
            tmp.domain_mapper.discretized_feature_names = rules
            tmp.domain_mapper.feature_values = explanation.X_test
            exp.append(tmp)
        return exp



    def as_shap(self):
        """transforms the explanation into a shap explanation object

        Returns:
            shap.Explanation : shap explanation object with the same values as the explanation
        """
        _, shap_exp = self.calibrated_explainer._preload_shap() # pylint: disable=protected-access
        shap_exp.base_values = np.resize(shap_exp.base_values, len(self))
        shap_exp.values = np.resize(shap_exp.values, (len(self), len(self.X_test[0, :])))
        shap_exp.data = self.X_test
        for i, explanation in enumerate(self.explanations): #range(len(self.X_test[:,0])):
            # shap_exp.base_values[i] = explanation.prediction['predict']
            for f in range(len(self.X_test[0, :])):
                shap_exp.values[i][f] = -explanation.feature_weights['predict'][f]
        return shap_exp

class AlternativeExplanations(CalibratedExplanations):
    '''
    A class for storing and visualizing alternative explanations, separating methods that are explicit for :class:`.AlternativeExplanation`.
    '''
    def super_explanations(self, only_ensured=False, include_potential=True):
        '''
        The function `super_explanations` returns a copy of this :class:`.AlternativeExplanations` object with only super-explanations.
        Super-explanations are individual rules with higher probability that support the predicted class. 

        Parameters
        ----------
        only_ensured : bool, default=False
            The `only_ensured` parameter is a boolean flag that determines whether to return only ensured explanations, 
            i.e., explanations with a smaller confidence interval. If set to `True`, the function will return only ensured
            explanations. If set to `False`, the function will return all super-explanations. 
        include_potential : bool, default=True
            The `include_potential` parameter is a boolean flag that determines whether to include potential explanations in the
            super-explanations. If set to `True`, the function will include super-potential explanations in the super-explanations.
            If set to `False`, the function will only include super-factual explanations.

        Returns
        -------
        super-explanations : :class:`.AlternativeExplanations`
            A new :class:`.AlternativeExplanations` object containing :class:`.AlternativeExplanation` objects only containing super-factual 
            or super-potential explanations. 

        Notes
        -----
        Super-explanations are only available for :class:`.AlternativeExplanation` explanations.

        Notes
        -----
        only_ensured and include_potential can interact in the following way:
        - only_ensured=True, include_potential=True: ensured explanations takes precedence meaning that unless the original explanation 
            is potential, no potential explanations will be included
        '''
        super_explanations = deepcopy(self)
        for explanation in super_explanations.explanations:
            explanation.super_explanations(only_ensured=only_ensured, include_potential=include_potential)
        return super_explanations


    def semi_explanations(self, only_ensured=False, include_potential=True):
        '''
        The function `semi_explanations` returns a copy of this :class:`.AlternativeExplanations` object with only semi-explanations.
        Semi-explanations are individual rules with lower probability that support the predicted class. 

        Parameters
        ----------
        only_ensured : bool, default=False
            The `only_ensured` parameter is a boolean flag that determines whether to return only ensured explanations, 
            i.e., explanations with a smaller confidence interval. If set to `True`, the function will return only ensured
            explanations. If set to `False`, the function will return all semi-explanations. 
        include_potential : bool, default=True
            The `include_potential` parameter is a boolean flag that determines whether to include potential explanations in the
            semi-explanations. If set to `True`, the function will include semi-potential explanations in the semi-explanations.
            If set to `False`, the function will only include semi-factual explanations.

        Returns
        -------
        semi-explanations : :class:`.AlternativeExplanations`
            A new :class:`.AlternativeExplanations` object containing :class:`.AlternativeExplanation` objects only containing semi-factual 
            or semi-potential explanations. 

        Notes
        -----
        Semi-explanations are only available for :class:`.AlternativeExplanation` explanations.

        Notes
        -----
        only_ensured and include_potential can interact in the following way:
        - only_ensured=True, include_potential=True: ensured explanations takes precedence meaning that unless the original explanation 
            is potential, no potential explanations will be included
        '''
        semi_explanations = deepcopy(self)
        for explanation in semi_explanations.explanations:
            explanation.semi_explanations(only_ensured=only_ensured, include_potential=include_potential)
        return semi_explanations


    def counter_explanations(self, only_ensured=False, include_potential=True):
        '''
        The function `counter_explanations` returns a copy of this :class:`.AlternativeExplanations` object with only counter-explanations.
        Counter-explanations are individual rules that does not support the predicted class. 

        Parameters
        ----------
        only_ensured : bool, default=False
            The `only_ensured` parameter is a boolean flag that determines whether to return only ensured explanations, 
            i.e., explanations with a smaller confidence interval. If set to `True`, the function will return only ensured
            explanations. If set to `False`, the function will return all counter-explanations.
        include_potential : bool, default=True
            The `include_potential` parameter is a boolean flag that determines whether to include potential explanations in the
            counter-explanations. If set to `True`, the function will include counter-potential explanations in the counter-explanations.
            If set to `False`, the function will only include counter-factual explanations.

        Returns
        -------
        counter-explanations : :class:`.AlternativeExplanations`
            A new :class:`.AlternativeExplanations` object containing :class:`.AlternativeExplanation` objects only containing counter-factual 
            or counter-potential explanations. 

        Notes
        -----
        Counter-explanations are only available for :class:`.AlternativeExplanation` explanations.

        Notes
        -----
        only_ensured and include_potential can interact in the following way:
        - only_ensured=True, include_potential=True: ensured explanations takes precedence meaning that unless the original explanation 
            is potential, no potential explanations will be included
        '''
        counter_explanations = deepcopy(self)
        for explanation in counter_explanations.explanations:
            explanation.counter_explanations(only_ensured=only_ensured, include_potential=include_potential)
        return counter_explanations

    def ensured_explanations(self):
        '''
        The function `ensured_explanations` returns a copy of this :class:`.AlternativeExplanations` object with only ensured explanations.
        Ensured explanations are individual rules that have a smaller confidence interval. 

        Returns
        -------
        ensured-explanations : AlternativeExplanations
            A new :class:`.AlternativeExplanations` object containing :class:`.AlternativeExplanation` objects only containing ensured 
            explanations. 
        '''
        ensured_explanations = deepcopy(self)
        for explanation in ensured_explanations.explanations:
            explanation.ensured_explanations()
        return ensured_explanations


# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-arguments
class CalibratedExplanation(ABC):
    '''
    A class for storing and visualizing calibrated explanations.
    '''
    def __init__(self, calibrated_explanations, index, X_test, binned, feature_weights, feature_predict, prediction, y_threshold=None, instance_bin=None):
        self.calibrated_explanations = calibrated_explanations
        self.index = index
        self.X_test = X_test
        self.binned = {}
        self.feature_weights = {}
        self.feature_predict = {}
        self.prediction = {}
        for key in binned.keys():
            self.binned[key] = deepcopy(binned[key][index])
        for key in feature_weights.keys():
            self.feature_weights[key] = deepcopy(feature_weights[key][index])
            self.feature_predict[key] = deepcopy(feature_predict[key][index])
        for key in prediction.keys():
            self.prediction[key] = deepcopy(prediction[key][index])
        self.y_threshold=y_threshold if np.isscalar(y_threshold) or isinstance(y_threshold, tuple) else \
                            None if y_threshold is None else \
                            y_threshold[index]

        self.conditions = []
        self.rules = None
        self.conjunctive_rules = None
        self._has_rules = False
        self._has_conjunctive_rules = False
        self.bin = [instance_bin] if instance_bin is not None else None
        self.explain_time = None
        # reduce dependence on Explainer class
        if not isinstance(self._get_explainer().y_cal, Categorical):
            self.y_minmax = [np.min(self._get_explainer().y_cal), np.max(self._get_explainer().y_cal)]
        else:
            self.y_minmax = [0,0]
        self.focus_columns = None

    def __len__(self):
        return len(self._get_rules()['rule'])


    def get_mode(self):
        '''
        Returns the mode of the explanation (classification or regression)
        '''
        return self._get_explainer().mode

    def get_class_labels(self):
        '''
        returns the class labels
        '''
        return self._get_explainer().class_labels

    def is_multiclass(self):
        '''
        returns whether the explanation is multiclass or not
        '''
        return self._get_explainer().is_multiclass()

    def _get_explainer(self):
        return self.calibrated_explanations._get_explainer() # pylint: disable=protected-access

    def _rank_features(self, feature_weights=None, width=None, num_to_show=None):
        assert feature_weights is not None or width is not None, 'Either feature_weights or width (or both) must not be None'
        num_features = len(feature_weights) if feature_weights is not None else len(width)
        if num_to_show is None or num_to_show > num_features:
            num_to_show = num_features
        # handle case where there are same weight but different uncertainty
        if feature_weights is not None and width is not None:
            # get the indeces by first sorting on the absolute value of the
            # feature_weight and then on the width
            sorted_indices = [i for i, x in
                                sorted(enumerate(list(zip(np.abs(feature_weights), width))),
                                key=lambda x: (x[1][0], x[1][1]))]
            return sorted_indices[-num_to_show:] # pylint: disable=invalid-unary-operand-type
        if width is not None:
            sorted_indices = np.argsort(width)
            return sorted_indices[-num_to_show:] # pylint: disable=invalid-unary-operand-type
        sorted_indices = np.argsort(np.abs(feature_weights))
        return sorted_indices[-num_to_show:] # pylint: disable=invalid-unary-operand-type

    def is_one_sided(self) -> bool:
        """test if a regression explanation is one-sided

        Returns:
            bool: True if one of the low or high percentiles is infinite
        """
        if self.calibrated_explanations.low_high_percentiles is None:
            return False
        return np.isinf(self.calibrated_explanations.get_low_percentile()) or \
                np.isinf(self.calibrated_explanations.get_high_percentile())

    def is_thresholded(self) -> bool:
        """test if the explanation is thresholded

        Returns:
            bool: True if the y_threshold is not None
        """
        return self.y_threshold is not None

    def is_regression(self) -> bool:
        """test if the explanation is for regression

        Returns:
            bool: True if mode is 'regression'
        """
        return 'regression' in self._get_explainer().mode

    def is_probabilistic(self) -> bool:
        """test if the explanation is probabilistic

        Returns:
            bool: True if mode is 'classification' or is_thresholded and is_regression are True
        """
        return 'classification' in self._get_explainer().mode or (self.is_regression() and self.is_thresholded())

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def plot(self, filter_top=None, **kwargs):
        '''The `plot` function plots explanations for a given
        instance, with the option to show or save the plots.
        '''
        # pass

    @abstractmethod
    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        '''The function `add_conjunctions` adds conjunctive rules to the factual or alternative
        explanations. The conjunctive rules are added to the `conjunctive_rules` attribute of the
        `CalibratedExplanations` object.
        '''

    @abstractmethod
    def _check_preconditions(self):
        pass

    @abstractmethod
    def _get_rules(self):
        pass

    def remove_conjunctions(self):
        """removes any conjunctive rules"""
        self._has_conjunctive_rules = False
        return self


    def _define_conditions(self):
        # """defines the rule conditions for an instance

        # Args:
        #     instance (n_features,): a test instance

        # Returns:
        #     list[str]: a list of conditions for each feature in the instance
        # """
        self.conditions = []
        # pylint: disable=invalid-name
        x = self._get_explainer().discretizer.discretize(self.X_test)
        for f in range(self._get_explainer().num_features):
            if f in self._get_explainer().categorical_features:
                if self._get_explainer().categorical_labels is not None:
                    try:
                        target = self._get_explainer().categorical_labels[f][int(x[f])]
                        rule = f"{self._get_explainer().feature_names[f]} = {target}"
                    except IndexError:
                        rule = f"{self._get_explainer().feature_names[f]} = {x[f]}"
                else:
                    rule = f"{self._get_explainer().feature_names[f]} = {x[f]}"
            else:
                rule = self._get_explainer().discretizer.names[f][int(x[f])]
            self.conditions.append(rule)
        return self.conditions


    def _predict_conjunctive(self, rule_value_set, original_features, perturbed, threshold, # pylint: disable=invalid-name, too-many-locals, too-many-arguments
                            predicted_class, bins=None):
        # """support function to calculate the prediction for a conjunctive rule
        # """
        assert len(original_features) >= 2, 'Conjunctive rules require at least two features'
        rule_predict, rule_low, rule_high, rule_count = 0,0,0,0
        of1, of2, of3 = 0,0,0
        rule_value1, rule_value2, rule_value3 = 0,0,0
        if len(original_features) == 2:
            of1, of2 = original_features[0], original_features[1]
            rule_value1, rule_value2 = rule_value_set[0], rule_value_set[1]
        elif len(original_features) >= 3:
            of1, of2, of3 = original_features[0], original_features[1], original_features[2]
            rule_value1, rule_value2, rule_value3 = rule_value_set[0], rule_value_set[1], rule_value_set[2]
        for value_1 in rule_value1:
            perturbed[of1] = value_1
            for value_2 in rule_value2:
                perturbed[of2] = value_2
                if len(original_features) >= 3:
                    for value_3 in rule_value3:
                        perturbed[of3] = value_3
                        p_value, low, high, _ = self._get_explainer()._predict(perturbed.reshape(1,-1), # pylint: disable=protected-access
                                            threshold=threshold, low_high_percentiles=self.calibrated_explanations.low_high_percentiles,
                                            classes=predicted_class, bins=bins)
                        rule_predict += p_value[0]
                        rule_low += low[0]
                        rule_high += high[0]
                        rule_count += 1
                else:
                    p_value, low, high, _ = self._get_explainer()._predict(perturbed.reshape(1,-1), # pylint: disable=protected-access
                                                threshold=threshold, low_high_percentiles=self.calibrated_explanations.low_high_percentiles,
                                                classes=predicted_class, bins=bins)
                    rule_predict += p_value[0]
                    rule_low += low[0]
                    rule_high += high[0]
                    rule_count += 1
        rule_predict /= rule_count
        rule_low /= rule_count
        rule_high /= rule_count
        return rule_predict, rule_low, rule_high

    @abstractmethod
    def _is_lesser(self, rule_boundary, instance_value):
        pass

    # pylint: disable=too-many-arguments, too-many-statements, too-many-branches, too-many-return-statements
    def add_new_explanation(self, feature, rule_boundary):
        """creates an explanation for a feature or a set of features with user defined values

        Parameters
        ----------
        feature : int or str
            the feature to focus attention on
        rule_boundary: int, float, str, or categorical
            the value to define as rule condition

        Returns
        -------
        :class:`.CalibratedExplanation`
        """
        try:
            f = feature if isinstance(feature, int)\
                                        else self._get_explainer().feature_names.index(feature)
        except ValueError:
            warnings.warn(f'Feature {feature} not found')
            return self
        if self._get_explainer().categorical_features is not None and f in self._get_explainer().categorical_features:
            warnings.warn('Alternatives for all categorical features are already included')
            return self

        X_copy = deepcopy(self.X_test[f])
        is_lesser = self._is_lesser(rule_boundary, X_copy)
        new_rule = self._get_rules()
        exists = False
        if is_lesser:
            if np.any([new_rule['rule'][i] == f'{feature} < {rule_boundary:.2f}' for i in range(len(new_rule['rule']))]):
                exists = True
        elif np.any([new_rule['rule'][i] == f'{feature} > {rule_boundary:.2f}' for i in range(len(new_rule['rule']))]):
            exists = True
        if exists:
            warnings.warn('Rule already included')
            return self

        threshold = self.y_threshold
        perturbed_threshold = self._get_explainer().assign_threshold(threshold)
        perturbed_bins = np.empty((0,)) if self.bin is not None else None
        perturbed_X = np.empty((0, self._get_explainer().num_features))
        perturbed_feature = np.empty((0,4)) # (feature, instance, bin_index, is_lesser)
        perturbed_class = np.empty((0,),dtype=int)

        cal_X_f = self._get_explainer().X_cal[:,f]
        feature_values = np.unique(np.array(cal_X_f))
        sample_percentiles = self._get_explainer().sample_percentiles

        if is_lesser:
            if not np.any(feature_values < rule_boundary):
                warnings.warn(f'Lowest feature value for feature {feature} is {np.min(feature_values)}')
                return self
            values = np.percentile(cal_X_f[cal_X_f < rule_boundary],
                                    sample_percentiles)
            covered = np.percentile(cal_X_f[cal_X_f >= rule_boundary],
                                    sample_percentiles)
        else:
            if not np.any(feature_values > rule_boundary):
                warnings.warn(f'Highest feature value for feature {feature} is {np.max(feature_values)}')
                return self
            values = np.percentile(cal_X_f[cal_X_f > rule_boundary],
                                    sample_percentiles)
            covered = np.percentile(cal_X_f[cal_X_f <= rule_boundary],
                                    sample_percentiles)

        for value in values:
            X_local = np.reshape(deepcopy(self.X_test), (1,-1))
            X_local[0,f] = value
            perturbed_X = np.concatenate((perturbed_X, np.array(X_local)))
            perturbed_feature = np.concatenate((perturbed_feature, [(f, 0, None, is_lesser)]))
            perturbed_bins = np.concatenate((perturbed_bins, self.bin)) if self.bin is not None else None
            perturbed_class = np.concatenate((perturbed_class, np.array([self.prediction['classes']])))
            if isinstance(threshold, tuple):
                perturbed_threshold = threshold
            else:
                perturbed_threshold = np.concatenate((perturbed_threshold, threshold))

        for value in covered:
            X_local = np.reshape(deepcopy(self.X_test), (1,-1))
            X_local[0,f] = value
            perturbed_X = np.concatenate((perturbed_X, np.array(X_local)))
            perturbed_feature = np.concatenate((perturbed_feature, [(f, 0, None, None)]))
            perturbed_bins = np.concatenate((perturbed_bins, self.bin)) if self.bin is not None else None
            perturbed_class = np.concatenate((perturbed_class, np.array([self.prediction['classes']])))
            if isinstance(threshold, tuple):
                perturbed_threshold = threshold
            else:
                perturbed_threshold = np.concatenate((perturbed_threshold, threshold))

        # pylint: disable=protected-access
        predict, low, high, _ = self._get_explainer()._predict(perturbed_X, threshold=perturbed_threshold, low_high_percentiles=self.calibrated_explanations.low_high_percentiles, classes=perturbed_class, bins=perturbed_bins)
        instance_predict = [predict[i] for i in range(len(predict)) if perturbed_feature[i][3] is None]
        rule_predict = [predict[i] for i in range(len(predict)) if perturbed_feature[i][3] is not None]
        rule_low = [low[i] for i in range(len(low)) if perturbed_feature[i][3] is not None]
        rule_high = [high[i] for i in range(len(high)) if perturbed_feature[i][3] is not None]

        # skip if identical to original
        if self.prediction['low'] == np.mean(rule_low) and self.prediction['high'] == np.mean(rule_high):
            warnings.warn('The alternative explanation is identical to the original explanation')
            return self
        new_rule['predict'].append(np.mean(rule_predict))
        new_rule['predict_low'].append(np.mean(rule_low))
        new_rule['predict_high'].append(np.mean(rule_high))
        new_rule['weight'].append(np.mean(rule_predict) - \
                                                            np.mean(instance_predict))
        new_rule['weight_low'].append(
                        np.mean(rule_low) -
                        np.mean(instance_predict) \
                                    if rule_low != -np.inf \
                                    else rule_low)
        new_rule['weight_high'].append(
                        np.mean(rule_high) -
                        np.mean(instance_predict) \
                                    if rule_high != np.inf \
                                    else rule_high)
        new_rule['value'].append(str(np.around(X_copy, decimals=2)))
        new_rule['feature'].append(f)
        new_rule['feature_value'].append(
                        self.binned['rule_values'][f][0][0])
        new_rule['is_conjunctive'].append(False)

        if is_lesser:
            new_rule['rule'].append(
                        f'{self._get_explainer().feature_names[f]} < {rule_boundary:.2f}')
        else:
            new_rule['rule'].append(
                        f'{self._get_explainer().feature_names[f]} > {rule_boundary:.2f}')
        self.rules = new_rule
        return self

# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-arguments
class FactualExplanation(CalibratedExplanation):
    '''
    A class for storing and visualizing factual explanations.
    '''
    def __init__(self, calibrated_explanations, index, X_test, binned, feature_weights, feature_predict, prediction, y_threshold=None, instance_bin=None):
        super().__init__(calibrated_explanations, index, X_test, binned, feature_weights, feature_predict, prediction, y_threshold, instance_bin)
        self._check_preconditions()
        self._get_rules()

    def __repr__(self):
        factual = self._get_rules()
        output = [
            f"{'Prediction':10} [{' Low':5}, {' High':5}]",
            f"{factual['base_predict'][0]:5.3f} [{factual['base_predict_low'][0]:5.3f}, {factual['base_predict_high'][0]:5.3f}]",
            f"{'Value':6}: {'Feature':40s} {'Weight':6} [{' Low':6}, {' High':6}]",
        ]
        feature_order = self._rank_features(factual['weight'],
                                width=np.array(factual['weight_high']) - np.array(factual['weight_low']),
                                num_to_show=len(factual['rule']))
        output.extend(
            f"{factual['value'][f]:6}: {factual['rule'][f]:40s} {factual['weight'][f]:>6.3f} [{factual['weight_low'][f]:>6.3f}, {factual['weight_high'][f]:>6.3f}]"
            for f in reversed(feature_order)
        )
        return "\n".join(output) + "\n"

    def _check_preconditions(self):
        if self.is_regression():
            if not isinstance(self._get_explainer().discretizer, BinaryRegressorDiscretizer):
                warnings.warn('Factual explanations for regression recommend using the binaryRegressor ' +\
                                    'discretizer. Consider extracting factual explanations using ' +\
                                    '`explainer.explain_factual(test_set)`')
        elif not isinstance(self._get_explainer().discretizer, BinaryEntropyDiscretizer):
            warnings.warn('Factual explanations for classification recommend using the ' +\
                                'binaryEntropy discretizer. Consider extracting factual ' +\
                                'explanations using `explainer.explain_factual(test_set)`')

    def _get_rules(self):
        # """creates factual rules

        # Returns:
        #     List[Dict[str, List]]: a list of dictionaries containing the factual rules, one for each test instance
        # """
        if self._has_conjunctive_rules:
            return self.conjunctive_rules
        if self._has_rules:
            return self.rules
        self._has_rules = False
        # i = self.index
        instance = deepcopy(self.X_test)
        factual = {
            'base_predict': [],
            'base_predict_low': [],
            'base_predict_high': [],
            'predict': [],
            'predict_low': [],
            'predict_high': [],
            'weight': [],
            'weight_low': [],
            'weight_high': [],
            'value': [],
            'rule': [],
            'feature': [],
            'feature_value': [],
            'is_conjunctive': [],
            'classes': self.prediction['classes'],
        }
        factual['base_predict'].append(self.prediction['predict'])
        factual['base_predict_low'].append(self.prediction['low'])
        factual['base_predict_high'].append(self.prediction['high'])
        rules = self._define_conditions()
        for f,_ in enumerate(instance): # pylint: disable=invalid-name
            if self.prediction['predict'] == self.feature_predict['predict'][f]:
                continue
            factual['predict'].append(self.feature_predict['predict'][f])
            factual['predict_low'].append(self.feature_predict['low'][f])
            factual['predict_high'].append(self.feature_predict['high'][f])
            factual['weight'].append(self.feature_weights['predict'][f])
            factual['weight_low'].append(self.feature_weights['low'][f])
            factual['weight_high'].append(self.feature_weights['high'][f])
            if f in self._get_explainer().categorical_features:
                if self._get_explainer().categorical_labels is not None:
                    factual['value'].append(
                        self._get_explainer().categorical_labels[f][int(instance[f])])
                else:
                    factual['value'].append(str(instance[f]))
            else:
                factual['value'].append(str(np.around(instance[f],decimals=2)))
            factual['rule'].append(rules[f])
            factual['feature'].append(f)
            factual['feature_value'].append(self.binned['rule_values'][f][0][-1])
            factual['is_conjunctive'].append(False)
        self.rules = factual
        self._has_rules = True
        return self.rules


    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        # """adds conjunctive factual rules

        # Args:
        #     n_top_features (int, optional): the number of most important factual rules to try to combine into conjunctive rules. Defaults to 5.
        #     max_rule_size (int, optional): the maximum size of the conjunctions. Defaults to 2 (meaning `rule_one and rule_two`).

        # Returns:
        #     CalibratedExplanations: Returns a self reference, to allow for method chaining
        # """
        if max_rule_size >= 4:
            raise ValueError('max_rule_size must be 2 or 3')
        if max_rule_size < 2:
            return self
        if not self._has_rules:
            factual = deepcopy(self._get_rules())
        else:
            factual = deepcopy(self.rules)
        if self._has_conjunctive_rules:
            conjunctive = self.conjunctive_rules
        else:
            conjunctive = deepcopy(factual)
        self._has_conjunctive_rules = False
        self.conjunctive_rules = []
        i =self.index
        # pylint: disable=unsubscriptable-object, invalid-name
        threshold = None if self.y_threshold is None else self.y_threshold
        x_original = deepcopy(self.X_test)

        num_rules = len(factual['rule'])
        predicted_class = factual['classes']
        conjunctive['classes'] = predicted_class
        if n_top_features is None:
            n_top_features = num_rules
        top_conjunctives = self._rank_features(np.reshape(conjunctive['weight'], (len(conjunctive['weight']))),
                            width=np.reshape(np.array(conjunctive['weight_high']) - np.array(conjunctive['weight_low']),
                            (len(conjunctive['weight']))), num_to_show= np.min([num_rules, n_top_features]))

        covered_features = []
        covered_combinations = [conjunctive['feature'][i] for i in range(len(conjunctive['rule']))]
        for f1, cf1 in enumerate(factual['feature']): # cf = factual feature
            covered_features.append(cf1)
            of1 = factual['feature'][f1] # of = original feature
            rule_value1 = factual['feature_value'][f1] \
                                if isinstance(factual['feature_value'][f1], np.ndarray) \
                                else [factual['feature_value'][f1]]
            for _, cf2 in enumerate(top_conjunctives): # cf = conjunctive feature
                if cf2 in covered_features:
                    continue
                rule_values = [rule_value1]
                original_features = [of1]
                of2 = conjunctive['feature'][cf2]
                if conjunctive['is_conjunctive'][cf2]:
                    if of1 in of2:
                        continue
                    for of in of2:
                        original_features.append(of)
                    for rule_value in conjunctive['feature_value'][cf2]:
                        rule_values.append(rule_value)
                else:
                    if of1 == of2:
                        continue
                    original_features.append(of2)
                    rule_values.append(conjunctive['feature_value'][cf2] \
                                    if isinstance(conjunctive['feature_value'][cf2], np.ndarray) \
                                    else [conjunctive['feature_value'][cf2]])
                skip = False
                for ofs in covered_combinations:
                    with contextlib.suppress(ValueError):
                        if np.all(np.sort(original_features) == ofs):
                            skip = True
                            break
                if skip:
                    continue
                covered_combinations.append(np.sort(original_features))

                rule_predict, rule_low, rule_high = self._predict_conjunctive(rule_values,
                                                                        original_features,
                                                                        deepcopy(x_original),
                                                                        threshold,
                                                                        predicted_class,
                                                                        bins=self.bin)

                conjunctive['predict'].append(rule_predict)
                conjunctive['predict_low'].append(rule_low)
                conjunctive['predict_high'].append(rule_high)
                conjunctive['weight'].append(rule_predict - self.prediction['predict'])
                conjunctive['weight_low'].append(rule_low - self.prediction['predict'] \
                                                    if rule_low != -np.inf else -np.inf)
                conjunctive['weight_high'].append(rule_high - self.prediction['predict'] \
                                                    if rule_high != np.inf else np.inf)
                conjunctive['value'].append(factual['value'][f1]+ '\n' +conjunctive['value'][cf2])
                conjunctive['feature'].append(original_features)
                conjunctive['feature_value'].append(rule_values)
                conjunctive['rule'].append(factual['rule'][f1]+ ' & \n' +conjunctive['rule'][cf2])
                conjunctive['is_conjunctive'].append(True)
        self.conjunctive_rules = conjunctive
        self._has_conjunctive_rules = True
        return self.add_conjunctions(n_top_features=n_top_features, max_rule_size=max_rule_size-1)

    def _is_lesser(self, rule_boundary, instance_value):
        return instance_value < rule_boundary

    def plot(self, filter_top=None, **kwargs):
        '''This function plots the factual explanation for a given instance using either probabilistic or
        regression plots.
        
        Parameters
        ----------
        filter_top : int, default=10
            The `filter_top` parameter determines the number of top features to display in the
            plot. If set to `None`, it will show all the features. Otherwise, it will show the specified
            number of features, up to the total number of features available.
        show : bool, default=False
            A boolean parameter that determines whether the plot should be displayed or not. If set to
            True, the plot will be displayed. If set to False, the plot will not be displayed.
        filename : str, default=''
            The filename parameter is a string that represents the full path and filename of the plot
            image file that will be saved. If this parameter is not provided or is an empty string, the plot
            will not be saved as an image file.
        uncertainty : bool, default=False
            The `uncertainty` parameter is a boolean flag that determines whether to plot the uncertainty
            intervals for the feature weights. If `uncertainty` is set to `True`, the plot will show the
            range of possible feature weights based on the lower and upper bounds of the uncertainty
            intervals. If `uncertainty` is set to `False`, the plot will only show the feature weights
        style : str, default='regular'
            The `style` parameter is a string that determines the style of the plot. Possible styles are for :class:`.FactualExplanation`:
            * 'regular' - a regular plot with feature weights and uncertainty intervals (if applicable)        
        '''
        show = kwargs.get('show', False)
        filename = kwargs.get('filename', '')
        uncertainty = kwargs.get('uncertainty', False)
        rnk_metric = kwargs.get('rnk_metric', 'feature_weight')
        rnk_weight = kwargs.get('rnk_weight', 0.5)
        if rnk_metric == 'uncertainty':
            rnk_weight = 1.0
            rnk_metric = 'ensured'

        factual = self._get_rules() #get_explanation(index)
        self._check_preconditions()
        predict = self.prediction
        num_features_to_show = len(factual['weight'])
        if filter_top is None:
            filter_top = num_features_to_show
        filter_top = np.min([num_features_to_show, filter_top])
        if filter_top <= 0:
            warnings.warn(f'The explanation has no rules to plot. The index of the instance is {self.index}')
            return

        if len(filename) > 0:
            path = f'{os.path.dirname(filename)}/'
            filename = os.path.basename(filename)
            title, ext = os.path.splitext(filename)
            make_directory(path, save_ext=np.array([ext]))
            path = f'plots/{path}'
            save_ext = [ext]
        else:
            path = ''
            title = ''
            save_ext = []
        if uncertainty:
            feature_weights = {'predict':factual['weight'],
                                'low':factual['weight_low'], 
                                'high':factual['weight_high']}
        else:
            feature_weights = factual['weight']
        width = np.reshape(np.array(factual['weight_high']) - np.array(factual['weight_low']),
                        (len(factual['weight'])))

        if rnk_metric == 'feature_weight':
            features_to_plot = self._rank_features(factual['weight'],
                                                width=width,
                                                num_to_show=filter_top)
        else:
            ranking = calculate_metrics(uncertainty=[factual['predict_high'][i]-factual['predict_low'][i] for i in range(len(factual['weight']))],
                                                prediction=factual['predict'],
                                                w=rnk_weight,
                                                metric=rnk_metric,
                                                )
            features_to_plot = self._rank_features(width=ranking,
                                                num_to_show=filter_top)

        column_names = factual['rule']
        if 'classification' in self._get_explainer().mode or self.is_thresholded():
            _plot_probabilistic(self, factual['value'], predict, feature_weights, features_to_plot,
                        filter_top, column_names, title=title, path=path, interval=uncertainty, show=show, idx=self.index,
                        save_ext=save_ext)
        else:
            _plot_regression(self, factual['value'], predict, feature_weights, features_to_plot,
                        filter_top, column_names, title=title, path=path, interval=uncertainty, show=show, idx=self.index,
                        save_ext=save_ext)



class AlternativeExplanation(CalibratedExplanation):
    '''This class represents an alternative explanation for a given instance. It is a subclass of
    :class:`.CalibratedExplanation` and inherits all its properties and methods. 
    '''
    def __init__(self, calibrated_explanations, index, X_test, binned, feature_weights, feature_predict, prediction, y_threshold=None, instance_bin=None):
        super().__init__(calibrated_explanations, index, X_test, binned, feature_weights, feature_predict, prediction, y_threshold, instance_bin)
        self._check_preconditions()
        self._has_rules = False
        self._get_rules()
        self.__is_super_explanation = False
        self.__is_semi_explanation = False
        self.__is_counter_explanation = False

    def __repr__(self):
        alternative = self._get_rules()
        output = [
            f"{'Prediction':10} [{' Low':5}, {' High':5}]",
            f"{alternative['base_predict'][0]:5.3f} [{alternative['base_predict_low'][0]:5.3f}, {alternative['base_predict_high'][0]:5.3f}]",
            f"{'Value':6}: {'Feature':40s} {'Prediction':10} [{' Low':6}, {' High':6}]",
        ]
        feature_order = self._rank_features(alternative['weight'],
                                width=np.array(alternative['weight_high']) - np.array(alternative['weight_low']),
                                num_to_show=len(alternative['rule']))
        output.extend(
            f"{alternative['value'][f]:6}: {alternative['rule'][f]:40s} {alternative['predict'][f]:>6.3f}     [{alternative['predict_low'][f]:>6.3f}, {alternative['predict_high'][f]:>6.3f}]"
            for f in reversed(feature_order)
        )
        return "\n".join(output) + "\n"

    def _check_preconditions(self):
        if self.is_regression():
            if not isinstance(self._get_explainer().discretizer, RegressorDiscretizer):
                warnings.warn('Alternative explanations for regression recommend using the ' +\
                                    'regressor discretizer. Consider extracting alternative ' +\
                                    'explanations using `explainer.explain_alternatives(test_set)`')
        elif not isinstance(self._get_explainer().discretizer, EntropyDiscretizer):
            warnings.warn('Alternative explanations for classification recommend using ' +\
                                'the entropy discretizer. Consider extracting alternative ' +\
                                'explanations using `explainer.explain_alternatives(test_set)`')

    # pylint: disable=too-many-statements, too-many-branches
    def _get_rules(self):
        # """creates alternative rules

        # Returns:
        #     List[Dict[str, List]]: a list of dictionaries containing the alternative rules, one for each test instance
        # """
        if self._has_conjunctive_rules:
            return self.conjunctive_rules
        if self._has_rules:
            return self.rules
        self.rules = []
        self.labels = {} # pylint: disable=attribute-defined-outside-init
        instance = deepcopy(self.X_test)
        discretized = self._get_explainer()._discretize(deepcopy(instance).reshape(1,-1))[0] # pylint: disable=protected-access
        instance_predict = self.binned['predict']
        instance_low = self.binned['low']
        instance_high = self.binned['high']
        alternative = self.__set_up_result()
        rule_boundaries = self._get_explainer().rule_boundaries(deepcopy(instance))
        for f,_ in enumerate(instance): # pylint: disable=invalid-name
            if f in self._get_explainer().categorical_features:
                values = np.array(self._get_explainer().feature_values[f])
                values = np.delete(values, values == discretized[f])
                for value_bin, value in enumerate(values):
                    # skip if identical to original
                    if self.prediction['low'] == instance_low[f][value_bin] and self.prediction['high'] == instance_high[f][value_bin]:
                        continue
                    alternative['predict'].append(instance_predict[f][value_bin])
                    alternative['predict_low'].append(instance_low[f][value_bin])
                    alternative['predict_high'].append(instance_high[f][value_bin])
                    alternative['weight'].append(instance_predict[f][value_bin] - \
                                                            self.prediction['predict'])
                    alternative['weight_low'].append(instance_low[f][value_bin] - \
                                                                self.prediction['predict'] \
                                                    if instance_low[f][value_bin] != -np.inf \
                                                    else instance_low[f][value_bin])
                    alternative['weight_high'].append(instance_high[f][value_bin] - \
                                                                self.prediction['predict'] \
                                                    if instance_high[f][value_bin] != np.inf \
                                                    else instance_high[f][value_bin])
                    if self._get_explainer().categorical_labels is not None:
                        alternative['value'].append(
                            self._get_explainer().categorical_labels[f][int(instance[f])])
                    else:
                        alternative['value'].append(str(np.around(instance[f],decimals=2)))
                    alternative['feature'].append(f)
                    alternative['feature_value'].append(value)
                    if self._get_explainer().categorical_labels is not None:
                        self.labels[len(alternative['rule'])] = f
                        alternative['rule'].append(
                                f'{self._get_explainer().feature_names[f]} = '+\
                                        f'{self._get_explainer().categorical_labels[f][int(value)]}')
                    else:
                        alternative['rule'].append(
                                f'{self._get_explainer().feature_names[f]} = {value}')
                    alternative['is_conjunctive'].append(False)
            else:
                values = np.array(self._get_explainer().X_cal[:,f])
                lesser = rule_boundaries[f][0]
                greater = rule_boundaries[f][1]

                value_bin = 0
                if np.any(values < lesser):
                    # skip if identical to original
                    if self.prediction['low'] == np.mean(instance_low[f][value_bin]) and self.prediction['high'] == np.mean(instance_high[f][value_bin]):
                        continue
                    alternative['predict'].append(np.mean(instance_predict[f][value_bin]))
                    alternative['predict_low'].append(np.mean(instance_low[f][value_bin]))
                    alternative['predict_high'].append(np.mean(instance_high[f][value_bin]))
                    alternative['weight'].append(np.mean(instance_predict[f][value_bin]) - \
                                                                    self.prediction['predict'])
                    alternative['weight_low'].append(
                                    np.mean(instance_low[f][value_bin]) -
                                    self.prediction['predict'] \
                                            if instance_low[f][value_bin] != -np.inf \
                                            else instance_low[f][value_bin])
                    alternative['weight_high'].append(
                                    np.mean(instance_high[f][value_bin]) -
                                    self.prediction['predict'] \
                                            if instance_high[f][value_bin] != np.inf \
                                            else instance_high[f][value_bin])
                    alternative['value'].append(str(np.around(instance[f],decimals=2)))
                    alternative['feature'].append(f)
                    alternative['feature_value'].append(
                                    self.binned['rule_values'][f][0][0])
                    alternative['rule'].append(
                                    f'{self._get_explainer().feature_names[f]} < {lesser:.2f}')
                    alternative['is_conjunctive'].append(False)
                    value_bin = 1

                if np.any(values > greater):
                    # skip if identical to original
                    if self.prediction['low'] == np.mean(instance_low[f][value_bin]) and self.prediction['high'] == np.mean(instance_high[f][value_bin]):
                        continue
                    alternative['predict'].append(np.mean(instance_predict[f][value_bin]))
                    alternative['predict_low'].append(np.mean(instance_low[f][value_bin]))
                    alternative['predict_high'].append(np.mean(instance_high[f][value_bin]))
                    alternative['weight'].append(
                                    np.mean(instance_predict[f][value_bin]) -
                                    self.prediction['predict'])
                    alternative['weight_low'].append(np.mean(instance_low[f][value_bin]) -
                                                                self.prediction['predict'] \
                                            if instance_low[f][value_bin] != -np.inf \
                                            else instance_low[f][value_bin])
                    alternative['weight_high'].append(np.mean(instance_high[f][value_bin]) -
                                                                self.prediction['predict'] \
                                            if instance_high[f][value_bin] != np.inf \
                                            else instance_high[f][value_bin])
                    alternative['value'].append(str(np.around(instance[f],decimals=2)))
                    alternative['feature'].append(f)
                    alternative['feature_value'].append(
                                    self.binned['rule_values'][f][0][1 \
                                            if len(self.binned['rule_values'][f][0]) == 3 else 0])
                    alternative['rule'].append(
                                    f'{self._get_explainer().feature_names[f]} > {greater:.2f}')
                    alternative['is_conjunctive'].append(False)

        self.rules = alternative
        self._has_rules = True
        return self.rules

    def __set_up_result(self):
        result = {
            'base_predict': [],
            'base_predict_low': [],
            'base_predict_high': [],
            'predict': [],
            'predict_low': [],
            'predict_high': [],
            'weight': [],
            'weight_low': [],
            'weight_high': [],
            'value': [],
            'rule': [],
            'feature': [],
            'feature_value': [],
            'is_conjunctive': [],
            'classes': self.prediction['classes'],
        }
        result['base_predict'].append(self.prediction['predict'])
        result['base_predict_low'].append(self.prediction['low'])
        result['base_predict_high'].append(self.prediction['high'])
        return result

    def is_super_explanation(self):
        '''
        This function returns a boolean value that indicates whether the explanation is a super-explanation or not.
        '''
        return self.__is_super_explanation

    def is_semi_explanation(self):
        '''
        This function returns a boolean value that indicates whether the explanation is a semi-explanation or not.
        '''
        return self.__is_semi_explanation

    def is_counter_explanation(self):
        '''
        This function returns a boolean value that indicates whether the explanation is a counter-explanation or not.
        '''
        return self.__is_counter_explanation

    def __filter_rules(self, only_ensured=False, make_super=False, make_semi=False, make_counter=False, include_potential=False):
        '''
        This is a support function to semi and counter explanations. It filters out rules that are not
        relevant to the explanation. 
        '''
        if self.is_regression() and not self.is_probabilistic():
            warnings.warn('Regression explanations are not probabilistic. Filtering rules may not be effective.')
        positive_class = self.prediction['predict'] > 0.5
        initial_uncertainty = np.abs(self.prediction['high'] - self.prediction['low'])

        new_rules = self.__set_up_result()
        rules = self._get_rules() # pylint: disable=protected-access
        for rule in range(len(rules['rule'])):
            # filter out potential rules if include_potential is False
            if not include_potential and (
                                rules['predict_low'][rule] < 0.5 < rules['predict_high'][rule]
                            ):
                continue
            if make_super and (
                                positive_class
                                and rules['predict'][rule] < self.prediction['predict']
                                or not positive_class
                                and rules['predict'][rule] > self.prediction['predict']
                            ):
                continue
            if make_semi:
                if positive_class:
                    if rules['predict'][rule] < 0.5 or rules['predict'][rule] > self.prediction['predict']:
                        continue
                elif rules['predict'][rule] > 0.5 or rules['predict'][rule] < self.prediction['predict']:
                    continue
            if make_counter and (
                                positive_class
                                and rules['predict'][rule] > 0.5
                                or not positive_class
                                and rules['predict'][rule] < 0.5
                            ):
                continue
            # if only_ensured is True, filter out rules that lead to increased uncertainty
            if only_ensured and rules['predict_high'][rule] - rules['predict_low'][rule] > initial_uncertainty:
                continue
            # filter out rules that does not provide a different prediction
            if rules['base_predict_low'] == rules['predict_low'][rule] and rules['base_predict_high'] == rules['predict_high'][rule]:
                continue
            new_rules['predict'].append(rules['predict'][rule])
            new_rules['predict_low'].append(rules['predict_low'][rule])
            new_rules['predict_high'].append(rules['predict_high'][rule])
            new_rules['weight'].append(rules['weight'][rule])
            new_rules['weight_low'].append(rules['weight_low'][rule])
            new_rules['weight_high'].append(rules['weight_high'][rule])
            new_rules['value'].append(rules['value'][rule])
            new_rules['rule'].append(rules['rule'][rule])
            new_rules['feature'].append(rules['feature'][rule])
            new_rules['feature_value'].append(rules['feature_value'][rule])
            new_rules['is_conjunctive'].append(rules['is_conjunctive'][rule])
        new_rules['classes'] = rules['classes']
        if self._has_conjunctive_rules: # pylint: disable=protected-access
            self.__extracted_non_conjunctive_rules(new_rules)
        self.rules = new_rules
        return self

    # extract non-conjunctive rules
    def __extracted_non_conjunctive_rules(self, new_rules):
        self.conjunctive_rules = deepcopy(new_rules)
        new_rules['predict'] = [value for i, value in enumerate(new_rules['predict']) if not new_rules['is_conjunctive'][i]]
        new_rules['predict_low'] = [value for i, value in enumerate(new_rules['predict_low']) if not new_rules['is_conjunctive'][i]]
        new_rules['predict_high'] = [value for i, value in enumerate(new_rules['predict_high']) if not new_rules['is_conjunctive'][i]]
        new_rules['weight'] = [value for i, value in enumerate(new_rules['weight']) if not new_rules['is_conjunctive'][i]]
        new_rules['weight_low'] = [value for i, value in enumerate(new_rules['weight_low']) if not new_rules['is_conjunctive'][i]]
        new_rules['weight_high'] = [value for i, value in enumerate(new_rules['weight_high']) if not new_rules['is_conjunctive'][i]]
        new_rules['value'] = [value for i, value in enumerate(new_rules['value']) if not new_rules['is_conjunctive'][i]]
        new_rules['rule'] = [value for i, value in enumerate(new_rules['rule']) if not new_rules['is_conjunctive'][i]]
        new_rules['feature'] = [value for i, value in enumerate(new_rules['feature']) if not new_rules['is_conjunctive'][i]]
        new_rules['feature_value'] = [value for i, value in enumerate(new_rules['feature_value']) if not new_rules['is_conjunctive'][i]]
        new_rules['is_conjunctive'] = [value for i, value in enumerate(new_rules['is_conjunctive']) if not new_rules['is_conjunctive'][i]]

    def super_explanations(self, only_ensured=False, include_potential=False):
        '''
        This function returns the super-explanations from this alternative explanation. 
        Super-explanations are individual rules that support the predicted class. 
        
        Parameters
        ----------
        only_ensured : bool, default=False            
            The `only_ensured` parameter is a boolean flag that determines whether to return only ensured explanations, 
            i.e., explanations with a smaller confidence interval. If set to `True`, the function will return only ensured
            explanations. If set to `False`, the function will return all super-explanations. 
        include_potential : bool, default=False            
            The `include_potential` parameter is a boolean flag that determines whether to include potential explanations, 
            i.e., explanations with a confidence interval covering 0.5. If set to `True`, the function will include potential
            explanations. If set to `False`, the function will return only super-factual explanations. 
        
        Returns
        -------
        self : :class:`.AlternativeExplanation`
            Returns self filtered to only contain super-factual or super-potential explanations. 
        '''
        self.__filter_rules(only_ensured=only_ensured, make_super=True, include_potential=include_potential)
        self.__is_super_explanation = True
        return self

    def semi_explanations(self, only_ensured=False, include_potential=False):
        '''
        This function returns the semi-explanations from this alternative explanation. 
        Semi-explanations are individual rules that support the predicted class. 
        
        Parameters
        ----------
        only_ensured : bool, default=False            
            The `only_ensured` parameter is a boolean flag that determines whether to return only ensured explanations, 
            i.e., explanations with a smaller confidence interval. If set to `True`, the function will return only ensured
            explanations. If set to `False`, the function will return all semi-explanations. 
        include_potential : bool, default=False            
            The `include_potential` parameter is a boolean flag that determines whether to include potential explanations, 
            i.e., explanations with a confidence interval covering 0.5. If set to `True`, the function will include potential
            explanations. If set to `False`, the function will return only semi-factual explanations.
        
        Returns
        -------
        self : :class:`.AlternativeExplanation`
            Returns self filtered to only contain semi-factual or semi-potential explanations. 
        '''
        self.__filter_rules(only_ensured=only_ensured, make_semi=True, include_potential=include_potential)
        self.__is_semi_explanation = True
        return self

    def counter_explanations(self, only_ensured=False, include_potential=False):
        '''
        This function returns the counter-explanations from this alternative explanation. 
        Counter-explanations are individual rules that does not support the predicted class. 

        Parameters
        ----------
        only_ensured : bool, default=False            
            The `only_ensured` parameter is a boolean flag that determines whether to return only ensured explanations, 
            i.e., explanations with a smaller confidence interval. If set to `True`, the function will return only ensured
            explanations. If set to `False`, the function will return all counter-explanations. 
        include_potential : bool, default=False            
            The `include_potential` parameter is a boolean flag that determines whether to include potential explanations, 
            i.e., explanations with a confidence interval covering 0.5. If set to `True`, the function will include potential
            explanations. If set to `False`, the function will return only counter-factual explanations.

        Returns
        -------
        self : :class:`.AlternativeExplanation`
            Returns self filtered to only contain counter-factual or counter-potential explanations. 
        '''
        self.__filter_rules(only_ensured=only_ensured, make_counter=True, include_potential=include_potential)
        self.__is_counter_explanation = True
        return self

    def ensured_explanations(self, include_potential=False):
        '''
        This function returns the ensured explanations from this alternative explanation. 
        Ensured explanations are individual rules that have a smaller confidence interval. 
        
        Parameters
        ----------        
        include_potential : bool, default=False            
            The `include_potential` parameter is a boolean flag that determines whether to include potential explanations, 
            i.e., explanations with a confidence interval covering 0.5. If set to `True`, the function will include potential
            explanations. If set to `False`, the function will return only ensured factual explanations.

        Returns
        -------
        self : :class:`.AlternativeExplanation`
            Returns self filtered to only contain ensured explanations. 
        '''
        self.__filter_rules(only_ensured=True, include_potential=include_potential)
        return self

    # pylint: disable=too-many-locals
    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        # """adds conjunctive alternative rules

        # Args:
        #     n_top_features (int, optional): the number of most important alternative rules to try to combine into conjunctive rules. Defaults to 5.
        #     max_rule_size (int, optional): the maximum size of the conjunctions. Defaults to 2 (meaning `rule_one and rule_two`).

        # Returns:
        #     CalibratedExplanations: Returns a self reference, to allow for method chaining
        # """
        if max_rule_size >= 4:
            raise ValueError('max_rule_size must be 2 or 3')
        if max_rule_size < 2:
            return self
        if not self._has_rules:
            alternative = deepcopy(self._get_rules())
        else:
            alternative = deepcopy(self.rules)
        if self._has_conjunctive_rules:
            conjunctive = self.conjunctive_rules
        else:
            conjunctive = deepcopy(alternative)
        if self._has_conjunctive_rules:
            return self
        self.conjunctive_rules = []
        # pylint: disable=unsubscriptable-object, invalid-name
        threshold = None if self.y_threshold is None else self.y_threshold
        x_original = deepcopy(self.X_test)

        num_rules = len(alternative['rule'])
        predicted_class = alternative['classes']
        conjunctive['classes'] = predicted_class
        if n_top_features is None:
            n_top_features = num_rules
        top_conjunctives = self._rank_features(np.reshape(conjunctive['weight'], (len(conjunctive['weight']))),
                            width=np.reshape(np.array(conjunctive['weight_high']) - np.array(conjunctive['weight_low']),
                            (len(conjunctive['weight']))), num_to_show= np.min([num_rules, n_top_features]))

        covered_features = []
        covered_combinations = [conjunctive['feature'][i] for i in range(len(conjunctive['rule']))]
        for f1, cf1 in enumerate(alternative['feature']): # cf = factual feature
            covered_features.append(cf1)
            of1 = alternative['feature'][f1] # of = original feature
            rule_value1 = alternative['feature_value'][f1] \
                                if isinstance(alternative['feature_value'][f1], np.ndarray) \
                                else [alternative['feature_value'][f1]]
            for _, cf2 in enumerate(top_conjunctives): # cf = conjunctive feature
                if cf2 in covered_features:
                    continue
                rule_values = [rule_value1]
                original_features = [of1]
                of2 = conjunctive['feature'][cf2]
                if conjunctive['is_conjunctive'][cf2]:
                    if of1 in of2:
                        continue
                    for of in of2:
                        original_features.append(of)
                    for rule_value in conjunctive['feature_value'][cf2]:
                        rule_values.append(rule_value)
                else:
                    if of1 == of2:
                        continue
                    original_features.append(of2)
                    rule_values.append(conjunctive['feature_value'][cf2] \
                                    if isinstance(conjunctive['feature_value'][cf2], np.ndarray) \
                                    else [conjunctive['feature_value'][cf2]])
                skip = any(
                    np.all(np.sort(original_features) == ofs)
                    for ofs in covered_combinations
                )
                if skip:
                    continue
                covered_combinations.append(np.sort(original_features))

                rule_predict, rule_low, rule_high = self._predict_conjunctive(rule_values,
                                                                        original_features,
                                                                        deepcopy(x_original),
                                                                        threshold,
                                                                        predicted_class,
                                                                        bins=self.bin)
                conjunctive['predict'].append(rule_predict)
                conjunctive['predict_low'].append(rule_low)
                conjunctive['predict_high'].append(rule_high)
                conjunctive['weight'].append(rule_predict - self.prediction['predict'])
                conjunctive['weight_low'].append(rule_low - self.prediction['predict'] \
                            if rule_low != -np.inf else -np.inf)
                conjunctive['weight_high'].append(rule_high - self.prediction['predict'] \
                            if rule_high != np.inf else np.inf)
                conjunctive['value'].append(alternative['value'][f1] + '\n' + \
                                                conjunctive['value'][cf2])
                conjunctive['feature'].append(original_features)
                conjunctive['feature_value'].append(rule_values)
                conjunctive['rule'].append(alternative['rule'][f1] + ' & \n' + \
                                                conjunctive['rule'][cf2])
                conjunctive['is_conjunctive'].append(True)
        self.conjunctive_rules = conjunctive
        self._has_conjunctive_rules = True
        return self.add_conjunctions(n_top_features=n_top_features, max_rule_size=max_rule_size-1)

    def _is_lesser(self, rule_boundary, instance_value):
        return rule_boundary < instance_value

    # pylint: disable=consider-iterating-dictionary
    def plot(self, filter_top=None, **kwargs):
        '''The function `plot_alternative` plots the alternative explanation for a given instance in
        a dataset.

        Parameters
        ----------
        filter_top : int, default=10
            The `filter_top` parameter determines the number of top features to display in the
            plot. If set to `None`, it will show all the features. Otherwise, it will show the specified
            number of features, up to the total number of features available.
        show : bool, default=False
            A boolean parameter that determines whether the plot should be displayed or not. If set to
            True, the plot will be displayed. If set to False, the plot will not be displayed.
        filename : str, default=''
            The filename parameter is a string that represents the full path and filename of the plot
            image file that will be saved. If this parameter is not provided or is an empty string, the plot
            will not be saved as an image file.
        style : str, default='regular'
            The `style` parameter is a string that determines the style of the plot. Possible styles for :class:`.AlternativeExplanation`:
            * 'regular' - a regular plot with feature weights and uncertainty intervals (if applicable)
            * 'triangular' - a triangular plot for alternative explanations highlighting the interplay between the calibrated probability and the uncertainty intervals
        '''
        show = kwargs.get('show', False)
        filename = kwargs.get('filename', '')
        rnk_metric = kwargs.get('rnk_metric', 'ensured')
        rnk_weight = kwargs.get('rnk_weight', 0.5)
        if rnk_metric == 'uncertainty':
            rnk_weight = 1.0
            rnk_metric = 'ensured'

        alternative = self._get_rules() #get_explanation(index)
        self._check_preconditions()
        predict = self.prediction
        if len(filename) > 0:
            path = f'{os.path.dirname(filename)}/'
            filename = os.path.basename(filename)
            title, ext = os.path.splitext(filename)
            make_directory(path, save_ext=np.array([ext]))
            path = f'plots/{path}'
            save_ext = [ext]
        else:
            path = ''
            title = ''
            save_ext = []
        feature_predict = {'predict': alternative['predict'],
                            'low': alternative['predict_low'], 
                            'high': alternative['predict_high']}
        feature_weights = np.reshape(alternative['weight'],
                                        (len(alternative['weight'])))
        width = np.reshape(np.array(alternative['weight_high']) -
                            np.array(alternative['weight_low']),
                            (len(alternative['weight'])))
        num_rules = len(alternative['rule'])
        if filter_top is None:
            filter_top = num_rules
        num_to_show_ = np.min([num_rules, filter_top])
        if num_to_show_ <= 0:
            warnings.warn(f'The explanation has no rules to plot. The index of the instance is {self.index}')
            return

        if rnk_metric == 'feature_weight':
            features_to_plot = self._rank_features(feature_weights,
                                                width=width,
                                                num_to_show=num_to_show_)
        else:
            # Always rank base on predicted class
            prediction = alternative['predict']
            if self.get_mode() == 'classification' or self.is_thresholded():
                prediction = prediction if predict['predict'] > 0.5 else [1-p for p in prediction]
            ranking = calculate_metrics(uncertainty=[alternative['predict_high'][i]-alternative['predict_low'][i] for i in range(num_rules)],
                                                prediction=prediction,
                                                w=rnk_weight,
                                                metric=rnk_metric,
                                                )
            features_to_plot = self._rank_features(width=ranking,
                                                num_to_show=num_to_show_)

        if 'style' in kwargs and kwargs['style'] == 'triangular':
            proba = predict['predict']
            uncertainty = np.abs(predict['high'] - predict['low'])
            rule_proba = alternative['predict']
            rule_uncertainty = np.abs(np.array(alternative['predict_high']) - np.array(alternative['predict_low']))
            # Use list comprehension or NumPy array indexing to select elements
            selected_rule_proba = [rule_proba[i] for i in features_to_plot]
            selected_rule_uncertainty = [rule_uncertainty[i] for i in features_to_plot]

            _plot_triangular(self, proba, uncertainty, selected_rule_proba, selected_rule_uncertainty, num_to_show_, title=title, path=path, show=show, save_ext=save_ext)
            return

        column_names = alternative['rule']
        _plot_alternative(self, alternative['value'], predict, feature_predict, \
                                        features_to_plot, num_to_show=num_to_show_, \
                                        column_names=column_names, title=title, path=path, show=show, save_ext=save_ext)



class FastExplanation(CalibratedExplanation):
    """
    Fast Explanation class, representing shap-like explanations.
    """
    def __init__(self, calibrated_explanations, index, X_test, feature_weights, feature_predict, prediction, y_threshold=None, instance_bin=None):
        super().__init__(calibrated_explanations, index, X_test, {}, feature_weights, feature_predict, prediction, y_threshold, instance_bin)
        self._check_preconditions()
        self._get_rules()

    def __repr__(self):
        fast = self._get_rules()
        output = [
            f"{'Prediction':10} [{' Low':5}, {' High':5}]",
            f"   {fast['base_predict'][0]:5.3f}   [{fast['base_predict_low'][0]:5.3f}, {fast['base_predict_high'][0]:5.3f}]",
            f"{'Value':6}: {'Feature':40s} {'Weight':6} [{' Low':6}, {' High':6}]",
        ]
        feature_order = self._rank_features(fast['weight'],
                                width=np.array(fast['weight_high']) - np.array(fast['weight_low']),
                                num_to_show=len(fast['rule']))
        # feature_order = range(len(fast['rule']))
        output.extend(
            f"{fast['value'][f]:6}: {fast['rule'][f]:40s} {fast['weight'][f]:>6.3f} [{fast['weight_low'][f]:>6.3f}, {fast['weight_high'][f]:>6.3f}]"
            for f in reversed(feature_order)
        )
        # sum_weights = np.sum((fast['weight']))
        # sum_weights_low = np.sum((fast['weight_low']))
        # sum_weights_high = np.sum((fast['weight_high']))
        # output.append(f"{'Mean':6}: {'':40s} {sum_weights:>6.3f} [{sum_weights_low:>6.3f}, {sum_weights_high:>6.3f}]")
        return "\n".join(output) + "\n"

    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        '''The function `add_conjunctions` adds conjunctive rules to the factual or alternative
        explanations. The conjunctive rules are added to the `conjunctive_rules` attribute of the
        `CalibratedExplanations` object.
        '''
        warnings.warn('The add_conjunctions method is currently not supported for `FastExplanation`, making this call resulting in no change.')
        # pass

    def _is_lesser(self, rule_boundary, instance_value):
        pass

    def add_new_explanation(self, feature, rule_boundary):
        """creates an explanation for a feature or a set of features with user defined values

        Parameters
        ----------
        feature : int or str
            the feature to focus attention on
        rule_boundary: int, float, str, or categorical
            the value to define as rule condition

        Returns
        -------
        :class:`.FastExplanation`
        """
        warnings.warn('The add_new_explanation method is currently not supported for `FastExplanation`, making this call resulting in no change.')
        # pass

    def _check_preconditions(self):
        pass

    # pylint: disable=too-many-statements, too-many-branches
    def _get_rules(self):
        # """creates factual rules

        # Returns:
        #     List[Dict[str, List]]: a list of dictionaries containing the factual rules, one for each test instance
        # """
        if self._has_conjunctive_rules:
            return self.conjunctive_rules
        if self._has_rules:
            return self.rules
        self._has_rules = False
        # i = self.index
        instance = deepcopy(self.X_test)
        fast = {
            'base_predict': [],
            'base_predict_low': [],
            'base_predict_high': [],
            'predict': [],
            'predict_low': [],
            'predict_high': [],
            'weight': [],
            'weight_low': [],
            'weight_high': [],
            'value': [],
            'rule': [],
            'feature': [],
            'feature_value': [],
            'is_conjunctive': [],
            'classes': self.prediction['classes'],
        }
        fast['base_predict'].append(self.prediction['predict'])
        fast['base_predict_low'].append(self.prediction['low'])
        fast['base_predict_high'].append(self.prediction['high'])
        rules = self._define_conditions()
        for f,_ in enumerate(instance): # pylint: disable=invalid-name
            if self.prediction['predict'] == self.feature_predict['predict'][f]:
                continue
            fast['predict'].append(self.feature_predict['predict'][f])
            fast['predict_low'].append(self.feature_predict['low'][f])
            fast['predict_high'].append(self.feature_predict['high'][f])
            fast['weight'].append(self.feature_weights['predict'][f])
            fast['weight_low'].append(self.feature_weights['low'][f])
            fast['weight_high'].append(self.feature_weights['high'][f])
            if f in self._get_explainer().categorical_features:
                if self._get_explainer().categorical_labels is not None:
                    fast['value'].append(
                        self._get_explainer().categorical_labels[f][int(instance[f])])
                else:
                    fast['value'].append(str(instance[f]))
            else:
                fast['value'].append(str(np.around(instance[f],decimals=2)))
            fast['rule'].append(rules[f])
            fast['feature'].append(f)
            fast['feature_value'].append(None)
            fast['is_conjunctive'].append(False)
        self.rules = fast
        self._has_rules = True
        return self.rules


    def _define_conditions(self):
        # """defines the rule conditions for an instance

        # Args:
        #     instance (n_features,): a test instance

        # Returns:
        #     list[str]: a list of conditions for each feature in the instance
        # """
        self.conditions = []
        for f in range(self._get_explainer().num_features):
            rule = f"{self._get_explainer().feature_names[f]}"
            self.conditions.append(rule)
        return self.conditions


    def plot(self, filter_top=None, **kwargs):
        '''This function plots the factual explanation for a given instance using either probabilistic or
        regression plots.
        
        Parameters
        ----------
        filter_top : int, default=10
            The `filter_top` parameter determines the number of top features to display in the
            plot. If set to `None`, it will show all the features. Otherwise, it will show the specified
            number of features, up to the total number of features available.
        show : bool, default=False
            A boolean parameter that determines whether the plot should be displayed or not. If set to
            True, the plot will be displayed. If set to False, the plot will not be displayed.
        filename : str, default=''
            The filename parameter is a string that represents the full path and filename of the plot
            image file that will be saved. If this parameter is not provided or is an empty string, the plot
            will not be saved as an image file.
        uncertainty : bool, default=False
            The `uncertainty` parameter is a boolean flag that determines whether to plot the uncertainty
            intervals for the feature weights. If `uncertainty` is set to `True`, the plot will show the
            range of possible feature weights based on the lower and upper bounds of the uncertainty
            intervals. If `uncertainty` is set to `False`, the plot will only show the feature weights
        style : str, default='regular'
            The `style` parameter is a string that determines the style of the plot. Possible styles are for :class:`.FactualExplanation`:
            * 'regular' - a regular plot with feature weights and uncertainty intervals (if applicable)        
        '''
        show = kwargs.get('show', False)
        filename = kwargs.get('filename', '')
        uncertainty = kwargs.get('uncertainty', False)
        rnk_metric = kwargs.get('rnk_metric', 'feature_weight')
        rnk_weight = kwargs.get('rnk_weight', 0.5)
        if rnk_metric == 'uncertainty':
            rnk_weight = 1.0
            rnk_metric = 'ensured'

        factual = self._get_rules() #get_explanation(index)
        self._check_preconditions()
        predict = self.prediction
        num_features_to_show = len(factual['weight'])
        if filter_top is None:
            filter_top = num_features_to_show
        filter_top = np.min([num_features_to_show, filter_top])
        if filter_top <= 0:
            warnings.warn(f'The explanation has no rules to plot. The index of the instance is {self.index}')
            return

        if len(filename) > 0:
            path = f'{os.path.dirname(filename)}/'
            filename = os.path.basename(filename)
            title, ext = os.path.splitext(filename)
            make_directory(path, save_ext=np.array([ext]))
            path = f'plots/{path}'
            save_ext = [ext]
        else:
            path = ''
            title = ''
            save_ext = []
        if uncertainty:
            feature_weights = {'predict':factual['weight'],
                                'low':factual['weight_low'], 
                                'high':factual['weight_high']}
        else:
            feature_weights = factual['weight']
        width = np.reshape(np.array(factual['weight_high']) - np.array(factual['weight_low']),
                        (len(factual['weight'])))

        if rnk_metric == 'feature_weight':
            features_to_plot = self._rank_features(factual['weight'],
                                                width=width,
                                                num_to_show=filter_top)
        else:
            ranking = calculate_metrics(uncertainty=[factual['predict_high'][i]-factual['predict_low'][i] for i in range(len(factual['weight']))],
                                                prediction=factual['predict'],
                                                w=rnk_weight,
                                                metric=rnk_metric,
                                                )
            features_to_plot = self._rank_features(width=ranking,
                                                num_to_show=filter_top)

        column_names = factual['rule']
        if 'classification' in self._get_explainer().mode or self.is_thresholded():
            _plot_probabilistic(self, factual['value'], predict, feature_weights, features_to_plot,
                        filter_top, column_names, title=title, path=path, interval=uncertainty, show=show, idx=self.index,
                        save_ext=save_ext)
        else:
            _plot_regression(self, factual['value'], predict, feature_weights, features_to_plot,
                        filter_top, column_names, title=title, path=path, interval=uncertainty, show=show, idx=self.index,
                        save_ext=save_ext)
