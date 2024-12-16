# pylint: disable=unknown-option-value
# pylint: disable=too-many-lines, too-many-public-methods, invalid-name, too-many-positional-arguments, line-too-long
"""
Contains classes for storing and visualizing calibrated explanations.

Classes
-------
    - :class:`.CalibratedExplanations`
    - :class:`.AlternativeExplanations`
    - :class:`.FrozenCalibratedExplainer`
"""
from copy import deepcopy
from time import time
import numpy as np
from .explanation import FactualExplanation, AlternativeExplanation, FastExplanation
from ..utils.discretizers import EntropyDiscretizer, RegressorDiscretizer
from ..utils.helper import prepare_for_saving

class CalibratedExplanations:  # pylint: disable=too-many-instance-attributes
    """A class for storing and visualizing calibrated explanations.

    This class is created by :class:`.CalibratedExplainer` and provides methods for managing
    and accessing explanations for test instances.
    """

    def __init__(self, calibrated_explainer, X_test, y_threshold, bins) -> None:
        """A class for storing and visualizing calibrated explanations.

        This class is created by :class:`.CalibratedExplainer` and provides methods for managing
        and accessing explanations for test instances.
        
        Initialize the CalibratedExplanations object.
        
        Parameters
        ----------
        calibrated_explainer : CalibratedExplainer
            The calibrated explainer object.
        X_test : array-like
            The test data.
        y_threshold : float or tuple
            The threshold for regression explanations.
        bins : array-like
            The bins for conditional explanations.
        """
        self.calibrated_explainer = FrozenCalibratedExplainer(calibrated_explainer)
        self.X_test = X_test
        self.y_threshold = y_threshold
        self.low_high_percentiles = None
        self.explanations = []
        self.start_index = 0
        self.current_index = self.start_index
        self.end_index = len(X_test[:, 0])
        self.bins = bins
        self.total_explain_time = None

    def __iter__(self):
        """Return an iterator for the explanations."""
        self.current_index = self.start_index
        return self

    def __next__(self):
        """Return the next explanation."""
        if self.current_index >= self.end_index:
            raise StopIteration
        result = self.get_explanation(self.current_index)
        self.current_index += 1
        return result

    def __len__(self):
        """Return the number of explanations."""
        return len(self.X_test[:, 0])

    def __getitem__(self, key):
        """Return the explanation for the given key.
        
        In case the index key is an integer (or results in a single result), the function returns the explanation
        corresponding to the index. If the key is a slice or an integer or boolean list (or numpy array)
        resulting in more than one explanation, the function returns a new `CalibratedExplanations`
        object with the indexed explanations.
        """
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
                    new_.explanations = [
                        exp for exp, include in zip(self.explanations, key) if include
                    ]
                elif isinstance(key[0], int):
                    # Handle integer list indexing
                    new_.explanations = [self.explanations[i] for i in key]
            if len(new_.explanations) == 1:
                return new_.explanations[0]
            new_.start_index = 0
            new_.current_index = new_.start_index
            new_.end_index = len(new_.explanations)
            new_.bins = None if self.bins is None else [self.bins[e.index] for e in new_]
            new_.X_test = np.array([self.X_test[e.index, :] for e in new_])
            new_.y_threshold = (
                None
                if self.y_threshold is None
                else (
                    self.y_threshold
                    if np.isscalar(self.y_threshold)
                    else [self.y_threshold[e.index] for e in new_]
                )
            )
            for i, e in enumerate(new_):
                e.index = i
            return new_
        raise TypeError("Invalid argument type.")

    def __repr__(self) -> str:
        """Return the string representation of the CalibratedExplanations object."""
        explanations_str = "\n".join([str(e) for e in self.explanations])
        return f"CalibratedExplanations({len(self)} explanations):\n{explanations_str}"

    def __str__(self) -> str:
        """Return the string representation of the CalibratedExplanations object."""
        return self.__repr__()

    def _is_thresholded(self) -> bool:
        """Check if the explanations are thresholded."""
        return self.y_threshold is not None

    def _is_one_sided(self) -> bool:
        """Check if the explanations are one-sided."""
        if self.low_high_percentiles is None:
            return False
        return np.isinf(self.get_low_percentile()) or np.isinf(self.get_high_percentile())

    def get_confidence(self) -> float:
        """Return the confidence level of the explanations.
        
        This method calculates the confidence interval for regression tasks by determining the distance between the lower and upper percentiles. By default, these percentiles are set to 5 and 95.
        
        Returns
        -------
        float
            The difference between the high and low percentiles, representing the confidence interval.
                
        Notes
        -----        
        - This method is only applicable to regression tasks.
        - If the high percentile is infinite, the confidence is calculated as `100 - low_percentile`.
        - If the low percentile is infinite, the confidence is calculated as `high_percentile`.
        """
        if np.isinf(self.get_high_percentile()):
            return 100 - self.get_low_percentile()
        if np.isinf(self.get_low_percentile()):
            return self.get_high_percentile()
        return self.get_high_percentile() - self.get_low_percentile()

    def get_low_percentile(self) -> float:
        """Return the low percentile of the explanations.
        
        This method returns the first element of the `low_high_percentiles` attribute,
        which represents the lower bound of the percentile range for the explanation.

        Returns
        -------
        float
            The low percentile value of the explanation.
        """
        return self.low_high_percentiles[0]  # pylint: disable=unsubscriptable-object

    def get_high_percentile(self) -> float:
        """Return the high percentile of the explanations.
        
        Returns
        -------
        float
            The high percentile value of the explanation.
        """
        return self.low_high_percentiles[1]  # pylint: disable=unsubscriptable-object

    # pylint: disable=too-many-arguments
    def finalize(
        self,
        binned,
        feature_weights,
        feature_predict,
        prediction,
        instance_time=None,
        total_time=None,
    ) -> None:
        """
        Finalize the explanation by adding the binned data and the feature weights.
        
        Parameters
        ----------
        binned : array-like
            The binned data for the features.
        feature_weights : array-like
            The weights of the features.
        feature_predict : array-like
            The predicted values for the features.
        prediction : array-like
            The prediction values.
        instance_time : array-like, optional
            The time taken to explain each instance, by default None.
        total_time : float, optional
            The total time taken to explain all instances, by default None.

        Returns
        -------
        self : object
            Returns the instance of the class with explanations finalized.
        """
        for i, instance in enumerate(self.X_test):
            instance_bin = self.bins[i] if self.bins is not None else None
            if self._is_alternative():
                explanation = AlternativeExplanation(
                    self,
                    i,
                    instance,
                    binned,
                    feature_weights,
                    feature_predict,
                    prediction,
                    self.y_threshold,
                    instance_bin=instance_bin,
                )
            else:
                explanation = FactualExplanation(
                    self,
                    i,
                    instance,
                    binned,
                    feature_weights,
                    feature_predict,
                    prediction,
                    self.y_threshold,
                    instance_bin=instance_bin,
                )
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
    def finalize_fast(
        self, feature_weights, feature_predict, prediction, instance_time=None, total_time=None
    ) -> None:
        """
        Finalize the explanation by adding the binned data and the feature weights.
        
        Parameters
        ----------
        binned : array-like
            The binned data for the features.
        feature_weights : array-like
            The weights of the features.
        feature_predict : array-like
            The predicted values for the features.
        prediction : array-like
            The prediction values.
        instance_time : array-like, optional
            The time taken to explain each instance, by default None.
        total_time : float, optional
            The total time taken to explain all instances, by default None.

        Notes
        -----
        - This method iterates over the test instances and creates a `FastExplanation` object for each instance.
        - The `FastExplanation` object is initialized with the provided feature weights, predictions, and other relevant data.
        - The explanation time for each instance is recorded if `instance_time` is provided.
        - The total explanation time is calculated if `total_time` is provided.
        """
        for i, instance in enumerate(self.X_test):
            instance_bin = self.bins[i] if self.bins is not None else None
            explanation = FastExplanation(
                self,
                i,
                instance,
                feature_weights,
                feature_predict,
                prediction,
                self.y_threshold,
                instance_bin=instance_bin,
            )
            explanation.explain_time = instance_time[i] if instance_time is not None else None
            self.explanations.append(explanation)
        self.total_explain_time = time() - total_time if total_time is not None else None

    def _get_explainer(self):
        # """get the explainer object
        # """
        return self.calibrated_explainer

    def _get_rules(self):
        return [
            # pylint: disable=protected-access
            explanation._get_rules() for explanation in self.explanations
        ]

    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        """
        Add conjunctive rules to the explanations.

        The conjunctive rules are added to the `conjunctive_rules` attribute of the `CalibratedExplanations`
        object.
        
        Parameters
        ----------
        n_top_features : int, optional
            The number of most important factual rules to try to combine into conjunctive rules. Defaults to 5.
        max_rule_size : int, optional
            The maximum size of the conjunctions. Defaults to 2 (meaning `rule_one and rule_two`).
        
        Returns
        -------
        CalibratedExplanations
            Returns a self reference, to allow for method chaining.
        """
        for explanation in self.explanations:
            explanation.remove_conjunctions()
            explanation.add_conjunctions(n_top_features, max_rule_size)
        return self

    def reset(self):
        """Reset the explanations to their original state."""
        for explanation in self.explanations:
            explanation.reset()
        return self

    def remove_conjunctions(self):
        """Remove any conjunctive rules."""
        for explanation in self.explanations:
            explanation.remove_conjunctions()
        return self

    def get_explanation(self, index):
        """Return the explanation corresponding to the index.
        
        Parameters
        ----------
        index : int
            The index of the explanation to retrieve.
        
        Returns
        -------
        CalibratedExplanation
            The explanation at the specified index.
        
        Warnings
        --------
        Deprecated: This method is deprecated and may be removed in future versions. Use indexing instead.
        """
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
        return isinstance(
            self.calibrated_explainer.discretizer, (RegressorDiscretizer, EntropyDiscretizer)
        )

    # pylint: disable=too-many-arguments, too-many-locals, unused-argument
    def plot(
        self,
        index=None,
        filter_top=10,
        show=False,
        filename="",
        uncertainty=False,
        style="regular",
        rnk_metric=None,
        rnk_weight=0.5,
        style_override=None
    ):
        """
        Plot explanations for a given instance, with the option to show or save the plots.

        Parameters
        ----------
        index : int or None, default=None
            The index of the instance for which you want to plot the explanation. If None, the function will plot all the explanations.
        filter_top :  int or None, default=10
            The number of top features to display in the plot. If set to `None`, all the features will be shown.
        show : bool, default=False
            Determines whether the plots should be displayed immediately after they are generated.
        filename : str, default=''
            The full path and filename of the plot image file that will be saved. If empty, the plot will not be saved.
        uncertainty : bool, default=False
            Determines whether to include uncertainty information in the plots.
        style : str, default='regular'
            The style of the plot. Supported styles are 'regular' and 'triangular' (experimental).
        rnk_metric : str, default=None
            The metric used to rank the features. Supported metrics are 'ensured', 'feature_weight', and 'uncertainty'.
            If None, the default from the explanation class is used.
        rnk_weight : float, default=0.5
            The weight of the uncertainty in the ranking. Used with the 'ensured' ranking metric.
        
        See Also
        --------
        :meth:`.FactualExplanation.plot` : Refer to the docstring for plot in FactualExplanation for details on default ranking ('feature_weight').
        :meth:`.AlternativeExplanation.plot` : Refer to the docstring for plot in AlternativeExplanation for details on default ranking ('ensured').
        :meth:`.FastExplanation.plot` : Refer to the docstring for plot in FastExplanation for details on default ranking ('feature_weight').
        """
        if len(filename) > 0:
            path, filename, title, ext = prepare_for_saving(filename)

        if index is not None:
            if len(filename) > 0:
                filename = path + title + str(index) + ext
            self.get_explanation(index).plot(
                filter_top=filter_top,
                show=show,
                filename=filename,
                uncertainty=uncertainty,
                style=style,
                rnk_metric=rnk_metric,
                rnk_weight=rnk_weight,
                style_override=style_override
            )
        else:
            for i, explanation in enumerate(self.explanations):
                if len(filename) > 0:
                    filename = path + title + str(i) + ext
                explanation.plot(
                    filter_top=filter_top,
                    show=show,
                    filename=filename,
                    uncertainty=uncertainty,
                    style=style,
                    rnk_metric=rnk_metric,
                    rnk_weight=rnk_weight,
                    style_override=style_override
                )

    # pylint: disable=protected-access
    def as_lime(self, num_features_to_show=None):
        """Transform the explanations into LIME explanation objects.

        Returns
        -------
        list of lime.Explanation
            List of LIME explanation objects with the same values as the `CalibratedExplanations`.
        """
        _, lime_exp = self.calibrated_explainer._preload_lime()  # pylint: disable=protected-access
        exp = []
        for explanation in self.explanations:  # range(len(self.X_test[:,0])):
            tmp = deepcopy(lime_exp)
            tmp.intercept[1] = 0
            tmp.local_pred = explanation.prediction["predict"]
            if "regression" in self.calibrated_explainer.mode:
                tmp.predicted_value = explanation.prediction["predict"]
                tmp.min_value = np.min(self.calibrated_explainer.y_cal)
                tmp.max_value = np.max(self.calibrated_explainer.y_cal)
            else:
                tmp.predict_proba[0], tmp.predict_proba[1] = (
                    1 - explanation.prediction["predict"],
                    explanation.prediction["predict"],
                )

            feature_weights = explanation.feature_weights["predict"]
            num_to_show = (
                num_features_to_show
                if num_features_to_show is not None
                else self.calibrated_explainer.num_features
            )
            features_to_plot = explanation._rank_features(feature_weights, num_to_show=num_to_show)
            rules = explanation._define_conditions()
            for j, f in enumerate(features_to_plot[::-1]):  # pylint: disable=invalid-name
                tmp.local_exp[1][j] = (f, feature_weights[f])
            del tmp.local_exp[1][num_to_show:]
            tmp.domain_mapper.discretized_feature_names = rules
            tmp.domain_mapper.feature_values = explanation.X_test
            exp.append(tmp)
        return exp

    def as_shap(self):
        """Transform the explanations into a SHAP explanation object.

        Returns
        -------
        shap.Explanation
            SHAP explanation object with the same values as the explanation.
        """
        _, shap_exp = self.calibrated_explainer._preload_shap()  # pylint: disable=protected-access
        shap_exp.base_values = np.resize(shap_exp.base_values, len(self))
        shap_exp.values = np.resize(shap_exp.values, (len(self), len(self.X_test[0, :])))
        shap_exp.data = self.X_test
        for i, explanation in enumerate(self.explanations):  # range(len(self.X_test[:,0])):
            # shap_exp.base_values[i] = explanation.prediction['predict']
            for f in range(len(self.X_test[0, :])):
                shap_exp.values[i][f] = -explanation.feature_weights["predict"][f]
        return shap_exp


class AlternativeExplanations(CalibratedExplanations):
    """A class for storing and visualizing alternative explanations.

    Inherits from :class:`.CalibratedExplanations` and provides methods specific to
    alternative explanations, such as filtering explanations by type.
    """

    def super_explanations(self, only_ensured=False, include_potential=True):
        """
        Return a copy with only super-explanations.

        Super-explanations are individual rules with higher probability that support the predicted class.
        
        Parameters
        ----------
        only_ensured : bool, default=False
            Determines whether to return only ensured explanations.
        include_potential : bool, default=True
            Determines whether to include potential explanations in the super-explanations.
        
        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only super-factual or super-potential explanations.
        
        Notes
        -----
        Super-explanations are only available for `AlternativeExplanation` explanations.
        """
        for explanation in self.explanations:
            explanation.super_explanations(
                only_ensured=only_ensured, include_potential=include_potential
            )
        return self

    def semi_explanations(self, only_ensured=False, include_potential=True):
        """
        Return a copy with only semi-explanations.

        Semi-explanations are individual rules with lower probability that support the predicted class.
        
        Parameters
        ----------
        only_ensured : bool, default=False
            Determines whether to return only ensured explanations.
        include_potential : bool, default=True
            Determines whether to include potential explanations in the semi-explanations.
        
        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only semi-factual or semi-potential explanations.
        
        Notes
        -----
        Semi-explanations are only available for `AlternativeExplanation` explanations.
        """
        for explanation in self.explanations:
            explanation.semi_explanations(
                only_ensured=only_ensured, include_potential=include_potential
            )
        return self

    def counter_explanations(self, only_ensured=False, include_potential=True):
        """
        Return a copy with only counter-explanations.

        Counter-explanations are individual rules that do not support the predicted class.

        Parameters
        ----------
        only_ensured : bool, default=False
            Determines whether to return only ensured explanations.
        include_potential : bool, default=True
            Determines whether to include potential explanations in the counter-explanations.
        
        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only counter-factual or counter-potential explanations.
        
        Notes
        -----
        Counter-explanations are only available for `AlternativeExplanation` explanations.
        """
        for explanation in self.explanations:
            explanation.counter_explanations(
                only_ensured=only_ensured, include_potential=include_potential
            )
        return self

    def ensured_explanations(self):
        """
        Return a copy with only ensured explanations.

        Ensured explanations are individual rules that have a smaller confidence interval.
        
        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only ensured explanations.
        """
        for explanation in self.explanations:
            explanation.ensured_explanations()
        return self


class FrozenCalibratedExplainer:
    """A class that wraps an explainer to provide a read-only interface.

    Prevents modification of the underlying explainer, ensuring its state remains unchanged.
    """

    def __init__(self, explainer):
        """Initialize a new instance of the FrozenCalibratedExplainer class.
        
        Parameters
        ----------
        explainer : CalibratedExplainer
            The explainer to be wrapped.
        """
        self._explainer = deepcopy(explainer)

    @property
    def X_cal(self):
        """
        Retrieves the calibrated feature matrix from the underlying explainer.
        
        This property provides access to the feature matrix used in the explainer, allowing users to understand the data being analyzed.
        
        Returns
        -------
            numpy.ndarray: The calibrated feature matrix.
        """
        return self._explainer.X_cal

    @property
    def y_cal(self):
        """
        Retrieves the calibrated target values from the underlying explainer.

        This property provides access to the target values used in the explainer, allowing users to understand the data being analyzed.
        
        Returns
        -------
            numpy.ndarray: The calibrated target values.
        """
        return self._explainer.y_cal

    @property
    def num_features(self):
        """
        Retrieves the number of features in the dataset.

        This property provides access to the count of features that the underlying explainer is using. 
        It is useful for understanding the dimensionality of the data being analyzed.

        Returns
        -------
            int: The number of features in the dataset.
        """
        return self._explainer.num_features

    @property
    def categorical_features(self):
        """
        Retrieves the indices of categorical features from the underlying explainer.
        
        This property provides access to the indices of categorical features used in the explainer, allowing users to understand the data being analyzed.
        
        Returns
        -------
            list: The indices of categorical features.
        """
        return self._explainer.categorical_features

    @property
    def categorical_labels(self):
        """
        Retrieves the labels for categorical features from the underlying explainer.
        
        This property provides access to the labels for categorical features used in the explainer, allowing users to understand the data being analyzed.
        
        Returns
        -------
            list: The labels for categorical features.
        """
        return self._explainer.categorical_labels

    @property
    def feature_values(self):
        """
        Retrieves the unique values for each feature from the underlying explainer.
        
        This property provides access to the unique values for each feature used in the explainer, allowing users to understand the data being analyzed.
        
        Returns
        -------
            list: The unique values for each feature.
        """
        return self._explainer.feature_values

    @property
    def feature_names(self):
        """
        Retrieves the names of the features from the underlying explainer.
        
        This property provides access to the names of the features used in the explainer, allowing users to understand the data being analyzed.
        
        Returns
        -------
            list: The names of the features.
        """
        return self._explainer.feature_names

    @property
    def class_labels(self):
        """
        Retrieves the labels for the classes from the underlying explainer.
            
        This property provides access to the labels for the classes used in the explainer, allowing users to understand the data being analyzed.
        
        Returns
        -------
            list: The labels for the classes.
        """
        return self._explainer.class_labels

    @property
    def assign_threshold(self):
        """
        Retrieves the threshold for assigning class labels from the underlying explainer.
        
        This property provides access to the threshold used for assigning class labels in the explainer, allowing users to understand the data being analyzed.
        
        Returns
        -------
            float: The threshold for assigning class labels.
        """
        return self._explainer.assign_threshold

    @property
    def sample_percentiles(self):
        """
        Retrieves the sample percentiles from the underlying explainer.

        This property provides access to the percentiles of the samples used in the explainer, 
        allowing users to understand the distribution of the data being analyzed.

        Returns
        -------
            list: The sample percentiles as a list.
        """
        return self._explainer.sample_percentiles

    @property
    def mode(self):
        """
        Retrieves the mode of the explainer from the underlying explainer.
        
        This property provides access to the mode of the explainer, allowing users to understand the type of problem being analyzed.
        
        Returns
        -------
            str: The mode of the explainer.
        """
        return self._explainer.mode

    @property
    def is_multiclass(self):
        """
        Retrieves a boolean indicating if the problem is multiclass from the underlying explainer.
        
        This property provides access to a boolean value indicating if the problem is multiclass, allowing users to understand the type of problem being analyzed.
        
        Returns
        -------
            bool: True if the problem is multiclass, False otherwise.
        """
        return self._explainer.is_multiclass

    @property
    def discretizer(self):
        """
        Retrieves the discretizer used by the explainer from the underlying explainer.
        
        This property provides access to the discretizer used by the explainer, allowing users to understand the discretization process.
        
        Returns
        -------
            Discretizer: The discretizer used by the explainer.
        """
        return self._explainer.discretizer

    @property
    def _discretize(self):
        """
        Retrieves the discretize function from the underlying explainer.
        
        This property provides access to the discretize function used by the explainer, allowing users to understand the discretization process.
        
        Returns
        -------
            function: The discretize function used by the explainer.
        """
        return self._explainer._discretize # pylint: disable=protected-access

    @property
    def rule_boundaries(self):
        """
        Retrieves the boundaries for rules in the explainer from the underlying explainer.
        
        This property provides access to the boundaries for rules used in the explainer, allowing users to understand the discretization process.
        
        Returns
        -------
            list: The boundaries for rules in the explainer.
        """
        return self._explainer.rule_boundaries

    @property
    def learner(self):
        """
        Retrieves the learner associated with the explainer from the underlying explainer.
        
        This property provides access to the learner associated with the explainer, allowing users to understand the learning process.
        
        Returns
        -------
            object: The learner associated with the explainer.
        """
        return self._explainer.learner

    @property
    def difficulty_estimator(self):
        """
        Retrieves the estimator for difficulty levels from the underlying explainer.
        
        This property provides access to the estimator for difficulty levels used in the explainer, allowing users to understand the learning process.
        
        Returns
        -------
            object: The estimator for difficulty levels.
        """
        return self._explainer.difficulty_estimator

    @property
    def _predict(self):
        """
        Retrieves the predict function from the underlying explainer.
        
        This property provides access to the predict function used by the explainer, allowing users to understand the prediction process.
        
        Returns
        -------
            function: The predict function used by the explainer.
        """
        return self._explainer._predict # pylint: disable=protected-access

    @property
    def _preload_lime(self):
        """
        Retrieves the preload_lime function from the underlying explainer.
        
        This property provides access to the preload_lime function used by the explainer, allowing users to understand the prediction process.
        
        Returns
        -------
            function: The preload_lime function used by the explainer.
        """
        return self._explainer._preload_lime # pylint: disable=protected-access

    @property
    def _preload_shap(self):
        """
        Retrieves the preload_shap function from the underlying explainer.
        
        This property provides access to the preload_shap function used by the explainer, allowing users to understand the prediction process.
        
        Returns
        -------
            function: The preload_shap function used by the explainer.
        """
        return self._explainer._preload_shap # pylint: disable=protected-access

    def __setattr__(self, key, value):
        """Prevent modification of attributes except for '_explainer'."""
        if key == '_explainer':
            super().__setattr__(key, value)
        else:
            raise AttributeError("Cannot modify frozen instance")
