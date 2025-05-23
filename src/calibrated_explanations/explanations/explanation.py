# pylint: disable=unknown-option-value
# pylint: disable=too-many-lines, too-many-arguments, invalid-name, too-many-positional-arguments, line-too-long
"""This module contains classes for storing and visualizing individual calibrated explanations.

Classes:
    :class:`.CalibratedExplanation`:
        Abstract base class for calibrated explanations. Defines the interface and shared functionality for different types of explanations.

    :class:`.FactualExplanation`:
        Provides factual explanations for a given instance, highlighting features that contribute to the model's prediction.

    :class:`.AlternativeExplanation`:
        Offers alternative explanations by exploring how changes to feature values could alter the model's prediction.

    :class:`.FastExplanation`:
        Represents fast explanations, enabling efficient interpretation of model behavior for large datasets.
"""
import contextlib
import warnings
# from dataclasses import dataclass
from copy import deepcopy
from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
from pandas import Categorical
from ..utils.discretizers import (
    BinaryEntropyDiscretizer,
    EntropyDiscretizer,
    RegressorDiscretizer,
    BinaryRegressorDiscretizer,
)
from ..utils.helper import calculate_metrics, prepare_for_saving
from .._plots import _plot_alternative, _plot_probabilistic, _plot_regression, _plot_triangular

# @dataclass
# class PredictionInterval:
#     """A dataclass representing a prediction interval for a single feature.

#     Attributes
#     ----------
#     predict: float
#         The model's prediction for this feature
#     low: float
#         The lower bound of the prediction interval
#     high: float
#         The upper bound of the prediction interval
#     """
#     predict: float
#     low: float
#     high: float

# @dataclass
# class FeatureRule:
#     """A dataclass representing a rule for a single feature in an explanation.

#     Attributes
#     ----------
#     weight : PredictionInterval
#         The weight/importance of this feature rule, containing prediction and interval values
#     prediction : PredictionInterval
#         The model's prediction and interval for this feature rule
#     instance_prediction : PredictionInterval
#         The binned prediction data for this feature rule
#     current_bin : int
#         The bin index containing the current feature value
#     rule : str
#         String representation of the rule
#     feature : int
#         Index of the feature this rule applies to
#     feature_value : float
#         The actual value of the feature
#     is_conjunctive : bool
#         Whether this rule is part of a conjunction
#     value_str : str
#         String representation of the feature value
#     """
#     weight: PredictionInterval
#     prediction: PredictionInterval
#     instance_prediction: PredictionInterval # from binned data
#     current_bin: int
#     rule: str
#     feature: int
#     feature_value: float
#     is_conjunctive: bool
#     value_str: str

# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-arguments
class CalibratedExplanation(ABC):
    """Abstract base class for storing and visualizing calibrated explanations.

    This class defines the interface and shared functionality for different types of calibrated explanations.
    """

    def __init__(
        self,
        calibrated_explanations,
        index,
        X_test,
        binned,
        feature_weights,
        feature_predict,
        prediction,
        y_threshold=None,
        instance_bin=None,
    ):
        """Abstract base class for storing and visualizing calibrated explanations.

        This class defines the interface and shared functionality for different types of calibrated explanations.
        
        Initialize a CalibratedExplanation instance.

        Parameters
        ----------
        calibrated_explanations : :class:`.CalibratedExplanations`
            The parent :class:`.CalibratedExplanations` object.
        index : int
            The index of the instance being explained.
        X_test : array-like
            The test dataset containing the instances to be explained.
        binned : dict
            A mapping of binned feature values.
        feature_weights : dict
            A mapping of feature weights.
        feature_predict : dict
            A mapping of feature predictions.
        prediction : dict
            A mapping containing the prediction results.
        y_threshold : float or tuple, optional
            The threshold for binary classification or regression explanations.
        instance_bin : int, optional
            The bin index of the instance.
        """
        binned = MappingProxyType(binned)
        feature_weights = MappingProxyType(feature_weights)
        feature_predict = MappingProxyType(feature_predict)
        prediction = MappingProxyType(prediction)
        self.calibrated_explanations = calibrated_explanations
        self.index = index
        self.X_test = X_test
        self.binned = {}
        self.feature_weights = {}
        self.feature_predict = {}
        self.prediction = {}
        for key in binned.keys():
            self.binned[key] = binned[key][index]
        for key in feature_weights.keys():
            self.feature_weights[key] = feature_weights[key][index]
            self.feature_predict[key] = feature_predict[key][index]
        for key in prediction.keys():
            self.prediction[key] = prediction[key][index]
        self.y_threshold = (
            y_threshold
            if np.isscalar(y_threshold) or isinstance(y_threshold, tuple)
            else None if y_threshold is None else y_threshold[index]
        )

        self.conditions = []
        self.rules = None
        self.conjunctive_rules = None
        self._has_rules = False
        self._has_conjunctive_rules = False
        self.bin = [instance_bin] if instance_bin is not None else None
        self.explain_time = None
        # reduce dependence on Explainer class
        if not isinstance(self._get_explainer().y_cal, Categorical):
            self.y_minmax = [
                np.min(self._get_explainer().y_cal),
                np.max(self._get_explainer().y_cal),
            ]
        else:
            self.y_minmax = [0, 0]
        self.focus_columns = None

    def __len__(self):
        """Return the number of rules in the explanation."""
        return len(self._get_rules()["rule"])

    def get_mode(self):
        """Return the mode of the explanation ('classification' or 'regression')."""
        return self._get_explainer().mode

    def get_class_labels(self):
        """Return the class labels."""
        return self._get_explainer().class_labels

    def is_multiclass(self):
        """Determine if the explanation is multiclass."""
        return self._get_explainer().is_multiclass()

    def _get_explainer(self):
        """Return the explainer object."""
        return self.calibrated_explanations._get_explainer()  # pylint: disable=protected-access

    def _rank_features(self, feature_weights=None, width=None, num_to_show=None):
        """Rank the features based on their weights.

        Parameters
        ----------
        feature_weights : dict, optional
            A mapping of feature weights.
        width : dict, optional
            A mapping of feature widths.
        num_to_show : int, optional
            The number of features to show.

        Returns
        -------
        list
            The sorted indices of the features.
        """
        assert (
            feature_weights is not None or width is not None
        ), "Either feature_weights or width (or both) must not be None"
        num_features = len(feature_weights) if feature_weights is not None else len(width)
        if num_to_show is None or num_to_show > num_features:
            num_to_show = num_features
        # handle case where there are same weight but different uncertainty
        if feature_weights is not None and width is not None:
            # get the indeces by first sorting on the absolute value of the
            # feature_weight and then on the width
            sorted_indices = [
                i
                for i, x in sorted(
                    enumerate(list(zip(np.abs(feature_weights), width))),
                    key=lambda x: (x[1][0], x[1][1]),
                )
            ]
            return sorted_indices[-num_to_show:]  # pylint: disable=invalid-unary-operand-type
        if width is not None:
            sorted_indices = np.argsort(width)
            return sorted_indices[-num_to_show:]  # pylint: disable=invalid-unary-operand-type
        sorted_indices = np.argsort(np.abs(feature_weights))
        return sorted_indices[-num_to_show:]  # pylint: disable=invalid-unary-operand-type

    def is_one_sided(self) -> bool:
        """Test if a regression explanation is one-sided.

        Returns
        -------
            bool: True if one of the low or high percentiles is infinite
        """
        if self.calibrated_explanations.low_high_percentiles is None:
            return False
        return np.isinf(self.calibrated_explanations.get_low_percentile()) or np.isinf(
            self.calibrated_explanations.get_high_percentile()
        )

    def is_thresholded(self) -> bool:
        """Check if the explanation is thresholded.

        Returns
        -------
            bool: True if the y_threshold is not None
        """
        return self.y_threshold is not None

    def is_regression(self) -> bool:
        """Check if the explanation is for regression.

        Returns
        -------
            bool: True if mode is 'regression'
        """
        return "regression" in self._get_explainer().mode

    def is_probabilistic(self) -> bool:
        """Check if the explanation is probabilistic.

        Returns
        -------
            bool: True if mode is 'classification' or is_thresholded and is_regression are True
        """
        return "classification" in self._get_explainer().mode or (
            self.is_regression() and self.is_thresholded()
        )

    @abstractmethod
    def __repr__(self):
        """Return a string representation of the explanation."""

    @abstractmethod
    def plot(self, filter_top=None, **kwargs):
        """
        Plot the explanation.

        Parameters
        ----------
        filter_top : int, optional
            The number of top features to display.
        **kwargs : dict
            Additional plotting arguments. See each subclass.
        
        See Also
        --------
        :meth:`.FactualExplanation.plot` : Refer to the docstring for plot in FactualExplanation for details.
        :meth:`.AlternativeExplanation.plot` : Refer to the docstring for plot in AlternativeExplanation for details.
        :meth:`.FastExplanation.plot` : Refer to the docstring for plot in FastExplanation for details.
        """

    @abstractmethod
    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        """
        Add conjunctive rules to the explanation.

        Parameters
        ----------
        n_top_features : int, optional
            Number of top features to combine.
        max_rule_size : int, optional
            Maximum size of the conjunctions.

        Returns
        -------
        :class:`.CalibratedExplanation`
        """

    @abstractmethod
    def _check_preconditions(self):
        pass

    @abstractmethod
    def _get_rules(self):
        pass

    def reset(self):
        """Reset the explanation to its original state."""
        self._has_rules = False
        self._get_rules()
        return self

    def remove_conjunctions(self):
        """Remove any conjunctive rules."""
        self._has_conjunctive_rules = False
        return self

    def _define_conditions(self):
        """
        Define the rule conditions for an instance.

        Returns
        -------
        list[str]
            A list of conditions for each feature in the instance.
        """
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

    def _predict_conjunctive(
        self,
        rule_value_set,
        original_features,
        perturbed,
        threshold,
        predicted_class,
        bins=None,
    ):
        """
        Calculate the prediction for a conjunctive rule.

        Parameters
        ----------
        rule_value_set : list
            The set of rule values.
        original_features : list
            The original feature indices.
        perturbed : array-like
            The perturbed dataset.
        threshold : float
            The threshold for classification or regression.
        predicted_class : int
            The predicted class label.
        bins : array-like, optional
            The bins for discretization.

        Returns
        -------
        tuple
            The predicted value, lower bound, upper bound, and count.
        """
        assert len(original_features) >= 2, "Conjunctive rules require at least two features"
        rule_predict, rule_low, rule_high, rule_count = 0, 0, 0, 0
        of1, of2, of3 = 0, 0, 0
        rule_value1, rule_value2, rule_value3 = 0, 0, 0
        if len(original_features) == 2:
            of1, of2 = original_features[0], original_features[1]
            rule_value1, rule_value2 = rule_value_set[0], rule_value_set[1]
        elif len(original_features) >= 3:
            of1, of2, of3 = original_features[0], original_features[1], original_features[2]
            rule_value1, rule_value2, rule_value3 = (
                rule_value_set[0],
                rule_value_set[1],
                rule_value_set[2],
            )
        for value_1 in rule_value1:
            perturbed[of1] = value_1
            for value_2 in rule_value2:
                perturbed[of2] = value_2
                if len(original_features) >= 3:
                    for value_3 in rule_value3:
                        perturbed[of3] = value_3
                        # pylint: disable=protected-access
                        p_value, low, high, _ = self._get_explainer()._predict(
                            perturbed.reshape(1, -1),
                            threshold=threshold,
                            low_high_percentiles=self.calibrated_explanations.low_high_percentiles,
                            classes=predicted_class,
                            bins=bins,
                        )
                        rule_predict += p_value[0]
                        rule_low += low[0]
                        rule_high += high[0]
                        rule_count += 1
                else:
                    p_value, low, high, _ = self._get_explainer()._predict( # pylint: disable=protected-access
                        perturbed.reshape(1, -1),
                        threshold=threshold,
                        low_high_percentiles=self.calibrated_explanations.low_high_percentiles,
                        classes=predicted_class,
                        bins=bins,
                    )
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
    def add_new_rule_condition(self, feature, rule_boundary):
        """
        Create a new rule condition for a numerical feature.

        Parameters
        ----------
        feature : int or str
            The feature index or name.
        rule_boundary : int or float
            The value to define as rule condition.

        Returns
        -------
        :class:`.CalibratedExplanation`

        Notes
        -----
        The function will return the same explanation if the rule is already included or if the feature is categorical.

        No implementation is provided for the :class:`.FastExplanation` class.
        """
        try:
            f = (
                feature
                if isinstance(feature, int)
                else self._get_explainer().feature_names.index(feature)
            )
        except ValueError:
            warnings.warn(f"Feature {feature} not found")
            return self
        if (
            self._get_explainer().categorical_features is not None
            and f in self._get_explainer().categorical_features
        ):
            warnings.warn("Alternatives for all categorical features are already included")
            return self

        X_copy = np.array(self.X_test, copy=True)
        is_lesser = self._is_lesser(rule_boundary, X_copy[f])
        new_rule = self._get_rules()
        rule = self._get_rule_str(is_lesser, f, rule_boundary)
        if np.any([new_rule["rule"][i] == rule for i in range(len(new_rule["rule"]))]):
            warnings.warn("Rule already included")
            return self

        threshold = self.y_threshold
        perturbed_threshold = self._get_explainer().assign_threshold(threshold)
        perturbed_bins = np.empty((0,)) if self.bin is not None else None
        perturbed_X = np.empty((0, self._get_explainer().num_features))
        perturbed_feature = np.empty((0, 4))  # (feature, instance, bin_index, is_lesser)
        perturbed_class = np.empty((0,), dtype=int)

        cal_X_f = self._get_explainer().X_cal[:, f]
        feature_values = np.unique(np.array(cal_X_f))
        sample_percentiles = self._get_explainer().sample_percentiles

        if is_lesser:
            if not np.any(feature_values < rule_boundary):
                warnings.warn(
                    f"Lowest feature value for feature {feature} is {np.min(feature_values)}"
                )
                return self
            values = np.percentile(cal_X_f[cal_X_f < rule_boundary], sample_percentiles)
            covered = np.percentile(cal_X_f[cal_X_f >= rule_boundary], sample_percentiles)
        else:
            if not np.any(feature_values > rule_boundary):
                warnings.warn(
                    f"Highest feature value for feature {feature} is {np.max(feature_values)}"
                )
                return self
            values = np.percentile(cal_X_f[cal_X_f > rule_boundary], sample_percentiles)
            covered = np.percentile(cal_X_f[cal_X_f <= rule_boundary], sample_percentiles)

        for value in values:
            X_local = np.reshape(X_copy, (1, -1))
            X_local[0, f] = value
            perturbed_X = np.concatenate((perturbed_X, np.array(X_local)))
            perturbed_feature = np.concatenate((perturbed_feature, [(f, 0, None, is_lesser)]))
            perturbed_bins = (
                np.concatenate((perturbed_bins, self.bin)) if self.bin is not None else None
            )
            perturbed_class = np.concatenate(
                (perturbed_class, np.array([self.prediction["classes"]]))
            )
            if isinstance(threshold, tuple):
                perturbed_threshold = threshold
            elif threshold is None:
                perturbed_threshold = None
            else:
                perturbed_threshold = np.concatenate((perturbed_threshold, threshold))

        for value in covered:
            X_local = np.reshape(X_copy, (1, -1))
            X_local[0, f] = value
            perturbed_X = np.concatenate((perturbed_X, np.array(X_local)))
            perturbed_feature = np.concatenate((perturbed_feature, [(f, 0, None, None)]))
            perturbed_bins = (
                np.concatenate((perturbed_bins, self.bin)) if self.bin is not None else None
            )
            perturbed_class = np.concatenate(
                (perturbed_class, np.array([self.prediction["classes"]]))
            )
            if isinstance(threshold, tuple):
                perturbed_threshold = threshold
            elif threshold is None:
                perturbed_threshold = None
            else:
                perturbed_threshold = np.concatenate((perturbed_threshold, threshold))

        # pylint: disable=protected-access
        predict, low, high, _ = self._get_explainer()._predict(
            perturbed_X,
            threshold=perturbed_threshold,
            low_high_percentiles=self.calibrated_explanations.low_high_percentiles,
            classes=perturbed_class,
            bins=perturbed_bins,
        )
        instance_predict = [
            predict[i] for i in range(len(predict)) if perturbed_feature[i][3] is None
        ]
        rule_predict = [
            predict[i] for i in range(len(predict)) if perturbed_feature[i][3] is not None
        ]
        rule_low = [low[i] for i in range(len(low)) if perturbed_feature[i][3] is not None]
        rule_high = [high[i] for i in range(len(high)) if perturbed_feature[i][3] is not None]

        # skip if identical to original
        if self.prediction["low"] == np.mean(rule_low) and self.prediction["high"] == np.mean(
            rule_high
        ):
            warnings.warn("The alternative explanation is identical to the original explanation", UserWarning)
            return self
        new_rule["predict"].append(np.mean(rule_predict))
        new_rule["predict_low"].append(np.mean(rule_low))
        new_rule["predict_high"].append(np.mean(rule_high))
        new_rule["weight"].append(np.mean(rule_predict) - np.mean(instance_predict))
        new_rule["weight_low"].append(
            np.mean(rule_low) - np.mean(instance_predict) if rule_low != -np.inf else rule_low
        )
        new_rule["weight_high"].append(
            np.mean(rule_high) - np.mean(instance_predict) if rule_high != np.inf else rule_high
        )
        new_rule["value"].append(str(np.around(X_copy, decimals=2)))
        new_rule["feature"].append(f)
        new_rule["feature_value"].append(self.binned["rule_values"][f][0][0])
        new_rule["is_conjunctive"].append(False)

        new_rule["rule"].append(rule)
        self.rules = new_rule
        return self

    def _get_rule_str(self, is_lesser, feature, rule_boundary):
        """Get the rule string for the explanation.

        Parameters
        ----------
        is_lesser : bool
            Whether the rule is a lesser condition.
        feature : str
            The feature name.
        rule_boundary : float
            The rule boundary value.

        Returns
        -------
        str
            The rule string.
        """
        if is_lesser:
            return f"{self._get_explainer().feature_names[feature]} < {rule_boundary:.2f}"
        return f"{self._get_explainer().feature_names[feature]} > {rule_boundary:.2f}"


# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-arguments
class FactualExplanation(CalibratedExplanation):
    """Class for storing and visualizing factual explanations.

    Provides factual explanations for a given instance, highlighting features that contribute to the model's prediction.
    """

    def __init__(
        self,
        calibrated_explanations,
        index,
        X_test,
        binned,
        feature_weights,
        feature_predict,
        prediction,
        y_threshold=None,
        instance_bin=None,
    ):
        """Class for storing and visualizing factual explanations.

        Provides factual explanations for a given instance, highlighting features that contribute to the model's prediction.
    
        Initialize a FactualExplanation instance.

        Parameters
        ----------
        calibrated_explanations : CalibratedExplanations
            The parent CalibratedExplanations object.
        index : int
            The index of the instance being explained.
        X_test : array-like
            The test dataset containing the instances to be explained.
        binned : dict
            A mapping of binned feature values.
        feature_weights : dict
            A mapping of feature weights.
        feature_predict : dict
            A mapping of feature predictions.
        prediction : dict
            A mapping containing the prediction results.
        y_threshold : float or tuple, optional
            The threshold for binary classification or regression explanations.
        instance_bin : int, optional
            The bin index of the instance.
        """
        super().__init__(
            calibrated_explanations,
            index,
            X_test,
            binned,
            feature_weights,
            feature_predict,
            prediction,
            y_threshold,
            instance_bin,
        )
        self._check_preconditions()
        self._get_rules()

    def __repr__(self):
        """Return a string representation of the factual explanation."""
        factual = self._get_rules()
        output = [
            f"{'Prediction':10} [{' Low':5}, {' High':5}]",
            f"{factual['base_predict'][0]:5.3f} [{factual['base_predict_low'][0]:5.3f}, {factual['base_predict_high'][0]:5.3f}]",
            f"{'Value':6}: {'Feature':40s} {'Weight':6} [{' Low':6}, {' High':6}]",
        ]
        feature_order = self._rank_features(
            factual["weight"],
            width=np.array(factual["weight_high"]) - np.array(factual["weight_low"]),
            num_to_show=len(factual["rule"]),
        )
        output.extend(
            f"{factual['value'][f]:6}: {factual['rule'][f]:40s} {factual['weight'][f]:>6.3f} [{factual['weight_low'][f]:>6.3f}, {factual['weight_high'][f]:>6.3f}]"
            for f in reversed(feature_order)
        )
        return "\n".join(output) + "\n"

    def _check_preconditions(self):
        if self.is_regression():
            if not isinstance(self._get_explainer().discretizer, BinaryRegressorDiscretizer):
                warnings.warn(
                    "Factual explanations for regression recommend using the binaryRegressor "
                    + "discretizer. Consider extracting factual explanations using "
                    + "`explainer.explain_factual(test_set)`"
                )
        elif not isinstance(self._get_explainer().discretizer, BinaryEntropyDiscretizer):
            warnings.warn(
                "Factual explanations for classification recommend using the "
                + "binaryEntropy discretizer. Consider extracting factual "
                + "explanations using `explainer.explain_factual(test_set)`"
            )

    def _get_rules(self):
        """
        Create factual rules.

        Returns
        -------
        List[Dict[str, List]]
            A list of dictionaries containing the factual rules.
        """
        if self._has_conjunctive_rules:
            return self.conjunctive_rules
        if self._has_rules:
            return self.rules
        self._has_rules = False
        # i = self.index
        instance = np.array(self.X_test, copy=True)
        factual = {
            "base_predict": [],
            "base_predict_low": [],
            "base_predict_high": [],
            "predict": [],
            "predict_low": [],
            "predict_high": [],
            "weight": [],
            "weight_low": [],
            "weight_high": [],
            "value": [],
            "rule": [],
            "feature": [],
            "feature_value": [],
            "is_conjunctive": [],
            "classes": self.prediction["classes"],
        }
        factual["base_predict"].append(self.prediction["predict"])
        factual["base_predict_low"].append(self.prediction["low"])
        factual["base_predict_high"].append(self.prediction["high"])
        rules = self._define_conditions()
        for f, _ in enumerate(instance):  # pylint: disable=invalid-name
            if self.prediction["predict"] == self.feature_predict["predict"][f]:
                continue
            factual["predict"].append(self.feature_predict["predict"][f])
            factual["predict_low"].append(self.feature_predict["low"][f])
            factual["predict_high"].append(self.feature_predict["high"][f])
            factual["weight"].append(self.feature_weights["predict"][f])
            factual["weight_low"].append(self.feature_weights["low"][f])
            factual["weight_high"].append(self.feature_weights["high"][f])
            if f in self._get_explainer().categorical_features:
                if self._get_explainer().categorical_labels is not None:
                    factual["value"].append(
                        self._get_explainer().categorical_labels[f][int(instance[f])]
                    )
                else:
                    factual["value"].append(str(instance[f]))
            else:
                factual["value"].append(str(np.around(instance[f], decimals=2)))
            factual["rule"].append(rules[f])
            factual["feature"].append(f)
            factual["feature_value"].append(self.binned["rule_values"][f][0][-1])
            factual["is_conjunctive"].append(False)
        self.rules = factual
        self._has_rules = True
        return self.rules

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        """
        Add conjunctive factual rules.

        Parameters
        ----------
        n_top_features : int, optional
            Number of top features to combine.
        max_rule_size : int, optional
            Maximum size of the conjunctions.

        Returns
        -------
        self : :class:`.FactualExplanation`
            Returns a self reference, to allow for method chaining
        """
        if max_rule_size >= 4:
            raise ValueError("max_rule_size must be 2 or 3")
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
        i = self.index
        # pylint: disable=unsubscriptable-object, invalid-name
        threshold = None if self.y_threshold is None else self.y_threshold
        x_original = deepcopy(self.X_test)

        num_rules = len(factual["rule"])
        predicted_class = factual["classes"]
        conjunctive["classes"] = predicted_class
        if n_top_features is None:
            n_top_features = num_rules
        top_conjunctives = self._rank_features(
            np.reshape(conjunctive["weight"], (len(conjunctive["weight"]))),
            width=np.reshape(
                np.array(conjunctive["weight_high"]) - np.array(conjunctive["weight_low"]),
                (len(conjunctive["weight"])),
            ),
            num_to_show=np.min([num_rules, n_top_features]),
        )

        covered_features = []
        covered_combinations = [conjunctive["feature"][i] for i in range(len(conjunctive["rule"]))]
        for f1, cf1 in enumerate(factual["feature"]):  # cf = factual feature
            covered_features.append(cf1)
            of1 = factual["feature"][f1]  # of = original feature
            rule_value1 = (
                factual["feature_value"][f1]
                if isinstance(factual["feature_value"][f1], np.ndarray)
                else [factual["feature_value"][f1]]
            )
            for _, cf2 in enumerate(top_conjunctives):  # cf = conjunctive feature
                if cf2 in covered_features:
                    continue
                rule_values = [rule_value1]
                original_features = [of1]
                of2 = conjunctive["feature"][cf2]
                if conjunctive["is_conjunctive"][cf2]:
                    if of1 in of2:
                        continue
                    original_features.extend(iter(of2))
                    rule_values.extend(iter(conjunctive["feature_value"][cf2]))
                else:
                    if of1 == of2:
                        continue
                    original_features.append(of2)
                    rule_values.append(
                        conjunctive["feature_value"][cf2]
                        if isinstance(conjunctive["feature_value"][cf2], np.ndarray)
                        else [conjunctive["feature_value"][cf2]]
                    )
                skip = False
                for ofs in covered_combinations:
                    with contextlib.suppress(ValueError):
                        if np.all(np.sort(original_features) == ofs):
                            skip = True
                            break
                if skip:
                    continue
                covered_combinations.append(np.sort(original_features))

                rule_predict, rule_low, rule_high = self._predict_conjunctive(
                    rule_values,
                    original_features,
                    deepcopy(x_original),
                    threshold,
                    predicted_class,
                    bins=self.bin,
                )

                conjunctive["predict"].append(rule_predict)
                conjunctive["predict_low"].append(rule_low)
                conjunctive["predict_high"].append(rule_high)
                conjunctive["weight"].append(rule_predict - self.prediction["predict"])
                conjunctive["weight_low"].append(
                    rule_low - self.prediction["predict"] if rule_low != -np.inf else -np.inf
                )
                conjunctive["weight_high"].append(
                    rule_high - self.prediction["predict"] if rule_high != np.inf else np.inf
                )
                conjunctive["value"].append(factual["value"][f1] + "\n" + conjunctive["value"][cf2])
                conjunctive["feature"].append(original_features)
                conjunctive["feature_value"].append(rule_values)
                conjunctive["rule"].append(factual["rule"][f1] + " & \n" + conjunctive["rule"][cf2])
                conjunctive["is_conjunctive"].append(True)
        self.conjunctive_rules = conjunctive
        self._has_conjunctive_rules = True
        return self.add_conjunctions(n_top_features=n_top_features, max_rule_size=max_rule_size - 1)

    def _is_lesser(self, rule_boundary, instance_value):
        return instance_value < rule_boundary

    def plot(self, filter_top=None, **kwargs):
        """
        Plot the factual explanation for a given instance.

        Parameters
        ----------
        filter_top : int, optional
            The number of top features to display.
        **kwargs : dict
            Additional plotting arguments, such as:
            
            show : bool, default=True if filename is empty, False otherwise
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
            rnk_metric : str, default='feature_weight'
                The metric used to rank the features. Supported metrics are 'ensured', 'feature_weight', and 'uncertainty'.
            rnk_weight : float, default=0.5
                The weight of the uncertainty in the ranking. Used with the 'ensured' ranking metric.
        """
        # Ensure style_override gets passed through
        style_override = kwargs.get('style_override', None)

        filename = kwargs.get("filename", "")
        show = kwargs.get("show", filename == "")
        uncertainty = kwargs.get("uncertainty", False)
        rnk_metric = kwargs.get("rnk_metric", "feature_weight")
        if rnk_metric is None:
            rnk_metric = "feature_weight"
        rnk_weight = kwargs.get("rnk_weight", 0.5)
        if rnk_metric == "uncertainty":
            rnk_weight = 1.0
            rnk_metric = "ensured"

        factual = self._get_rules()  # get_explanation(index)
        self._check_preconditions()
        predict = self.prediction
        num_features_to_show = len(factual["weight"])
        if filter_top is None:
            filter_top = num_features_to_show
        filter_top = np.min([num_features_to_show, filter_top])
        if filter_top <= 0:
            warnings.warn(
                f"The explanation has no rules to plot. The index of the instance is {self.index}"
            )
            return

        if len(filename) > 0:
            path, filename, title, ext = prepare_for_saving(filename)
            path = f"plots/{path}"
            save_ext = [ext]
        else:
            path = ""
            title = ""
            save_ext = []
        if uncertainty:
            feature_weights = {
                "predict": factual["weight"],
                "low": factual["weight_low"],
                "high": factual["weight_high"],
            }
        else:
            feature_weights = factual["weight"]
        width = np.reshape(
            np.array(factual["weight_high"]) - np.array(factual["weight_low"]),
            (len(factual["weight"])),
        )

        if rnk_metric == "feature_weight":
            features_to_plot = self._rank_features(
                factual["weight"], width=width, num_to_show=filter_top
            )
        else:
            ranking = calculate_metrics(
                uncertainty=[
                    factual["predict_high"][i] - factual["predict_low"][i]
                    for i in range(len(factual["weight"]))
                ],
                prediction=factual["predict"],
                w=rnk_weight,
                metric=rnk_metric,
            )
            features_to_plot = self._rank_features(width=ranking, num_to_show=filter_top)

        column_names = factual["rule"]
        if "classification" in self._get_explainer().mode or self.is_thresholded():
            _plot_probabilistic(
                self,
                factual["value"],
                predict,
                feature_weights,
                features_to_plot,
                filter_top,
                column_names,
                title=title,
                path=path,
                interval=uncertainty,
                show=show,
                idx=self.index,
                save_ext=save_ext,
                style_override=style_override,
            )
        else:
            _plot_regression(
                self,
                factual["value"],
                predict,
                feature_weights,
                features_to_plot,
                filter_top,
                column_names,
                title=title,
                path=path,
                interval=uncertainty,
                show=show,
                idx=self.index,
                save_ext=save_ext,
                style_override=style_override,
            )


class AlternativeExplanation(CalibratedExplanation):
    """Class representing an alternative explanation for a given instance.

    Offers alternative explanations by exploring how changes to feature values could alter the model's prediction.
    """

    def __init__(
        self,
        calibrated_explanations,
        index,
        X_test,
        binned,
        feature_weights,
        feature_predict,
        prediction,
        y_threshold=None,
        instance_bin=None,
    ):
        """Class representing an alternative explanation for a given instance.

        Offers alternative explanations by exploring how changes to feature values could alter the model's prediction.
    
        Initialize an AlternativeExplanation instance.

        Parameters
        ----------
        calibrated_explanations : CalibratedExplanations
            The parent CalibratedExplanations object.
        index : int
            The index of the instance being explained.
        X_test : array-like
            The test dataset containing the instances to be explained.
        binned : dict
            A mapping of binned feature values.
        feature_weights : dict
            A mapping of feature weights.
        feature_predict : dict
            A mapping of feature predictions.
        prediction : dict
            A mapping containing the prediction results.
        y_threshold : float or tuple, optional
            The threshold for binary classification or regression explanations.
        instance_bin : int, optional
            The bin index of the instance.
        """
        super().__init__(
            calibrated_explanations,
            index,
            X_test,
            binned,
            feature_weights,
            feature_predict,
            prediction,
            y_threshold,
            instance_bin,
        )
        self._check_preconditions()
        self._has_rules = False
        self._get_rules()
        self.__is_super_explanation = False
        self.__is_semi_explanation = False
        self.__is_counter_explanation = False

    def __repr__(self):
        """Return a string representation of the alternative explanation."""
        alternative = self._get_rules()
        output = [
            f"{'Prediction':10} [{' Low':5}, {' High':5}]",
            f"{alternative['base_predict'][0]:5.3f} [{alternative['base_predict_low'][0]:5.3f}, {alternative['base_predict_high'][0]:5.3f}]",
            f"{'Value':6}: {'Feature':40s} {'Prediction':10} [{' Low':6}, {' High':6}]",
        ]
        feature_order = self._rank_features(
            alternative["weight"],
            width=np.array(alternative["weight_high"]) - np.array(alternative["weight_low"]),
            num_to_show=len(alternative["rule"]),
        )
        output.extend(
            f"{alternative['value'][f]:6}: {alternative['rule'][f]:40s} {alternative['predict'][f]:>6.3f}     [{alternative['predict_low'][f]:>6.3f}, {alternative['predict_high'][f]:>6.3f}]"
            for f in reversed(feature_order)
        )
        return "\n".join(output) + "\n"

    def _check_preconditions(self):
        if self.is_regression():
            if not isinstance(self._get_explainer().discretizer, RegressorDiscretizer):
                warnings.warn(
                    "Alternative explanations for regression recommend using the "
                    + "regressor discretizer. Consider extracting alternative "
                    + "explanations using `explainer.explain_alternatives(test_set)`"
                )
        elif not isinstance(self._get_explainer().discretizer, EntropyDiscretizer):
            warnings.warn(
                "Alternative explanations for classification recommend using "
                + "the entropy discretizer. Consider extracting alternative "
                + "explanations using `explainer.explain_alternatives(test_set)`"
            )

    # pylint: disable=too-many-statements, too-many-branches
    def _get_rules(self):
        """
        Create alternative rules.

        Returns
        -------
        Array-like : List[Dict[str, List]]
            A list of dictionaries containing the alternative rules.
        """
        if self._has_conjunctive_rules:
            return self.conjunctive_rules
        if self._has_rules:
            return self.rules
        self.rules = []
        self.labels = {}  # pylint: disable=attribute-defined-outside-init
        instance = np.array(self.X_test, copy=True)
        instance.flags.writeable = False
        # pylint: disable=protected-access
        discretized = self._get_explainer()._discretize(instance.reshape(1, -1))[
            0
        ]
        instance_predict = self.binned["predict"]
        instance_low = self.binned["low"]
        instance_high = self.binned["high"]
        alternative = self.__set_up_result()
        rule_boundaries = self._get_explainer().rule_boundaries(instance)
        for f, _ in enumerate(instance):  # pylint: disable=invalid-name
            if f in self._get_explainer().categorical_features:
                values = np.array(self._get_explainer().feature_values[f])
                values = np.delete(values, values == discretized[f])
                for value_bin, value in enumerate(values):
                    # skip if identical to original
                    if (
                        self.prediction["low"] == instance_low[f][value_bin]
                        and self.prediction["high"] == instance_high[f][value_bin]
                    ):
                        continue
                    alternative["predict"].append(instance_predict[f][value_bin])
                    alternative["predict_low"].append(instance_low[f][value_bin])
                    alternative["predict_high"].append(instance_high[f][value_bin])
                    alternative["weight"].append(
                        instance_predict[f][value_bin] - self.prediction["predict"]
                    )
                    alternative["weight_low"].append(
                        instance_low[f][value_bin] - self.prediction["predict"]
                        if instance_low[f][value_bin] != -np.inf
                        else instance_low[f][value_bin]
                    )
                    alternative["weight_high"].append(
                        instance_high[f][value_bin] - self.prediction["predict"]
                        if instance_high[f][value_bin] != np.inf
                        else instance_high[f][value_bin]
                    )
                    if self._get_explainer().categorical_labels is not None:
                        alternative["value"].append(
                            self._get_explainer().categorical_labels[f][int(instance[f])]
                        )
                    else:
                        alternative["value"].append(str(np.around(instance[f], decimals=2)))
                    alternative["feature"].append(f)
                    alternative["feature_value"].append(value)
                    if self._get_explainer().categorical_labels is not None:
                        self.labels[len(alternative["rule"])] = f
                        alternative["rule"].append(
                            f"{self._get_explainer().feature_names[f]} = "
                            + f"{self._get_explainer().categorical_labels[f][int(value)]}"
                        )
                    else:
                        alternative["rule"].append(
                            f"{self._get_explainer().feature_names[f]} = {value}"
                        )
                    alternative["is_conjunctive"].append(False)
            else:
                values = np.array(self._get_explainer().X_cal[:, f])
                lesser = rule_boundaries[f][0]
                greater = rule_boundaries[f][1]

                value_bin = 0
                if np.any(values < lesser):
                    # skip if identical to original
                    if self.prediction["low"] == np.mean(
                        instance_low[f][value_bin]
                    ) and self.prediction["high"] == np.mean(instance_high[f][value_bin]):
                        continue
                    alternative["predict"].append(np.mean(instance_predict[f][value_bin]))
                    alternative["predict_low"].append(np.mean(instance_low[f][value_bin]))
                    alternative["predict_high"].append(np.mean(instance_high[f][value_bin]))
                    alternative["weight"].append(
                        np.mean(instance_predict[f][value_bin]) - self.prediction["predict"]
                    )
                    alternative["weight_low"].append(
                        np.mean(instance_low[f][value_bin]) - self.prediction["predict"]
                        if instance_low[f][value_bin] != -np.inf
                        else instance_low[f][value_bin]
                    )
                    alternative["weight_high"].append(
                        np.mean(instance_high[f][value_bin]) - self.prediction["predict"]
                        if instance_high[f][value_bin] != np.inf
                        else instance_high[f][value_bin]
                    )
                    alternative["value"].append(str(np.around(instance[f], decimals=2)))
                    alternative["feature"].append(f)
                    alternative["feature_value"].append(self.binned["rule_values"][f][0][0])
                    alternative["rule"].append(
                        f"{self._get_explainer().feature_names[f]} < {lesser:.2f}"
                    )
                    alternative["is_conjunctive"].append(False)
                    value_bin = 1

                if np.any(values > greater):
                    # skip if identical to original
                    if self.prediction["low"] == np.mean(
                        instance_low[f][value_bin]
                    ) and self.prediction["high"] == np.mean(instance_high[f][value_bin]):
                        continue
                    alternative["predict"].append(np.mean(instance_predict[f][value_bin]))
                    alternative["predict_low"].append(np.mean(instance_low[f][value_bin]))
                    alternative["predict_high"].append(np.mean(instance_high[f][value_bin]))
                    alternative["weight"].append(
                        np.mean(instance_predict[f][value_bin]) - self.prediction["predict"]
                    )
                    alternative["weight_low"].append(
                        np.mean(instance_low[f][value_bin]) - self.prediction["predict"]
                        if instance_low[f][value_bin] != -np.inf
                        else instance_low[f][value_bin]
                    )
                    alternative["weight_high"].append(
                        np.mean(instance_high[f][value_bin]) - self.prediction["predict"]
                        if instance_high[f][value_bin] != np.inf
                        else instance_high[f][value_bin]
                    )
                    alternative["value"].append(str(np.around(instance[f], decimals=2)))
                    alternative["feature"].append(f)
                    alternative["feature_value"].append(
                        self.binned["rule_values"][f][0][
                            1 if len(self.binned["rule_values"][f][0]) == 3 else 0
                        ]
                    )
                    alternative["rule"].append(
                        f"{self._get_explainer().feature_names[f]} > {greater:.2f}"
                    )
                    alternative["is_conjunctive"].append(False)

        self.rules = alternative
        self._has_rules = True
        return self.rules

    def __set_up_result(self):
        result = {
            "base_predict": [],
            "base_predict_low": [],
            "base_predict_high": [],
            "predict": [],
            "predict_low": [],
            "predict_high": [],
            "weight": [],
            "weight_low": [],
            "weight_high": [],
            "value": [],
            "rule": [],
            "feature": [],
            "feature_value": [],
            "is_conjunctive": [],
            "classes": self.prediction["classes"],
        }
        result["base_predict"].append(self.prediction["predict"])
        result["base_predict_low"].append(self.prediction["low"])
        result["base_predict_high"].append(self.prediction["high"])
        return result

    def is_super_explanation(self):
        """Determine if the explanation is a super-explanation."""
        return self.__is_super_explanation

    def is_semi_explanation(self):
        """Determine if the explanation is a semi-explanation."""
        return self.__is_semi_explanation

    def is_counter_explanation(self):
        """Determine if the explanation is a counter-explanation."""
        return self.__is_counter_explanation

    def __filter_rules(
        self,
        only_ensured=False,
        make_super=False,
        make_semi=False,
        make_counter=False,
        include_potential=False,
    ):
        """Filter rules based on the explanation type."""
        if self.is_regression() and not self.is_probabilistic():
            warnings.warn(
                "Regression explanations are not probabilistic. Filtering rules may not be effective."
            )
        positive_class = self.prediction["predict"] > 0.5
        initial_uncertainty = np.abs(self.prediction["high"] - self.prediction["low"])

        new_rules = self.__set_up_result()
        rules = self._get_rules()  # pylint: disable=protected-access
        for rule in range(len(rules["rule"])):
            # filter out potential rules if include_potential is False
            if not include_potential and (
                rules["predict_low"][rule] < 0.5 < rules["predict_high"][rule]
            ):
                continue
            if make_super and (
                positive_class
                and rules["predict"][rule] <= self.prediction["predict"]
                or not positive_class
                and rules["predict"][rule] >= self.prediction["predict"]
            ):
                continue
            if make_semi:
                if positive_class:
                    if (
                        rules["predict"][rule] < 0.5
                        or rules["predict"][rule] > self.prediction["predict"]
                    ):
                        continue
                elif (
                    rules["predict"][rule] > 0.5
                    or rules["predict"][rule] < self.prediction["predict"]
                ):
                    continue
            if make_counter and (
                positive_class
                and rules["predict"][rule] > 0.5
                or not positive_class
                and rules["predict"][rule] < 0.5
            ):
                continue
            # if only_ensured is True, filter out rules that lead to increased uncertainty
            if (
                only_ensured
                and rules["predict_high"][rule] - rules["predict_low"][rule] > initial_uncertainty
            ):
                continue
            # filter out rules that does not provide a different prediction
            if (
                rules["base_predict_low"] == rules["predict_low"][rule]
                and rules["base_predict_high"] == rules["predict_high"][rule]
            ):
                continue
            new_rules["predict"].append(rules["predict"][rule])
            new_rules["predict_low"].append(rules["predict_low"][rule])
            new_rules["predict_high"].append(rules["predict_high"][rule])
            new_rules["weight"].append(rules["weight"][rule])
            new_rules["weight_low"].append(rules["weight_low"][rule])
            new_rules["weight_high"].append(rules["weight_high"][rule])
            new_rules["value"].append(rules["value"][rule])
            new_rules["rule"].append(rules["rule"][rule])
            new_rules["feature"].append(rules["feature"][rule])
            new_rules["feature_value"].append(rules["feature_value"][rule])
            new_rules["is_conjunctive"].append(rules["is_conjunctive"][rule])
        new_rules["classes"] = rules["classes"]

        if self._has_conjunctive_rules:  # pylint: disable=protected-access
            self.__extracted_non_conjunctive_rules(new_rules)
        self.rules = new_rules
        return self

    # extract non-conjunctive rules
    def __extracted_non_conjunctive_rules(self, new_rules):
        self.conjunctive_rules = MappingProxyType(new_rules)
        new_rules["predict"] = [
            value
            for i, value in enumerate(new_rules["predict"])
            if not new_rules["is_conjunctive"][i]
        ]
        new_rules["predict_low"] = [
            value
            for i, value in enumerate(new_rules["predict_low"])
            if not new_rules["is_conjunctive"][i]
        ]
        new_rules["predict_high"] = [
            value
            for i, value in enumerate(new_rules["predict_high"])
            if not new_rules["is_conjunctive"][i]
        ]
        new_rules["weight"] = [
            value
            for i, value in enumerate(new_rules["weight"])
            if not new_rules["is_conjunctive"][i]
        ]
        new_rules["weight_low"] = [
            value
            for i, value in enumerate(new_rules["weight_low"])
            if not new_rules["is_conjunctive"][i]
        ]
        new_rules["weight_high"] = [
            value
            for i, value in enumerate(new_rules["weight_high"])
            if not new_rules["is_conjunctive"][i]
        ]
        new_rules["value"] = [
            value
            for i, value in enumerate(new_rules["value"])
            if not new_rules["is_conjunctive"][i]
        ]
        new_rules["rule"] = [
            value for i, value in enumerate(new_rules["rule"]) if not new_rules["is_conjunctive"][i]
        ]
        new_rules["feature"] = [
            value
            for i, value in enumerate(new_rules["feature"])
            if not new_rules["is_conjunctive"][i]
        ]
        new_rules["feature_value"] = [
            value
            for i, value in enumerate(new_rules["feature_value"])
            if not new_rules["is_conjunctive"][i]
        ]
        new_rules["is_conjunctive"] = [
            value
            for i, value in enumerate(new_rules["is_conjunctive"])
            if not new_rules["is_conjunctive"][i]
        ]

    def reset(self):
        """Reset the explanation to its original state."""
        self.__is_super_explanation = False
        self.__is_semi_explanation = False
        self.__is_counter_explanation = False
        self._has_rules = False
        self._get_rules()
        return self

    def super_explanations(self, only_ensured=False, include_potential=False):
        """
        Provide super-explanations that support the predicted class.

        Returns
        -------
        :class:`.AlternativeExplanation`
        """
        self.__filter_rules(
            only_ensured=only_ensured, make_super=True, include_potential=include_potential
        )
        self.__is_super_explanation = True
        return self

    def semi_explanations(self, only_ensured=False, include_potential=False):
        """
        Provide semi-explanations that partially support the predicted class.

        Returns
        -------
        :class:`.AlternativeExplanation`
        """
        self.__filter_rules(
            only_ensured=only_ensured, make_semi=True, include_potential=include_potential
        )
        self.__is_semi_explanation = True
        return self

    def counter_explanations(self, only_ensured=False, include_potential=False):
        """
        Provide counter-explanations that do not support the predicted class.

        Returns
        -------
        :class:`.AlternativeExplanation`
        """
        self.__filter_rules(
            only_ensured=only_ensured, make_counter=True, include_potential=include_potential
        )
        self.__is_counter_explanation = True
        return self

    def ensured_explanations(self, include_potential=False):
        """
        Provide ensured explanations with smaller confidence intervals.

        Returns
        -------
        :class:`.AlternativeExplanation`
        """
        self.__filter_rules(only_ensured=True, include_potential=include_potential)
        return self

    # pylint: disable=too-many-locals
    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        """
        Add conjunctive alternative rules.

        Parameters
        ----------
        n_top_features : int, optional
            Number of top features to combine.
        max_rule_size : int, optional
            Maximum size of the conjunctions.

        Returns
        -------
        self : :class:`.AlternativeExplanation`
            Returns a self reference, to allow for method chaining
        """
        if max_rule_size >= 4:
            raise ValueError("max_rule_size must be 2 or 3")
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

        num_rules = len(alternative["rule"])
        predicted_class = alternative["classes"]
        conjunctive["classes"] = predicted_class
        if n_top_features is None:
            n_top_features = num_rules
        top_conjunctives = self._rank_features(
            np.reshape(conjunctive["weight"], (len(conjunctive["weight"]))),
            width=np.reshape(
                np.array(conjunctive["weight_high"]) - np.array(conjunctive["weight_low"]),
                (len(conjunctive["weight"])),
            ),
            num_to_show=np.min([num_rules, n_top_features]),
        )

        covered_features = []
        covered_combinations = [conjunctive["feature"][i] for i in range(len(conjunctive["rule"]))]
        for f1, cf1 in enumerate(alternative["feature"]):  # cf = factual feature
            covered_features.append(cf1)
            of1 = alternative["feature"][f1]  # of = original feature
            rule_value1 = (
                alternative["feature_value"][f1]
                if isinstance(alternative["feature_value"][f1], np.ndarray)
                else [alternative["feature_value"][f1]]
            )
            for _, cf2 in enumerate(top_conjunctives):  # cf = conjunctive feature
                if cf2 in covered_features:
                    continue
                rule_values = [rule_value1]
                original_features = [of1]
                of2 = conjunctive["feature"][cf2]
                if conjunctive["is_conjunctive"][cf2]:
                    if of1 in of2:
                        continue
                    original_features.extend(iter(of2))
                    rule_values.extend(iter(conjunctive["feature_value"][cf2]))
                else:
                    if of1 == of2:
                        continue
                    original_features.append(of2)
                    rule_values.append(
                        conjunctive["feature_value"][cf2]
                        if isinstance(conjunctive["feature_value"][cf2], np.ndarray)
                        else [conjunctive["feature_value"][cf2]]
                    )
                skip = any(
                    np.all(np.sort(original_features) == ofs) for ofs in covered_combinations
                )
                if skip:
                    continue
                covered_combinations.append(np.sort(original_features))

                rule_predict, rule_low, rule_high = self._predict_conjunctive(
                    rule_values,
                    original_features,
                    deepcopy(x_original),
                    threshold,
                    predicted_class,
                    bins=self.bin,
                )
                conjunctive["predict"].append(rule_predict)
                conjunctive["predict_low"].append(rule_low)
                conjunctive["predict_high"].append(rule_high)
                conjunctive["weight"].append(rule_predict - self.prediction["predict"])
                conjunctive["weight_low"].append(
                    rule_low - self.prediction["predict"] if rule_low != -np.inf else -np.inf
                )
                conjunctive["weight_high"].append(
                    rule_high - self.prediction["predict"] if rule_high != np.inf else np.inf
                )
                conjunctive["value"].append(
                    alternative["value"][f1] + "\n" + conjunctive["value"][cf2]
                )
                conjunctive["feature"].append(original_features)
                conjunctive["feature_value"].append(rule_values)
                conjunctive["rule"].append(
                    alternative["rule"][f1] + " & \n" + conjunctive["rule"][cf2]
                )
                conjunctive["is_conjunctive"].append(True)
        self.conjunctive_rules = conjunctive
        self._has_conjunctive_rules = True
        return self.add_conjunctions(n_top_features=n_top_features, max_rule_size=max_rule_size - 1)

    def _is_lesser(self, rule_boundary, instance_value):
        return rule_boundary < instance_value

    # pylint: disable=consider-iterating-dictionary
    def plot(self, filter_top=None, **kwargs):
        """
        Plot the alternative explanation.

        Parameters
        ----------
        filter_top : int, optional
            The number of top features to display.
        **kwargs : dict
            Additional plotting arguments, such as:
            
            show : bool, default=True if filename is empty, False otherwise
                A boolean parameter that determines whether the plot should be displayed or not. If set to
                True, the plot will be displayed. If set to False, the plot will not be displayed.
            filename : str, default=''
                The filename parameter is a string that represents the full path and filename of the plot
                image file that will be saved. If this parameter is not provided or is an empty string, the plot
                will not be saved as an image file.
            style : str, default='regular'
                The `style` parameter is a string that determines the style of the plot. Possible styles are for :class:`.AlternativeExplanation`:
                
                * 'regular' - a regular plot with feature weights and uncertainty intervals (if applicable)
                * 'triangular' - a triangular plot for alternative explanations highlighting the interplay between the calibrated probability and the uncertainty intervals
            rnk_metric : str, default='ensured'
                The metric used to rank the features. Supported metrics are 'ensured', 'feature_weight', and 'uncertainty'.
            rnk_weight : float, default=0.5
                The weight of the uncertainty in the ranking. Used with the 'ensured' ranking metric.
        """
        # Ensure style_override gets passed through
        style_override = kwargs.get('style_override', None)

        filename = kwargs.get("filename", "")
        show = kwargs.get("show", filename == "")
        rnk_metric = kwargs.get("rnk_metric", "ensured")
        if rnk_metric is None:
            rnk_metric = "ensured"
        rnk_weight = kwargs.get("rnk_weight", 0.5)
        if rnk_metric == "uncertainty":
            rnk_weight = 1.0
            rnk_metric = "ensured"

        alternative = self._get_rules()  # get_explanation(index)
        self._check_preconditions()
        predict = self.prediction
        if len(filename) > 0:
            path, filename, title, ext = prepare_for_saving(filename)
            path = f"plots/{path}"
            save_ext = [ext]
        else:
            path = ""
            title = ""
            save_ext = []
        feature_predict = {
            "predict": alternative["predict"],
            "low": alternative["predict_low"],
            "high": alternative["predict_high"],
        }
        feature_weights = np.reshape(alternative["weight"], (len(alternative["weight"])))
        width = np.reshape(
            np.array(alternative["weight_high"]) - np.array(alternative["weight_low"]),
            (len(alternative["weight"])),
        )
        num_rules = len(alternative["rule"])
        if filter_top is None:
            filter_top = num_rules
        num_to_show_ = np.min([num_rules, filter_top])
        if num_to_show_ <= 0:
            warnings.warn(
                f"The explanation has no rules to plot. The index of the instance is {self.index}"
            )
            return

        if rnk_metric == "feature_weight":
            features_to_plot = self._rank_features(
                feature_weights, width=width, num_to_show=num_to_show_
            )
        else:
            # Always rank base on predicted class
            prediction = alternative["predict"]
            if self.get_mode() == "classification" or self.is_thresholded():
                prediction = prediction if predict["predict"] > 0.5 else [1 - p for p in prediction]
            ranking = calculate_metrics(
                uncertainty=[
                    alternative["predict_high"][i] - alternative["predict_low"][i]
                    for i in range(num_rules)
                ],
                prediction=prediction,
                w=rnk_weight,
                metric=rnk_metric,
            )
            features_to_plot = self._rank_features(width=ranking, num_to_show=num_to_show_)

        if "style" in kwargs and kwargs["style"] == "triangular":
            proba = predict["predict"]
            uncertainty = np.abs(predict["high"] - predict["low"])
            rule_proba = alternative["predict"]
            rule_uncertainty = np.abs(
                np.array(alternative["predict_high"]) - np.array(alternative["predict_low"])
            )
            # Use list comprehension or NumPy array indexing to select elements
            selected_rule_proba = [rule_proba[i] for i in features_to_plot]
            selected_rule_uncertainty = [rule_uncertainty[i] for i in features_to_plot]

            _plot_triangular(
                self,
                proba,
                uncertainty,
                selected_rule_proba,
                selected_rule_uncertainty,
                num_to_show_,
                title=title,
                path=path,
                show=show,
                save_ext=save_ext,
                style_override=style_override,
            )
            return

        column_names = alternative["rule"]
        _plot_alternative(
            self,
            alternative["value"],
            predict,
            feature_predict,
            features_to_plot,
            num_to_show=num_to_show_,
            column_names=column_names,
            title=title,
            path=path,
            show=show,
            save_ext=save_ext,
            style_override=style_override,
        )


class FastExplanation(CalibratedExplanation):
    """Class representing fast explanations.

    Represents fast, SHAP-like explanations, enabling efficient interpretation of model behavior for large datasets.
    """

    def __init__(
        self,
        calibrated_explanations,
        index,
        X_test,
        feature_weights,
        feature_predict,
        prediction,
        y_threshold=None,
        instance_bin=None,
    ):
        """Class representing fast explanations.

        Represents fast, SHAP-like explanations, enabling efficient interpretation of model behavior for large datasets.
    
        Initialize a FastExplanation instance.

        Parameters
        ----------
        calibrated_explanations : CalibratedExplanations
            The parent CalibratedExplanations object.
        index : int
            The index of the instance being explained.
        X_test : array-like
            The test dataset containing the instances to be explained.
        feature_weights : dict
            A mapping of feature weights.
        feature_predict : dict
            A mapping of feature predictions.
        prediction : dict
            A mapping containing the prediction results.
        y_threshold : float or tuple, optional
            The threshold for binary classification or regression explanations.
        instance_bin : int, optional
            The bin index of the instance.
        """
        super().__init__(
            calibrated_explanations,
            index,
            X_test,
            {},
            feature_weights,
            feature_predict,
            prediction,
            y_threshold,
            instance_bin,
        )
        self._check_preconditions()
        self._get_rules()

    def __repr__(self):
        """Return a string representation of the fast explanation."""
        fast = self._get_rules()
        output = [
            f"{'Prediction':10} [{' Low':5}, {' High':5}]",
            f"   {fast['base_predict'][0]:5.3f}   [{fast['base_predict_low'][0]:5.3f}, {fast['base_predict_high'][0]:5.3f}]",
            f"{'Value':6}: {'Feature':40s} {'Weight':6} [{' Low':6}, {' High':6}]",
        ]
        feature_order = self._rank_features(
            fast["weight"],
            width=np.array(fast["weight_high"]) - np.array(fast["weight_low"]),
            num_to_show=len(fast["rule"]),
        )
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
        """This method is currently not supported for `FastExplanation`, making this call result in no change.
        
        Parameters
        ----------
        n_top_features : int
            The number of top features to consider for conjunctions. Default is 5.
        max_rule_size : int
            The maximum size of the conjunctive rules. Default is 2.

        Warning
        -------
        This method is not supported for :class:`.FastExplanation` and will not alter the explanation.
        """
        warnings.warn(
            "The add_conjunctions method is currently not supported for `FastExplanation`, making this call resulting in no change."
        )
        # pass

    def _is_lesser(self, rule_boundary, instance_value):
        pass

    def add_new_rule_condition(self, feature, rule_boundary):
        """Create a new rule condition for a numerical feature.

        Warning
        -------
        This method is not supported for :class:`.FastExplanation` and will not alter the explanation.
        """
        warnings.warn(
            "The add_new_rule_condition method is currently not supported for `FastExplanation`, making this call resulting in no change."
        )
        # pass

    def _check_preconditions(self):
        pass

    # pylint: disable=too-many-statements, too-many-branches
    def _get_rules(self):
        """
        Create fast explanation rules.

        Returns
        -------
        dict
            A dictionary containing the fast explanation rules.
        """
        if self._has_conjunctive_rules:
            return self.conjunctive_rules
        if self._has_rules:
            return self.rules
        self._has_rules = False
        # i = self.index
        instance = np.array(self.X_test, copy=True)
        fast = {
            "base_predict": [],
            "base_predict_low": [],
            "base_predict_high": [],
            "predict": [],
            "predict_low": [],
            "predict_high": [],
            "weight": [],
            "weight_low": [],
            "weight_high": [],
            "value": [],
            "rule": [],
            "feature": [],
            "feature_value": [],
            "is_conjunctive": [],
            "classes": self.prediction["classes"],
        }
        fast["base_predict"].append(self.prediction["predict"])
        fast["base_predict_low"].append(self.prediction["low"])
        fast["base_predict_high"].append(self.prediction["high"])
        rules = self._define_conditions()
        for f, _ in enumerate(instance):  # pylint: disable=invalid-name
            if self.prediction["predict"] == self.feature_predict["predict"][f]:
                continue
            fast["predict"].append(self.feature_predict["predict"][f])
            fast["predict_low"].append(self.feature_predict["low"][f])
            fast["predict_high"].append(self.feature_predict["high"][f])
            fast["weight"].append(self.feature_weights["predict"][f])
            fast["weight_low"].append(self.feature_weights["low"][f])
            fast["weight_high"].append(self.feature_weights["high"][f])
            if f in self._get_explainer().categorical_features:
                if self._get_explainer().categorical_labels is not None:
                    fast["value"].append(
                        self._get_explainer().categorical_labels[f][int(instance[f])]
                    )
                else:
                    fast["value"].append(str(instance[f]))
            else:
                fast["value"].append(str(np.around(instance[f], decimals=2)))
            fast["rule"].append(rules[f])
            fast["feature"].append(f)
            fast["feature_value"].append(None)
            fast["is_conjunctive"].append(False)
        self.rules = fast
        self._has_rules = True
        return self.rules

    def _define_conditions(self):
        """
        Define the rule conditions for the fast explanation.

        Returns
        -------
        list[str]
            A list of conditions for each feature.
        """
        self.conditions = []
        for f in range(self._get_explainer().num_features):
            rule = f"{self._get_explainer().feature_names[f]}"
            self.conditions.append(rule)
        return self.conditions

    def plot(self, filter_top=None, **kwargs):
        """
        Plot the fast explanation.

        Parameters
        ----------
        filter_top : int, optional
            The number of top features to display.
        **kwargs : dict
            Additional plotting arguments, such as:
            
            show : bool, default=True if filename is empty, False otherwise
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
                The `style` parameter is a string that determines the style of the plot. Possible styles are for :class:`.FastExplanation`:
                
                * 'regular' - a regular plot with feature weights and uncertainty intervals (if applicable)
            rnk_metric : str, default='feature_weight'
                The metric used to rank the features. Supported metrics are 'ensured', 'feature_weight', and 'uncertainty'.
            rnk_weight : float, default=0.5
                The weight of the uncertainty in the ranking. Used with the 'ensured' ranking metric.
        """
        # Ensure style_override gets passed through
        style_override = kwargs.get('style_override', None)

        filename = kwargs.get("filename", "")
        show = kwargs.get("show", filename == "")
        uncertainty = kwargs.get("uncertainty", False)
        rnk_metric = kwargs.get("rnk_metric", "feature_weight")
        if rnk_metric is None:
            rnk_metric = "feature_weight"
        rnk_weight = kwargs.get("rnk_weight", 0.5)
        if rnk_metric == "uncertainty":
            rnk_weight = 1.0
            rnk_metric = "ensured"

        factual = self._get_rules()  # get_explanation(index)
        self._check_preconditions()
        predict = self.prediction
        num_features_to_show = len(factual["weight"])
        if filter_top is None:
            filter_top = num_features_to_show
        filter_top = np.min([num_features_to_show, filter_top])
        if filter_top <= 0:
            warnings.warn(
                f"The explanation has no rules to plot. The index of the instance is {self.index}"
            )
            return

        if len(filename) > 0:
            path, filename, title, ext = prepare_for_saving(filename)
            path = f"plots/{path}"
            save_ext = [ext]
        else:
            path = ""
            title = ""
            save_ext = []
        if uncertainty:
            feature_weights = {
                "predict": factual["weight"],
                "low": factual["weight_low"],
                "high": factual["weight_high"],
            }
        else:
            feature_weights = factual["weight"]
        width = np.reshape(
            np.array(factual["weight_high"]) - np.array(factual["weight_low"]),
            (len(factual["weight"])),
        )

        if rnk_metric == "feature_weight":
            features_to_plot = self._rank_features(
                factual["weight"], width=width, num_to_show=filter_top
            )
        else:
            ranking = calculate_metrics(
                uncertainty=[
                    factual["predict_high"][i] - factual["predict_low"][i]
                    for i in range(len(factual["weight"]))
                ],
                prediction=factual["predict"],
                w=rnk_weight,
                metric=rnk_metric,
            )
            features_to_plot = self._rank_features(width=ranking, num_to_show=filter_top)

        column_names = factual["rule"]
        if "classification" in self._get_explainer().mode or self.is_thresholded():
            _plot_probabilistic(
                self,
                factual["value"],
                predict,
                feature_weights,
                features_to_plot,
                filter_top,
                column_names,
                title=title,
                path=path,
                interval=uncertainty,
                show=show,
                idx=self.index,
                save_ext=save_ext,
                style_override=style_override
            )
        else:
            _plot_regression(
                self,
                factual["value"],
                predict,
                feature_weights,
                features_to_plot,
                filter_top,
                column_names,
                title=title,
                path=path,
                interval=uncertainty,
                show=show,
                idx=self.index,
                save_ext=save_ext,
                style_override=style_override
            )
