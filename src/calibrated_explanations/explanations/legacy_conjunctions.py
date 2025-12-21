"""
Legacy conjunction implementations.

This module contains the legacy implementations of add_conjunctions for
FactualExplanation and AlternativeExplanation, as well as the helper
_predict_conjunctive. These are preserved for parity testing and must
NOT be modified.
"""

from typing import Any, Dict, Tuple

import numpy as np

from ..utils.helper import safe_first_element


def _predict_conjunctive_legacy(
    self,
    rule_value_set,
    original_features,
    perturbed,
    threshold,
    predicted_class,
    bins=None,
):
    """
    Calculate the prediction for a conjunctive rule (Legacy).

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
    if len(original_features) < 2:
        from ..utils.exceptions import ValidationError

        raise ValidationError(
            "Conjunctive rules require at least two features",
            details={
                "param": "original_features",
                "count": len(original_features),
                "requirement": "minimum 2 features",
            },
        )

    predict_fn = self._get_explainer()._predict  # pylint: disable=protected-access
    # Ensure perturbed is a writable copy to avoid "read-only" errors
    perturbed = np.array(perturbed, copy=True)

    base_values = np.array([perturbed[idx] for idx in original_features], copy=True)

    rule_predict = 0.0
    rule_low = 0.0
    rule_high = 0.0
    rule_count = 0

    value_iterables = [np.asarray(values) for values in rule_value_set[: len(original_features)]]

    def _restore() -> None:
        for pos, feat_idx in enumerate(original_features):
            perturbed[feat_idx] = base_values[pos]

    try:
        if len(original_features) == 2:
            of1, of2 = original_features[:2]
            values1, values2 = value_iterables[:2]
            for value_1 in values1:
                perturbed[of1] = value_1
                for value_2 in values2:
                    perturbed[of2] = value_2
                    perturbed_row = perturbed.reshape(1, -1)
                    p_value, low, high, _ = predict_fn(
                        perturbed_row,
                        threshold=threshold,
                        low_high_percentiles=self.calibrated_explanations.low_high_percentiles,
                        classes=predicted_class,
                        bins=bins,
                    )
                    rule_predict += float(safe_first_element(p_value))
                    rule_low += float(safe_first_element(low))
                    rule_high += float(safe_first_element(high))
                    rule_count += 1
        else:
            of1, of2, of3 = original_features[:3]
            values1, values2, values3 = value_iterables[:3]
            for value_1 in values1:
                perturbed[of1] = value_1
                for value_2 in values2:
                    perturbed[of2] = value_2
                    for value_3 in values3:
                        perturbed[of3] = value_3
                        perturbed_row = perturbed.reshape(1, -1)
                        p_value, low, high, _ = predict_fn(
                            perturbed_row,
                            threshold=threshold,
                            low_high_percentiles=self.calibrated_explanations.low_high_percentiles,
                            classes=predicted_class,
                            bins=bins,
                        )
                        rule_predict += float(safe_first_element(p_value))
                        rule_low += float(safe_first_element(low))
                        rule_high += float(safe_first_element(high))
                        rule_count += 1
    finally:
        _restore()

    if rule_count:
        rule_predict /= rule_count
        rule_low /= rule_count
        rule_high /= rule_count
    return rule_predict, rule_low, rule_high


def add_conjunctions_factual_legacy(self, n_top_features=5, max_rule_size=2):
    """Add conjunctive factual rules (Legacy)."""
    if max_rule_size >= 4:
        from ..utils.exceptions import ConfigurationError

        raise ConfigurationError(
            "max_rule_size must be 2 or 3",
            details={
                "param": "max_rule_size",
                "value": max_rule_size,
                "valid_range": [2, 3],
            },
        )
    if max_rule_size < 2:
        return self

    factual = self._get_rules() if not self._has_rules else self.rules

    def _clone_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        cloned: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = value
        return cloned

    conjunctive_state = (
        _clone_payload(self.conjunctive_rules)
        if self._has_conjunctive_rules and self.conjunctive_rules is not None
        else _clone_payload(factual)
    )

    self._has_conjunctive_rules = False
    self.conjunctive_rules = []

    threshold = None if self.y_threshold is None else self.y_threshold
    scratch = np.array(self.x_test, copy=True)
    predicted_class = factual["classes"]
    conjunctive_state["classes"] = predicted_class

    if n_top_features is None:
        n_top_features = len(factual["rule"])

    def _normalise_features(values: Any) -> Tuple[int, ...]:
        if isinstance(values, (list, tuple, np.ndarray)):
            return tuple(sorted(int(v) for v in np.asarray(values).ravel()))
        return (int(values),)

    def _feature_length(candidate: Any) -> int:
        if isinstance(candidate, (list, tuple, np.ndarray)):
            return len(candidate)
        return 1

    for current_size in range(2, max_rule_size + 1):
        num_rules = len(factual["rule"])
        if num_rules == 0:
            break

        weights_array = np.asarray(conjunctive_state["weight"], dtype=float)
        width_array = np.asarray(conjunctive_state["weight_high"], dtype=float) - np.asarray(
            conjunctive_state["weight_low"], dtype=float
        )
        top_conjunctives = list(
            self._rank_features(
                weights_array,
                width=width_array,
                num_to_show=min(num_rules, n_top_features),
            )
        )

        covered_combinations = {
            _normalise_features(conjunctive_state["feature"][i])
            for i in range(len(conjunctive_state["feature"]))
        }

        for f1, _ in enumerate(factual["feature"]):
            of1 = factual["feature"][f1]
            sampled_values1 = factual["sampled_values"][f1]
            rule_value1 = (
                sampled_values1 if isinstance(sampled_values1, np.ndarray) else [sampled_values1]
            )

            for cf2 in top_conjunctives:
                rule_values = [rule_value1]
                original_features = [of1]
                of2 = conjunctive_state["feature"][cf2]
                target_length = current_size - 1
                if _feature_length(of2) != target_length:
                    continue
                if conjunctive_state["is_conjunctive"][cf2]:
                    if of1 in of2:
                        continue
                    original_features.extend(int(v) for v in of2)
                    rule_values.extend(list(conjunctive_state["sampled_values"][cf2]))
                else:
                    if of1 == of2:
                        continue
                    original_features.append(of2)
                    sampled_values2 = conjunctive_state["sampled_values"][cf2]
                    rule_values.append(
                        sampled_values2
                        if isinstance(sampled_values2, np.ndarray)
                        else [sampled_values2]
                    )

                combo_key = _normalise_features(original_features)
                if combo_key in covered_combinations:
                    continue
                covered_combinations.add(combo_key)

                rule_predict, rule_low, rule_high = _predict_conjunctive_legacy(
                    self,
                    rule_values,
                    original_features,
                    scratch,
                    threshold,
                    predicted_class,
                    bins=self.bin,
                )

                conjunctive_state["predict"].append(rule_predict)
                conjunctive_state["predict_low"].append(rule_low)
                conjunctive_state["predict_high"].append(rule_high)
                conjunctive_state["weight"].append(rule_predict - self.prediction["predict"])
                conjunctive_state["weight_low"].append(
                    rule_low - self.prediction["predict"] if rule_low != -np.inf else -np.inf
                )
                conjunctive_state["weight_high"].append(
                    rule_high - self.prediction["predict"] if rule_high != np.inf else np.inf
                )
                conjunctive_state["value"].append(
                    factual["value"][f1] + "\n" + conjunctive_state["value"][cf2]
                )
                conjunctive_state["feature"].append(list(original_features))
                conjunctive_state["sampled_values"].append(list(rule_values))
                conjunctive_state["feature_value"].append(None)
                conjunctive_state["rule"].append(
                    factual["rule"][f1] + " & \n" + conjunctive_state["rule"][cf2]
                )
                conjunctive_state["is_conjunctive"].append(True)

    self.conjunctive_rules = conjunctive_state
    self._has_conjunctive_rules = True
    return self


def add_conjunctions_alternative_legacy(self, n_top_features=5, max_rule_size=2):
    """
    Add conjunctive alternative rules (Legacy).

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
        from ..utils.exceptions import ConfigurationError

        raise ConfigurationError(
            "max_rule_size must be 2 or 3",
            details={
                "param": "max_rule_size",
                "value": max_rule_size,
                "valid_range": [2, 3],
            },
        )
    if max_rule_size < 2:
        return self

    alternative = self._get_rules() if not self._has_rules else self.rules

    def _clone_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        cloned: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = value
        return cloned

    conjunctive_state = (
        _clone_payload(self.conjunctive_rules)
        if self._has_conjunctive_rules and self.conjunctive_rules is not None
        else _clone_payload(alternative)
    )

    self._has_conjunctive_rules = False
    self.conjunctive_rules = []

    threshold = None if self.y_threshold is None else self.y_threshold
    scratch = np.array(self.x_test, copy=True)
    predicted_class = alternative["classes"]
    conjunctive_state["classes"] = predicted_class

    if n_top_features is None:
        n_top_features = len(alternative["rule"])

    def _normalise_features(values: Any) -> Tuple[int, ...]:
        if isinstance(values, (list, tuple, np.ndarray)):
            return tuple(sorted(int(v) for v in np.asarray(values).ravel()))
        return (int(values),)

    def _feature_length(candidate: Any) -> int:
        if isinstance(candidate, (list, tuple, np.ndarray)):
            return len(candidate)
        return 1

    for current_size in range(2, max_rule_size + 1):
        num_rules = len(alternative["rule"])
        if num_rules == 0:
            break

        weights_array = np.asarray(conjunctive_state["weight"], dtype=float)
        width_array = np.asarray(conjunctive_state["weight_high"], dtype=float) - np.asarray(
            conjunctive_state["weight_low"], dtype=float
        )
        top_conjunctives = list(
            self._rank_features(
                weights_array,
                width=width_array,
                num_to_show=min(num_rules, n_top_features),
            )
        )

        covered_combinations = {
            _normalise_features(conjunctive_state["feature"][i])
            for i in range(len(conjunctive_state["feature"]))
        }

        for f1, _ in enumerate(alternative["feature"]):
            of1 = alternative["feature"][f1]
            sampled_values1 = alternative["sampled_values"][f1]
            rule_value1 = (
                sampled_values1 if isinstance(sampled_values1, np.ndarray) else [sampled_values1]
            )

            for cf2 in top_conjunctives:
                rule_values = [rule_value1]
                original_features = [of1]
                original_feature_values = [alternative["feature_value"][f1]]
                of2 = conjunctive_state["feature"][cf2]
                target_length = current_size - 1
                if _feature_length(of2) != target_length:
                    continue
                if conjunctive_state["is_conjunctive"][cf2]:
                    if of1 in of2:
                        continue
                    original_features.extend(int(v) for v in of2)
                    rule_values.extend(list(conjunctive_state["sampled_values"][cf2]))
                    original_feature_values.extend(conjunctive_state["feature_value"][cf2])
                else:
                    if of1 == of2:
                        continue
                    original_features.append(of2)
                    original_feature_values.append(alternative["feature_value"][cf2])
                    sampled_values2 = conjunctive_state["sampled_values"][cf2]
                    rule_values.append(
                        sampled_values2
                        if isinstance(sampled_values2, np.ndarray)
                        else [sampled_values2]
                    )

                combo_key = _normalise_features(original_features)
                if combo_key in covered_combinations:
                    continue
                covered_combinations.add(combo_key)

                rule_predict, rule_low, rule_high = _predict_conjunctive_legacy(
                    self,
                    rule_values,
                    original_features,
                    scratch,
                    threshold,
                    predicted_class,
                    bins=self.bin,
                )

                conjunctive_state["predict"].append(rule_predict)
                conjunctive_state["predict_low"].append(rule_low)
                conjunctive_state["predict_high"].append(rule_high)
                conjunctive_state["weight"].append(rule_predict - self.prediction["predict"])
                conjunctive_state["weight_low"].append(
                    rule_low - self.prediction["predict"] if rule_low != -np.inf else -np.inf
                )
                conjunctive_state["weight_high"].append(
                    rule_high - self.prediction["predict"] if rule_high != np.inf else np.inf
                )
                conjunctive_state["value"].append(
                    alternative["value"][f1] + "\n" + conjunctive_state["value"][cf2]
                )
                conjunctive_state["feature"].append(list(original_features))
                conjunctive_state["sampled_values"].append(list(rule_values))
                conjunctive_state["feature_value"].append(list(original_feature_values))
                conjunctive_state["rule"].append(
                    alternative["rule"][f1] + " & \n" + conjunctive_state["rule"][cf2]
                )
                conjunctive_state["is_conjunctive"].append(True)

    self.conjunctive_rules = conjunctive_state
    self._has_conjunctive_rules = True
    return self
