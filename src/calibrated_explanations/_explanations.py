# pylint: disable=too-many-lines, trailing-whitespace, line-too-long, too-many-public-methods
# flake8: noqa: E501
"""contains the CalibratedExplanations class created by the CalibratedExplainer class
"""
import os
import warnings
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from ._discretizers import BinaryDiscretizer, BinaryEntropyDiscretizer, EntropyDiscretizer, DecileDiscretizer

class CalibratedExplanations: # pylint: disable=too-many-instance-attributes
    """
    A class for storing and visualizing calibrated explanations.
    """
    def __init__(self, calibrated_explainer, test_objects, y_threshold) -> None:
        self.calibrated_explainer = calibrated_explainer
        self.test_objects = test_objects
        self.y_threshold = y_threshold
        self.low_high_percentiles = None
        self._has_conjunctive_counterfactual_rules = False
        self._has_counterfactual_rules = False
        self._has_conjunctive_factual_rules = False
        self._has_factual_rules = False
        self.binned = None
        self.feature_weights = None
        self.feature_predict = None
        self.predict = None
        self.rules = []
        self.factual_rules = []
        self.counterfactual_rules = []
        self.conjunctive_factual_rules = []
        self.conjunctive_counterfactual_rules = []
        self.counterfactual_rules = []
        self.counterfactual_labels = {}



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
        """get the confidence of the explanation
        
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




    def _finalize(self, binned, feature_weights, feature_predict, prediction) -> None:
        # """finalize the explanation by adding the binned data and the feature weights
        # """
        self.binned = binned
        self.feature_weights = feature_weights
        self.feature_predict = feature_predict
        self.predict = prediction
        self.calibrated_explainer._set_latest_explanation(self) # pylint: disable=protected-access



    def _define_rules(self, instance):
        # """defines the rule conditions for an instance

        # Args:
        #     instance (n_features,): a test instance

        # Returns:
        #     list[str]: a list of conditioins for each feature in the instance
        # """
        self.rules = []
        # pylint: disable=invalid-name
        x = self.calibrated_explainer.discretizer.discretize(instance)
        for f in range(self.calibrated_explainer.num_features):
            if f in self.calibrated_explainer.categorical_features:
                if self.calibrated_explainer.categorical_labels is not None:
                    try:
                        target = self.calibrated_explainer.categorical_labels[f][int(x[f])]
                        rule = f"{self.calibrated_explainer.feature_names[f]} = {target}"
                    except IndexError:
                        rule = f"{self.calibrated_explainer.feature_names[f]} = {x[f]}"
                else:
                    rule = f"{self.calibrated_explainer.feature_names[f]} = {x[f]}"
            else:
                rule = self.calibrated_explainer.discretizer.names[f][int(x[f])]
            self.rules.append(rule)
        return self.rules



    def __rank_features(self, feature_weights, width=None, num_to_show=None):
        if num_to_show is None or num_to_show > len(feature_weights):
            num_to_show = len(feature_weights)
        # handle case where there are same weight but different uncertainty
        if width is not None:
            # get the indeces by first sorting on the absolute value of the
            # feature_weight and then on the width
            sorted_indices = [i for i, x in
                                sorted(enumerate(list(zip(np.abs(feature_weights), -width))),
                                key=lambda x: (x[1][0], x[1][1]))]
        else:
            sorted_indices = np.argsort(np.abs(feature_weights))
        return sorted_indices[-num_to_show:] # pylint: disable=invalid-unary-operand-type



    def __make_directory(self, path: str, save_ext=None) -> None:
        # """ create directory if it does not exist
        # """
        if save_ext is not None and len(save_ext) == 0:
            return
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        if not os.path.isdir(path):
            os.mkdir(path)



    def _predict_conjunctive(self, rule_value_set, original_features, perturbed, y, # pylint: disable=invalid-name, too-many-locals, too-many-arguments
                            predicted_class):
        # """support function to calculate the prediction for a conjunctive rule
        # """
        rule_predict, rule_low, rule_high, rule_count = 0,0,0,0
        if len(original_features) == 2:
            of1, of2 = original_features[0], original_features[1]
            rule_value1, rule_value2 = rule_value_set[0], rule_value_set[1]
        elif len(original_features) == 3:
            of1, of2, of3 = original_features[0], original_features[1], original_features[2]
            rule_value1, rule_value2, rule_value3 = rule_value_set[0], rule_value_set[1], rule_value_set[2]
        # elif len(original_features) == 4:
        #     of1, of2, of3, of4 = original_features[0], original_features[1], original_features[2], original_features[3]
        #     rule_value1, rule_value2, rule_value3, rule_value4 = rule_value_set[0], rule_value_set[1], rule_value_set[2], rule_value_set[3]    
        for value_1 in rule_value1:
            perturbed[of1] = value_1
            for value_2 in rule_value2:
                perturbed[of2] = value_2
                if len(original_features) >= 3:
                    for value_3 in rule_value3:
                        perturbed[of3] = value_3
                        # if len(original_features) == 4:
                        #     for value_4 in rule_value4:
                        #         perturbed[of4] = value_4
                        #         p_value, low, high, _ = self.calibrated_explainer.predict(perturbed.reshape(1,-1),
                        #                             y=y, low_high_percentiles=self.low_high_percentiles,
                        #                             classes=predicted_class)
                        #         rule_predict += p_value[0]
                        #         rule_low += low[0]
                        #         rule_high += high[0]
                        #         rule_count += 1
                        # else:                       
                        p_value, low, high, _ = self.calibrated_explainer._predict(perturbed.reshape(1,-1), # pylint: disable=protected-access
                                            y=y, low_high_percentiles=self.low_high_percentiles,
                                            classes=predicted_class)
                        rule_predict += p_value[0]
                        rule_low += low[0]
                        rule_high += high[0]
                        rule_count += 1
                else:                    
                    p_value, low, high, _ = self.calibrated_explainer._predict(perturbed.reshape(1,-1), # pylint: disable=protected-access
                                                y=y, low_high_percentiles=self.low_high_percentiles,
                                                classes=predicted_class)
                    rule_predict += p_value[0]
                    rule_low += low[0]
                    rule_high += high[0]
                    rule_count += 1
        rule_predict /= rule_count
        rule_low /= rule_count
        rule_high /= rule_count
        return rule_predict, rule_low, rule_high



    def _get_factual_rules(self):
        # """creates factual rules

        # Returns:
        #     List[Dict[str, List]]: a list of dictionaries containing the factual rules, one for each test instance
        # """
        if self._has_conjunctive_factual_rules:
            return self.conjunctive_factual_rules
        if self._has_factual_rules:
            return self.factual_rules        
        self._has_factual_rules = False
        factual_rules = []
        for i in range(len(self.test_objects)):
            instance = self.test_objects[i, :]
            factual = {'base_predict': [],
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
                        'classes': None, 
                        'is_conjunctive': []
                        }
            factual['classes'] = self.predict['classes'][i]
            factual['base_predict'].append(self.predict['predict'][i])
            factual['base_predict_low'].append(self.predict['low'][i])
            factual['base_predict_high'].append(self.predict['high'][i])
            rules = self._define_rules(instance)
            for f,_ in enumerate(instance): # pylint: disable=invalid-name
                factual['predict'].append(self.feature_predict['predict'][i][f])
                factual['predict_low'].append(self.feature_predict['low'][i][f])
                factual['predict_high'].append(self.feature_predict['high'][i][f])
                factual['weight'].append(self.feature_weights['predict'][i][f])
                factual['weight_low'].append(self.feature_weights['low'][i][f])
                factual['weight_high'].append(self.feature_weights['high'][i][f])
                if f in self.calibrated_explainer.categorical_features:
                    if self.calibrated_explainer.categorical_labels is not None:
                        factual['value'].append(
                            self.calibrated_explainer.categorical_labels[f][int(instance[f])])
                else:
                    factual['value'].append(str(np.around(instance[f],decimals=2)))
                factual['rule'].append(rules[f])
                factual['feature'].append(f)
                factual['feature_value'].append(self.binned['rule_values'][i][f][0][-1])
                factual['is_conjunctive'].append(False)
            factual_rules.append(factual)
        self.factual_rules = factual_rules
        self._has_factual_rules = True
        return self.factual_rules



    # pylint: disable=too-many-statements, too-many-branches
    def _get_counterfactual_rules(self):
        # """creates counterfactual rules

        # Returns:
        #     List[Dict[str, List]]: a list of dictionaries containing the counterfactual rules, one for each test instance
        # """
        if self._has_conjunctive_counterfactual_rules:
            return self.conjunctive_counterfactual_rules
        if self._has_counterfactual_rules:
            return self.counterfactual_rules
        self.counterfactual_rules = []
        self.counterfactual_labels = {}
        for i in range(len(self.test_objects)):
            self.counterfactual_labels[i] = {}
            instance = self.test_objects[i, :]
            discretized = self.calibrated_explainer._discretize(deepcopy(instance).reshape(1,-1))[0] # pylint: disable=protected-access
            instance_predict = self.binned['predict'][i]
            instance_low = self.binned['low'][i]
            instance_high = self.binned['high'][i]
            counterfactual = {'base_predict': [],
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
                            'classes': None, 
                            'is_conjunctive': []
                            }

            counterfactual['classes'] = self.predict['classes'][i]
            counterfactual['base_predict'].append(self.predict['predict'][i])
            counterfactual['base_predict_low'].append(self.predict['low'][i])
            counterfactual['base_predict_high'].append(self.predict['high'][i])
            rule_boundaries = self.calibrated_explainer._rule_boundaries(instance) # pylint: disable=protected-access
            for f,_ in enumerate(instance): # pylint: disable=invalid-name
                if f in self.calibrated_explainer.categorical_features:
                    values = np.array(self.calibrated_explainer.feature_values[f])
                    values = np.delete(values, values == discretized[f])
                    for value_bin, value in enumerate(values):
                        counterfactual['predict'].append(instance_predict[f][value_bin])
                        counterfactual['predict_low'].append(instance_low[f][value_bin])
                        counterfactual['predict_high'].append(instance_high[f][value_bin])
                        counterfactual['weight'].append(instance_predict[f][value_bin] - \
                                                        self.predict['predict'][i])
                        counterfactual['weight_low'].append(instance_low[f][value_bin] - \
                                                            self.predict['predict'][i] \
                                                if instance_low[f][value_bin] != -np.inf \
                                                else instance_low[f][value_bin])
                        counterfactual['weight_high'].append(instance_high[f][value_bin] - \
                                                            self.predict['predict'][i] \
                                                if instance_high[f][value_bin] != np.inf \
                                                else instance_high[f][value_bin])
                        if self.calibrated_explainer.categorical_labels is not None:
                            counterfactual['value'].append(
                                self.calibrated_explainer.categorical_labels[f][int(instance[f])])
                        else:
                            counterfactual['value'].append(str(np.around(instance[f],decimals=2)))
                        counterfactual['feature'].append(f)
                        counterfactual['feature_value'].append(value)
                        if self.calibrated_explainer.categorical_labels is not None:
                            self.counterfactual_labels[i][len(counterfactual['rule'])] = f
                            counterfactual['rule'].append(
                                    f'{self.calibrated_explainer.feature_names[f]} = '+\
                                    f'{self.calibrated_explainer.categorical_labels[f][int(value)]}')
                        else:
                            counterfactual['rule'].append(
                                    f'{self.calibrated_explainer.feature_names[f]} = {value}')
                        counterfactual['is_conjunctive'].append(False)
                else:
                    values = np.array(self.calibrated_explainer.cal_X[:,f])
                    lesser = rule_boundaries[f][0]
                    greater = rule_boundaries[f][1]

                    value_bin = 0
                    if np.any(values < lesser):
                        counterfactual['predict'].append(np.mean(instance_predict[f][value_bin]))
                        counterfactual['predict_low'].append(np.mean(instance_low[f][value_bin]))
                        counterfactual['predict_high'].append(np.mean(instance_high[f][value_bin]))
                        counterfactual['weight'].append(np.mean(instance_predict[f][value_bin]) - \
                                                                self.predict['predict'][i])
                        counterfactual['weight_low'].append(
                                        np.mean(instance_low[f][value_bin]) -
                                        self.predict['predict'][i] \
                                        if instance_low[f][value_bin] != -np.inf \
                                        else instance_low[f][value_bin])
                        counterfactual['weight_high'].append(
                                        np.mean(instance_high[f][value_bin]) -
                                        self.predict['predict'][i] \
                                        if instance_high[f][value_bin] != np.inf \
                                        else instance_high[f][value_bin])
                        counterfactual['value'].append(str(np.around(instance[f],decimals=2)))
                        counterfactual['feature'].append(f)
                        counterfactual['feature_value'].append(
                                        self.binned['rule_values'][i][f][0][0])
                        counterfactual['rule'].append(
                                        f'{self.calibrated_explainer.feature_names[f]} < {lesser}')
                        counterfactual['is_conjunctive'].append(False)
                        value_bin = 1

                    if np.any(values > greater):
                        counterfactual['predict'].append(np.mean(instance_predict[f][value_bin]))
                        counterfactual['predict_low'].append(np.mean(instance_low[f][value_bin]))
                        counterfactual['predict_high'].append(np.mean(instance_high[f][value_bin]))
                        counterfactual['weight'].append(
                                        np.mean(instance_predict[f][value_bin]) -
                                        self.predict['predict'][i])
                        counterfactual['weight_low'].append(np.mean(instance_low[f][value_bin]) -
                                                                    self.predict['predict'][i] \
                                        if instance_low[f][value_bin] != -np.inf \
                                        else instance_low[f][value_bin])
                        counterfactual['weight_high'].append(np.mean(instance_high[f][value_bin]) -
                                                                    self.predict['predict'][i] \
                                        if instance_high[f][value_bin] != np.inf \
                                        else instance_high[f][value_bin])
                        counterfactual['value'].append(str(np.around(instance[f],decimals=2)))
                        counterfactual['feature'].append(f)
                        counterfactual['feature_value'].append(
                                        self.binned['rule_values'][i][f][0][1 \
                                        if len(self.binned['rule_values'][i][f][0]) == 3 else 0])
                        counterfactual['rule'].append(
                                        f'{self.calibrated_explainer.feature_names[f]} > {greater}')
                        counterfactual['is_conjunctive'].append(False)

            self.counterfactual_rules.append(counterfactual)
        self._has_counterfactual_rules = True
        return self.counterfactual_rules



    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        """_summary_

        Args:
            n_top_features (int, optional): the number of most important factual rules to try to combine into conjunctive rules. Defaults to 5.
            max_rule_size (int, optional): the maximum size of the conjunctions. Defaults to 2 (meaning `rule_one and rule_two`).

        Returns:
            CalibratedExplanations: Returns a self reference, to allow for method chaining
        """
        if self._is_counterfactual():
            return self._add_conjunctive_counterfactual_rules(n_top_features, max_rule_size)
        return self._add_conjunctive_factual_rules(n_top_features, max_rule_size)



    def remove_conjunctions(self):
        """removes any conjunctive rules"""
        self._has_conjunctive_counterfactual_rules = False
        self._has_conjunctive_factual_rules = False
        return self



    # pylint: disable=too-many-locals
    def _add_conjunctive_factual_rules(self, n_top_features=5, max_rule_size=2):
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
        if not self._has_factual_rules:
            factuals = deepcopy(self._get_factual_rules())
        else:
            factuals = deepcopy(self.factual_rules)
        if self._has_conjunctive_factual_rules:
            conjunctives = self.conjunctive_factual_rules
        else:
            conjunctives = factuals
        self._has_conjunctive_factual_rules = False
        self.conjunctive_factual_rules = []
        for i in range(len(self.test_objects)):
            factual = deepcopy(factuals[i])
            conjunctive = conjunctives[i]
            # pylint: disable=unsubscriptable-object, invalid-name
            y = None if self.y_threshold is None else self.y_threshold \
                    if np.isscalar(self.y_threshold) else self.y_threshold[i]
            x_original = deepcopy(self.test_objects[i, :])

            num_rules = len(factual['rule'])
            predicted_class = factual['classes']
            conjunctive['classes'] = predicted_class
            if n_top_features is None:
                n_top_features = num_rules
            # top_factuals = self.__rank_features(np.reshape(factual['weight'], (len(factual['weight']))), 
            #                     width=np.reshape(np.array(factual['weight_high']) - np.array(factual['weight_low']),
            #                     (len(factual['weight']))), num_to_show=np.min([num_rules, n_top_features]))
            top_conjunctives = self.__rank_features(np.reshape(conjunctive['weight'], (len(conjunctive['weight']))), 
                                width=np.reshape(np.array(conjunctive['weight_high']) - np.array(conjunctive['weight_low']),
                                (len(conjunctive['weight']))), num_to_show=np.min([num_rules, n_top_features]))

            covered_features = []
            covered_combinations = [conjunctive['feature'][i] for i in range(len(conjunctive['rule']))]
            for _, cf1 in enumerate(factual['feature']): # cf = factual feature
                covered_features.append(cf1)
                of1 = factual['feature'][cf1] # of = original feature
                rule_value1 = factual['feature_value'][cf1] \
                                if isinstance(factual['feature_value'][cf1], np.ndarray) \
                                else [factual['feature_value'][cf1]]
                for _, cf2 in enumerate(top_conjunctives): # cf = conjunctive feature
                    if cf2 in covered_features:
                        continue
                    rule_values = [rule_value1]
                    original_features = [of1]
                    if conjunctive['is_conjunctive'][cf2]:
                        of2 = conjunctive['feature'][cf2]
                        if of1 in of2:
                            continue
                        for of in of2:
                            original_features.append(of)
                        for rule_value in conjunctive['feature_value'][cf2]:
                            rule_values.append(rule_value)
                    else:
                        of2 = conjunctive['feature'][cf2] # of = original feature
                        if of1 == of2:
                            continue
                        original_features.append(of2)
                        rule_values.append(conjunctive['feature_value'][cf2] \
                                    if isinstance(conjunctive['feature_value'][cf2], np.ndarray) \
                                    else [conjunctive['feature_value'][cf2]])
                    skip = False
                    for ofs in covered_combinations:
                        if np.all(np.sort(original_features) == ofs):
                            skip = True
                            break
                    if skip:
                        continue
                    covered_combinations.append(np.sort(original_features))

                    rule_predict, rule_low, rule_high = self._predict_conjunctive(rule_values,
                                                                            original_features,
                                                                            deepcopy(x_original),
                                                                            y,
                                                                            predicted_class)

                    conjunctive['predict'].append(rule_predict)
                    conjunctive['predict_low'].append(rule_low)
                    conjunctive['predict_high'].append(rule_high)
                    conjunctive['weight'].append(rule_predict - self.predict['predict'][i])
                    conjunctive['weight_low'].append(rule_low - self.predict['predict'][i] \
                                                    if rule_low != -np.inf else -np.inf)
                    conjunctive['weight_high'].append(rule_high - self.predict['predict'][i] \
                                                    if rule_high != np.inf else np.inf)
                    conjunctive['value'].append(factual['value'][cf1]+ '\n' +conjunctive['value'][cf2])
                    conjunctive['feature'].append(original_features)
                    conjunctive['feature_value'].append(rule_values)
                    conjunctive['rule'].append(factual['rule'][cf1]+ ' & \n' +conjunctive['rule'][cf2])
                    conjunctive['is_conjunctive'].append(True)
            self.conjunctive_factual_rules.append(conjunctive)
        self._has_conjunctive_factual_rules = True
        return self._add_conjunctive_factual_rules(n_top_features=n_top_features, max_rule_size=max_rule_size-1)



    # pylint: disable=too-many-locals
    def _add_conjunctive_counterfactual_rules(self, n_top_features=5, max_rule_size=2):
        # """adds conjunctive counterfactual rules

        # Args:
        #     n_top_features (int, optional): the number of most important counterfactual rules to try to combine into conjunctive rules. Defaults to 5.
        #     max_rule_size (int, optional): the maximum size of the conjunctions. Defaults to 2 (meaning `rule_one and rule_two`).

        # Returns:
        #     CalibratedExplanations: Returns a self reference, to allow for method chaining
        # """
        if max_rule_size >= 4:
            raise ValueError('max_rule_size must be 2 or 3')
        if max_rule_size < 2:
            return self
        if not self._has_factual_rules:
            counterfactuals = deepcopy(self._get_counterfactual_rules())
        else:
            counterfactuals = deepcopy(self.counterfactual_rules)
        if self._has_conjunctive_counterfactual_rules:
            conjunctives = self.conjunctive_counterfactual_rules
        else:
            conjunctives = counterfactuals
        if self._has_conjunctive_counterfactual_rules:
            return self
        counterfactuals = deepcopy(self._get_counterfactual_rules())
        self.conjunctive_counterfactual_rules = []
        for i in range(len(self.test_objects)):
            counterfactual = deepcopy(counterfactuals[i])
            conjunctive = conjunctives[i]
            # pylint: disable=unsubscriptable-object, invalid-name
            y = None if self.y_threshold is None else self.y_threshold \
                                if np.isscalar(self.y_threshold) else self.y_threshold[i]
            x_original = deepcopy(self.test_objects[i, :])

            num_rules = len(counterfactual['rule'])
            predicted_class = counterfactual['classes']
            conjunctive['classes'] = predicted_class
            if n_top_features is None:
                n_top_features = num_rules
            # top_factuals = self.__rank_features(np.reshape(factual['weight'], (len(factual['weight']))), 
            #                     width=np.reshape(np.array(factual['weight_high']) - np.array(factual['weight_low']),
            #                     (len(factual['weight']))), num_to_show=np.min([num_rules, n_top_features]))
            top_conjunctives = self.__rank_features(np.reshape(conjunctive['weight'], (len(conjunctive['weight']))), 
                                width=np.reshape(np.array(conjunctive['weight_high']) - np.array(conjunctive['weight_low']),
                                (len(conjunctive['weight']))), num_to_show=np.min([num_rules, n_top_features]))

            covered_features = []
            covered_combinations = [conjunctive['feature'][i] for i in range(len(conjunctive['rule']))]
            for _, cf1 in enumerate(counterfactual['feature']): # cf = factual feature
                covered_features.append(cf1)
                of1 = counterfactual['feature'][cf1] # of = original feature
                rule_value1 = counterfactual['feature_value'][cf1] \
                                if isinstance(counterfactual['feature_value'][cf1], np.ndarray) \
                                else [counterfactual['feature_value'][cf1]]
                for _, cf2 in enumerate(top_conjunctives): # cf = conjunctive feature
                    if cf2 in covered_features:
                        continue
                    rule_values = [rule_value1]
                    original_features = [of1]
                    if conjunctive['is_conjunctive'][cf2]:
                        of2 = conjunctive['feature'][cf2]
                        if of1 in of2:
                            continue
                        for of in of2:
                            original_features.append(of)
                        for rule_value in conjunctive['feature_value'][cf2]:
                            rule_values.append(rule_value)
                    else:
                        of2 = conjunctive['feature'][cf2] # of = original feature
                        if of1 == of2:
                            continue
                        original_features.append(of2)
                        rule_values.append(conjunctive['feature_value'][cf2] \
                                    if isinstance(conjunctive['feature_value'][cf2], np.ndarray) \
                                    else [conjunctive['feature_value'][cf2]])
                    skip = False
                    for ofs in covered_combinations:
                        if np.all(np.sort(original_features) == ofs):
                            skip = True
                            break
                    if skip:
                        continue
                    covered_combinations.append(np.sort(original_features))

                    rule_predict, rule_low, rule_high = self._predict_conjunctive(rule_values,
                                                                            original_features,
                                                                            deepcopy(x_original),
                                                                            y,
                                                                            predicted_class)

                    conjunctive['predict'].append(rule_predict)
                    conjunctive['predict_low'].append(rule_low)
                    conjunctive['predict_high'].append(rule_high)
                    conjunctive['weight'].append(rule_predict - self.predict['predict'][i])
                    conjunctive['weight_low'].append(rule_low - self.predict['predict'][i] \
                            if rule_low != -np.inf else -np.inf)
                    conjunctive['weight_high'].append(rule_high - self.predict['predict'][i] \
                            if rule_high != np.inf else np.inf)
                    conjunctive['value'].append(counterfactual['value'][cf1] + '\n' + \
                                                conjunctive['value'][cf2])
                    conjunctive['feature'].append(original_features)
                    conjunctive['feature_value'].append(rule_values)
                    conjunctive['rule'].append(counterfactual['rule'][cf1] + ' & \n' + \
                                                conjunctive['rule'][cf2])
                    conjunctive['is_conjunctive'].append(True)
            self.conjunctive_counterfactual_rules.append(conjunctive)
        self._has_conjunctive_counterfactual_rules = True
        return self._add_conjunctive_counterfactual_rules(n_top_features=n_top_features, max_rule_size=max_rule_size-1)



    def _check_preconditions(self, counterfactuals=False):
        # """checks that the recommended discretizer is used for the type of explanation

        # Args:
        #     counterfactuals (bool, optional): if true, the check assumes a counterfactual explanation, otherwise factual explanations are assumed. Defaults to False.
        # """
        if self == self.calibrated_explainer.latest_explanation:
            if counterfactuals:
                if 'regression' in self.calibrated_explainer.mode:
                    if not isinstance(self.calibrated_explainer.discretizer, DecileDiscretizer):
                        warnings.warn('Counterfactual explanations for regression recommend using the ' +\
                                        'decile discretizer. Consider extracting counterfactual ' +\
                                        'explanations using `explainer.explain_counterfactual(test_set)`')
                else:
                    if not isinstance(self.calibrated_explainer.discretizer, EntropyDiscretizer):
                        warnings.warn('Counterfactual explanations for classification recommend using ' +\
                                        'the entropy discretizer. Consider extracting counterfactual ' +\
                                        'explanations using `explainer.explain_counterfactual(test_set)`')

            else:
                if 'regression' in self.calibrated_explainer.mode:
                    if not isinstance(self.calibrated_explainer.discretizer, BinaryDiscretizer):
                        warnings.warn('Factual explanations for regression recommend using the binary ' +\
                                        'discretizer. Consider extracting factual explanations using ' +\
                                        '`explainer.explain_factual(test_set)`')
                else:
                    if not isinstance(self.calibrated_explainer.discretizer, BinaryEntropyDiscretizer):
                        warnings.warn('Factual explanations for classification recommend using the ' +\
                                        'binaryEntropy discretizer. Consider extracting factual ' +\
                                        'explanations using `explainer.explain_factual(test_set)`')


    # pylint: disable=dangerous-default-value, too-many-arguments, too-many-locals
    def _plot_regular(self, title="", n_features_to_show=10, show=False,
                    path='plots/', save_ext=['svg','pdf','png']):
        # """creates regular plots for factual explanations

        # Args:
        #     title (str, optional): The title of each plot. Defaults to "".
        #     n_features_to_show (int, optional): number of features to include in the plot. None == all features. Defaults to 10.
        #     show (bool, optional): determines whether created plots are shown, to avoid showing plots that are saved to disk. Defaults to False.
        #     path (str, optional): path to saving location. Defaults to 'plots/'.
        #     save_ext (list, optional): the file formats used to save plots to disk. Defaults to ['svg','pdf','png'].
        # """
        self._check_preconditions()
        factuals = self._get_factual_rules()
        predict = self.predict
        num_features = len(factuals[0]['weight'])
        num_instances = len(factuals)
        if n_features_to_show is None:
            n_features_to_show = num_features
        n_features_to_show = np.min([num_features, n_features_to_show])

        self.__make_directory(path+title, save_ext=save_ext)

        for i in range(num_instances):
            factual = factuals[i]
            feature_weights = factual['weight']
            width = np.reshape(np.array(factual['weight_high']) - np.array(factual['weight_low']),
                            (len(factual['weight'])))
            features_to_plot = self.__rank_features(feature_weights,
                                                    width=width,
                                                    num_to_show=n_features_to_show)
            column_names = factual['rule']
            if 'classification' in self.calibrated_explainer.mode or self._is_thresholded():
                self.__plot_probabilistic(factual['value'], predict, feature_weights, features_to_plot,
                            n_features_to_show, column_names, title, str(i), path, show, idx=i,
                            save_ext=save_ext)
            else:                
                self.__plot_regression(factual['value'], predict, feature_weights, features_to_plot,
                            n_features_to_show, column_names, title, str(i), path, show, idx=i,
                            save_ext=save_ext)


    # pylint: disable=dangerous-default-value
    def _plot_uncertainty(self,
                        title="",
                        n_features_to_show=10,
                        show=False,
                        path='plots/',
                        save_ext=['svg','pdf','png']):
        # """creates uncertainty plots for factual explanations

        # Args:
        #     title (str, optional): The title of each plot. Defaults to "".
        #     n_features_to_show (int, optional): number of features to include in the plot. None == all features. Defaults to 10.
        #     show (bool, optional): determines whether created plots are shown, to avoid showing plots that are saved to disk. Defaults to False.
        #     path (str, optional): path to saving location. Defaults to 'plots/'.
        #     save_ext (list, optional): the file formats used to save plots to disk. Defaults to ['svg','pdf','png'].
        # """
        self._check_preconditions()
        factuals = self._get_factual_rules()
        predict = self.predict
        num_features = len(factuals[0]['weight'])
        num_instances = len(factuals)
        if n_features_to_show is None:
            n_features_to_show = num_features
        n_features_to_show = np.min([num_features, n_features_to_show])

        self.__make_directory(path+title, save_ext=save_ext)

        for i in range(num_instances):
            factual = factuals[i]
            feature_weights = {'predict':factual['weight'],
                                'low':factual['weight_low'], 
                                'high':factual['weight_high']}
            width = np.reshape(np.array(factual['weight_high']) - np.array(factual['weight_low']),
                                (len(factual['weight'])))
            features_to_plot = self.__rank_features(feature_weights['predict'],
                                                    width=width,
                                                    num_to_show=n_features_to_show)
            column_names = factual['rule']
            if 'classification' in self.calibrated_explainer.mode or self._is_thresholded():
                self.__plot_probabilistic(factual['value'], predict, feature_weights, features_to_plot,
                                n_features_to_show, column_names, title, str(i), path, show,
                                interval=True, idx=i, save_ext=save_ext)
            else:
                self.__plot_regression(factual['value'], predict, feature_weights, features_to_plot,
                                n_features_to_show, column_names, title, str(i), path, show,
                                interval=True, idx=i, save_ext=save_ext)


    # pylint: disable=dangerous-default-value
    def _plot_counterfactuals(self,
                            title="",
                            n_features_to_show=10,
                            show=False,
                            path='plots/',
                            save_ext=['svg','pdf','png']):
        # """creates plots for counterfactual explanations

        # Args:
        #     title (str, optional): The title of each plot. Defaults to "".
        #     n_features_to_show (int, optional): number of features to include in the plot. None == all features. Defaults to 10.
        #     show (bool, optional): determines whether created plots are shown, to avoid showing plots that are saved to disk. Defaults to False.
        #     path (str, optional): path to saving location. Defaults to 'plots/'.
        #     save_ext (list, optional): the file formats used to save plots to disk. Defaults to ['svg','pdf','png'].
        # """
        self._check_preconditions(counterfactuals=True)
        predict = self.predict
        counterfactuals = self._get_counterfactual_rules()

        self.__make_directory(path+title, save_ext=save_ext)

        for i, _ in enumerate(self.test_objects):
            counterfactual = counterfactuals[i]
            feature_predict = {'predict': counterfactual['predict'],
                                'low': counterfactual['predict_low'], 
                                'high': counterfactual['predict_high']}
            feature_weights = np.reshape(counterfactual['weight'],
                                        (len(counterfactual['weight'])))
            width = np.reshape(np.array(counterfactual['weight_high']) -
                                np.array(counterfactual['weight_low']),
                                (len(counterfactual['weight'])))
            num_rules = len(counterfactual['rule'])
            if n_features_to_show is None:
                n_features_to_show = num_rules
            num_to_show_ = np.min([num_rules, n_features_to_show])
            features_to_plot = self.__rank_features(feature_weights,
                                                    width=width,
                                                    num_to_show=num_to_show_)
            column_names = counterfactual['rule']
            self.__plot_counterfactual(counterfactual['value'], predict, feature_predict, \
                                        features_to_plot, num_to_show=num_to_show_, \
                                        column_names=column_names, title=title, postfix=str(i), \
                                        path=path, show=show, idx=i, save_ext=save_ext)

    def get_explanation(self, instance_index):
        '''The function `get_explanation` returns either a counterfactual rule or a factual rule based on
        the value of `self._is_counterfactual()`.
        
        Parameters
        ----------
        instance_index
            The `instance_index` parameter is an integer that represents the index of the explanation
        instance that you want to retrieve. It is used to specify which explanation instance you want to
        get from either the counterfactual rules or the factual rules.
        
        Returns
        -------
            The method `get_explanation` returns either a counterfactual rule or a factual rule, depending
        on the condition `self._is_counterfactual()`. If the condition is true, it returns the
        counterfactual rule at the specified `instance_index` from the list of counterfactual rules. If
        the condition is false, it returns the factual rule at the specified `instance_index` from the
        
        '''
        assert isinstance(instance_index, int), "instance_index must be an integer"
        assert instance_index >= 0, "instance_index must be greater than or equal to 0"
        assert instance_index < len(self.test_objects), "instance_index must be less than the number of test instances"
        if self._is_counterfactual():
            return self._get_counterfactual_rules()[instance_index]
        return self._get_factual_rules()[instance_index]

    def _is_counterfactual(self):
        # '''The function checks if the explanations are counterfactuals by checking if the `discretizer` attribute of the `calibrated_explainer` object is an
        # instance of either `DecileDiscretizer` or `EntropyDiscretizer`.
        
        # Returns
        # -------
        #     a boolean value indicating whether the explanations are counterfactuals.        
        # '''        
        return isinstance(self.calibrated_explainer.discretizer, (DecileDiscretizer, EntropyDiscretizer))


    def plot_all(self,
                n_features_to_show=10,
                show=False,
                path='',
                uncertainty=False):
        '''The function `plot_all` plots either counterfactual or factual explanations for a given
        instance, with the option to show or save the plots.
        
        Parameters
        ----------
        n_features_to_show, optional
            The parameter "n_features_to_show" determines the number of top features to display in the
            plot. It specifies how many of the most important features should be shown in the plot.
        show, optional
            The "show" parameter determines whether the plots should be displayed immediately after they
            are generated. If set to True, the plots will be shown; if set to False, the plots will not be
            shown.
        path, optional
            The `path` parameter is the directory path where the plots will be saved. If you don't provide
            a value for `path`, the plots will not be saved and will only be displayed if `show` is set to
            `True`.
        uncertainty, optional
            The "uncertainty" parameter is a boolean flag that determines whether to include uncertainty
            information in the plots. If set to True, the plots will show uncertainty measures, if
            available, along with the explanations. If set to False, the plots will only show the
            explanations without uncertainty information.
        
        '''
        is_counterfactual = self._is_counterfactual()
        self._check_preconditions(counterfactuals=is_counterfactual)
        for instance_index in range(len(self.test_objects)):
            if is_counterfactual:
                self.plot_counterfactual(instance_index, n_features_to_show, show, path)
            else:
                self.plot_factual(instance_index, n_features_to_show, show, path, uncertainty)



    def plot_factual(self, instance_index, n_features_to_show=10, show=False, full_filename='', uncertainty=False):
        '''This function plots the factual explanation for a given instance using either probabilistic or
        regression plots.
        
        Parameters
        ----------
        instance_index : int
            The index of the instance for which you want to plot the factual explanation.
        factual
            The `factual` parameter is a dictionary that contains the explanation for a specific instance.
            It includes the following keys:
        n_features_to_show, optional
            The `n_features_to_show` parameter determines the number of top features to display in the
            plot. If set to `None`, it will show all the features. Otherwise, it will show the specified
            number of features, up to the total number of features available.
        show, optional
            A boolean parameter that determines whether the plot should be displayed or not. If set to
            True, the plot will be displayed. If set to False, the plot will not be displayed.
        full_filename
            The full_filename parameter is a string that represents the full path and filename of the plot
            image file that will be saved. If this parameter is not provided or is an empty string, the plot
            will not be saved as an image file.
        uncertainty, optional
            The `uncertainty` parameter is a boolean flag that determines whether to plot the uncertainty
        intervals for the feature weights. If `uncertainty` is set to `True`, the plot will show the
        range of possible feature weights based on the lower and upper bounds of the uncertainty
        intervals. If `uncertainty
        
        '''
        factual = self.get_explanation(instance_index)
        self._check_preconditions()
        predict = self.predict
        num_features = len(factual['weight'])
        if n_features_to_show is None:
            n_features_to_show = num_features
        n_features_to_show = np.min([num_features, n_features_to_show])

        if len(full_filename) > 0:
            path = os.path.dirname(full_filename)
            filename = os.path.basename(full_filename)
            title, ext = os.path.splitext(filename)
            self.__make_directory(path+title, save_ext=np.array([ext]))
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
        features_to_plot = self.__rank_features(factual['weight'],
                                                width=width,
                                                num_to_show=n_features_to_show)
        column_names = factual['rule']
        if 'classification' in self.calibrated_explainer.mode or self._is_thresholded():
            self.__plot_probabilistic(factual['value'], predict, feature_weights, features_to_plot,
                        n_features_to_show, column_names, title=title, postfix=str(instance_index), path=path, interval=uncertainty, show=show, idx=instance_index,
                        save_ext=save_ext)
        else:                
            self.__plot_regression(factual['value'], predict, feature_weights, features_to_plot,
                        n_features_to_show, column_names, title=title, postfix=str(instance_index), path=path, interval=uncertainty, show=show, idx=instance_index,
                        save_ext=save_ext)


    def plot_counterfactual(self, instance_index, n_features_to_show=10, show=False, full_filename=''):        
        '''The function `plot_counterfactual` plots the counterfactual explanation for a given instance in
        a dataset.
        
        Parameters
        ----------
        instance_index
            The index of the instance for which you want to plot the counterfactual explanation.
        n_features_to_show, optional
            The parameter "n_features_to_show" determines the number of features to show in the plot. It
            specifies how many of the top-ranked features based on their weights and widths should be
            displayed in the plot.
        show, optional
            A boolean parameter that determines whether or not to display the plot. If set to True, the
            plot will be displayed. If set to False, the plot will not be displayed.
        full_filename
            The full_filename parameter is a string that represents the full path and filename of the plot
            image file that will be saved. If this parameter is not provided or an empty string is passed,
            the plot will not be saved as an image file.
        
        '''
        counterfactual = self.get_explanation(instance_index)
        self._check_preconditions(counterfactuals=True)      
        predict = self.predict      
        if len(full_filename) > 0:
            path = os.path.dirname(full_filename)
            filename = os.path.basename(full_filename)
            title, ext = os.path.splitext(filename)
            self.__make_directory(path+title, save_ext=np.array([ext]))
            save_ext = [ext]
        else:
            path = ''
            title = ''
            save_ext = []
        feature_predict = {'predict': counterfactual['predict'],
                            'low': counterfactual['predict_low'], 
                            'high': counterfactual['predict_high']}
        feature_weights = np.reshape(counterfactual['weight'],
                                        (len(counterfactual['weight'])))
        width = np.reshape(np.array(counterfactual['weight_high']) -
                            np.array(counterfactual['weight_low']),
                            (len(counterfactual['weight'])))
        num_rules = len(counterfactual['rule'])
        if n_features_to_show is None:
            n_features_to_show = num_rules
        num_to_show_ = np.min([num_rules, n_features_to_show])
        features_to_plot = self.__rank_features(feature_weights,
                                                width=width,
                                                num_to_show=num_to_show_)
        column_names = counterfactual['rule']
        self.__plot_counterfactual(counterfactual['value'], predict, feature_predict, \
                                    features_to_plot, num_to_show=num_to_show_, \
                                    column_names=column_names, title=title, postfix='', \
                                    path=path, show=show, idx=instance_index, save_ext=save_ext)



    # pylint: disable=dangerous-default-value, too-many-arguments, too-many-locals, invalid-name, too-many-branches, too-many-statements
    def __plot_counterfactual(self, instance, predict, feature_predict, features_to_plot, \
                            num_to_show, column_names, title, postfix, path, show,
                            idx=None, save_ext=['svg','pdf','png']):
        """plots counterfactual explanations"""
        fig = plt.figure(figsize=(10,num_to_show*.5))
        ax_main = fig.add_subplot(111)

        x = np.linspace(0, num_to_show-1, num_to_show)
        p_l = predict['low'][idx] if predict['low'][idx] != -np.inf \
                            else min(self.calibrated_explainer.cal_y)
        p_h = predict['high'][idx] if predict['high'][idx] != np.inf \
                            else max(self.calibrated_explainer.cal_y)
        p = predict['predict'][idx]
        venn_abers={'low_high': [p_l,p_h],'predict':p}
        # Fill original Venn Abers interval
        xl = np.linspace(-0.5, x[0], 2)
        xh = np.linspace(x[-1], x[-1]+0.5, 2)
        if (p_l < 0.5 and p_h < 0.5) or (p_l > 0.5 and p_h > 0.5) or \
                            'regression' in self.calibrated_explainer.mode:
            color = self.__get_fill_color({'predict':1},0.15) \
                            if 'regression' in self.calibrated_explainer.mode \
                            else self.__get_fill_color(venn_abers,0.15)
            ax_main.fill_betweenx(x, [p_l]*(num_to_show), [p_h]*(num_to_show),color=color)
            # Fill up to the edges
            ax_main.fill_betweenx(xl, [p_l]*(2), [p_h]*(2),color=color)
            ax_main.fill_betweenx(xh, [p_l]*(2), [p_h]*(2),color=color)
            if 'regression' in self.calibrated_explainer.mode:
                ax_main.fill_betweenx(x, p, p, color='r', alpha=0.3)  
                # Fill up to the edges                
                ax_main.fill_betweenx(xl, p, p, color='r', alpha=0.3)  
                ax_main.fill_betweenx(xh, p, p, color='r', alpha=0.3)           
        else:
            venn_abers['predict'] = p_l
            color = self.__get_fill_color(venn_abers, 0.15)
            ax_main.fill_betweenx(x, [p_l]*(num_to_show), [0.5]*(num_to_show),color=color)
            # Fill up to the edges
            ax_main.fill_betweenx(xl, [p_l]*(2), [0.5]*(2),color=color)
            ax_main.fill_betweenx(xh, [p_l]*(2), [0.5]*(2),color=color)
            venn_abers['predict'] = p_h
            color = self.__get_fill_color(venn_abers, 0.15)
            ax_main.fill_betweenx(x, [0.5]*(num_to_show), [p_h]*(num_to_show),color=color)
            # Fill up to the edges
            ax_main.fill_betweenx(xl, [0.5]*(2), [p_h]*(2),color=color)
            ax_main.fill_betweenx(xh, [0.5]*(2), [p_h]*(2),color=color)

        for jx, j in enumerate(features_to_plot):
            p_l = feature_predict['low'][j] if feature_predict['low'][j] != -np.inf \
                                            else min(self.calibrated_explainer.cal_y)
            p_h = feature_predict['high'][j] if feature_predict['high'][j] != np.inf \
                                            else max(self.calibrated_explainer.cal_y)
            p = feature_predict['predict'][j]
            xj = np.linspace(x[jx]-0.2, x[jx]+0.2,2)
            venn_abers={'low_high': [p_l,p_h],'predict':p}
            # Fill each feature impact
            if 'regression' in self.calibrated_explainer.mode:
                ax_main.fill_betweenx(xj, p_l,p_h, color='r', alpha= 0.40)
                ax_main.fill_betweenx(xj, p, p, color='r')  
            elif (p_l < 0.5 and p_h < 0.5) or (p_l > 0.5 and p_h > 0.5) :
                ax_main.fill_betweenx(xj, p_l,p_h,color=self.__get_fill_color(venn_abers, 0.99))
            else:
                venn_abers['predict'] = p_l
                ax_main.fill_betweenx(xj, p_l,0.5,color=self.__get_fill_color(venn_abers, 0.99))
                venn_abers['predict'] = p_h
                ax_main.fill_betweenx(xj, 0.5,p_h,color=self.__get_fill_color(venn_abers, 0.99))

        ax_main.set_yticks(range(num_to_show))
        ax_main.set_yticklabels(labels=[column_names[i] for i in features_to_plot]) \
            if column_names is not None else ax_main.set_yticks(range(num_to_show)) # pylint: disable=expression-not-assigned
        ax_main.set_ylim(-0.5,x[-1]+0.5)
        ax_main.set_ylabel('Counterfactual rules')
        ax_main_twin = ax_main.twinx()
        ax_main_twin.set_yticks(range(num_to_show))
        ax_main_twin.set_yticklabels([instance[i] for i in features_to_plot])
        ax_main_twin.set_ylim(-0.5,x[-1]+0.5)
        ax_main_twin.set_ylabel('Instance values')
        if self._is_thresholded():
            # pylint: disable=unsubscriptable-object
            if np.isscalar(self.y_threshold):
                ax_main.set_xlabel('Probability of target being below '+\
                                    f'{float(self.y_threshold) :.2f}')
            else:
                ax_main.set_xlabel('Probability of target being below '+\
                                    f'{float(self.y_threshold[idx]) :.2f}') 
            ax_main.set_xlim(0,1)
            ax_main.set_xticks(np.linspace(0, 1, 11))
        elif 'regression' in self.calibrated_explainer.mode:
            ax_main.set_xlabel(f'Prediction interval with {self.get_confidence()}% confidence')
            ax_main.set_xlim([min(self.calibrated_explainer.cal_y),
                        max(self.calibrated_explainer.cal_y)])
        else:
            if self.calibrated_explainer.class_labels is not None:
                if self.calibrated_explainer._is_multiclass(): # pylint: disable=protected-access
                    ax_main.set_xlabel('Probability for class '+\
                                f'\'{self.calibrated_explainer.class_labels[self.predict["classes"][idx]]}\'') # pylint: disable=line-too-long
                else:
                    ax_main.set_xlabel('Probability for class '+\
                                f'\'{self.calibrated_explainer.class_labels[1]}\'')
            else:
                if self.calibrated_explainer._is_multiclass(): # pylint: disable=protected-access
                    ax_main.set_xlabel(f'Probability for class \'{self.predict["classes"][idx]}\'')
                else:
                    ax_main.set_xlabel('Probability for the positive class')
            ax_main.set_xlim(0,1)
            ax_main.set_xticks(np.linspace(0, 1, 11))

        fig.tight_layout()
        for ext in save_ext:
            fig.savefig(path + title + '/' + title + '_' + postfix +'.'+ext, bbox_inches='tight')
        if show:
            fig.show()


    def __plot_probabilistic(self, instance, predict, feature_weights, features_to_plot, num_to_show,
                    column_names, title, postfix, path, show, interval=False, idx=None,
                    save_ext=['svg','pdf','png']):
        """plots regular and uncertainty explanations"""
        if interval is True:
            assert idx is not None
        fig = plt.figure(figsize=(10,num_to_show*.5+2))
        subfigs = fig.subfigures(4, 1, height_ratios=[1, 1, 1, num_to_show+2])

        if interval and (self._is_one_sided()):
            raise Warning('Interval plot is not supported for one-sided explanations.')

        ax_positive = subfigs[0].add_subplot(111)
        ax_negative = subfigs[1].add_subplot(111)
        
        ax_main = subfigs[3].add_subplot(111)

        # plot the probabilities at the top
        x = np.linspace(0, 1, 2)
        xj = np.linspace(x[0]-0.2, x[0]+0.2,2)
        p = predict['predict'][idx]
        pl = predict['low'][idx] if predict['low'][idx] != -np.inf \
                                else min(self.calibrated_explainer.cal_y)
        ph = predict['high'][idx] if predict['high'][idx] != np.inf \
                                else max(self.calibrated_explainer.cal_y)

        ax_negative.fill_betweenx(xj, 1-p, 1-p, color='b')
        ax_negative.fill_betweenx(xj, 0, 1-ph, color='b')
        ax_negative.fill_betweenx(xj, 1-pl, 1-ph, color='b', alpha=0.2)
        ax_negative.set_xlim([0,1])
        ax_negative.set_yticks(range(1))
        ax_negative.set_xticks(np.linspace(0,1,6))
        ax_positive.fill_betweenx(xj, p, p, color='r')
        ax_positive.fill_betweenx(xj, 0, pl, color='r')
        ax_positive.fill_betweenx(xj, pl, ph, color='r', alpha=0.2)
        ax_positive.set_xlim([0,1])
        ax_positive.set_yticks(range(1))
        ax_positive.set_xticks([])

        if self._is_thresholded():
            if np.isscalar(self.y_threshold):
                ax_negative.set_yticklabels(labels=[f'P(y>{float(self.y_threshold) :.2f})'])
                ax_positive.set_yticklabels(labels=[f'P(y<={float(self.y_threshold) :.2f})'])
            else:                    
                ax_negative.set_yticklabels(labels=[f'P(y>{float(self.y_threshold[idx]) :.2f})']) # pylint: disable=unsubscriptable-object
                ax_positive.set_yticklabels(labels=[f'P(y<={float(self.y_threshold[idx]) :.2f})']) # pylint: disable=unsubscriptable-object
        else:
            if self.calibrated_explainer.class_labels is not None:
                if self.calibrated_explainer._is_multiclass(): # pylint: disable=protected-access
                    ax_negative.set_yticklabels(labels=[f'P(y!={self.calibrated_explainer.class_labels[self.predict["classes"][idx]]})']) # pylint: disable=line-too-long
                    ax_positive.set_yticklabels(labels=[f'P(y={self.calibrated_explainer.class_labels[self.predict["classes"][idx]]})']) # pylint: disable=line-too-long
                else:
                    ax_negative.set_yticklabels(labels=[f'P(y={self.calibrated_explainer.class_labels[0]})']) # pylint: disable=line-too-long
                    ax_positive.set_yticklabels(labels=[f'P(y={self.calibrated_explainer.class_labels[1]})']) # pylint: disable=line-too-long
            else: 
                if self.calibrated_explainer._is_multiclass(): # pylint: disable=protected-access
                    ax_negative.set_yticklabels(labels=[f'P(y!={self.predict["classes"][idx]})'])
                    ax_positive.set_yticklabels(labels=[f'P(y={self.predict["classes"][idx]})'])
                else:
                    ax_negative.set_yticklabels(labels=['P(y=0)'])
                    ax_positive.set_yticklabels(labels=['P(y=1)'])
        ax_negative.set_xlabel('Probability')

        # Plot the base prediction in black/grey
        x = np.linspace(0, num_to_show-1, num_to_show)
        xl = np.linspace(-0.5, x[0], 2)
        xh = np.linspace(x[-1], x[-1]+0.5, 2)
        ax_main.fill_betweenx(x, [0], [0], color='k')
        ax_main.fill_betweenx(xl, [0], [0], color='k')
        ax_main.fill_betweenx(xh, [0], [0], color='k')
        if interval:           
            p = predict['predict'][idx]
            gwl = predict['low'][idx] - p
            gwh = predict['high'][idx] - p
            
            gwh, gwl = np.max([gwh, gwl]), np.min([gwh, gwl])
            ax_main.fill_betweenx([-0.5,num_to_show-0.5], gwl, gwh, color='k', alpha=0.2)

        # For each feature, plot the weight
        for jx, j in enumerate(features_to_plot):
            xj = np.linspace(x[jx]-0.2, x[jx]+0.2,2)
            min_val,max_val = 0,0
            if interval:
                width = feature_weights['predict'][j]
                wl = feature_weights['low'][j]
                wh = feature_weights['high'][j]
                wh, wl = np.max([wh, wl]), np.min([wh, wl])
                max_val = wh if width < 0 else 0
                min_val = wl if width > 0 else 0
                # If uncertainty cover zero, then set to 0 to avoid solid plotting
                if wl < 0 < wh:
                    min_val = 0
                    max_val = 0
            else:                
                width = feature_weights[j]
                min_val = width if width < 0 else 0
                max_val = width if width > 0 else 0
            color = 'b' if width > 0 else 'r'
            ax_main.fill_betweenx(xj, min_val, max_val, color=color)
            if interval:
                if wl < 0 < wh and self.calibrated_explainer.mode == 'classification':
                    ax_main.fill_betweenx(xj, 0, wl, color='r', alpha=0.2)
                    ax_main.fill_betweenx(xj, wh, 0, color='b', alpha=0.2)
                else:
                    ax_main.fill_betweenx(xj, wl, wh, color=color, alpha=0.2)

        ax_main.set_yticks(range(num_to_show))
        ax_main.set_yticklabels(labels=[column_names[i] for i in features_to_plot]) \
            if column_names is not None else ax_main.set_yticks(range(num_to_show)) # pylint: disable=expression-not-assigned
        ax_main.set_ylim(-0.5,x[-1]+0.5)
        ax_main.set_ylabel('Rules')
        ax_main.set_xlabel('Feature weights')
        ax_main_twin = ax_main.twinx()
        ax_main_twin.set_yticks(range(num_to_show))
        ax_main_twin.set_yticklabels([instance[i] for i in features_to_plot])
        ax_main_twin.set_ylim(-0.5,x[-1]+0.5)
        ax_main_twin.set_ylabel('Instance values')
        for ext in save_ext:
            fig.savefig(path + title + '/' + title + '_' + postfix +'.'+ext,
                    bbox_inches='tight') 
        if show:
            fig.show()


    # pylint: disable=dangerous-default-value, too-many-branches, too-many-statements
    def __plot_regression(self, instance, predict, feature_weights, features_to_plot, num_to_show,
                    column_names, title, postfix, path, show, interval=False, idx=None,
                    save_ext=['svg','pdf','png']):
        """plots regular and uncertainty explanations"""
        if interval is True:
            assert idx is not None
        fig = plt.figure(figsize=(10,num_to_show*.5+2))
        subfigs = fig.subfigures(2, 1, height_ratios=[1, num_to_show+2])

        if interval and (self._is_one_sided()):
            raise Warning('Interval plot is not supported for one-sided explanations.')

        ax_regression = subfigs[0].add_subplot(111)        
        ax_main = subfigs[1].add_subplot(111)

        # plot the probabilities at the top
        x = np.linspace(0, 1, 2)
        xj = np.linspace(x[0]-0.2, x[0]+0.2,2)
        p = predict['predict'][idx]
        pl = predict['low'][idx] if predict['low'][idx] != -np.inf \
                                else min(self.calibrated_explainer.cal_y)
        ph = predict['high'][idx] if predict['high'][idx] != np.inf \
                                else max(self.calibrated_explainer.cal_y)
        
        ax_regression.fill_betweenx(xj, pl, ph, color='r', alpha=0.2)
        ax_regression.fill_betweenx(xj, p, p, color='r')
        ax_regression.set_xlim([min(self.calibrated_explainer.cal_y),max(self.calibrated_explainer.cal_y)])
        ax_regression.set_yticks(range(1))

        ax_regression.set_xlabel(f'Prediction interval with {self.get_confidence()}% confidence')
        ax_regression.set_yticklabels(labels=['Median prediction'])

        # Plot the base prediction in black/grey
        x = np.linspace(0, num_to_show-1, num_to_show)
        xl = np.linspace(-0.5, x[0], 2)
        xh = np.linspace(x[-1], x[-1]+0.5, 2)
        ax_main.fill_betweenx(x, [0], [0], color='k')
        ax_main.fill_betweenx(xl, [0], [0], color='k')
        ax_main.fill_betweenx(xh, [0], [0], color='k')
        if interval:            
            p = predict['predict'][idx]
            gwl = p - predict['low'][idx]
            gwh = p - predict['high'][idx]
            
            gwh, gwl = np.max([gwh, gwl]), np.min([gwh, gwl])
            ax_main.fill_betweenx([-0.5,num_to_show-0.5], gwl, gwh, color='k', alpha=0.2)

        # For each feature, plot the weight
        for jx, j in enumerate(features_to_plot):
            xj = np.linspace(x[jx]-0.2, x[jx]+0.2,2)
            min_val,max_val = 0,0
            if interval:
                width = - feature_weights['predict'][j]
                wl = - feature_weights['low'][j]
                wh = - feature_weights['high'][j]
                wh, wl = np.max([wh, wl]), np.min([wh, wl])
                max_val = wh if width < 0 else 0
                min_val = wl if width > 0 else 0
                # If uncertainty cover zero, then set to 0 to avoid solid plotting
                if wl < 0 < wh:
                    min_val = 0
                    max_val = 0
            else:                
                width = - feature_weights[j]
                min_val = width if width < 0 else 0
                max_val = width if width > 0 else 0
            color = 'b' if width > 0 else 'r'
            ax_main.fill_betweenx(xj, min_val, max_val, color=color)
            if interval:
                ax_main.fill_betweenx(xj, wl, wh, color=color, alpha=0.2)

        ax_main.set_yticks(range(num_to_show))
        ax_main.set_yticklabels(labels=[column_names[i] for i in features_to_plot]) \
            if column_names is not None else ax_main.set_yticks(range(num_to_show)) # pylint: disable=expression-not-assigned
        ax_main.set_ylim(-0.5,x[-1]+0.5)
        ax_main.set_ylabel('Rules')
        ax_main.set_xlabel('Feature weights')
        ax_main_twin = ax_main.twinx()
        ax_main_twin.set_yticks(range(num_to_show))
        ax_main_twin.set_yticklabels([instance[i] for i in features_to_plot])
        ax_main_twin.set_ylim(-0.5,x[-1]+0.5)
        ax_main_twin.set_ylabel('Instance values')
        for ext in save_ext:
            fig.savefig(path + title + '/' + title + '_' + postfix +'.'+ext,
                    bbox_inches='tight') 
        if show:
            fig.show()



    def as_lime(self):
        """transforms the explanation into a lime explanation object

        Returns:
            lime explanation object with the same values as the explanation
        """
        _, lime_exp = self.calibrated_explainer.preload_lime()
        exp = []
        for i in range(len(self.test_objects[:,0])):
            tmp = deepcopy(lime_exp)
            tmp.intercept[1] = 0
            tmp.local_pred = self.predict['predict'][i]
            if 'regression' in self.calibrated_explainer.mode:
                tmp.predicted_value = self.predict['predict'][i]
                tmp.min_value = min(self.calibrated_explainer.cal_y)
                tmp.max_value = max(self.calibrated_explainer.cal_y)
            else:
                tmp.predict_proba[0], tmp.predict_proba[1] = \
                        1-self.predict['predict'][i], self.predict['predict'][i]

            feature_weights = self.feature_weights['predict'][i]
            features_to_plot = self.__rank_features(feature_weights, 
                        num_to_show=self.calibrated_explainer.num_features)
            rules = self._define_rules(self.test_objects[i, :])
            for j,f in enumerate(features_to_plot[::-1]): # pylint: disable=invalid-name
                tmp.local_exp[1][j] = (f, feature_weights[f])
            tmp.domain_mapper.discretized_feature_names = rules
            tmp.domain_mapper.feature_values = self.test_objects[i, :]
            exp.append(tmp)
        return exp



    def as_shap(self):
        """transforms the explanation into a shap explanation object

        Returns:
            shap explanation object with the same values as the explanation
        """
        _, shap_exp = self.calibrated_explainer.preload_shap(len(self.test_objects[:,0]))
        for i in range(len(self.test_objects[:,0])):
            shap_exp.base_values[i] = self.predict['predict'][i]
            for f in range(len(self.test_objects[0, :])):
                shap_exp.values[i][f] = -self.feature_weights['predict'][i][f]
        return shap_exp


    # pylint: disable=invalid-name
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
        winner_class = int(venn_abers['predict']>= 0.5)
        color = colors[winner_class]

        alpha = venn_abers['predict'] if winner_class==1 else 1-venn_abers['predict']
        alpha = ((alpha - 0.5) / (1 - 0.5)) * (1-.25) + .25 # normalize values to the range [.25,1]
        if reduction != 1:
            alpha = reduction

        # unpack numpy scalars
        alpha = float(alpha)
        # compute the color as alpha against white
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return '#%2x%2x%2x' % tuple(color) # pylint: disable=consider-using-f-string
