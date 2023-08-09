# pylint: disable=too-many-lines, trailing-whitespace, line-too-long
# flake8: noqa: E501
"""contains the CalibratedExplanation class created by the CalibratedExplainer class
"""
import os
import warnings
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class CalibratedExplanation: # pylint: disable=too-many-instance-attributes
    """
    A class for storing and visualizing calibrated explanations.
    """
    def __init__(self, calibrated_explainer, test_objects) -> None:
        self.calibrated_explainer = calibrated_explainer
        self.test_objects = test_objects
        self.test_targets = None
        self.low_high_percentiles = None
        self._has_conjunctive_counterfactual_rules = False
        self._has_counterfactual_rules = False
        self._has_conjunctive_regular_rules = False
        self.binned = None
        self.feature_weights = None
        self.feature_predict = None
        self.predict = None
        self.rules = []
        self.factual_rules = []
        self.counterfactual_rules = []
        self.conjunctive_regular_rules = []
        self.conjunctive_counterfactual_rules = []
        self.counterfactuals = []
        self.counterfactual_labels = {}



    def is_thresholded(self) -> bool:
        """test if the explanation is thresholded

        Returns:
            bool: True if the test_targets is not None
        """
        return self.test_targets is not None



    def is_one_sided(self) -> bool:
        """test if a regression explanation is one-sided

        Returns:
            bool: True if one of the low or high percentiles is infinite
        """
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
        elif np.isinf(self.get_low_percentile()):
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




    def finalize(self, binned, feature_weights, feature_predict, prediction) -> None:
        """finalize the explanation by adding the binned data and the feature weights
        """
        self.binned = binned
        self.feature_weights = feature_weights
        self.feature_predict = feature_predict
        self.predict = prediction
        self.calibrated_explainer.set_latest_explanation(self)



    def define_rules(self, instance):
        """defines the rule conditions for an instance

        Args:
            instance (n_features,): a test instance

        Returns:
            list[str]: a list of conditioins for each feature in the instance
        """
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
        """ create directory if it does not exist
        """
        if save_ext is not None and len(save_ext) == 0:
            return
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        if not os.path.isdir(path):
            os.mkdir(path)



    def predict_conjunctive(self, rule_value1, rule_value2, of1, of2, perturbed, y, # pylint: disable=invalid-name, too-many-locals, too-many-arguments
                            predicted_class):
        """support function to calculate the prediction for a conjunctive rule
        """
        rule_predict, rule_low, rule_high, rule_count = 0,0,0,0
        for value_1 in rule_value1:
            perturbed[of1] = value_1
            for value_2 in rule_value2:
                perturbed[of2] = value_2
                p_value, low, high, _ = self.calibrated_explainer.predict(perturbed.reshape(1,-1),
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



    def get_factual_rules(self):
        """creates factual rules

        Returns:
            List[Dict[str, List]]: a list of dictionaries containing the factual rules, one for each test instance
        """
        if self._has_conjunctive_regular_rules:
            return self.conjunctive_regular_rules
        factual_rules = []
        for i in range(len(self.test_objects)):
            instance = self.test_objects[i, :]
            factual = {'weight': [],
                       'weight_low': [],
                       'weight_high': [],
                       'predict': [],
                       'predict_low': [],
                       'predict_high': [],
                       'value': [],
                       'rule': [],
                       'feature': [],
                       'feature_value': [],
                       'classes': []}
            factual['classes'].append(self.predict['classes'][i])
            rules = self.define_rules(instance)
            for f,_ in enumerate(instance): # pylint: disable=invalid-name
                factual['weight'].append(self.feature_weights['predict'][i][f])
                factual['weight_low'].append(self.feature_weights['low'][i][f])
                factual['weight_high'].append(self.feature_weights['high'][i][f])
                factual['predict'].append(self.feature_predict['predict'][i][f])
                factual['predict_low'].append(self.feature_predict['low'][i][f])
                factual['predict_high'].append(self.feature_predict['high'][i][f])
                if f in self.calibrated_explainer.categorical_features:
                    if self.calibrated_explainer.categorical_labels is not None:
                        factual['value'].append(
                            self.calibrated_explainer.categorical_labels[f][int(instance[f])])
                else:
                    factual['value'].append(str(np.around(instance[f],decimals=2)))
                factual['rule'].append(rules[f])
                factual['feature'].append(f)
                factual['feature_value'].append(self.binned['rule_values'][i][f][0][-1])
            factual_rules.append(factual)
        self.factual_rules = factual_rules
        return self.factual_rules



    def get_counterfactual_rules(self):
        """creates counterfactual rules

        Returns:
            List[Dict[str, List]]: a list of dictionaries containing the counterfactual rules, one for each test instance
        """
        if self._has_conjunctive_counterfactual_rules:
            return self.conjunctive_counterfactual_rules
        self.counterfactuals = []
        self.counterfactual_labels = {}
        for i in range(len(self.test_objects)):
            self.counterfactual_labels[i] = {}
            instance = self.test_objects[i, :]
            discretized = self.calibrated_explainer.discretize(deepcopy(instance).reshape(1,-1))[0]
            instance_predict = self.binned['predict'][i]
            instance_low = self.binned['low'][i]
            instance_high = self.binned['high'][i]
            counterfactual = {'weight': [],
                              'weight_low': [],
                              'weight_high': [],
                              'predict': [],
                              'predict_low': [],
                              'predict_high': [],
                              'value': [],
                              'rule': [],
                              'feature': [],
                              'feature_value': [],
                              'classes': []}

            counterfactual['classes'].append(self.predict['classes'][i])
            rule_boundaries = self.calibrated_explainer.rule_boundaries(instance)
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
                                    f'{self.calibrated_explainer.feature_names[f]} = \
                                    {self.calibrated_explainer.categorical_labels[f][int(value)]}')
                        else:
                            counterfactual['rule'].append(
                                    f'{self.calibrated_explainer.feature_names[f]} = {value}')
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

            self.counterfactuals.append(counterfactual)
        self._has_counterfactual_rules = True
        return self.counterfactuals



    def add_conjunctive_factual_rules(self, num_to_include=5):
        """adds conjunctive factual rules

        Args:
            num_to_include (int, optional): the number of most important factual rules to try to combine into conjunctive rules. Defaults to 5.

        Returns:
            List[Dict[str, List]]: a list of dictionaries containing the factual rules (including conjunctive rules), one for each test instance
        """
        if self._has_conjunctive_regular_rules:
            return self
        factuals = self.get_factual_rules()
        self._has_conjunctive_regular_rules = False
        self.conjunctive_regular_rules = []
        for i in range(len(self.test_objects)):
            factual = factuals[i]
            # pylint: disable=unsubscriptable-object, invalid-name
            y = None if self.test_targets is None else self.test_targets \
                    if np.isscalar(self.test_targets) else self.test_targets[i]
            x_original = deepcopy(self.test_objects[i, :])
            conjunctive = factual

            feature_weights = np.reshape(factual['weight'], (len(factual['weight'])))
            width = np.reshape(np.array(factual['weight_high']) - np.array(factual['weight_low']),
                               (len(factual['weight'])))
            num_rules = len(factual['rule'])
            predicted_class = factual['classes'][0]
            conjunctive['classes'] = predicted_class
            if num_to_include is None:
                num_to_include = num_rules
            features_to_plot = self.__rank_features(feature_weights, width=width,
                                                    num_to_show=np.min([num_rules, num_to_include]))

            for j, cf1 in enumerate(features_to_plot): # cf = counterfactual feature
                of1 = factual['feature'][cf1] # of = original feature
                rule_value1 = factual['feature_value'][cf1] \
                                if isinstance(factual['feature_value'][cf1], np.ndarray) \
                                else [factual['feature_value'][cf1]]
                for cf2 in features_to_plot[j+1:]: # cf = counterfactual feature
                    of2 = factual['feature'][cf2] # of = original feature
                    if of1 == of2:
                        continue
                    rule_value2 = factual['feature_value'][cf2] \
                                    if isinstance(factual['feature_value'][cf2], np.ndarray) \
                                    else [factual['feature_value'][cf2]]

                    rule_predict, rule_low, rule_high = self.predict_conjunctive(rule_value1,
                                                                            rule_value2,
                                                                            of1,
                                                                            of2,
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
                    conjunctive['value'].append(factual['value'][cf1]+ '\n' +factual['value'][cf2])
                    conjunctive['feature'].append((of1,cf1,of2,cf2))
                    conjunctive['feature_value'].append((rule_value1,rule_value2))
                    conjunctive['rule'].append(factual['rule'][cf1]+ ' & \n' +factual['rule'][cf2])
            self.conjunctive_regular_rules.append(conjunctive)
        self._has_conjunctive_regular_rules = True
        return self



    def add_conjunctive_counterfactual_rules(self, num_to_include=5):
        """adds conjunctive counterfactual rules

        Args:
            num_to_include (int, optional): the number of most important counterfactual rules to try to combine into conjunctive rules. Defaults to 5.

        Returns:
            List[Dict[str, List]]: a list of dictionaries containing the counterfactual rules (including conjunctive rules), one for each test instance
        """
        if self._has_counterfactual_rules:
            counterfactuals = self.counterfactuals
        else:
            counterfactuals = self.get_counterfactual_rules()
        self.conjunctive_counterfactual_rules = []
        for i in range(len(self.test_objects)):
            counterfactual = counterfactuals[i]
            # pylint: disable=unsubscriptable-object, invalid-name
            y = None if self.test_targets is None else self.test_targets \
                                if np.isscalar(self.test_targets) else self.test_targets[i]
            x_original = deepcopy(self.test_objects[i, :])
            conjunctive = counterfactual

            feature_weights = np.reshape(counterfactual['weight'], (len(counterfactual['weight'])))
            width = np.reshape(np.array(counterfactual['weight_high']) -
                               np.array(counterfactual['weight_low']),
                               (len(counterfactual['weight'])))
            num_rules = len(counterfactual['rule'])
            predicted_class = counterfactual['classes'][0]
            conjunctive['classes'] = predicted_class
            if num_to_include is None:
                num_to_include = num_rules
            features_to_plot = self.__rank_features(feature_weights,
                                                    width=width,
                                                    num_to_show=np.min([num_rules, num_to_include]))

            for j, cf1 in enumerate(features_to_plot): # cf = counterfactual feature
                of1 = counterfactual['feature'][cf1] # of = original feature
                rule_value1 = counterfactual['feature_value'][cf1] \
                        if isinstance(counterfactual['feature_value'][cf1], np.ndarray) \
                        else [counterfactual['feature_value'][cf1]]
                for cf2 in features_to_plot[j+1:]: # cf = counterfactual feature
                    of2 = counterfactual['feature'][cf2] # of = original feature
                    if of1 == of2:
                        continue
                    rule_value2 = counterfactual['feature_value'][cf2] \
                            if isinstance(counterfactual['feature_value'][cf2], np.ndarray) \
                            else [counterfactual['feature_value'][cf2]]

                    rule_predict, rule_low, rule_high = self.predict_conjunctive(rule_value1,
                                                                            rule_value2,
                                                                            of1,
                                                                            of2,
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
                                                counterfactual['value'][cf2])
                    conjunctive['feature'].append((of1,cf1,of2,cf2))
                    conjunctive['feature_value'].append((rule_value1,rule_value2))
                    conjunctive['rule'].append(counterfactual['rule'][cf1] + ' & \n' + \
                                               counterfactual['rule'][cf2])
            self.conjunctive_counterfactual_rules.append(conjunctive)
        self._has_conjunctive_counterfactual_rules = True
        return self



    def remove_conjunctive_rules(self):
        """removes any conjunctive factual rules"""
        self._has_conjunctive_counterfactual_rules = False
        self._has_conjunctive_regular_rules = False
        return self



    def remove_counterfactual_rules(self):
        """removes any conjunctive counterfactual rules"""
        self._has_counterfactual_rules = False
        return self



    def check_preconditions(self, counterfactuals=False):
        """checks that the recommended discretizer is used for the type of explanation

        Args:
            counterfactuals (bool, optional): if true, the check assumes a counterfactual explanation, otherwise factual explanations are assumed. Defaults to False.
        """
        if counterfactuals:
            if 'regression' in self.calibrated_explainer.mode:
                if self.calibrated_explainer.discretizer != 'decile':
                    warnings.warn('Counterfactual explanations for regressoin recommend using the \
                                    decile discretizer. Consider extracting counterfactual \
                                    explanations using `explainer.get_counterfactuals(test_set)`')
            else:
                if self.calibrated_explainer.discretizer != 'entropy':
                    warnings.warn('Counterfactual explanations for classification recommend using \
                                    the entropy discretizer. Consider extracting counterfactual \
                                    explanations using `explainer.get_counterfactuals(test_set)`')

        else:
            if 'regression' in self.calibrated_explainer.mode:
                if self.calibrated_explainer.discretizer != 'binary':
                    warnings.warn('Factual explanations for regressoin recommend using the decile \
                                    discretizer. Consider extracting factual explanations using \
                                    `explainer.get_factuals(test_set)`')
            else:
                if self.calibrated_explainer.discretizer != 'binaryEntropy':
                    warnings.warn('Factual explanations for classification recommend using the \
                                    binaryEntropy discretizer. Consider extracting factual \
                                    explanations using `explainer.get_factuals(test_set)`')


    # pylint: disable=dangerous-default-value, too-many-arguments, too-many-locals
    def plot_regular(self, title="", n_features_to_show=10, show=False,
                     path='plots/', save_ext=['svg','pdf','png']):
        """creates regular plots for factual explanations

        Args:
            title (str, optional): The title of each plot. Defaults to "".
            n_features_to_show (int, optional): number of features to include in the plot. None == all features. Defaults to 10.
            show (bool, optional): determines whether created plots are shown, to avoid showing plots that are saved to disk. Defaults to False.
            path (str, optional): path to saving location. Defaults to 'plots/'.
            save_ext (list, optional): the file formats used to save plots to disk. Defaults to ['svg','pdf','png'].
        """
        self.check_preconditions()
        factuals = self.get_factual_rules()
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
            self.__plot_weight(factual['value'], predict, feature_weights, features_to_plot,
                               n_features_to_show, column_names, title, str(i), path, show, idx=i,
                               save_ext=save_ext)


    # pylint: disable=dangerous-default-value
    def plot_uncertainty(self,
                         title="",
                         n_features_to_show=10,
                         show=False,
                         path='plots/',
                         save_ext=['svg','pdf','png']):
        """creates uncetainty plots for factual explanations

        Args:
            title (str, optional): The title of each plot. Defaults to "".
            n_features_to_show (int, optional): number of features to include in the plot. None == all features. Defaults to 10.
            show (bool, optional): determines whether created plots are shown, to avoid showing plots that are saved to disk. Defaults to False.
            path (str, optional): path to saving location. Defaults to 'plots/'.
            save_ext (list, optional): the file formats used to save plots to disk. Defaults to ['svg','pdf','png'].
        """
        self.check_preconditions()
        factuals = self.get_factual_rules()
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
            self.__plot_weight(factual['value'], predict, feature_weights, features_to_plot,
                               n_features_to_show, column_names, title, str(i), path, show,
                               interval=True, idx=i, save_ext=save_ext)


    # pylint: disable=dangerous-default-value
    def plot_counterfactuals(self,
                             title="",
                             n_features_to_show=10,
                             show=False,
                             path='plots/',
                             save_ext=['svg','pdf','png']):
        """creates plots for counterfactual explanations

        Args:
            title (str, optional): The title of each plot. Defaults to "".
            n_features_to_show (int, optional): number of features to include in the plot. None == all features. Defaults to 10.
            show (bool, optional): determines whether created plots are shown, to avoid showing plots that are saved to disk. Defaults to False.
            path (str, optional): path to saving location. Defaults to 'plots/'.
            save_ext (list, optional): the file formats used to save plots to disk. Defaults to ['svg','pdf','png'].
        """
        self.check_preconditions(counterfactuals=True)
        predict = self.predict
        counterfactuals = self.get_counterfactual_rules()

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


    # pylint: disable=dangerous-default-value, too-many-arguments, too-many-locals, invalid-name, too-many-branches, too-many-statements
    def __plot_counterfactual(self, instance, predict, feature_predict, features_to_plot, \
                              num_to_show, column_names, title, postfix, path, show,
                              idx=None, save_ext=['svg','pdf','png']):
        """plots counterfactual explanations"""
        fig = plt.figure(figsize=(10,num_to_show*.5))
        ax1 = fig.add_subplot(111)

        x = np.linspace(0, num_to_show-1, num_to_show)
        p_l = predict['low'][idx] if predict['low'][idx] != -np.inf \
                            else min(self.calibrated_explainer.calY)
        p_h = predict['high'][idx] if predict['high'][idx] != np.inf \
                            else max(self.calibrated_explainer.calY)
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
            ax1.fill_betweenx(x, [p_l]*(num_to_show), [p_h]*(num_to_show),color=color)
            # Fill up to the edges
            ax1.fill_betweenx(xl, [p_l]*(2), [p_h]*(2),color=color)
            ax1.fill_betweenx(xh, [p_l]*(2), [p_h]*(2),color=color)
            if 'regression' in self.calibrated_explainer.mode:
                ax1.fill_betweenx(x, p, p, color='r', alpha=0.3)  
                # Fill up to the edges                
                ax1.fill_betweenx(xl, p, p, color='r', alpha=0.3)  
                ax1.fill_betweenx(xh, p, p, color='r', alpha=0.3)           
        else:
            venn_abers['predict'] = p_l
            color = self.__get_fill_color(venn_abers, 0.15)
            ax1.fill_betweenx(x, [p_l]*(num_to_show), [0.5]*(num_to_show),color=color)
            # Fill up to the edges
            ax1.fill_betweenx(xl, [p_l]*(2), [0.5]*(2),color=color)
            ax1.fill_betweenx(xh, [p_l]*(2), [0.5]*(2),color=color)
            venn_abers['predict'] = p_h
            color = self.__get_fill_color(venn_abers, 0.15)
            ax1.fill_betweenx(x, [0.5]*(num_to_show), [p_h]*(num_to_show),color=color)
            # Fill up to the edges
            ax1.fill_betweenx(xl, [0.5]*(2), [p_h]*(2),color=color)
            ax1.fill_betweenx(xh, [0.5]*(2), [p_h]*(2),color=color)

        for jx, j in enumerate(features_to_plot):
            p_l = feature_predict['low'][j] if feature_predict['low'][j] != -np.inf \
                                            else min(self.calibrated_explainer.calY)
            p_h = feature_predict['high'][j] if feature_predict['high'][j] != np.inf \
                                             else max(self.calibrated_explainer.calY)
            p = feature_predict['predict'][j]
            xj = np.linspace(x[jx]-0.2, x[jx]+0.2,2)
            venn_abers={'low_high': [p_l,p_h],'predict':p}
            # Fill each feature impact
            if 'regression' in self.calibrated_explainer.mode:
                ax1.fill_betweenx(xj, p_l,p_h, color='r', alpha= 0.40)
                ax1.fill_betweenx(xj, p, p, color='r')  
            elif (p_l < 0.5 and p_h < 0.5) or (p_l > 0.5 and p_h > 0.5) :
                ax1.fill_betweenx(xj, p_l,p_h,color=self.__get_fill_color(venn_abers, 0.99))
            else:
                venn_abers['predict'] = p_l
                ax1.fill_betweenx(xj, p_l,0.5,color=self.__get_fill_color(venn_abers, 0.99))
                venn_abers['predict'] = p_h
                ax1.fill_betweenx(xj, 0.5,p_h,color=self.__get_fill_color(venn_abers, 0.99))

        ax1.set_yticks(range(num_to_show))
        ax1.set_yticklabels(labels=[column_names[i] for i in features_to_plot]) \
            if column_names is not None else ax1.set_yticks(range(num_to_show)) # pylint: disable=expression-not-assigned
        ax1.set_ylim(-0.5,x[-1]+0.5)
        ax1.set_ylabel('Counterfactual rules')
        ax2 = ax1.twinx()
        ax2.set_yticks(range(num_to_show))
        ax2.set_yticklabels([instance[i] for i in features_to_plot])
        ax2.set_ylim(-0.5,x[-1]+0.5)
        ax2.set_ylabel('Instance values')
        if self.is_thresholded():
            # pylint: disable=unsubscriptable-object
            if np.isscalar(self.test_targets):
                ax1.set_xlabel(f'Probability of target being above \
                                    {float(self.test_targets) :.2f}')
            else:
                ax1.set_xlabel(f'Probability of target being above \
                                    {float(self.test_targets[idx]) :.2f}') 
            ax1.set_xlim(0,1)
            ax1.set_xticks(np.linspace(0, 1, 11))
        elif 'regression' in self.calibrated_explainer.mode:
            ax1.set_xlabel(f'Prediction interval with {self.get_confidence()}% confidence')
            ax1.set_xlim([min(self.calibrated_explainer.calY),
                          max(self.calibrated_explainer.calY)])
        else:
            if self.calibrated_explainer.class_labels is not None:
                if self.calibrated_explainer.is_multiclass():
                    ax1.set_xlabel(f'Probability for class \
                                \'{self.calibrated_explainer.class_labels[self.predict["classes"][idx]]}\'') # pylint: disable=line-too-long
                else:
                    ax1.set_xlabel(f'Probability for class \
                                \'{self.calibrated_explainer.class_labels[1]}\'')
            else:
                if self.calibrated_explainer.is_multiclass():
                    ax1.set_xlabel(f'Probability for class \'{self.predict["classes"][idx]}\'')
                else:
                    ax1.set_xlabel('Probability for the positive class')
            ax1.set_xlim(0,1)
            ax1.set_xticks(np.linspace(0, 1, 11))

        fig.tight_layout()
        for ext in save_ext:
            fig.savefig(path + title + '/' + title + '_' + postfix +'.'+ext, bbox_inches='tight')
        if show:
            fig.show()


    # pylint: disable=dangerous-default-value, too-many-branches, too-many-statements
    def __plot_weight(self, instance, predict, feature_weights, features_to_plot, num_to_show,
                      column_names, title, postfix, path, show, interval=False, idx=None,
                      save_ext=['svg','pdf','png']):
        """plots regular and uncertainty explanations"""
        if interval is True:
            assert idx is not None
        fig = plt.figure(figsize=(10,num_to_show*.5+2))
        subfigs = fig.subfigures(4, 1, height_ratios=[1, 1, 1, num_to_show+2])

        if interval and (self.is_one_sided()):
            raise Warning('Interval plot is not supported for one-sided explanations.')

        if self.calibrated_explainer.mode == 'classification' or self.is_thresholded():
            ax00 = subfigs[0].add_subplot(111)
            ax01 = subfigs[1].add_subplot(111)
        else:
            ax01 = subfigs[2].add_subplot(111)
        
        ax1 = subfigs[3].add_subplot(111)

        # plot the probabilities
        x = np.linspace(0, 1, 2)
        xj = np.linspace(x[0]-0.2, x[0]+0.2,2)
        p = predict['predict'][idx]
        pl = predict['low'][idx] if predict['low'][idx] != -np.inf \
                                 else min(self.calibrated_explainer.calY)
        ph = predict['high'][idx] if predict['high'][idx] != np.inf \
                                  else max(self.calibrated_explainer.calY)
        if self.calibrated_explainer.mode == 'classification':
            ax00.fill_betweenx(xj, 1-p, 1-p, color='b')
            ax00.fill_betweenx(xj, 0, 1-ph, color='b')
            ax00.fill_betweenx(xj, 1-pl, 1-ph, color='b', alpha=0.2)
            ax01.fill_betweenx(xj, p, p, color='r')
            ax01.fill_betweenx(xj, 0, pl, color='r')
            ax01.fill_betweenx(xj, pl, ph, color='r', alpha=0.2)
            ax00.set_xlim([0,1])
            ax01.set_xlim([0,1])
            ax00.set_yticks(range(1))
            ax00.set_xticks([])
        elif ('regression' in self.calibrated_explainer.mode and self.is_thresholded()):
            ax00.fill_betweenx(xj, 0, 1-p, color='b')
            ax01.fill_betweenx(xj, 0, p, color='r')
            ax00.set_xlim([0,1])
            ax01.set_xlim([0,1])
            ax00.set_yticks(range(1))
            ax00.set_xticks([])
            
        else:     
            ax01.fill_betweenx(xj, pl, ph, color='r', alpha=0.2)
            ax01.fill_betweenx(xj, p, p, color='r')
            ax01.set_xlim([min(self.calibrated_explainer.calY),max(self.calibrated_explainer.calY)])
        ax01.set_yticks(range(1))
        
        if 'regression' in self.calibrated_explainer.mode:
            if self.is_thresholded():
                if np.isscalar(self.test_targets):
                    ax00.set_yticklabels(labels=[f'P(y<{float(self.test_targets) :.2f})'])
                    ax01.set_yticklabels(labels=[f'P(y>={float(self.test_targets) :.2f})'])
                else:                    
                    ax00.set_yticklabels(labels=[f'P(y<{float(self.test_targets[idx]) :.2f})']) # pylint: disable=unsubscriptable-object
                    ax01.set_yticklabels(labels=[f'P(y>={float(self.test_targets[idx]) :.2f})']) # pylint: disable=unsubscriptable-object
                ax01.set_xlabel('Probability')
            else:
                ax01.set_xlabel(f'Prediction interval with {self.get_confidence()}% confidence')
                ax01.set_yticklabels(labels=['Median prediction'])
        else:
            if self.calibrated_explainer.class_labels is not None:
                if self.calibrated_explainer.is_multiclass():
                    ax00.set_yticklabels(labels=[f'P(y!={self.calibrated_explainer.class_labels[self.predict["classes"][idx]]})']) # pylint: disable=line-too-long
                    ax01.set_yticklabels(labels=[f'P(y={self.calibrated_explainer.class_labels[self.predict["classes"][idx]]})']) # pylint: disable=line-too-long
                else:
                    ax00.set_yticklabels(labels=[f'P(y={self.calibrated_explainer.class_labels[0]})']) # pylint: disable=line-too-long
                    ax01.set_yticklabels(labels=[f'P(y={self.calibrated_explainer.class_labels[1]})']) # pylint: disable=line-too-long
            else: 
                if self.calibrated_explainer.is_multiclass():                
                    ax00.set_yticklabels(labels=[f'P(y!={self.predict["classes"][idx]})'])
                    ax01.set_yticklabels(labels=[f'P(y={self.predict["classes"][idx]})'])
                else:
                    ax00.set_yticklabels(labels=['P(y=0)'])
                    ax01.set_yticklabels(labels=['P(y=1)'])
            ax01.set_xlabel('Probability')
        
        x = np.linspace(0, num_to_show-1, num_to_show)
        xl = np.linspace(-0.5, x[0], 2)
        xh = np.linspace(x[-1], x[-1]+0.5, 2)
        ax1.fill_betweenx(x, [0], [0], color='k')
        ax1.fill_betweenx(xl, [0], [0], color='k')
        ax1.fill_betweenx(xh, [0], [0], color='k')
        if interval:
            p = predict['predict'][idx]
            gwl = p - predict['low'][idx]
            gwh = p - predict['high'][idx]
            
            gwh, gwl = np.max([gwh, gwl]), np.min([gwh, gwl])
            ax1.fill_betweenx([-0.5,num_to_show-0.5], gwl, gwh, color='k', alpha=0.2)

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
                # If uncertainty cover zero, then set to w to avoid solid plotting
                if wh > 0 and wl < 0:
                    min_val = width
                    max_val = width
            else:
                width = - feature_weights[j]
                min_val = width if width < 0 else 0
                max_val = width if width > 0 else 0
            color = 'b' if width > 0 else 'r'
            ax1.fill_betweenx(xj, min_val, max_val, color=color)
            ax1.fill_betweenx(xj, width, width, color=color)
            if interval:
                if wh > 0 and wl < 0 and self.calibrated_explainer.mode == 'classification':
                    ax1.fill_betweenx(xj, 0, wl, color='r', alpha=0.2)
                    ax1.fill_betweenx(xj, wh, 0, color='b', alpha=0.2)
                else:
                    ax1.fill_betweenx(xj, wl, wh, color=color, alpha=0.2)
       
        ax1.set_yticks(range(num_to_show))
        ax1.set_yticklabels(labels=[column_names[i] for i in features_to_plot]) \
            if column_names is not None else ax1.set_yticks(range(num_to_show)) # pylint: disable=expression-not-assigned
        ax1.set_ylim(-0.5,x[-1]+0.5)
        ax1.set_ylabel('Rules')
        ax1.set_xlabel('Feature weights')
        ax12 = ax1.twinx()
        ax12.set_yticks(range(num_to_show))
        ax12.set_yticklabels([instance[i] for i in features_to_plot])
        ax12.set_ylim(-0.5,x[-1]+0.5)
        ax12.set_ylabel('Instance values')
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
        _, lime_exp = self.calibrated_explainer.preload_LIME()
        exp = []
        for i in range(len(self.test_objects[:,0])):
            tmp = deepcopy(lime_exp)
            tmp.intercept[1] = 0
            tmp.local_pred = self.predict['predict'][i]
            if 'regression' in self.calibrated_explainer.mode:
                tmp.predicted_value = self.predict['predict'][i]
                tmp.min_value = min(self.calibrated_explainer.calY)
                tmp.max_value = max(self.calibrated_explainer.calY)
            else:
                tmp.predict_proba[0], tmp.predict_proba[1] = \
                        1-self.predict['predict'][i], self.predict['predict'][i]

            feature_weights = self.feature_weights['predict'][i]
            features_to_plot = self.__rank_features(feature_weights, 
                        num_to_show=self.calibrated_explainer.num_features)
            rules = self.define_rules(self.test_objects[i, :])
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
        _, shap_exp = self.calibrated_explainer.preload_SHAP(len(self.test_objects[:,0]))
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
    