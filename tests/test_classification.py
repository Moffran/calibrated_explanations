# pylint: disable=missing-docstring, missing-module-docstring, invalid-name, protected-access, too-many-locals, line-too-long, duplicate-code
# flake8: noqa: E501
from __future__ import absolute_import
# import tempfile
# import os

import unittest
import pytest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import CalibratedExplainer, EntropyDiscretizer, BinaryEntropyDiscretizer
from calibrated_explanations.utils.helper import transform_to_numeric, safe_isinstance


MODEL = 'RF'
def load_binary_dataset():
    dataSet = 'diabetes_full'
    delimiter = ','
    num_to_test = 2
    target = 'Y'

    fileName = 'data/' + dataSet + ".csv"
    df = pd.read_csv(fileName, delimiter=delimiter, dtype=np.float64)

    X, y = df.drop(target,axis=1), df[target]
    no_of_classes = len(np.unique(y))
    no_of_features = X.shape[1]
    no_of_instances = X.shape[0]
    columns = X.columns
    categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:,i])) < 10]
    # # sort targets to make sure equal presence of both classes in test set (see definition of test_index after outer loop below)
    idx = np.argsort(y.values).astype(int)
    X, y = X.values[idx, :], y.values[idx]
    # Select num_to_test/2 from top and num_to_test/2 from bottom of list of instances
    test_index = np.array([*range(int(num_to_test/2)), *range(no_of_instances-1, no_of_instances-int(num_to_test/2)-1,-1)])
    train_index = np.setdiff1d(np.array(range(no_of_instances)), test_index)
    traincal_X, testX = X[train_index, :], X[test_index, :]
    trainCalY, testY = y[train_index], y[test_index]
    # traincal_X,trainCalY = shuffle(traincal_X, trainCalY)
    trainX, cal_X, trainY, calY = train_test_split(traincal_X, trainCalY, test_size=0.33, random_state=42, stratify=trainCalY)
    return trainX, trainY, cal_X, calY, testX, testY, no_of_classes, no_of_features, categorical_features, columns

def load_multiclass_dataset():
    dataSet = 'glass'
    delimiter = ','
    num_to_test = 6
    # print(dataSet)

    fileName = 'data/Multiclass/' + dataSet + ".csv"
    df = pd.read_csv(fileName, delimiter=delimiter)
    target = 'Type'

    # df.convert_objects()

    df = df.dropna()
    df, categorical_features, categorical_labels, target_labels, mappings = transform_to_numeric(df, target) # pylint: disable=unused-variable

    X, y = df.drop(target,axis=1), df[target]
    columns = X.columns
    no_of_classes = len(np.unique(y))
    no_of_features = X.shape[1]
    no_of_instances = X.shape[0]
    categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:,i])) < 10]
    # # sort targets to make sure equal presence of both classes in test set (see definition of test_index after outer loop below)
    idx = np.argsort(y.values).astype(int)
    X, y = X.values[idx, :], y.values[idx]
    test_idx = []
    idx = list(range(no_of_instances))
    for i in range(no_of_classes):
        test_idx.append(np.where(y == i)[0][0:int(num_to_test/no_of_classes)])
    test_index = np.array(test_idx).flatten()
    # Select num_to_test/2 from top and num_to_test/2 from bottom of list of instances
    train_index = np.setdiff1d(np.array(range(no_of_instances)), test_index)
    traincal_X, testX = X[train_index, :], X[test_index, :]
    trainCalY, testY = y[train_index], y[test_index]
    # traincal_X,trainCalY = shuffle(traincal_X, trainCalY)
    trainX, cal_X, trainY, calY = train_test_split(traincal_X, trainCalY, test_size=0.33,random_state=42, stratify=trainCalY)
    return trainX, trainY, cal_X, calY, testX, testY, no_of_classes, no_of_features, categorical_features, categorical_labels, target_labels, columns

def get_classification_model(model_name, trainX, trainY):
    t1 = DecisionTreeClassifier()
    r1 = RandomForestClassifier(n_estimators=10)
    model_dict = {'RF':(r1,"RF"),'DT': (t1,"DT")}

    model, model_name = model_dict[model_name] # pylint: disable=redefined-outer-name
    model.fit(trainX,trainY)
    return model, model_name



class TestCalibratedExplainer_classification(unittest.TestCase):
    def assertExplanation(self, exp):
        for _, instance in enumerate(exp.test_objects):
            boundaries = exp.calibrated_explainer.rule_boundaries(instance)
            for f in range(exp.calibrated_explainer.num_features):
                # assert that instance values are covered by the rule conditions
                assert instance[f] >= boundaries[f][0] and instance[f] <= boundaries[f][1]
        for explanation in exp:
            assert safe_isinstance(explanation, ['calibrated_explanations.FactualExplanation',
                                                 'calibrated_explanations.CounterfactualExplanation'])
        return True

    def test_binary_ce(self):
        trainX, trainY, cal_X, calY, testX, _, _, _, categorical_features, feature_names = load_binary_dataset()
        model, _ = get_classification_model('RF', trainX, trainY) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            cal_X,
            calY,
            feature_names=feature_names,
            categorical_features=categorical_features,
            mode='classification',
        )
        factual_explanation = cal_exp.explain_factual(testX)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryEntropyDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        factual_explanation.remove_conjunctions()
        try:
            factual_explanation.plot_all()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot_all() raised unexpected exception: {e}")
        try:
            factual_explanation.plot_all(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot_all(uncertainty=True) raised unexpected exception: {e}")
        with pytest.raises(AssertionError):
            semi = factual_explanation.get_semi_explanations()
        with pytest.raises(AssertionError):
            counter = factual_explanation.get_counter_explanations()
        factual_explanation.add_conjunctions(max_rule_size=3)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, EntropyDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.remove_conjunctions()
        try:
            counterfactual_explanation.plot_all()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot_all() raised unexpected exception: {e}")
        try:
            counterfactual_explanation.plot_all(style='triangular')
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot_all() raised unexpected exception: {e}")
        semi = counterfactual_explanation.get_semi_explanations()
        self.assertExplanation(semi)
        counter = counterfactual_explanation.get_counter_explanations()
        self.assertExplanation(counter)
        counterfactual_explanation.add_conjunctions(max_rule_size=3)

    # @unittest.skip('Test passes locally.  Skipping provisionally.')
    def test_multiclass_ce(self):
        trainX, trainY, cal_X, calY, testX, _, _, _, _, categorical_labels, target_labels, feature_names = load_multiclass_dataset()
        model, _ = get_classification_model('RF', trainX, trainY) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            cal_X,
            calY,
            feature_names=feature_names,
            categorical_labels=categorical_labels,
            class_labels=target_labels,
            mode='classification',
            verbose=True
        )
        print(cal_exp)
        factual_explanation = cal_exp.explain_factual(testX)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryEntropyDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        factual_explanation.remove_conjunctions()
        try:
            factual_explanation.plot_all()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot_all() raised unexpected exception: {e}")
        try:
            factual_explanation.plot_all(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot_all(uncertainty=True) raised unexpected exception: {e}")
        with pytest.raises(AssertionError):
            semi = factual_explanation.get_semi_explanations()
        with pytest.raises(AssertionError):
            counter = factual_explanation.get_counter_explanations()
        factual_explanation.add_conjunctions(max_rule_size=3)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, EntropyDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.remove_conjunctions()
        try:
            counterfactual_explanation.plot_all()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot_all() raised unexpected exception: {e}")
        try:
            counterfactual_explanation.plot_all(style='triangular')
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot_all() raised unexpected exception: {e}")
        semi = counterfactual_explanation.get_semi_explanations()
        self.assertExplanation(semi)
        counter = counterfactual_explanation.get_counter_explanations()
        self.assertExplanation(counter)
        counterfactual_explanation.add_conjunctions(max_rule_size=3)

    # @unittest.skip('Skipping provisionally.')
    def test_binary_conditional_ce(self):
        trainX, trainY, cal_X, calY, testX, _, _, _, categorical_features, feature_names = load_binary_dataset()
        model, _ = get_classification_model('RF', trainX, trainY) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            cal_X,
            calY,
            feature_names=feature_names,
            categorical_features=categorical_features,
            mode='classification',
            bins=cal_X[:,0]
        )
        factual_explanation = cal_exp.explain_factual(testX, bins=testX[:,0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryEntropyDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot_all()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot_all() raised unexpected exception: {e}")
        try:
            factual_explanation.plot_all(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot_all(uncertainty=True) raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, bins=testX[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, EntropyDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot_all()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot_all() raised unexpected exception: {e}")

    # @unittest.skip('Test passes locally.  Skipping provisionally.')
    def test_multiclass_conditional_ce(self):
        trainX, trainY, cal_X, calY, testX, _, _, _, _, categorical_labels, target_labels, feature_names = load_multiclass_dataset()
        model, _ = get_classification_model('RF', trainX, trainY) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            cal_X,
            calY,
            feature_names=feature_names,
            # categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            class_labels=target_labels,
            mode='classification',
            bins=cal_X[:,0]
        )
        factual_explanation = cal_exp.explain_factual(testX, bins=testX[:,0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryEntropyDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot_all()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot_all() raised unexpected exception: {e}")
        try:
            factual_explanation.plot_all(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot_all(uncertainty=True) raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, bins=testX[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, EntropyDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot_all()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot_all() raised unexpected exception: {e}")


if __name__ == '__main__':
    # unittest.main()
    pytest.main()
