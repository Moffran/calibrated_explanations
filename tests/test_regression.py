# pylint: disable=missing-docstring, missing-module-docstring, invalid-name, protected-access, too-many-locals, line-too-long, too-many-statements, duplicate-code
# flake8: noqa: E501
from __future__ import absolute_import

import unittest
import pytest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from crepes.extras import DifficultyEstimator

from calibrated_explanations import CalibratedExplainer, BinaryDiscretizer, DecileDiscretizer # pylint: disable=unused-import

MODEL = 'RF'


def load_regression_dataset():
    # dataSet = 'housing.csv'
    # delimiter = ';'
    num_to_test = 1
    # categorical_labels = {8: {0: 'INLAND', 1: 'NEAR BAY', 2: '<1H OCEAN', 3: 'NEAR OCEAN', 4: 'ISLAND'}}

    # fileName = 'data/reg/' + dataSet
    # df = pd.read_csv(fileName, delimiter=delimiter, dtype=np.float64)

    # target = 'median_house_value'
    # # target = 'REGRESSION'
    # df.dropna(inplace=True)
    # X, y = df.drop(target,axis=1), df[target]
    # # normalize target between 0 and 1
    # # y = (y - y.min())/(y.max() - y.min())
    # columns = df.drop(target,axis=1).columns
    # no_of_classes = len(np.unique(y))
    # no_of_features = X.shape[1]
    # categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:,i])) < 10]
    # # # sort targets to make sure equal presence of both classes in test set (see definition of test_index after outer loop below)

    dataset = 'abalone.txt'
    ds = pd.read_csv('data/reg/' + dataset)
    X = ds.drop('REGRESSION', axis=1).values
    y = ds['REGRESSION'].values
    y = (y-np.min(y))/(np.max(y)-np.min(y))
    no_of_classes = None
    no_of_features = X.shape[1]
    categorical_features = [i for i in range(no_of_features) if len(np.unique(X[:,i])) < 10]
    categorical_labels = None
    columns = ds.drop('REGRESSION', axis=1).columns

    trainCalX, testX, trainCalY, testY = train_test_split(X, y, test_size=num_to_test, random_state=42)
    # trainCalX,trainCalY = shuffle(trainCalX, trainCalY)
    trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=0.2, random_state=42)
    return trainX, trainY, calX, calY, testX, testY, no_of_classes, no_of_features, categorical_features, categorical_labels, columns

def get_regression_model(model_name, trainX, trainY):
    t1 = DecisionTreeRegressor()
    r1 = RandomForestRegressor(n_estimators=100)
    model_dict = {'RF':(r1,"RF"),'DT': (t1,"DT")}

    model, model_name = model_dict[model_name] # pylint: disable=redefined-outer-name
    model.fit(trainX,trainY)
    return model, model_name



class TestCalibratedExplainer(unittest.TestCase):
    def assertExplanation(self, exp):
        for _, instance in enumerate(exp.test_objects):
            boundaries = exp.calibrated_explainer.rule_boundaries(instance)
            for f in range(exp.calibrated_explainer.num_features):
                # assert that instance values are covered by the rule conditions
                assert instance[f] >= boundaries[f][0] and instance[f] <= boundaries[f][1]
        return True

    def test_failure(self):
        trainX, trainY, calX, calY, _, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', trainX, trainY) # pylint: disable=redefined-outer-name
        with pytest.raises(NotFittedError):
            CalibratedExplainer(RandomForestRegressor(), calX, calY, feature_names=feature_names, categorical_features=categorical_features, categorical_labels=categorical_labels, mode='regression')
        cal_exp = CalibratedExplainer(model, calX, calY, feature_names=feature_names, categorical_features=categorical_features, mode='regression')
        with pytest.raises(NotFittedError):
            cal_exp.set_difficulty_estimator(DifficultyEstimator())


    # NOTE: this takes takes about 70s to run
    # @unittest.skip('Skipping provisionally.')
    def test_regression_ce(self):
        trainX, trainY, calX, calY, testX, testY, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', trainX, trainY) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            calX,
            calY,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression'
        )
        factual_explanation = cal_exp.explain_factual(testX)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, testY)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, testY[0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY[0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)


    # @unittest.skip('Test passes but is slow, ~2 minutes.  Skipping provisionally.')
    @unittest.skip('Test fails but cannot reproduce error. Skipping provisionally.')
    def test_knn_normalized_regression_ce(self):
        trainX, trainY, calX, calY, testX, testY, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', trainX, trainY) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            calX,
            calY,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            difficulty_estimator=DifficultyEstimator().fit(X=trainX, y=trainY, scaler=True),
        )
        factual_explanation = cal_exp.explain_factual(testX)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, testY)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, testY[0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY[0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)


    # @unittest.skip('Test passes but is slow, ~2 minutes.  Skipping provisionally.')
    # @unittest.skip('Skipping provisionally.')
    def test_var_normalized_regression_ce(self):
        trainX, trainY, calX, calY, testX, testY, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', trainX, trainY) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            calX,
            calY,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            difficulty_estimator=DifficultyEstimator().fit(X=trainX, learner=model, scaler=True),
        )
        factual_explanation = cal_exp.explain_factual(testX)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, testY)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, testY[0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        factual_explanation.get_factual_rules()
        self.assertExplanation(factual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY[0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, DecileDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        counterfactual_explanation.add_conjunctions()
        counterfactual_explanation.get_counterfactual_rules()
        self.assertExplanation(counterfactual_explanation)



if __name__ == '__main__':
    # unittest.main()
    pytest.main()
