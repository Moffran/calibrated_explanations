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
from crepes.extras import DifficultyEstimator

from calibrated_explanations import CalibratedExplainer, BinaryRegressorDiscretizer, RegressorDiscretizer
from calibrated_explanations.utils.helper import safe_isinstance

MODEL = 'RF'


def load_regression_dataset():
    num_to_test = 1
    calibration_size = 200
    dataset = 'abalone.txt'

    ds = pd.read_csv('data/reg/' + dataset)
    X = ds.drop('REGRESSION', axis=1).values[:2000,:]
    y = ds['REGRESSION'].values[:2000]
    y = (y-np.min(y))/(np.max(y)-np.min(y))
    no_of_classes = None
    no_of_features = X.shape[1]
    categorical_features = [i for i in range(no_of_features) if len(np.unique(X[:,i])) < 10]
    categorical_labels = None
    columns = ds.drop('REGRESSION', axis=1).columns

    trainCalX, testX, trainCalY, testY = train_test_split(X, y, test_size=num_to_test, random_state=42)
    # trainCalX,trainCalY = shuffle(trainCalX, trainCalY)
    trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=calibration_size, random_state=42)
    return trainX, trainY, calX, calY, testX, testY, no_of_classes, no_of_features, categorical_features, categorical_labels, columns


def get_regression_model(model_name, trainX, trainY):
    t1 = DecisionTreeRegressor()
    r1 = RandomForestRegressor(n_estimators=10)
    model_dict = {'RF':(r1,"RF"),'DT': (t1,"DT")}

    model, model_name = model_dict[model_name] # pylint: disable=redefined-outer-name
    model.fit(trainX,trainY)
    return model, model_name



class TestCalibratedExplainer_regression(unittest.TestCase):
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

    def test_failure_regression(self):
        trainX, trainY, calX, calY, _, _, _, _, categorical_features, _, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', trainX, trainY) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(model, calX, calY, feature_names=feature_names, categorical_features=categorical_features, mode='regression')
        with pytest.raises(RuntimeError):
            cal_exp.set_difficulty_estimator(DifficultyEstimator())
        with pytest.raises(RuntimeError):
            cal_exp.set_difficulty_estimator(DifficultyEstimator)


    # NOTE: this takes takes about 70s to run
    # @unittest.skip('Test fails online but passes locally. Error/warning raised by crepes. Skipping provisionally.')
    def test_regression_ce(self):
        trainX, trainY, calX, calY, testX, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
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
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)
        with pytest.raises(AssertionError):
            semi = factual_explanation.get_semi_explanations()
        with pytest.raises(AssertionError):
            counter = factual_explanation.get_counter_explanations()

        counterfactual_explanation = cal_exp.explain_counterfactual(testX)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")
        semi = counterfactual_explanation.get_semi_explanations()
        self.assertExplanation(semi)
        counter = counterfactual_explanation.get_counter_explanations()
        self.assertExplanation(counter)


    def test_probabilistic_regression_ce(self):
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

        factual_explanation = cal_exp.explain_factual(testX, testY)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        factual_explanation = cal_exp.explain_factual(testX, testY[0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")
        with pytest.raises(AssertionError):
            semi = factual_explanation.get_semi_explanations()
        with pytest.raises(AssertionError):
            counter = factual_explanation.get_counter_explanations()

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY[0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")
        try:
            counterfactual_explanation.plot(style='triangular')
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")
        semi = counterfactual_explanation.get_semi_explanations()
        self.assertExplanation(semi)
        counter = counterfactual_explanation.get_counter_explanations()
        self.assertExplanation(counter)


    # NOTE: this takes takes about 70s to run
    # @unittest.skip('Test fails online but passes locally. Error/warning raised by crepes. Skipping provisionally.')
    def test_regression_conditional_ce(self):
        trainX, trainY, calX, calY, testX, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', trainX, trainY) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            calX,
            calY,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            bins=calX[:,0]
        )
        factual_explanation = cal_exp.explain_factual(testX, bins=testX[:,0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(0.1, np.inf), bins=testX[:,0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(-np.inf, 0.9), bins=testX[:,0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, bins=testX[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(0.1, np.inf), bins=testX[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(-np.inf, 0.9), bins=testX[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


    def test_probabilistic_regression_conditional_ce(self):
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
            bins=calX[:,0]
        )

        factual_explanation = cal_exp.explain_factual(testX, testY, bins=testX[:,0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        factual_explanation = cal_exp.explain_factual(testX, testY[0], bins=testX[:,0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY, bins=testX[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY[0], bins=testX[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


    # @unittest.skip('Test fails online but passes locally. Error/warning raised by crepes. Skipping provisionally.')
    def test_knn_normalized_regression_ce(self):
        trainX, trainY, calX, calY, testX, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
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
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


    # @unittest.skip('Test passes but is extremely slow.  Skipping provisionally.')
    def test_knn_normalized_probabilistic_regression_ce(self):
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

        factual_explanation = cal_exp.explain_factual(testX, testY)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        factual_explanation = cal_exp.explain_factual(testX, testY[0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY[0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


    # @unittest.skip('Test fails online but passes locally. Error/warning raised by crepes. Skipping provisionally.')
    def test_var_normalized_regression_ce(self):
        trainX, trainY, calX, calY, testX, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
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
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        factual_explanation = cal_exp.explain_factual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        counterfactual_explanation = cal_exp.explain_counterfactual(testX)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


    # @unittest.skip('Test passes but is extremely slow.  Skipping provisionally.')
    def test_var_normalized_probabilistic_regression_ce(self):
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

        factual_explanation = cal_exp.explain_factual(testX, testY)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        factual_explanation = cal_exp.explain_factual(testX, testY[0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(testX, testY[0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


if __name__ == '__main__':
    # unittest.main()
    pytest.main()
