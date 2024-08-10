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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_to_test, random_state=42)
    # X_train,y_train = shuffle(X_train, y_train)
    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train, test_size=calibration_size, random_state=42)
    return X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, no_of_classes, no_of_features, categorical_features, categorical_labels, columns


def get_regression_model(model_name, X_prop_train, y_prop_train):
    t1 = DecisionTreeRegressor()
    r1 = RandomForestRegressor(n_estimators=10)
    model_dict = {'RF':(r1,"RF"),'DT': (t1,"DT")}

    model, model_name = model_dict[model_name] # pylint: disable=redefined-outer-name
    model.fit(X_prop_train,y_prop_train)
    return model, model_name



class TestCalibratedExplainer_regression(unittest.TestCase):
    def assertExplanation(self, exp):
        for _, instance in enumerate(exp.X_test):
            boundaries = exp.calibrated_explainer.rule_boundaries(instance)
            for f in range(exp.calibrated_explainer.num_features):
                # assert that instance values are covered by the rule conditions
                assert instance[f] >= boundaries[f][0] and instance[f] <= boundaries[f][1]
        for explanation in exp:
            assert safe_isinstance(explanation, ['calibrated_explanations.FactualExplanation',
                                                'calibrated_explanations.CounterfactualExplanation'])
        return True

    def test_failure_regression(self):
        X_prop_train, y_prop_train, X_cal, y_cal, _, _, _, _, categorical_features, _, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(model, X_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features, mode='regression')
        with pytest.raises(RuntimeError):
            cal_exp.set_difficulty_estimator(DifficultyEstimator())
        with pytest.raises(RuntimeError):
            cal_exp.set_difficulty_estimator(DifficultyEstimator)


    # @unittest.skip('Skipping provisionally.')
    def test_regression_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression'
        )
        factual_explanation = cal_exp.explain_factual(X_test)
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        factual_explanation.add_conjunctions()
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            factual_explanation[0].plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")
        try:
            factual_explanation[0].plot(style='triangular')
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(-np.inf, 0.9))
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

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, low_high_percentiles=(-np.inf, 0.9))
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
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression'
        )

        factual_explanation = cal_exp.explain_factual(X_test, y_test)
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

        factual_explanation = cal_exp.explain_factual(X_test, y_test[0])
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

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, y_test)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, y_test[0])
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


    # @unittest.skip('Skipping provisionally.')
    def test_regression_conditional_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            bins=X_cal[:,0]
        )
        factual_explanation = cal_exp.explain_factual(X_test, bins=X_test[:,0])
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

        factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(0.1, np.inf), bins=X_test[:,0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(-np.inf, 0.9), bins=X_test[:,0])
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, bins=X_test[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, low_high_percentiles=(0.1, np.inf), bins=X_test[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, low_high_percentiles=(-np.inf, 0.9), bins=X_test[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


    # @unittest.skip('Skipping provisionally.')
    def test_probabilistic_regression_conditional_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            bins=X_cal[:,0]
        )

        factual_explanation = cal_exp.explain_factual(X_test, y_test, bins=X_test[:,0])
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

        factual_explanation = cal_exp.explain_factual(X_test, y_test[0], bins=X_test[:,0])
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

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, y_test, bins=X_test[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, y_test[0], bins=X_test[:,0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


    # @unittest.skip('Skipping provisionally.')
    def test_knn_normalized_regression_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, y=y_prop_train, scaler=True),
        )
        factual_explanation = cal_exp.explain_factual(X_test)
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

        factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


    # @unittest.skip('Test passes but is extremely slow.  Skipping provisionally.')
    def test_knn_normalized_probabilistic_regression_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, y=y_prop_train, scaler=True),
        )

        factual_explanation = cal_exp.explain_factual(X_test, y_test)
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

        factual_explanation = cal_exp.explain_factual(X_test, y_test[0])
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

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, y_test)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, y_test[0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


    # @unittest.skip('Skipping provisionally.')
    def test_var_normalized_regression_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, learner=model, scaler=True),
        )
        factual_explanation = cal_exp.explain_factual(X_test)
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

        factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        factual_explanation = cal_exp.explain_factual(X_test, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(factual_explanation.calibrated_explainer.discretizer, BinaryRegressorDiscretizer)
        self.assertExplanation(factual_explanation)
        try:
            factual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            factual_explanation.plot(uncertainty=True)

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, low_high_percentiles=(0.1, np.inf))
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, low_high_percentiles=(-np.inf, 0.9))
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


    # @unittest.skip('Test passes but is extremely slow.  Skipping provisionally.')
    def test_var_normalized_probabilistic_regression_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, learner=model, scaler=True),
        )

        factual_explanation = cal_exp.explain_factual(X_test, y_test)
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

        factual_explanation = cal_exp.explain_factual(X_test, y_test[0])
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

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, y_test)
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")

        counterfactual_explanation = cal_exp.explain_counterfactual(X_test, y_test[0])
        self.assertIsInstance(counterfactual_explanation.calibrated_explainer.discretizer, RegressorDiscretizer)
        self.assertExplanation(counterfactual_explanation)
        try:
            counterfactual_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"counterfactual_explanation.plot() raised unexpected exception: {e}")


# ------------------------------------------------------------------------------


    # NOTE: this takes takes about 70s to run
    # @unittest.skip('Test fails online but passes locally. Error/warning raised by crepes. Skipping provisionally.')
    def test_regression_perturbed_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            perturb=True
        )
        perturbed_explanation = cal_exp.explain_perturbed(X_test)
        perturbed_explanation.add_conjunctions()
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"factual_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        perturbed_explanation = cal_exp.explain_perturbed(X_test, low_high_percentiles=(0.1, np.inf))
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            perturbed_explanation.plot(uncertainty=True)

        perturbed_explanation = cal_exp.explain_perturbed(X_test, low_high_percentiles=(-np.inf, 0.9))
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            perturbed_explanation.plot(uncertainty=True)
        with pytest.raises(AssertionError):
            _ = perturbed_explanation.get_semi_explanations()
        with pytest.raises(AssertionError):
            _ = perturbed_explanation.get_counter_explanations()


    def test_probabilistic_regression_perturbed_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            perturb=True
        )

        perturbed_explanation = cal_exp.explain_perturbed(X_test, y_test)
        perturbed_explanation.add_conjunctions()
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        perturbed_explanation = cal_exp.explain_perturbed(X_test, y_test[0])
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")
        with pytest.raises(AssertionError):
            _ = perturbed_explanation.get_semi_explanations()
        with pytest.raises(AssertionError):
            _ = perturbed_explanation.get_counter_explanations()


    # NOTE: this takes takes about 70s to run
    # @unittest.skip('Test fails online but passes locally. Error/warning raised by crepes. Skipping provisionally.')
    def test_regression_conditional_perturbed_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            bins=X_cal[:,0],
            perturb=True
        )
        perturbed_explanation = cal_exp.explain_perturbed(X_test, bins=X_test[:,0])
        perturbed_explanation.add_conjunctions()
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        perturbed_explanation = cal_exp.explain_perturbed(X_test, low_high_percentiles=(0.1, np.inf), bins=X_test[:,0])
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            perturbed_explanation.plot(uncertainty=True)

        perturbed_explanation = cal_exp.explain_perturbed(X_test, low_high_percentiles=(-np.inf, 0.9), bins=X_test[:,0])
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            perturbed_explanation.plot(uncertainty=True)


    def test_probabilistic_regression_conditional_perturbed_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            bins=y_cal > y_test[0],
            perturb=True
        )

        perturbed_explanation = cal_exp.explain_perturbed(X_test, y_test, bins=y_test > y_test[0])
        perturbed_explanation.add_conjunctions()
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        perturbed_explanation = cal_exp.explain_perturbed(X_test, y_test[0], bins=y_test > y_test[0])
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")


    # @unittest.skip('Test fails online but passes locally. Error/warning raised by crepes. Skipping provisionally.')
    def test_knn_normalized_regression_perturbed_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, y=y_prop_train, scaler=True),
            perturb=True
        )
        perturbed_explanation = cal_exp.explain_perturbed(X_test)
        perturbed_explanation.add_conjunctions()
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        perturbed_explanation = cal_exp.explain_perturbed(X_test, low_high_percentiles=(0.1, np.inf))
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            perturbed_explanation.plot(uncertainty=True)

        perturbed_explanation = cal_exp.explain_perturbed(X_test, low_high_percentiles=(-np.inf, 0.9))
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            perturbed_explanation.plot(uncertainty=True)


    # @unittest.skip('Test passes but is extremely slow.  Skipping provisionally.')
    def test_knn_normalized_probabilistic_regression_perturbed_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, y=y_prop_train, scaler=True),
            perturb=True
        )

        perturbed_explanation = cal_exp.explain_perturbed(X_test, y_test)
        perturbed_explanation.add_conjunctions()
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        perturbed_explanation = cal_exp.explain_perturbed(X_test, y_test[0])
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")


    # @unittest.skip('Test fails online but passes locally. Error/warning raised by crepes. Skipping provisionally.')
    def test_var_normalized_regression_perturbed_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, learner=model, scaler=True),
            perturb=True
        )
        perturbed_explanation = cal_exp.explain_perturbed(X_test)
        perturbed_explanation.add_conjunctions()
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        perturbed_explanation = cal_exp.explain_perturbed(X_test, low_high_percentiles=(0.1, np.inf))
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            perturbed_explanation.plot(uncertainty=True)

        perturbed_explanation = cal_exp.explain_perturbed(X_test, low_high_percentiles=(-np.inf, 0.9))
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        with pytest.raises(Warning):
            perturbed_explanation.plot(uncertainty=True)


    # @unittest.skip('Test passes but is extremely slow.  Skipping provisionally.')
    def test_var_normalized_probabilistic_regression_perturbed_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        model, _ = get_regression_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        cal_exp = CalibratedExplainer(
            model,
            X_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_labels=categorical_labels,
            mode='regression',
            difficulty_estimator=DifficultyEstimator().fit(X=X_prop_train, learner=model, scaler=True),
            perturb=True
        )

        perturbed_explanation = cal_exp.explain_perturbed(X_test, y_test)
        perturbed_explanation.add_conjunctions()
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")

        perturbed_explanation = cal_exp.explain_perturbed(X_test, y_test[0])
        try:
            perturbed_explanation.plot()
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot() raised unexpected exception: {e}")
        try:
            perturbed_explanation.plot(uncertainty=True)
        except Exception as e: # pylint: disable=broad-except
            pytest.fail(f"perturbed_explanation.plot(uncertainty=True) raised unexpected exception: {e}")


if __name__ == '__main__':
    # unittest.main()
    pytest.main()
