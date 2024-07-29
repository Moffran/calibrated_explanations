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

from calibrated_explanations import WrapCalibratedExplainer
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
    r1 = RandomForestRegressor(n_estimators=100)
    model_dict = {'RF':(r1,"RF"),'DT': (t1,"DT")}

    model, model_name = model_dict[model_name] # pylint: disable=redefined-outer-name
    model.fit(trainX,trainY)
    return model, model_name



class TestCalibratedExplainer_regression(unittest.TestCase):
    def assertBetween(self, value, low, high):
        self.assertTrue(low <= value <= high, f"Expected {low} <= {value} <= {high}")

    # pylint: disable=unused-variable, unsubscriptable-object
    def test_wrap_regression_ce(self):
        trainX, trainY, calX, calY, testX, testY, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        cal_exp = WrapCalibratedExplainer(RandomForestRegressor())
        self.assertFalse(cal_exp.fitted)
        self.assertFalse(cal_exp.calibrated)
        print(cal_exp)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explain_factual(testX)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explain_counterfactual(testX)

        cal_exp.fit(trainX, trainY)
        self.assertTrue(cal_exp.fitted)
        self.assertFalse(cal_exp.calibrated)
        print(cal_exp)
        testY_hat1 = cal_exp.predict(testX)
        testY_hat2, (low, high) = cal_exp.predict(testX, uq_interval=True)
        for i, y_hat in enumerate(testY_hat2):
            self.assertEqual(testY_hat1[i], y_hat)
            self.assertEqual(low[i], y_hat)
            self.assertEqual(high[i], y_hat)
        # An uncalibrated regression model does not support predict thresholded labels as no conformal predictive system is available
        with pytest.raises(ValueError):
            testY_hat = cal_exp.predict(testX, threshold=testY)
        # An uncalibrated regression model does not support predict thresholded labels as no conformal predictive system is available
        with pytest.raises(ValueError):
            testY_hat, (low, high) = cal_exp.predict(testX, uq_interval=True, threshold=testY)
        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(testX)
        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(testX, uq_interval=True)
        # An uncalibrated regression model does not support predict_proba as no conformal predictive system is available
        with pytest.raises(RuntimeError):
            cal_exp.predict_proba(testX, threshold=testY)
        # An uncalibrated regression model does not support predict_proba as no conformal predictive system is available
        with pytest.raises(RuntimeError):
            cal_exp.predict_proba(testX, uq_interval=True, threshold=testY)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explain_factual(testX)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explain_counterfactual(testX)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explain_factual(testX, threshold=testY)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explain_counterfactual(testX, threshold=testY)

        # calibrate initialize the conformal predictive system
        # Note that the difficulty estimation works in the same way as when using CalibratedExplainer
        # No additional testing of difficulty estimation is deemed necessary
        cal_exp.calibrate(calX, calY, feature_names=feature_names, categorical_labels=categorical_labels)
        self.assertTrue(cal_exp.fitted)
        self.assertTrue(cal_exp.calibrated)
        print(cal_exp)
        # predict calibrated regression output using the conformal predictive system
        testY_hat1 = cal_exp.predict(testX)
        # predict calibrated regression output using the conformal predictive system, with uncertainty quantification
        testY_hat2, (low, high) = cal_exp.predict(testX, uq_interval=True)
        for i, y_hat in enumerate(testY_hat2):
            self.assertEqual(testY_hat1[i], y_hat)
            self.assertBetween(y_hat, low[i], high[i])
        # predict thresholded labels using the conformal predictive system
        testY_hat1 = cal_exp.predict(testX, threshold=testY)
        # predict thresholded labels using the conformal predictive system, with uncertainty quantification
        testY_hat2, (low, high) = cal_exp.predict(testX, uq_interval=True, threshold=testY)
        for i, y_hat in enumerate(testY_hat2):
            self.assertEqual(testY_hat1[i], y_hat)
            # testY_hat2 is a string in the form 'y_hat > threshold' so we cannot compare it to low and high
            # self.assertBetween(y_hat, low[i], high[i])
        explanation = cal_exp.explain_factual(testX)
        explanation = cal_exp.explain_counterfactual(testX)
        explanation = cal_exp.explain_factual(testX, threshold=testY)
        explanation = cal_exp.explain_counterfactual(testX, threshold=testY)

        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(testX)
        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(testX, uq_interval=True)
        testY_hat1 = cal_exp.predict_proba(testX, threshold=testY[0])
        testY_hat2, (low, high) = cal_exp.predict_proba(testX, uq_interval=True, threshold=testY[0])
        for i, y_hat in enumerate(testY_hat2):
            # Due to that random_state can not be set to guarantee identical results in
            # ConformalPredictiveSystem, the probabilities will differ slightly. This is a known issue.
            # for j in range(len(y_hat)):
            #     self.assertEqual(testY_hat1[i][j], y_hat[j])
            self.assertBetween(testY_hat2[i,1], low[i], high[i])
        testY_hat1 = cal_exp.predict_proba(testX, threshold=testY)
        testY_hat2, (low, high) = cal_exp.predict_proba(testX, uq_interval=True, threshold=testY)
        for i, y_hat in enumerate(testY_hat2):
            # Due to that random_state can not be set to guarantee identical results in
            # ConformalPredictiveSystem, the probabilities will differ slightly. This is a known issue.
            # for j in range(len(y_hat)):
            #     self.assertEqual(testY_hat1[i][j], y_hat[j])
            self.assertBetween(testY_hat2[i,1], low[i], high[i])

        cal_exp.fit(trainX, trainY)
        self.assertTrue(cal_exp.fitted)
        self.assertTrue(cal_exp.calibrated)

        learner = cal_exp.learner
        explainer = cal_exp.explainer

        new_exp = WrapCalibratedExplainer(learner)
        self.assertTrue(new_exp.fitted)
        self.assertFalse(new_exp.calibrated)
        self.assertEqual(new_exp.learner, learner)

        new_exp = WrapCalibratedExplainer(explainer)
        self.assertTrue(new_exp.fitted)
        self.assertTrue(new_exp.calibrated)
        self.assertEqual(new_exp.explainer, explainer)
        self.assertEqual(new_exp.learner, learner)


if __name__ == '__main__':
    # unittest.main()
    pytest.main()
