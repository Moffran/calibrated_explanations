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

from crepes.extras import MondrianCategorizer

from calibrated_explanations import WrapCalibratedExplainer
MODEL = 'RF'


def load_regression_dataset():
    num_to_test = 1
    calibration_size = 200
    dataset = 'abalone.txt'

    ds = pd.read_csv(f'data/reg/{dataset}')
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
    r1 = RandomForestRegressor(n_estimators=100)
    model_dict = {'RF':(r1,"RF"),'DT': (t1,"DT")}

    model, model_name = model_dict[model_name] # pylint: disable=redefined-outer-name
    model.fit(X_prop_train,y_prop_train)
    return model, model_name



class TestCalibratedExplainer_regression(unittest.TestCase):
    def assertBetween(self, value, low, high):
        self.assertTrue(low <= value <= high, f"Expected {low} <= {value} <= {high}")

    # pylint: disable=unused-variable, unsubscriptable-object
    def test_wrap_regression_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        cal_exp = WrapCalibratedExplainer(RandomForestRegressor())
        self.assertFalse(cal_exp.fitted)
        self.assertFalse(cal_exp.calibrated)
        print(cal_exp)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explain_factual(X_test)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explore_alternatives(X_test)

        cal_exp.fit(X_prop_train, y_prop_train)
        self.assertTrue(cal_exp.fitted)
        self.assertFalse(cal_exp.calibrated)
        print(cal_exp)
        y_test_hat1 = cal_exp.predict(X_test)
        y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True)
        for i, y_hat in enumerate(y_test_hat2):
            self.assertEqual(y_test_hat1[i], y_hat)
            self.assertEqual(low[i], y_hat)
            self.assertEqual(high[i], y_hat)
        # An uncalibrated regression model does not support predict thresholded labels as no conformal predictive system is available
        with pytest.raises(ValueError):
            y_test_hat = cal_exp.predict(X_test, threshold=y_test)
        # An uncalibrated regression model does not support predict thresholded labels as no conformal predictive system is available
        with pytest.raises(ValueError):
            y_test_hat, (low, high) = cal_exp.predict(X_test, uq_interval=True, threshold=y_test)
        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(X_test)
        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(X_test, uq_interval=True)
        # An uncalibrated regression model does not support predict_proba as no conformal predictive system is available
        with pytest.raises(RuntimeError):
            cal_exp.predict_proba(X_test, threshold=y_test)
        # An uncalibrated regression model does not support predict_proba as no conformal predictive system is available
        with pytest.raises(RuntimeError):
            cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explain_factual(X_test)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explore_alternatives(X_test)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explain_factual(X_test, threshold=y_test)
        with pytest.raises(RuntimeError):
            explanation = cal_exp.explore_alternatives(X_test, threshold=y_test)

        # calibrate initialize the conformal predictive system
        # Note that the difficulty estimation works in the same way as when using CalibratedExplainer
        # No additional testing of difficulty estimation is deemed necessary
        cal_exp.calibrate(X_cal, y_cal, feature_names=feature_names, categorical_labels=categorical_labels)
        self.assertTrue(cal_exp.fitted)
        self.assertTrue(cal_exp.calibrated)
        print(cal_exp)
        # predict calibrated regression output using the conformal predictive system
        y_test_hat1 = cal_exp.predict(X_test)
        # predict calibrated regression output using the conformal predictive system, with uncertainty quantification
        y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True)
        for i, y_hat in enumerate(y_test_hat2):
            self.assertEqual(y_test_hat1[i], y_hat)
            self.assertBetween(y_hat, low[i], high[i])
        # predict thresholded labels using the conformal predictive system
        y_test_hat1 = cal_exp.predict(X_test, threshold=y_test)
        # predict thresholded labels using the conformal predictive system, with uncertainty quantification
        y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True, threshold=y_test)
        # Due to that random_state can not be set to guarantee identical results in
        # ConformalPredictiveSystem, the probabilities will differ slightly, sometimes resulting in different
        # predicted class labels (depending on whether it is above or below the threshold). This is a known issue.
        # for i, y_hat in enumerate(y_test_hat2):
            # self.assertEqual(y_test_hat1[i], y_hat)
            # y_test_hat2 is a string in the form 'y_hat > threshold' so we cannot compare it to low and high
            # self.assertBetween(y_hat, low[i], high[i])
        explanation = cal_exp.explain_factual(X_test)
        explanation = cal_exp.explore_alternatives(X_test)
        explanation = cal_exp.explain_factual(X_test, threshold=y_test)
        explanation = cal_exp.explore_alternatives(X_test, threshold=y_test)

        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(X_test)
        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(X_test, uq_interval=True)
        y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test[0])
        y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test[0])
        for i, y_hat in enumerate(y_test_hat2):
            # Due to that random_state can not be set to guarantee identical results in
            # ConformalPredictiveSystem, the probabilities will differ slightly. This is a known issue.
            # for j in range(len(y_hat)):
            #     self.assertEqual(y_test_hat1[i][j], y_hat[j])
            self.assertBetween(y_test_hat2[i,1], low[i], high[i])
        y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test)
        y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test)
        for i, y_hat in enumerate(y_test_hat2):
            # Due to that random_state can not be set to guarantee identical results in
            # ConformalPredictiveSystem, the probabilities will differ slightly. This is a known issue.
            # for j in range(len(y_hat)):
            #     self.assertEqual(y_test_hat1[i][j], y_hat[j])
            self.assertBetween(y_test_hat2[i,1], low[i], high[i])

        cal_exp.fit(X_prop_train, y_prop_train)
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

        cal_exp.plot(X_test) # pylint: disable=no-member
        cal_exp.plot(X_test, y_test) # pylint: disable=no-member

        cal_exp.plot(X_test, threshold=y_test[0]) # pylint: disable=no-member
        cal_exp.plot(X_test, y_test, threshold=y_test[0]) # pylint: disable=no-member

        # with pytest.raises(AssertionError):
        #     cal_exp.plot_global(X_test, threshold=y_test) # pylint: disable=no-member
        # with pytest.raises(AssertionError):
        #     cal_exp.plot_global(X_test, y_test, threshold=y_test) # pylint: disable=no-member

    # pylint: disable=unused-variable, unsubscriptable-object
    def test_wrap_conditional_regression_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        cal_exp = WrapCalibratedExplainer(RandomForestRegressor())
        cal_exp.fit(X_prop_train, y_prop_train)

        # calibrate initialize the conformal predictive system
        # Note that the difficulty estimation works in the same way as when using CalibratedExplainer
        # No additional testing of difficulty estimation is deemed necessary
        mc = MondrianCategorizer()
        mc.fit(X_cal, f=cal_exp.learner.predict, no_bins=5)

        cal_exp.calibrate(X_cal, y_cal, mc=mc, feature_names=feature_names, categorical_labels=categorical_labels)
        self.assertTrue(cal_exp.fitted)
        self.assertTrue(cal_exp.calibrated)
        print(cal_exp)
        # predict calibrated regression output using the conformal predictive system
        y_test_hat1 = cal_exp.predict(X_test)
        # predict calibrated regression output using the conformal predictive system, with uncertainty quantification
        y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True)
        for i, y_hat in enumerate(y_test_hat2):
            self.assertEqual(y_test_hat1[i], y_hat)
            self.assertBetween(y_hat, low[i], high[i])
        # predict thresholded labels using the conformal predictive system
        y_test_hat1 = cal_exp.predict(X_test, threshold=y_test)
        # predict thresholded labels using the conformal predictive system, with uncertainty quantification
        y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True, threshold=y_test)
        # Due to that random_state can not be set to guarantee identical results in
        # ConformalPredictiveSystem, the probabilities will differ slightly, sometimes resulting in different
        # predicted class labels (depending on whether it is above or below the threshold). This is a known issue.
        # for i, y_hat in enumerate(y_test_hat2):
            # self.assertEqual(y_test_hat1[i], y_hat)
            # y_test_hat2 is a string in the form 'y_hat > threshold' so we cannot compare it to low and high
            # self.assertBetween(y_hat, low[i], high[i])
        explanation = cal_exp.explain_factual(X_test)
        explanation = cal_exp.explore_alternatives(X_test)
        explanation = cal_exp.explain_factual(X_test, threshold=y_test)
        explanation = cal_exp.explore_alternatives(X_test, threshold=y_test)

        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(X_test)
        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(X_test, uq_interval=True)
        y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test[0])
        y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test[0])
        for i, y_hat in enumerate(y_test_hat2):
            # Due to that random_state can not be set to guarantee identical results in
            # ConformalPredictiveSystem, the probabilities will differ slightly. This is a known issue.
            # for j in range(len(y_hat)):
            #     self.assertEqual(y_test_hat1[i][j], y_hat[j])
            self.assertBetween(y_test_hat2[i,1], low[i], high[i])
        y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test)
        y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test)
        for i, y_hat in enumerate(y_test_hat2):
            # Due to that random_state can not be set to guarantee identical results in
            # ConformalPredictiveSystem, the probabilities will differ slightly. This is a known issue.
            # for j in range(len(y_hat)):
            #     self.assertEqual(y_test_hat1[i][j], y_hat[j])
            self.assertBetween(y_test_hat2[i,1], low[i], high[i])

        cal_exp.fit(X_prop_train, y_prop_train)
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

        # with pytest.raises(AssertionError):
        #     cal_exp.plot_global(X_test, threshold=y_test) # pylint: disable=no-member
        # with pytest.raises(AssertionError):
        #     cal_exp.plot_global(X_test, y_test, threshold=y_test) # pylint: disable=no-member



    # pylint: disable=unused-variable, unsubscriptable-object
    def test_wrap_regression_fast_ce(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, categorical_labels, feature_names = load_regression_dataset()
        cal_exp = WrapCalibratedExplainer(RandomForestRegressor())
        cal_exp.fit(X_prop_train, y_prop_train)
        # calibrate initialize the conformal predictive system
        # Note that the difficulty estimation works in the same way as when using CalibratedExplainer
        # No additional testing of difficulty estimation is deemed necessary
        cal_exp.calibrate(X_cal, y_cal, feature_names=feature_names, categorical_labels=categorical_labels, perturb=True)
        self.assertTrue(cal_exp.fitted)
        self.assertTrue(cal_exp.calibrated)
        print(cal_exp)
        # predict calibrated regression output using the conformal predictive system
        y_test_hat1 = cal_exp.predict(X_test)
        # predict calibrated regression output using the conformal predictive system, with uncertainty quantification
        y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True)
        for i, y_hat in enumerate(y_test_hat2):
            self.assertEqual(y_test_hat1[i], y_hat)
            self.assertBetween(y_hat, low[i], high[i])
        # predict thresholded labels using the conformal predictive system
        y_test_hat1 = cal_exp.predict(X_test, threshold=y_test)
        # predict thresholded labels using the conformal predictive system, with uncertainty quantification
        y_test_hat2, (low, high) = cal_exp.predict(X_test, uq_interval=True, threshold=y_test)
        # Due to that random_state can not be set to guarantee identical results in
        # ConformalPredictiveSystem, the probabilities will differ slightly, sometimes resulting in different
        # predicted class labels (depending on whether it is above or below the threshold). This is a known issue.
        # for i, y_hat in enumerate(y_test_hat2):
            # self.assertEqual(y_test_hat1[i], y_hat)
            # y_test_hat2 is a string in the form 'y_hat > threshold' so we cannot compare it to low and high
            # self.assertBetween(y_hat, low[i], high[i])
        explanation = cal_exp.explain_factual(X_test)
        explanation = cal_exp.explore_alternatives(X_test)
        explanation = cal_exp.explain_factual(X_test, threshold=y_test)
        explanation = cal_exp.explore_alternatives(X_test, threshold=y_test)
        explanation = cal_exp.explain_fast(X_test)
        explanation = cal_exp.explain_fast(X_test, threshold=y_test)

        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(X_test)
        # predict_proba without a threshold is not supported for regression models, regardless of calibration
        with pytest.raises(ValueError):
            cal_exp.predict_proba(X_test, uq_interval=True)
        y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test[0])
        y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test[0])
        for i, y_hat in enumerate(y_test_hat2):
            # Due to that random_state can not be set to guarantee identical results in
            # ConformalPredictiveSystem, the probabilities will differ slightly. This is a known issue.
            # for j in range(len(y_hat)):
            #     self.assertEqual(y_test_hat1[i][j], y_hat[j])
            self.assertBetween(y_test_hat2[i,1], low[i], high[i])
        y_test_hat1 = cal_exp.predict_proba(X_test, threshold=y_test)
        y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, uq_interval=True, threshold=y_test)
        for i, y_hat in enumerate(y_test_hat2):
            # Due to that random_state can not be set to guarantee identical results in
            # ConformalPredictiveSystem, the probabilities will differ slightly. This is a known issue.
            # for j in range(len(y_hat)):
            #     self.assertEqual(y_test_hat1[i][j], y_hat[j])
            self.assertBetween(y_test_hat2[i,1], low[i], high[i])

        cal_exp.fit(X_prop_train, y_prop_train)
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

        cal_exp.plot(X_test) # pylint: disable=no-member
        cal_exp.plot(X_test, y_test) # pylint: disable=no-member

        cal_exp.plot(X_test, threshold=y_test[0]) # pylint: disable=no-member
        cal_exp.plot(X_test, y_test, threshold=y_test[0]) # pylint: disable=no-member

        # with pytest.raises(AssertionError):
        #     cal_exp.plot_global(X_test, threshold=y_test) # pylint: disable=no-member
        # with pytest.raises(AssertionError):
        #     cal_exp.plot_global(X_test, y_test, threshold=y_test) # pylint: disable=no-member


if __name__ == '__main__':
    # unittest.main()
    pytest.main()
