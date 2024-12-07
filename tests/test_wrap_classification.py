# pylint: disable=invalid-name, line-too-long, too-many-locals, too-many-statements, redefined-outer-name, duplicate-code
"""
This module contains unit tests for the `WrapCalibratedExplainer` class from the `calibrated_explanations` package.
The tests cover both binary and multiclass classification scenarios.
Fixtures:
    binary_dataset: Prepares a binary classification dataset for testing.
    multiclass_dataset: Prepares a multiclass classification dataset for testing.
Tests:
    test_wrap_binary_ce: Tests the `WrapCalibratedExplainer` with a binary classification dataset.
    test_wrap_multiclass_ce: Tests the `WrapCalibratedExplainer` with a multiclass classification dataset.
The tests ensure that:
    - The `WrapCalibratedExplainer` can be fitted and calibrated correctly.
    - Predictions and prediction intervals are consistent before and after calibration.
    - The `WrapCalibratedExplainer` raises appropriate errors when methods are called before fitting or calibration.
    - The `WrapCalibratedExplainer` can be re-initialized with an existing learner or explainer.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.utils.helper import transform_to_numeric

from tests.test_wrap_regression import generic_test

@pytest.fixture
def binary_dataset():
    """
    Generates a binary classification dataset from a CSV file.
    The function reads a dataset from a CSV file, processes it, and splits it into training, calibration, and test sets.
    It also identifies the number of classes, number of features, and categorical features.
    Returns:
        tuple: A tuple containing the following elements:
            - X_prop_train (numpy.ndarray): The training features for the model.
            - y_prop_train (numpy.ndarray): The training labels for the model.
            - X_cal (numpy.ndarray): The calibration features.
            - y_cal (numpy.ndarray): The calibration labels.
            - X_test (numpy.ndarray): The test features.
            - y_test (numpy.ndarray): The test labels.
            - no_of_classes (int): The number of unique classes in the target variable.
            - no_of_features (int): The number of features in the dataset.
            - categorical_features (list): A list of indices of categorical features.
            - columns (pandas.Index): The column names of the features.
    """
    dataSet = 'diabetes_full'
    delimiter = ','
    num_to_test = 2
    target_column = 'Y'

    fileName = f'data/{dataSet}.csv'
    df = pd.read_csv(fileName, delimiter=delimiter, dtype=np.float64)

    columns = df.drop(target_column, axis=1).columns
    num_classes = len(np.unique(df[target_column]))
    num_features = df.drop(target_column, axis=1).shape[1]

    sorted_indices = np.argsort(df[target_column].values).astype(int)
    X, y = df.drop(target_column, axis=1).values[sorted_indices, :], df[target_column].values[sorted_indices]

    categorical_features = [i for i in range(num_features) if len(np.unique(df.drop(target_column, axis=1).iloc[:, i])) < 10]

    test_index = np.array([*range(num_to_test // 2), *range(len(y) - 1, len(y) - num_to_test // 2 - 1, -1)])
    train_index = np.setdiff1d(np.array(range(len(y))), test_index)

    trainX_cal, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(trainX_cal, y_train, test_size=0.33, random_state=42, stratify=y_train)

    return X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, num_classes, num_features, categorical_features, columns

@pytest.fixture
def multiclass_dataset():
    """
    Prepares and splits a multiclass dataset for training, calibration, and testing.

    Returns:
        X_prop_train (np.ndarray): Training features for the proper training set.
        y_prop_train (np.ndarray): Training labels for the proper training set.
        X_cal (np.ndarray): Features for the calibration set.
        y_cal (np.ndarray): Labels for the calibration set.
        X_test (np.ndarray): Features for the test set.
        y_test (np.ndarray): Labels for the test set.
        no_of_classes (int): Number of unique classes in the target variable.
        no_of_features (int): Number of features in the dataset.
        categorical_features (list): List of categorical feature names.
        categorical_labels (list): List of categorical labels.
        target_labels (list): List of target labels.
        columns (pd.Index): Column names of the feature set.
    """
    dataset_name = 'glass'
    delimiter = ','
    num_test_samples = 6
    file_path = f'data/Multiclass/{dataset_name}.csv'

    df = pd.read_csv(file_path, delimiter=delimiter).dropna()
    target_column = 'Type'

    df, categorical_features, categorical_labels, target_labels, _ = transform_to_numeric(df, target_column)

    columns = df.drop(target_column, axis=1).columns
    num_classes = len(np.unique(df[target_column]))
    num_features = df.drop(target_column, axis=1).shape[1]

    sorted_indices = np.argsort(df[target_column].values).astype(int)
    X, y = df.drop(target_column, axis=1).values[sorted_indices, :], df[target_column].values[sorted_indices]

    test_indices = np.hstack([np.where(y == i)[0][:num_test_samples // num_classes] for i in range(num_classes)])
    train_indices = np.setdiff1d(np.arange(len(y)), test_indices)

    X_train_cal, X_test = X[train_indices, :], X[test_indices, :]
    y_train, y_test = y[train_indices], y[test_indices]

    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train_cal, y_train, test_size=0.33, random_state=42, stratify=y_train)

    return X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, num_classes, num_features, categorical_features, categorical_labels, target_labels, columns

def test_wrap_binary_ce(binary_dataset):
    """
    Test the WrapCalibratedExplainer class for binary classification.
    This test function performs the following steps:
    1. Initializes the WrapCalibratedExplainer with a RandomForestClassifier.
    2. Checks that the explainer is neither fitted nor calibrated initially.
    3. Ensures that plotting without fitting raises a RuntimeError.
    4. Fits the explainer and verifies it is fitted but not calibrated.
    5. Tests various prediction methods (with and without calibration) and 
       ensures consistency in the predictions.
    6. Tests the predict_proba method (with and without calibration) and 
       ensures consistency in the probability predictions.
    7. Calibrates the explainer and verifies it is both fitted and calibrated.
    8. Re-tests the prediction methods to ensure consistency post-calibration.
    9. Re-fits the explainer and verifies it remains calibrated.
    10. Tests the ability to create new instances of WrapCalibratedExplainer 
        with the same learner and explainer, ensuring they inherit the correct 
        fitted and calibrated states.
    11. Plots the results to visually inspect the predictions.
    Args:
        binary_dataset (tuple): A tuple containing the training, calibration, 
                                and test datasets along with additional 
                                metadata such as categorical features and 
                                feature names.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, feature_names = binary_dataset
    cal_exp = WrapCalibratedExplainer(RandomForestClassifier())
    assert not cal_exp.fitted
    assert not cal_exp.calibrated

    with pytest.raises(RuntimeError):
        cal_exp.plot(X_test, show=False)
    with pytest.raises(RuntimeError):
        cal_exp.plot(X_test, y_test, show=False)

    cal_exp.fit(X_prop_train, y_prop_train)
    assert cal_exp.fitted
    assert not cal_exp.calibrated

    y_test_hat1 = cal_exp.predict(X_test)
    y_test_hat2, (low, high) = cal_exp.predict(X_test, True)
    y_test_hat3 = cal_exp.predict(X_test, calibrated=False)
    y_test_hat4, (low4, high4) = cal_exp.predict(X_test, uq_interval=True, calibrated=False)

    for i, y_hat in enumerate(y_test_hat1):
        assert y_test_hat2[i] == y_hat
        assert y_test_hat3[i] == y_hat
        assert y_test_hat4[i] == y_hat
        assert low[i] == y_hat
        assert high[i] == y_hat
        assert low4[i] == y_hat
        assert high4[i] == y_hat

    y_test_proba1 = cal_exp.predict_proba(X_test)
    y_test_proba2, (low_proba, high_proba) = cal_exp.predict_proba(X_test, True)
    y_test_proba3 = cal_exp.predict_proba(X_test, calibrated=False)
    y_test_proba4, (low_proba4, high_proba4) = cal_exp.predict_proba(X_test, True, calibrated=False)

    for i, y_proba in enumerate(y_test_proba1):
        for j, y_proba_j in enumerate(y_proba):
            assert y_test_proba2[i][j] == y_proba_j
            assert y_test_proba3[i][j] == y_proba_j
            assert y_test_proba4[i][j] == y_proba_j
        assert low_proba[i] == y_test_proba1[i, 1]
        assert high_proba[i] == y_test_proba1[i, 1]
        assert low_proba4[i] == y_test_proba1[i, 1]
        assert high_proba4[i] == y_test_proba1[i, 1]

    cal_exp.calibrate(X_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features)
    assert cal_exp.fitted
    assert cal_exp.calibrated

    y_test_hat3 = cal_exp.predict(X_test, calibrated=False)
    y_test_hat4, (low4, high4) = cal_exp.predict(X_test, uq_interval=True, calibrated=False)

    for i, y_hat in enumerate(y_test_hat1):
        assert y_test_hat3[i] == y_hat
        assert y_test_hat4[i] == y_hat
        assert low4[i] == y_hat
        assert high4[i] == y_hat

    y_test_proba3 = cal_exp.predict_proba(X_test, calibrated=False)
    y_test_proba4, (low_proba4, high_proba4) = cal_exp.predict_proba(X_test, True, calibrated=False)

    for i, y_proba in enumerate(y_test_proba1):
        for j, y_proba_j in enumerate(y_proba):
            assert y_test_proba3[i][j] == y_proba_j
            assert y_test_proba4[i][j] == y_proba_j
        assert low_proba4[i] == y_test_proba1[i, 1]
        assert high_proba4[i] == y_test_proba1[i, 1]

    y_test_hat1 = cal_exp.predict(X_test)
    y_test_hat2, (low, high) = cal_exp.predict(X_test, True)

    for i, y_hat in enumerate(y_test_hat2):
        assert y_test_hat1[i] == y_hat

    y_test_hat1 = cal_exp.predict_proba(X_test)
    y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, True)

    for i, y_hat in enumerate(y_test_hat2):
        for j, y_hat_j in enumerate(y_hat):
            assert y_test_hat1[i][j] == y_hat_j
        assert low[i] <= y_test_hat2[i, 1] <= high[i]

    generic_test(cal_exp, X_prop_train, y_prop_train, X_test, y_test)

def test_wrap_multiclass_ce(multiclass_dataset):
    """
    Test the WrapCalibratedExplainer class for a multiclass classification problem.
    This test performs the following steps:
    1. Initializes the WrapCalibratedExplainer with a RandomForestClassifier.
    2. Checks that the explainer is neither fitted nor calibrated initially.
    3. Ensures that plotting methods raise RuntimeError before fitting.
    4. Fits the explainer and verifies it is fitted but not calibrated.
    5. Tests the predict and predict_proba methods before calibration.
    6. Calibrates the explainer and verifies it is both fitted and calibrated.
    7. Tests the predict and predict_proba methods after calibration.
    8. Re-fits the explainer and verifies it remains calibrated.
    9. Tests the ability to create new WrapCalibratedExplainer instances with the same learner and explainer.
    10. Ensures that plotting methods work after fitting and calibration.
    Args:
        multiclass_dataset (tuple): A tuple containing the training, calibration, and test datasets along with 
                                    additional metadata such as categorical features and feature names.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, _, _, feature_names = multiclass_dataset
    cal_exp = WrapCalibratedExplainer(RandomForestClassifier())
    assert not cal_exp.fitted
    assert not cal_exp.calibrated
    repr(cal_exp)

    with pytest.raises(RuntimeError):
        cal_exp.calibrate(X_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features)
    with pytest.raises(RuntimeError):
        cal_exp.plot(X_test, show=False)
    with pytest.raises(RuntimeError):
        cal_exp.plot(X_test, y_test, show=False)
    with pytest.raises(RuntimeError):
        cal_exp.calibrated_confusion_matrix()
    with pytest.raises(RuntimeError):
        cal_exp.initialize_reject_learner()
    with pytest.raises(RuntimeError):
        cal_exp.predict_reject(X_test)

    cal_exp.fit(X_prop_train, y_prop_train)
    assert cal_exp.fitted
    assert not cal_exp.calibrated
    repr(cal_exp)

    with pytest.raises(RuntimeError):
        cal_exp.plot(X_test, show=False)
    with pytest.raises(RuntimeError):
        cal_exp.plot(X_test, y_test, show=False)
    with pytest.raises(RuntimeError):
        cal_exp.calibrated_confusion_matrix()
    with pytest.raises(RuntimeError):
        cal_exp.initialize_reject_learner()
    with pytest.raises(RuntimeError):
        cal_exp.predict_reject(X_test)

    y_test_hat1 = cal_exp.predict(X_test)
    y_test_hat2, (low, high) = cal_exp.predict(X_test, True)

    for i, y_hat in enumerate(y_test_hat2):
        assert y_test_hat1[i] == y_hat
        assert low[i] == y_hat
        assert high[i] == y_hat

    y_test_hat1 = cal_exp.predict_proba(X_test)
    y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, True)

    for i, y_hat in enumerate(y_test_hat2):
        for j, y_hat_j in enumerate(y_hat):
            assert y_test_hat1[i][j] == y_hat_j
            assert low[i][j] <= y_hat_j <= high[i][j]

    cal_exp.calibrate(X_cal, y_cal, mode='classification', feature_names=feature_names, categorical_features=categorical_features)
    assert cal_exp.fitted
    assert cal_exp.calibrated
    repr(cal_exp)

    cal_exp.calibrated_confusion_matrix()
    cal_exp.initialize_reject_learner()
    cal_exp.predict_reject(X_test)

    y_test_hat1 = cal_exp.predict(X_test)
    y_test_hat2, (low, high) = cal_exp.predict(X_test, True)

    for i, y_hat in enumerate(y_test_hat2):
        assert y_test_hat1[i] == y_hat

    y_test_hat1 = cal_exp.predict_proba(X_test)
    y_test_hat2, (low, high) = cal_exp.predict_proba(X_test, True)

    for i, y_hat in enumerate(y_test_hat2):
        for j, y_hat_j in enumerate(y_hat):
            assert y_test_hat1[i][j] == y_hat_j
            assert low[i][j] <= y_hat_j <= high[i][j]

    generic_test(cal_exp, X_prop_train, y_prop_train, X_test, y_test)
