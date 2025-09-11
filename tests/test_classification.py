# pylint: disable=invalid-name, protected-access, too-many-locals, too-many-arguments, too-many-positional-arguments, line-too-long, redefined-outer-name, no-member
"""
This module contains unit tests for the `CalibratedExplainer` class from the `calibrated_explanations` package.
The tests cover both binary and multiclass classification scenarios.
Fixtures:
    binary_dataset: Prepares a binary classification dataset for testing.
    multiclass_dataset: Prepares a multiclass classification dataset for testing.
Tests:
    test_binary_ce: Tests the `CalibratedExplainer` with a binary classification dataset.
    test_multiclass_ce: Tests the `CalibratedExplainer` with a multiclass classification dataset.
    test_binary_conditional_ce: Tests the `CalibratedExplainer` with a binary classification dataset and conditional bins.
    test_multiclass_conditional_ce: Tests the `CalibratedExplainer` with a multiclass classification dataset and conditional bins.
    test_binary_fast_ce: Tests the `CalibratedExplainer` with a binary classification dataset and perturbation.
    test_multiclass_fast_ce: Tests the `CalibratedExplainer` with a multiclass classification dataset and perturbation.
    test_binary_conditional_fast_ce: Tests the `CalibratedExplainer` with a binary classification dataset, conditional bins, and perturbation.
    test_multiclass_fast_conditional_ce: Tests the `CalibratedExplainer` with a multiclass classification dataset, conditional bins, and perturbation.
"""

import numpy as np
import pandas as pd
import pytest
from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.utils.helper import transform_to_numeric
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os


@pytest.fixture
def binary_dataset():
    """
    Generates a binary classification dataset from a CSV file.
    Returns:
        tuple: A tuple containing the training, calibration, and test datasets along with additional metadata.
    """
    dataSet = "diabetes_full"
    delimiter = ","
    num_to_test = 2
    target = "Y"

    fileName = f"data/{dataSet}.csv"
    df = pd.read_csv(fileName, delimiter=delimiter, dtype=np.float64)
    # Limit rows for test speed while preserving behavior
    df = df.iloc[:500, :]

    X, y = df.drop(target, axis=1), df[target]
    no_of_classes = len(np.unique(y))
    no_of_features = X.shape[1]
    columns = X.columns
    categorical_features = [i for i in range(no_of_features) if len(np.unique(X.iloc[:, i])) < 10]

    idx = np.argsort(y.values).astype(int)
    X, y = X.values[idx, :], y.values[idx]

    test_index = np.array(
        [*range(num_to_test // 2), *range(len(y) - 1, len(y) - num_to_test // 2 - 1, -1)]
    )
    train_index = np.setdiff1d(np.array(range(len(y))), test_index)

    trainX_cal, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(
        trainX_cal, y_train, test_size=0.33, random_state=42, stratify=y_train
    )
    return (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        no_of_classes,
        no_of_features,
        categorical_features,
        columns,
    )


@pytest.fixture
def multiclass_dataset():
    """
    Prepares and splits a multiclass dataset for training, calibration, and testing.
    Returns:
        tuple: A tuple containing the training, calibration, and test datasets along with additional metadata.
    """
    dataSet = "glass"
    delimiter = ","
    num_to_test = 6

    fileName = f"data/Multiclass/{dataSet}.csv"
    df = pd.read_csv(fileName, delimiter=delimiter)
    # Limit rows for test speed; multiclass fixtures only need a small sample
    df = df.iloc[:500, :]
    target = "Type"

    df = df.dropna()
    df, categorical_features, categorical_labels, target_labels, _ = transform_to_numeric(
        df, target
    )

    X, y = df.drop(target, axis=1), df[target]
    columns = X.columns
    no_of_classes = len(np.unique(y))
    no_of_features = X.shape[1]

    idx = np.argsort(y.values).astype(int)
    X, y = X.values[idx, :], y.values[idx]

    test_idx = [np.where(y == i)[0][: num_to_test // no_of_classes] for i in range(no_of_classes)]
    test_index = np.array(test_idx).flatten()
    train_index = np.setdiff1d(np.array(range(len(y))), test_index)

    trainX_cal, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(
        trainX_cal, y_train, test_size=0.33, random_state=42, stratify=y_train
    )
    return (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        no_of_classes,
        no_of_features,
        categorical_features,
        categorical_labels,
        target_labels,
        columns,
    )


def get_classification_model(model_name, X_prop_train, y_prop_train):
    """
    Initializes and fits a classification model.
    Args:
        model_name (str): The name of the model to initialize.
        X_prop_train (np.ndarray): Training features.
        y_prop_train (np.ndarray): Training labels.
    Returns:
        tuple: A tuple containing the fitted model and its name.
    """
    # Allow a fast-test mode to reduce training time (smaller forests)
    fast = bool(os.getenv("FAST_TESTS"))
    t1 = DecisionTreeClassifier()
    r1 = RandomForestClassifier(n_estimators=3 if fast else 10)
    model_dict = {"RF": (r1, "RF"), "DT": (t1, "DT")}

    model, model_name = model_dict[model_name]
    model.fit(X_prop_train, y_prop_train)
    return model, model_name


def initiate_explainer(
    model,
    X_cal,
    y_cal,
    feature_names,
    categorical_features,
    mode,
    class_labels=None,
    difficulty_estimator=None,
    bins=None,
    fast=False,
    verbose=False,
):
    """
    Initialize a CalibratedExplainer instance.

    Parameters:
    model : object
        The machine learning model to be explained.
    X_cal : array-like
        The calibration dataset features.
    y_cal : array-like
        The calibration dataset labels.
    feature_names : list of str
        The names of the features.
    categorical_features : list of str
        The names of the categorical features.
    mode : str
        The mode of the explainer, e.g., 'classification' or 'regression'.
    class_labels : list of str, optional
        The class labels for classification tasks. Default is None.
    difficulty_estimator : object, optional
        The difficulty estimator object. Default is None.
    bins : int, optional
        The number of bins for calibration. Default is None.
    fast : bool, optional
        Whether to use a faster, less accurate method. Default is False.
    verbose : bool, optional
        Whether to print detailed information during initialization. Default is False.

    Returns:
    CalibratedExplainer
        An instance of the CalibratedExplainer class.
    """
    return CalibratedExplainer(
        model,
        X_cal,
        y_cal,
        feature_names=feature_names,
        categorical_features=categorical_features,
        mode=mode,
        class_labels=class_labels,
        bins=bins,
        fast=fast,
        difficulty_estimator=difficulty_estimator,
        verbose=verbose,
    )


def test_binary_ce(binary_dataset):
    """
    Tests the CalibratedExplainer with a binary classification dataset.
    Args:
        binary_dataset (tuple): The binary classification dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, X_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    cal_exp.initialize_reject_learner()
    cal_exp.predict_reject(X_test)

    factual_explanation = cal_exp.explain_factual(X_test)
    factual_explanation[0].add_new_rule_condition(feature_names[0], X_cal[0, 0])
    factual_explanation.add_conjunctions()
    factual_explanation.remove_conjunctions()
    factual_explanation[:1].plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)
    factual_explanation.add_conjunctions(max_rule_size=3)

    alternative_explanation = cal_exp.explore_alternatives(X_test)
    alternative_explanation.add_conjunctions()
    alternative_explanation.remove_conjunctions()
    alternative_explanation[:1].plot(show=False)
    alternative_explanation[X_test == X_test[0]].plot(show=False, style="triangular")
    alternative_explanation.semi_explanations()
    alternative_explanation.counter_explanations()
    alternative_explanation.ensured_explanations()
    alternative_explanation.add_conjunctions(max_rule_size=3)


def test_multiclass_ce_str_target(multiclass_dataset):
    """
    Tests the CalibratedExplainer with a multiclass classification dataset.
    Args:
        multiclass_dataset (tuple): The multiclass classification dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
        _,
        categorical_labels,
        target_labels,
        feature_names,
    ) = multiclass_dataset
    y_prop_train = y_prop_train.astype(str)
    y_cal = y_cal.astype(str)
    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_labels,
        mode="classification",
        class_labels=target_labels,
        verbose=True,
    )

    cal_exp.initialize_reject_learner()
    cal_exp.predict_reject(X_test)

    factual_explanation = cal_exp.explain_factual(X_test)
    factual_explanation.add_conjunctions()
    factual_explanation.remove_conjunctions()
    factual_explanation[:1].plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)
    factual_explanation.add_conjunctions(max_rule_size=3)

    alternative_explanation = cal_exp.explore_alternatives(X_test)
    alternative_explanation.add_conjunctions()
    alternative_explanation.remove_conjunctions()
    alternative_explanation[:1].plot(show=False)
    alternative_explanation[X_test == X_test[0]].plot(show=False, style="triangular")
    alternative_explanation.semi_explanations()
    alternative_explanation.counter_explanations()
    alternative_explanation.add_conjunctions(max_rule_size=3, n_top_features=None)
    alternative_explanation.semi_explanations(only_ensured=True)
    alternative_explanation.counter_explanations(only_ensured=True)


def test_binary_ce_str_target(binary_dataset):
    """
    Tests the CalibratedExplainer with a binary classification dataset.
    Args:
        binary_dataset (tuple): The binary classification dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    y_prop_train = y_prop_train.astype(str)
    y_cal = y_cal.astype(str)
    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, X_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    cal_exp.initialize_reject_learner()
    cal_exp.predict_reject(X_test)

    factual_explanation = cal_exp.explain_factual(X_test)
    factual_explanation[0].add_new_rule_condition(feature_names[0], X_cal[0, 0])
    factual_explanation.add_conjunctions()
    factual_explanation.remove_conjunctions()
    factual_explanation[:1].plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)
    factual_explanation.add_conjunctions(max_rule_size=3)

    alternative_explanation = cal_exp.explore_alternatives(X_test)
    alternative_explanation.add_conjunctions()
    alternative_explanation.remove_conjunctions()
    alternative_explanation[:1].plot(show=False)
    alternative_explanation[X_test == X_test[0]].plot(show=False, style="triangular")
    alternative_explanation.semi_explanations()
    alternative_explanation.counter_explanations()
    alternative_explanation.ensured_explanations()
    alternative_explanation.add_conjunctions(max_rule_size=3)


def test_multiclass_ce(multiclass_dataset):
    """
    Tests the CalibratedExplainer with a multiclass classification dataset.
    Args:
        multiclass_dataset (tuple): The multiclass classification dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
        _,
        categorical_labels,
        target_labels,
        feature_names,
    ) = multiclass_dataset
    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_labels,
        mode="classification",
        class_labels=target_labels,
        verbose=True,
    )

    cal_exp.initialize_reject_learner()
    cal_exp.predict_reject(X_test)

    cal_exp.predict(X_test)
    cal_exp.predict_proba(X_test)

    factual_explanation = cal_exp.explain_factual(X_test)
    factual_explanation.add_conjunctions()
    factual_explanation.remove_conjunctions()
    factual_explanation[:1].plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)
    factual_explanation.add_conjunctions(max_rule_size=3)

    alternative_explanation = cal_exp.explore_alternatives(X_test)
    alternative_explanation.add_conjunctions()
    alternative_explanation.remove_conjunctions()
    alternative_explanation[:1].plot(show=False)
    alternative_explanation[X_test == X_test[0]].plot(show=False, style="triangular")
    alternative_explanation.semi_explanations()
    alternative_explanation.counter_explanations()
    alternative_explanation.add_conjunctions(max_rule_size=3, n_top_features=None)
    alternative_explanation.semi_explanations(only_ensured=True)
    alternative_explanation.counter_explanations(only_ensured=True)


def test_binary_conditional_ce(binary_dataset):
    """
    Tests the CalibratedExplainer with a binary classification dataset and conditional bins.
    Args:
        binary_dataset (tuple): The binary classification dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    target_labels = ["No", "Yes"]
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="classification",
        class_labels=target_labels,
        bins=X_cal[:, 0],
    )

    cal_exp.initialize_reject_learner()
    cal_exp.predict_reject(X_test, bins=X_test[:, 0])

    factual_explanation = cal_exp.explain_factual(X_test, bins=X_test[:, 0])
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_alternatives(X_test, bins=X_test[:, 0])
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)


@pytest.mark.slow
def test_multiclass_conditional_ce(multiclass_dataset):
    """
    Tests the CalibratedExplainer with a multiclass classification dataset and conditional bins.
    Args:
        multiclass_dataset (tuple): The multiclass classification dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
        _,
        categorical_labels,
        _,
        feature_names,
    ) = multiclass_dataset
    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_labels,
        mode="classification",
        bins=X_cal[:, 0],
    )

    factual_explanation = cal_exp.explain_factual(X_test, bins=X_test[:, 0])
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_alternatives(X_test, bins=X_test[:, 0])
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)


def test_binary_fast_ce(binary_dataset):
    """
    Tests the CalibratedExplainer with a binary classification dataset and perturbation.
    Args:
        binary_dataset (tuple): The binary classification dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, X_cal, y_cal, feature_names, categorical_features, mode="classification", fast=True
    )

    fast_explanation = cal_exp.explain_fast(X_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()
    fast_explanation.remove_conjunctions()
    fast_explanation[:1].plot(show=False)
    fast_explanation[0].plot(show=False, uncertainty=True)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions(max_rule_size=3)


def test_multiclass_fast_ce(multiclass_dataset):
    """
    Tests the CalibratedExplainer with a multiclass classification dataset and perturbation.
    Args:
        multiclass_dataset (tuple): The multiclass classification dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
        _,
        categorical_labels,
        target_labels,
        feature_names,
    ) = multiclass_dataset
    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_labels,
        mode="classification",
        class_labels=target_labels,
        verbose=True,
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(X_test)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()
    fast_explanation.remove_conjunctions()
    fast_explanation[:1].plot(show=False)
    fast_explanation[0].plot(show=False, uncertainty=True)
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions(max_rule_size=3)


def test_binary_conditional_fast_ce(binary_dataset):
    """
    Tests the CalibratedExplainer with a binary classification dataset, conditional bins, and perturbation.
    Args:
        binary_dataset (tuple): The binary classification dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    target_labels = ["No", "Yes"]
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="classification",
        class_labels=target_labels,
        bins=X_cal[:, 0],
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(X_test, bins=X_test[:, 0])
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()
    fast_explanation[:1].plot(show=False)
    fast_explanation[0].plot(show=False, uncertainty=True)


def test_multiclass_fast_conditional_ce(multiclass_dataset):
    """
    Tests the CalibratedExplainer with a multiclass classification dataset, conditional bins, and perturbation.
    Args:
        multiclass_dataset (tuple): The multiclass classification dataset.
    """
    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
        _,
        categorical_labels,
        _,
        feature_names,
    ) = multiclass_dataset
    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_labels,
        mode="classification",
        bins=X_cal[:, 0],
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(X_test, bins=X_test[:, 0])
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()
    fast_explanation[:1].plot(show=False)
    fast_explanation[0].plot(show=False, uncertainty=True)
