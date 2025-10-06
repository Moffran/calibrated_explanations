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

import pytest
from tests._helpers import get_classification_model, initiate_explainer


def test_binary_ce(binary_dataset):
    """
    Tests the CalibratedExplainer with a binary classification dataset.
    Args:
        binary_dataset (tuple): The binary classification dataset.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    cal_exp.initialize_reject_learner()
    cal_exp.predict_reject(x_test)

    factual_explanation = cal_exp.explain_factual(x_test)
    factual_explanation[0].add_new_rule_condition(feature_names[0], x_cal[0, 0])
    factual_explanation.add_conjunctions()
    factual_explanation.remove_conjunctions()
    factual_explanation[:1].plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)
    factual_explanation.add_conjunctions(max_rule_size=3)

    alternative_explanation = cal_exp.explore_alternatives(x_test)
    alternative_explanation.add_conjunctions()
    alternative_explanation.remove_conjunctions()
    alternative_explanation[:1].plot(show=False)
    alternative_explanation[x_test == x_test[0]].plot(show=False, style="triangular")
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
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
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
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_labels,
        mode="classification",
        class_labels=target_labels,
        verbose=True,
    )

    cal_exp.initialize_reject_learner()
    cal_exp.predict_reject(x_test)

    factual_explanation = cal_exp.explain_factual(x_test)
    factual_explanation.add_conjunctions()
    factual_explanation.remove_conjunctions()
    factual_explanation[:1].plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)
    factual_explanation.add_conjunctions(max_rule_size=3)

    alternative_explanation = cal_exp.explore_alternatives(x_test)
    alternative_explanation.add_conjunctions()
    alternative_explanation.remove_conjunctions()
    alternative_explanation[:1].plot(show=False)
    alternative_explanation[x_test == x_test[0]].plot(show=False, style="triangular")
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
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    y_prop_train = y_prop_train.astype(str)
    y_cal = y_cal.astype(str)
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    cal_exp.initialize_reject_learner()
    cal_exp.predict_reject(x_test)

    factual_explanation = cal_exp.explain_factual(x_test)
    factual_explanation[0].add_new_rule_condition(feature_names[0], x_cal[0, 0])
    factual_explanation.add_conjunctions()
    factual_explanation.remove_conjunctions()
    factual_explanation[:1].plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)
    factual_explanation.add_conjunctions(max_rule_size=3)

    alternative_explanation = cal_exp.explore_alternatives(x_test)
    alternative_explanation.add_conjunctions()
    alternative_explanation.remove_conjunctions()
    alternative_explanation[:1].plot(show=False)
    alternative_explanation[x_test == x_test[0]].plot(show=False, style="triangular")
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
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        _,
        categorical_labels,
        target_labels,
        feature_names,
    ) = multiclass_dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_labels,
        mode="classification",
        class_labels=target_labels,
        verbose=True,
    )

    cal_exp.initialize_reject_learner()
    cal_exp.predict_reject(x_test)

    cal_exp.predict(x_test)
    cal_exp.predict_proba(x_test)

    factual_explanation = cal_exp.explain_factual(x_test)
    factual_explanation.add_conjunctions()
    factual_explanation.remove_conjunctions()
    factual_explanation[:1].plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)
    factual_explanation.add_conjunctions(max_rule_size=3)

    alternative_explanation = cal_exp.explore_alternatives(x_test)
    alternative_explanation.add_conjunctions()
    alternative_explanation.remove_conjunctions()
    alternative_explanation[:1].plot(show=False)
    alternative_explanation[x_test == x_test[0]].plot(show=False, style="triangular")
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
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    target_labels = ["No", "Yes"]
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="classification",
        class_labels=target_labels,
        bins=x_cal[:, 0],
    )

    cal_exp.initialize_reject_learner()
    cal_exp.predict_reject(x_test, bins=x_test[:, 0])

    factual_explanation = cal_exp.explain_factual(x_test, bins=x_test[:, 0])
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_alternatives(x_test, bins=x_test[:, 0])
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
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        _,
        categorical_labels,
        _,
        feature_names,
    ) = multiclass_dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_labels,
        mode="classification",
        bins=x_cal[:, 0],
    )

    factual_explanation = cal_exp.explain_factual(x_test, bins=x_test[:, 0])
    factual_explanation.add_conjunctions()
    factual_explanation.plot(show=False)
    factual_explanation[0].plot(show=False, uncertainty=True)

    alternative_explanation = cal_exp.explore_alternatives(x_test, bins=x_test[:, 0])
    alternative_explanation.add_conjunctions()
    alternative_explanation.plot(show=False)


def test_binary_fast_ce(binary_dataset):
    """
    Tests the CalibratedExplainer with a binary classification dataset and perturbation.
    Args:
        binary_dataset (tuple): The binary classification dataset.
    """
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification", fast=True
    )

    fast_explanation = cal_exp.explain_fast(x_test)
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
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        _,
        categorical_labels,
        target_labels,
        feature_names,
    ) = multiclass_dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_labels,
        mode="classification",
        class_labels=target_labels,
        verbose=True,
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(x_test)
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
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    target_labels = ["No", "Yes"]
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="classification",
        class_labels=target_labels,
        bins=x_cal[:, 0],
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(x_test, bins=x_test[:, 0])
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
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        _,
        categorical_labels,
        _,
        feature_names,
    ) = multiclass_dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_labels,
        mode="classification",
        bins=x_cal[:, 0],
        fast=True,
    )

    fast_explanation = cal_exp.explain_fast(x_test, bins=x_test[:, 0])
    with pytest.warns(UserWarning):
        fast_explanation.add_conjunctions()
    fast_explanation[:1].plot(show=False)
    fast_explanation[0].plot(show=False, uncertainty=True)
