# pylint: disable=invalid-name, protected-access, too-many-locals, too-many-arguments, too-many-positional-arguments, line-too-long, redefined-outer-name, no-member
"""
Duplicated multiclass tests exercising the `multi_labels_enabled=True` path.

These are duplicates of the important integration tests that normally run the
argmax-class path. They ensure the multi-label (per-class) generation path
creates rules and explanation containers without regressions.
"""

import pytest
from tests.helpers.explainer_utils import initiate_explainer
from tests.helpers.model_utils import get_classification_model

pytestmark = pytest.mark.integration


@pytest.mark.viz
def test_multiclass_ce_multi_labels_dup(multiclass_dataset):
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

    # Run the multiclass per-class generation path
    multi_factual = cal_exp.explain_factual(x_test, multi_labels_enabled=True)
    multi_alternative = cal_exp.explore_alternatives(x_test, multi_labels_enabled=True)

    # Basic sanity: per-instance containers created and per-class explanations available
    assert len(multi_factual.explanations) == len(x_test)
    assert len(multi_alternative.explanations) == len(x_test)
    assert set(multi_factual.explanations[0].keys())
    assert set(multi_alternative.explanations[0].keys())


@pytest.mark.viz
def test_multiclass_ce_str_target_multi_labels_dup(multiclass_dataset):
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
    # use string labels like the original test
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

    multi_factual = cal_exp.explain_factual(x_test, multi_labels_enabled=True)
    multi_alternative = cal_exp.explore_alternatives(x_test, multi_labels_enabled=True)

    # ensure containers created and basic plotting paths are exercised
    multi_factual.plot(show=False, uncertainty=True)
    multi_alternative.plot(show=False)
    assert multi_factual is not None
    assert multi_alternative is not None


@pytest.mark.slow
@pytest.mark.viz
def test_multiclass_conditional_ce_multi_labels_dup(multiclass_dataset):
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

    # exercise conditional + multi-label generation
    multi_factual = cal_exp.explain_factual(x_test, bins=x_test[:, 0], multi_labels_enabled=True)
    multi_factual.add_conjunctions()
    assert multi_factual is not None
    assert len(multi_factual.explanations) == len(x_test)
    # Plot a per-class explanation for the first instance to avoid container-level
    # dict-plot branches that expect internal `index` bookkeeping.
    first_class_key = int(next(iter(multi_factual.explanations[0].keys())))
    ex = multi_factual.get_explanation(0, first_class_key)
    assert ex is not None
    ex.plot(show=False, uncertainty=True)
