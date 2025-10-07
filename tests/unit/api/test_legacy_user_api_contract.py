"""Regression tests for the legacy user-facing API surface."""

from __future__ import annotations

import inspect

import calibrated_explanations as ce
from calibrated_explanations.explanations.explanations import CalibratedExplanations


def _param_names(func):
    sig = inspect.signature(func)
    return [name for name in sig.parameters if name != "self"]


def test_wrap_calibrated_explainer_lifecycle_methods_exist():
    required = [
        "fit",
        "calibrate",
        "predict",
        "predict_proba",
        "explain_factual",
        "explore_alternatives",
        "set_difficulty_estimator",
    ]
    for attr in required:
        assert hasattr(ce.WrapCalibratedExplainer, attr), attr


def test_wrap_predict_signature_includes_legacy_parameters():
    params = _param_names(ce.WrapCalibratedExplainer.predict)
    assert params[:3] == ["x", "uq_interval", "calibrated"]
    assert params[-1] == "kwargs"  # **kwargs preserves legacy aliases


def test_wrap_predict_proba_signature_includes_threshold():
    params = _param_names(ce.WrapCalibratedExplainer.predict_proba)
    assert params[:3] == ["x", "uq_interval", "calibrated"]
    assert "threshold" in params
    assert params[-1] == "kwargs"


def test_calibrated_explainer_constructor_signature():
    params = _param_names(ce.CalibratedExplainer.__init__)
    expected_prefix = [
        "learner",
        "x_cal",
        "y_cal",
        "mode",
        "feature_names",
        "categorical_features",
        "categorical_labels",
        "class_labels",
        "bins",
        "difficulty_estimator",
    ]
    assert params[: len(expected_prefix)] == expected_prefix
    assert params[-1] == "kwargs"


def test_calibrated_explainer_factories_support_expected_keywords():
    factual_params = _param_names(ce.CalibratedExplainer.explain_factual)
    alternative_params = _param_names(ce.CalibratedExplainer.explore_alternatives)
    for params in (factual_params, alternative_params):
        assert "x" in params
        assert "threshold" in params
        assert "low_high_percentiles" in params
        assert "bins" in params
        assert params[-2] == "features_to_ignore"
        assert params[-1].startswith("_use_plugin") or params[-1] == "kwargs"


def test_explanation_collection_api_is_stable():
    required = [
        "plot",
        "add_conjunctions",
        "remove_conjunctions",
        "get_explanation",
        "__getitem__",
    ]
    for attr in required:
        assert hasattr(CalibratedExplanations, attr), attr

    plot_params = _param_names(CalibratedExplanations.plot)
    for expected in [
        "index",
        "filter_top",
        "show",
        "filename",
        "uncertainty",
        "style",
        "rnk_metric",
        "rnk_weight",
    ]:
        assert expected in plot_params

    add_params = _param_names(CalibratedExplanations.add_conjunctions)
    assert add_params == ["n_top_features", "max_rule_size"]
