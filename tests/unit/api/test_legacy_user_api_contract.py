"""Regression tests for the legacy user-facing API surface."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import calibrated_explanations as ce
from calibrated_explanations.api.config import ExplainerConfig
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.explanations import CalibratedExplanations


def param_names(func):
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
    params = param_names(ce.WrapCalibratedExplainer.predict)
    assert params[:3] == ["x", "uq_interval", "calibrated"]
    assert params[-1] == "kwargs"  # **kwargs preserves legacy aliases


def test_wrap_predict_proba_signature_includes_threshold():
    params = param_names(ce.WrapCalibratedExplainer.predict_proba)
    assert params[:3] == ["x", "uq_interval", "calibrated"]
    assert "threshold" in params
    assert params[-1] == "kwargs"


def test_calibrated_explainer_constructor_signature():
    params = param_names(ce.CalibratedExplainer.__init__)
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
    factual_params = param_names(ce.CalibratedExplainer.explain_factual)
    alternative_params = param_names(ce.CalibratedExplainer.explore_alternatives)
    for params in (factual_params, alternative_params):
        assert "x" in params
        assert "threshold" in params
        assert "low_high_percentiles" in params
        assert "bins" in params

        # Handle optional kwargs and _use_plugin
        if params[-1] == "kwargs":
            if params[-2] == "_use_plugin":
                assert params[-3] == "features_to_ignore"
            else:
                assert params[-2] == "features_to_ignore"
        else:
            assert params[-2] == "features_to_ignore"
            assert params[-1].startswith("_use_plugin")


def test_explanation_collection_api_is_stable():
    required = [
        "plot",
        "add_conjunctions",
        "remove_conjunctions",
        "__getitem__",
    ]
    for attr in required:
        assert hasattr(CalibratedExplanations, attr), attr

    plot_params = param_names(CalibratedExplanations.plot)
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

    add_params = param_names(CalibratedExplanations.add_conjunctions)
    assert add_params == ["n_top_features", "max_rule_size", "kwargs"]


# ---------------------------------------------------------------------------
# ADR-020 gap 2 — parity assertions for explain_factual / explore_alternatives
# ---------------------------------------------------------------------------


def make_parity_wrapper(threshold, low_high_percentiles):
    """Return a (wrapper, captured) pair for delegation parity tests."""
    captured: dict[str, dict] = {}

    class CapturingExplainer:
        def explain_factual(self, x, **kwargs):
            captured["factual"] = dict(kwargs)
            return MagicMock()

        def explore_alternatives(self, x, **kwargs):
            captured["alternatives"] = dict(kwargs)
            return MagicMock()

    model = MagicMock()
    cfg = ExplainerConfig(
        model=model, threshold=threshold, low_high_percentiles=low_high_percentiles
    )
    w = WrapCalibratedExplainer.from_config(cfg)
    w.fitted = True
    w.calibrated = True
    w.explainer = CapturingExplainer()
    return w, captured


def test_wrap_explain_factual_parity_explicit_threshold_is_forwarded():
    """Explicit threshold passed to explain_factual must arrive at the underlying explainer."""
    w, captured = make_parity_wrapper(threshold=0.3, low_high_percentiles=(10, 90))
    w.explain_factual([[0.0]], threshold=0.8)
    assert captured["factual"]["threshold"] == 0.8


def test_wrap_explain_factual_parity_config_defaults_are_injected_when_absent():
    """When no explicit threshold/percentiles passed, config defaults must be injected."""
    w, captured = make_parity_wrapper(threshold=0.3, low_high_percentiles=(10, 90))
    w.explain_factual([[0.0]])
    assert captured["factual"]["threshold"] == 0.3
    assert captured["factual"]["low_high_percentiles"] == (10, 90)


def test_wrap_explore_alternatives_parity_explicit_kwargs_forwarded():
    """Explicit kwargs passed to explore_alternatives must arrive at the underlying explainer."""
    w, captured = make_parity_wrapper(threshold=0.5, low_high_percentiles=(5, 95))
    w.explore_alternatives([[0.0]], threshold=0.7)
    assert captured["alternatives"]["threshold"] == 0.7


def test_wrap_explore_alternatives_parity_config_defaults_injected():
    """Config defaults must be injected into explore_alternatives when not explicitly passed."""
    w, captured = make_parity_wrapper(threshold=0.5, low_high_percentiles=(5, 95))
    w.explore_alternatives([[0.0]])
    assert captured["alternatives"]["threshold"] == 0.5
    assert captured["alternatives"]["low_high_percentiles"] == (5, 95)


def test_wrap_explain_methods_accept_var_keyword_for_ce_parity():
    """Both wrapper explain methods must accept **kwargs so all CalibratedExplainer kwargs are passable."""
    for method_name in ("explain_factual", "explore_alternatives"):
        sig = inspect.signature(getattr(WrapCalibratedExplainer, method_name))
        var_kw_params = [
            p for p in sig.parameters.values() if p.kind == inspect.Parameter.VAR_KEYWORD
        ]
        assert var_kw_params, (
            f"WrapCalibratedExplainer.{method_name} must accept **kwargs "
            "for parity with CalibratedExplainer parameter surface"
        )
