"""Minimal tests for the config builder scaffolding (Phase 2).

These tests intentionally avoid changing public APIs by exercising only
the internal/private `_from_config` helper and the fluent builder.
"""

from __future__ import annotations

from dataclasses import is_dataclass

from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations.api.config import ExplainerBuilder, ExplainerConfig
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer


def test_explainer_config_dataclass_and_defaults():
    model = RandomForestClassifier()
    cfg = ExplainerConfig(model=model)
    assert is_dataclass(cfg)
    assert cfg.task == "auto"
    assert cfg.low_high_percentiles == (5, 95)
    assert cfg.threshold is None
    assert cfg.preprocessor is None
    assert cfg.auto_encode == "auto"
    assert cfg.unseen_category_policy == "error"


def test_explainer_builder_fluent_roundtrip():
    model = RandomForestClassifier()
    b = (
        ExplainerBuilder(model)
        .task("classification")
        .low_high_percentiles((10, 90))
        .threshold(0.7)
        .preprocessor(None)
        .auto_encode("auto")
        .unseen_category_policy("ignore")
        .parallel_workers(2)
    )
    cfg = b.build_config()
    assert isinstance(cfg, ExplainerConfig)
    assert cfg.model is model
    assert cfg.task == "classification"
    assert cfg.low_high_percentiles == (10, 90)
    assert cfg.threshold == 0.7
    assert cfg.unseen_category_policy == "ignore"


def test_wrap_from_config_private_helper():
    model = RandomForestClassifier()
    cfg = ExplainerConfig(model=model)
    w = WrapCalibratedExplainer._from_config(cfg)  # private, intentional
    assert isinstance(w, WrapCalibratedExplainer)
    assert w.learner is model


def test_wrap_from_config_applies_defaults(monkeypatch):
    # Configure defaults
    model = RandomForestClassifier()
    cfg = ExplainerConfig(model=model, low_high_percentiles=(10, 90), threshold=0.3)
    w = WrapCalibratedExplainer._from_config(cfg)

    # Monkeypatch underlying explainer to capture kwargs passed through
    class DummyExplainer:
        def explain_factual(self, X, **kwargs):  # noqa: D401
            return kwargs

        def explore_alternatives(self, X, **kwargs):  # noqa: D401
            return kwargs

    w.fitted = True
    w.calibrated = True
    w.explainer = DummyExplainer()  # type: ignore[assignment]

    # factual inherits defaults
    out = w.explain_factual([[0.0]])
    assert out["low_high_percentiles"] == (10, 90)
    assert out["threshold"] == 0.3

    # explicit kwargs override config defaults
    out2 = w.explain_factual([[0.0]], low_high_percentiles=(5, 95), threshold=None)
    assert out2["low_high_percentiles"] == (5, 95)
    assert out2["threshold"] is None

    # alternatives also inherit
    out3 = w.explore_alternatives([[0.0]])
    assert out3["low_high_percentiles"] == (10, 90)
    assert out3["threshold"] == 0.3


def test_wrap_from_config_applies_defaults_fast():
    # Configure defaults
    model = RandomForestClassifier()
    cfg = ExplainerConfig(model=model, low_high_percentiles=(20, 80), threshold=0.4)
    w = WrapCalibratedExplainer._from_config(cfg)

    class DummyExplainerFast:
        def explain_fast(self, X, **kwargs):  # noqa: D401
            return kwargs

    w.fitted = True
    w.calibrated = True
    w.explainer = DummyExplainerFast()  # type: ignore[assignment]

    out = w.explain_fast([[0.0]])
    assert out["low_high_percentiles"] == (20, 80)
    assert out["threshold"] == 0.4
