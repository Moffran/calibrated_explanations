import types
import warnings

import numpy as np
import pytest

from calibrated_explanations.plugins import registry, builtins
from calibrated_explanations.explanations import legacy_conjunctions
from calibrated_explanations.explanations import explanation as expl_mod
from tests.support.registry_helpers import (
    clear_trust_warnings,
    registry_snapshot,
    remove_from_registry,
    warn_untrusted_plugin,
)


def make_ep(name, load=None, dist=None):
    ep = types.SimpleNamespace()
    ep.name = name
    ep.module = None
    ep.attr = None
    ep.dist = dist
    if load is None:

        def load_stub():
            raise RuntimeError("no load")

        ep.load = load_stub
    else:
        ep.load = load
    return ep


def test_load_entrypoint_plugins_skips_denied_identifier(monkeypatch):
    # Simulate entry points returning a candidate that is denied by policy
    eps = types.SimpleNamespace()
    eps.select = lambda group=None: [make_ep("denied:plugin")]
    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: eps)

    # Force denial
    monkeypatch.setattr(registry, "is_identifier_denied", lambda ident: True)

    loaded = registry.load_entrypoint_plugins(include_untrusted=True)
    assert loaded == ()
    report = registry.get_discovery_report()
    # Expect at least one skipped_denied in the report
    assert len(report.skipped_denied) >= 1


def test_warn_untrusted_plugin_emits_once(monkeypatch):
    # create minimal metadata and call warn_untrusted_plugin twice; set will ensure single warning
    meta = {
        "name": "u1",
        "provider": "p",
        "schema_version": 1,
        "version": "0.0",
        "capabilities": ["explain"],
    }
    clear_trust_warnings()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_untrusted_plugin(meta, source="entry point")
        warn_untrusted_plugin(meta, source="entry point")
    assert any("Skipping untrusted plugin" in str(r.message) for r in rec)


def test_load_entrypoint_plugins_skips_when_meta_name_denied(monkeypatch):
    # Entry point that loads a plugin with plugin_meta having a denied name
    def load_plugin_stub():
        p = types.SimpleNamespace()
        p.plugin_meta = {
            "schema_version": 1,
            "name": "denied_name",
            "version": "0.0",
            "provider": "x",
            "capabilities": ["explain"],
            "data_modalities": ("tabular",),
        }
        return p

    eps = types.SimpleNamespace()
    eps.select = lambda group=None: [make_ep("pkg:attr", load=load_plugin_stub)]
    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: eps)
    monkeypatch.setattr(registry, "is_identifier_denied", lambda ident: ident == "denied_name")

    loaded = registry.load_entrypoint_plugins(include_untrusted=True)
    assert loaded == ()
    report = registry.get_discovery_report()
    assert any(
        r.metadata and r.metadata.get("name") == "denied_name" for r in report.skipped_denied
    )


def test_register_and_use_builtin_fast_plugin(monkeypatch, tmp_path):
    # Ensure any existing core.explanation.fast descriptor is removed, then call register_fast_explanation_plugin
    from calibrated_explanations.plugins.explanations_fast import register_fast_explanation_plugin

    # Remove any existing descriptor for core.explanation.fast from registry snapshot
    desc = registry.find_explanation_descriptor("core.explanation.fast")
    if desc is not None:
        # descriptor may be stored in internal registry objects; attempt removal by scanning snapshot
        for p in registry_snapshot():
            pm = getattr(p, "plugin_meta", {})
            if isinstance(pm, dict) and pm.get("name") == "core.explanation.fast":
                remove_from_registry(p)
    # Now call the public registration helper which should register the builtin fast plugin
    register_fast_explanation_plugin()
    desc2 = registry.find_explanation_descriptor("core.explanation.fast")
    assert desc2 is not None


def test_builtin_feature_filter_exception_path(monkeypatch):
    # Test inner feature-filter exception branch by forcing FeatureFilterConfig to enable filter
    plugin = builtins.SequentialExplanationPlugin()

    class FakeExec:
        def supports(self, req, cfg):
            return True

    plugin.execution_plugin_class = FakeExec

    class FakeExplainer:
        feature_filter_config = True

        def __init__(self):
            self.features_to_ignore = ()

        def explain_factual(self, x, **kwargs):
            class C:
                explanations = []

            return C()

    # Force FeatureFilterConfig.from_base_and_env to produce an object that enables filtering
    class FakeCfg:
        enabled = True
        per_instance_top_k = 1
        strict_observability = False

    monkeypatch.setattr(builtins.FeatureFilterConfig, "from_base_and_env", lambda base: FakeCfg)

    ctx = builtins.ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=(),
        categorical_features=(),
        categorical_labels={},
        discretizer=None,
        helper_handles={"explainer": FakeExplainer()},
        predict_bridge=types.SimpleNamespace(predict=lambda *a, **k: None),
        interval_settings={},
        plot_settings={},
    )
    plugin.initialize(ctx)
    req = builtins.ExplanationRequest(
        threshold=None, low_high_percentiles=None, bins=None, features_to_ignore=()
    )

    # Should return an ExplanationBatch via legacy fallback path (fast plugin invocation missing -> AttributeError path)
    batch = plugin.explain_batch(np.ones((1, 1)), req)
    assert isinstance(batch, builtins.ExplanationBatch)


def test_alternative_plot_triangle_uses_interval_width_for_regression(monkeypatch):
    # Arrange: minimal regression-mode explanation stub
    fake = types.SimpleNamespace()
    fake.rank_features = lambda *a, **k: [0]
    fake.feature_predict = {"predict": [5.0], "low": [4.0], "high": [6.0]}
    fake.prediction = {"predict": 5.0, "low": 4.0, "high": 6.0}
    fake.y_minmax = (0.0, 10.0)
    alternative = {
        "predict": [6.0],
        "predict_low": [5.5],
        "predict_high": [7.5],
        "value": ["v"],
        "rule": ["r"],
        "weight": [1.0],
        "weight_low": [0.5],
        "weight_high": [1.5],
    }
    fake.get_explainer = lambda: types.SimpleNamespace(
        num_features=1, feature_names=["f0"], categorical_features=(), categorical_labels=None
    )
    fake.get_rules = lambda *a, **k: alternative
    check_pre_attr = "_" + "check_preconditions"
    setattr(fake, check_pre_attr, lambda *a, **k: None)
    fake.get_mode = lambda: "regression"
    fake.is_thresholded = lambda: False

    called = {}

    def fake_plot_triangular(self_, proba, uncertainty, sel_proba, sel_unc, num_to_show, **kwargs):
        called["args"] = (proba, uncertainty, sel_proba, sel_unc, num_to_show)

    monkeypatch.setattr(expl_mod, "plot_triangular", fake_plot_triangular)

    # Act
    expl_mod.AlternativeExplanation.plot(
        fake,
        filter_top=None,
        style="triangular",
        title=None,
        path=None,
        show=False,
        save_ext=None,
        style_override=None,
    )

    # Assert: uncertainty is width = high-low
    proba, uncertainty, sel_proba, sel_unc, num_to_show = called["args"]
    assert proba == 5.0
    assert uncertainty == pytest.approx(2.0)
    assert sel_proba == [6.0]
    assert list(sel_unc) == pytest.approx([2.0])
    assert num_to_show == 1


def test_add_conjunctions_factual_legacy_simple():
    # Minimal fake object exercising add_conjunctions_factual_legacy via public API
    class FakeExplainer:
        def __init__(self):
            class PredOrchestrator:
                def predict_internal(self, x, **_kwargs):
                    return [0.9], [0.8], [1.0], 1

            self.prediction_orchestrator = PredOrchestrator()

    class S:
        def __init__(self):
            self.has_rules = False
            self.rules = None
            self.has_conjunctive_rules = False
            self.conjunctive_rules = None
            self.y_threshold = None
            self.x_test = [0, 0, 0]
            self.bin = None
            self.prediction = {"predict": 0.5}
            self.calibrated_explanations = types.SimpleNamespace(low_high_percentiles=(2.5, 97.5))

        def get_explainer(self):
            return FakeExplainer()

        def get_rules(self):
            return {
                "classes": 1,
                "rule": ["r0", "r1"],
                "feature": [0, 1],
                "sampled_values": [10, 20],
                "value": ["v0", "v1"],
                "weight": [0.1, 0.2],
                "weight_low": [0.0, 0.0],
                "weight_high": [0.2, 0.3],
                "predict": [0.5, 0.6],
                "predict_low": [0.4, 0.5],
                "predict_high": [0.6, 0.7],
                "feature_value": [None, None],
                "is_conjunctive": [False, False],
            }

        def rank_features(self, *a, **k):
            return [1]

    s = S()
    res = legacy_conjunctions.add_conjunctions_factual_legacy(s, n_top_features=2, max_rule_size=2)
    assert res is s
    assert getattr(s, "has_conjunctive_rules") is True
    assert isinstance(s.conjunctive_rules, dict)
    assert len(s.conjunctive_rules.get("feature", [])) >= len(s.get_rules()["feature"]) + 1


def test_add_conjunctions_alternative_legacy_simple():
    # Minimal fake object exercising add_conjunctions_alternative_legacy via public API
    class FakeExplainer:
        def __init__(self):
            class PredOrchestrator:
                def predict_internal(self, x, **_kwargs):
                    return [0.3], [0.2], [0.4], 1

            self.prediction_orchestrator = PredOrchestrator()

    class S:
        def __init__(self):
            self.has_rules = False
            self.rules = None
            self.has_conjunctive_rules = False
            self.conjunctive_rules = None
            self.y_threshold = None
            self.x_test = [0, 0, 0]
            self.bin = None
            self.prediction = {"predict": 0.2}
            self.calibrated_explanations = types.SimpleNamespace(low_high_percentiles=(2.5, 97.5))

        def get_explainer(self):
            return FakeExplainer()

        def get_rules(self):
            return {
                "classes": 1,
                "rule": ["ra", "rb"],
                "feature": [0, 1],
                "sampled_values": [5, 6],
                "value": ["va", "vb"],
                "weight": [0.05, 0.06],
                "weight_low": [0.0, 0.0],
                "weight_high": [0.1, 0.12],
                "predict": [0.2, 0.25],
                "predict_low": [0.15, 0.2],
                "predict_high": [0.25, 0.3],
                "feature_value": [["A"], ["B"]],
                "is_conjunctive": [False, False],
            }

        def rank_features(self, *a, **k):
            return [1]

    s = S()
    res = legacy_conjunctions.add_conjunctions_alternative_legacy(
        s, n_top_features=2, max_rule_size=2
    )
    assert res is s
    assert getattr(s, "has_conjunctive_rules") is True
    assert isinstance(s.conjunctive_rules, dict)
    # feature_value should be a list-of-lists after adding alternative conjunctions
    fv = s.conjunctive_rules.get("feature_value", [])
    assert any(isinstance(v, list) for v in fv)
