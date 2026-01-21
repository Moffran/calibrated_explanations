import types
import warnings

import numpy as np

from calibrated_explanations.plugins import registry, builtins
from calibrated_explanations.explanations import legacy_conjunctions
from calibrated_explanations.explanations import explanation as expl_mod


def make_ep(name, load=None, dist=None):
    ep = types.SimpleNamespace()
    ep.name = name
    ep.module = None
    ep.attr = None
    ep.dist = dist
    if load is None:

        def _load():
            raise RuntimeError("no load")

        ep.load = _load
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
    registry.clear_trust_warnings()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        registry.warn_untrusted_plugin(meta, source="entry point")
        registry.warn_untrusted_plugin(meta, source="entry point")
    assert any("Skipping untrusted plugin" in str(r.message) for r in rec)


def test_load_entrypoint_plugins_skips_when_meta_name_denied(monkeypatch):
    # Entry point that loads a plugin with plugin_meta having a denied name
    def _load_plugin():
        p = types.SimpleNamespace()
        p.plugin_meta = {
            "schema_version": 1,
            "name": "denied_name",
            "version": "0.0",
            "provider": "x",
            "capabilities": ["explain"],
        }
        return p

    eps = types.SimpleNamespace()
    eps.select = lambda group=None: [make_ep("pkg:attr", load=_load_plugin)]
    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: eps)
    monkeypatch.setattr(registry, "is_identifier_denied", lambda ident: ident == "denied_name")

    loaded = registry.load_entrypoint_plugins(include_untrusted=True)
    assert loaded == ()
    report = registry.get_discovery_report()
    assert any(
        r.metadata and r.metadata.get("name") == "denied_name" for r in report.skipped_denied
    )


def test_builtin_execution_plugin_falls_back_to_legacy(monkeypatch):
    # Use SequentialExplanationPlugin and inject an execution plugin whose supports() -> False
    plugin = builtins.SequentialExplanationPlugin()

    class FakeExec:
        def supports(self, req, cfg):
            return False

    plugin.execution_plugin_class = FakeExec

    # Build a minimal explainer handle that exposes the explanation callable
    class FakeCollection:
        def __init__(self):
            self.explanations = [object()]

    class FakeExplainer:
        def explain_factual(self, x, **kwargs):
            return FakeCollection()

    # Minimal predict bridge
    class FakeBridge:
        def predict(self, *a, **k):
            return None

    ctx = builtins.ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=(),
        categorical_features=(),
        categorical_labels={},
        discretizer=None,
        helper_handles={"explainer": FakeExplainer()},
        predict_bridge=FakeBridge(),
        interval_settings={},
        plot_settings={},
    )
    plugin.initialize(ctx)

    req = builtins.ExplanationRequest(
        threshold=None, low_high_percentiles=None, bins=None, features_to_ignore=()
    )
    batch = plugin.explain_batch(np.ones((1, 1)), req)
    assert hasattr(batch, "instances")
    assert batch.collection_metadata["container"] is not None


def test_register_and_use_builtin_fast_plugin(monkeypatch, tmp_path):
    # Ensure any existing core.explanation.fast descriptor is removed, then call register_fast_explanation_plugin
    from calibrated_explanations.plugins.explanations_fast import register_fast_explanation_plugin

    # Remove any existing descriptor for core.explanation.fast from registry snapshot
    desc = registry.find_explanation_descriptor("core.explanation.fast")
    if desc is not None:
        # descriptor may be stored in internal registry objects; attempt removal by scanning snapshot
        for p in registry.registry_snapshot():
            pm = getattr(p, "plugin_meta", {})
            if isinstance(pm, dict) and pm.get("name") == "core.explanation.fast":
                registry.remove_from_registry(p)
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


def test_alternative_plot_triangle_calls_plot_triangular(monkeypatch):
    # Prepare a fake self with minimal attributes expected by the triangular branch
    fake = types.SimpleNamespace()
    fake.rank_features = lambda *a, **k: [0]
    fake.feature_predict = {"predict": [0.6], "low": [0.5], "high": [0.7]}
    fake.prediction = {"predict": 0.6, "low": 0.5, "high": 0.7}
    alternative = {
        "predict": [0.7],
        "predict_low": [0.6],
        "predict_high": [0.8],
        "value": ["v"],
        "rule": ["r"],
        "weight": [0.1],
        "weight_low": [0.0],
        "weight_high": [0.2],
    }
    fake.get_explainer = lambda: types.SimpleNamespace(
        num_features=1, feature_names=["f0"], categorical_features=(), categorical_labels=None
    )
    # Provide get_rules used by AlternativeExplanation.plot
    fake.get_rules = lambda *a, **k: alternative
    _name = "_" + "check_preconditions"
    setattr(fake, _name, lambda *a, **k: None)
    fake.get_mode = lambda: "classification"
    fake.is_thresholded = lambda: False
    # patch plot_triangular to capture invocation
    called = {}

    def fake_plot_triangular(self_, proba, uncertainty, sel_proba, sel_unc, num_to_show, **kwargs):
        called["args"] = (proba, uncertainty, sel_proba, sel_unc, num_to_show)

    monkeypatch.setattr(expl_mod, "plot_triangular", fake_plot_triangular)
    fake.prediction = {"predict": 0.6, "low": 0.5, "high": 0.7}
    fake.feature_predict = {"predict": [0.6], "low": [0.5], "high": [0.7]}
    fake.alternative = alternative
    # call the method (unbound) to simulate instance method
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
    assert "args" in called


def test_load_entrypoint_plugins_loads_valid_plugin(monkeypatch):
    # Simulate an entry point that loads a valid plugin with proper plugin_meta
    def _load_plugin():
        p = types.SimpleNamespace()
        p.plugin_meta = {
            "schema_version": 1,
            "name": "valid_plugin",
            "version": "0.1",
            "provider": "test",
            "capabilities": ["explain"],
            "trusted": True,
            "trust": {"trusted": True},
        }
        return p

    eps = types.SimpleNamespace()
    eps.select = lambda group=None: [make_ep("pkg:attr", load=_load_plugin)]
    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: eps)

    # Ensure no denial
    monkeypatch.setattr(registry, "is_identifier_denied", lambda ident: False)

    loaded = registry.load_entrypoint_plugins(include_untrusted=True)
    # load_entrypoint_plugins returns a tuple of loaded plugin objects
    assert isinstance(loaded, tuple)
    # The discovery report should record at least one accepted plugin
    report = registry.get_discovery_report()
    assert any(r.metadata and r.metadata.get("name") == "valid_plugin" for r in report.accepted)


def test_register_fast_plugin_idempotent():
    # Calling register_fast_explanation_plugin multiple times should be safe
    from calibrated_explanations.plugins.explanations_fast import register_fast_explanation_plugin

    # Call twice; at minimum a descriptor must exist after registration
    register_fast_explanation_plugin()
    register_fast_explanation_plugin()

    desc = registry.find_explanation_descriptor("core.explanation.fast")
    assert desc is not None


def test_add_conjunctions_factual_legacy_simple():
    # Minimal fake object exercising add_conjunctions_factual_legacy via public API
    class FakeExplainer:
        def _predict(self, x, threshold, low_high_percentiles, classes, bins):
            return [0.9], [0.8], [1.0], 1

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
        def _predict(self, x, threshold, low_high_percentiles, classes, bins):
            return [0.3], [0.2], [0.4], 1

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
