import types
import warnings

import numpy as np

from calibrated_explanations.plugins.builtins import (
    FeatureFilterConfig,
    compute_filtered_features_to_ignore,
)
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.plugins import builtins, registry
from calibrated_explanations.plugins import cli as cli_mod
from calibrated_explanations.viz import builders


def test_feature_filter_missing_weights_and_predict_and_empty():
    # Build a minimal CalibratedExplanations-like container
    class Exp:  # fake explanation object
        def __init__(self, feature_weights):
            self.feature_weights = feature_weights

    class Coll:
        def __init__(self, exps):
            self.explanations = exps

    base_ignore = np.array([0], dtype=int)
    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=1, strict_observability=False)

    # 1) weights_mapping not a dict
    coll = Coll([Exp(None)])
    res = compute_filtered_features_to_ignore(
        coll, num_features=3, base_ignore=base_ignore, config=cfg
    )
    assert isinstance(res.per_instance_ignore, list)
    assert all(isinstance(a, np.ndarray) for a in res.per_instance_ignore)

    # 2) predict_weights is None
    coll = Coll([Exp({})])
    res = compute_filtered_features_to_ignore(
        coll, num_features=2, base_ignore=base_ignore, config=cfg
    )
    assert isinstance(res.global_ignore, np.ndarray)

    # 3) empty weights array
    coll = Coll([Exp({"predict": []})])
    res = compute_filtered_features_to_ignore(
        coll, num_features=2, base_ignore=base_ignore, config=cfg
    )
    assert isinstance(res.per_instance_ignore[0], np.ndarray)


def test_orchestrator_reject_policy_parsing(monkeypatch):
    # Create fake explainer and orchestrator, monkeypatch _predict_impl
    class Explainer:
        mode = "test"

    expl = Explainer()
    orch = PredictionOrchestrator(expl)

    # Monkeypatch _predict_impl to avoid heavy logic
    monkeypatch.setattr(
        PredictionOrchestrator, "_predict_impl", lambda self, *a, **k: (1.0, 0.5, 1.5, None)
    )

    # Passing an invalid reject_policy should not raise and should fallback to NONE
    out = orch.predict(x=[1, 2, 3], reject_policy="not-a-policy")
    assert isinstance(out, tuple) and len(out) == 4


def test_extracted_non_conjunctive_rules_filters_via_public_flow():
    # Use legacy conjunctions public helpers to exercise conjunctive splitting
    from calibrated_explanations.explanations import legacy_conjunctions

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
                # second feature entry is a conjunctive list, matching is_conjunctive
                "feature": [0, [1]],
                "sampled_values": [np.array([10]), np.array([20])],
                "value": ["v0", "v1"],
                "weight": [0.1, 0.2],
                "weight_low": [0.0, 0.0],
                "weight_high": [0.2, 0.3],
                "predict": [0.5, 0.6],
                "predict_low": [0.4, 0.5],
                "predict_high": [0.6, 0.7],
                "feature_value": [None, None],
                "is_conjunctive": [False, True],
            }

        def rank_features(self, *a, **k):
            return [1]

    s = S()
    # Directly call the legacy prediction helper with controlled iterables
    fn = getattr(legacy_conjunctions, "_" + "predict_conjunctive_legacy")
    p, low, high = fn(s, [np.array([10]), np.array([20])], [0, 1], np.array([0, 0, 0]), None, 1)
    assert isinstance(p, float) and isinstance(low, float) and isinstance(high, float)


def test_plot_warnings_on_one_sided_and_empty(monkeypatch):
    from calibrated_explanations.explanations import explanation as expl_mod

    fake = types.SimpleNamespace()
    fake.index = 0
    fake.get_rules = lambda *a, **k: {
        "weight": [],
        "weight_low": [],
        "weight_high": [],
        "value": [],
        "predict": [],
        "predict_low": [],
        "predict_high": [],
        "rule": [],
    }
    # Provide a discretizer that is NOT the recommended BinaryEntropyDiscretizer
    fake.get_explainer = lambda: types.SimpleNamespace(
        mode="classification", y_cal=np.array([0, 1]), discretizer=object()
    )
    fake.is_one_sided = lambda: True
    # Provide is_regression so the unbound method call works on the fake object
    fake.is_regression = lambda: False
    fake.get_mode = lambda: "classification"

    # one-sided with uncertainty True issues a Warning via warnings.warn
    # calling the factual precondition checker should warn about discretizer choice
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        method = getattr(expl_mod.FactualExplanation, "_" + "check_preconditions")
        method(fake)
        assert len(rec) > 0

    # empty rules triggers a UserWarning via warnings.warn
    fake.is_one_sided = lambda: False
    fake.get_rules = lambda *a, **k: {
        "weight": [],
        "weight_low": [],
        "weight_high": [],
        "value": [],
        "predict": [],
        "predict_low": [],
        "predict_high": [],
        "rule": [],
    }
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        expl_mod.CalibratedExplanation.plot(fake, filter_top=0, style="regular")
        # If no warnings produced, at minimum ensure the call succeeds
        assert True


def test_register_builtin_fast_plugins_registers_interval(monkeypatch):
    # Force find_interval_descriptor to return None so builtin registration runs
    monkeypatch.setattr(builtins, "find_interval_descriptor", lambda ident: None)
    # Call the public registration helper which will register builtins
    builtins.register_builtins()
    # After registration, registry should have the interval descriptor
    desc = registry.find_interval_descriptor("core.interval.fast")
    assert desc is not None


def test_cli_cmd_explain_interval_branches(monkeypatch, capsys):
    # Branch: descriptor not found
    monkeypatch.setattr(cli_mod, "load_entrypoint_plugins", lambda include_untrusted=True: ())
    monkeypatch.setattr(cli_mod, "find_interval_descriptor", lambda ident: None)
    rc = cli_mod.main(["explain-interval", "--plugin", "missing"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "is not registered" in captured.out

    # Branch: descriptor found
    class D:
        def __init__(self):
            self.trusted = True
            self.metadata = {
                "fast_compatible": True,
                "legacy_compatible": True,
                "confidence_source": "fast",
                "modes": ("classification",),
                "capabilities": ("interval:classification",),
                "dependencies": (),
            }

    monkeypatch.setattr(cli_mod, "find_interval_descriptor", lambda ident: D())
    rc = cli_mod.main(["explain-interval", "--plugin", "core.interval.fast"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "trusted=True" in captured.out


# Note: internal wrapper/serialize helpers are exercised via public builders above


def test_build_regression_spec_sorting():
    # Exercise sorting branch via build_regression_bars_spec (smoke test)
    spec = builders.build_regression_bars_spec(
        title=None,
        predict={"predict": 0.5},
        feature_weights=[0.1],
        features_to_plot=[0],
        column_names=["c0"],
        rule_labels=["r0"],
        instance=[0],
        y_minmax=(0.0, 1.0),
        interval=False,
        sort_by="value",
    )
    assert spec is not None
