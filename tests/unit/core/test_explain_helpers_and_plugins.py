import numpy as np

from calibrated_explanations.core.explain import _computation as comp
from calibrated_explanations.core.explain import _helpers as helpers
from calibrated_explanations.core.explain import sequential, parallel_feature, parallel_instance


def test_assign_weight_scalar_variants():
    # scalar
    assert np.isclose(comp.assign_weight_scalar(0.1, 0.6), 0.5)

    # array differences -> return first element
    inst = np.array([0.2, 0.3])
    pred = np.array([0.7, 0.8])
    assert np.isclose(comp.assign_weight_scalar(inst, pred), 0.5)


def test_merge_ignore_features_union():
    class DummyExplainer:
        def __init__(self):
            self.features_to_ignore = [1, 3]

    expl = DummyExplainer()
    out = helpers.merge_ignore_features(expl, [2, 3, 4])
    assert set(out.tolist()) == {1, 2, 3, 4}


def test_merge_feature_result_updates_buffers():
    # Prepare small buffers
    n = 2
    f = 1
    weights_predict = np.zeros((n, 3))
    weights_low = np.zeros((n, 3))
    weights_high = np.zeros((n, 3))
    predict_matrix = np.zeros((n, 3))
    low_matrix = np.zeros((n, 3))
    high_matrix = np.zeros((n, 3))
    rule_values = [{}, {}]
    instance_binned = [
        {"predict": {}, "low": {}, "high": {}, "current_bin": {}, "counts": {}, "fractions": {}},
        {"predict": {}, "low": {}, "high": {}, "current_bin": {}, "counts": {}, "fractions": {}},
    ]
    # Create a fake feature result for feature index 1
    feature_predict_values = np.array([0.1, 0.2])
    low_vals = np.array([0.05, 0.15])
    high_vals = np.array([0.2, 0.25])
    feature_weights_predict = np.array([0.9, 0.8])
    feature_weights_low = np.array([0.4, 0.3])
    feature_weights_high = np.array([1.0, 1.1])
    rule_values_entries = [None, ([], 1.0, 1.0)]
    binned_entries = [
        (np.array([0.0]), np.array([0.0]), np.array([0.0]), -1, np.array([0.0]), np.array([0.0])),
        (feature_predict_values, low_vals, high_vals, 0, np.array([1.0]), np.array([1.0])),
    ]
    lower_update = np.array([0.0, 0.0])
    upper_update = np.array([1.0, 1.0])

    result = (
        f,
        feature_weights_predict,
        feature_weights_low,
        feature_weights_high,
        feature_predict_values,
        low_vals,
        high_vals,
        rule_values_entries,
        binned_entries,
        lower_update,
        upper_update,
    )

    rule_boundaries = np.zeros((n, 3, 2))

    helpers.merge_feature_result(
        result,
        weights_predict,
        weights_low,
        weights_high,
        predict_matrix,
        low_matrix,
        high_matrix,
        rule_values,
        instance_binned,
        rule_boundaries,
    )

    # Check that buffers updated
    assert np.allclose(weights_predict[:, f], feature_weights_predict)
    assert np.allclose(predict_matrix[:, f], feature_predict_values)
    assert rule_boundaries[0, f, 0] == 0.0


def make_executor(enabled=True, min_batch_size=1):
    class Config:
        def __init__(self):
            self.enabled = enabled
            self.min_batch_size = min_batch_size

    class Exec:
        def __init__(self):
            self.config = Config()

        def map(self, func, items, work_items=None):
            return [func(it) for it in items]

    return Exec()


def test_sequential_plugin_execute_minimal(monkeypatch):
    # Monkeypatch explain_predict_step and initialize_explanation to return minimal data
    def fake_explain_predict_step(explainer, x, threshold, low_high_percentiles, bins, features_to_ignore):
        n = x.shape[0]
        # predict, low, high arrays sized for the n instances (no perturbed entries)
        predict = np.zeros((n,))
        low = np.zeros((n,))
        high = np.zeros((n,))
        prediction = {"predict": np.zeros(n), "classes": np.zeros(n, dtype=int)}
        perturbed_feature = np.empty((0, 4))
        rule_boundaries = np.zeros((n, 1, 2))
        lesser_values = {}
        greater_values = {}
        covered_values = {}
        x_cal = np.zeros((n, 1))
        return (predict, low, high, prediction, perturbed_feature, rule_boundaries, lesser_values, greater_values, covered_values, x_cal)

    class SimpleExplanation:
        def __init__(self, x):
            self.x_test = x
            self.explanations = []

        def finalize(self, *args, **kwargs):
            return self

    def fake_initialize_explanation(explainer, x, low_high_percentiles, threshold, bins, features_to_ignore):
        return SimpleExplanation(x)

    monkeypatch.setattr(helpers, "explain_predict_step", fake_explain_predict_step)
    monkeypatch.setattr(helpers, "initialize_explanation", fake_initialize_explanation)
    # Also patch module-level references imported by plugins
    monkeypatch.setattr(sequential, "explain_predict_step", fake_explain_predict_step)
    monkeypatch.setattr(sequential, "initialize_explanation", fake_initialize_explanation)

    # Create request/config
    from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest

    req = ExplainRequest(x=np.zeros((1, 1)), threshold=None, low_high_percentiles=(5, 95), bins=None, features_to_ignore=np.array([], dtype=int))
    cfg = ExplainConfig(executor=make_executor(), num_features=1, categorical_features=(), feature_values={0: np.array([])})

    explainer = type(
        "E",
        (),
        {
            "_get_calibration_summaries": lambda self, x: ({}, {}),
            "_infer_explanation_mode": lambda self: "factual",
            "_is_mondrian": lambda self: False,
            "mode": "factual",
            "_merge_feature_result": lambda self, *a, **k: helpers.merge_feature_result(*a, **k),
        },
    )()

    plugin = sequential.SequentialExplainPlugin()
    out = plugin.execute(req, cfg, explainer)
    # expect an explanation object
    assert hasattr(out, "explanations")


def test_instance_parallel_plugin_empty_input(monkeypatch):
    # Test empty input path
    from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest

    req = ExplainRequest(x=np.zeros((0, 1)), threshold=None, low_high_percentiles=(5, 95), bins=None, features_to_ignore=np.array([], dtype=int))
    cfg = ExplainConfig(executor=make_executor(), num_features=1, categorical_features=(), feature_values={0: np.array([])})

    def fake_empty_initialize(explainer, x, low_high_percentiles, threshold, bins, features_to_ignore):
        class EmptyExplanation:
            def __init__(self):
                self.explanations = []

        return EmptyExplanation()

    # Patch the module-level initializer used by the plugin
    monkeypatch.setattr(parallel_instance, "initialize_explanation", fake_empty_initialize)

    explainer = type(
        "E",
        (),
        {
            "_get_calibration_summaries": lambda self, x: ({}, {}),
            "_infer_explanation_mode": lambda self: "factual",
            "_is_mondrian": lambda self: False,
            "mode": "factual",
            "_merge_feature_result": lambda self, *a, **k: helpers.merge_feature_result(*a, **k),
        },
    )()

    plugin = parallel_instance.InstanceParallelExplainPlugin()
    result = plugin.execute(req, cfg, explainer)
    assert hasattr(result, "explanations")


def test_feature_parallel_supports_and_execute(monkeypatch):
    # minimal smoke test for supports and execute
    from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest

    execr = make_executor()
    cfg = ExplainConfig(executor=execr, num_features=1, categorical_features=(), feature_values={0: np.array([])})
    req = ExplainRequest(x=np.zeros((1, 1)), threshold=None, low_high_percentiles=(5, 95), bins=None, features_to_ignore=np.array([], dtype=int))

    # monkeypatch helpers similar to sequential test
    def fake_explain_predict_step(*args, **kwargs):
        n = args[1].shape[0]
        predict = np.zeros((n,))
        low = np.zeros((n,))
        high = np.zeros((n,))
        prediction = {"predict": np.zeros(n), "classes": np.zeros(n, dtype=int)}
        perturbed_feature = np.empty((0, 4))
        rule_boundaries = np.zeros((n, 1, 2))
        lesser_values = {}
        greater_values = {}
        covered_values = {}
        x_cal = np.zeros((n, 1))
        return (predict, low, high, prediction, perturbed_feature, rule_boundaries, lesser_values, greater_values, covered_values, x_cal)

    class SimpleExplanation:
        def __init__(self, x):
            self.x_test = x
            self.explanations = []

        def finalize(self, *args, **kwargs):
            return self

    monkeypatch.setattr(helpers, "explain_predict_step", fake_explain_predict_step)
    monkeypatch.setattr(helpers, "initialize_explanation", lambda *a, **k: SimpleExplanation(a[1]))
    # Also patch module-level references imported by plugins
    monkeypatch.setattr(parallel_feature, "explain_predict_step", fake_explain_predict_step)
    monkeypatch.setattr(parallel_feature, "initialize_explanation", lambda *a, **k: SimpleExplanation(a[1]))

    # simple explainer stub
    explainer = type(
        "E",
        (),
        {
            "_get_calibration_summaries": lambda self, x: ({}, {}),
            "_infer_explanation_mode": lambda self: "factual",
            "_is_mondrian": lambda self: False,
            "mode": "factual",
            "_merge_feature_result": lambda self, *a, **k: helpers.merge_feature_result(*a, **k),
        },
    )()

    plugin = parallel_feature.FeatureParallelExplainPlugin()
    assert plugin.supports(req, cfg)
    out = plugin.execute(req, cfg, explainer)
    assert hasattr(out, "explanations")
