import numpy as np

from calibrated_explanations.core.explain import helpers as helpers
from calibrated_explanations.core.explain import sequential, parallel_instance
from calibrated_explanations.core.explain import feature_task as feature_task_module


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
            self.instance_chunk_size = None
            self.feature_chunk_size = None

    class Exec:
        def __init__(self):
            self.config = Config()

        def map(self, func, items, work_items=None, chunksize=None):
            return [func(it) for it in items]

    return Exec()


def test_sequential_plugin_execute_minimal(monkeypatch):
    # Monkeypatch explain_predict_step and initialize_explanation to return minimal data
    def fake_explain_predict_step(
        explainer,
        x,
        threshold,
        low_high_percentiles,
        bins,
        features_to_ignore,
        *,
        feature_filter_per_instance_ignore=None,
    ):
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
        perturbed_threshold = None
        perturbed_bins = None
        perturbed_x = np.empty((0, x.shape[1]))
        perturbed_class = np.empty((0,), dtype=int)
        return (
            predict,
            low,
            high,
            prediction,
            perturbed_feature,
            rule_boundaries,
            lesser_values,
            greater_values,
            covered_values,
            x_cal,
            perturbed_threshold,
            perturbed_bins,
            perturbed_x,
            perturbed_class,
        )

    class SimpleExplanation:
        def __init__(self, x):
            self.x_test = x
            self.explanations = []

        def finalize(self, *args, **kwargs):
            return self

    def fake_initialize_explanation(
        explainer, x, low_high_percentiles, threshold, bins, features_to_ignore
    ):
        return SimpleExplanation(x)

    monkeypatch.setattr(helpers, "explain_predict_step", fake_explain_predict_step)
    monkeypatch.setattr(helpers, "initialize_explanation", fake_initialize_explanation)
    # Also patch module-level references imported by plugins
    monkeypatch.setattr(sequential, "explain_predict_step", fake_explain_predict_step)
    monkeypatch.setattr(sequential, "initialize_explanation", fake_initialize_explanation)

    # Create request/config
    from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest

    req = ExplainRequest(
        x=np.zeros((1, 1)),
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=np.array([], dtype=int),
    )
    cfg = ExplainConfig(
        executor=make_executor(),
        num_features=1,
        categorical_features=(),
        feature_values={0: np.array([])},
    )

    explainer = type(
        "E",
        (),
        {
            "get_calibration_summaries": lambda self, x: ({}, {}),
            "infer_explanation_mode": lambda self: "factual",
            "is_mondrian": property(lambda self: False),
            "mode": "factual",
            "_merge_feature_result": lambda self, *a, **k: helpers.merge_feature_result(*a, **k),
        },
    )()

    plugin = sequential.SequentialExplainExecutor()
    out = plugin.execute(req, cfg, explainer)
    # expect an explanation object
    assert hasattr(out, "explanations")


def test_instance_parallel_plugin_empty_input(monkeypatch):
    # Test empty input path
    from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest

    req = ExplainRequest(
        x=np.zeros((0, 1)),
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=np.array([], dtype=int),
    )
    cfg = ExplainConfig(
        executor=make_executor(),
        num_features=1,
        categorical_features=(),
        feature_values={0: np.array([])},
    )

    def fake_empty_initialize(
        explainer, x, low_high_percentiles, threshold, bins, features_to_ignore
    ):
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
            "get_calibration_summaries": lambda self, x: ({}, {}),
            "infer_explanation_mode": lambda self: "factual",
            "is_mondrian": property(lambda self: False),
            "mode": "factual",
            "_merge_feature_result": lambda self, *a, **k: helpers.merge_feature_result(*a, **k),
        },
    )()

    plugin = parallel_instance.InstanceParallelExplainExecutor()
    result = plugin.execute(req, cfg, explainer)
    assert hasattr(result, "explanations")


def test_feature_parallel_supports_and_execute(monkeypatch):
    # minimal smoke test for supports and execute
    from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest

    execr = make_executor()
    cfg = ExplainConfig(
        executor=execr,
        num_features=1,
        categorical_features=(),
        feature_values={0: np.array([])},
        granularity="instance",
        min_instances_for_parallel=1,
    )
    req = ExplainRequest(
        x=np.zeros((1, 1)),
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=np.array([], dtype=int),
    )

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
        perturbed_threshold = None
        perturbed_bins = None
        perturbed_x = np.empty((0, args[1].shape[1]))
        perturbed_class = np.empty((0,), dtype=int)
        return (
            predict,
            low,
            high,
            prediction,
            perturbed_feature,
            rule_boundaries,
            lesser_values,
            greater_values,
            covered_values,
            x_cal,
            perturbed_threshold,
            perturbed_bins,
            perturbed_x,
            perturbed_class,
        )

    class SimpleExplanation:
        def __init__(self, x):
            self.x_test = x
            self.explanations = []

        def finalize(self, *args, **kwargs):
            return self

        monkeypatch.setattr(helpers, "explain_predict_step", fake_explain_predict_step)
        monkeypatch.setattr(sequential, "explain_predict_step", fake_explain_predict_step)
        monkeypatch.setattr(
            helpers, "initialize_explanation", lambda *a, **k: SimpleExplanation(a[1])
        )
        monkeypatch.setattr(
            sequential, "initialize_explanation", lambda *a, **k: SimpleExplanation(a[1])
        )
        # Also patch module-level references imported by plugins

    monkeypatch.setattr(
        parallel_instance, "initialize_explanation", lambda *a, **k: SimpleExplanation(a[1])
    )

    # simple explainer stub
    explainer = type(
        "E",
        (),
        {
            "get_calibration_summaries": lambda self, x: ({}, {}),
            "infer_explanation_mode": lambda self: "factual",
            "is_mondrian": property(lambda self: False),
            "mode": "factual",
            "_merge_feature_result": lambda self, *a, **k: helpers.merge_feature_result(*a, **k),
            "num_features": 1,
            "x_cal": np.zeros((1, 1)),
            "_predict": lambda self, *a, **k: (
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1, dtype=int),
            ),
            "is_multiclass": lambda self: False,
            "discretizer": None,
            "sample_percentiles": [25, 50, 75],
        },
    )()

    plugin = parallel_instance.InstanceParallelExplainExecutor()
    assert plugin.supports(req, cfg)
    out = plugin.execute(req, cfg, explainer)
    assert hasattr(out, "explanations")


def test_sequential_and_feature_parallel_equivalence(monkeypatch):
    """Regression test: sequential and feature-parallel should produce identical
    finalized explanation payloads when given the same inputs and deterministic
    per-feature task behaviour.
    """
    from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest

    n_instances = 3
    num_features = 4

    # Fake explain_predict_step to produce deterministic, minimal structures
    def fake_explain_predict_step(*args, **kwargs):
        x = args[1]
        n = x.shape[0]
        predict = np.zeros((n,))
        low = np.zeros((n,))
        high = np.zeros((n,))
        prediction = {"predict": np.zeros(n), "low": np.zeros(n), "high": np.zeros(n)}
        perturbed_feature = np.empty((0, 4))
        rule_boundaries = np.zeros((n, num_features, 2))
        lesser_values = {}
        greater_values = {}
        covered_values = {}
        x_cal = np.zeros((n, num_features))
        perturbed_threshold = None
        perturbed_bins = None
        perturbed_x = np.empty((0, num_features))
        perturbed_class = np.empty((0,), dtype=int)
        return (
            predict,
            low,
            high,
            prediction,
            perturbed_feature,
            rule_boundaries,
            lesser_values,
            greater_values,
            covered_values,
            x_cal,
            perturbed_threshold,
            perturbed_bins,
            perturbed_x,
            perturbed_class,
        )

    class SimpleExplanation:
        def __init__(self, x):
            self.x_test = x
            self.explanations = []
            # placeholders set by finalize
            self.binned_predict = None
            self.feature_weights = None
            self.feature_predict = None
            self.prediction = None

        def finalize(self, binned_predict, feature_weights, feature_predict, prediction, **kwargs):
            self.binned_predict = binned_predict
            self.feature_weights = feature_weights
            self.feature_predict = feature_predict
            self.prediction = prediction
            return self

    monkeypatch.setattr(helpers, "explain_predict_step", fake_explain_predict_step)
    monkeypatch.setattr(helpers, "initialize_explanation", lambda *a, **k: SimpleExplanation(a[1]))
    monkeypatch.setattr(sequential, "explain_predict_step", fake_explain_predict_step)
    monkeypatch.setattr(
        sequential, "initialize_explanation", lambda *a, **k: SimpleExplanation(a[1])
    )
    monkeypatch.setattr(
        parallel_instance, "initialize_explanation", lambda *a, **k: SimpleExplanation(a[1])
    )

    # Mock the internal feature_task to produce deterministic per-feature tuples
    def fake_feature_task(task):
        f = int(task[0])
        n = int(len(task[1]))
        feature_predict_values = np.full(n, 0.1 * (f + 1))
        low_vals = feature_predict_values - 0.01
        high_vals = feature_predict_values + 0.01
        feature_weights_predict = np.full(n, 0.5 + 0.1 * f)
        feature_weights_low = np.full(n, 0.2 + 0.05 * f)
        feature_weights_high = np.full(n, 0.6 + 0.05 * f)
        rule_values_entries = [None] * n
        # binned_entries per instance: (predict_arr, low_arr, high_arr, current_bin, counts, fractions)
        binned_entries = [
            (
                np.array([feature_predict_values[i]]),
                np.array([low_vals[i]]),
                np.array([high_vals[i]]),
                -1,
                np.array([1.0]),
                np.array([1.0]),
            )
            for i in range(n)
        ]
        lower_update = np.zeros(n)
        upper_update = np.ones(n)

        return (
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

    monkeypatch.setattr(feature_task_module, "feature_task", fake_feature_task)

    # Setup request/config and a simple explainer stub
    req = ExplainRequest(
        x=np.zeros((n_instances, num_features)),
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=np.array([], dtype=int),
    )
    cfg_seq = ExplainConfig(
        executor=None,
        num_features=num_features,
        categorical_features=(),
        feature_values={i: np.array([]) for i in range(num_features)},
    )
    cfg_par = ExplainConfig(
        executor=make_executor(),
        num_features=num_features,
        categorical_features=(),
        feature_values={i: np.array([]) for i in range(num_features)},
    )

    explainer = type(
        "E",
        (),
        {
            "get_calibration_summaries": lambda self, x: ({}, {}),
            "infer_explanation_mode": lambda self: "factual",
            "is_mondrian": property(lambda self: False),
            "mode": "factual",
        },
    )()

    seq_plugin = sequential.SequentialExplainExecutor()
    par_plugin = parallel_instance.InstanceParallelExplainExecutor()

    out_seq = seq_plugin.execute(req, cfg_seq, explainer)
    out_par = par_plugin.execute(req, cfg_par, explainer)

    # Compare the aggregated per-instance per-feature predict matrices and weights
    for i in range(n_instances):
        seq_pred = np.asarray(out_seq.feature_predict["predict"][i])
        par_pred = np.asarray(out_par.feature_predict["predict"][i])
        assert np.allclose(seq_pred, par_pred)

        seq_w = np.asarray(out_seq.feature_weights["predict"][i])
        par_w = np.asarray(out_par.feature_weights["predict"][i])
        assert np.allclose(seq_w, par_w)


def test_build_feature_tasks_minimal():
    """Focused unit test for build_feature_tasks with minimal inputs."""
    from calibrated_explanations.core.explain._shared import (
        build_feature_tasks,
        ExplainConfig,
    )

    # Minimal explainer stub
    explainer = type(
        "E",
        (),
        {"get_calibration_summaries": lambda self, x: ({}, {})},
    )()

    n_instances = 2
    num_features = 3
    x_input = np.zeros((n_instances, num_features))
    perturbed_feature = np.empty((0, 4))
    x_cal = np.zeros((n_instances, num_features))
    features_to_ignore_array = np.array([], dtype=int)
    config = ExplainConfig(
        num_features=num_features,
        categorical_features=(),
        feature_values={i: np.array([]) for i in range(num_features)},
    )
    rule_boundaries = np.zeros((n_instances, num_features, 2))
    lesser_values = {}
    greater_values = {}
    covered_values = {}

    predict = np.zeros((n_instances,))
    low = np.zeros((n_instances,))
    high = np.zeros((n_instances,))
    baseline_predict = np.zeros((n_instances,))

    tasks = build_feature_tasks(
        explainer,
        x_input,
        perturbed_feature,
        x_cal,
        features_to_ignore_array,
        config,
        rule_boundaries,
        lesser_values,
        greater_values,
        covered_values,
        predict,
        low,
        high,
        baseline_predict,
    )

    # Expect one task per feature
    assert len(tasks) == num_features
    # Each task first element is feature index and second is x_column
    for idx, task in enumerate(tasks):
        assert int(task[0]) == idx
        x_col = task[1]
        assert x_col.shape == (n_instances,)


def test_finalize_explanation_aggregation():
    """Focused unit test for finalize_explanation to ensure buffers aggregate correctly."""
    from calibrated_explanations.core.explain._shared import finalize_explanation

    n_instances = 2
    num_features = 2

    # Prepare buffers
    weights_predict = np.array([[0.1, 0.2], [0.3, 0.4]])
    weights_low = weights_predict * 0.5
    weights_high = weights_predict * 1.5
    predict_matrix = np.array([[0.11, 0.21], [0.31, 0.41]])
    low_matrix = predict_matrix - 0.01
    high_matrix = predict_matrix + 0.01

    # rule_values and instance_binned expected shapes
    rule_values = [{0: 0.1}, {1: 0.2}]
    instance_binned = [
        {
            "predict": {0: np.array([0.11])},
            "low": {},
            "high": {},
            "current_bin": {},
            "counts": {},
            "fractions": {},
        },
        {
            "predict": {1: np.array([0.31])},
            "low": {},
            "high": {},
            "current_bin": {},
            "counts": {},
            "fractions": {},
        },
    ]

    rule_boundaries = np.zeros((n_instances, num_features, 2))
    prediction = {"predict": np.zeros(n_instances)}

    # Simple explanation object that records finalize inputs
    class SimpleExplanation:
        def __init__(self):
            self.finalized = None

        def finalize(self, binned_predict, feature_weights, feature_predict, prediction, **kwargs):
            self.finalized = {
                "binned_predict": binned_predict,
                "feature_weights": feature_weights,
                "feature_predict": feature_predict,
                "prediction": prediction,
            }
            return self

    explanation = SimpleExplanation()

    # explainer stub with inference method
    explainer = type(
        "E",
        (),
        {
            "infer_explanation_mode": lambda self: "factual",
        },
    )()

    out = finalize_explanation(
        explanation,
        weights_predict,
        weights_low,
        weights_high,
        predict_matrix,
        low_matrix,
        high_matrix,
        rule_values,
        instance_binned,
        rule_boundaries,
        prediction,
        instance_start_time=0.0,
        total_start_time=0.0,
        explainer=explainer,
    )

    # validate returned explanation and contents
    assert isinstance(out, SimpleExplanation)
    assert out.finalized is not None
    assert "feature_weights" in out.finalized
    # feature_weights['predict'] should be list of arrays equal to rows of weights_predict
    fw = out.finalized["feature_weights"]["predict"]
    assert np.allclose(fw[0], weights_predict[0])
    assert np.allclose(fw[1], weights_predict[1])


def test_finalize_explanation_attaches_per_instance_ignore():
    """finalize_explanation should propagate per-instance ignore masks from explainer."""
    from calibrated_explanations.core.explain._shared import finalize_explanation

    n_instances = 2
    num_features = 2

    weights_predict = np.zeros((n_instances, num_features))
    weights_low = np.zeros_like(weights_predict)
    weights_high = np.zeros_like(weights_predict)
    predict_matrix = np.zeros_like(weights_predict)
    low_matrix = np.zeros_like(weights_predict)
    high_matrix = np.zeros_like(weights_predict)

    rule_values = [{} for _ in range(n_instances)]
    instance_binned = [
        {"predict": {}, "low": {}, "high": {}, "current_bin": {}, "counts": {}, "fractions": {}},
        {"predict": {}, "low": {}, "high": {}, "current_bin": {}, "counts": {}, "fractions": {}},
    ]
    rule_boundaries = np.zeros((n_instances, num_features, 2))
    prediction = {"predict": np.zeros(n_instances)}

    class SimpleExplanationWithIgnore:
        def __init__(self):
            self.finalized = None

        def finalize(self, binned_predict, feature_weights, feature_predict, prediction, **kwargs):
            self.finalized = {
                "binned_predict": binned_predict,
                "feature_weights": feature_weights,
                "feature_predict": feature_predict,
                "prediction": prediction,
                "kwargs": kwargs,
            }
            return self

    explanation = SimpleExplanationWithIgnore()

    # explainer stub with feature-filter state
    per_instance_ignore = [np.array([0], dtype=int), np.array([1], dtype=int)]

    class ExplainerWithFilterState:
        def __init__(self):
            self.feature_filter_per_instance_ignore = per_instance_ignore

        def infer_explanation_mode(self):
            return "factual"

    explainer = ExplainerWithFilterState()

    out = finalize_explanation(
        explanation,
        weights_predict,
        weights_low,
        weights_high,
        predict_matrix,
        low_matrix,
        high_matrix,
        rule_values,
        instance_binned,
        rule_boundaries,
        prediction,
        instance_start_time=0.0,
        total_start_time=0.0,
        explainer=explainer,
    )

    assert hasattr(out, "feature_filter_per_instance_ignore")
    assert out.feature_filter_per_instance_ignore == per_instance_ignore
