import numpy as np

from calibrated_explanations.core.explain.parallel_instance import (
    _instance_parallel_task,
    InstanceParallelExplainExecutor,
)
from calibrated_explanations.core.explain.parallel_runtime import worker_init_from_explainer_spec
from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest


class DummyExecutor:
    def __init__(self, cfg):
        self.config = cfg

    def map(self, fn, items, *, work_items=None, chunksize=None):
        # Capture tasks and return a simple simulated result for each
        self.captured = list(items)
        # Simulate worker results: return (start_idx, sentinel)
        return [(t[0], {"ok": True}) for t in items]


def make_request(x, low_high_percentiles=(5, 95)):
    return ExplainRequest(
        x=np.asarray(x),
        threshold=None,
        low_high_percentiles=low_high_percentiles,
        bins=None,
        features_to_ignore=np.array([]),
        features_to_ignore_per_instance=None,
        use_plugin=False,
        skip_instance_parallel=False,
    )


def test_tasks_omit_explainer_when_worker_initializer_present(monkeypatch):
    # Build an executor-like object with a worker_initializer set
    from calibrated_explanations.parallel.parallel import ParallelConfig

    cfg = ParallelConfig(enabled=True)
    cfg.worker_initializer = lambda spec: None

    dummy_exec = DummyExecutor(cfg)

    # Prepare a small explainer-like object with minimal attributes used
    class FakeExplainer:
        num_features = 3
        mode = "classification"

        def _infer_explanation_mode(self):
            return "factual"

    explainer = FakeExplainer()

    plugin = InstanceParallelExplainExecutor()

    # Use multiple instances so partitioning creates several tasks
    request = make_request(np.zeros((10, 3)))
    config = ExplainConfig(
        executor=dummy_exec,
        granularity="instance",
        min_instances_for_parallel=1,
        chunk_size=4,
        num_features=3,
        features_to_ignore_default=(),
        categorical_features=(),
        feature_values={},
        mode="classification",
    )

    # Monkeypatch initialize_explanation in the parallel_instance module to avoid
    # heavy explainer dependencies during this unit test.
    import calibrated_explanations.core.explain.parallel_instance as pi_mod

    class SimpleCombined:
        def __init__(self, x):
            self.explanations = []
            self.x_test = x
            self.start_index = 0
            self.current_index = 0
            self.end_index = 0
            self.total_explain_time = 0.0

    monkeypatch.setattr(
        pi_mod, "initialize_explanation", lambda explainer, x, a, b, c, d: SimpleCombined(x)
    )

    # Monkeypatch the executor.map to capture tasks
    class ChunkResult:
        def __init__(self, n=1):
            class ExplanationObj:
                def __init__(self):
                    self.calibrated_explanations = None
                    self.index = 0
                    self.x_test = None

            self.explanations = [ExplanationObj() for _ in range(n)]

    def fake_map(fn, items, *, work_items=None, chunksize=None):
        fake_map.captured = list(items)
        return [(t[0], ChunkResult()) for t in items]

    dummy_exec.map = fake_map

    plugin.execute(request, config, explainer)

    captured = getattr(fake_map, "captured", [])
    assert captured, "Expected tasks to be submitted"
    for _, _, state in captured:
        assert (
            "explainer" not in state
        ), "Explainer should not be shipped when worker_initializer is present"


def test_worker_harness_used_when_present():
    # Install a worker harness in-process
    spec = {"dummy": True}
    worker_init_from_explainer_spec(spec)

    # Create a compact task payload (no explainer)
    state = {"config_state": {}}
    start, stop = 0, 5
    res = _instance_parallel_task((start, stop, state))

    assert res[0] == start
    # Harness returns a dict with 'spec' key as implemented
    assert isinstance(res[1], dict) and res[1].get("spec") == spec


def test_fallback_runs_with_explainer_when_no_harness(monkeypatch):
    # Ensure no harness installed
    import calibrated_explanations.core.explain.parallel_runtime as pr_mod

    if hasattr(pr_mod, "_worker_harness"):
        delattr(pr_mod, "_worker_harness")

    # Create serialized_state including a dummy explainer and config_state
    class FakeExplainer:
        def __init__(self):
            self.latest_explanation = None

        def _infer_explanation_mode(self):
            return "factual"

    fake_expl = FakeExplainer()

    # Monkeypatch SequentialExplainExecutor.execute to return sentinel
    from calibrated_explanations.core.explain.parallel_instance import SequentialExplainExecutor

    monkeypatch.setattr(
        SequentialExplainExecutor, "execute", lambda self, r, c, e: {"from": "sequential"}
    )

    state = {
        "subset": np.zeros((2, 1)),
        "threshold_slice": None,
        "bins_slice": None,
        "low_high_percentiles": (5, 95),
        "features_to_ignore_array": np.array([]),
        "features_to_ignore_per_instance": None,
        "explainer": fake_expl,
        "config_state": {},
    }

    res = _instance_parallel_task((0, 2, state))
    assert res[0] == 0
    assert isinstance(res[1], dict) and res[1].get("from") == "sequential"
