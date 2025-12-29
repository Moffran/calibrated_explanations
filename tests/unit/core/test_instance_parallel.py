"""Tests for instance-level parallel explanations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest
from calibrated_explanations.core.explain.parallel_instance import InstanceParallelExplainExecutor
import calibrated_explanations.core.explain.parallel_instance as parallel_instance_mod
from calibrated_explanations.perf import ParallelConfig, ParallelExecutor


def make_dataset_helper() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    x_cal = rng.randn(20, 3)
    y_cal = (x_cal[:, 0] + x_cal[:, 1] > 0).astype(int)
    x_test = rng.randn(4, 3)
    return x_cal, y_cal, x_test


def test_instance_parallel_matches_sequential_output() -> None:
    x_cal, y_cal, x_test = make_dataset_helper()
    learner = LogisticRegression(random_state=0, solver="liblinear")
    learner.fit(x_cal, y_cal)

    baseline = CalibratedExplainer(learner, x_cal, y_cal, mode="classification")
    baseline.set_discretizer(None)
    baseline_result = baseline.explain_factual(x_test, _use_plugin=False)

    parallel_executor = ParallelExecutor(
        ParallelConfig(
            enabled=True,
            strategy="sequential",
            min_batch_size=2,
            granularity="instance",
        )
    )

    parallel = CalibratedExplainer(
        learner,
        x_cal,
        y_cal,
        mode="classification",
        perf_parallel=parallel_executor,
    )
    parallel.set_discretizer(None)
    parallel_result = parallel.explain_factual(x_test, _use_plugin=False)

    assert len(parallel_result) == len(baseline_result)

    for idx in range(len(baseline_result)):
        base_exp = baseline_result[idx]
        par_exp = parallel_result[idx]
        np.testing.assert_allclose(
            par_exp.feature_weights["predict"],
            base_exp.feature_weights["predict"],
        )
        np.testing.assert_allclose(
            par_exp.feature_predict["predict"],
            base_exp.feature_predict["predict"],
        )
        np.testing.assert_allclose(
            par_exp.prediction["predict"],
            base_exp.prediction["predict"],
        )

    assert parallel_result.explanations[0].calibrated_explanations is parallel_result
    assert parallel_result.total_explain_time is not None


def test_uses_process_like_strategy_on_windows_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instance-parallel should keep process/joblib strategies on Windows unless opted out."""
    import warnings

    class DummyExplanation:
        def __init__(self) -> None:
            self.calibrated_explanations = None
            self.index = None
            self.x_test = None

    class DummyChunkResult:
        def __init__(self) -> None:
            self.explanations = [DummyExplanation()]

    def fake_instance_parallel_task(task):  # noqa: ANN001
        start_idx = int(task[0])
        return start_idx, DummyChunkResult()

    def fake_initialize_explanation(  # noqa: ANN001
        _explainer,
        x_input,
        _low_high_percentiles,
        _threshold,
        _bins,
        _features_to_ignore_array,
    ):
        class DummyCombined:
            def __init__(self, x_values):
                self.x_test = np.asarray(x_values)
                self.start_index = 0
                self.explanations: list[DummyExplanation] = []

        return DummyCombined(x_input)

    monkeypatch.setattr(
        parallel_instance_mod, "_instance_parallel_task", fake_instance_parallel_task
    )
    monkeypatch.setattr(
        parallel_instance_mod, "initialize_explanation", fake_initialize_explanation
    )
    monkeypatch.setattr(parallel_instance_mod.os, "name", "nt", raising=False)

    from calibrated_explanations.plugins.manager import PluginManager

    class DummyExplainer:
        def __init__(self) -> None:
            self.latest_explanation = None
            self.last_explanation_mode = None
            self._plugin_manager = PluginManager(self)

        @property
        def plugin_manager(self):
            return self._plugin_manager

        @plugin_manager.setter
        def plugin_manager(self, value):
            self._plugin_manager = value

        def infer_explanation_mode(self):
            return "factual"

        def is_mondrian(self):
            return False

        @property
        def mode(self):
            return "classification"

        @property
        def num_features(self):
            return 1

        @property
        def x_cal(self):
            return np.zeros((1, 1))

        def predict(self, x, threshold=None, low_high_percentiles=None, bins=None, classes=None):
            n = x.shape[0]
            return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)

        def is_multiclass(self):
            return False

        @property
        def discretizer(self):
            return None

        @property
        def sample_percentiles(self):
            return [25, 50, 75]

        def get_calibration_summaries(self, x_cal):
            return {}, {}

        @property
        def y_cal(self):
            return np.zeros(1)

        @property
        def feature_names(self):
            return ["feature_0"]

    class DummyExecutor:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                enabled=True,
                max_workers=1,
                strategy="processes",
                instance_chunk_size=1,
            )
            self.active_strategy_name = "processes"
            self.map_called = False
            self.thread_strategy_called = False

        def map(self, func, items, **kwargs):  # noqa: ANN001,ARG002
            self.map_called = True
            return [func(item) for item in items]

        def thread_strategy(self, *args, **kwargs):  # noqa: ANN001,ARG002
            self.thread_strategy_called = True
            raise AssertionError(
                "thread_strategy should not be used when process backends are enabled by default"
            )

    executor = DummyExecutor()
    request = ExplainRequest(
        x=np.zeros((2, 1)),
        threshold=None,
        low_high_percentiles=(0.05, 0.95),
        bins=None,
        features_to_ignore=np.array([], dtype=int),
        use_plugin=False,
        skip_instance_parallel=False,
    )
    config = ExplainConfig(
        executor=executor,
        granularity="instance",
        min_instances_for_parallel=1,
        chunk_size=1,
        num_features=1,
        features_to_ignore_default=(),
        categorical_features=(),
        feature_values={},
        mode="classification",
    )

    plugin = InstanceParallelExplainExecutor()
    explainer = DummyExplainer()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = plugin.execute(request, config, explainer)

    assert executor.map_called is True
    assert executor.thread_strategy_called is False
    assert len(result.explanations) == 2
    assert not any("Instance-parallel execution on Windows" in str(w.message) for w in recorded)
