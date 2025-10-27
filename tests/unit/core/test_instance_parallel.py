"""Tests for instance-level parallel explanations."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.perf import ParallelConfig, ParallelExecutor


def _make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    x_cal = rng.randn(20, 3)
    y_cal = (x_cal[:, 0] + x_cal[:, 1] > 0).astype(int)
    x_test = rng.randn(4, 3)
    return x_cal, y_cal, x_test


def test_instance_parallel_matches_sequential_output() -> None:
    x_cal, y_cal, x_test = _make_dataset()
    learner = LogisticRegression(random_state=0, solver="liblinear")
    learner.fit(x_cal, y_cal)

    baseline = CalibratedExplainer(learner, x_cal, y_cal, mode="classification")
    baseline.set_discretizer(None)
    baseline_result = baseline.explain(x_test, _use_plugin=False)

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
    parallel_result = parallel.explain(x_test, _use_plugin=False)

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
