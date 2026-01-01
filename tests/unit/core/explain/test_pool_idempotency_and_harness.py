import numpy as np
from sklearn.dummy import DummyClassifier

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


def make_explainer():
    learner = DummyClassifier(strategy="most_frequent")
    x_cal = np.arange(30).reshape(10, 3).astype(float)
    y_cal = np.concatenate([np.zeros(5), np.ones(5)])
    learner.fit(x_cal, y_cal)
    return CalibratedExplainer(learner, x_cal, y_cal, mode="classification")


def test_initialize_pool_idempotent_and_config_contains_spec():
    expl = make_explainer()

    # First initialization should create and enter the pool
    expl.initialize_pool(n_workers=1, pool_at_init=True)
    first = getattr(expl, "_perf_parallel", None)
    assert first is not None

    # Re-initializing should be a no-op and keep the same executor
    expl.initialize_pool(n_workers=1, pool_at_init=True)
    assert getattr(expl, "_perf_parallel") is first

    cfg = first.config
    # Worker initializer should be configured when pool_at_init=True
    assert getattr(cfg, "worker_initializer", None) is not None
    assert isinstance(getattr(cfg, "worker_init_args", ()), tuple)

    # The spec argument should contain learner_bytes (possibly None if not picklable)
    spec = cfg.worker_init_args[0] if cfg.worker_init_args else {}
    assert "x_cal" in spec and "y_cal" in spec

    # Closing should clean up and be idempotent
    expl.close()
    assert getattr(expl, "_perf_parallel", None) is None
    expl.close()
    assert getattr(expl, "_perf_parallel", None) is None


def test_worker_init_callable_installs_harness():
    expl = make_explainer()
    expl.initialize_pool(n_workers=1, pool_at_init=False)
    cfg = expl.parallel_executor.config
    spec = cfg.worker_init_args[0] if cfg.worker_init_args else {}

    # Call worker initializer directly in-process to validate it can install the harness
    from calibrated_explanations.core.explain.parallel_runtime import (
        worker_init_from_explainer_spec,
    )

    worker_init_from_explainer_spec(spec)

    import calibrated_explanations.core.explain.parallel_runtime as pr

    assert hasattr(pr, "_worker_harness")
    assert hasattr(pr._worker_harness, "explain_slice")

    expl.close()
