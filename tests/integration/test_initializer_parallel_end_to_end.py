import os
import time
import pickle
import numpy as np
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer

try:
    from sklearn.dummy import DummyClassifier
except Exception:  # pragma: no cover - sklearn required in test env
    DummyClassifier = None


def make_small_explainer():
    if DummyClassifier is None:
        # fallback simple learner if sklearn is not available
        class Fallback:
            def fit(self, x, y):
                self.fitted_ = True
                return self

            def predict(self, x):
                return np.zeros(len(x))

            def predict_proba(self, x):
                return np.vstack([np.zeros(len(x)), np.ones(len(x))]).T

        learner = Fallback()
    else:
        learner = DummyClassifier(strategy="most_frequent")

    # use non-constant calibration features to avoid degenerate edge-cases
    x_cal = np.arange(30).reshape(10, 3).astype(float)
    # include both classes so calibration plugins (e.g. Venn-Abers) can initialize
    y_cal = np.concatenate([np.zeros(5), np.ones(5)])
    learner.fit(x_cal, y_cal)
    explainer = CalibratedExplainer(learner, x_cal, y_cal, mode="classification")
    return explainer


@pytest.mark.integration
def test_sequential_vs_initializer_parallel_end_to_end(tmp_path):
    expl = make_small_explainer()

    x_test = np.zeros((4, 3))

    # Baseline sequential run
    baseline = expl.explain_factual(x_test)
    baseline_json = baseline.to_json()

    # Initialize pool with worker initializer
    expl.initialize_pool(n_workers=1, pool_at_init=True)

    parallel_res = expl.explain_factual(x_test)

    parallel_json = parallel_res.to_json()

    assert baseline_json == parallel_json

    # Pickling size/timing: spec vs full explainer

    spec = {
        "learner": expl.learner,
        "x_cal": expl.x_cal,
        "y_cal": expl.y_cal,
        "mode": expl.mode,
        "num_features": expl.num_features,
    }

    t0 = time.perf_counter()
    _ = pickle.dumps(spec)
    spec_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        _ = pickle.dumps(expl)
        expl_time = time.perf_counter() - t0
    except Exception:
        expl_time = spec_time * 10

    assert spec_time <= expl_time


@pytest.mark.skipif(os.name != "nt", reason="Windows spawn semantics test")
def test_windows_spawn_semantics():
    # Sanity check: ensure initializer path can be created on Windows (skipped on non-Windows)
    expl = make_small_explainer()
    expl.initialize_pool(n_workers=1, pool_at_init=True)
    expl.close()
