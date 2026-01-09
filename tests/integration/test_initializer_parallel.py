import numpy as np

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


def test_initializer_based_pool_warmup():
    # Minimal dummy learner for regression
    class DummyLearner:
        def fit(self, x, y):
            self.fitted_ = True
            return self

        def predict(self, x):
            return np.zeros(len(x))

    learner = DummyLearner()
    x_cal = np.zeros((5, 2))
    y_cal = np.zeros(5)

    learner.fit(x_cal, y_cal)
    explainer = CalibratedExplainer(learner, x_cal, y_cal, mode="regression")

    # Initialize pool with warm-up initializer; this should create a ParallelExecutor
    explainer.initialize_pool(n_workers=1, pool_at_init=True)

    assert getattr(explainer, "perf_parallel", None) is not None
    # Close resources
    explainer.close()
