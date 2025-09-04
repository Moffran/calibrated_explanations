from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer


class SmallModel:
    def __init__(self) -> None:
        self.fitted_ = False

    def fit(self, X, y, **kwargs):
        self.fitted_ = True

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):  # classification path
        return np.tile([0.3, 0.7], (len(X), 1))


def _prepare_wrapper():
    m = SmallModel()
    w = WrapCalibratedExplainer(m)
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1])
    w.fit(X, y)
    # Minimal calibration stub using real calibrate with empty kwargs
    w.calibrate(X, y)
    return w, X


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_warn_on_aliases_emits_deprecation_once_predict():
    w, X = _prepare_wrapper()
    with pytest.warns(DeprecationWarning):
        # use alias 'n_jobs' to trigger warning mapping to parallel_workers
        w.predict(X, n_jobs=2)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_warn_on_aliases_emits_deprecation_once_predict_proba():
    w, X = _prepare_wrapper()
    with pytest.warns(DeprecationWarning):
        w.predict_proba(X, n_jobs=2)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_warn_on_aliases_emits_deprecation_once_explain_factual():
    w, X = _prepare_wrapper()
    with pytest.warns(DeprecationWarning):
        w.explain_factual(X, alpha=0.1)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_warn_on_aliases_emits_deprecation_once_explore_alternatives():
    w, X = _prepare_wrapper()
    with pytest.warns(DeprecationWarning):
        w.explore_alternatives(X, alphas=(0.05, 0.95))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_warn_on_aliases_emits_deprecation_once_explain_fast():
    w, X = _prepare_wrapper()
    with pytest.warns(DeprecationWarning):
        w.explain_fast(X, n_jobs=1)
