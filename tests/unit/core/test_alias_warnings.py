from __future__ import annotations

import warnings
import numpy as np
import pytest

from calibrated_explanations.api.params import ALIAS_MAP, warn_on_aliases
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer


def test_warn_on_aliases_emits_deprecation():
    # pick an alias from the map
    alias = next(iter(ALIAS_MAP.keys()))
    kwargs = {alias: 123}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_aliases(kwargs)
        assert any(issubclass(item.category, DeprecationWarning) for item in w)


class SmallModel:
    def __init__(self) -> None:
        self.fitted_ = False

    def fit(self, x, y, **kwargs):
        self.fitted_ = True

    def predict(self, x):
        return np.zeros(len(x))

    def predict_proba(self, x):  # classification path
        return np.tile([0.3, 0.7], (len(x), 1))


def _prepare_wrapper():
    m = SmallModel()
    w = WrapCalibratedExplainer(m)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1])
    w.fit(x, y)
    # Minimal calibration stub using real calibrate with empty kwargs
    w.calibrate(x, y)
    return w, x


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_warn_on_aliases_emits_deprecation_once_predict():
    w, x = _prepare_wrapper()
    with pytest.warns(DeprecationWarning):
        # use alias 'n_jobs' to trigger warning mapping to parallel_workers
        w.predict(x, n_jobs=2)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_warn_on_aliases_emits_deprecation_once_predict_proba():
    w, x = _prepare_wrapper()
    with pytest.warns(DeprecationWarning):
        w.predict_proba(x, n_jobs=2)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_warn_on_aliases_emits_deprecation_once_explain_factual():
    w, x = _prepare_wrapper()
    with pytest.warns(DeprecationWarning):
        w.explain_factual(x, alpha=0.1)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_warn_on_aliases_emits_deprecation_once_explore_alternatives():
    w, x = _prepare_wrapper()
    with pytest.warns(DeprecationWarning):
        w.explore_alternatives(x, alphas=(0.05, 0.95))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_warn_on_aliases_emits_deprecation_once_explain_fast():
    w, x = _prepare_wrapper()
    with pytest.warns(DeprecationWarning):
        w.explain_fast(x, n_jobs=1)
