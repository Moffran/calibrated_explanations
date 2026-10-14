"""CE_CACHE on the plain constructor path (#220, ADR-003).

``CE_CACHE`` provisions a calibrator cache for ``WrapCalibratedExplainer`` and
``CalibratedExplainer`` without the builder, mirroring ``CE_PARALLEL``. An
explicit ``perf_cache=`` keeps precedence, and the cache stays off when
``CE_CACHE`` is unset.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder
from calibrated_explanations.cache import CacheConfig, CalibratorCache
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import ConfigurationError


@pytest.fixture
def learner_and_data():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((30, 2))
    y = (x[:, 0] > 0).astype(int)
    learner = LogisticRegression(solver="liblinear")
    learner.fit(x, y)
    return learner, x, y


def _calibrated_wrapper(learner, x, y, **calibrate_kwargs):
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x, y)
    wrapper.calibrate(x, y, **calibrate_kwargs)
    return wrapper


def test_wrapper_should_provision_cache_when_ce_cache_enables_it(learner_and_data, monkeypatch):
    learner, x, y = learner_and_data
    monkeypatch.setenv("CE_CACHE", "enable,max_items=7,namespace=svc")

    wrapper = _calibrated_wrapper(learner, x, y)

    cache = wrapper.explainer.perf_cache
    assert isinstance(cache, CalibratorCache)
    assert cache.enabled is True
    assert cache.config.max_items == 7
    assert cache.config.namespace == "svc"


def test_wrapper_env_cache_should_serve_repeated_predictions(learner_and_data, monkeypatch):
    learner, x, y = learner_and_data
    monkeypatch.setenv("CE_CACHE", "on")
    wrapper = _calibrated_wrapper(learner, x, y)

    first = wrapper.predict(x[:3], uq_interval=True)
    second = wrapper.predict(x[:3], uq_interval=True)

    metrics = wrapper.explainer.perf_cache.metrics
    assert metrics.hits >= 1
    np.testing.assert_array_equal(first[0], second[0])


def test_calibrated_explainer_should_provision_cache_when_ce_cache_enables_it(
    learner_and_data, monkeypatch
):
    learner, x, y = learner_and_data
    monkeypatch.setenv("CE_CACHE", "1")

    explainer = CalibratedExplainer(learner, x, y)

    assert isinstance(explainer.perf_cache, CalibratorCache)
    assert explainer.perf_cache.enabled is True


@pytest.mark.parametrize("env_value", [None, "off"])
def test_cache_should_stay_off_when_ce_cache_is_unset_or_disabled(
    learner_and_data, monkeypatch, env_value
):
    learner, x, y = learner_and_data
    if env_value is None:
        monkeypatch.delenv("CE_CACHE", raising=False)
    else:
        monkeypatch.setenv("CE_CACHE", env_value)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapper = _calibrated_wrapper(learner, x, y)
        explainer = CalibratedExplainer(learner, x, y)

    assert not [w for w in caught if "CE_CACHE" in str(w.message)]
    assert wrapper.explainer.perf_cache is None
    assert explainer.perf_cache is None


def test_explicit_perf_cache_should_take_precedence_over_ce_cache(learner_and_data, monkeypatch):
    learner, x, y = learner_and_data
    monkeypatch.setenv("CE_CACHE", "enable,max_items=7")
    explicit = CalibratorCache(CacheConfig(enabled=True, max_items=3))

    wrapper = _calibrated_wrapper(learner, x, y, perf_cache=explicit)
    explainer = CalibratedExplainer(learner, x, y, perf_cache=explicit)

    assert wrapper.explainer.perf_cache is explicit
    assert explainer.perf_cache is explicit


def test_builder_cache_should_not_be_replaced_on_constructor_path(learner_and_data, monkeypatch):
    learner, x, y = learner_and_data
    monkeypatch.setenv("CE_CACHE", "off")
    config = ExplainerBuilder(learner).perf_cache(True).build_config()

    wrapper = WrapCalibratedExplainer.from_config(config)
    wrapper.fit(x, y)
    wrapper.calibrate(x, y)

    assert wrapper.explainer.perf_cache is wrapper.perf_cache
    assert wrapper.explainer.perf_cache.enabled is False


def test_wrapper_should_fail_closed_when_ce_cache_is_malformed(learner_and_data, monkeypatch):
    learner, x, y = learner_and_data
    monkeypatch.setenv("CE_CACHE", "enable,max_items=many")
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x, y)

    with pytest.raises(ConfigurationError, match="max_items=many") as excinfo:
        wrapper.calibrate(x, y)

    assert excinfo.value.details["env_var"] == "CE_CACHE"
