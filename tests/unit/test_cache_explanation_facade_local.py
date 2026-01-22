import numpy as np

from calibrated_explanations.cache.explanation_cache import ExplanationCacheFacade


class FakeCache:
    def __init__(self):
        self.enabled = True
        self.store = {}
        self.version = None

    def get(self, *, stage, parts):
        return self.store.get((stage, parts))

    def set(self, *, stage, parts, value):
        self.store[(stage, parts)] = value

    def compute(self, *, stage, parts, fn):
        key = (stage, parts)
        if key in self.store:
            return self.store[key]
        val = fn()
        self.store[key] = val
        return val

    def flush(self):
        self.store.clear()

    def reset_version(self, new_version: str):
        self.version = new_version


def test_explanation_cache_facade_enabled_flag():
    facade_none = ExplanationCacheFacade(None)
    assert not facade_none.enabled

    fake = FakeCache()
    facade = ExplanationCacheFacade(fake)
    assert facade.enabled


def test_calibration_summaries_get_set_compute():
    fake = FakeCache()
    facade = ExplanationCacheFacade(fake)

    explainer_id = "e1"
    x_hash = "h1"
    categorical = {0: {"a": 1}}
    numeric = {0: np.array([1.0, 2.0])}

    # initially missing
    assert facade.get_calibration_summaries(explainer_id=explainer_id, x_cal_hash=x_hash) is None

    # set and get
    facade.set_calibration_summaries(
        explainer_id=explainer_id,
        x_cal_hash=x_hash,
        categorical_counts=categorical,
        numeric_sorted=numeric,
    )
    got = facade.get_calibration_summaries(explainer_id=explainer_id, x_cal_hash=x_hash)
    assert got == (categorical, numeric)

    # compute should return cached value without calling fn again
    called = {"count": 0}

    def compute_fn():
        called["count"] += 1
        return categorical, numeric

    res1 = facade.compute_calibration_summaries(
        explainer_id=explainer_id, x_cal_hash=x_hash, compute_fn=compute_fn
    )
    assert res1 == (categorical, numeric)
    assert called["count"] == 0


def test_feature_names_cache_and_invalidate_and_reset_version():
    fake = FakeCache()
    facade = ExplanationCacheFacade(fake)
    explainer_id = "e2"
    names = ("f1", "f2")

    assert facade.get_feature_names_cache(explainer_id=explainer_id) is None
    facade.set_feature_names_cache(explainer_id=explainer_id, feature_names=names)
    assert facade.get_feature_names_cache(explainer_id=explainer_id) == names

    # also set a calibration summary for the same explainer so we can validate
    x_hash = "h2"
    facade.set_calibration_summaries(
        explainer_id=explainer_id,
        x_cal_hash=x_hash,
        categorical_counts={0: {"x": 1}},
        numeric_sorted={0: np.array([1.0])},
    )

    # invalidate clears caches and is observable via public API
    facade.invalidate_all()
    assert facade.get_feature_names_cache(explainer_id=explainer_id) is None
    assert facade.get_calibration_summaries(explainer_id=explainer_id, x_cal_hash=x_hash) is None

    # reset_version sets version on underlying cache
    facade.reset_version("v2")
    assert fake.version == "v2"
