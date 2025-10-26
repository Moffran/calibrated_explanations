import pytest

import calibrated_explanations as ce


def test_interval_regressor_lazy_import(monkeypatch):
    # Ensure IntervalRegressor isn't cached before access.
    monkeypatch.delitem(ce.__dict__, "IntervalRegressor", raising=False)

    IntervalRegressor = ce.IntervalRegressor

    from calibrated_explanations.core.interval_regressor import IntervalRegressor as Impl

    assert IntervalRegressor is Impl
    assert ce.__dict__["IntervalRegressor"] is Impl


def test_venn_abers_lazy_import(monkeypatch):
    monkeypatch.delitem(ce.__dict__, "VennAbers", raising=False)

    VennAbers = ce.VennAbers

    from calibrated_explanations.core.venn_abers import VennAbers as Impl

    assert VennAbers is Impl
    assert ce.__dict__["VennAbers"] is Impl


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(ce, "NotARealAttribute")


