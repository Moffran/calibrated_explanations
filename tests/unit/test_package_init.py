import pytest

import calibrated_explanations as ce


def test_interval_regressor_lazy_import(monkeypatch):
    # Ensure IntervalRegressor isn't cached before access.
    monkeypatch.delitem(ce.__dict__, "IntervalRegressor", raising=False)

    with pytest.warns(DeprecationWarning, match="IntervalRegressor.*deprecated"):
        interval_regressor = ce.IntervalRegressor

    from calibrated_explanations.calibration.interval_regressor import (
        IntervalRegressor as IntervalRegressorImpl,
    )

    assert interval_regressor is IntervalRegressorImpl
    assert ce.__dict__["IntervalRegressor"] is IntervalRegressorImpl


def test_venn_abers_lazy_import(monkeypatch):
    monkeypatch.delitem(ce.__dict__, "VennAbers", raising=False)

    with pytest.warns(DeprecationWarning, match="VennAbers.*deprecated"):
        venn_abers = ce.VennAbers

    from calibrated_explanations.calibration.venn_abers import VennAbers as VennAbersImpl

    assert venn_abers is VennAbersImpl
    assert ce.__dict__["VennAbers"] is VennAbersImpl


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(ce, "NotARealAttribute")
