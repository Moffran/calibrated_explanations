import pytest

import calibrated_explanations as ce
from tests.helpers.deprecation import warns_or_raises, deprecations_error_enabled


def test_interval_regressor_lazy_import(monkeypatch):
    # Ensure IntervalRegressor isn't cached before access.
    monkeypatch.delitem(ce.__dict__, "IntervalRegressor", raising=False)

    if deprecations_error_enabled():
        with pytest.raises(DeprecationWarning):
            _ = ce.IntervalRegressor
        assert "IntervalRegressor" not in ce.__dict__
    else:
        with warns_or_raises(match="IntervalRegressor.*deprecated"):
            interval_regressor = ce.IntervalRegressor

        from calibrated_explanations.calibration import IntervalRegressor as IntervalRegressorImpl

        assert interval_regressor is IntervalRegressorImpl
        assert ce.__dict__["IntervalRegressor"] is IntervalRegressorImpl


def test_venn_abers_lazy_import(monkeypatch):
    monkeypatch.delitem(ce.__dict__, "VennAbers", raising=False)

    if deprecations_error_enabled():
        with pytest.raises(DeprecationWarning):
            _ = ce.VennAbers
        assert "VennAbers" not in ce.__dict__
    else:
        with warns_or_raises(match="VennAbers.*deprecated"):
            venn_abers = ce.VennAbers

        from calibrated_explanations.calibration import VennAbers as VennAbersImpl

        assert venn_abers is VennAbersImpl
        assert ce.__dict__["VennAbers"] is VennAbersImpl


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(ce, "NotARealAttribute")
