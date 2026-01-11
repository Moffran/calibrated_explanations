from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from calibrated_explanations.calibration.interval_wrappers import (
    FastIntervalCalibrator,
    is_fast_interval_collection,
)
from calibrated_explanations.core.exceptions import CalibratedError


def test_fast_interval_calibrator__should_raise_when_empty_collection():
    with pytest.raises(CalibratedError, match="cannot be empty"):
        FastIntervalCalibrator([])


def test_fast_interval_calibrator__should_delegate_to_default_calibrator():
    # Arrange
    calibrator0 = MagicMock()
    calibrator1 = MagicMock()

    calibrator1.predict_proba.return_value = ("p", "low", "high", "classes")
    calibrator1.predict_probability.return_value = ("p2", "low2", "high2", None)
    calibrator1.predict_uncertainty.return_value = "uq"
    calibrator1.is_multiclass.return_value = True
    calibrator1.is_mondrian.return_value = False
    calibrator1.compute_proba_cal.return_value = "adj"

    wrapper = FastIntervalCalibrator([calibrator0, calibrator1])

    x = np.zeros((2, 3))
    y = np.zeros(2)

    # Act
    got_proba = wrapper.predict_proba(x, some_kw=1)
    got_prob = wrapper.predict_probability(x)
    got_uq = wrapper.predict_uncertainty(x)
    got_multiclass = wrapper.is_multiclass()
    got_mondrian = wrapper.is_mondrian()
    wrapper.pre_fit_for_probabilistic(x, y)
    got_adj = wrapper.compute_proba_cal(x, y, weights=None)
    wrapper.insert_calibration(x, y, warm_start=True)

    # Assert
    assert got_proba == ("p", "low", "high", "classes")
    assert got_prob == ("p2", "low2", "high2", None)
    assert got_uq == "uq"
    assert got_multiclass is True
    assert got_mondrian is False
    assert got_adj == "adj"

    calibrator1.predict_proba.assert_called_once_with(x, some_kw=1)
    calibrator1.predict_probability.assert_called_once_with(x)
    calibrator1.predict_uncertainty.assert_called_once_with(x)
    calibrator1.is_multiclass.assert_called_once_with()
    calibrator1.is_mondrian.assert_called_once_with()
    calibrator1.pre_fit_for_probabilistic.assert_called_once_with(x, y)
    calibrator1.compute_proba_cal.assert_called_once_with(x, y, weights=None)
    calibrator1.insert_calibration.assert_called_once_with(x, y, warm_start=True)


def test_fast_interval_calibrator__should_behave_like_sequence():
    calibrator0 = object()
    calibrator1 = object()
    wrapper = FastIntervalCalibrator([calibrator0, calibrator1])

    assert len(wrapper) == 2
    assert wrapper[0] is calibrator0
    assert list(wrapper) == [calibrator0, calibrator1]
    assert wrapper.calibrators == (calibrator0, calibrator1)


def test_is_fast_interval_collection__should_return_true_for_supported_types():
    assert is_fast_interval_collection([]) is True
    assert is_fast_interval_collection(()) is True
    assert is_fast_interval_collection(FastIntervalCalibrator([object()])) is True
    assert is_fast_interval_collection(object()) is False
