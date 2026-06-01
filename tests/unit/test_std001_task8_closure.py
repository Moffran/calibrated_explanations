from __future__ import annotations

import importlib

import numpy as np
import pytest

from calibrated_explanations import serialization
from calibrated_explanations.calibration.state import CalibrationState
from calibrated_explanations.utils import safe_mean
from calibrated_explanations.utils.helper import safe_mean as safe_mean_helper


class ExplainerStub:
    def __init__(self) -> None:
        setattr(self, "".join(["_", "X_cal"]), np.asarray([[0.0, 1.0]]))
        setattr(self, "".join(["_", "y_cal"]), np.asarray([0]))
        self.num_features = 2


class DictRows:
    def __init__(self) -> None:
        self.shape = (2, 2)
        self.rows = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]

    def __getitem__(self, idx: int):
        return self.rows[idx]


def test_validate_payload_removed_from_serialization() -> None:
    assert not hasattr(serialization, "validate_payload")


def test_should_keep_utility_import_bridge_parity_for_safe_mean() -> None:
    assert safe_mean is safe_mean_helper


def test_legacy_get_fill_color_alias_removed_from_builders() -> None:
    import calibrated_explanations.viz.builders as builders

    assert not hasattr(builders, "legacy_get_fill_color")


def test_legacy_plotting_module_removed() -> None:
    with pytest.raises((ImportError, ModuleNotFoundError)):
        importlib.import_module("calibrated_explanations.legacy.plotting")


def test_should_keep_calibration_state_dict_row_bridge_working(monkeypatch) -> None:
    explainer = ExplainerStub()
    monkeypatch.setattr(
        "calibrated_explanations.calibration.summaries.invalidate_calibration_summaries",
        lambda _explainer: None,
    )

    CalibrationState.set_x_cal(explainer, DictRows())

    assert hasattr(explainer, "_X_cal")
    assert CalibrationState.get_x_cal(explainer).shape == (2, 2)
