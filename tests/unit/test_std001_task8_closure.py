from __future__ import annotations

import numpy as np

from calibrated_explanations import serialization
from calibrated_explanations.calibration.state import CalibrationState
from calibrated_explanations.utils import safe_mean
from calibrated_explanations.utils.helper import safe_mean as safe_mean_helper
from calibrated_explanations.viz.builders import legacy_get_fill_color


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


def test_should_delegate_serialization_validate_payload_to_schema_validator(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_schema_validate(payload):
        calls.append(payload)

    monkeypatch.setattr(serialization, "_schema_validate_payload", _fake_schema_validate)
    payload = {"schema_version": "1.0.0"}

    serialization.validate_payload(payload)

    assert calls == [payload]


def test_should_keep_utility_import_bridge_parity_for_safe_mean() -> None:
    assert safe_mean is safe_mean_helper


def test_should_keep_builders_legacy_color_alias_parity() -> None:
    assert legacy_get_fill_color(0.7, 0.9) == "#e8594d"
    assert legacy_get_fill_color(1.0, 1.0) == "#ff0000"


def test_should_keep_calibration_state_dict_row_bridge_working(monkeypatch) -> None:
    explainer = ExplainerStub()
    monkeypatch.setattr(
        "calibrated_explanations.calibration.summaries.invalidate_calibration_summaries",
        lambda _explainer: None,
    )

    CalibrationState.set_x_cal(explainer, DictRows())

    assert hasattr(explainer, "_CalibratedExplainer__X_cal")
    assert CalibrationState.get_x_cal(explainer).shape == (2, 2)
