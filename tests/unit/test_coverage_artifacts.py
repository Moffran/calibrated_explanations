from __future__ import annotations

import importlib
import builtins

import numpy as np
import pytest

from calibrated_explanations.calibration.state import CalibrationState
from calibrated_explanations.core.explain.sequential import SequentialExplainExecutor
from calibrated_explanations.core.reject.orchestrator import RejectOrchestrator
from calibrated_explanations.core.test import JoblibBackend, sequential_map
from calibrated_explanations.plugins.predict_monitor import PredictBridgeMonitor


class ExplainerStub:
    def __init__(self) -> None:
        setattr(self, "".join(["_", "X_cal"]), np.asarray([[0.0, 1.0]]))
        setattr(self, "".join(["_", "y_cal"]), np.asarray([0]))
        self.num_features = 2


class DictRows:
    def __init__(self) -> None:
        self.shape = (2, 2)
        self.rows = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]

    def __getitem__(self, idx):
        return self.rows[idx]

    def __iter__(self):
        return iter(self.rows)


class BridgeStub:
    def predict(self, x, *, mode, task, bins=None):
        _ = (x, mode, task, bins)
        return {
            "predict": np.asarray([0.4]),
            "low": np.asarray([0.8]),
            "high": np.asarray([0.5]),
        }

    def predict_interval(self, x, *, task, bins=None):
        _ = (x, task, bins)
        return (np.asarray([0.2]), np.asarray([0.4]), np.asarray([0.3]))

    def predict_proba(self, x, bins=None):
        _ = (x, bins)
        return np.asarray([[0.1, 0.9]])






def test_calibration_state_dict_rows_append_and_getters(monkeypatch: pytest.MonkeyPatch) -> None:
    explainer = ExplainerStub()
    monkeypatch.setattr(
        "calibrated_explanations.calibration.summaries.invalidate_calibration_summaries",
        lambda _e: None,
    )

    CalibrationState.set_x_cal(explainer, DictRows())
    x_cal = CalibrationState.get_x_cal(explainer)
    assert isinstance(x_cal, np.ndarray)
    assert x_cal.shape == (2, 2)

    CalibrationState.set_y_cal(explainer, np.asarray([[1], [0]]))
    assert CalibrationState.get_y_cal(explainer).shape == (2,)

    CalibrationState.append_calibration(explainer, np.asarray([[5.0, 6.0]]), np.asarray([1]))
    assert CalibrationState.get_x_cal(explainer).shape[0] == 3
    assert CalibrationState.get_y_cal(explainer).shape[0] == 3


def test_sequential_executor_identity_contract() -> None:
    executor = SequentialExplainExecutor()
    assert executor.name == "sequential"
    assert executor.priority == 10
    assert executor.supports(request=object(), config=object()) is True


def test_joblib_backend_falls_back_when_joblib_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = JoblibBackend()
    original_import = builtins.__import__

    def fail_joblib(name, *args, **kwargs):
        if name == "joblib":
            raise ImportError("joblib missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_joblib)

    result = backend.map(lambda x: x + 1, [1, 2, 3], workers=2)
    assert result == [2, 3, 4]


def test_joblib_backend_falls_back_when_joblib_submodule_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = JoblibBackend()
    original_import = builtins.__import__

    def fail_any_joblib(name, *args, **kwargs):
        if name.startswith("joblib"):
            raise ImportError("joblib unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_any_joblib)

    result = backend.map(lambda x: x * 2, [2, 3], workers=1)
    assert result == [4, 6]




