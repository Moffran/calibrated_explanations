from __future__ import annotations

import importlib

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


def test_parallel_shim_behavior_is_functional() -> None:
    backend = JoblibBackend()
    assert backend.map(lambda x: x + 1, [1, 2, 3], workers=1) == [2, 3, 4]
    assert sequential_map(lambda x: x * x, [1, 2, 3]) == [1, 4, 9]


def test_schema_lazy_export_and_validation_path() -> None:
    schema_mod = importlib.import_module("calibrated_explanations.schema")
    validate_payload = getattr(schema_mod, "validate_payload")
    payload = {
        "task": "regression",
        "index": 0,
        "explanation_type": "factual",
        "prediction": {"predict": 1.0, "low": 0.5, "high": 1.5},
        "rules": [
            {
                "feature": 0,
                "rule": "x > 0",
                "rule_weight": {"predict": 0.1, "low": 0.0, "high": 0.2},
                "rule_prediction": {"predict": 1.0, "low": 0.9, "high": 1.1},
            }
        ],
    }
    validate_payload(payload)


def test_viz_lazy_import_requires_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    viz_mod = importlib.import_module("calibrated_explanations.viz")
    monkeypatch.setattr(
        viz_mod,
        "_require_matplotlib",
        lambda: (_ for _ in ()).throw(ModuleNotFoundError("missing")),
    )
    with pytest.raises(ModuleNotFoundError):
        viz_mod.__getattr__("render")


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


def test_predict_monitor_tracks_calls_and_warns_on_invalid_intervals() -> None:
    monitor = PredictBridgeMonitor(BridgeStub())
    with pytest.warns(UserWarning, match="low > high"):
        monitor.predict(np.asarray([[1.0]]), mode="factual", task="regression")
    with pytest.warns(UserWarning, match="low > high"):
        monitor.predict_interval(np.asarray([[1.0]]), task="regression")
    proba = monitor.predict_proba(np.asarray([[1.0]]))
    assert proba.shape == (1, 2)
    assert monitor.calls == ("predict", "predict_interval", "predict_proba")


def test_reject_orchestrator_initialization_and_pickle_state() -> None:
    class StubExplainer:
        bins = None
        x_cal = np.asarray([[1.0, 2.0], [2.0, 3.0]])
        y_cal = np.asarray([0, 1])
        mode = "classification"
        interval_learner = type(
            "IntervalLearner",
            (),
            {
                "predict_proba": staticmethod(
                    lambda x, bins=None: np.tile(np.asarray([0.4, 0.6]), (len(x), 1))
                )
            },
        )()

        @staticmethod
        def is_multiclass():
            return False

    orchestrator = RejectOrchestrator(StubExplainer())
    state = orchestrator.__getstate__()
    assert "_strategies_lock" not in state
    assert "_logger" not in state
    restored = RejectOrchestrator.__new__(RejectOrchestrator)
    restored.__setstate__(state)
    assert callable(restored.resolve_strategy(None))
