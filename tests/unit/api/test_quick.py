import pytest

from calibrated_explanations.api.quick import quick_explain
from calibrated_explanations.core import WrapCalibratedExplainer


class DummyWrapper:
    """Lightweight wrapper stub to capture quick_explain interactions."""

    def __init__(self) -> None:
        self.fit_args = None
        self.calibrate_args = None
        self.explain_args = None

    def fit(self, x_train, y_train):
        self.fit_args = (x_train, y_train)
        return self

    def calibrate(self, x_cal, y_cal, **kwargs):
        self.calibrate_args = (x_cal, y_cal, kwargs)
        return self

    def explain_factual(self, x):
        self.explain_args = x
        return {"payload": x}


@pytest.fixture
def dummy_wrapper(monkeypatch):
    wrapper = DummyWrapper()

    def fake_from_config(cls, cfg):
        # Store the cfg on the wrapper so tests can assert forwarded fields.
        wrapper.cfg = cfg  # type: ignore[attr-defined]
        return wrapper

    # monkeypatch expects a descriptor when replacing a classmethod
    monkeypatch.setattr(
        WrapCalibratedExplainer,
        "_from_config",
        classmethod(fake_from_config),
    )
    return wrapper


def test_quick_explain_forwards_optional_config(dummy_wrapper):
    result = quick_explain(
        model="sentinel-model",
        x_train=[1, 2],
        y_train=[0, 1],
        x_cal=[3, 4],
        y_cal=[1, 0],
        x=["target"],
        threshold=0.42,
        low_high_percentiles=(10, 90),
        preprocessor="prep",
    )

    assert dummy_wrapper.cfg.model == "sentinel-model"
    assert dummy_wrapper.cfg.threshold == 0.42
    assert dummy_wrapper.cfg.low_high_percentiles == (10, 90)
    assert dummy_wrapper.cfg.preprocessor == "prep"

    # The wrapper should receive the exact inputs provided by quick_explain.
    assert dummy_wrapper.fit_args == ([1, 2], [0, 1])
    assert dummy_wrapper.calibrate_args == ([3, 4], [1, 0], {})
    assert dummy_wrapper.explain_args == ["target"]
    assert result == {"payload": ["target"]}


def test_quick_explain_infers_task_when_unspecified(dummy_wrapper):
    quick_explain(
        model="sentinel-model",
        x_train=[1],
        y_train=[0],
        x_cal=[2],
        y_cal=[1],
        x=["target"],
    )

    # When task is omitted quick_explain should let calibration infer the mode.
    assert dummy_wrapper.calibrate_args[2] == {}


def test_quick_explain_overrides_task_when_requested(dummy_wrapper):
    quick_explain(
        model="sentinel-model",
        x_train=[1],
        y_train=[0],
        x_cal=[2],
        y_cal=[1],
        x=["target"],
        task="regression",
    )

    assert dummy_wrapper.calibrate_args[2] == {"mode": "regression"}
