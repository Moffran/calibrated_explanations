"""Tests for lightweight helpers in :mod:`calibrated_explainer`."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from calibrated_explanations.plugins import EXPLANATION_PROTOCOL_VERSION
from calibrated_explanations.core.config_helpers import (
    coerce_string_tuple as _coerce_string_tuple,
    read_pyproject_section as _read_pyproject_section,
    split_csv as _split_csv,
)
from calibrated_explanations.plugins.predict_monitor import PredictBridgeMonitor


class DummyBridge:
    """Simple bridge to validate monitoring behaviour."""

    def __init__(self) -> None:
        self.predictions: Dict[str, Any] = {
            "predict": np.array([0.1, 0.9]),
            "predict_interval": (np.array([0.2]), np.array([0.8])),
            "predict_proba": np.array([[0.2, 0.8]]),
        }

    def predict(self, x: Any, *, mode: str, task: str, bins: Any | None = None) -> Any:
        return self.predictions["predict"], mode, task, bins

    def predict_interval(self, x: Any, *, task: str, bins: Any | None = None) -> Any:
        return self.predictions["predict_interval"], task, bins

    def predict_proba(self, x: Any, bins: Any | None = None) -> Any:
        return self.predictions["predict_proba"], bins





@pytest.mark.parametrize(
    "value, expected",
    [
        (None, ()),
        ("", ()),
        ("a, b ,c", ("a", "b", "c")),
    ],
)
def test_split_csv(value, expected):
    assert _split_csv(value) == expected


def test_coerce_string_tuple_handles_iterables():
    assert _coerce_string_tuple("value") == ("value",)
    assert _coerce_string_tuple(["x", "", "y", 1, None]) == ("x", "y")








def test_instantiate_plugin_prefers_fresh_instances(explainer_factory):
    explainer = explainer_factory()

    class Proto:
        plugin_meta = {}

        def __call__(self):
            return self

    proto = Proto()
    assert explainer._instantiate_plugin(None) is None
    assert explainer._instantiate_plugin(proto) is proto

    class RequiresArgs:
        def __init__(self, value):
            self.value = value

    original = RequiresArgs(3)
    cloned = explainer._instantiate_plugin(original)
    assert isinstance(cloned, RequiresArgs)
    assert cloned is not original
    assert cloned.value == 3


