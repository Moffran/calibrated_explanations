"""Tests for lightweight helpers in :mod:`calibrated_explainer`."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


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


def test_instantiate_plugin_prefers_fresh_instances(explainer_factory):
    explainer = explainer_factory()

    class Proto:
        plugin_meta = {}

        def __call__(self):
            return self

    proto = Proto()
    assert explainer.instantiate_plugin(None) is None
    assert explainer.instantiate_plugin(proto) is proto

    class RequiresArgs:
        def __init__(self, value):
            self.value = value

    original = RequiresArgs(3)
    cloned = explainer.instantiate_plugin(original)
    assert isinstance(cloned, RequiresArgs)
    assert cloned is not original
    assert cloned.value == 3
