"""Tests for lightweight helpers in :mod:`calibrated_explainer`."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from calibrated_explanations.plugins.registry import EXPLANATION_PROTOCOL_VERSION
from calibrated_explanations.core.config_helpers import (
    coerce_string_tuple as _coerce_string_tuple,
    read_pyproject_section as _read_pyproject_section,
    split_csv as _split_csv,
)
from calibrated_explanations.plugins.predict_monitor import PredictBridgeMonitor


class _DummyBridge:
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


def test_read_pyproject_section(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
        [tool.calibrated_explanations.explanations]
        factual = "py.identifier"
        factual_fallbacks = ["fb.one", "", "fb.two"]
        """.strip()
    )
    monkeypatch.chdir(tmp_path)

    result = _read_pyproject_section(("tool", "calibrated_explanations", "explanations"))

    assert result == {
        "factual": "py.identifier",
        "factual_fallbacks": ["fb.one", "", "fb.two"],
    }


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


def test_predict_bridge_monitor_tracks_usage():
    """Test that PredictBridgeMonitor correctly tracks bridge method calls."""
    bridge = _DummyBridge()
    monitor = PredictBridgeMonitor(bridge)

    predict_result = monitor.predict(np.array([[1.0]]), mode="factual", task="classification")
    interval_result = monitor.predict_interval(np.array([[1.0]]), task="classification")
    proba_result = monitor.predict_proba(np.array([[1.0]]))

    assert monitor.calls == ("predict", "predict_interval", "predict_proba")
    assert monitor.used is True
    # Ensure the wrapped bridge is called transparently.
    assert predict_result[0] is bridge.predictions["predict"]
    assert interval_result[0] is bridge.predictions["predict_interval"]
    assert proba_result[0] is bridge.predictions["predict_proba"]


def test_check_explanation_runtime_metadata_validations(explainer_factory):
    explainer = explainer_factory()

    assert (
        explainer._check_explanation_runtime_metadata(None, identifier="plugin", mode="factual")
        == "plugin: plugin metadata unavailable"
    )

    incompatible = {
        "schema_version": "0",
        "tasks": ["classification"],
        "modes": ["factual"],
        "capabilities": {"requires_predict_proba": False},
    }
    assert "unsupported" in explainer._check_explanation_runtime_metadata(
        incompatible, identifier="plugin", mode="factual"
    )

    missing_tasks = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "modes": ["factual"],
        "capabilities": {"requires_predict_proba": False},
    }
    assert "missing tasks" in explainer._check_explanation_runtime_metadata(
        missing_tasks, identifier="plugin", mode="factual"
    )

    wrong_mode = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "tasks": ["classification"],
        "modes": ["alternative"],
        "capabilities": {"requires_predict_proba": False},
    }
    assert "does not declare mode" in explainer._check_explanation_runtime_metadata(
        wrong_mode, identifier="plugin", mode="factual"
    )


def test_check_explanation_runtime_metadata_capabilities(explainer_factory):
    explainer = explainer_factory()
    metadata = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "tasks": ["classification"],
        "modes": ["factual"],
        "capabilities": ["explain", "mode:factual"],
    }
    message = explainer._check_explanation_runtime_metadata(
        metadata, identifier="plugin", mode="factual"
    )
    assert "missing required capabilities" in message


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


def test_check_interval_runtime_metadata_validations(explainer_factory):
    explainer = explainer_factory()

    assert (
        explainer._check_interval_runtime_metadata(None, identifier="interval", fast=False)
        == "interval: interval metadata unavailable"
    )

    incompatible = {
        "schema_version": 2,
    }
    assert "unsupported" in explainer._check_interval_runtime_metadata(
        incompatible, identifier="interval", fast=False
    )

    missing_modes = {
        "schema_version": 1,
        "capabilities": ["interval:classification"],
    }
    assert "missing modes" in explainer._check_interval_runtime_metadata(
        missing_modes, identifier="interval", fast=False
    )

    missing_cap = {
        "schema_version": 1,
        "modes": ["classification"],
        "capabilities": ["interval:regression"],
    }
    assert "missing capability" in explainer._check_interval_runtime_metadata(
        missing_cap, identifier="interval", fast=False
    )

    not_fast = {
        "schema_version": 1,
        "modes": ["classification"],
        "capabilities": ["interval:classification"],
        "fast_compatible": False,
    }
    assert "not marked fast_compatible" in explainer._check_interval_runtime_metadata(
        not_fast, identifier="interval", fast=True
    )

    requires_bins = {
        "schema_version": 1,
        "modes": ["classification"],
        "capabilities": ["interval:classification"],
        "fast_compatible": True,
        "requires_bins": True,
    }
    assert "requires bins" in explainer._check_interval_runtime_metadata(
        requires_bins, identifier="interval", fast=True
    )
