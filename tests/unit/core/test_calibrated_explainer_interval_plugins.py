from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from calibrated_explanations.core.calibrated_explainer import (
    CalibratedExplainer,
    ConfigurationError,
)
from calibrated_explanations.core.prediction import orchestrator as prediction_orchestrator_module
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator


@dataclass
class DummyDescriptor:
    plugin: Any | None
    metadata: dict[str, Any]
    trusted: bool = False


def _make_explainer(*, mode: str = "regression", bins=None) -> CalibratedExplainer:
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    explainer.mode = mode
    explainer.bins = bins
    explainer._interval_plugin_override = None
    explainer._fast_interval_plugin_override = None
    explainer._interval_plugin_fallbacks = {"default": (), "fast": ()}
    explainer._interval_preferred_identifier = {"default": None, "fast": None}
    explainer._interval_plugin_identifiers = {"default": None, "fast": None}
    explainer._interval_plugin_hints = {}
    # Initialize orchestrators so tests can call delegation methods
    explainer._prediction_orchestrator = PredictionOrchestrator(explainer)
    explainer._explanation_orchestrator = ExplanationOrchestrator(explainer)
    return explainer


def test_resolve_interval_plugin_prefers_override_instance(monkeypatch):
    monkeypatch.setattr(prediction_orchestrator_module, "ensure_builtin_plugins", lambda: None)
    explainer = _make_explainer()

    class Override:
        plugin_meta = {"name": "override-plugin"}

    override_instance = Override()
    explainer._interval_plugin_override = override_instance

    plugin, identifier = explainer._resolve_interval_plugin(fast=False)

    assert plugin is override_instance
    assert identifier == "override-plugin"


def test_resolve_interval_plugin_collects_errors_before_success(monkeypatch):
    monkeypatch.setattr(prediction_orchestrator_module, "ensure_builtin_plugins", lambda: None)
    explainer = _make_explainer()
    explainer._interval_plugin_fallbacks["default"] = ("alpha", "omega")
    explainer._interval_preferred_identifier["default"] = None

    beta_plugin_type = type("BetaPlugin", (), {})
    alpha_plugin_type = type("AlphaPlugin", (), {})
    omega_plugin_type = type("OmegaPlugin", (), {})

    descriptors = {
        "beta": DummyDescriptor(
            plugin=beta_plugin_type,
            metadata={
                "name": "beta",
                "capabilities": ("interval:regression",),
            },
            trusted=True,
        ),
        "alpha": DummyDescriptor(
            plugin=alpha_plugin_type,
            metadata={
                "name": "alpha",
                "modes": ("regression",),
                "capabilities": ("interval:regression",),
                "requires_bins": True,
            },
            trusted=True,
        ),
        "omega": DummyDescriptor(
            plugin=omega_plugin_type,
            metadata={
                "name": "omega",
                "modes": ("regression",),
                "capabilities": ("interval:regression",),
            },
            trusted=True,
        ),
    }

    def fake_find_descriptor(identifier: str):
        return descriptors.get(identifier)

    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_descriptor", fake_find_descriptor
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin", lambda identifier: None
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin_trusted", lambda identifier: None
    )

    plugin, identifier = explainer._resolve_interval_plugin(
        fast=False,
        hints=("beta", "alpha"),
    )

    assert identifier == "omega"
    assert plugin is omega_plugin_type


def test_resolve_interval_plugin_reports_aggregated_errors(monkeypatch):
    monkeypatch.setattr(prediction_orchestrator_module, "ensure_builtin_plugins", lambda: None)
    explainer = _make_explainer()
    explainer._interval_plugin_fallbacks["default"] = ("missing", "absent")

    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_descriptor", lambda identifier: None
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin", lambda identifier: None
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin_trusted", lambda identifier: None
    )

    with pytest.raises(ConfigurationError) as excinfo:
        explainer._resolve_interval_plugin(fast=False)

    message = str(excinfo.value)
    assert "missing: not registered" in message
    assert "absent: not registered" in message


def test_resolve_interval_plugin_fast_mode_requires_flag(monkeypatch):
    monkeypatch.setattr(prediction_orchestrator_module, "ensure_builtin_plugins", lambda: None)
    explainer = _make_explainer()
    explainer._interval_plugin_fallbacks["fast"] = ("fast-primary", "fast-backup")

    fast_primary_plugin_type = type("FastPrimaryPlugin", (), {})
    fast_backup_plugin_type = type("FastBackupPlugin", (), {})

    descriptors = {
        "fast-primary": DummyDescriptor(
            plugin=fast_primary_plugin_type,
            metadata={
                "name": "fast-primary",
                "modes": ("regression",),
                "capabilities": ("interval:regression",),
                "fast_compatible": False,
            },
            trusted=True,
        ),
        "fast-backup": DummyDescriptor(
            plugin=fast_backup_plugin_type,
            metadata={
                "name": "fast-backup",
                "modes": ("regression",),
                "capabilities": ("interval:regression",),
                "fast_compatible": True,
            },
            trusted=True,
        ),
    }

    def fake_find_descriptor(identifier: str):
        return descriptors.get(identifier)

    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_descriptor", fake_find_descriptor
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin", lambda identifier: None
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin_trusted", lambda identifier: None
    )

    plugin, identifier = explainer._resolve_interval_plugin(fast=True)

    assert identifier == "fast-backup"
    assert plugin is fast_backup_plugin_type


def test_resolve_interval_plugin_override_string_failure(monkeypatch):
    monkeypatch.setattr(prediction_orchestrator_module, "ensure_builtin_plugins", lambda: None)
    explainer = _make_explainer()
    explainer._interval_plugin_override = "preferred.interval"
    explainer._interval_plugin_fallbacks["default"] = ("preferred.interval",)

    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_descriptor", lambda identifier: None
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin", lambda identifier: None
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin_trusted", lambda identifier: None
    )

    with pytest.raises(ConfigurationError) as excinfo:
        explainer._resolve_interval_plugin(fast=False)

    assert "Interval plugin override failed" in str(excinfo.value)
    assert "preferred.interval: not registered" in str(excinfo.value)


def test_resolve_interval_plugin_aggregates_metadata_errors(monkeypatch):
    monkeypatch.setattr(prediction_orchestrator_module, "ensure_builtin_plugins", lambda: None)
    explainer = _make_explainer()
    explainer._interval_plugin_fallbacks["default"] = ("needs-bins", "missing")

    class NeedsBins:
        plugin_meta = {"name": "needs-bins"}

    descriptor = DummyDescriptor(
        plugin=NeedsBins,
        metadata={
            "schema_version": 1,
            "modes": ("regression",),
            "capabilities": ("interval:regression",),
            "requires_bins": True,
        },
        trusted=True,
    )

    monkeypatch.setattr(
        prediction_orchestrator_module,
        "find_interval_descriptor",
        lambda identifier: descriptor if identifier == "needs-bins" else None,
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin", lambda identifier: None
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin_trusted", lambda identifier: None
    )

    with pytest.raises(ConfigurationError) as excinfo:
        explainer._resolve_interval_plugin(fast=False)

    message = str(excinfo.value)
    assert "needs-bins: requires bins but explainer has none configured" in message
    assert "missing: not registered" in message
