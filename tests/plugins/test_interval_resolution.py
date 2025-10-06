from __future__ import annotations

import pytest

from calibrated_explanations._venn_abers import VennAbers
from calibrated_explanations.core import calibration_helpers as ch
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.exceptions import ConfigurationError
from calibrated_explanations.plugins.intervals import IntervalCalibratorPlugin
from calibrated_explanations.plugins.registry import (
    clear_interval_plugins,
    ensure_builtin_plugins,
    register_interval_plugin,
)


class UntrustedIntervalPlugin(IntervalCalibratorPlugin):
    plugin_meta = {
        "name": "tests.interval.untrusted",
        "schema_version": 1,
        "version": "0.0-test",
        "provider": "tests",
        "capabilities": ["interval:classification"],
        "modes": ("classification",),
        "dependencies": (),
        "trusted": False,
        "trust": {"trusted": False},
        "fast_compatible": False,
        "requires_bins": False,
        "confidence_source": "tests",
        "legacy_compatible": False,
    }

    def create(self, context, *, fast: bool = False):  # pragma: no cover - defensive
        raise AssertionError("Untrusted interval plugin should not be used")


class RecordingIntervalPlugin(IntervalCalibratorPlugin):
    plugin_meta = {
        "name": "tests.interval.recording",
        "schema_version": 1,
        "version": "0.0-test",
        "provider": "tests",
        "capabilities": ["interval:classification"],
        "modes": ("classification",),
        "dependencies": (),
        "trusted": True,
        "trust": {"trusted": True},
        "fast_compatible": False,
        "requires_bins": False,
        "confidence_source": "tests",
        "legacy_compatible": True,
    }

    last_context = None
    last_calibrator = None

    def create(self, context, *, fast: bool = False):
        calibrator = object()
        type(self).last_context = context
        type(self).last_calibrator = calibrator
        return calibrator


class RecordingFastIntervalPlugin(IntervalCalibratorPlugin):
    plugin_meta = {
        "name": "tests.interval.fast_recording",
        "schema_version": 1,
        "version": "0.0-test",
        "provider": "tests",
        "capabilities": ["interval:classification"],
        "modes": ("classification",),
        "dependencies": (),
        "trusted": True,
        "trust": {"trusted": True},
        "fast_compatible": True,
        "requires_bins": False,
        "confidence_source": "tests",
        "legacy_compatible": True,
    }

    last_context = None
    last_calibrators = ()

    def create(self, context, *, fast: bool = False):
        num_features = int(context.metadata.get("num_features", 0) or 0)
        calibrators = [object() for _ in range(num_features + 1)]
        type(self).last_context = context
        type(self).last_calibrators = tuple(calibrators)
        return calibrators


class MissingCapabilityIntervalPlugin(IntervalCalibratorPlugin):
    plugin_meta = {
        "name": "tests.interval.missing_cap",
        "schema_version": 1,
        "version": "0.0-test",
        "provider": "tests",
        "capabilities": ["interval:regression"],
        "modes": ("classification",),
        "dependencies": (),
        "trusted": True,
        "trust": {"trusted": True},
        "fast_compatible": False,
        "requires_bins": False,
        "confidence_source": "tests",
        "legacy_compatible": True,
    }

    def create(self, context, *, fast: bool = False):  # pragma: no cover - unreachable
        return object()


class SlowFastIntervalPlugin(IntervalCalibratorPlugin):
    plugin_meta = {
        "name": "tests.interval.slow_fast",
        "schema_version": 1,
        "version": "0.0-test",
        "provider": "tests",
        "capabilities": ["interval:classification"],
        "modes": ("classification",),
        "dependencies": (),
        "trusted": True,
        "trust": {"trusted": True},
        "fast_compatible": False,
        "requires_bins": False,
        "confidence_source": "tests",
        "legacy_compatible": True,
    }

    def create(self, context, *, fast: bool = False):  # pragma: no cover - defensive
        return [object()]


def _make_explainer(binary_dataset, **overrides):
    from tests._helpers import get_classification_model

    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="classification",
        feature_names=feature_names,
        categorical_features=categorical_features,
        class_labels=["No", "Yes"],
        seed=42,
        **overrides,
    )
    return explainer, x_test


def test_interval_resolution_skips_untrusted_fallback(monkeypatch, binary_dataset):
    ensure_builtin_plugins()
    descriptor = register_interval_plugin("tests.interval.untrusted", UntrustedIntervalPlugin())
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FALLBACKS", descriptor.identifier)
    try:
        explainer, _ = _make_explainer(binary_dataset)
        ch.initialize_interval_learner(explainer)
        identifier = explainer._interval_plugin_identifiers.get("default")
        assert identifier == "core.interval.legacy"
        assert isinstance(explainer.interval_learner, VennAbers)
    finally:
        monkeypatch.delenv("CE_INTERVAL_PLUGIN_FALLBACKS", raising=False)
        clear_interval_plugins()
        ensure_builtin_plugins()


def test_fast_interval_plugin_constructs_calibrators(binary_dataset):
    ensure_builtin_plugins()
    explainer, _ = _make_explainer(binary_dataset, fast=True)
    ch.initialize_interval_learner(explainer)
    calibrators = explainer.interval_learner
    assert isinstance(calibrators, list)
    assert len(calibrators) == explainer.num_features + 1
    assert all(isinstance(cal, VennAbers) for cal in calibrators)
    assert explainer._interval_plugin_identifiers.get("fast") == "core.interval.fast"


def test_interval_override_uses_untrusted_plugin(monkeypatch, binary_dataset):
    ensure_builtin_plugins()
    descriptor = register_interval_plugin(
        "tests.interval.explicit_untrusted",
        type(
            "ExplicitUntrusted",
            (IntervalCalibratorPlugin,),
            {
                "plugin_meta": {
                    "name": "tests.interval.explicit_untrusted",
                    "schema_version": 1,
                    "version": "0.0-test",
                    "provider": "tests",
                    "capabilities": ["interval:classification"],
                    "modes": ("classification",),
                    "dependencies": (),
                    "trusted": False,
                    "trust": {"trusted": False},
                    "fast_compatible": False,
                    "requires_bins": False,
                    "confidence_source": "tests",
                    "legacy_compatible": True,
                },
                "create": lambda self, context, fast=False: object(),
            },
        )(),
    )
    monkeypatch.setenv("CE_INTERVAL_PLUGIN", descriptor.identifier)
    try:
        explainer, _ = _make_explainer(binary_dataset)
        ch.initialize_interval_learner(explainer)
        assert explainer._interval_plugin_identifiers.get("default") == descriptor.identifier
    finally:
        monkeypatch.delenv("CE_INTERVAL_PLUGIN", raising=False)
        clear_interval_plugins()
        ensure_builtin_plugins()


def test_interval_metadata_captures_calibrator(monkeypatch, binary_dataset):
    ensure_builtin_plugins()
    RecordingIntervalPlugin.last_context = None
    RecordingIntervalPlugin.last_calibrator = None
    descriptor = register_interval_plugin("tests.interval.recording", RecordingIntervalPlugin())
    monkeypatch.setenv("CE_INTERVAL_PLUGIN", descriptor.identifier)
    try:
        explainer, _ = _make_explainer(binary_dataset)
        ch.initialize_interval_learner(explainer)
        assert explainer.interval_learner is RecordingIntervalPlugin.last_calibrator
        context = RecordingIntervalPlugin.last_context
        assert context is not None
        assert context.metadata.get("calibrator") is RecordingIntervalPlugin.last_calibrator
    finally:
        monkeypatch.delenv("CE_INTERVAL_PLUGIN", raising=False)
        clear_interval_plugins()
        ensure_builtin_plugins()


def test_fast_interval_metadata_captures_calibrators(monkeypatch, binary_dataset):
    ensure_builtin_plugins()
    RecordingFastIntervalPlugin.last_context = None
    RecordingFastIntervalPlugin.last_calibrators = ()
    descriptor = register_interval_plugin(
        "tests.interval.fast_recording",
        RecordingFastIntervalPlugin(),
    )
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FAST", descriptor.identifier)
    try:
        explainer, _ = _make_explainer(
            binary_dataset,
            fast=True,
        )
        ch.initialize_interval_learner(explainer)
        assert tuple(explainer.interval_learner) == RecordingFastIntervalPlugin.last_calibrators
        context = RecordingFastIntervalPlugin.last_context
        assert context is not None
        assert context.metadata.get("fast_calibrators") == RecordingFastIntervalPlugin.last_calibrators
    finally:
        monkeypatch.delenv("CE_INTERVAL_PLUGIN_FAST", raising=False)
        clear_interval_plugins()
        ensure_builtin_plugins()


def test_interval_resolution_skips_missing_capability(monkeypatch, binary_dataset):
    ensure_builtin_plugins()
    descriptor = register_interval_plugin(
        "tests.interval.missing_cap",
        MissingCapabilityIntervalPlugin(),
    )
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FALLBACKS", descriptor.identifier)
    try:
        explainer, _ = _make_explainer(binary_dataset)
        ch.initialize_interval_learner(explainer)
        identifier = explainer._interval_plugin_identifiers.get("default")
        assert identifier == "core.interval.legacy"
    finally:
        monkeypatch.delenv("CE_INTERVAL_PLUGIN_FALLBACKS", raising=False)
        clear_interval_plugins()
        ensure_builtin_plugins()


def test_fast_interval_override_requires_fast_capability(monkeypatch, binary_dataset):
    ensure_builtin_plugins()
    descriptor = register_interval_plugin(
        "tests.interval.slow_fast",
        SlowFastIntervalPlugin(),
    )
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FAST", descriptor.identifier)
    try:
        with pytest.raises(ConfigurationError):
            _make_explainer(binary_dataset, fast=True)
    finally:
        monkeypatch.delenv("CE_INTERVAL_PLUGIN_FAST", raising=False)
        clear_interval_plugins()
        ensure_builtin_plugins()
