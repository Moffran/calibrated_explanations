from __future__ import annotations

from calibrated_explanations._VennAbers import VennAbers
from calibrated_explanations.core import calibration_helpers as ch
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
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


def _make_explainer(binary_dataset, **overrides):
    from tests._helpers import get_classification_model

    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset

    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    explainer = CalibratedExplainer(
        model,
        X_cal,
        y_cal,
        mode="classification",
        feature_names=feature_names,
        categorical_features=categorical_features,
        class_labels=["No", "Yes"],
        seed=42,
        **overrides,
    )
    return explainer, X_test


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
