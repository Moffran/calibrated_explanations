from __future__ import annotations

import numpy as np


def _build_dummy_model():
    class DummyModel:
        def predict_proba(self, x):
            return np.column_stack([1 - x, x])

    return DummyModel()


class HelloCalibratedPlugin:
    """Minimal example that returns calibrated outputs for a wrapped model."""

    plugin_meta = {
        "schema_version": 1,
        "name": "hello.calibrated.plugin",
        "version": "0.1.0",
        "provider": "example-team",
        "capabilities": ("binary-classification", "probabilistic-regression"),
        "dependencies": (),
        "modes": ("factual", "alternative"),
        "tasks": ("classification", "regression"),
        "trusted": False,
    }

    def supports(self, model):
        """Return whether the plugin can work with *model*."""

        return hasattr(model, "predict_proba")

    def explain(self, model, x, **kwargs):
        """Produce a calibrated explanation payload for ``x``."""

        probabilities = model.predict_proba(x)
        return {
            "prediction": probabilities[:, 1],
            "uncertainty_interval": (probabilities[:, 0], probabilities[:, 1]),
            "modes": self.plugin_meta["modes"],
        }


def test_plugin_contract_class_behaviour():
    from typing import Any
    from calibrated_explanations.plugins.base import ExplainerPlugin

    plugin = HelloCalibratedPlugin()
    _ = ExplainerPlugin
    model: Any = _build_dummy_model()
    sample = np.array([[0.2], [0.8]])
    assert plugin.supports(model)
    payload = plugin.explain(model, sample)
    assert np.allclose(payload["prediction"], sample.ravel())
    assert payload["modes"] == plugin.plugin_meta["modes"]


def test_plugin_contract_registration():
    from calibrated_explanations.plugins.base import validate_plugin_meta
    from calibrated_explanations.plugins import (
        register_explanation_plugin,
        _EXPLANATION_PLUGINS,
    )

    plugin = HelloCalibratedPlugin()
    validate_plugin_meta(dict(plugin.plugin_meta))

    identifier = "external.hello.calibrated"
    try:
        register_explanation_plugin(identifier, plugin)
        assert identifier in _EXPLANATION_PLUGINS
    finally:
        _EXPLANATION_PLUGINS.pop(identifier, None)
