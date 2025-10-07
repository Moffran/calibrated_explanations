from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.core import calibrated_explainer as explainer_module
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.plugins.registry import ensure_builtin_plugins


def _make_simple_model():
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression(random_state=0)
    model.fit(x, y)
    return model, x, y


def test_pyproject_sections_seed_interval_and_plot_chains(monkeypatch):
    ensure_builtin_plugins()

    def fake_read(path):
        if tuple(path) == ("tool", "calibrated_explanations", "intervals"):
            return {
                "default": "tests.interval.pyproject",
                "default_fallbacks": ["tests.interval.secondary"],
            }
        if tuple(path) == ("tool", "calibrated_explanations", "plots"):
            return {
                "style": "tests.plot.pyproject",
                "style_fallbacks": ["tests.plot.secondary"],
            }
        return {}

    monkeypatch.setattr(
        explainer_module,
        "_read_pyproject_section",
        fake_read,
    )

    model, x_cal, y_cal = _make_simple_model()

    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="classification",
        feature_names=["f0", "f1"],
        categorical_features=[],
        class_labels=["No", "Yes"],
    )

    chain = explainer._interval_plugin_fallbacks["default"]
    assert chain[0] == "tests.interval.pyproject"
    assert "tests.interval.secondary" in chain
    assert chain[-1] == "core.interval.legacy"

    plot_chain = explainer._plot_style_chain
    assert plot_chain[0] == "tests.plot.pyproject"
    assert plot_chain[-1] == "legacy"
