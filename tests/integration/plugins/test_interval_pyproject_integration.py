from __future__ import annotations

import textwrap

import numpy as np
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.plugins.intervals import IntervalCalibratorPlugin
from calibrated_explanations.plugins import (
    clear_interval_plugins,
    ensure_builtin_plugins,
    register_interval_plugin,
    mark_interval_trusted,
)


class PyprojectRecordingIntervalPlugin(IntervalCalibratorPlugin):
    invocations: list[tuple[bool, object]] = []
    plugin_meta = {
        "name": "tests.interval.pyproject_recording",
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

    def create(self, context, *, fast: bool = False):
        type(self).invocations.append((fast, context))
        return object()


def make_simple_classifier_helper():
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


def test_pyproject_interval_override_resolves_plugin(tmp_path, monkeypatch):
    ensure_builtin_plugins()
    PyprojectRecordingIntervalPlugin.invocations = []
    plugin = PyprojectRecordingIntervalPlugin()
    descriptor = register_interval_plugin(plugin.plugin_meta["name"], plugin)
    mark_interval_trusted(descriptor.identifier)

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """
            [tool.calibrated_explanations.intervals]
            default = "tests.interval.pyproject_recording"
            default_fallbacks = ["core.interval.legacy"]
            """
        ).strip()
        + "\n"
    )

    monkeypatch.chdir(tmp_path)

    try:
        model, x_cal, y_cal = make_simple_classifier_helper()
        explainer = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
            mode="classification",
            feature_names=["f0", "f1"],
            categorical_features=[],
            class_labels=["No", "Yes"],
        )

        assert (
            explainer.plugin_manager.interval_plugin_identifiers["default"] == descriptor.identifier
        )
        assert (
            explainer.plugin_manager.interval_plugin_fallbacks["default"][0]
            == descriptor.identifier
        )
        assert (
            descriptor.identifier in explainer.plugin_manager.interval_plugin_fallbacks["default"]
        )
        assert (
            PyprojectRecordingIntervalPlugin.invocations
        ), "pyproject override should invoke the registered plugin"

        fast_flag, context = PyprojectRecordingIntervalPlugin.invocations[-1]
        assert fast_flag is False
        assert context.metadata.get("operation") == "initialize"
    finally:
        clear_interval_plugins()
        ensure_builtin_plugins()
