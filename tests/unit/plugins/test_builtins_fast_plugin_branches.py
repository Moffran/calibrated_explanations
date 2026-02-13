from __future__ import annotations

import numpy as np
import pytest

import calibrated_explanations.plugins.builtins as builtins_mod
from calibrated_explanations.plugins.intervals import IntervalCalibratorContext


def test_builtin_fast_interval_plugin_create_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    venn_calls: list[tuple] = []
    reg_calls: list[object] = []

    class FakeVennAbers:
        def __init__(self, *args, **kwargs):
            venn_calls.append((args, kwargs))

    class FakeIntervalRegressor:
        def __init__(self, explainer):
            reg_calls.append(explainer)

    def fake_register_interval(identifier, plugin, source="builtin"):
        if identifier == "core.interval.fast":
            captured["plugin"] = plugin

    monkeypatch.setattr(builtins_mod, "find_interval_descriptor", lambda _identifier: None)
    monkeypatch.setattr(
        builtins_mod,
        "register_interval_plugin",
        fake_register_interval,
    )
    monkeypatch.setattr(builtins_mod, "register_explanation_plugin", lambda *a, **k: None)
    monkeypatch.setattr(builtins_mod, "register_plot_builder", lambda *a, **k: None)
    monkeypatch.setattr(builtins_mod, "register_plot_renderer", lambda *a, **k: None)
    monkeypatch.setattr(builtins_mod, "register_plot_style", lambda *a, **k: None)
    monkeypatch.setattr(
        "calibrated_explanations.plugins.explanations_fast.register_fast_explanation_plugin",
        lambda: None,
    )
    monkeypatch.setattr(
        "calibrated_explanations.plugins.builtins.perturb_dataset",
        lambda x, y, categorical_features, **kwargs: (x, x.copy(), y.copy(), 1),
    )
    monkeypatch.setattr(
        "calibrated_explanations.calibration.venn_abers.VennAbers",
        FakeVennAbers,
    )
    monkeypatch.setattr(
        "calibrated_explanations.calibration.interval_regressor.IntervalRegressor",
        FakeIntervalRegressor,
    )

    builtins_mod.register_builtins()
    plugin = captured["plugin"]

    explainer = type(
        "ExplainerStub",
        (),
        {
            "bins": np.asarray([0, 1]),
            "x_cal": np.asarray([[0.0, 1.0], [1.0, 2.0]]),
            "y_cal": np.asarray([0, 1]),
            "predict_function": None,
        },
    )()

    x_cal = np.asarray([[0.0, 1.0], [1.0, 2.0]])
    y_cal = np.asarray([0, 1])
    context_cls = IntervalCalibratorContext(
        learner=object(),
        calibration_splits=[(x_cal, y_cal)],
        bins={"calibration": np.asarray([0, 1])},
        residuals={},
        difficulty={},
        metadata={
            "task": "classification",
            "explainer": explainer,
            "num_features": 2,
            "categorical_features": (),
            "noise_config": {},
        },
        fast_flags={},
    )

    wrapper_cls = plugin.create(context_cls, fast=True)
    assert len(wrapper_cls) == 3
    assert len(venn_calls) == 3

    context_reg = IntervalCalibratorContext(
        learner=object(),
        calibration_splits=[(x_cal, y_cal)],
        bins={"calibration": np.asarray([0, 1])},
        residuals={},
        difficulty={},
        metadata={
            "task": "regression",
            "explainer": explainer,
            "num_features": 2,
            "categorical_features": (),
            "noise_config": {},
        },
        fast_flags={},
    )

    wrapper_reg = plugin.create(context_reg, fast=True)
    assert len(wrapper_reg) == 3
    assert len(reg_calls) == 3
