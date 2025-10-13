from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


def test_plots_module_deprecated_and_calls_adapter(monkeypatch):
    sys.modules.pop("calibrated_explanations._plots", None)
    with pytest.warns(DeprecationWarning):
        plots_module = importlib.import_module("calibrated_explanations._plots")

    sentinel_spec = object()
    build_calls: dict[str, dict] = {}

    def fake_builder(**kwargs):
        build_calls["kwargs"] = kwargs
        return sentinel_spec

    render_calls: list[dict] = []

    def fake_render(spec, **kwargs):
        render_calls.append({"spec": spec, **kwargs})

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_regression_bars_spec", fake_builder
    )
    monkeypatch.setattr("calibrated_explanations.viz.matplotlib_adapter.render", fake_render)

    explanation = types.SimpleNamespace(y_minmax=[0.0, 1.0])
    plots_module._plot_regression(
        explanation,
        instance=np.array([0.1, 0.2]),
        predict={"predict": np.array([0.5]), "low": np.array([0.2]), "high": np.array([0.8])},
        feature_weights={"f1": np.array([0.1])},
        features_to_plot=["f1"],
        num_to_show=1,
        column_names=["f1"],
        title="example",
        path="",
        show=False,
        interval=False,
        idx=None,
        save_ext=[".png"],
        use_legacy=False,
    )

    assert build_calls["kwargs"]["title"] == "example"
    assert render_calls[0]["spec"] is sentinel_spec
    assert render_calls[0]["show"] is False
    assert render_calls[1]["save_path"] == "example.png"


def test_plots_legacy_shim_reexports(monkeypatch):
    sys.modules.pop("calibrated_explanations._plots_legacy", None)
    with pytest.warns(DeprecationWarning):
        legacy_module = importlib.import_module("calibrated_explanations._plots_legacy")

    from calibrated_explanations.legacy import plotting as legacy_plotting

    assert legacy_module._plot_regression is legacy_plotting._plot_regression
