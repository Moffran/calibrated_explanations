from __future__ import annotations

import importlib
import os
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
    assert render_calls[0]["save_path"] is None
    assert render_calls[1]["save_path"] == "example.png"


def test_probabilistic_shim_uses_default_save_extensions(monkeypatch, tmp_path):
    sys.modules.pop("calibrated_explanations._plots", None)
    with pytest.warns(DeprecationWarning):
        plots_module = importlib.import_module("calibrated_explanations._plots")

    render_calls: list[dict] = []

    def fake_render(spec, **kwargs):
        render_calls.append({"spec": spec, **kwargs})

    monkeypatch.setattr("calibrated_explanations.viz.matplotlib_adapter.render", fake_render)

    explanation = types.SimpleNamespace(y_minmax=(0.0, 1.0), prediction={"classes": 1})
    plots_module._plot_probabilistic(
        explanation,
        instance=np.array([0.2, 0.4]),
        predict={"predict": 0.6, "low": -np.inf, "high": np.inf},
        feature_weights=np.array([0.1, -0.2]),
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["a", "b"],
        title="prob",
        path=str(tmp_path) + "/",
        show=True,
        interval=False,
        idx=None,
        save_ext=None,
        use_legacy=False,
    )

    # First render call honours caller's show flag, subsequent calls save default extensions.
    assert render_calls[0]["show"] is True
    assert render_calls[0]["save_path"] is None
    spec = render_calls[0]["spec"]
    assert spec.header.low == pytest.approx(0.0)  # type: ignore[union-attr]
    assert spec.header.high == pytest.approx(1.0)  # type: ignore[union-attr]
    saved_paths = [call["save_path"] for call in render_calls[1:]]
    assert saved_paths == [
        os.path.join(str(tmp_path), "prob" + ext) for ext in ["svg", "pdf", "png"]
    ]


def test_plots_module_defaults_save_extensions(monkeypatch, tmp_path):
    sys.modules.pop("calibrated_explanations._plots", None)
    with pytest.warns(DeprecationWarning):
        plots_module = importlib.import_module("calibrated_explanations._plots")

    sentinel_spec = object()

    def fake_builder(**kwargs):
        return sentinel_spec

    render_calls: list[dict] = []

    def fake_render(spec, **kwargs):
        render_calls.append({"spec": spec, **kwargs})

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_regression_bars_spec", fake_builder
    )
    monkeypatch.setattr("calibrated_explanations.viz.matplotlib_adapter.render", fake_render)

    out_dir = tmp_path / "plots"
    out_dir.mkdir()
    plots_module._plot_regression(
        types.SimpleNamespace(y_minmax=[0.0, 1.0]),
        instance=np.array([0.1, 0.2]),
        predict={"predict": np.array([0.5]), "low": np.array([0.2]), "high": np.array([0.8])},
        feature_weights={"f1": np.array([0.1])},
        features_to_plot=["f1"],
        num_to_show=1,
        column_names=["f1"],
        title="example",
        path=str(out_dir) + "/",
        show=True,
        interval=False,
        idx=None,
        save_ext=None,
        use_legacy=False,
    )

    assert render_calls[0]["show"] is True
    saved_paths = [call["save_path"] for call in render_calls[1:]]
    assert saved_paths == [
        str(out_dir / "examplesvg"),
        str(out_dir / "examplepdf"),
        str(out_dir / "examplepng"),
    ]


def test_plots_module_defaults_save_ext_when_not_provided(monkeypatch):
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
        "calibrated_explanations.viz.builders.build_probabilistic_bars_spec", fake_builder
    )
    monkeypatch.setattr("calibrated_explanations.viz.matplotlib_adapter.render", fake_render)

    explanation = types.SimpleNamespace(
        y_minmax=[0.0, 1.0],
        prediction={"classes": 1},
        get_class_labels=lambda: ["neg", "pos"],
        is_thresholded=lambda: False,
        get_mode=lambda: "classification",
        is_one_sided=lambda: False,
    )
    setattr(explanation, "_get_explainer", lambda: None)

    plots_module._plot_probabilistic(
        explanation,
        instance=np.array([0.1, 0.2]),
        predict={"predict": 0.5, "low": 0.2, "high": 0.8},
        feature_weights={
            "predict": np.array([0.1, -0.1]),
            "low": np.array([0.0, 0.0]),
            "high": np.array([0.2, 0.2]),
        },
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["f0", "f1"],
        title="defaults",
        path="/tmp/",
        show=True,
        interval=True,
        idx=0,
        save_ext=None,
        use_legacy=False,
    )

    assert build_calls["kwargs"]["title"] == "defaults"
    assert render_calls[0]["show"] is True
    assert render_calls[0]["save_path"] is None
    paths = [call["save_path"] for call in render_calls[1:]]
    assert paths == ["/tmp/defaultssvg", "/tmp/defaultspdf", "/tmp/defaultspng"]


def test_plots_legacy_shim_reexports(monkeypatch):
    sys.modules.pop("calibrated_explanations._plots_legacy", None)
    with pytest.warns(DeprecationWarning):
        legacy_module = importlib.import_module("calibrated_explanations._plots_legacy")

    from calibrated_explanations.legacy import plotting as legacy_plotting

    assert legacy_module._plot_regression is legacy_plotting._plot_regression
