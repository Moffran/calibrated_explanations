from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import calibrated_explanations.plugins as ce_plugins
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.plugins.plots import PlotRenderResult
from calibrated_explanations import plotting


@pytest.fixture(autouse=True)
def _reset_plotting_config_manager() -> None:
    plotting.reset_plotting_config_manager()
    yield
    plotting.reset_plotting_config_manager()


def configure_legacy_style_preference(
    monkeypatch: pytest.MonkeyPatch,
    config_source: str,
) -> None:
    if config_source == "env":
        monkeypatch.setenv("CE_PLOT_STYLE", "legacy")
        monkeypatch.delenv("CE_PLOT_STYLE_FALLBACKS", raising=False)
        return
    if config_source == "pyproject":
        monkeypatch.setattr(plotting, "_read_plot_pyproject", lambda: {"style": "legacy"})
        return
    raise AssertionError(f"Unsupported config source: {config_source}")


def explanation_without_explainer() -> SimpleNamespace:
    return SimpleNamespace(
        prediction={"classes": 0},
        y_minmax=(0.0, 1.0),
        get_class_labels=lambda: ("neg", "pos"),
        is_thresholded=lambda: False,
        y_threshold=None,
    )


def make_public_plot_wrapper() -> WrapCalibratedExplainer:
    x_proper = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y_proper = np.array([0, 0, 1, 1], dtype=int)
    x_cal = np.array([[0.5], [1.5], [2.5], [3.5]], dtype=float)
    y_cal = np.array([0, 0, 1, 1], dtype=int)

    wrapper = WrapCalibratedExplainer(LogisticRegression(random_state=0, solver="liblinear"))
    wrapper.fit(x_proper, y_proper)
    wrapper.calibrate(x_cal, y_cal)
    return wrapper


def install_public_global_plot_plugin(
    monkeypatch: pytest.MonkeyPatch,
    explainer: CalibratedExplainer,
    *,
    style: str = "plotly.global.instance_explorer",
) -> list[tuple[str | None, str | None]]:
    class PlotPlugin:
        plugin_meta = {"style": style}

        def build(self, context: object) -> object:
            return {"plot_spec": context.style, "context": context}

        def render(self, artifact: object, *, context: object) -> PlotRenderResult:
            return PlotRenderResult(artifact=artifact, figure=None, saved_paths=(), extras={})

    resolve_calls: list[tuple[str | None, str | None]] = []

    monkeypatch.setattr(ce_plugins, "ensure_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        explainer.plugin_manager,
        "resolve_plot_plugin",
        lambda *, explicit_style=None, renderer_override=None: resolve_calls.append(
            (explicit_style, renderer_override)
        )
        or (
            PlotPlugin(),
            explicit_style or "plot_spec.default",
            (explicit_style or "plot_spec.default", "legacy"),
        ),
    )
    return resolve_calls


def common_chain_kwargs(plot_func_name: str) -> dict[str, object]:
    common: dict[str, object] = {
        "instance": [0.5],
        "predict": {"predict": 0.5, "low": 0.4, "high": 0.6},
        "features_to_plot": [0],
        "num_to_show": 1,
        "column_names": ["f1"],
        "title": "default",
        "path": None,
        "show": False,
        "use_legacy": None,
    }
    if plot_func_name == "plot_alternative":
        common["feature_predict"] = {"predict": [0.5], "low": [0.4], "high": [0.6]}
    else:
        common["feature_weights"] = {"predict": [0.5], "low": [0.4], "high": [0.6]}
    return common


@pytest.mark.parametrize(
    "plot_func_name", ("plot_probabilistic", "plot_regression", "plot_alternative")
)
def test_should_use_plotspec_default_for_chain_based_plotters_when_use_legacy_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    plot_func_name: str,
) -> None:
    captured_styles: list[str | None] = []
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        plotting, "_resolve_explainer_from_explanation", lambda explanation: object()
    )
    monkeypatch.setattr(
        plotting,
        "_resolve_plot_style_chain",
        lambda explainer, style: ("legacy", "plot_spec.default"),
    )
    monkeypatch.setattr(
        plotting,
        "_render_instance_plot_plugin",
        lambda explanation, explicit_style=None, **kwargs: captured_styles.append(explicit_style)
        or {"style": explicit_style},
    )
    monkeypatch.setattr(
        plotting.legacy,
        "_plot_probabilistic",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        plotting.legacy,
        "plot_regression",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        plotting.legacy,
        "plot_alternative",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    plot_func = getattr(plotting, plot_func_name)
    plot_func(explanation_without_explainer(), **common_chain_kwargs(plot_func_name))

    assert captured_styles == ["plot_spec.default"]
    assert not legacy_calls


@pytest.mark.parametrize(
    ("plot_func_name", "legacy_attr"),
    (
        ("plot_probabilistic", "_plot_probabilistic"),
        ("plot_regression", "plot_regression"),
        ("plot_alternative", "plot_alternative"),
    ),
)
def test_should_preserve_legacy_opt_out_for_chain_based_plotters(
    monkeypatch: pytest.MonkeyPatch,
    plot_func_name: str,
    legacy_attr: str,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    plotspec_calls: list[str | None] = []

    monkeypatch.setattr(
        plotting,
        "_render_instance_plot_plugin",
        lambda explanation, explicit_style=None, **kwargs: plotspec_calls.append(explicit_style),
    )
    monkeypatch.setattr(
        plotting.legacy,
        legacy_attr,
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    call_kwargs = common_chain_kwargs(plot_func_name)
    call_kwargs["use_legacy"] = True

    plot_func = getattr(plotting, plot_func_name)
    plot_func(explanation_without_explainer(), **call_kwargs)

    assert legacy_calls
    assert not plotspec_calls


@pytest.mark.parametrize(
    ("plot_func_name", "legacy_attr"),
    (
        ("plot_probabilistic", "_plot_probabilistic"),
        ("plot_regression", "plot_regression"),
        ("plot_alternative", "plot_alternative"),
    ),
)
def test_should_preserve_explicit_legacy_style_for_chain_based_plotters(
    monkeypatch: pytest.MonkeyPatch,
    plot_func_name: str,
    legacy_attr: str,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        plotting.legacy,
        legacy_attr,
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    call_kwargs = common_chain_kwargs(plot_func_name)
    call_kwargs["style_override"] = "legacy"

    plot_func = getattr(plotting, plot_func_name)
    plot_func(explanation_without_explainer(), **call_kwargs)

    assert legacy_calls


@pytest.mark.parametrize(
    ("plot_func_name", "legacy_attr", "config_source"),
    (
        ("plot_probabilistic", "_plot_probabilistic", "env"),
        ("plot_probabilistic", "_plot_probabilistic", "pyproject"),
        ("plot_regression", "plot_regression", "env"),
        ("plot_regression", "plot_regression", "pyproject"),
        ("plot_alternative", "plot_alternative", "env"),
        ("plot_alternative", "plot_alternative", "pyproject"),
    ),
)
def test_should_preserve_configured_legacy_opt_out_for_chain_based_plotters_when_style_config_prefers_legacy(
    monkeypatch: pytest.MonkeyPatch,
    plot_func_name: str,
    legacy_attr: str,
    config_source: str,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    plotspec_calls: list[str | None] = []

    configure_legacy_style_preference(monkeypatch, config_source)
    monkeypatch.setattr(
        plotting, "_resolve_explainer_from_explanation", lambda explanation: object()
    )
    monkeypatch.setattr(
        plotting,
        "_render_instance_plot_plugin",
        lambda explanation, explicit_style=None, **kwargs: plotspec_calls.append(explicit_style),
    )
    monkeypatch.setattr(
        plotting.legacy,
        legacy_attr,
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    plot_func = getattr(plotting, plot_func_name)
    plot_func(explanation_without_explainer(), **common_chain_kwargs(plot_func_name))

    assert legacy_calls
    assert not plotspec_calls


def test_should_use_plotspec_default_for_triangular_when_use_legacy_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    render_calls: list[Any] = []
    spec = object()

    monkeypatch.setattr(plotting, "plt", object())
    monkeypatch.setattr(
        plotting.legacy,
        "plot_triangular",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    import calibrated_explanations.viz.builders as builders
    import calibrated_explanations.viz.matplotlib_adapter as matplotlib_adapter

    monkeypatch.setattr(builders, "build_triangular_plotspec", lambda **kwargs: spec)
    monkeypatch.setattr(
        matplotlib_adapter,
        "render",
        lambda plot_spec, **kwargs: render_calls.append(plot_spec),
    )

    plotting.plot_triangular(
        explanation=None,
        proba=[0.2],
        uncertainty=[0.1],
        rule_proba=[0.3],
        rule_uncertainty=[0.05],
        num_to_show=1,
        title="triangular",
        path=None,
        show=False,
    )

    assert render_calls == [spec]
    assert not legacy_calls


def test_should_mark_triangular_plotspec_as_regression_when_plain_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    spec = object()

    class PlainRegressionExplanation:
        def get_mode(self) -> str:
            return "regression"

        def is_thresholded(self) -> bool:
            return False

    monkeypatch.setattr(plotting, "plt", object())

    import calibrated_explanations.viz.builders as builders
    import calibrated_explanations.viz.matplotlib_adapter as matplotlib_adapter

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return spec

    monkeypatch.setattr(builders, "build_triangular_plotspec", fake_builder)
    monkeypatch.setattr(matplotlib_adapter, "render", lambda plot_spec, **kwargs: None)

    plotting.plot_triangular(
        explanation=PlainRegressionExplanation(),
        proba=[185000.0],
        uncertainty=[26000.0],
        rule_proba=[210000.0],
        rule_uncertainty=[30000.0],
        num_to_show=1,
        title="triangular_regression",
        path=None,
        show=False,
    )

    assert captured["is_probabilistic"] is False


def test_should_preserve_legacy_opt_out_for_triangular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        plotting.legacy,
        "plot_triangular",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    plotting.plot_triangular(
        explanation=None,
        proba=[0.2],
        uncertainty=[0.1],
        rule_proba=[0.3],
        rule_uncertainty=[0.05],
        num_to_show=1,
        title="triangular",
        path=None,
        show=False,
        use_legacy=True,
    )

    assert legacy_calls


@pytest.mark.parametrize("config_source", ("env", "pyproject"))
def test_should_preserve_configured_legacy_opt_out_for_triangular_when_style_config_prefers_legacy(
    monkeypatch: pytest.MonkeyPatch,
    config_source: str,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    render_calls: list[Any] = []

    configure_legacy_style_preference(monkeypatch, config_source)
    monkeypatch.setattr(
        plotting.legacy,
        "plot_triangular",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    import calibrated_explanations.viz.matplotlib_adapter as matplotlib_adapter

    monkeypatch.setattr(
        matplotlib_adapter,
        "render",
        lambda plot_spec, **kwargs: render_calls.append(plot_spec),
    )

    plotting.plot_triangular(
        explanation=None,
        proba=[0.2],
        uncertainty=[0.1],
        rule_proba=[0.3],
        rule_uncertainty=[0.05],
        num_to_show=1,
        title="triangular",
        path=None,
        show=False,
    )

    assert legacy_calls
    assert not render_calls


def test_should_fallback_visibly_when_triangular_plotspec_rendering_fails(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    spec = object()

    monkeypatch.setattr(plotting, "plt", object())
    monkeypatch.setattr(
        plotting.legacy,
        "plot_triangular",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    import calibrated_explanations.viz.builders as builders
    import calibrated_explanations.viz.matplotlib_adapter as matplotlib_adapter

    monkeypatch.setattr(builders, "build_triangular_plotspec", lambda **kwargs: spec)
    monkeypatch.setattr(
        matplotlib_adapter,
        "render",
        lambda plot_spec, **kwargs: (_ for _ in ()).throw(RuntimeError("render boom")),
    )

    with (
        caplog.at_level(logging.INFO, logger="calibrated_explanations.plotting"),
        pytest.warns(UserWarning, match="Falling back to legacy plot"),
    ):
        plotting.plot_triangular(
            explanation=None,
            proba=[0.2],
            uncertainty=[0.1],
            rule_proba=[0.3],
            rule_uncertainty=[0.05],
            num_to_show=1,
            title="triangular",
            path=None,
            show=False,
        )

    assert legacy_calls
    assert "Falling back to legacy plot" in caplog.text


def test_should_use_plotspec_default_for_global_when_use_legacy_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    render_calls: list[tuple[object, str, tuple[str, ...]]] = []

    monkeypatch.setattr(
        plotting.legacy,
        "plot_global",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    class PlotSpecPlugin:
        plugin_meta = {"style": "plot_spec.default"}

        def build(self, context: object) -> object:
            return {"plot_spec": "global"}

        def render(self, artifact: object, *, context: object) -> object:
            render_calls.append((artifact, "plot_spec.default", ("plot_spec.default", "legacy")))
            return {"rendered": artifact}

    class PluginManager:
        def resolve_plot_plugin(self, *, explicit_style=None, renderer_override=None):
            return PlotSpecPlugin(), "plot_spec.default", ("plot_spec.default", "legacy")

    class Learner:
        def predict_proba(self) -> None:
            return None

    explainer = SimpleNamespace(
        learner=Learner(),
        plugin_manager=PluginManager(),
        class_labels=("negative", "positive"),
        latest_explanation=None,
        _last_explanation_mode="classification",
        predict_proba=lambda x, uq_interval=True, threshold=None, bins=None: (
            [0.4, 0.6],
            ([0.3, 0.5], [0.5, 0.7]),
        ),
    )

    result = plotting.plot_global(explainer, x=[1, 2], show=False)

    assert result == {"rendered": {"plot_spec": "global"}}
    assert render_calls
    assert not legacy_calls


def test_should_not_treat_unimported_matplotlib_as_global_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    render_contexts: list[object] = []

    monkeypatch.setattr(plotting, "plt", None)

    class PlotSpecPlugin:
        plugin_meta = {"style": "plot_spec.default"}

        def build(self, context: object) -> object:
            return {"plot_spec": "global"}

        def render(self, artifact: object, *, context: object) -> object:
            render_contexts.append(context)
            return {"rendered": artifact}

    class PluginManager:
        def resolve_plot_plugin(self, *, explicit_style=None, renderer_override=None):
            return PlotSpecPlugin(), "plot_spec.default", ("plot_spec.default", "legacy")

    class Learner:
        def predict_proba(self) -> None:
            return None

    explainer = SimpleNamespace(
        learner=Learner(),
        plugin_manager=PluginManager(),
        class_labels=("negative", "positive"),
        latest_explanation=None,
        _last_explanation_mode="classification",
        predict_proba=lambda x, uq_interval=True, threshold=None, bins=None: (
            [0.4, 0.6],
            ([0.3, 0.5], [0.5, 0.7]),
        ),
    )

    result = plotting.plot_global(explainer, x=[1, 2], show=True)

    assert result == {"rendered": {"plot_spec": "global"}}
    assert render_contexts


def test_should_preserve_legacy_opt_out_for_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        plotting.legacy,
        "plot_global",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    plotting.plot_global(SimpleNamespace(), x=[1, 2], show=False, use_legacy=True)

    assert legacy_calls


def test_should_preserve_configured_legacy_opt_out_for_global_without_fallback_warning_when_manager_prefers_legacy(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    resolve_calls: list[tuple[str | None, str | None]] = []

    monkeypatch.setattr(
        plotting.legacy,
        "plot_global",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    class PluginManager:
        plot_style_override = "legacy"

        def resolve_plot_plugin(self, *, explicit_style=None, renderer_override=None):
            resolve_calls.append((explicit_style, renderer_override))
            raise AssertionError("resolve_plot_plugin should not run for configured legacy opt-out")

    class Learner:
        def predict_proba(self) -> None:
            return None

    explainer = SimpleNamespace(
        learner=Learner(),
        plugin_manager=PluginManager(),
        class_labels=("negative", "positive"),
        latest_explanation=None,
        _last_explanation_mode="classification",
        predict_proba=lambda x, uq_interval=True, threshold=None, bins=None: (
            [0.4, 0.6],
            ([0.3, 0.5], [0.5, 0.7]),
        ),
    )

    with (
        caplog.at_level(logging.INFO, logger="calibrated_explanations.plotting"),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        plotting.plot_global(explainer, x=[1, 2], show=False)

    assert legacy_calls
    assert not resolve_calls
    assert not caught
    assert "Falling back to legacy plot" not in caplog.text


def test_should_fallback_visibly_when_global_plotspec_rendering_fails(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        plotting.legacy,
        "plot_global",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    class BrokenPlotSpecPlugin:
        plugin_meta = {"style": "plot_spec.default"}

        def build(self, context: object) -> object:
            return {"plot_spec": "global"}

        def render(self, artifact: object, *, context: object) -> object:
            raise RuntimeError("render boom")

    class PluginManager:
        def resolve_plot_plugin(self, *, explicit_style=None, renderer_override=None):
            return BrokenPlotSpecPlugin(), "plot_spec.default", ("plot_spec.default", "legacy")

    class Learner:
        def predict_proba(self) -> None:
            return None

    explainer = SimpleNamespace(
        learner=Learner(),
        plugin_manager=PluginManager(),
        class_labels=("negative", "positive"),
        latest_explanation=None,
        _last_explanation_mode="classification",
        predict_proba=lambda x, uq_interval=True, threshold=None, bins=None: (
            [0.4, 0.6],
            ([0.3, 0.5], [0.5, 0.7]),
        ),
    )

    with (
        caplog.at_level(logging.INFO, logger="calibrated_explanations.plotting"),
        pytest.warns(UserWarning, match="Falling back to legacy plot"),
    ):
        plotting.plot_global(explainer, x=[1, 2], show=False)

    assert legacy_calls
    assert "Falling back to legacy plot" in caplog.text


def test_should_return_plugin_result_from_calibrated_explainer_plot_when_explicit_global_style_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = make_public_plot_wrapper()
    explainer = wrapper.explainer
    resolve_calls = install_public_global_plot_plugin(monkeypatch, explainer)

    result = explainer.plot(
        x=[[1.0], [2.0]],
        style="plotly.global.instance_explorer",
        show=False,
    )

    assert isinstance(result, PlotRenderResult)
    assert result.artifact is not None
    assert result.artifact["plot_spec"] == "plotly.global.instance_explorer"
    assert resolve_calls == [("plotly.global.instance_explorer", None)]


def test_should_return_plugin_result_from_wrap_plot_with_targets_when_explicit_global_style_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = make_public_plot_wrapper()
    resolve_calls = install_public_global_plot_plugin(monkeypatch, wrapper.explainer)

    result = wrapper.plot(
        x=[[1.0], [2.0]],
        y=[0, 1],
        style="plotly.global.instance_explorer",
        show=False,
    )

    assert isinstance(result, PlotRenderResult)
    assert result.artifact is not None
    assert result.artifact["plot_spec"] == "plotly.global.instance_explorer"
    assert resolve_calls == [("plotly.global.instance_explorer", None)]


def test_should_preserve_default_global_plot_behavior_from_wrap_plot_when_style_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = make_public_plot_wrapper()
    resolve_calls = install_public_global_plot_plugin(monkeypatch, wrapper.explainer)

    result = wrapper.plot(x=[[1.0], [2.0]], show=False)

    assert isinstance(result, PlotRenderResult)
    assert result.artifact is not None
    assert result.artifact["plot_spec"] == "plot_spec.default"
    assert resolve_calls == [("plot_spec.default", None)]


def test_should_force_legacy_global_plot_from_wrap_plot_when_use_legacy_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = make_public_plot_wrapper()
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    resolve_calls: list[tuple[str | None, str | None]] = []
    monkeypatch.setattr(ce_plugins, "ensure_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        wrapper.explainer.plugin_manager,
        "resolve_plot_plugin",
        lambda *, explicit_style=None, renderer_override=None: resolve_calls.append(
            (explicit_style, renderer_override)
        ),
    )
    monkeypatch.setattr(
        plotting.legacy,
        "plot_global",
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    result = wrapper.plot(x=[[1.0], [2.0]], show=False, use_legacy=True)

    assert result is None
    assert legacy_calls
    assert resolve_calls == []
