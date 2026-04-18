from __future__ import annotations

import importlib
import importlib.util
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_viz_coloring_helpers_cover_primary_and_fallback_paths() -> None:
    coloring = importlib.import_module("calibrated_explanations.viz.coloring")

    colors = coloring.color_brew(3)
    assert len(colors) == 3
    assert all(len(rgb) == 3 for rgb in colors)
    assert all(isinstance(component, int) for rgb in colors for component in rgb)

    assert coloring.get_fill_color({"predict": 0.8}).startswith("#")
    assert coloring.get_fill_color({"predict": 0.2}, reduction=0.4).startswith("#")
    # Non-numeric prediction triggers the defensive fallback path.
    assert coloring.get_fill_color({"predict": "not-a-number"}).startswith("#")


def test_fast_explanation_plugin_registration_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_mod = importlib.import_module("calibrated_explanations.plugins.explanations_fast")
    calls: dict[str, object] = {}

    monkeypatch.setattr(fast_mod, "find_explanation_descriptor", lambda _name: None)

    def fake_register(name, plugin, source):
        calls["name"] = name
        calls["plugin"] = plugin
        calls["source"] = source

    monkeypatch.setattr(fast_mod, "register_explanation_plugin", fake_register)
    fast_mod.register_fast_explanation_plugin()

    assert calls["name"] == "core.explanation.fast"
    assert calls["source"] == "builtin"
    plugin = calls["plugin"]
    assert plugin.plugin_meta["name"] == "core.explanation.fast"
    assert "explanation:fast" in plugin.plugin_meta["capabilities"]

    # Descriptor present -> early return, no registration call.
    calls.clear()
    monkeypatch.setattr(fast_mod, "find_explanation_descriptor", lambda _name: object())
    fast_mod.register_fast_explanation_plugin()
    assert calls == {}


def test_lazy_package_surfaces_cache_and_parallel() -> None:
    cache_pkg = importlib.import_module("calibrated_explanations.cache")
    parallel_pkg = importlib.import_module("calibrated_explanations.parallel")

    assert cache_pkg.CacheConfig is not None
    assert cache_pkg.ExplanationCacheFacade is not None
    assert parallel_pkg.ParallelConfig is not None
    assert parallel_pkg.ParallelExecutor is not None

    with pytest.raises(AttributeError):
        _ = cache_pkg.not_exported
    with pytest.raises(AttributeError):
        _ = parallel_pkg.not_exported


def test_viz_package_requires_matplotlib_for_render(monkeypatch: pytest.MonkeyPatch) -> None:
    viz_pkg = importlib.import_module("calibrated_explanations.viz")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(ModuleNotFoundError, match="Visualization requires matplotlib"):
        _ = viz_pkg.render
    with pytest.raises(AttributeError):
        _ = viz_pkg.not_exported


def test_core_reject_shim_package_import_warns() -> None:
    shim_path = Path("src/calibrated_explanations/core/reject.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "calibrated_explanations.core.reject_compat_tmp",
        shim_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        spec.loader.exec_module(module)
    assert module.RejectPolicy is not None
    assert any(isinstance(item.message, DeprecationWarning) for item in caught)


def test_plotting_config_helpers_and_style_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plotting = importlib.import_module("calibrated_explanations.plotting")
    plotting.reset_plotting_config_manager()

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.calibrated_explanations.plots]\nstyle = 'custom.primary'\nfallbacks = 'extra.one, extra.two'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    settings = plotting.read_plot_pyproject()
    assert settings["style"] == "custom.primary"

    monkeypatch.setenv("CE_PLOT_STYLE", "env.primary")
    monkeypatch.setenv("CE_PLOT_STYLE_FALLBACKS", "env.one, env.two")
    explainer = SimpleNamespace(
        last_explanation_mode="classification",
        plot_plugin_fallbacks={"classification": ("mode.one", "mode.two")},
    )
    chain = plotting.resolve_plot_style_chain(explainer, explicit_style="explicit.primary")
    assert chain[0] == "explicit.primary"
    assert "plot_spec.default" in chain
    assert "legacy" in chain
    assert chain.index("legacy") < chain.index("plot_spec.default")
    assert "mode.one" in chain

    monkeypatch.delenv("CE_PLOT_STYLE", raising=False)
    monkeypatch.delenv("CE_PLOT_STYLE_FALLBACKS", raising=False)
    pyproject.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "calibrated_explanations.plugins.registry.list_plot_style_descriptors",
        lambda: [SimpleNamespace(identifier="registry.default", metadata={"is_default": True})],
    )
    plotting.reset_plotting_config_manager()
    chain2 = plotting.resolve_plot_style_chain(SimpleNamespace(last_explanation_mode=None))
    assert "registry.default" in chain2
    plotting.reset_plotting_config_manager()
    assert "legacy" in chain2
    assert chain2.index("legacy") < chain2.index("plot_spec.default")


def test_plotting_update_plot_config_writes_normalized_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plotting = importlib.import_module("calibrated_explanations.plotting")
    config_path = tmp_path / "plot_config.ini"
    monkeypatch.setattr(plotting, "_plot_config_path", lambda: config_path)

    plotting.update_plot_config({"style": {"base": "ggplot"}})
    text = config_path.read_text(encoding="utf-8")
    assert "base = ggplot" in text
    assert text.endswith("\n")


def test_plotspec_serializer_boundary_helpers() -> None:
    plotspec = importlib.import_module("calibrated_explanations.viz.plotspec")
    serializers = importlib.import_module("calibrated_explanations.viz.serializers")

    spec = plotspec.GlobalPlotSpec(
        title="demo",
        kind="global_probabilistic",
        mode="classification",
        global_entries=plotspec.GlobalSpec(proba=[0.2, 0.8], uncertainty=[0.1, 0.2]),
    )

    envelope = serializers.global_plotspec_to_dict(spec)
    assert isinstance(envelope, dict)
    assert envelope.get("plot_spec", {}).get("kind") == "global_probabilistic"


def test_base_explain_executor_abstract_method_bodies_are_callable() -> None:
    base_mod = importlib.import_module("calibrated_explanations.core.explain._base")
    assert base_mod.BaseExplainExecutor.supports(object(), None, None) is None
    assert base_mod.BaseExplainExecutor.execute(object(), None, None, None) is None
    assert base_mod.BaseExplainExecutor.name.fget(object()) is None
    assert base_mod.BaseExplainExecutor.priority.fget(object()) is None


def test_core_reject_shim_re_raises_when_absolute_fallback_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shim_path = Path("src/calibrated_explanations/core/reject.py").resolve()
    spec = importlib.util.spec_from_file_location("ce_reject_fallback_fail_tmp", shim_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "calibrated_explanations.core.reject.policy":
            raise ImportError("simulated fallback failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    with pytest.raises(ImportError, match="simulated fallback failure"):
        spec.loader.exec_module(module)


def test_plot_probabilistic_resolves_explainer_then_fails_without_matplotlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plotting = importlib.reload(importlib.import_module("calibrated_explanations.plotting"))
    from calibrated_explanations.utils.exceptions import ConfigurationError

    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("simulated matplotlib failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    explanation = SimpleNamespace(
        calibrated_explanations=SimpleNamespace(
            get_explainer=lambda: SimpleNamespace(
                last_explanation_mode=None, plot_plugin_fallbacks={}
            )
        )
    )

    with pytest.raises(ConfigurationError, match="Plotting requires matplotlib"):
        plotting.plot_probabilistic(
            explanation=explanation,
            instance=[1.0],
            predict={"predict": 0.5},
            feature_weights={"predict": [0.5]},
            features_to_plot=[0],
            num_to_show=1,
            column_names=["x0"],
            title="demo",
            path=".",
            show=False,
        )


def test_instantiate_discretizer_observed_source_uses_y_cal_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = importlib.import_module("calibrated_explanations.core.discretizer_config")

    class DummyDiscretizer:
        def __init__(self, *_args, **kwargs):
            self.labels = kwargs["labels"]

    monkeypatch.setattr(mod, "EntropyDiscretizer", DummyDiscretizer)
    out = mod.instantiate_discretizer(
        discretizer_name="entropy",
        x_cal=np.asarray([[0.0], [1.0]]),
        features_to_ignore=np.asarray([], dtype=int),
        feature_names=["x0"],
        y_cal=np.asarray([0, 1]),
        seed=0,
        condition_source="observed",
    )
    assert out.labels.tolist() == [0, 1]


def test_instantiate_discretizer_label_array_failure_falls_back_to_y_cal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = importlib.import_module("calibrated_explanations.core.discretizer_config")
    x_cal = np.asarray([[0.0], [1.0]])
    y_cal = np.asarray([2, 3])
    condition_labels = np.asarray([9, 9])
    features_to_ignore = np.asarray([], dtype=int)

    class DummyDiscretizer:
        def __init__(self, *_args, **kwargs):
            self.labels = kwargs["labels"]

    monkeypatch.setattr(mod, "EntropyDiscretizer", DummyDiscretizer)
    monkeypatch.setattr(
        mod.np, "asarray", lambda _value: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    out = mod.instantiate_discretizer(
        discretizer_name="entropy",
        x_cal=x_cal,
        features_to_ignore=features_to_ignore,
        feature_names=["x0"],
        y_cal=y_cal,
        seed=0,
        condition_labels=condition_labels,
        condition_source="prediction",
    )
    assert out.labels.tolist() == [2, 3]
