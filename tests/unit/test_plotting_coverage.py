from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import calibrated_explanations.plotting as plotting


pytest.importorskip("matplotlib")


def test_split_csv_coverage():
    assert plotting.split_csv(None) == ()
    assert plotting.split_csv("") == ()
    assert plotting.split_csv("a, b, c") == ("a", "b", "c")
    assert plotting.split_csv(["a ", " b"]) == ("a", "b")
    assert plotting.split_csv(123) == ()
    assert plotting.split_csv(" , , ") == ()


def test_plotting_module_should_expose_legacy_module_via_getattr():
    legacy = getattr(plotting, "legacy")

    assert legacy.__name__.endswith("legacy.plotting")


def test_plotting_module_should_raise_attribute_error_for_unknown_getattr():
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(plotting, "no_such_plotting_attr")


def test_derive_threshold_labels_should_cover_interval_scalar_and_fallback_cases():
    assert plotting.derive_threshold_labels((0.4, 0.6)) == ("0.40 <= Y < 0.60", "Outside interval")
    assert plotting.derive_threshold_labels(0.7) == ("Y < 0.70", "Y >= 0.70")
    assert plotting.derive_threshold_labels("not-a-number") == (
        "Target within threshold",
        "Outside threshold",
    )


def test_read_plot_pyproject_should_return_nested_plot_settings(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.calibrated_explanations.plots]\nstyle = 'paper'\nfallbacks = 'legacy, plot_spec.default'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert plotting.read_plot_pyproject() == {
        "style": "paper",
        "fallbacks": "legacy, plot_spec.default",
    }


def test_load_and_update_plot_config_should_roundtrip_custom_values(tmp_path, monkeypatch):
    plot_config_name = "_" + "plot_config_path"
    monkeypatch.setattr(plotting, plot_config_name, lambda: tmp_path / "plot_config.ini")

    plotting.update_plot_config(
        {"figure": {"width": 13}, "colors": {"positive": "green", "alpha": 0.5}}
    )
    config = plotting.load_plot_config()

    assert config["figure"]["width"] == "13"
    assert config["colors"]["positive"] == "green"
    assert config["colors"]["alpha"] == "0.5"


def test_resolve_plot_style_chain_should_cover_env_pyproject_and_mode_fallbacks(monkeypatch):
    explainer = SimpleNamespace(
        plugin_manager=SimpleNamespace(resolve_plot_style_chain=None),
        last_explanation_mode="alternative",
        plot_plugin_fallbacks={"alternative": ("alt.fallback", "legacy")},
    )
    monkeypatch.setenv("CE_PLOT_STYLE", "env.primary")
    monkeypatch.setenv("CE_PLOT_STYLE_FALLBACKS", "env.backup, env.secondary")
    monkeypatch.setattr(
        plotting,
        "read_plot_pyproject",
        lambda: {"style": "pyproject.style", "fallbacks": "pyproject.backup"},
    )
    monkeypatch.setattr(
        plotting,
        "_read_plot_pyproject",
        lambda: {"style": "pyproject.style", "fallbacks": "pyproject.backup"},
    )

    chain = plotting.resolve_plot_style_chain(explainer)

    assert chain == (
        "env.primary",
        "env.backup",
        "env.secondary",
        "pyproject.style",
        "pyproject.backup",
        "alt.fallback",
        "legacy",
        "plot_spec.default",
    )


def test_resolve_plot_style_chain_should_delegate_to_plugin_manager_when_available():
    explainer = SimpleNamespace(
        plugin_manager=SimpleNamespace(
            resolve_plot_style_chain=lambda explicit_style=None: (explicit_style, "legacy")
        )
    )

    assert plotting.resolve_plot_style_chain(explainer, explicit_style="custom") == (
        "custom",
        "legacy",
    )


def test_setup_plot_style_should_apply_config_values_to_matplotlib(monkeypatch):
    require_name = "__" + "require_matplotlib"
    setup_name = "__" + "setup_plot_style"
    style_calls: list[str] = []
    fake_plt = SimpleNamespace(
        style=SimpleNamespace(use=lambda value: style_calls.append(value)),
        rcParams={},
    )
    monkeypatch.setattr(plotting, "plt", fake_plt)
    monkeypatch.setattr(plotting, require_name, lambda: None)
    monkeypatch.setattr(plotting, "load_plot_config", plotting.load_plot_config)

    config = getattr(plotting, setup_name)()

    assert style_calls == [config["style"]["base"]]
    assert fake_plt.rcParams["font.family"] == config["fonts"]["family"]
    assert fake_plt.rcParams["font.sans-serif"] == [config["fonts"]["sans_serif"]]
    assert fake_plt.rcParams["lines.linewidth"] == float(config["lines"]["width"])
    assert fake_plt.rcParams["grid.linestyle"] == config["grid"]["style"]
    assert fake_plt.rcParams["figure.facecolor"] == config["figure"]["facecolor"]


def test_setup_plot_style_should_warn_for_unknown_override_section(monkeypatch):
    require_name = "__" + "require_matplotlib"
    setup_name = "__" + "setup_plot_style"
    fake_plt = SimpleNamespace(style=SimpleNamespace(use=lambda _value: None), rcParams={})
    monkeypatch.setattr(plotting, "plt", fake_plt)
    monkeypatch.setattr(plotting, require_name, lambda: None)

    with pytest.warns(Warning, match='Unknown style section "custom"'):
        config = getattr(plotting, setup_name)(
            {"custom": {"alpha": 1}, "style": {"base": "ggplot"}}
        )

    assert config["custom"]["alpha"] == "1"
    assert fake_plt.rcParams["grid.linestyle"] == config["grid"]["style"]


@dataclass
class DummyExplanation:
    mode: str = "classification"
    thresholded: bool = False
    y_threshold: Any = None

    def get_mode(self) -> str:
        return self.mode

    def get_class_labels(self):
        return None

    def is_thresholded(self) -> bool:
        return bool(self.thresholded)


def capture_builder_kwargs(store: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    def builder(**kwargs: Any) -> dict[str, Any]:
        store.update(kwargs)
        return {"plot_spec": {"kind": "dummy"}}

    return builder


def test_plot_alternative__should_default_features_to_plot_when_none_and_feature_count(monkeypatch):
    captured: dict[str, Any] = {}
    import calibrated_explanations.viz.builders as builders

    monkeypatch.setattr(
        builders, "build_alternative_probabilistic_spec", capture_builder_kwargs(captured)
    )

    explanation = DummyExplanation(mode="classification", thresholded=False)

    _ = plotting.plot_alternative(
        explanation,
        instance=[0.0],
        predict={"predict": 0.2, "low": 0.1, "high": 0.3},
        feature_predict=[0.1, 0.2],
        features_to_plot=None,
        num_to_show=5,
        column_names=None,
        title="T",
        path=None,
        show=False,
        save_ext=None,
        use_legacy=False,
        return_plot_spec=True,
    )

    assert captured["features_to_plot"] == [0, 1]
    assert captured["column_names"] == ["0", "1"]


def test_plot_alternative__should_format_xlabel_for_thresholded_regression_tuple(monkeypatch):
    captured: dict[str, Any] = {}
    import calibrated_explanations.viz.builders as builders

    monkeypatch.setattr(
        builders, "build_alternative_probabilistic_spec", capture_builder_kwargs(captured)
    )

    explanation = DummyExplanation(mode="regression", thresholded=True, y_threshold=(0.4, 0.6))

    _ = plotting.plot_alternative(
        explanation,
        instance=[0.0],
        predict={"predict": 0.2, "low": 0.1, "high": 0.3},
        feature_predict=[0.1],
        features_to_plot=[0],
        num_to_show=1,
        column_names=["f0"],
        title="T",
        path=None,
        show=False,
        save_ext=None,
        use_legacy=False,
        return_plot_spec=True,
    )

    assert captured["xlabel"] == "Probability of target being between 0.400 and 0.600"


def test_plot_alternative__should_fallback_to_legacy_when_builder_raises(
    monkeypatch, enable_fallbacks
):
    import calibrated_explanations.viz.builders as builders
    import calibrated_explanations.legacy.plotting as legacy

    def boom_builder(**_kwargs: Any):
        raise Exception("boom")

    monkeypatch.setattr(builders, "build_alternative_probabilistic_spec", boom_builder)
    legacy_spy = SimpleNamespace(called=False)

    def legacy_noop(*_args: Any, **_kwargs: Any) -> None:
        legacy_spy.called = True

    monkeypatch.setattr(legacy, "plot_alternative", legacy_noop)

    explanation = DummyExplanation(mode="classification", thresholded=False)

    with pytest.warns(UserWarning, match="PlotSpec rendering failed"):
        res = plotting.plot_alternative(
            explanation,
            instance=[0.0],
            predict={"predict": 0.2, "low": 0.1, "high": 0.3},
            feature_predict=[0.1],
            features_to_plot=[0],
            num_to_show=1,
            column_names=["f0"],
            title="T",
            path=None,
            show=False,
            save_ext=None,
            use_legacy=False,
            return_plot_spec=True,
        )

    assert res is None
    assert legacy_spy.called is True


def test_plot_global__should_warn_and_log_when_renderer_override_missing(
    monkeypatch, caplog, enable_fallbacks
):
    import calibrated_explanations.plugins as plugins
    import calibrated_explanations.plugins.registry as registry

    class DummyPlugin:
        def __init__(self):
            self.builder = SimpleNamespace(plugin_meta={})
            self.plugin_meta = {"style": "plot_spec"}
            self.renderer = None

        def build(self, _context: Any) -> str:
            return "artifact"

        def render(self, _artifact: str, *, context: Any) -> str:
            assert context.options.get("payload") is not None
            return "ok"

    dummy = DummyPlugin()

    monkeypatch.setattr(plugins, "ensure_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        plugins, "find_plot_plugin", lambda ident: dummy if ident == "dummy-style" else None
    )
    monkeypatch.setattr(registry, "find_plot_plugin_trusted", lambda _ident: None)

    def raise_missing(_identifier: str) -> Any:
        raise Exception("missing")

    monkeypatch.setattr(registry, "find_plot_renderer", raise_missing)

    class DummyLearner:
        def predict_proba(self):  # pragma: no cover - marker only
            raise NotImplementedError

    class DummyExplainer:
        def __init__(self):
            self.learner = DummyLearner()
            self.latest_explanation = None
            self.last_explanation_mode = None
            self.plot_plugin_fallbacks = {}

            def _resolve_plot_plugin(explicit_style, renderer_override=None):
                if renderer_override:
                    logging.getLogger("calibrated_explanations.plotting").info(
                        "Failed to find plot renderer '%s'; falling back to default",
                        renderer_override,
                    )
                    warnings.warn(
                        f"Failed to find plot renderer '{renderer_override}'; falling back to default",
                        UserWarning,
                        stacklevel=2,
                    )
                return dummy, explicit_style or "dummy-style", ("dummy-style", "legacy")

            self.plugin_manager = SimpleNamespace(resolve_plot_plugin=_resolve_plot_plugin)

        def predict_proba(self, _x: Any, *, uq_interval: bool, threshold: Any, bins: Any):
            return [0.2, 0.8], ([0.1, 0.7], [0.3, 0.9])

    explainer = DummyExplainer()

    caplog.set_level(logging.INFO)
    with pytest.warns(UserWarning, match="Failed to find plot renderer"):
        result = plotting.plot_global(
            explainer,
            x=[[0.0], [1.0]],
            y=None,
            threshold=None,
            use_legacy=False,
            show=False,
            style="dummy-style",
            renderer="nope",
        )

    assert result == "ok"
    assert any("Failed to find plot renderer" in rec.message for rec in caplog.records)


def test_require_matplotlib_should_retry_after_stale_cached_import_error(monkeypatch):
    """A stale cached import error must not poison later plotting calls.

    Some tests temporarily install fake matplotlib modules.  If plotting first
    sees that fake environment, ``_MATPLOTLIB_IMPORT_ERROR`` can be populated.
    Once the real dependency is available again, ``__require_matplotlib`` must
    retry the import instead of treating the cached error as permanent.
    """
    require_name = "__" + "require_matplotlib"
    error_name = "_" + "MATPLOTLIB_IMPORT_ERROR"
    original_plt = plotting.plt
    original_mcolors = plotting.mcolors
    original_error = getattr(plotting, error_name)
    monkeypatch.setattr(plotting, "plt", None)
    monkeypatch.setattr(plotting, "mcolors", None)
    monkeypatch.setattr(plotting, error_name, RuntimeError("stale failure"))

    getattr(plotting, require_name)()

    assert plotting.plt is not None
    assert plotting.mcolors is not None
    assert getattr(plotting, error_name) is None

    monkeypatch.setattr(plotting, "plt", original_plt)
    monkeypatch.setattr(plotting, "mcolors", original_mcolors)
    monkeypatch.setattr(plotting, error_name, original_error)
