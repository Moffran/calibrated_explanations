from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import calibrated_explanations.plotting as plotting


def test_split_csv_coverage():
    assert plotting.split_csv(None) == ()
    assert plotting.split_csv("") == ()
    assert plotting.split_csv("a, b, c") == ("a", "b", "c")
    assert plotting.split_csv(["a ", " b"]) == ("a", "b")
    assert plotting.split_csv(123) == ()
    assert plotting.split_csv(" , , ") == ()


def test_split_csv_sequence_non_str():
    assert plotting.split_csv([1, 2, "a"]) == ("a",)


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


def test_plot_alternative__should_normalize_features_to_plot_when_mixed_inputs(monkeypatch):
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
        feature_predict=[0.1, 0.2, 0.3],
        features_to_plot=["0", -1, "bad", 2.7],
        num_to_show=5,
        column_names=None,
        title="T",
        path=None,
        show=False,
        save_ext=None,
        use_legacy=False,
        return_plot_spec=True,
    )

    assert captured["features_to_plot"] == [0, 2]
    assert captured["column_names"] == ["0", "1", "2"]


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


def test_plot_alternative__should_format_xlabel_for_thresholded_regression_scalar(monkeypatch):
    captured: dict[str, Any] = {}
    import calibrated_explanations.viz.builders as builders

    monkeypatch.setattr(
        builders, "build_alternative_probabilistic_spec", capture_builder_kwargs(captured)
    )
    monkeypatch.setattr(
        builders,
        "build_alternative_regression_spec",
        lambda **_kwargs: pytest.fail(
            "regression builder should not be used for thresholded regression"
        ),
    )

    explanation = DummyExplanation(mode="regression", thresholded=True, y_threshold=0.5)

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

    assert captured["xlabel"] == "Probability of target being below 0.50"
    assert captured["xlim"] == (0.0, 1.0)
    assert len(list(captured["xticks"])) == 11


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
        plugins, "find_plot_plugin_trusted", lambda ident: dummy if ident == "dummy-style" else None
    )
    monkeypatch.setattr(
        plugins, "find_plot_plugin", lambda ident: dummy if ident == "dummy-style" else None
    )

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
