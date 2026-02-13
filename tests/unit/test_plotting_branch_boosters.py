from __future__ import annotations

import types

import numpy as np
import pytest

import calibrated_explanations.plotting as plotting
import calibrated_explanations.viz.builders as viz_builders
import calibrated_explanations.viz.matplotlib_adapter as viz_adapter


class ExplanationStub:
    def __init__(
        self,
        *,
        mode: str,
        thresholded: bool,
        y_threshold,
        prediction,
        y_minmax,
        class_labels=None,
        explainer=None,
    ) -> None:
        self.mode = mode
        self.thresholded = thresholded
        self.y_threshold = y_threshold
        self.prediction = prediction
        self.y_minmax = y_minmax
        self.class_labels = class_labels
        self.explainer = explainer

    def get_mode(self):
        return self.mode

    def is_thresholded(self):
        return self.thresholded

    def get_class_labels(self):
        return self.class_labels

    def get_explainer(self):
        return self.explainer


def test_plot_triangular_legacy_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def legacy_plot_triangular(*args):
        calls.append(args)

    monkeypatch.setattr(
        "calibrated_explanations.legacy.plotting.plot_triangular",
        legacy_plot_triangular,
    )

    plotting.plot_triangular(
        explanation=object(),
        proba=np.asarray([0.5]),
        uncertainty=np.asarray([0.1]),
        rule_proba=np.asarray([0.6]),
        rule_uncertainty=np.asarray([0.2]),
        num_to_show=1,
        title="tri",
        path=None,
        show=False,
        save_ext=None,
        use_legacy=True,
    )

    assert len(calls) == 1


def test_plot_alternative_regression_thresholded_return_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    render_calls: list[tuple] = []

    monkeypatch.setattr(
        viz_builders,
        "build_alternative_probabilistic_spec",
        lambda **kwargs: {"plot_spec": kwargs},
    )
    monkeypatch.setattr(
        viz_builders,
        "build_alternative_regression_spec",
        lambda **kwargs: {"plot_spec": kwargs},
    )
    monkeypatch.setattr(
        viz_adapter,
        "render",
        lambda spec, show, save_path=None: render_calls.append((spec, show, save_path)),
    )

    explainer = types.SimpleNamespace(
        is_multiclass=lambda: False,
        get_confidence=lambda: 95,
    )
    explanation = ExplanationStub(
        mode="regression",
        thresholded=True,
        y_threshold=("bad", "value"),
        prediction={"predict": 0.1, "low": 0.0, "high": 1.0},
        y_minmax=("not-a-float", object()),
        class_labels=None,
        explainer=explainer,
    )

    spec = plotting.plot_alternative(
        explanation=explanation,
        instance=[1, 2],
        predict={"predict": "x", "low": None, "high": np.nan},
        feature_predict={"predict": [None, 0.3], "low": [None, 0.1], "high": [None, 0.5]},
        features_to_plot=["bad-index", -3, 0, 1],
        num_to_show=2,
        column_names=["f0", "f1"],
        title="alt",
        path=None,
        show=False,
        save_ext=[],
        return_plot_spec=True,
    )

    assert isinstance(spec, dict)
    assert len(render_calls) >= 1


def test_plot_alternative_classification_render_failure_falls_back_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls: list[tuple] = []

    monkeypatch.setattr(
        viz_builders,
        "build_alternative_probabilistic_spec",
        lambda **kwargs: {"plot_spec": kwargs},
    )
    monkeypatch.setattr(
        viz_builders,
        "build_alternative_regression_spec",
        lambda **kwargs: {"plot_spec": kwargs},
    )
    monkeypatch.setattr(
        viz_adapter,
        "render",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("render boom")),
    )
    monkeypatch.setattr(
        "calibrated_explanations.legacy.plotting.plot_alternative",
        lambda *args: legacy_calls.append(args),
    )

    explanation = ExplanationStub(
        mode="classification",
        thresholded=False,
        y_threshold=None,
        prediction={"classes": 1, "predict": 0.6, "low": 0.4, "high": 0.8},
        y_minmax=(0.0, 1.0),
        class_labels=("neg", "pos"),
        explainer=types.SimpleNamespace(is_multiclass=lambda: False),
    )

    with pytest.warns(Warning, match="Falling back to legacy plot"):
        plotting.plot_alternative(
            explanation=explanation,
            instance=[1, 2],
            predict={"predict": 0.7, "low": 0.5, "high": 0.9},
            feature_predict=[0.3, 0.2],
            features_to_plot=[0, 1],
            num_to_show=2,
            column_names=["a", "b"],
            title="fallback",
            path=None,
            show=False,
            save_ext=[],
            return_plot_spec=True,
        )

    assert len(legacy_calls) == 1
