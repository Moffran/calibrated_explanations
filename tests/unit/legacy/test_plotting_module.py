"""Extensive tests for the legacy plotting helpers.

These tests intentionally exercise a wide variety of branches in
``calibrated_explanations.legacy.plotting`` so that the module is well
covered even though it relies heavily on optional matplotlib features.
"""

from __future__ import annotations


import numpy as np
import pytest

from calibrated_explanations.legacy import plotting as legacy_plotting

from tests.helpers.explainer_utils import (
    FakeExplanation,
    FakeNonProbExplainer,
    FakeProbExplainer,
)


matplotlib = pytest.importorskip(
    "matplotlib",
    reason="matplotlib is required for legacy plotting tests",
)


matplotlib.use("Agg", force=True)


def string_path(path) -> str:
    """Return a string path with trailing separator suitable for plotting helpers."""

    return str(path) + "/"


def test_plot_probabilistic_rejects_mismatched_lengths():
    """Feature/instance size mismatches should be rejected to avoid mis-rendering."""

    instance = np.array([0.1])
    predict = {"predict": 0.5, "low": 0.2, "high": 0.7}
    feature_weights = np.array([0.3, -0.2])

    with pytest.raises(IndexError):
        legacy_plotting._plot_probabilistic(
            explanation=FakeExplanation(),
            instance=instance,
            predict=predict,
            feature_weights=feature_weights,
            features_to_plot=[0, 1],
            num_to_show=2,
            column_names=["f0", "f1"],
            title="mismatch",
            path="",  # unused because an exception is expected
            show=True,
            interval=False,
            save_ext=[".png"],
        )


def test_plot_probabilistic_headless_short_circuit(monkeypatch):
    """Headless invocations without save metadata must return without importing mpl."""

    monkeypatch.setattr(legacy_plotting, "plt", None)
    monkeypatch.setattr(legacy_plotting, "_MATPLOTLIB_IMPORT_ERROR", ImportError("backend missing"))
    monkeypatch.setattr(
        legacy_plotting,
        "__require_matplotlib",
        lambda: (_ for _ in ()).throw(RuntimeError("should not import")),
    )

    legacy_plotting._plot_probabilistic(
        explanation=FakeExplanation(),
        instance=np.array([]),
        predict={"predict": 0.5, "low": 0.2, "high": 0.7},
        feature_weights=np.array([]),
        features_to_plot=[],
        num_to_show=0,
        column_names=None,
        title=None,
        path=None,
        show=False,
        interval=False,
        save_ext=None,
    )


def test_plot_probabilistic_defaults_save_extensions(monkeypatch, tmp_path):
    """Default save extensions should be emitted in svg/pdf/png order with title concatenation."""

    saved: list[str] = []

    from matplotlib.figure import Figure as MplFigure

    def capture(self, filename, **kwargs):  # pragma: no cover - trivial hook
        saved.append(filename)

    monkeypatch.setattr(MplFigure, "savefig", capture, raising=False)

    legacy_plotting._plot_probabilistic(
        explanation=FakeExplanation(),
        instance=np.array([]),
        predict={"predict": 0.5, "low": 0.2, "high": 0.7},
        feature_weights=np.array([]),
        features_to_plot=[],
        num_to_show=0,
        column_names=None,
        title="default",
        path=string_path(tmp_path),
        show=False,
        interval=False,
        save_ext=None,
    )

    assert saved == [
        str(tmp_path / "defaultsvg"),
        str(tmp_path / "defaultpdf"),
        str(tmp_path / "defaultpng"),
    ]
    legacy_plotting.plt.close("all")


def test_plot_probabilistic_interval_requires_idx():
    """Interval rendering without ``idx`` should raise to enforce ADR guardrails."""

    with pytest.raises(AssertionError):
        legacy_plotting._plot_probabilistic(
            explanation=FakeExplanation(),
            instance=np.array([0.1]),
            predict={"predict": 0.5, "low": 0.2, "high": 0.8},
            feature_weights={
                "predict": np.array([0.1]),
                "low": np.array([0.0]),
                "high": np.array([0.2]),
            },
            features_to_plot=[0],
            num_to_show=1,
            column_names=["feat"],
            title="needs-idx",
            path="",
            show=True,
            interval=True,
            idx=None,
            save_ext=[],
        )


def test_require_matplotlib_raises_when_import_failed(monkeypatch):
    """``__require_matplotlib`` should surface a readable error if ``plt`` is missing."""
    from calibrated_explanations.core.exceptions import ConfigurationError

    monkeypatch.setattr(legacy_plotting, "plt", None)
    monkeypatch.setattr(
        legacy_plotting,
        "_MATPLOTLIB_IMPORT_ERROR",
        ImportError("backend failed"),
    )

    with pytest.raises(ConfigurationError) as excinfo:
        legacy_plotting.__require_matplotlib()

    assert "backend failed" in str(excinfo.value)


def test_plot_probabilistic_covers_threshold_and_label_branches(tmp_path):
    """Exercise the major branches in ``_plot_probabilistic``."""

    path = tmp_path / "probabilistic"
    path.mkdir()
    instance = np.array([0.1, 0.2, 0.3])
    base_predict = {"predict": 0.6, "low": -np.inf, "high": np.inf}
    interval_weights = {
        "predict": np.array([0.3, -0.4, 0.1]),
        "low": np.array([0.2, -0.5, 0.05]),
        "high": np.array([0.4, -0.3, 0.15]),
    }
    features = [0, 1, 2]
    columns = ["f0", "f1", "f2"]

    # Thresholded regression hits the first label branch.
    explanation = FakeExplanation(
        mode="regression",
        thresholded=True,
        y_threshold=0.25,
    )
    legacy_plotting._plot_probabilistic(
        explanation=explanation,
        instance=instance,
        predict=base_predict,
        feature_weights=interval_weights,
        features_to_plot=features,
        num_to_show=len(features),
        column_names=columns,
        title="thresholded",
        path=string_path(path),
        show=False,
        interval=True,
        idx=0,
        save_ext=[".png"],
    )
    assert (path / "thresholded.png").exists()

    # Multiclass without explicit labels should use the explainer to derive names.
    explanation = FakeExplanation(
        class_labels=None,
        is_multiclass=True,
    )
    explanation.prediction = {"classes": 2}
    explanation.is_multiclass = False  # toggle branch using _get_explainer().is_multiclass
    legacy_plotting._plot_probabilistic(
        explanation=explanation,
        instance=instance,
        predict={**base_predict, "predict": 0.4},
        feature_weights=np.array([0.2, -0.1, 0.05]),
        features_to_plot=features,
        num_to_show=len(features),
        column_names=None,
        title="explainer_labels",
        path=string_path(path),
        show=False,
        interval=False,
        save_ext=[".svg"],
    )
    assert (path / "explainer_labels.svg").exists()

    # Explicit multiclass labels exercise the dedicated branch.
    explanation = FakeExplanation(
        class_labels=["neg", "pos", "maybe"],
        is_multiclass=True,
    )
    explanation.prediction = {"classes": 1}
    legacy_plotting._plot_probabilistic(
        explanation=explanation,
        instance=instance,
        predict=base_predict,
        feature_weights=interval_weights,
        features_to_plot=features,
        num_to_show=len(features),
        column_names=columns,
        title="multiclass",
        path=string_path(path),
        show=False,
        interval=True,
        idx=1,
        save_ext=[".pdf"],
    )
    assert (path / "multiclass.pdf").exists()

    # Binary labels (not multiclass) should fall back to the final else branch.
    explanation = FakeExplanation(
        class_labels=["no", "yes"],
        is_multiclass=False,
    )
    explanation.prediction = {"classes": 1}
    legacy_plotting._plot_probabilistic(
        explanation=explanation,
        instance=instance,
        predict=base_predict,
        feature_weights=np.array([0.5, -0.2, 0.0]),
        features_to_plot=features,
        num_to_show=len(features),
        column_names=columns,
        title="binary",
        path=string_path(path),
        show=False,
        interval=False,
        save_ext=[".png"],
    )
    assert (path / "binary.png").exists()

    legacy_plotting.plt.close("all")


def test_plot_probabilistic_saves_before_show(monkeypatch, tmp_path):
    """``_plot_probabilistic`` must save files before displaying the figure."""

    instance = np.array([0.1])
    feature_weights = np.array([0.2])
    predict = {"predict": 0.5, "low": 0.3, "high": 0.7}

    calls: list[str] = []

    import matplotlib.figure

    def fake_savefig(self, path, **_):  # pragma: no cover - minimal spy
        calls.append(f"save:{path}")

    def fake_show(self):  # pragma: no cover - minimal spy
        calls.append("show")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake_savefig, raising=False)
    monkeypatch.setattr(matplotlib.figure.Figure, "show", fake_show, raising=False)

    legacy_plotting._plot_probabilistic(
        explanation=FakeExplanation(),
        instance=instance,
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=[0],
        num_to_show=1,
        column_names=["f0"],
        title="order",
        path=str(tmp_path / "order"),
        show=True,
        interval=False,
        idx=None,
        save_ext=[".png"],
    )

    assert calls[0].startswith("save:")
    assert calls[-1] == "show"


def test_plot_probabilistic_figsize_scales_with_num_to_show(monkeypatch, tmp_path):
    """Figure height should grow with ``num_to_show`` as documented in ADR-025."""

    seen: list[tuple[float, float] | None] = []

    real_figure = legacy_plotting.plt.figure

    def spy_figure(*args, **kwargs):
        seen.append(kwargs.get("figsize"))
        return real_figure(*args, **kwargs)

    monkeypatch.setattr(legacy_plotting.plt, "figure", spy_figure)

    instance = np.linspace(0.0, 1.0, 6)
    feature_weights = np.linspace(0.2, -0.1, 6)
    predict = {"predict": 0.4, "low": 0.3, "high": 0.5}

    legacy_plotting._plot_probabilistic(
        explanation=FakeExplanation(),
        instance=instance,
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=list(range(6)),
        num_to_show=6,
        column_names=[f"f{i}" for i in range(6)],
        title="figsize",
        path=str(tmp_path / "figsize"),
        show=False,
        interval=False,
        idx=None,
        save_ext=[".png"],
    )

    assert seen and seen[0] == (10, 6 * 0.5 + 2)
    legacy_plotting.plt.close("all")


def test_plot_regression_interval_and_non_interval(tmp_path):
    path = tmp_path / "regression"
    path.mkdir()
    instance = np.array([0.5, 0.1])
    columns = ["a", "b"]
    features = [0, 1]
    explanation = FakeExplanation(mode="regression")
    predict = {"predict": 0.25, "low": 0.2, "high": 0.35}
    interval_weights = {
        "predict": np.array([0.2, -0.3]),
        "low": np.array([0.1, -0.4]),
        "high": np.array([0.3, -0.2]),
    }

    legacy_plotting._plot_regression(
        explanation=explanation,
        instance=instance,
        predict=predict,
        feature_weights=interval_weights,
        features_to_plot=features,
        num_to_show=len(features),
        column_names=columns,
        title="interval",
        path=string_path(path),
        show=False,
        interval=True,
        idx=0,
        save_ext=[".png"],
    )
    assert (path / "interval.png").exists()

    legacy_plotting._plot_regression(
        explanation=explanation,
        instance=instance,
        predict=predict,
        feature_weights=np.array([0.4, -0.1]),
        features_to_plot=features,
        num_to_show=len(features),
        column_names=None,
        title="point",
        path=string_path(path),
        show=False,
        interval=False,
        save_ext=[".pdf"],
    )
    assert (path / "point.pdf").exists()

    legacy_plotting.plt.close("all")


def test_plot_regression_interval_requires_index():
    """Regression interval rendering must assert a provided index."""

    instance = np.array([0.2, 0.3])
    predict = {"predict": 0.5, "low": 0.4, "high": 0.6}
    weights = {
        "predict": np.array([0.1, -0.2]),
        "low": np.array([0.05, -0.3]),
        "high": np.array([0.2, -0.1]),
    }

    with pytest.raises(AssertionError):
        legacy_plotting._plot_regression(
            explanation=FakeExplanation(mode="regression"),
            instance=instance,
            predict=predict,
            feature_weights=weights,
            features_to_plot=[0, 1],
            num_to_show=2,
            column_names=["a", "b"],
            title="missing-idx",
            path="/tmp/",  # any non-None path avoids the headless guard
            show=False,
            interval=True,
            idx=None,
            save_ext=[".png"],
        )


def test_plot_triangular_for_classification_and_regression(tmp_path):
    path = tmp_path / "triangular"
    path.mkdir()
    explanation = FakeExplanation(mode="classification")
    rule_proba = np.array([0.6, 0.2, 0.9])
    rule_uncertainty = np.array([0.1, 0.3, 0.2])

    legacy_plotting._plot_triangular(
        explanation=explanation,
        proba=0.7,
        uncertainty=0.15,
        rule_proba=rule_proba,
        rule_uncertainty=rule_uncertainty,
        num_to_show=len(rule_proba),
        title="class",
        path=string_path(path),
        show=False,
        save_ext=[".png"],
    )
    assert (path / "class.png").exists()

    regression = FakeExplanation(mode="regression")
    legacy_plotting._plot_triangular(
        explanation=regression,
        proba=0.4,
        uncertainty=0.05,
        rule_proba=np.array([0.2, 0.25, 0.3]),
        rule_uncertainty=np.array([0.02, 0.05, 0.03]),
        num_to_show=3,
        title="reg",
        path=string_path(path),
        show=False,
        save_ext=[".svg"],
    )
    assert (path / "reg.svg").exists()

    legacy_plotting.plt.close("all")


def test_plot_alternative_covers_probability_and_threshold(tmp_path):
    path = tmp_path / "alternative"
    path.mkdir()
    instance = np.array([1.0, 2.0, 3.0])
    features = [0, 1, 2]
    feature_predict = {
        "predict": np.array([0.2, 0.8, 0.55]),
        "low": np.array([0.1, 0.7, 0.5]),
        "high": np.array([0.3, 0.9, 0.6]),
    }
    predict = {"predict": 0.6, "low": 0.55, "high": 0.65}

    explanation = FakeExplanation(mode="regression")
    legacy_plotting._plot_alternative(
        explanation=explanation,
        instance=instance,
        predict=predict,
        feature_predict=feature_predict,
        features_to_plot=features,
        num_to_show=len(features),
        column_names=["x", "y", "z"],
        title="regression",
        path=string_path(path),
        show=True,
        save_ext=[".png"],
    )
    assert (path / "regression.png").exists()

    thresholded = FakeExplanation(
        mode="classification",
        thresholded=True,
        y_threshold=(0.2, 0.8),
        class_labels=["neg", "pos"],
    )
    thresholded.prediction = {"classes": 1}
    legacy_plotting._plot_alternative(
        explanation=thresholded,
        instance=instance,
        predict=predict,
        feature_predict=feature_predict,
        features_to_plot=features,
        num_to_show=len(features),
        column_names=["x", "y", "z"],
        title="thresholded",
        path=string_path(path),
        show=False,
        save_ext=[".pdf"],
    )
    assert (path / "thresholded.pdf").exists()

    legacy_plotting.plt.close("all")


def test_plot_probabilistic_errors_for_misaligned_instance():
    """Ensure ``_plot_probabilistic`` rejects instances shorter than features."""

    explanation = FakeExplanation()
    instance = np.array([0.1])
    feature_weights = np.array([0.2, -0.1])

    with pytest.raises(IndexError):
        legacy_plotting._plot_probabilistic(
            explanation=explanation,
            instance=instance,
            predict={"predict": 0.5, "low": 0.3, "high": 0.7},
            feature_weights=feature_weights,
            features_to_plot=[0, 1],
            num_to_show=2,
            column_names=["f0", "f1"],
            title="misaligned_instance",
            path="",
            show=False,
            interval=False,
            save_ext=[".png"],
        )


def test_plot_probabilistic_errors_for_num_to_show_mismatch():
    """``_plot_probabilistic`` should fail when ``num_to_show`` exceeds feature count."""

    explanation = FakeExplanation()
    instance = np.array([0.1, 0.2])
    feature_weights = np.array([0.2, -0.1])

    with pytest.raises(ValueError) as excinfo:
        legacy_plotting._plot_probabilistic(
            explanation=explanation,
            instance=instance,
            predict={"predict": 0.5, "low": 0.3, "high": 0.7},
            feature_weights=feature_weights,
            features_to_plot=[0, 1],
            num_to_show=3,
            column_names=["f0", "f1"],
            title="num_to_show_mismatch",
            path="",
            show=False,
            interval=False,
            save_ext=[".png"],
        )

    assert "FixedLocator" in str(excinfo.value)


def test_plot_probabilistic_headless_noop_without_save_metadata(monkeypatch):
    """When show/save are disabled the helper should short-circuit without matplotlib."""

    explanation = FakeExplanation()

    def fail():  # pragma: no cover - only used when guard regresses
        raise AssertionError("matplotlib should not be required for headless no-op")

    monkeypatch.setattr(legacy_plotting, "plt", None)
    monkeypatch.setattr(legacy_plotting, "__require_matplotlib", fail)

    legacy_plotting._plot_probabilistic(
        explanation=explanation,
        instance=np.array([0.1]),
        predict={"predict": 0.5, "low": 0.3, "high": 0.7},
        feature_weights=np.array([0.2]),
        features_to_plot=[0],
        num_to_show=1,
        column_names=["f0"],
        title=None,
        path=None,
        show=False,
        interval=False,
        save_ext=None,
    )


def test_plot_global_for_probabilistic_and_non_probabilistic(tmp_path):
    x_values = np.zeros((5, 2))

    # Non probabilistic branch should honour early return when show=False and plt present.
    non_prob = FakeNonProbExplainer()
    legacy_plotting._plot_global(
        non_prob,
        x_values,
        y=None,
        threshold=None,
        show=False,
    )

    # Probabilistic explainer with labels to exercise legend creation.
    prob = FakeProbExplainer(classes=(0, 1, 2))
    y = np.array([0, 1, 2, 1, 0])
    legacy_plotting._plot_global(
        prob,
        x_values,
        y=y,
        threshold=None,
        show=False,
    )

    legacy_plotting.plt.close("all")


def test_plot_global_headless_noop(monkeypatch):
    """When headless the global helper should not attempt to import matplotlib."""

    monkeypatch.setattr(legacy_plotting, "plt", None)
    monkeypatch.setattr(
        legacy_plotting,
        "__require_matplotlib",
        lambda: (_ for _ in ()).throw(RuntimeError("should not import")),
    )

    legacy_plotting._plot_global(FakeNonProbExplainer(), np.zeros((2, 1)), show=False)


def test_plot_global_requires_scalar_threshold_for_non_probabilistic():
    """Non-probabilistic explainers must receive a scalar threshold."""

    class ThresholdExplainer:
        def __init__(self):
            self.learner = object()
            self.class_labels = None
            self.y_cal = np.array([0.2, 0.6])

        def predict(self, x, uq_interval=True, **kwargs):  # pragma: no cover - unused path
            preds = np.linspace(0.3, 0.7, len(x))
            low = preds - 0.1
            high = preds + 0.1
            return preds, (low, high)

        def predict_proba(self, x, uq_interval=True, threshold=None, **kwargs):
            preds = np.linspace(0.2, 0.8, len(x))
            proba = np.column_stack([1 - preds, preds])
            low = np.zeros_like(proba)
            high = np.ones_like(proba)
            return proba, (low, high)

        def is_multiclass(self):  # pragma: no cover - deterministic helper
            return False

    explainer = ThresholdExplainer()
    x_vals = np.zeros((3, 1))
    y_vals = np.array([0.1, 0.3, 0.5])

    with pytest.warns(RuntimeWarning), pytest.raises(AssertionError):
        legacy_plotting._plot_global(
            explainer,
            x_vals,
            y=y_vals,
            threshold=(0.2, 0.8),
            show=True,
        )

    legacy_plotting.plt.close("all")


def test_plot_global_headless_short_circuit(monkeypatch):
    """Global helper should avoid importing matplotlib when nothing is rendered."""

    monkeypatch.setattr(legacy_plotting, "plt", None)

    def fail():  # pragma: no cover - ensures guard triggers
        raise AssertionError("matplotlib import guard should short-circuit")

    monkeypatch.setattr(legacy_plotting, "__require_matplotlib", fail)

    explainer = FakeProbExplainer()
    legacy_plotting._plot_global(explainer, np.zeros((1, 1)), show=False)


def test_plot_global_requires_scalar_threshold():
    """Non-probabilistic explainers must provide a scalar threshold."""

    class GuardExplainer(FakeNonProbExplainer):
        def predict_proba(
            self, x, uq_interval=True, threshold=None, **kwargs
        ):  # pragma: no cover - simple shim
            preds, (low, high) = self.predict(x, uq_interval=uq_interval, **kwargs)
            stacked = np.column_stack([preds, preds])
            lower = np.column_stack([low, low])
            upper = np.column_stack([high, high])
            return stacked, (lower, upper)

    explainer = GuardExplainer()
    x_values = np.zeros((3, 1))
    y = np.array([0.1, 0.2, 0.3])

    with pytest.warns(RuntimeWarning), pytest.raises(AssertionError):
        legacy_plotting._plot_global(
            explainer,
            x_values,
            y=y,
            threshold=(0.2, 0.5),
        )

    legacy_plotting.plt.close("all")


def test_plot_proba_triangle_returns_figure():
    legacy_plotting._plot_proba_triangle()
    assert legacy_plotting.plt.get_fignums(), "Expected matplotlib figure to be created"

    legacy_plotting.plt.close("all")


def test_color_brew_and_fill_color_behaviour():
    colors = legacy_plotting.__color_brew(3)
    assert len(colors) == 3
    assert all(len(rgb) == 3 for rgb in colors)

    palette = legacy_plotting.__color_brew(2)
    high_color = palette[1]
    alpha = 0.5  # reduction overrides computed alpha
    expected_high = "#%02x%02x%02x" % tuple(
        int(round(alpha * channel + (1 - alpha) * 255, 0)) for channel in high_color
    )
    shade = legacy_plotting.__get_fill_color({"predict": 0.8}, reduction=0.5)
    assert shade == expected_high

    low_color = palette[0]
    # when predict < 0.5 the helper mirrors the palette and keeps computed alpha
    computed_alpha = ((1 - 0.2) - 0.5) / 0.5 * (1 - 0.25) + 0.25
    expected_low = "#%02x%02x%02x" % tuple(
        int(round(computed_alpha * channel + (1 - computed_alpha) * 255, 0))
        for channel in low_color
    )
    assert legacy_plotting.__get_fill_color({"predict": 0.2}) == expected_low


def test_plot_probabilistic_raises_when_feature_lengths_mismatch(tmp_path):
    """Mismatched feature/index sizes should not silently succeed."""

    explanation = FakeExplanation()
    instance = np.array([0.2, 0.4])
    predict = {"predict": 0.6, "low": 0.2, "high": 0.8}
    feature_weights = {
        "predict": np.array([0.1, 0.2, 0.3]),
        "low": np.array([0.0, 0.1, 0.2]),
        "high": np.array([0.2, 0.3, 0.4]),
    }

    with pytest.raises(IndexError):
        legacy_plotting._plot_probabilistic(
            explanation=explanation,
            instance=instance,
            predict=predict,
            feature_weights=feature_weights,
            features_to_plot=[0, 1, 2],
            num_to_show=2,
            column_names=["a", "b", "c"],
            title="mismatch",
            path=string_path(tmp_path),
            show=False,
            interval=True,
            idx=0,
            save_ext=[".png"],
        )


def test_plot_probabilistic_short_circuits_without_show_or_save(monkeypatch):
    """Headless mode should exit before importing matplotlib."""

    explanation = FakeExplanation()
    instance = np.array([0.1])
    predict = {"predict": 0.2, "low": 0.1, "high": 0.3}
    feature_weights = np.array([0.05])

    def boom():  # pragma: no cover - ensures the guard fires before import
        raise AssertionError("matplotlib should not be required")

    monkeypatch.setattr(legacy_plotting, "__require_matplotlib", boom)

    # With no show/save metadata, the helper should be a no-op.
    legacy_plotting._plot_probabilistic(
        explanation=explanation,
        instance=instance,
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=[0],
        num_to_show=1,
        column_names=["f0"],
        title=None,
        path=None,
        show=False,
        interval=False,
        idx=None,
        save_ext=None,
    )
