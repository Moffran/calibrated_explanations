"""Extensive tests for the legacy plotting helpers.

These tests intentionally exercise a wide variety of branches in
``calibrated_explanations.legacy.plotting`` so that the module is well
covered even though it relies heavily on optional matplotlib features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pytest

matplotlib = pytest.importorskip(
    "matplotlib", reason="matplotlib is required for legacy plotting tests"
)


matplotlib.use("Agg", force=True)

from calibrated_explanations.legacy import plotting as legacy_plotting


@dataclass
class _FakeExplainer:
    """Simple explainer exposing the minimal API used by the plots."""

    is_multiclass_flag: bool = False

    def is_multiclass(self) -> bool:  # pragma: no cover - trivial passthrough
        return self.is_multiclass_flag


class _CalibrationEnvelope:
    def get_confidence(self) -> int:  # pragma: no cover - trivial passthrough
        return 90


class _FakeExplanation:
    """Light-weight explanation carrying attributes accessed by the plots."""

    def __init__(
        self,
        mode: str = "classification",
        *,
        thresholded: bool = False,
        y_threshold: float | tuple[float, float] = 0.5,
        class_labels: Sequence[str] | None = None,
        is_multiclass: bool = False,
    ) -> None:
        self._mode = mode
        self._thresholded = thresholded
        self.y_threshold = y_threshold
        self._class_labels = list(class_labels) if class_labels is not None else None
        self.y_minmax = (0.0, 1.0)
        self.prediction = {"classes": 1}
        self.is_multiclass = is_multiclass
        self._explainer = _FakeExplainer(is_multiclass_flag=is_multiclass)
        self.calibrated_explanations = _CalibrationEnvelope()

    # The legacy plotting code still calls a number of "private" helpers.
    def _get_explainer(self) -> _FakeExplainer:  # pragma: no cover - passthrough
        return self._explainer

    def get_mode(self) -> str:  # pragma: no cover - passthrough
        return self._mode

    def get_class_labels(self) -> Sequence[str] | None:  # pragma: no cover
        return self._class_labels

    def is_thresholded(self) -> bool:  # pragma: no cover - passthrough
        return self._thresholded

    def is_one_sided(self) -> bool:  # pragma: no cover - constant behaviour
        return False


class _FakeNonProbExplainer:
    """Explainer without ``predict_proba`` to drive the non-probabilistic path."""

    def __init__(self) -> None:
        self.learner = object()
        self.y_cal = np.array([0.1, 0.2, 0.3])

    def predict(self, x, uq_interval=False, **kwargs):  # pragma: no cover - passthrough
        size = len(x)
        preds = np.linspace(0.2, 0.8, size)
        low = preds - 0.1
        high = preds + 0.1
        return preds, (low, high)

    def is_multiclass(self) -> bool:  # pragma: no cover - deterministic
        return False


class _FakeProbExplainer:
    """Explainer exposing ``predict_proba`` to trigger the probabilistic branch."""

    def __init__(self, *, classes: Iterable[int] = (0, 1)) -> None:
        self.learner = self
        self.class_labels = {i: c for i, c in enumerate(classes)}

    def predict_proba(self, x, uq_interval=False, threshold=None, **kwargs):
        proba = np.full((len(x), len(self.class_labels)), 0.5)
        low = np.zeros_like(proba)
        high = np.ones_like(proba)
        return proba, (low, high)

    def is_multiclass(self) -> bool:  # pragma: no cover - deterministic
        return len(self.class_labels) > 2

    def predict(self, x, uq_interval=False, **kwargs):  # pragma: no cover - passthrough
        preds = np.linspace(0.1, 0.9, len(x))
        low = preds - 0.05
        high = preds + 0.05
        return preds, (low, high)


def _string_path(path) -> str:
    """Return a string path with trailing separator suitable for plotting helpers."""

    return str(path) + "/"


def test_plot_probabilistic_rejects_mismatched_lengths():
    """Feature/instance size mismatches should be rejected to avoid mis-rendering."""

    instance = np.array([0.1])
    predict = {"predict": 0.5, "low": 0.2, "high": 0.7}
    feature_weights = np.array([0.3, -0.2])

    with pytest.raises(IndexError):
        legacy_plotting._plot_probabilistic(
            explanation=_FakeExplanation(),
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
    monkeypatch.setattr(legacy_plotting, "__require_matplotlib", lambda: (_ for _ in ()).throw(RuntimeError("should not import")))

    legacy_plotting._plot_probabilistic(
        explanation=_FakeExplanation(),
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


def test_require_matplotlib_raises_when_import_failed(monkeypatch):
    """``__require_matplotlib`` should surface a readable error if ``plt`` is missing."""

    monkeypatch.setattr(legacy_plotting, "plt", None)
    monkeypatch.setattr(
        legacy_plotting,
        "_MATPLOTLIB_IMPORT_ERROR",
        ImportError("backend failed"),
    )

    with pytest.raises(RuntimeError) as excinfo:
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
    explanation = _FakeExplanation(
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
        path=_string_path(path),
        show=False,
        interval=True,
        idx=0,
        save_ext=[".png"],
    )
    assert (path / "thresholded.png").exists()

    # Multiclass without explicit labels should use the explainer to derive names.
    explanation = _FakeExplanation(
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
        path=_string_path(path),
        show=False,
        interval=False,
        save_ext=[".svg"],
    )
    assert (path / "explainer_labels.svg").exists()

    # Explicit multiclass labels exercise the dedicated branch.
    explanation = _FakeExplanation(
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
        path=_string_path(path),
        show=False,
        interval=True,
        idx=1,
        save_ext=[".pdf"],
    )
    assert (path / "multiclass.pdf").exists()

    # Binary labels (not multiclass) should fall back to the final else branch.
    explanation = _FakeExplanation(
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
        path=_string_path(path),
        show=False,
        interval=False,
        save_ext=[".png"],
    )
    assert (path / "binary.png").exists()

    legacy_plotting.plt.close("all")


def test_plot_regression_interval_and_non_interval(tmp_path):
    path = tmp_path / "regression"
    path.mkdir()
    instance = np.array([0.5, 0.1])
    columns = ["a", "b"]
    features = [0, 1]
    explanation = _FakeExplanation(mode="regression")
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
        path=_string_path(path),
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
        path=_string_path(path),
        show=False,
        interval=False,
        save_ext=[".pdf"],
    )
    assert (path / "point.pdf").exists()

    legacy_plotting.plt.close("all")


def test_plot_triangular_for_classification_and_regression(tmp_path):
    path = tmp_path / "triangular"
    path.mkdir()
    explanation = _FakeExplanation(mode="classification")
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
        path=_string_path(path),
        show=False,
        save_ext=[".png"],
    )
    assert (path / "class.png").exists()

    regression = _FakeExplanation(mode="regression")
    legacy_plotting._plot_triangular(
        explanation=regression,
        proba=0.4,
        uncertainty=0.05,
        rule_proba=np.array([0.2, 0.25, 0.3]),
        rule_uncertainty=np.array([0.02, 0.05, 0.03]),
        num_to_show=3,
        title="reg",
        path=_string_path(path),
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

    explanation = _FakeExplanation(mode="regression")
    legacy_plotting._plot_alternative(
        explanation=explanation,
        instance=instance,
        predict=predict,
        feature_predict=feature_predict,
        features_to_plot=features,
        num_to_show=len(features),
        column_names=["x", "y", "z"],
        title="regression",
        path=_string_path(path),
        show=True,
        save_ext=[".png"],
    )
    assert (path / "regression.png").exists()

    thresholded = _FakeExplanation(
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
        path=_string_path(path),
        show=False,
        save_ext=[".pdf"],
    )
    assert (path / "thresholded.pdf").exists()

    legacy_plotting.plt.close("all")


def test_plot_probabilistic_errors_for_misaligned_instance():
    """Ensure ``_plot_probabilistic`` rejects instances shorter than features."""

    explanation = _FakeExplanation()
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

    explanation = _FakeExplanation()
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

    explanation = _FakeExplanation()

    def _fail():  # pragma: no cover - only used when guard regresses
        raise AssertionError("matplotlib should not be required for headless no-op")

    monkeypatch.setattr(legacy_plotting, "plt", None)
    monkeypatch.setattr(legacy_plotting, "__require_matplotlib", _fail)

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
    non_prob = _FakeNonProbExplainer()
    legacy_plotting._plot_global(
        non_prob,
        x_values,
        y=None,
        threshold=None,
        show=False,
    )

    # Probabilistic explainer with labels to exercise legend creation.
    prob = _FakeProbExplainer(classes=(0, 1, 2))
    y = np.array([0, 1, 2, 1, 0])
    legacy_plotting._plot_global(
        prob,
        x_values,
        y=y,
        threshold=None,
        show=False,
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

    shade = legacy_plotting.__get_fill_color({"predict": 0.8}, reduction=0.5)
    # The helper returns an HTML colour string.
    assert shade.startswith("#") and len(shade) == 7


def test_plot_probabilistic_raises_when_feature_lengths_mismatch(tmp_path):
    """Mismatched feature/index sizes should not silently succeed."""

    explanation = _FakeExplanation()
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
            path=_string_path(tmp_path),
            show=False,
            interval=True,
            idx=0,
            save_ext=[".png"],
        )


def test_plot_probabilistic_short_circuits_without_show_or_save(monkeypatch):
    """Headless mode should exit before importing matplotlib."""

    explanation = _FakeExplanation()
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

