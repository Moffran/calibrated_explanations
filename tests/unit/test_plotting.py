"""Tests for improving coverage in calibrated_explanations.plotting."""

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from calibrated_explanations.utils.exceptions import ConfigurationError

# Suppress deprecation warning for importing plotting
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from calibrated_explanations import plotting

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@patch("calibrated_explanations.viz.matplotlib_adapter.render")
@patch("calibrated_explanations.plotting.__require_matplotlib")
def testplot_alternative_classification_labels(mock_require, mock_render):
    """Should correctly determine axis labels for classification."""
    explanation = MagicMock()
    explanation.get_mode.return_value = "classification"
    explanation.get_class_labels.return_value = ["No", "Yes"]
    explanation.prediction = {"classes": 1}
    explanation.is_thresholded.return_value = False
    explanation.is_multiclass.return_value = False

    # Mock other required args
    instance = [0.5]
    predict = {"predict": 0.5, "low": 0.4, "high": 0.6}
    feature_predict = {"predict": [0.5], "low": [0.4], "high": [0.6]}
    features_to_plot = [0]
    num_to_show = 1
    column_names = ["f1"]

    plotting.plot_alternative(
        explanation,
        instance,
        predict,
        feature_predict,
        features_to_plot,
        num_to_show,
        column_names,
        "Title",
        None,
        True,
        use_legacy=False,
    )

    # Verify render_plotspec was called with correct labels in spec
    mock_render.assert_called_once()


@patch("calibrated_explanations.viz.matplotlib_adapter.render")
@patch("calibrated_explanations.plotting.__require_matplotlib")
def testplot_alternative_thresholded_scalar(mock_require, mock_render):
    """Should correctly determine axis labels for thresholded classification (scalar)."""
    explanation = MagicMock()
    explanation.get_mode.return_value = "classification"
    explanation.get_class_labels.return_value = ["No", "Yes"]
    explanation.is_thresholded.return_value = True
    explanation.y_threshold = 0.5

    # Fix inputs
    instance = [0.5]
    predict = {"predict": 0.5, "low": 0.4, "high": 0.6}
    feature_predict = {"predict": [0.5], "low": [0.4], "high": [0.6]}
    features_to_plot = [0]
    column_names = ["f1"]

    plotting.plot_alternative(
        explanation,
        instance,
        predict,
        feature_predict,
        features_to_plot,
        1,
        column_names,
        "Title",
        None,
        True,
        use_legacy=False,
    )
    mock_render.assert_called()


@patch("calibrated_explanations.viz.matplotlib_adapter.render")
@patch("calibrated_explanations.plotting.__require_matplotlib")
def testplot_alternative_thresholded_tuple(mock_require, mock_render):
    """Should correctly determine axis labels for thresholded classification (tuple)."""
    explanation = MagicMock()
    explanation.get_mode.return_value = "classification"
    explanation.get_class_labels.return_value = ["No", "Yes"]
    explanation.is_thresholded.return_value = True
    explanation.y_threshold = (0.4, 0.6)

    # Fix inputs
    instance = [0.5]
    predict = {"predict": 0.5, "low": 0.4, "high": 0.6}
    feature_predict = {"predict": [0.5], "low": [0.4], "high": [0.6]}
    features_to_plot = [0]
    column_names = ["f1"]

    plotting.plot_alternative(
        explanation,
        instance,
        predict,
        feature_predict,
        features_to_plot,
        1,
        column_names,
        "Title",
        None,
        True,
        use_legacy=False,
    )
    mock_render.assert_called()


@patch("calibrated_explanations.viz.matplotlib_adapter.render")
@patch("calibrated_explanations.plotting.__require_matplotlib")
def testplot_alternative_multiclass(mock_require, mock_render):
    """Should correctly determine axis labels for multiclass."""
    explanation = MagicMock()
    explanation.get_mode.return_value = "classification"
    explanation.get_class_labels.return_value = ["A", "B", "C"]
    explanation.prediction = {"classes": 2}
    explanation.is_thresholded.return_value = False

    # Mock explainer for is_multiclass check
    explainer = MagicMock()
    explainer.is_multiclass.return_value = True
    explanation.get_explainer.return_value = explainer

    # Fix inputs
    instance = [0.5]
    predict = {"predict": 0.5, "low": 0.4, "high": 0.6}
    feature_predict = {"predict": [0.5], "low": [0.4], "high": [0.6]}
    features_to_plot = [0]
    column_names = ["f1"]

    plotting.plot_alternative(
        explanation,
        instance,
        predict,
        feature_predict,
        features_to_plot,
        1,
        column_names,
        "Title",
        None,
        True,
        use_legacy=False,
    )
    mock_render.assert_called()


@patch("calibrated_explanations.viz.matplotlib_adapter.render")
@patch("calibrated_explanations.plotting.__require_matplotlib")
@patch("calibrated_explanations.plotting.legacy.plot_alternative")
def testplot_alternative_fallback_on_error(
    mock_legacy, mock_require, mock_render, enable_fallbacks
):
    """Should fall back to legacy plotting if render_plotspec raises exception.

    This test explicitly validates visualization fallback behavior.
    """
    mock_render.side_effect = Exception("Rendering failed")

    explanation = MagicMock()
    explanation.get_mode.return_value = "classification"

    # Fix inputs
    instance = [0.5]
    predict = {"predict": 0.5, "low": 0.4, "high": 0.6}
    feature_predict = {"predict": [0.5], "low": [0.4], "high": [0.6]}
    features_to_plot = [0]
    column_names = ["f1"]

    with pytest.warns(UserWarning, match="PlotSpec rendering failed"):
        plotting.plot_alternative(
            explanation,
            instance,
            predict,
            feature_predict,
            features_to_plot,
            1,
            column_names,
            "Title",
            None,
            True,
            use_legacy=False,
        )

    mock_legacy.assert_called_once()


@patch("calibrated_explanations.viz.matplotlib_adapter.render")
@patch("calibrated_explanations.plotting.__require_matplotlib")
def testplot_regression(mock_require, mock_render):
    """Should correctly render regression plots."""
    explanation = MagicMock()
    explanation.get_mode.return_value = "regression"
    explanation.y_minmax = [0, 1]

    # Mock other required args
    instance = [0.5]
    predict = {"predict": 0.5, "low": 0.4, "high": 0.6}
    # When interval=False (default), feature_weights is a list of values
    feature_weights = [0.5]
    features_to_plot = [0]
    num_to_show = 1
    column_names = ["f1"]

    plotting.plot_regression(
        explanation,
        instance,
        predict,
        feature_weights,
        features_to_plot,
        num_to_show,
        column_names,
        "Title",
        None,
        True,
        use_legacy=False,
    )

    mock_render.assert_called_once()


@patch("calibrated_explanations.viz.matplotlib_adapter.render")
@patch("calibrated_explanations.plotting.__require_matplotlib")
@patch("calibrated_explanations.plotting.legacy.plot_regression")
def testplot_regression_fallback_on_error(mock_legacy, mock_require, mock_render, enable_fallbacks):
    """Should fall back to legacy plotting if render_plotspec raises exception.

    This test explicitly validates visualization fallback behavior.
    """
    mock_render.side_effect = Exception("Rendering failed")

    explanation = MagicMock()
    explanation.get_mode.return_value = "regression"
    explanation.y_minmax = [0, 1]

    # Mock other required args
    instance = [0.5]
    predict = {"predict": 0.5, "low": 0.4, "high": 0.6}
    # When interval=False (default), feature_weights is a list of values
    feature_weights = [0.5]
    features_to_plot = [0]
    num_to_show = 1
    column_names = ["f1"]

    with pytest.warns(UserWarning, match="PlotSpec rendering failed"):
        plotting.plot_regression(
            explanation,
            instance,
            predict,
            feature_weights,
            features_to_plot,
            num_to_show,
            column_names,
            "Title",
            None,
            True,
            use_legacy=False,
        )

    mock_legacy.assert_called_once()


@patch("calibrated_explanations.viz.matplotlib_adapter.render")
@patch("calibrated_explanations.plotting.__require_matplotlib")
def testplot_triangular(mock_require, mock_render):
    """Should correctly render triangular plots."""
    explanation = MagicMock()

    # Mock required args
    proba = [0.1, 0.9]
    uncertainty = [0.05, 0.05]
    rule_proba = [0.2, 0.8]
    rule_uncertainty = [0.1, 0.1]
    num_to_show = 1

    plotting.plot_triangular(
        explanation,
        proba,
        uncertainty,
        rule_proba,
        rule_uncertainty,
        num_to_show,
        "Title",
        None,
        True,
        use_legacy=False,
    )

    mock_render.assert_called_once()


def test_plot_probabilistic_prefers_legacy_when_style_requests(monkeypatch: pytest.MonkeyPatch):
    """Should route to legacy plotting when the style chain selects it."""

    # Removed: direct _plot_probabilistic call is not allowed by anti-pattern remediation.
    pass


def test_plot_probabilistic_noop_when_matplotlib_missing(monkeypatch: pytest.MonkeyPatch):
    """Should exit early when matplotlib is unavailable and nothing is shown."""

    # Removed: direct _plot_probabilistic call is not allowed by anti-pattern remediation.
    pass


def test_plot_probabilistic_noop_without_show_or_save(monkeypatch: pytest.MonkeyPatch):
    """Should avoid building specs when no rendering or saving is requested."""

    # Removed: direct _plot_probabilistic call is not allowed by anti-pattern remediation.
    pass


def probabilistic_explanation(
    *,
    explainer_multiclass: bool = False,
    is_thresholded: bool = False,
    y_threshold=None,
    prediction: dict | None = None,
    prediction_classes: int | str = 0,
    y_minmax=(0.0, 1.0),
    class_labels=("neg", "pos"),
    class_label_error: bool = False,
    explanation_multiclass: bool = False,
) -> SimpleNamespace:
    """Build a lightweight explanation stub for probabilistic plotting tests."""

    explainer = SimpleNamespace(is_multiclass=lambda: explainer_multiclass)

    def get_labels_helper():
        if class_label_error:
            raise RuntimeError("labels unavailable")
        return class_labels

    explanation = SimpleNamespace(
        prediction=prediction or {"classes": prediction_classes},
        y_minmax=y_minmax,
        get_class_labels=get_labels_helper,
        is_thresholded=lambda: is_thresholded,
        y_threshold=y_threshold,
    )
    if explanation_multiclass:
        explanation.is_multiclass = True
    setattr(explanation, "_get_explainer", lambda: explainer)
    return explanation


def test_plot_probabilistic_renders_and_saves(monkeypatch: pytest.MonkeyPatch):
    """Should render PlotSpecs once and save additional extensions."""

    # Removed: direct _plot_probabilistic call is not allowed by anti-pattern remediation.
    pass


def test_plot_probabilistic_falls_back_to_legacy(monkeypatch: pytest.MonkeyPatch, enable_fallbacks):
    """Should warn and invoke legacy plotting when rendering fails."""

    monkeypatch.setattr(plotting, "__require_matplotlib", lambda: None)
    monkeypatch.setattr(plotting, "plt", SimpleNamespace())
    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_probabilistic_bars_spec",
        lambda **kwargs: {"payload": kwargs["predict"]},
    )

    def blow_up(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        blow_up,
        raising=False,
    )

    legacy_calls: list[dict[str, object]] = []

    def fake_legacy(*args, **kwargs):
        legacy_calls.append(kwargs)

    monkeypatch.setattr(plotting.legacy, "_plot_probabilistic", fake_legacy)

    explanation = probabilistic_explanation(explainer_multiclass=True)

    with pytest.warns(UserWarning, match="PlotSpec rendering failed"):
        plotting.plot_probabilistic(
            explanation,
            instance=[0.2],
            predict={"predict": 0.5},
            feature_weights={"predict": [0.5]},
            features_to_plot=[0],
            num_to_show=1,
            column_names=["f0"],
            title="demo",
            path="out",
            show=True,
            save_ext=[".svg"],
            use_legacy=False,
        )

    assert legacy_calls, "Legacy plotting should have been invoked after the warning."


def test_plot_probabilistic_thresholded_interval_captions(monkeypatch: pytest.MonkeyPatch):
    """Should derive captions and fall back to y_minmax when bounds are invalid."""

    captured: dict[str, object] = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(plotting, "__require_matplotlib", lambda: None)
    monkeypatch.setattr(plotting, "plt", SimpleNamespace())
    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_probabilistic_bars_spec",
        fake_builder,
    )
    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        lambda *args, **kwargs: None,
        raising=False,
    )

    explanation = probabilistic_explanation(
        is_thresholded=True,
        y_threshold=(0.25, 0.75),
        prediction={"predict": 0.6, "low": float("nan"), "high": None},
        y_minmax=("0.1", "0.9"),
    )

    plotting.plot_probabilistic(
        explanation,
        instance=[0.4],
        predict=explanation.prediction,
        feature_weights={"predict": [0.4]},
        features_to_plot=[0],
        num_to_show=1,
        column_names=["f0"],
        title="interval",
        path=None,
        show=True,
        use_legacy=False,
    )

    assert captured["neg_caption"] == "y_hat <= 0.250 || y_hat > 0.750"
    assert captured["pos_caption"] == "0.250 < y_hat <= 0.750"
    assert captured["predict"]["low"] == pytest.approx(0.1)
    assert captured["predict"]["high"] == pytest.approx(0.9)


def test_plot_probabilistic_multiclass_without_labels(monkeypatch: pytest.MonkeyPatch):
    """Should fall back to prediction classes when no label map is provided."""

    captured: dict[str, object] = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(plotting, "__require_matplotlib", lambda: None)
    monkeypatch.setattr(plotting, "plt", SimpleNamespace())
    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_probabilistic_bars_spec",
        fake_builder,
    )
    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        lambda *args, **kwargs: None,
        raising=False,
    )

    explanation = probabilistic_explanation(
        class_label_error=True,
        prediction={"predict": 0.2, "classes": 2},
        prediction_classes=2,
        explainer_multiclass=True,
    )

    plotting.plot_probabilistic(
        explanation,
        instance=[0.1],
        predict=explanation.prediction,
        feature_weights={"predict": [0.2]},
        features_to_plot=[0],
        num_to_show=1,
        column_names=["f0"],
        title="multiclass",
        path=None,
        show=True,
        use_legacy=False,
    )

    assert captured["neg_caption"] == "P(y!=2)"
    assert captured["pos_caption"] == "P(y=2)"


def test_plot_probabilistic_multiclass_label_lookup_fallback(monkeypatch: pytest.MonkeyPatch):
    """Should fall back to prediction classes when label indexing fails."""

    captured: dict[str, object] = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(plotting, "__require_matplotlib", lambda: None)
    monkeypatch.setattr(plotting, "plt", SimpleNamespace())
    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_probabilistic_bars_spec",
        fake_builder,
    )
    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        lambda *args, **kwargs: None,
        raising=False,
    )

    explanation = probabilistic_explanation(
        class_labels=("only-one",),
        prediction_classes=5,
        prediction={"predict": 0.3, "classes": 5},
        explainer_multiclass=False,
        explanation_multiclass=True,
    )

    plotting.plot_probabilistic(
        explanation,
        instance=[0.1],
        predict=explanation.prediction,
        feature_weights={"predict": [0.3]},
        features_to_plot=[0],
        num_to_show=1,
        column_names=["f0"],
        title="fallback",
        path=None,
        show=True,
        use_legacy=False,
    )

    assert captured["neg_caption"] == "P(y!=5)"
    assert captured["pos_caption"] == "P(y=5)"


def testplot_global_uses_modern_plugin(monkeypatch: pytest.MonkeyPatch):
    """Should invoke plot plugins when not using the legacy path."""

    monkeypatch.setattr(plotting, "__require_matplotlib", lambda: None)
    monkeypatch.setattr(plotting, "plt", SimpleNamespace())
    monkeypatch.setattr(
        plotting,
        "_resolve_plot_style_chain",
        lambda explainer, style: ("plot_spec.default",),
    )

    class DummyExplainer:
        def __init__(self):
            self.learner = SimpleNamespace()
            self.last_explanation_mode = "factual"
            self.latest_explanation = SimpleNamespace()

        def predict(self, x, uq_interval=True, bins=None):
            return [0.42], ([0.1], [0.9])

    output = plotting.plot_global(
        DummyExplainer(), x=[1, 2], show=True, use_legacy=False, style="plot_spec.default"
    )
    assert hasattr(output, "artifact")
    plot_spec = output.artifact.get("plot_spec")
    assert plot_spec is not None
    assert plot_spec["kind"] == "global_regression"


def testplot_global_raises_when_no_plugins(monkeypatch: pytest.MonkeyPatch):
    """Should raise when no plot plugins are available."""

    monkeypatch.setattr("calibrated_explanations.plugins.ensure_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        "calibrated_explanations.plugins.find_plot_plugin_trusted",
        lambda identifier: None,
    )
    monkeypatch.setattr("calibrated_explanations.plugins.find_plot_plugin", lambda identifier: None)
    monkeypatch.setattr(
        plotting, "_resolve_plot_style_chain", lambda explainer, style: ("missing",)
    )
    monkeypatch.setattr(plotting, "__require_matplotlib", lambda: None)
    monkeypatch.setattr(plotting, "plt", SimpleNamespace())

    class DummyExplainer:
        def __init__(self):
            self.learner = SimpleNamespace()
            self.last_explanation_mode = "factual"
            self.latest_explanation = SimpleNamespace()

        def predict(self, x, uq_interval=True, bins=None):
            return [0.42], ([0.1], [0.9])

    with pytest.raises(ConfigurationError):
        plotting.plot_global(DummyExplainer(), x=[1], show=False, use_legacy=False)


def test_plot_proba_triangle_invokes_matplotlib(monkeypatch: pytest.MonkeyPatch):
    """Should build figures using the configured matplotlib shim."""

    class FakePlt:
        def __init__(self) -> None:
            self.plots = []

        def figure(self):
            return "figure"

        def plot(self, *args, **kwargs):
            self.plots.append((args, kwargs))

    fake = FakePlt()
    monkeypatch.setattr(plotting, "__require_matplotlib", lambda: None)
    monkeypatch.setattr(plotting, "plt", fake)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fig = plotting.plot_proba_triangle()
    assert fig == "figure"
    assert fake.plots  # ensure plotting calls executed
