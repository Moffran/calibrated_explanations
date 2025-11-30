import types

import numpy as np
import pytest

from calibrated_explanations.legacy import plotting

matplotlib = pytest.importorskip(
    "matplotlib", reason="matplotlib is required for legacy plotting tests"
)

matplotlib.use("Agg", force=True)


class _CalibratedStub:
    def __init__(self, confidence=95):
        self._confidence = confidence

    def get_confidence(self):
        return self._confidence


class _InnerExplainer:
    def __init__(self, multiclass=False):
        self._multiclass = multiclass

    def is_multiclass(self):
        return self._multiclass


_DEFAULT_LABELS = object()


class DummyExplanation:
    def __init__(
        self,
        mode="classification",
        thresholded=False,
        y_threshold=0.7,
        class_labels=_DEFAULT_LABELS,
        multiclass=False,
        y_minmax=None,
        prediction_class=1,
        inner_multiclass=None,
        confidence=95,
        one_sided=False,
    ):
        self._mode = mode
        self._thresholded = thresholded
        self.y_threshold = y_threshold
        if y_minmax is None:
            y_minmax = (0.0, 1.0) if "regression" not in mode else (0.0, 10.0)
        self.y_minmax = y_minmax
        self.prediction = {"classes": prediction_class}
        self.is_multiclass = multiclass
        if class_labels is _DEFAULT_LABELS:
            class_labels = ["neg", "pos"]
        self._class_labels = class_labels
        self.calibrated_explanations = _CalibratedStub(confidence)
        if inner_multiclass is None:
            inner_multiclass = multiclass
        self._explainer = _InnerExplainer(inner_multiclass)
        self._one_sided = one_sided

    def is_one_sided(self):
        return self._one_sided

    def is_thresholded(self):
        return self._thresholded

    def get_class_labels(self):
        return self._class_labels

    def get_mode(self):
        return self._mode

    def _get_explainer(self):
        return self._explainer


@pytest.fixture(autouse=True)
def close_figures():
    yield
    from matplotlib import pyplot as plt

    plt.close("all")


@pytest.fixture
def disable_show(monkeypatch):
    from matplotlib.figure import Figure

    monkeypatch.setattr(Figure, "show", lambda self: None)
    monkeypatch.setattr(plotting.plt, "show", lambda *args, **kwargs: None)


def test_probabilistic_plot_creates_expected_image(tmp_path):
    explanation = DummyExplanation()
    instance = [1.0, 2.0]
    predict = {"predict": 0.6, "low": -np.inf, "high": np.inf}
    feature_weights = [0.4, -0.25]
    features_to_plot = [0, 1]
    column_names = ["feature_a", "feature_b"]

    plotting._plot_probabilistic(
        explanation,
        instance,
        predict,
        feature_weights,
        features_to_plot,
        num_to_show=len(features_to_plot),
        column_names=column_names,
        title="probabilistic",
        path=str(tmp_path) + "/",
        show=False,
        save_ext=[".png"],
    )

    assert (tmp_path / "probabilistic.png").exists()


def test_plotting_reload_handles_missing_matplotlib(monkeypatch):
    import builtins
    import importlib

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("matplotlib"):
            raise ImportError("matplotlib missing")
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as ctx:
        ctx.setattr(builtins, "__import__", fake_import)
        importlib.reload(plotting)
        assert plotting.plt is None
        assert plotting._MATPLOTLIB_IMPORT_ERROR is not None

    importlib.reload(plotting)


def test_probabilistic_interval_branches(tmp_path, disable_show):
    explanation = DummyExplanation(thresholded=True, y_threshold=0.4)
    instance = [1.0, -1.0]
    predict = {"predict": 0.3, "low": 0.1, "high": 0.6}
    feature_weights = {
        "predict": np.array([0.2, -0.3]),
        "low": np.array([-0.4, 0.2]),
        "high": np.array([0.5, 0.6]),
    }
    plotting._plot_probabilistic(
        explanation,
        instance,
        predict,
        feature_weights,
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["f0", "f1"],
        title="prob_interval",
        path=str(tmp_path) + "/",
        show=True,
        interval=True,
        idx=0,
        save_ext=None,
    )

    saved = sorted(p.name for p in tmp_path.iterdir())
    assert len(saved) == 3


def test_probabilistic_default_save_extensions_use_title(tmp_path):
    explanation = DummyExplanation()
    instance = [0.5, -0.1]
    predict = {"predict": 0.55, "low": 0.2, "high": 0.8}

    plotting._plot_probabilistic(
        explanation,
        instance,
        predict,
        feature_weights=[0.3, -0.2],
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["a", "b"],
        title="auto",
        path=str(tmp_path) + "/",
        show=False,
        save_ext=None,
    )

    saved = sorted(p.name for p in tmp_path.iterdir())
    assert len(saved) == 3
    for ext in ("svg", "pdf", "png"):
        assert any(name.startswith(f"auto{ext}") for name in saved), saved


def test_probabilistic_interval_requires_index():
    explanation = DummyExplanation(thresholded=True)
    feature_weights = {
        "predict": np.array([0.1, -0.2]),
        "low": np.array([0.0, -0.3]),
        "high": np.array([0.2, 0.4]),
    }

    with pytest.raises(AssertionError):
        plotting._plot_probabilistic(
            explanation,
            instance=[0.5, 0.1],
            predict={"predict": 0.6, "low": 0.4, "high": 0.8},
            feature_weights=feature_weights,
            features_to_plot=[0, 1],
            num_to_show=2,
            column_names=["f0", "f1"],
            title="missing_idx",
            path="",
            show=False,
            interval=True,
            idx=None,
            save_ext=[".png"],
        )


def test_probabilistic_threshold_and_label_variants(tmp_path):
    instance = [0.5, 0.9]
    features = [0, 1]
    columns = ["c0", "c1"]

    # Tuple threshold branch
    explanation = DummyExplanation(thresholded=True, y_threshold=(0.2, 0.8))
    plotting._plot_probabilistic(
        explanation,
        instance,
        {"predict": 0.7, "low": 0.3, "high": 0.9},
        [0.1, -0.2],
        features,
        num_to_show=2,
        column_names=columns,
        title="prob_tuple",
        path=str(tmp_path) + "/",
        show=False,
        save_ext=[".png"],
    )

    # No class labels with multiclass explainer
    explanation = DummyExplanation(class_labels=None, inner_multiclass=True)
    plotting._plot_probabilistic(
        explanation,
        instance,
        {"predict": 0.5, "low": 0.4, "high": 0.6},
        [0.3, -0.1],
        features,
        num_to_show=2,
        column_names=columns,
        title="prob_no_labels_multi",
        path=str(tmp_path) + "/",
        show=False,
        save_ext=[".png"],
    )

    # No class labels with binary explainer
    explanation = DummyExplanation(class_labels=None, inner_multiclass=False)
    plotting._plot_probabilistic(
        explanation,
        instance,
        {"predict": 0.4, "low": 0.2, "high": 0.7},
        [0.2, -0.3],
        features,
        num_to_show=2,
        column_names=columns,
        title="prob_no_labels_binary",
        path=str(tmp_path) + "/",
        show=False,
        save_ext=[".png"],
    )

    # Multiclass labels branch
    explanation = DummyExplanation(
        class_labels=["zero", "one", "two"], multiclass=True, prediction_class=2
    )
    plotting._plot_probabilistic(
        explanation,
        instance,
        {"predict": 0.6, "low": 0.5, "high": 0.7},
        [0.25, 0.15],
        features,
        num_to_show=2,
        column_names=columns,
        title="prob_multiclass",
        path=str(tmp_path) + "/",
        show=False,
        save_ext=[".png"],
    )


def test_probabilistic_interval_one_sided_error(tmp_path):
    explanation = DummyExplanation(one_sided=True)
    with pytest.raises(Warning):
        plotting._plot_probabilistic(
            explanation,
            instance=[0.3],
            predict={"predict": 0.5, "low": 0.2, "high": 0.8},
            feature_weights={
                "predict": np.array([0.1]),
                "low": np.array([0.0]),
                "high": np.array([0.2]),
            },
            features_to_plot=[0],
            num_to_show=1,
            column_names=["feat"],
            title="prob_error",
            path=str(tmp_path) + "/",
            show=False,
            interval=True,
            idx=0,
            save_ext=[".png"],
        )


def test_probabilistic_returns_without_output():
    plotting._plot_probabilistic(
        DummyExplanation(),
        instance=[0.1],
        predict={"predict": 0.5, "low": 0.2, "high": 0.8},
        feature_weights=[0.1],
        features_to_plot=[0],
        num_to_show=1,
        column_names=["feat"],
        title="unused",
        path=None,
        show=False,
    )


def test_plot_global_requires_scalar_threshold_for_non_probabilistic():
    class _ThresholdWrapper:
        def __init__(self):
            self.learner = types.SimpleNamespace()
            self.y_cal = np.array([0.1, 0.2, 0.3])

        def predict(self, x, uq_interval=False, **kwargs):
            preds = np.linspace(0.2, 0.6, len(x))
            low = preds - 0.1
            high = preds + 0.1
            return preds, (low, high)

        def predict_proba(self, x, uq_interval=False, threshold=None, **kwargs):
            preds, (low, high) = self.predict(x, uq_interval=uq_interval)
            proba = np.column_stack([1 - preds, preds])
            return proba, (low, high)

        def is_multiclass(self):  # pragma: no cover - deterministic helper
            return False

    explainer = _ThresholdWrapper()
    x_vals = np.zeros((3, 1))
    y_vals = np.array([0, 1, 0])

    with pytest.warns(RuntimeWarning), pytest.raises(AssertionError):
        plotting._plot_global(
            explainer,
            x_vals,
            y=y_vals,
            threshold=(0.2, 0.8),
            show=False,
        )


def test_regression_interval_plot_saves_image(tmp_path):
    explanation = DummyExplanation(mode="regression", thresholded=False, y_minmax=(-2.0, 2.0))
    instance = [0.1, -1.5]
    predict = {"predict": 0.2, "low": -0.3, "high": 0.7}
    feature_weights = {
        "predict": np.array([0.5, -0.2]),
        "low": np.array([-0.1, -0.6]),
        "high": np.array([0.8, 0.4]),
    }
    features_to_plot = [0, 1]
    column_names = ["rule_a", "rule_b"]

    plotting._plot_regression(
        explanation,
        instance,
        predict,
        feature_weights,
        features_to_plot,
        num_to_show=len(features_to_plot),
        column_names=column_names,
        title="regression",
        path=str(tmp_path) + "/",
        show=False,
        interval=True,
        idx=0,
        save_ext=[".png"],
    )

    assert (tmp_path / "regression.png").exists()


def test_regression_interval_requires_index(tmp_path):
    explanation = DummyExplanation(mode="regression")
    feature_weights = {
        "predict": np.array([0.5, -0.2]),
        "low": np.array([0.3, -0.4]),
        "high": np.array([0.7, -0.1]),
    }

    with pytest.raises(AssertionError):
        plotting._plot_regression(
            explanation,
            instance=[1.0, -1.0],
            predict={"predict": 0.2, "low": -0.1, "high": 0.5},
            feature_weights=feature_weights,
            features_to_plot=[0, 1],
            num_to_show=2,
            column_names=["f0", "f1"],
            title="reg_idx",
            path=str(tmp_path) + "/",
            show=False,
            interval=True,
            idx=None,
            save_ext=[".png"],
        )


def test_regression_non_interval_branches(tmp_path, disable_show):
    explanation = DummyExplanation(mode="regression", y_minmax=(-1.0, 1.0))
    instance = [0.3, -0.8]
    predict = {"predict": -0.1, "low": -0.4, "high": 0.2}
    plotting._plot_regression(
        explanation,
        instance,
        predict,
        feature_weights=[0.4, -0.6],
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["nr0", "nr1"],
        title="regression_non_interval",
        path=str(tmp_path) + "/",
        show=True,
        interval=False,
        save_ext=None,
    )

    saved = sorted(p.name for p in tmp_path.iterdir())
    assert len(saved) == 3


def test_interval_requires_idx_and_two_sided(tmp_path):
    instance = [1.0]
    features = [0]
    columns = ["f0"]
    predict = {"predict": 0.3, "low": 0.2, "high": 0.4}
    feature_weights = {"predict": np.array([0.1]), "low": np.array([0.0]), "high": np.array([0.2])}

    explanation = DummyExplanation()
    with pytest.raises(AssertionError):
        plotting._plot_probabilistic(
            explanation,
            instance,
            predict,
            feature_weights,
            features,
            num_to_show=1,
            column_names=columns,
            title="missing_idx",
            path=str(tmp_path) + "/",
            show=False,
            interval=True,
            idx=None,
            save_ext=[".png"],
        )

    one_sided = DummyExplanation(one_sided=True)
    with pytest.raises(Warning):
        plotting._plot_probabilistic(
            one_sided,
            instance,
            predict,
            feature_weights,
            features,
            num_to_show=1,
            column_names=columns,
            title="one_sided",
            path=str(tmp_path) + "/",
            show=False,
            interval=True,
            idx=0,
            save_ext=[".png"],
        )


def test_plot_global_threshold_requires_scalar():
    class _NonProbExplainer:
        def __init__(self):
            self.learner = object()

        def predict(self, x, uq_interval=False, **kwargs):
            preds = np.linspace(0.2, 0.4, len(x))
            return preds, (preds - 0.05, preds + 0.05)

        def predict_proba(self, x, uq_interval=False, threshold=None, **kwargs):
            preds, (low, high) = self.predict(x, uq_interval=uq_interval, **kwargs)
            # Mimic binary proba output for thresholded regression fallback path
            stacked = np.vstack([preds, preds]).T
            return stacked, (low, high)

        def is_multiclass(self):
            return False

    explainer = _NonProbExplainer()
    x = np.zeros((3, 1))
    y = np.array([0.1, 0.2, 0.3])

    with pytest.raises(AssertionError):
        plotting._plot_global(
            explainer,
            x,
            y=y,
            threshold=(0.2, 0.5),
            show=False,
        )


def test_regression_interval_one_sided_error(tmp_path):
    explanation = DummyExplanation(mode="regression", one_sided=True)
    with pytest.raises(Warning):
        plotting._plot_regression(
            explanation,
            instance=[0.2],
            predict={"predict": 0.0, "low": -0.1, "high": 0.1},
            feature_weights={
                "predict": np.array([0.1]),
                "low": np.array([-0.2]),
                "high": np.array([0.3]),
            },
            features_to_plot=[0],
            num_to_show=1,
            column_names=["nr"],
            title="reg_error",
            path=str(tmp_path) + "/",
            show=False,
            interval=True,
            idx=0,
            save_ext=[".png"],
        )


def test_regression_returns_without_output():
    plotting._plot_regression(
        DummyExplanation(mode="regression"),
        instance=[0.1],
        predict={"predict": 0.0, "low": -0.2, "high": 0.2},
        feature_weights=[0.5],
        features_to_plot=[0],
        num_to_show=1,
        column_names=["nr"],
        title="unused",
        path=None,
        show=False,
    )


def test_alternative_plot_handles_thresholded_classification(tmp_path):
    explanation = DummyExplanation(thresholded=True, y_threshold=0.65)
    instance = ["high", "low"]
    predict = {"predict": 0.55, "low": 0.4, "high": 0.8}
    feature_predict = {
        "predict": np.array([0.55, 0.75]),
        "low": np.array([0.4, 0.6]),
        "high": np.array([0.7, 0.9]),
    }
    features_to_plot = [0, 1]
    column_names = ["alt_a", "alt_b"]

    plotting._plot_alternative(
        explanation,
        instance,
        predict,
        feature_predict,
        features_to_plot,
        num_to_show=len(features_to_plot),
        column_names=column_names,
        title="alternative",
        path=str(tmp_path) + "/",
        show=False,
        save_ext=[".png"],
    )

    assert (tmp_path / "alternative.png").exists()


def test_alternative_plot_regression_branch(tmp_path, disable_show):
    explanation = DummyExplanation(mode="regression", y_minmax=(-1.0, 1.0))
    instance = [0.2, -0.4]
    predict = {"predict": 0.1, "low": -0.2, "high": 0.3}
    feature_predict = {
        "predict": np.array([0.15, -0.25]),
        "low": np.array([-0.1, -0.5]),
        "high": np.array([0.2, -0.1]),
    }
    plotting._plot_alternative(
        explanation,
        instance,
        predict,
        feature_predict,
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["ra", "rb"],
        title="alternative_regression",
        path=str(tmp_path) + "/",
        show=True,
        save_ext=None,
    )

    saved = sorted(p.name for p in tmp_path.iterdir())
    assert len(saved) == 3


def test_alternative_returns_without_output():
    plotting._plot_alternative(
        DummyExplanation(),
        instance=["only"],
        predict={"predict": 0.5, "low": 0.4, "high": 0.6},
        feature_predict={
            "predict": np.array([0.5]),
            "low": np.array([0.3]),
            "high": np.array([0.7]),
        },
        features_to_plot=[0],
        num_to_show=1,
        column_names=["c"],
        title="unused",
        path=None,
        show=False,
    )


def test_triangular_plot_uses_probability_triangle(tmp_path):
    explanation = DummyExplanation()
    rule_proba = np.array([0.4, 0.6, 0.8])
    rule_uncertainty = np.array([0.1, 0.2, 0.3])

    plotting._plot_triangular(
        explanation,
        proba=0.5,
        uncertainty=0.25,
        rule_proba=rule_proba,
        rule_uncertainty=rule_uncertainty,
        num_to_show=3,
        title="triangular",
        path=str(tmp_path) + "/",
        show=False,
        save_ext=[".png"],
    )

    assert (tmp_path / "triangular.png").exists()


def test_triangular_plot_regression_mode(tmp_path, disable_show):
    explanation = DummyExplanation(mode="regression", thresholded=False)
    rule_proba = np.array([0.5, 0.5, 0.5])
    rule_uncertainty = np.array([0.1, 0.2, 0.3])

    with pytest.warns(Warning):
        plotting._plot_triangular(
            explanation,
            proba=0.5,
            uncertainty=0.25,
            rule_proba=rule_proba,
            rule_uncertainty=rule_uncertainty,
            num_to_show=3,
            title="triangular_regression",
            path=str(tmp_path) + "/",
            show=True,
            save_ext=None,
        )

    saved = sorted(p.name for p in tmp_path.iterdir())
    assert len(saved) == 3


def test_triangular_returns_without_output():
    plotting._plot_triangular(
        DummyExplanation(),
        proba=0.4,
        uncertainty=0.1,
        rule_proba=np.array([0.4]),
        rule_uncertainty=np.array([0.1]),
        num_to_show=1,
        title="unused",
        path=None,
        show=False,
    )


def test_probabilistic_figsize_scales_with_num_to_show(monkeypatch, tmp_path, disable_show):
    recorded: list[tuple[float, float] | None] = []
    real_figure = plotting.plt.figure

    def record_figure(*args, **kwargs):
        recorded.append(kwargs.get("figsize"))
        return real_figure(*args, **kwargs)

    monkeypatch.setattr(plotting.plt, "figure", record_figure)

    plotting._plot_probabilistic(
        DummyExplanation(),
        instance=[0.3, -0.2, 0.1, 0.4],
        predict={"predict": 0.6, "low": 0.4, "high": 0.8},
        feature_weights=[0.2, -0.1, 0.05, -0.3],
        features_to_plot=[0, 1, 2, 3],
        num_to_show=4,
        column_names=["f0", "f1", "f2", "f3"],
        title="fig_prob",
        path=str(tmp_path) + "/",
        show=True,
        interval=False,
        idx=None,
        save_ext=[],
    )

    plotting._plot_regression(
        DummyExplanation(mode="regression"),
        instance=[0.2, -0.5, 0.1, -0.3, 0.4, -0.2],
        predict={"predict": 0.1, "low": -0.2, "high": 0.5},
        feature_weights=[0.3, -0.1, 0.2, -0.4, 0.5, -0.2],
        features_to_plot=[0, 1, 2, 3, 4, 5],
        num_to_show=6,
        column_names=[f"r{i}" for i in range(6)],
        title="fig_reg",
        path=str(tmp_path) + "/",
        show=True,
        interval=False,
        idx=None,
        save_ext=[],
    )

    assert recorded[0] == (10, pytest.approx(4.0))
    assert recorded[1] == (10, pytest.approx(5.0))


def test_plot_global_non_probabilistic(monkeypatch):
    class _Learner:
        pass

    class _Explainer:
        def __init__(self):
            self.learner = _Learner()
            self.y_cal = np.array([0.1, 0.4, 0.9])

        def predict(self, x, uq_interval=True, **kwargs):
            del x, uq_interval, kwargs
            predict = np.array([0.2, 0.7])
            low = np.array([0.1, 0.6])
            high = np.array([0.3, 0.9])
            return predict, (low, high)

    explainer = _Explainer()

    monkeypatch.setattr(plotting.plt, "show", lambda *args, **kwargs: None)

    plotting._plot_global(explainer, x=np.array([[0.0], [1.0]]), show=True)


def test_plot_global_threshold_branch(monkeypatch):
    class _Learner:
        pass

    class _Explainer:
        def __init__(self):
            self.learner = _Learner()
            self.y_cal = np.array([0.2, 0.5, 0.9])
            self.class_labels = None

        def predict_proba(self, x, uq_interval=True, threshold=None, **kwargs):
            del x, uq_interval, threshold, kwargs
            proba = np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
            low = np.array([0.2, 0.3, 0.4])
            high = np.array([0.4, 0.5, 0.6])
            return proba, (low, high)

    explainer = _Explainer()
    monkeypatch.setattr(plotting.plt, "show", lambda *args, **kwargs: None)

    plotting._plot_global(
        explainer,
        x=np.zeros((3, 1)),
        y=np.array([0.9, 0.3, 0.4]),
        threshold=0.5,
        show=True,
    )


def test_plot_global_probabilistic_variants(monkeypatch):
    class _Learner:
        def predict_proba(self, *args, **kwargs):  # pragma: no cover - interface placeholder
            return None

    class _MultiExplainer:
        def __init__(self):
            self.learner = _Learner()
            self.y_cal = np.array([0.1, 0.4, 0.7])
            self.class_labels = {0: "zero", 1: "one", 2: "two"}

        def predict_proba(self, x, uq_interval=True, threshold=None, **kwargs):
            del x, uq_interval, threshold, kwargs
            proba = np.array(
                [
                    [0.2, 0.5, 0.3],
                    [0.1, 0.6, 0.3],
                    [0.7, 0.2, 0.1],
                ]
            )
            low = proba - 0.05
            high = proba + 0.05
            return proba, (low, high)

        def is_multiclass(self):
            return True

    class _BinaryExplainer:
        def __init__(self):
            self.learner = _Learner()
            self.y_cal = np.array([0.2, 0.6])
            self.class_labels = None

        def predict_proba(self, x, uq_interval=True, threshold=None, **kwargs):
            del x, uq_interval, threshold, kwargs
            proba = np.array([[0.3, 0.7], [0.6, 0.4]])
            low = np.array([0.1, 0.2])
            high = np.array([0.2, 0.3])
            return proba, (low, high)

        def is_multiclass(self):
            return False

    monkeypatch.setattr(plotting.plt, "show", lambda *args, **kwargs: None)

    multi_explainer = _MultiExplainer()
    plotting._plot_global(multi_explainer, x=np.zeros((3, 1)), show=True)
    plotting._plot_global(
        multi_explainer,
        x=np.zeros((3, 1)),
        y=np.array([0, 1, 2]),
        show=True,
    )

    binary_explainer = _BinaryExplainer()
    plotting._plot_global(
        binary_explainer,
        x=np.zeros((2, 1)),
        y=np.array([0, 1]),
        show=True,
    )


def test_plot_global_headless_short_circuit(monkeypatch):
    explainer = types.SimpleNamespace(learner=types.SimpleNamespace())

    def fail():  # pragma: no cover - guard should prevent execution
        raise AssertionError("__require_matplotlib should not be called")

    monkeypatch.setattr(plotting, "plt", None)
    monkeypatch.setattr(plotting, "__require_matplotlib", fail)

    plotting._plot_global(explainer, x=np.zeros((1, 1)), show=False)


def test_plot_global_requires_scalar_threshold_for_predict_only(disable_show):
    class _PredictOnlyExplainer:
        def __init__(self):
            self.learner = types.SimpleNamespace()
            self.y_cal = np.array([0.1, 0.2, 0.3])

        def predict(self, x, uq_interval=False, **kwargs):
            preds = np.linspace(0.2, 0.4, len(x))
            return preds, (preds - 0.05, preds + 0.05)

        def predict_proba(self, x, uq_interval=False, threshold=None, **kwargs):
            preds = np.tile([0.6, 0.4], (len(x), 1))
            lows = np.tile([0.5, 0.3], (len(x), 1))
            highs = np.tile([0.7, 0.5], (len(x), 1))
            return preds, (lows, highs)

        def is_multiclass(self):
            return False

    explainer = _PredictOnlyExplainer()
    x = np.zeros((3, 1))
    y = np.array([0.1, 0.2, 0.3])

    with pytest.raises(AssertionError):
        plotting._plot_global(
            explainer,
            x,
            y=y,
            threshold=(0.2, 0.4),
            show=False,
        )


def test_plot_proba_triangle_helper():
    plotting._plot_proba_triangle()


def test_probabilistic_saves_before_show(monkeypatch, tmp_path):
    order: list[str] = []

    def fake_savefig(self, *args, **kwargs):
        order.append("save")

    def fake_show(self):
        order.append("show")

    from matplotlib.figure import Figure

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    monkeypatch.setattr(Figure, "show", fake_show)

    plotting._plot_probabilistic(
        DummyExplanation(),
        instance=[0.2, 0.3],
        predict={"predict": 0.6, "low": 0.4, "high": 0.8},
        feature_weights=[0.1, -0.2],
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["f0", "f1"],
        title="ordered",
        path=str(tmp_path) + "/",
        show=True,
        interval=False,
        idx=None,
        save_ext=[".png"],
    )

    assert order[:2] == ["save", "show"]


def test_require_matplotlib_raises(monkeypatch):
    from calibrated_explanations.core.exceptions import ConfigurationError
    monkeypatch.setattr(plotting, "plt", None)
    monkeypatch.setattr(plotting, "_MATPLOTLIB_IMPORT_ERROR", ImportError("missing backend"))

    with pytest.raises(ConfigurationError) as excinfo:
        plotting.__require_matplotlib()

    assert "missing backend" in str(excinfo.value)
