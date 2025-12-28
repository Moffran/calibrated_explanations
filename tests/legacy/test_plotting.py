import types
import numpy as np
import pytest

from calibrated_explanations.legacy import plotting


class FakeAxes:
    def __init__(self):
        self.xlabel = None
        self.ylabel = None
        self.xlim = None
        self.xticks = None
        self.yticks = None
        self.yticklabels = None
        self.twin_axes = []

    def fill_betweenx(self, *args, **kwargs):  # pragma: no cover - behaviour not under test
        return None

    def set_xlabel(self, value):
        self.xlabel = value

    def set_ylabel(self, value):
        self.ylabel = value

    def set_xlim(self, *args, **kwargs):
        self.xlim = (args, kwargs)

    def set_xticks(self, value, **_kwargs):
        self.xticks = value

    def set_yticks(self, value, **_kwargs):
        self.yticks = value

    def set_yticklabels(self, value=None, **kwargs):
        if "labels" in kwargs:
            value = kwargs["labels"]
        self.yticklabels = value

    def set_ylim(self, *args, **kwargs):
        return None

    def twinx(self):
        twin = FakeAxes()
        self.twin_axes.append(twin)
        return twin

    def scatter(self, *args, **kwargs):  # pragma: no cover - behaviour not under test
        return None


class FakeFigure:
    def __init__(self, registry):
        self.registry = registry
        self.axes = []
        self.saved = []
        self.tight_layout_called = False
        self.show_called = False

    def add_subplot(self, *_args, **_kwargs):
        ax = FakeAxes()
        self.axes.append(ax)
        self.registry.last_axes = ax
        return ax

    def savefig(self, path, **kwargs):
        self.saved.append((path, kwargs))
        self.registry.saved_paths.append(path)

    def tight_layout(self):
        self.tight_layout_called = True

    def show(self):
        self.show_called = True
        self.registry.figure_show_calls += 1


class FakePlotLib:
    def __init__(self):
        self.last_axes = None
        self.last_figure = None
        self.saved_paths = []
        self.figure_show_calls = 0
        self.xlabel_calls = []
        self.ylabel_calls = []
        self.quiver_calls = []
        self.scatter_calls = []
        self.legend_calls = 0
        self.plot_calls = []
        self.xlim_calls = []
        self.ylim_calls = []
        self.cm = types.SimpleNamespace(viridis=lambda values: values)

    def figure(self, *args, **kwargs):
        self.last_figure = FakeFigure(self)
        return self.last_figure

    def subplots(self, *args, **kwargs):
        fig = FakeFigure(self)
        ax = FakeAxes()
        fig.axes.append(ax)
        self.last_axes = ax
        return fig, ax

    def quiver(self, *args, **kwargs):
        self.quiver_calls.append((args, kwargs))

    def plot(self, *args, **kwargs):
        self.plot_calls.append((args, kwargs))

    def scatter(self, *args, **kwargs):
        self.scatter_calls.append((args, kwargs))

    def legend(self, *args, **kwargs):
        self.legend_calls += 1

    def xlabel(self, text, **kwargs):
        self.xlabel_calls.append((text, kwargs))

    def ylabel(self, text, **kwargs):
        self.ylabel_calls.append((text, kwargs))

    def xlim(self, *args, **kwargs):
        self.xlim_calls.append((args, kwargs))

    def ylim(self, *args, **kwargs):
        self.ylim_calls.append((args, kwargs))

    def title(self, *args, **kwargs):  # pragma: no cover - behaviour not under test
        return None

    def show(self):
        self.figure_show_calls += 1

    def get_cmap(self, *_args, **_kwargs):
        return lambda index: index


class DummyExplainer:
    def __init__(self, *, mode="classification", multiclass=False):
        self.mode = mode
        self.multiclass_flag = multiclass
        self.prediction = {"classes": 0}
        self.y_minmax = (0.0, 1.0)
        self.calibrated_explanations = types.SimpleNamespace(get_confidence=lambda: 90)
        self.class_labels = None
        self.learner = types.SimpleNamespace()
        self.y_cal = np.array([0.5, 0.5])

    def get_mode(self):
        return self.mode

    def is_thresholded(self):
        return False

    def get_class_labels(self):
        return None

    def _get_explainer(self):
        return types.SimpleNamespace(is_multiclass=lambda: self.multiclass_flag)

    def is_multiclass(self):
        return self.multiclass_flag

    def predict_proba(self, *_args, **_kwargs):
        proba = np.array([[0.3, 0.7]])
        low = np.array([[0.2, 0.6]])
        high = np.array([[0.4, 0.8]])
        return proba, (low, high)

    def predict(self, x, uq_interval=False, **_kwargs):
        preds = np.full(len(x), 0.42)
        if not uq_interval:
            return preds

        low = np.full(len(x), 0.32)
        high = np.full(len(x), 0.52)
        return preds, (low, high)


class DummyThresholdExplanation(DummyExplainer):
    def __init__(self, y_threshold):
        super().__init__(mode="classification")
        self.threshold_value = y_threshold

    def is_thresholded(self):
        return True

    @property
    def y_threshold(self):
        return self.threshold_value


class DummyClassificationExplanation(DummyExplainer):
    def __init__(self, *, with_labels=False, multiclass=False):
        super().__init__(mode="classification", multiclass=multiclass)
        self.with_labels_flag = with_labels
        if with_labels:
            self.class_labels = {0: "zero", 1: "one"}
        self.prediction = {"classes": 1}

    def get_class_labels(self):
        return None if not self._with_labels else self.class_labels


@pytest.fixture
def fake_matplotlib(monkeypatch):
    fake_plt = FakePlotLib()
    monkeypatch.setattr(plotting, "plt", fake_plt)
    monkeypatch.setattr(plotting, "_MATPLOTLIB_IMPORT_ERROR", None)
    return fake_plt


def test_compose_save_target_handles_directories(tmp_path):
    folder = tmp_path / "plots"
    folder.mkdir()
    result_dir = plotting._compose_save_target(folder, "demo", ".png")
    assert result_dir == str(folder / "demo.png")

    file_base = tmp_path / "base"
    result_file = plotting._compose_save_target(file_base, "demo", ".png")
    assert result_file == str(file_base) + "demo.png"


def test_require_matplotlib_reports_original_error(monkeypatch):
    from calibrated_explanations.utils.exceptions import ConfigurationError

    monkeypatch.setattr(plotting, "plt", None)
    monkeypatch.setattr(plotting, "_MATPLOTLIB_IMPORT_ERROR", ImportError("boom"))
    with pytest.raises(ConfigurationError) as exc_info:
        plotting.__require_matplotlib()
    assert "Original import error: boom" in str(exc_info.value)


def test_plot_alternative_sets_positive_class_label(fake_matplotlib, tmp_path):
    explanation = DummyClassificationExplanation()
    predict = {"predict": 0.6, "low": 0.4, "high": 0.8}
    feature_predict = {
        "predict": np.array([0.6, 0.3]),
        "low": np.array([0.2, 0.1]),
        "high": np.array([0.9, 0.7]),
    }
    plotting.plot_alternative(
        explanation,
        instance=[1.0, 2.0],
        predict=predict,
        feature_predict=feature_predict,
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["f0", "f1"],
        title="alt",
        path=tmp_path,
        show=False,
        save_ext=[".png"],
    )

    ax_main = fake_matplotlib.last_figure.axes[0]
    assert ax_main.xlabel == "Probability for the positive class"


def test_plot_alternative_threshold_array_uses_fallback(fake_matplotlib, tmp_path):
    explanation = DummyThresholdExplanation(np.array(0.42))
    predict = {"predict": 0.6, "low": 0.4, "high": 0.8}
    feature_predict = {
        "predict": np.array([0.6]),
        "low": np.array([0.2]),
        "high": np.array([0.9]),
    }
    plotting.plot_alternative(
        explanation,
        instance=[1.0],
        predict=predict,
        feature_predict=feature_predict,
        features_to_plot=[0],
        num_to_show=1,
        column_names=["f0"],
        title="alt",
        path=tmp_path,
        show=False,
        save_ext=[".png"],
    )

    ax_main = fake_matplotlib.last_figure.axes[0]
    assert ax_main.xlabel == "Probability of target being below 0.42"


def test_plot_global_warns_for_identical_uncertainty(fake_matplotlib):
    explanation = DummyExplainer()
    x = np.array([[1.0, 2.0]])

    plotting._plot_global(
        explanation,
        x,
        y=None,
        threshold=(0.2, 0.8),
        show=False,
    )

    # The last xlabel call corresponds to the probability interval branch
    assert fake_matplotlib.xlabel_calls[-1][0] == "Probability of 0.2 <= Y < 0.8"

    fake_matplotlib.xlabel_calls.clear()

    with pytest.warns(Warning):
        plotting._plot_global(
            explanation,
            x,
            y=None,
            threshold=None,
            show=False,
        )

    assert fake_matplotlib.xlabel_calls[-1][0] == "Predictions"
