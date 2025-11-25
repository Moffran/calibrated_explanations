import os
from pathlib import Path

import numpy as np

from calibrated_explanations.viz import plots as _plots


class _FakeExplanation:
    def __init__(self, mode="regression"):
        self._mode = mode
        self.y_minmax = (0.0, 1.0)

        class _CE:
            def get_confidence(self):
                return 95

        self.calibrated_explanations = _CE()

    def get_mode(self):
        return self._mode

    def is_thresholded(self):
        return False

    def get_class_labels(self):
        return None


class _FakeExplainer:
    def __init__(self):
        self.learner = object()  # no predict_proba -> non-regularized branch
        # sample calibration values
        self.y_cal = np.array([0.1, 0.2, 0.3])

    def predict(self, x, uq_interval=False, bins=None):
        # return (predictions, (low, high)) similar shape to usage
        preds = np.zeros(len(x))
        low = np.zeros(len(x))
        high = np.ones(len(x))
        return preds, (low, high)

    def is_multiclass(self):
        return False


def test_plot_regression_writes_file(tmp_path):
    """Test that plot_regression saves files to disk.

    Refactored to use pathlib.Path and semantic assertions.
     Tests that files are created, not exact path strings.
    """
    # prepare feature weights as dict with interval arrays
    n = 4
    fw = {
        "predict": np.array([0.1, -0.2, 0.3, 0.0]),
        "low": np.array([0.05, -0.25, 0.25, -0.05]),
        "high": np.array([0.15, -0.15, 0.35, 0.05]),
    }
    expl = _FakeExplanation(mode="regression")
    outdir = str(tmp_path)
    title = "reg_test"
    # call with save_ext to force render+save behavior
    _plots._plot_regression(
        explanation=expl,
        instance=[0, 0, 0, 0],
        predict={"predict": 0.5, "low": 0.4, "high": 0.6},
        feature_weights=fw,
        features_to_plot=list(range(n)),
        num_to_show=n,
        column_names=[f"f{i}" for i in range(n)],
        title=title,
        path=outdir + os.path.sep,
        show=False,
        interval=True,
        save_ext=[".png"],
        use_legacy=False,
    )
    # Semantic assertion: verify that a PNG file with the title was created
    # (not an exact path comparison)
    png_files = list(Path(outdir).glob("*" + title + "*.png"))
    assert len(png_files) > 0, f"Expected PNG file with title '{title}' in {outdir}"

    # Verify file is non-trivial in size
    for png_file in png_files:
        assert png_file.stat().st_size > 100, f"PNG file {png_file} is too small"


def test_plot_alternative_writes_file(tmp_path):
    """Test that plot_alternative saves files to disk.

    Refactored to use pathlib.Path and semantic assertions.
    """
    # n is implied by feature_predict length
    expl = _FakeExplanation(mode="regression")
    feature_predict = {
        "predict": np.array([0.2, 0.8, 0.4]),
        "low": np.array([0.1, 0.7, 0.35]),
        "high": np.array([0.3, 0.9, 0.45]),
    }
    outdir = str(tmp_path)
    title = "alt_test"
    _plots._plot_alternative(
        explanation=expl,
        instance=[1, 2, 3],
        predict={"predict": 0.6, "low": 0.5, "high": 0.7},
        feature_predict=feature_predict,
        features_to_plot=[0, 1, 2],
        num_to_show=3,
        column_names=["a", "b", "c"],
        title=title,
        path=outdir + os.path.sep,
        show=False,
        save_ext=[".png"],
        use_legacy=False,
    )
    # Semantic assertion: verify that a PNG file with the title was created
    png_files = list(Path(outdir).glob("*" + title + "*.png"))
    assert len(png_files) > 0, f"Expected PNG file with title '{title}' in {outdir}"


def test_plot_global_non_probabilistic_runs_without_error():
    expl = _FakeExplainer()
    # small x_test
    x_test = np.zeros((3, 2))
    # should not raise
    _plots._plot_global(expl, x_test, y_test=None, threshold=None, show=False, use_legacy=False)


def test_plot_proba_triangle_returns_figure():
    fig = _plots._plot_proba_triangle()
    assert fig is not None


def test_plot_alternative_thresholded_writes_file(tmp_path):
    """Test that plot_alternative with thresholded explanation saves files.

    Refactored to use pathlib.Path and semantic assertions.
     Thresholded explanation should hit the label/xticks/xlim branch.
    """

    class _FakeExplanationThresh(_FakeExplanation):
        def __init__(self):
            super().__init__(mode="regression")
            self.y_threshold = 0.25

        def is_thresholded(self):
            return True

    # n is implied by feature_predict length
    expl = _FakeExplanationThresh()
    feature_predict = {
        "predict": np.array([0.2, 0.8]),
        "low": np.array([0.1, 0.7]),
        "high": np.array([0.3, 0.9]),
    }
    outdir = str(tmp_path)
    title = "alt_thresh"
    _plots._plot_alternative(
        explanation=expl,
        instance=[1, 2],
        predict={"predict": 0.6, "low": 0.5, "high": 0.7},
        feature_predict=feature_predict,
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["a", "b"],
        title=title,
        path=outdir + os.path.sep,
        show=False,
        save_ext=[".png"],
        use_legacy=False,
    )
    # Semantic assertion: verify that a PNG file with the title was created
    png_files = list(Path(outdir).glob("*" + title + "*.png"))
    assert len(png_files) > 0, f"Expected PNG file with title '{title}' in {outdir}"


def test_plot_global_probabilistic_branch_runs():
    # fake explainer with predict_proba to exercise probabilistic branch
    class _FakeExplainerProba:
        def __init__(self):
            # the plotting helper checks explainer.learner for predict_proba
            self.learner = self

        # accept optional args used by plotting helper
        def predict_proba(self, x, uq_interval=False, threshold=None, bins=None):
            proba = np.tile(np.array([[0.3, 0.7]]), (len(x), 1))
            # return per-class low/high arrays matching proba shape
            low = np.zeros_like(proba)
            high = np.ones_like(proba)
            return proba, (low, high)

        def is_multiclass(self):
            return True

        def predict(self, x, uq_interval=False, bins=None):
            return np.zeros(len(x)), (np.zeros(len(x)), np.ones(len(x)))

    expl = _FakeExplainerProba()
    x_test = np.zeros((3, 2))
    # should not raise
    _plots._plot_global(expl, x_test, y_test=None, threshold=None, show=False, use_legacy=False)


def test_plot_alternative_early_noop_when_not_saving():
    # if not showing and save_ext is None, function returns early without matplotlib
    expl = _FakeExplanation(mode="regression")
    feature_predict = {
        "predict": np.array([0.2]),
        "low": np.array([0.1]),
        "high": np.array([0.3]),
    }
    # should not raise even if matplotlib not available because save_ext is None
    _plots._plot_alternative(
        explanation=expl,
        instance=[1],
        predict={"predict": 0.6, "low": 0.5, "high": 0.7},
        feature_predict=feature_predict,
        features_to_plot=[0],
        num_to_show=1,
        column_names=["a"],
        title="noop",
        path=None,
        show=False,
        save_ext=None,
        use_legacy=False,
    )


def test_plot_alternative_probabilistic_headless_noop():
    class _ProbabilisticExplanation(_FakeExplanation):
        def __init__(self):
            super().__init__(mode="classification")
            self.prediction = {"classes": 1}

        def get_class_labels(self):
            return ["no", "yes"]

    feature_predict = {
        "predict": np.array([0.3, 0.7]),
        "low": np.array([0.2, 0.6]),
        "high": np.array([0.4, 0.8]),
    }

    _plots._plot_alternative(
        explanation=_ProbabilisticExplanation(),
        instance=[0.1, 0.2],
        predict={"predict": 0.55, "low": 0.4, "high": 0.7},
        feature_predict=feature_predict,
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["a", "b"],
        title="noop",
        path=None,
        show=False,
        save_ext=None,
        use_legacy=False,
    )


def test_plot_alternative_infers_features_to_plot(tmp_path):
    """Test that plot_alternative infers features and saves files.

    Refactored to use pathlib.Path and semantic assertions.
    """
    expl = _FakeExplanation(mode="regression")
    feature_predict = {
        "predict": np.array([0.2, 0.8, 0.4]),
        "low": np.array([0.1, 0.7, 0.35]),
        "high": np.array([0.3, 0.9, 0.45]),
    }

    outdir = str(tmp_path)
    title = "alt_infer"

    _plots._plot_alternative(
        explanation=expl,
        instance=[1, 2, 3],
        predict={"predict": 0.6},
        feature_predict=feature_predict,
        features_to_plot=None,
        num_to_show=3,
        column_names=None,
        title=title,
        path=outdir + os.path.sep,
        show=False,
        save_ext=[".png"],
        use_legacy=False,
    )

    # Semantic assertion: verify that a PNG file with the title was created
    png_files = list(Path(outdir).glob("*" + title + "*.png"))
    assert len(png_files) > 0, f"Expected PNG file with title '{title}' in {outdir}"
