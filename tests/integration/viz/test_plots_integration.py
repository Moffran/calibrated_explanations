import os

import numpy as np

from calibrated_explanations import _plots


class _FakeExplanation:
    def __init__(self, mode="regression"):
        self._mode = mode
        self.y_minmax = (0.0, 1.0)

        class _CE:
            def get_confidence(self_inner):
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

    def predict(self, X, uq_interval=False, bins=None):
        # return (predictions, (low, high)) similar shape to usage
        preds = np.zeros(len(X))
        low = np.zeros(len(X))
        high = np.ones(len(X))
        return preds, (low, high)

    def is_multiclass(self):
        return False


def test_plot_regression_writes_file(tmp_path):
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
    )
    assert os.path.exists(os.path.join(outdir, title + ".png"))


def test_plot_alternative_writes_file(tmp_path):
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
    )
    assert os.path.exists(os.path.join(outdir, title + ".png"))


def test_plot_global_non_probabilistic_runs_without_error():
    expl = _FakeExplainer()
    # small X_test
    X_test = np.zeros((3, 2))
    # should not raise
    _plots._plot_global(expl, X_test, y_test=None, threshold=None, show=False)


def test_plot_proba_triangle_returns_figure():
    fig = _plots._plot_proba_triangle()
    assert fig is not None


def test_plot_alternative_thresholded_writes_file(tmp_path):
    # thresholded explanation should hit the label/xticks/xlim branch
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
    )
    assert os.path.exists(os.path.join(outdir, title + ".png"))


def test_plot_global_probabilistic_branch_runs():
    # fake explainer with predict_proba to exercise probabilistic branch
    class _FakeExplainerProba:
        def __init__(self):
            # the plotting helper checks explainer.learner for predict_proba
            self.learner = self

        # accept optional args used by plotting helper
        def predict_proba(self, X, uq_interval=False, threshold=None, bins=None):
            proba = np.tile(np.array([[0.3, 0.7]]), (len(X), 1))
            # return per-class low/high arrays matching proba shape
            low = np.zeros_like(proba)
            high = np.ones_like(proba)
            return proba, (low, high)

        def is_multiclass(self):
            return True

        def predict(self, X, uq_interval=False, bins=None):
            return np.zeros(len(X)), (np.zeros(len(X)), np.ones(len(X)))

    expl = _FakeExplainerProba()
    X_test = np.zeros((3, 2))
    # should not raise
    _plots._plot_global(expl, X_test, y_test=None, threshold=None, show=False)


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
    )
