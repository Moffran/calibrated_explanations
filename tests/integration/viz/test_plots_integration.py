import os
from pathlib import Path

import numpy as np
import pytest

import calibrated_explanations.viz.plots as _plots


from tests.helpers.explainer_utils import FakeExplanation


pytestmark = pytest.mark.viz


def testplot_alternative_thresholded_writes_file(tmp_path):
    """Test that plot_alternative with thresholded explanation saves files.

    Refactored to use pathlib.Path and semantic assertions.
     Thresholded explanation should hit the label/xticks/xlim branch.
    """

    class FakeExplanationThresh(FakeExplanation):
        def __init__(self):
            super().__init__(mode="regression")
            self.y_threshold = 0.25

        def is_thresholded(self):
            return True

    # n is implied by feature_predict length
    expl = FakeExplanationThresh()
    feature_predict = {
        "predict": np.array([0.2, 0.8]),
        "low": np.array([0.1, 0.7]),
        "high": np.array([0.3, 0.9]),
    }
    outdir = str(tmp_path)
    title = "alt_thresh"
    _plots.plot_alternative(
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


def test_plot_regression_legacy_fallback(tmp_path):
    """Test that plot_regression falls back to legacy when use_legacy=True."""
    # prepare feature weights as dict with interval arrays
    n = 4
    fw = {
        "predict": np.array([0.1, -0.2, 0.3, 0.0]),
        "low": np.array([0.05, -0.25, 0.25, -0.05]),
        "high": np.array([0.15, -0.15, 0.35, 0.05]),
    }
    expl = FakeExplanation(mode="regression")
    outdir = str(tmp_path)
    title = "reg_legacy"
    # Force legacy plotting
    _plots.plot_regression(
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
        idx=0,
        save_ext=[".png"],
        use_legacy=True,
    )
    # Semantic assertion: verify that a PNG file with the title was created
    png_files = list(Path(outdir).glob("*" + title + "*.png"))
    assert len(png_files) > 0, f"Expected PNG file with title '{title}' in {outdir}"

    # Verify file is non-trivial in size
    for png_file in png_files:
        assert png_file.stat().st_size > 100, f"PNG file {png_file} is too small"
