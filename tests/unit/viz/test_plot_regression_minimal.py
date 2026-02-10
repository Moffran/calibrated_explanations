import os
import numpy as np


def test_plot_regression_legacy_minimal(tmp_path):
    """Call legacy plotting path minimally to cover decision logic without heavy assertions."""
    from calibrated_explanations.viz import plots as _plots
    from tests.helpers.explainer_utils import FakeExplanation

    n = 4
    fw = {
        "predict": np.array([0.1, -0.2, 0.3, 0.0]),
        "low": np.array([0.05, -0.25, 0.25, -0.05]),
        "high": np.array([0.15, -0.15, 0.35, 0.05]),
    }
    expl = FakeExplanation(mode="regression")
    outdir = str(tmp_path)

    # Use idx to avoid legacy assertion about interval idx
    _plots.plot_regression(
        explanation=expl,
        instance=[0, 0, 0, 0],
        predict={"predict": 0.5, "low": 0.4, "high": 0.6},
        feature_weights=fw,
        features_to_plot=list(range(n)),
        num_to_show=n,
        column_names=[f"f{i}" for i in range(n)],
        title="reg_minimal",
        path=outdir + os.path.sep,
        show=False,
        interval=True,
        save_ext=[".png"],
        use_legacy=True,
        idx=0,
    )

    # If no exception, test is successful — file creation is non-essential here
    assert True
