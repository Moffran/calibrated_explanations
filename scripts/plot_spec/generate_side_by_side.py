"""Generate side-by-side PlotSpec vs legacy visual evidence for Task 9 review.

Run with:
    python scripts/plot_spec/generate_side_by_side.py [--out reports/plot_parity/v0.11.2_side_by_side]

Produces one PNG per family showing legacy (left) and PlotSpec-mended (right)
renders of an identical deterministic input fixture.  Output filenames follow
the convention ``<family>_legacy.png`` and ``<family>_plotspec.png`` so they
can be reviewed side-by-side in any image viewer or placed in the gallery doc.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

# ensure package is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from calibrated_explanations.viz import (
    build_probabilistic_bars_spec,
    build_regression_bars_spec,
    build_alternative_probabilistic_spec,
    build_alternative_regression_spec,
    build_triangular_plotspec,
    build_global_plotspec,
)
from calibrated_explanations.viz import matplotlib_adapter as mpl_adapter


# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------

class _MinimalExplanation:
    """Minimal stub satisfying all legacy plotting function attribute requirements."""

    def __init__(self, mode="classification", y_minmax=(0.0, 1.0)):
        self._mode = mode
        self.y_minmax = y_minmax
        self.prediction = {"classes": 1}

        class _CE:
            def get_confidence(self_inner):
                return 95

        self.calibrated_explanations = _CE()

    def get_mode(self):
        return self._mode

    def is_thresholded(self):
        return False

    def is_one_sided(self):
        return False

    def get_class_labels(self):
        return None


# ---------------------------------------------------------------------------
# Factual probabilistic
# ---------------------------------------------------------------------------

def _factual_probabilistic_inputs():
    predict = {"predict": 0.72, "low": 0.65, "high": 0.79}
    feature_weights = {"predict": [0.35, -0.18, 0.12], "low": [0.28, -0.23, 0.07], "high": [0.42, -0.13, 0.17]}
    features_to_plot = [0, 1, 2]
    column_names = ["age > 45", "income < 30k", "education = high"]
    instance = [52.0, 28500.0, "high"]
    return predict, feature_weights, features_to_plot, column_names, instance


def render_factual_probabilistic_legacy(out_dir: Path) -> None:
    from calibrated_explanations.legacy import plotting as legacy

    predict, feature_weights, features_to_plot, column_names, instance = _factual_probabilistic_inputs()
    exp = _MinimalExplanation()

    legacy._plot_probabilistic(
        exp, instance, predict, feature_weights, features_to_plot,
        len(features_to_plot), column_names,
        "factual_probabilistic_legacy",
        str(out_dir) + "/", False, interval=True, idx=0, save_ext=[".png"],
    )


def render_factual_probabilistic_plotspec(out_dir: Path) -> None:
    predict, feature_weights, features_to_plot, column_names, instance = _factual_probabilistic_inputs()
    spec = build_probabilistic_bars_spec(
        title=None,
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=features_to_plot,
        column_names=column_names,
        instance=instance,
        y_minmax=[0.0, 1.0],
        interval=True,
        neg_caption="P(y=0)",
        pos_caption="P(y=1)",
    )
    mpl_adapter.render(spec, show=False, save_path=str(out_dir / "factual_probabilistic_plotspec.png"))


# ---------------------------------------------------------------------------
# Factual regression
# ---------------------------------------------------------------------------

def _factual_regression_inputs():
    predict = {"predict": 185000.0, "low": 172000.0, "high": 198000.0}
    feature_weights = {"predict": [12000.0, -8500.0, 5200.0], "low": [9000.0, -11000.0, 3800.0], "high": [15000.0, -6000.0, 6600.0]}
    features_to_plot = [0, 1, 2]
    column_names = ["floor_area > 80m²", "location = rural", "age_of_building < 10y"]
    instance = [95.0, 0.0, 3.0]
    return predict, feature_weights, features_to_plot, column_names, instance


def render_factual_regression_legacy(out_dir: Path) -> None:
    from calibrated_explanations.legacy import plotting as legacy

    predict, feature_weights, features_to_plot, column_names, instance = _factual_regression_inputs()
    exp = _MinimalExplanation(mode="regression", y_minmax=(100000.0, 500000.0))
    legacy.plot_regression(
        exp, instance, predict, feature_weights, features_to_plot,
        len(features_to_plot), column_names,
        "factual_regression_legacy",
        str(out_dir) + "/", False, interval=True, idx=0, save_ext=[".png"],
    )


def render_factual_regression_plotspec(out_dir: Path) -> None:
    predict, feature_weights, features_to_plot, column_names, instance = _factual_regression_inputs()
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=features_to_plot,
        column_names=column_names,
        instance=instance,
        y_minmax=[100000.0, 500000.0],
        interval=True,
    )
    mpl_adapter.render(spec, show=False, save_path=str(out_dir / "factual_regression_plotspec.png"))


# ---------------------------------------------------------------------------
# Alternative probabilistic
# ---------------------------------------------------------------------------

def _alternative_probabilistic_inputs():
    predict = {"predict": 0.65, "low": 0.52, "high": 0.78}
    feature_predict = {
        "predict": [0.35, 0.72, 0.48],
        "low": [0.22, 0.61, 0.38],
        "high": [0.48, 0.83, 0.58],
    }
    features_to_plot = [0, 1, 2]
    column_names = ["age ≤ 40", "income ≥ 50k", "education = low"]
    instance = [35.0, 52000.0, "low"]
    return predict, feature_predict, features_to_plot, column_names, instance


def render_alternative_probabilistic_legacy(out_dir: Path) -> None:
    from calibrated_explanations.legacy import plotting as legacy

    predict, feature_predict, features_to_plot, column_names, instance = _alternative_probabilistic_inputs()
    exp = _MinimalExplanation()
    legacy.plot_alternative(
        exp, instance, predict, feature_predict, features_to_plot,
        len(features_to_plot), column_names,
        "alternative_probabilistic_legacy",
        str(out_dir) + "/", False, save_ext=[".png"],
    )


def render_alternative_probabilistic_plotspec(out_dir: Path) -> None:
    predict, feature_predict, features_to_plot, column_names, instance = _alternative_probabilistic_inputs()
    spec = build_alternative_probabilistic_spec(
        title=None,
        predict=predict,
        feature_weights=feature_predict,
        features_to_plot=features_to_plot,
        column_names=column_names,
        instance=instance,
        y_minmax=[0.0, 1.0],
        interval=True,
        xlim=(0.0, 1.0),
    )
    mpl_adapter.render(spec, show=False, save_path=str(out_dir / "alternative_probabilistic_plotspec.png"))


# ---------------------------------------------------------------------------
# Alternative regression
# ---------------------------------------------------------------------------

def _alternative_regression_inputs():
    predict = {"predict": 185000.0, "low": 162000.0, "high": 208000.0}
    feature_predict = {
        "predict": [210000.0, 168000.0, 192000.0],
        "low": [195000.0, 152000.0, 179000.0],
        "high": [225000.0, 184000.0, 205000.0],
    }
    features_to_plot = [0, 1, 2]
    column_names = ["floor_area ≤ 60m²", "location = urban", "age_of_building > 30y"]
    instance = [55.0, 1.0, 35.0]
    return predict, feature_predict, features_to_plot, column_names, instance


def render_alternative_regression_legacy(out_dir: Path) -> None:
    from calibrated_explanations.legacy import plotting as legacy

    predict, feature_predict, features_to_plot, column_names, instance = _alternative_regression_inputs()
    exp = _MinimalExplanation(mode="regression", y_minmax=(100000.0, 500000.0))
    legacy.plot_alternative(
        exp, instance, predict, feature_predict, features_to_plot,
        len(features_to_plot), column_names,
        "alternative_regression_legacy",
        str(out_dir) + "/", False, save_ext=[".png"],
    )


def render_alternative_regression_plotspec(out_dir: Path) -> None:
    predict, feature_predict, features_to_plot, column_names, instance = _alternative_regression_inputs()
    spec = build_alternative_regression_spec(
        title=None,
        predict=predict,
        feature_weights=feature_predict,
        features_to_plot=features_to_plot,
        column_names=column_names,
        instance=instance,
        y_minmax=[100000.0, 500000.0],
        interval=True,
    )
    mpl_adapter.render(spec, show=False, save_path=str(out_dir / "alternative_regression_plotspec.png"))


# ---------------------------------------------------------------------------
# Triangular
# ---------------------------------------------------------------------------

def render_triangular_legacy(out_dir: Path) -> None:
    from calibrated_explanations.legacy import plotting as legacy

    proba = np.array([0.72])
    uncertainty = np.array([0.08])
    rule_proba = np.array([0.45, 0.55, 0.60])
    rule_uncertainty = np.array([0.05, 0.03, 0.04])
    exp = _MinimalExplanation()
    legacy.plot_triangular(
        exp, proba, uncertainty, rule_proba, rule_uncertainty,
        3, "triangular_legacy",
        str(out_dir) + "/", False, save_ext=[".png"],
    )


def render_triangular_plotspec(out_dir: Path) -> None:
    proba = np.array([0.72])
    uncertainty = np.array([0.08])
    rule_proba = np.array([0.45, 0.55, 0.60])
    rule_uncertainty = np.array([0.05, 0.03, 0.04])
    spec = build_triangular_plotspec(
        title=None,
        proba=proba,
        uncertainty=uncertainty,
        rule_proba=rule_proba,
        rule_uncertainty=rule_uncertainty,
        num_to_show=3,
    )
    mpl_adapter.render(spec, show=False, save_path=str(out_dir / "triangular_plotspec.png"))


# ---------------------------------------------------------------------------
# Shared legacy plt initializer
# ---------------------------------------------------------------------------

def _init_legacy_plt(legacy) -> None:
    """Pre-initialize legacy.plt so plot_global uses the same matplotlib instance."""
    if legacy.plt is None:
        import matplotlib.pyplot as _mpl_plt
        import matplotlib.colors as _mpl_colors
        legacy.plt = _mpl_plt
        legacy.mcolors = _mpl_colors


def _capture_new_figure(legacy, figs_before: set):
    """Return the figure created since figs_before was recorded."""
    figs_after = set(legacy.plt.get_fignums())
    new_figs = figs_after - figs_before
    if new_figs:
        return legacy.plt.figure(sorted(new_figs)[-1])
    return legacy.plt.gcf()


# ---------------------------------------------------------------------------
# Global (classification / probabilistic)
# ---------------------------------------------------------------------------

def render_global_legacy(out_dir: Path) -> None:
    from calibrated_explanations.legacy import plotting as legacy
    _init_legacy_plt(legacy)

    rng = np.random.default_rng(42)
    n = 20
    raw = rng.uniform(0, 1, n)
    proba_arr = np.column_stack([1 - raw, raw])
    # 1D low/high (column 1 only) to avoid shape mismatch in legacy scatter
    low_1d = np.clip(raw - 0.05, 0.0, 1.0)
    high_1d = np.clip(raw + 0.05, 0.0, 1.0)
    y_test = (raw > 0.5).astype(int)

    class _FakeExplainer:
        class learner:
            predict_proba = staticmethod(lambda X: np.column_stack([1 - X[:, 0], X[:, 0]]))

        class_labels = None

        def is_multiclass(self):
            return False

        def predict_proba(self, X, uq_interval=False, threshold=None, **kwargs):
            return proba_arr, (low_1d, high_1d)

    explainer = _FakeExplainer()
    try:
        x_dummy = np.zeros((n, 1))
        figs_before = set(legacy.plt.get_fignums())
        legacy.plot_global(
            explainer, x_dummy, y_test, threshold=None,
            show=False,
        )
        fig = _capture_new_figure(legacy, figs_before)
        fig.savefig(str(out_dir / "global_legacy.png"), bbox_inches="tight")
        legacy.plt.close(fig)
    except Exception as exc:
        import traceback
        print(f"  [warn] Legacy global render skipped: {exc}")
        traceback.print_exc()


def render_global_plotspec(out_dir: Path) -> None:
    rng = np.random.default_rng(42)
    n = 20
    raw = rng.uniform(0, 1, n)
    proba = np.column_stack([1 - raw, raw])
    low = proba - 0.05
    high = proba + 0.05
    uncertainty = high - low
    y_test = (raw > 0.5).astype(int)
    spec = build_global_plotspec(
        title=None,
        proba=proba,
        predict=None,
        low=low,
        high=high,
        uncertainty=uncertainty,
        y_test=y_test,
        is_regularized=True,
    )
    mpl_adapter.render(spec, show=False, save_path=str(out_dir / "global_plotspec.png"))


# ---------------------------------------------------------------------------
# Global (regression)
# ---------------------------------------------------------------------------

def _global_regression_inputs():
    """Synthetic regression global data with DifficultyEstimator-style varying uncertainties.

    The half-interval width is drawn from a log-normal distribution so that
    harder (less-certain) instances have much wider intervals than easy ones.
    Constant uncertainty collapses the y-axis in legacy's plot_global.
    """
    rng = np.random.default_rng(42)
    n = 30
    predict_1d = rng.uniform(100, 500, n)
    y_test = predict_1d + rng.normal(0, 30, n)
    # Uncertainty proportional to prediction magnitude + noise → mimics DifficultyEstimator
    half_width = np.abs(rng.normal(0, 1, n)) * 20 + 5   # range roughly [5, 60]
    low_1d = predict_1d - half_width
    high_1d = predict_1d + half_width
    uncertainty_1d = high_1d - low_1d
    return predict_1d, low_1d, high_1d, uncertainty_1d, y_test


def render_global_legacy_regression(out_dir: Path) -> None:
    from calibrated_explanations.legacy import plotting as legacy
    _init_legacy_plt(legacy)

    predict_1d, low_1d, high_1d, _, y_test = _global_regression_inputs()

    class _FakeRegExplainer:
        class learner:
            pass  # no predict_proba attribute → regression path

        y_cal = y_test

        def predict(self_inner, X, uq_interval=False, **kwargs):
            return predict_1d, (low_1d, high_1d)

    explainer = _FakeRegExplainer()
    try:
        x_dummy = np.zeros((len(predict_1d), 1))
        figs_before = set(legacy.plt.get_fignums())
        legacy.plot_global(
            explainer, x_dummy, y_test, threshold=None,
            show=False,
        )
        fig = _capture_new_figure(legacy, figs_before)
        fig.savefig(str(out_dir / "global_legacy_regression.png"), bbox_inches="tight")
        legacy.plt.close(fig)
    except Exception as exc:
        import traceback
        print(f"  [warn] Legacy global regression render skipped: {exc}")
        traceback.print_exc()


def render_global_plotspec_regression(out_dir: Path) -> None:
    predict_1d, low_1d, high_1d, uncertainty_1d, y_test = _global_regression_inputs()
    spec = build_global_plotspec(
        title=None,
        proba=None,
        predict=predict_1d.tolist(),
        low=low_1d.tolist(),
        high=high_1d.tolist(),
        uncertainty=uncertainty_1d.tolist(),
        y_test=y_test.tolist(),
        is_regularized=False,
    )
    mpl_adapter.render(spec, show=False, save_path=str(out_dir / "global_plotspec_regression.png"))


# ---------------------------------------------------------------------------
# Conjunction (multi-line labels — demonstrates \n height fix)
# ---------------------------------------------------------------------------
# These fixtures use conjunction rule descriptions with \n line breaks, which
# is the real-world format produced by conjunctive explanations.  Legacy plots
# clip or overlap such labels; the PlotSpec adapter auto-expands figure height
# by 0.5" per extra line so all labels remain legible.

def _conjunction_inputs():
    predict = {"predict": 0.68, "low": 0.58, "high": 0.78}
    feature_weights = {
        "predict": [0.31, -0.22, 0.14, -0.09],
        "low":     [0.24, -0.28, 0.08, -0.15],
        "high":    [0.38, -0.16, 0.20, -0.03],
    }
    features_to_plot = [0, 1, 2, 3]
    column_names = [
        "age > 45\nAND income < 30k",
        "education = high\nAND credit_score > 700",
        "loan_term <= 24m\nAND employment = full-time\nAND region = urban",
        "dependents = 0",
    ]
    instance = [52.0, 28500.0, "high\n700\n18m\nfull-time\nurban", 0]
    return predict, feature_weights, features_to_plot, column_names, instance


def render_conjunction_legacy(out_dir: Path) -> None:
    from calibrated_explanations.legacy import plotting as legacy

    predict, feature_weights, features_to_plot, column_names, instance = _conjunction_inputs()
    exp = _MinimalExplanation()
    legacy._plot_probabilistic(
        exp, instance, predict, feature_weights, features_to_plot,
        len(features_to_plot), column_names,
        "conjunction_legacy",
        str(out_dir) + "/", False, interval=True, idx=0, save_ext=[".png"],
    )


def render_conjunction_plotspec(out_dir: Path) -> None:
    predict, feature_weights, features_to_plot, column_names, instance = _conjunction_inputs()
    spec = build_probabilistic_bars_spec(
        title=None,
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=features_to_plot,
        column_names=column_names,
        instance=instance,
        y_minmax=[0.0, 1.0],
        interval=True,
        neg_caption="P(y=0)",
        pos_caption="P(y=1)",
    )
    mpl_adapter.render(spec, show=False, save_path=str(out_dir / "conjunction_plotspec.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FAMILIES = [
    ("factual_probabilistic", render_factual_probabilistic_legacy, render_factual_probabilistic_plotspec),
    ("factual_regression", render_factual_regression_legacy, render_factual_regression_plotspec),
    ("alternative_probabilistic", render_alternative_probabilistic_legacy, render_alternative_probabilistic_plotspec),
    ("alternative_regression", render_alternative_regression_legacy, render_alternative_regression_plotspec),
    ("triangular", render_triangular_legacy, render_triangular_plotspec),
    ("global", render_global_legacy, render_global_plotspec),
    ("global_regression", render_global_legacy_regression, render_global_plotspec_regression),
    ("conjunction", render_conjunction_legacy, render_conjunction_plotspec),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="reports/plot_parity/v0.11.2_side_by_side",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--family",
        choices=[f[0] for f in FAMILIES],
        default=None,
        help="Render a single family (default: all)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [f for f in FAMILIES if args.family is None or f[0] == args.family]

    for name, render_legacy, render_plotspec in targets:
        print(f"  {name}...")
        try:
            render_legacy(out_dir)
        except Exception as exc:
            print(f"    [warn] legacy render failed: {exc}")
        try:
            render_plotspec(out_dir)
        except Exception as exc:
            print(f"    [warn] plotspec render failed: {exc}")

    print(f"\nImages written to: {out_dir}")
    written = sorted(out_dir.glob("*.png"))
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
