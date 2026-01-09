from calibrated_explanations.viz.builders import build_alternative_probabilistic_spec
from calibrated_explanations.viz.plotspec import PlotSpec
from calibrated_explanations.viz.plotspec import BarHPanelSpec, BarItem


def minimal_args():
    predict = {"predict": 0.5, "low": 0.2, "high": 0.8}
    feature_weights = {"predict": [0.5], "low": [0.2], "high": [0.8]}
    return dict(
        title="t",
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=[0],
        column_names=None,
        rule_labels=None,
        instance=None,
        y_minmax=None,
        interval=True,
    )


def test_builder_does_not_create_header_without_explicit_xticks():
    args = minimal_args()
    # do not supply xticks -> should not force a header unless other hints present
    ps = build_alternative_probabilistic_spec(**args)
    if isinstance(ps, PlotSpec):
        assert ps.header is None
    else:
        assert "header" not in ps.get("plot_spec", {})


def test_builder_attaches_base_lines_when_header_present():
    args = minimal_args()
    # supply explicit xticks so header_needed becomes True
    args["xticks"] = (0.0, 1.0)
    ps = build_alternative_probabilistic_spec(**args)
    if isinstance(ps, PlotSpec):
        assert getattr(ps.body, "base_lines", None) is not None
    else:
        assert ps.get("plot_spec", {}).get("body", {}).get("base_lines") is not None
