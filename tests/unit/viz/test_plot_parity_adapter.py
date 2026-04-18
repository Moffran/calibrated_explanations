import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

from calibrated_explanations.viz import matplotlib_adapter as mpl_adapter
from calibrated_explanations.viz import (
    REGRESSION_BAR_COLOR,
    REGRESSION_BASE_COLOR,
)

from tests.unit.viz.test_plot_parity_fixtures import (
    factual_probabilistic_no_uncertainty,
    alternative_probabilistic_cross_05,
    global_probabilistic_multiclass,
)


REG_BAR_COLOR = REGRESSION_BAR_COLOR
REG_BASE_COLOR = REGRESSION_BASE_COLOR


def test_alternative_probabilistic_cross_primitives():
    spec = alternative_probabilistic_cross_05()
    assert spec.header is None
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    assert not primitives.get("header")
    overlays = primitives.get("overlays", [])
    assert isinstance(overlays, list) and len(overlays) >= 1
    indices = {item.get("index") for item in overlays}
    # Base interval should be present (index -1) alongside feature overlays (>=0)
    assert -1 in indices
    assert any(idx is not None and idx >= 0 for idx in indices)


def test_global_probabilistic_multiclass_saved(tmp_path):
    spec = global_probabilistic_multiclass()
    # builder returns canonical dataclass; save behavior is configured on the
    # dataclass and converted at serializer boundary.
    assert spec.global_entries is not None
    spec.save_behavior.path = str(tmp_path)
    assert spec.save_behavior is not None
    assert tuple(spec.save_behavior.default_exts or ()) == ("svg", "png")


def test_adapter_returns_normalized_and_legacy_for_plotspec():
    """Ensure adapter returns both normalized `primitives` and legacy keys for PlotSpec dataclass."""
    # use factual_probabilistic_no_uncertainty (returns a PlotSpec dataclass)
    spec = factual_probabilistic_no_uncertainty()
    wrapper = mpl_adapter.render(spec, export_drawn_primitives=True)
    # wrapper must include top-level plot_spec and primitives list
    assert isinstance(wrapper, dict)
    assert "plot_spec" in wrapper and "primitives" in wrapper
    # legacy keys like solids/overlays/header should also be present (may be empty)
    assert any(k in wrapper for k in ("solids", "overlays", "header", "base_interval"))


def testplot_triangular_delegates_to_adapter(monkeypatch, tmp_path):
    """Ensure `plot_triangular` delegates to builder+adapter and handles save_ext."""
    from calibrated_explanations.viz import plots as _plots

    calls = []

    def fake_render(spec, *, show=False, save_path=None, **kwargs):
        calls.append({"spec": spec, "show": show, "save_path": save_path})
        return {}

    monkeypatch.setattr("calibrated_explanations.viz.matplotlib_adapter.render", fake_render)

    # prepare simple numeric arrays for triangular plot
    proba = [0.2]
    uncertainty = [0.1]
    rule_proba = [0.3]
    rule_uncertainty = [0.05]

    # call with show=False and no save_ext -> should no-op and not call adapter.render
    _plots.plot_triangular(
        None,
        proba,
        uncertainty,
        rule_proba,
        rule_uncertainty,
        1,
        "t",
        None,
        False,
        save_ext=None,
        use_legacy=False,
    )
    assert len(calls) == 0

    # Reset and call with save_ext to trigger adapter.save behavior
    calls.clear()
    _plots.plot_triangular(
        None,
        proba,
        uncertainty,
        rule_proba,
        rule_uncertainty,
        1,
        "t",
        str(tmp_path) + "/",
        False,
        save_ext=["png"],
        use_legacy=False,
    )  # noqa: E501
    # adapter.render should be invoked for initial render + each save ext
    assert len(calls) >= 2
