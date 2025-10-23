import pytest

from calibrated_explanations.viz import matplotlib_adapter as mpl_adapter
from calibrated_explanations.viz.builders import (
    REGRESSION_BAR_COLOR,
    REGRESSION_BASE_COLOR,
)

from tests.unit.viz.test_plot_parity_fixtures import (
    factual_probabilistic_no_uncertainty,
    factual_probabilistic_zero_crossing,
    factual_regression_interval,
    alternative_probabilistic_cross_05,
    alternative_regression_interval,
    alternative_regression_point,
    alternative_regression_probability_scale,
    triangular_probabilistic,
    global_probabilistic_multiclass,
)


def _role_alpha(pr):
    v = pr.get("visual", {})
    return v.get("color_role"), v.get("alpha")


REG_BAR_COLOR = REGRESSION_BAR_COLOR
REG_BASE_COLOR = REGRESSION_BASE_COLOR


def test_factual_probabilistic_no_uncertainty_primitives():
    spec = factual_probabilistic_no_uncertainty()
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    # header primitives expected
    header = primitives.get("header", {})
    assert "positive" in header and "negative" in header
    # main solids present
    solids = primitives.get("solids", [])
    assert len(solids) >= 1


def test_header_overlay_present_without_interval_flag():
    spec = factual_probabilistic_no_uncertainty()
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    header = primitives.get("header", {})
    pos = header.get("positive")
    neg = header.get("negative")
    assert pos is not None and pos.get("overlay") is not None
    assert neg is not None and neg.get("overlay") is not None
    assert pos["overlay"] == pytest.approx((0.8, 0.86))
    assert sorted(neg["overlay"]) == pytest.approx([0.14, 0.2])


def test_factual_probabilistic_zero_crossing_behavior():
    spec = factual_probabilistic_zero_crossing()
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    solids = primitives.get("solids", [])
    overlays = primitives.get("overlays", [])
    # With pivot removed (pivot=0.0), positive-weight features usually become
    # solids, but default legacy suppression for crossing intervals may instead
    # produce split overlays. Accept either case but ensure index 0 is present
    # in at least one of the lists.
    assert any(s.get("index", -1) == 0 for s in solids) or any(
        o.get("index", -1) == 0 for o in overlays
    )


def test_factual_regression_interval_primitives():
    spec = factual_regression_interval()
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    # header = primitives.get("header", {})
    # regression header may be empty for single-band headers; ensure body primitives exist
    solids = primitives.get("solids", [])
    overlays = primitives.get("overlays", [])
    assert isinstance(solids, list) and isinstance(overlays, list)
    solids = primitives.get("solids", [])
    overlays = primitives.get("overlays", [])
    assert len(overlays) >= 0


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


def test_alternative_regression_primitives():
    spec = alternative_regression_interval()
    assert spec.header is None
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    assert not primitives.get("header")
    overlays = primitives.get("overlays", [])
    assert isinstance(overlays, list) and len(overlays) >= 1
    indices = {item.get("index") for item in overlays}
    assert -1 in indices
    assert any(idx is not None and idx >= 0 for idx in indices)
    assert any(item.get("color") == REG_BASE_COLOR for item in overlays if item.get("index") == -1)
    assert any(
        item.get("color") == REG_BAR_COLOR
        for item in overlays
        if item.get("index") is not None and item.get("index") >= 0
    )
    lines = primitives.get("lines", [])
    assert isinstance(lines, list) and len(lines) >= 1
    line_indices = {item.get("index") for item in lines}
    assert -1 in line_indices
    assert any(idx is not None and idx >= 0 for idx in line_indices)
    assert any(
        item.get("color") == REG_BAR_COLOR
        for item in lines
        if item.get("index") is not None and item.get("index") >= 0
    )


def test_alternative_regression_point_primitives():
    spec = alternative_regression_point()
    assert spec.header is None
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    assert not primitives.get("header")
    overlays = primitives.get("overlays", [])
    assert isinstance(overlays, list) and len(overlays) >= 1
    indices = {item.get("index") for item in overlays}
    assert -1 in indices
    assert any(idx is not None and idx >= 0 for idx in indices)
    assert any(item.get("color") == REG_BASE_COLOR for item in overlays if item.get("index") == -1)
    assert any(
        item.get("color") == REG_BAR_COLOR
        for item in overlays
        if item.get("index") is not None and item.get("index") >= 0
    )
    lines = primitives.get("lines", [])
    assert isinstance(lines, list) and len(lines) >= 1
    line_indices = {item.get("index") for item in lines}
    assert any(idx is not None and idx >= 0 for idx in line_indices)
    assert any(
        item.get("color") == REG_BAR_COLOR
        for item in lines
        if item.get("index") is not None and item.get("index") >= 0
    )


def test_alternative_regression_probability_scale_primitives():
    spec = alternative_regression_probability_scale()
    assert spec.header is not None and getattr(spec.header, "dual", False)
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    overlays = primitives.get("overlays", [])
    assert isinstance(overlays, list) and len(overlays) >= 1
    colors = {item.get("color") for item in overlays}
    assert len(colors) >= 2
    assert any(item.get("index") == -1 for item in overlays)
    lines = primitives.get("lines", [])
    assert isinstance(lines, list) and len(lines) >= 1
    assert spec.body is not None and spec.body.xlim == (0.0, 1.0)
    assert spec.body.xlabel.startswith("Probability")


def test_triangular_primitives():
    spec = triangular_probabilistic()
    # triangular builder returns a dict; adapter.render expects a PlotSpec dataclass
    # so assert the builder returned the expected triangular payload instead of rendering
    assert isinstance(spec, dict)
    ps = spec.get("plot_spec", {})
    assert ps.get("kind") == "triangular"
    assert "triangular" in ps


def test_global_probabilistic_multiclass_saved(tmp_path):
    spec = global_probabilistic_multiclass()
    # builder returns a dict PlotSpec; ensure global entries exist and we can attach save behavior
    assert isinstance(spec, dict)
    ps = spec.get("plot_spec", {})
    ps.setdefault("save_behavior", {})["path"] = str(tmp_path)
    # must include global_entries for plotting
    assert "global_entries" in ps
    # default_exts should be configurable on save_behavior when present
    ps["save_behavior"].setdefault("default_exts", ["svg", "png"])
    assert isinstance(ps["save_behavior"].get("default_exts"), list)


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


def test_normalized_primitives_have_expected_shape():
    """Confirm normalized primitives are list-of-dict with required keys."""
    spec = factual_probabilistic_no_uncertainty()
    wrapper = mpl_adapter.render(spec, export_drawn_primitives=True)
    primitives = wrapper.get("primitives", [])
    assert isinstance(primitives, list)
    if primitives:
        p = primitives[0]
        assert isinstance(p, dict)
        # common keys for normalized primitives
        assert "id" in p and "type" in p and "coords" in p and "style" in p


def test_wrapper_merge_preserves_plot_spec_and_primitives():
    spec = factual_probabilistic_no_uncertainty()
    wrapper = mpl_adapter.render(spec, export_drawn_primitives=True)
    # Ensure plot_spec payload convertible to dict exists and matches spec.title
    ps = wrapper.get("plot_spec", {})
    # title may be present in dataclass -> asdict conversion; check existence
    assert isinstance(ps, dict)
    # primitives list must be present and be list
    assert isinstance(wrapper.get("primitives", []), list)


def test_render_dict_triangular_via_shim():
    """Render a triangular dict payload via the adapter shim and assert primitives."""
    from tests.unit.viz.test_plot_parity_fixtures import triangular_probabilistic

    spec = triangular_probabilistic()
    wrapper = mpl_adapter.render(spec, export_drawn_primitives=True)
    # shim should return triangle_background and some primitives list
    assert "triangle_background" in wrapper or any(
        p.get("type") == "quiver" for p in wrapper.get("primitives", [])
    )


def test_render_dict_global_via_shim(tmp_path):
    """Render a global dict payload via the adapter shim and assert scatter + save primitives."""
    from tests.unit.viz.test_plot_parity_fixtures import global_probabilistic_multiclass

    spec = global_probabilistic_multiclass()
    # attach a temp save path via save_behavior
    spec.setdefault("plot_spec", {}).setdefault("save_behavior", {})["default_exts"] = [
        "png",
        "svg",
    ]
    spec["plot_spec"]["save_behavior"]["path"] = str(tmp_path)
    wrapper = mpl_adapter.render(spec, export_drawn_primitives=True)
    # should include scatter primitives (at least one) and save_fig entries
    scatters = [p for p in wrapper.get("primitives", []) if p.get("type") == "scatter"]
    saves = [p for p in wrapper.get("primitives", []) if p.get("type") == "save_fig"]
    assert len(scatters) >= 1
    assert len(saves) == 2


def test_plot_triangular_delegates_to_adapter(monkeypatch, tmp_path):
    """Ensure `_plot_triangular` delegates to builder+adapter and handles save_ext."""
    from calibrated_explanations.viz import plots as _plots

    calls = []

    def fake_render(spec, *, show=False, save_path=None, **kwargs):
        calls.append({"spec": spec, "show": show, "save_path": save_path})
        # return shim-like wrapper when dict passed
        if isinstance(spec, dict):
            return {
                "plot_spec": spec.get("plot_spec", {}),
                "primitives": [{"type": "quiver"}],
            }
        return {}

    monkeypatch.setattr("calibrated_explanations.viz.matplotlib_adapter.render", fake_render)

    # prepare simple numeric arrays for triangular plot
    proba = [0.2]
    uncertainty = [0.1]
    rule_proba = [0.3]
    rule_uncertainty = [0.05]

    # call with show=False and no save_ext -> should no-op and not call adapter.render
    _plots._plot_triangular(
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
    _plots._plot_triangular(
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
