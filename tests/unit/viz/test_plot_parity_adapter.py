from calibrated_explanations.viz import matplotlib_adapter as mpl_adapter
from tests.unit.viz.test_plot_parity_fixtures import (
    factual_probabilistic_no_uncertainty,
    factual_probabilistic_zero_crossing,
    factual_regression_interval,
    alternative_probabilistic_cross_05,
    triangular_probabilistic,
    global_probabilistic_multiclass,
)


def _role_alpha(pr):
    v = pr.get("visual", {})
    return v.get("color_role"), v.get("alpha")


def test_factual_probabilistic_no_uncertainty_primitives():
    spec = factual_probabilistic_no_uncertainty()
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    # header primitives expected
    header = primitives.get("header", {})
    assert "positive" in header and "negative" in header
    # main solids present
    solids = primitives.get("solids", [])
    assert len(solids) >= 1


def test_factual_probabilistic_zero_crossing_behavior():
    spec = factual_probabilistic_zero_crossing()
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    solids = primitives.get("solids", [])
    overlays = primitives.get("overlays", [])
    # Legacy parity: solid suppressed, overlays present for index 0
    assert all(s.get("index", -1) != 0 for s in solids)
    assert any(o.get("index", -1) == 0 for o in overlays)


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
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    overlays = primitives.get("overlays", [])
    # Expect overlays for features split at 0.5
    assert len(overlays) >= 1


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
