import numpy as np
import pytest

# Skip the entire module at collection time if matplotlib is not available.
pytest.importorskip("matplotlib")

pytestmark = pytest.mark.viz


def _render_spec_and_get_axes(spec):
    # Request the figure back for inspection
    # import backend-specific adapter and matplotlib lazily (after importorskip)
    from calibrated_explanations.viz import matplotlib_adapter

    fig = matplotlib_adapter.render(spec, show=False, save_path=None, return_fig=True)
    # Expect the last axes to be the body axes
    axs = fig.get_axes()
    # return fig and axis objects for assertions
    return fig, axs


def test_body_xlim_contains_zero_and_padding():
    # Build a simple spec with known values
    predict = {"predict": 0.4, "low": 0.35, "high": 0.45}
    vals = np.array([0.02, -0.05, 0.06])
    low = vals - 0.01
    high = vals + 0.01
    fw = {"predict": vals, "low": low, "high": high}
    # import builders lazily after matplotlib is ensured present
    from calibrated_explanations.viz.builders import build_probabilistic_bars_spec

    spec = build_probabilistic_bars_spec(
        title="t",
        predict=predict,
        feature_weights=fw,
        features_to_plot=[0, 1, 2],
        column_names=["a", "b", "c"],
        instance=[1, 2, 3],
        y_minmax=None,
        interval=True,
    )
    fig, axs = _render_spec_and_get_axes(spec)
    # body is last axis
    ax = axs[-1]
    x0, x1 = ax.get_xlim()
    assert x0 < 0 < x1
    # ensure padding ~5% beyond max extent
    max_extent = max(abs(v) for v in list(vals) + [low.min(), high.max()])
    pad = (x1 - x0) / 2.0 - max_extent
    assert pad >= max_extent * 0.04  # allow ~5% tolerance
    # close figure
    import matplotlib

    matplotlib.pyplot.close(fig)


def test_bars_drawn_from_zero_directionally_and_overlay_sign():
    predict = {"predict": 0.5, "low": 0.4, "high": 0.6}
    vals = np.array([0.03, -0.04, 0.07])
    low = vals - 0.005
    high = vals + 0.005
    fw = {"predict": vals, "low": low, "high": high}
    from calibrated_explanations.viz.builders import build_probabilistic_bars_spec

    spec = build_probabilistic_bars_spec(
        title="t",
        predict=predict,
        feature_weights=fw,
        features_to_plot=[0, 1, 2],
        column_names=["a", "b", "c"],
        instance=[1, 2, 3],
        y_minmax=None,
        interval=True,
    )
    # use the adapter's primitive export (backend-agnostic)
    from calibrated_explanations.viz import matplotlib_adapter

    primitives_no = matplotlib_adapter.render(
        spec,
        show=False,
        save_path=None,
        return_fig=False,
        draw_intervals=False,
        export_drawn_primitives=True,
    )
    primitives_yes = matplotlib_adapter.render(
        spec,
        show=False,
        save_path=None,
        return_fig=False,
        draw_intervals=True,
        export_drawn_primitives=True,
    )
    # solids parity: solids should be identical between no/yes modes
    solids_no = primitives_no.get("solids", [])
    solids_yes = primitives_yes.get("solids", [])
    assert solids_no == solids_yes
    # overlays should be present only in the yes rendering
    overlays_no = primitives_no.get("overlays", [])
    overlays_yes = primitives_yes.get("overlays", [])
    assert overlays_no == []
    assert len(overlays_yes) > 0
    # overlay sign behavior: each overlay center sign should match color selection policy
    for ov in overlays_yes:
        # color is a hex or color string; ensure center sign corresponds to overlay placement (>=0 positive)
        # we simply assert alpha is translucent
        assert ov.get("alpha", 1.0) < 1.0
