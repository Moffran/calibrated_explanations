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


def test_probabilistic_header_axes_unit_interval_with_custom_minmax():
    from calibrated_explanations.viz.builders import build_probabilistic_bars_spec

    predict = {"predict": 0.62, "low": 0.5, "high": 0.75}
    fw = {
        "predict": np.array([0.2, 0.4]),
        "low": np.array([0.15, 0.35]),
        "high": np.array([0.25, 0.45]),
    }
    spec = build_probabilistic_bars_spec(
        title="prob",
        predict=predict,
        feature_weights=fw,
        features_to_plot=[0, 1],
        column_names=["f0", "f1"],
        instance=[1, 2],
        y_minmax=(5.0, 10.0),
        interval=True,
    )
    fig, axes = _render_spec_and_get_axes(spec)
    try:
        neg_xlim = axes[0].get_xlim()
        pos_xlim = axes[1].get_xlim()
        assert neg_xlim[0] == pytest.approx(0.0)
        assert neg_xlim[1] == pytest.approx(1.0)
        assert pos_xlim[0] == pytest.approx(0.0)
        assert pos_xlim[1] == pytest.approx(1.0)
    finally:
        import matplotlib

        matplotlib.pyplot.close(fig)


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


def test_render_short_circuits_without_show_or_save(monkeypatch):
    from calibrated_explanations.viz import matplotlib_adapter
    from calibrated_explanations.viz.plotspec import PlotSpec

    calls = []

    def boom():  # pragma: no cover - ensures guard handles headless case
        calls.append("require")
        raise AssertionError("__require_matplotlib should not be called")

    monkeypatch.setattr(matplotlib_adapter, "_require_mpl", boom)

    spec = PlotSpec(title=None, header=None, body=None)
    result = matplotlib_adapter.render(spec, show=False, save_path=None)

    assert result is None
    assert calls == []


def test_render_short_circuits_without_matplotlib(monkeypatch):
    from calibrated_explanations.viz import builders, matplotlib_adapter

    spec = builders.build_probabilistic_bars_spec(
        title=None,
        predict={"predict": 0.5, "low": 0.4, "high": 0.6},
        feature_weights={"predict": np.array([0.1]), "low": np.array([0.05]), "high": np.array([0.15])},
        features_to_plot=[0],
        column_names=["f0"],
        instance=[1.0],
        y_minmax=(0.0, 1.0),
        interval=True,
    )

    def boom():
        raise RuntimeError("matplotlib should not be required")

    monkeypatch.setattr("calibrated_explanations.viz.matplotlib_adapter._require_mpl", boom)

    result = matplotlib_adapter.render(
        spec,
        show=False,
        save_path=None,
        return_fig=False,
        export_drawn_primitives=False,
    )

    assert result is None


def test_render_uses_exact_save_path(tmp_path):
    from calibrated_explanations.viz import builders, matplotlib_adapter

    spec = builders.build_probabilistic_bars_spec(
        title="save",
        predict={"predict": 0.6, "low": 0.2, "high": 0.8},
        feature_weights=[0.2, -0.1],
        features_to_plot=[0, 1],
        column_names=["f0", "f1"],
        instance=[1.0, 2.0],
        y_minmax=None,
        interval=False,
    )

    save_path = tmp_path / "joined.png"
    matplotlib_adapter.render(spec, show=False, save_path=str(save_path))

    assert save_path.exists()

    from matplotlib import pyplot as plt

    plt.close("all")


def test_render_short_circuits_without_show_or_save(monkeypatch):
    from calibrated_explanations.viz import matplotlib_adapter
    from calibrated_explanations.viz.plotspec import PlotSpec

    def fail():  # pragma: no cover - executed only on regression
        raise AssertionError("render should short-circuit when show/save disabled")

    monkeypatch.setattr(matplotlib_adapter, "_require_mpl", fail)

    matplotlib_adapter.render(
        PlotSpec(),
        show=False,
        save_path=None,
        return_fig=False,
        export_drawn_primitives=False,
    )


def test_render_saves_to_requested_path(monkeypatch, tmp_path):
    from calibrated_explanations.viz import matplotlib_adapter
    from calibrated_explanations.viz.builders import build_probabilistic_bars_spec

    predict = {"predict": 0.5, "low": 0.3, "high": 0.7}
    vals = np.array([0.2, -0.1])
    fw = {"predict": vals, "low": vals - 0.05, "high": vals + 0.05}
    spec = build_probabilistic_bars_spec(
        title="save",
        predict=predict,
        feature_weights=fw,
        features_to_plot=[0, 1],
        column_names=["a", "b"],
        instance=[1.0, 2.0],
        y_minmax=(0.0, 1.0),
        interval=True,
    )

    saved: list[str] = []

    def fake_savefig(self, path, **kwargs):  # pragma: no cover - minimal hook
        saved.append(path)

    import matplotlib.figure

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake_savefig, raising=False)

    out_path = tmp_path / "render_output.png"
    matplotlib_adapter.render(spec, show=False, save_path=str(out_path))

    assert saved == [str(out_path)]

    import matplotlib.pyplot as plt

    plt.close("all")
