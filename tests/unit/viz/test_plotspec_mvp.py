import os
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

from calibrated_explanations import plotting
from calibrated_explanations.viz import build_regression_bars_spec, matplotlib_adapter


pytest.importorskip("matplotlib")
pytestmark = pytest.mark.viz


def test_plotspec_regression_render_smoke():
    rng = np.random.default_rng(0)
    nfeat = 5
    # Fake inputs similar to _plot_regression
    predict = {"predict": 0.5, "low": 0.2, "high": 0.8}
    fw = {
        "predict": rng.normal(0, 0.2, size=nfeat),
        "low": rng.normal(-0.1, 0.1, size=nfeat),
        "high": rng.normal(0.1, 0.1, size=nfeat),
    }
    feats = list(range(nfeat))
    cols = [f"f{i}" for i in range(nfeat)]
    instance = rng.normal(size=nfeat)
    spec = build_regression_bars_spec(
        title="MVP",
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=instance,
        y_minmax=(0.0, 1.0),
        interval=True,
    )
    # Render to a temp file; should not raise
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "mvp.png")
        matplotlib_adapter.render(spec, show=False, save_path=out)
        assert os.path.exists(out)


def test_plot_probabilistic_requires_idx_when_interval(monkeypatch):
    """PlotSpec-backed probabilistic plots must enforce the interval guard."""

    monkeypatch.setattr(plotting, "__require_matplotlib", lambda: None)

    explanation = types.SimpleNamespace(
        y_minmax=(0.0, 1.0),
        prediction={"classes": 1},
        get_class_labels=lambda: ["neg", "pos"],
        is_thresholded=lambda: False,
        get_mode=lambda: "classification",
        is_one_sided=lambda: False,
    )
    setattr(explanation, "_get_explainer", lambda: None)

    with pytest.raises(AssertionError):
        plotting._plot_probabilistic(
            explanation,
            instance=np.array([0.2, 0.4]),
            predict={"predict": 0.6, "low": 0.2, "high": 0.8},
            feature_weights={
                "predict": np.array([0.1, -0.1]),
                "low": np.array([0.0, 0.0]),
                "high": np.array([0.2, 0.2]),
            },
            features_to_plot=[0, 1],
            num_to_show=2,
            column_names=["f0", "f1"],
            title="interval",
            path="",
            show=True,
            interval=True,
            idx=None,
            save_ext=None,
            use_legacy=False,
        )


@pytest.mark.platform_dependent
def test_plot_regression_default_save_paths_include_title(monkeypatch, tmp_path):
    """PlotSpec regression helper should save to multiple formats when path specified.

    Refactored from brittle os.path.join assertion to semantic
    assertions on file formats and presence. Tests that:
    - Files are created in the specified directory
    - Files use expected formats (svg, pdf, png)
    - Files contain the title in their names

    Note: Does NOT test exact path strings (platform-dependent, fragile).
    Uses pathlib.Path for cross-platform compatibility.

    Marked as @pytest.mark.platform_dependent because path concatenation
    behavior may vary across Windows/POSIX systems despite pathlib usage.
    """

    render_calls: list[dict] = []

    def fake_render(spec, **kwargs):  # pragma: no cover - spy helper
        render_calls.append({"show": kwargs.get("show"), "save_path": kwargs.get("save_path")})

    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        fake_render,
    )

    explanation = types.SimpleNamespace(y_minmax=(0.0, 1.0))
    plotting._plot_regression(
        explanation,
        instance=np.array([0.3]),
        predict={"predict": 0.5, "low": 0.2, "high": 0.8},
        feature_weights=np.array([0.1]),
        features_to_plot=[0],
        num_to_show=1,
        column_names=["f0"],
        title="reg",
        path=str(tmp_path) + "/",
        show=True,
        interval=False,
        idx=None,
        save_ext=None,
        use_legacy=False,
    )

    # Verify first call has show=True and no save_path (initial render)
    assert render_calls[0] == {"show": True, "save_path": None}

    # Semantic assertions on saved paths (format-independent)
    saved_paths = [call["save_path"] for call in render_calls[1:]]
    assert len(saved_paths) >= 3, "Should save in at least 3 formats"

    # Verify all saved paths exist and contain expected format indicators
    # (NOTE: The code concatenates title + ext where ext is "svg", "pdf", "png"
    # so the resulting filenames are "regsvg", "regpdf", "regpng" without dots)
    format_indicators = {"svg", "pdf", "png"}
    found_indicators = set()
    for path_str in saved_paths:
        if path_str is not None:
            path = Path(path_str)
            # Verify path is under tmp_path (semantic, not string comparison)
            try:
                path.relative_to(tmp_path)
            except ValueError:
                pytest.fail(f"Path {path} is not under tmp_path {tmp_path}")

            # Verify filename contains title
            assert "reg" in path.name, f"Path {path.name} should contain title 'reg'"

            # Check which format indicators appear in the filename
            for indicator in format_indicators:
                if indicator in path.name:
                    found_indicators.add(indicator)

    # Verify at least the expected formats are present in the filenames
    assert (
        len(found_indicators) >= 3
    ), f"Should find indicators for 3 formats, got {found_indicators}"


@pytest.mark.platform_dependent
def test_format_save_path_concatenates_title(tmp_path):
    """Test that _format_save_path combines base directory and filename correctly.

    Refactored to use pathlib.Path and semantic assertions.
     Tests that the function:
     - Concatenates base directory and filename
     - Handles trailing slashes correctly
     - Returns a valid path that can be converted back to Path

     Note: Uses pathlib for cross-platform path handling; does not compare
     exact string representations (platform-dependent).

     Marked as @pytest.mark.platform_dependent because path handling
     behavior may vary across Windows/POSIX systems.
    """
    base = tmp_path / "plots"
    base.mkdir()

    # Test 1: Base path as Path object
    result1 = plotting._format_save_path(base, "figurepng")
    result_path1 = Path(result1)
    assert result_path1.name == "figurepng", "Filename should be in result"
    assert result_path1.parent == base, "Parent directory should match base"

    # Test 2: Base path as string with trailing slash
    result2 = plotting._format_save_path(str(base) + "/", "figurepdf")
    result_path2 = Path(result2)
    assert result_path2.name == "figurepdf", "Filename should be in result"
    # Normalize for comparison (pathlib handles trailing slashes)
    assert (
        result_path2.parent.resolve() == base.resolve()
    ), f"Parent should match base: {result_path2.parent} vs {base}"

    # Test 3: Empty base path should return filename only
    result3 = plotting._format_save_path("", "figurepng")
    assert result3 == "figurepng", "Empty base should return just filename"


def test_plotspec_sorting_abs_desc():
    nfeat = 6
    predict = {"predict": 0.4, "low": 0.1, "high": 0.7}
    vals = np.array([0.2, -0.9, 0.5, -0.1, 0.7, -0.6])
    fw = {"predict": vals, "low": vals - 0.1, "high": vals + 0.1}
    feats = list(range(nfeat))
    cols = [f"f{i}" for i in range(nfeat)]
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by="abs",
        ascending=False,
    )
    bars = spec.body.bars  # type: ignore[union-attr]
    magnitudes = [abs(b.value) for b in bars]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_plotspec_sorting_width_and_interval_equivalence():
    rng = np.random.default_rng(2)
    nfeat = 4
    predict = {"predict": 0.5, "low": 0.3, "high": 0.7}
    vals = rng.normal(0, 0.2, size=nfeat)
    low = vals - rng.uniform(0.05, 0.15, size=nfeat)
    high = vals + rng.uniform(0.05, 0.25, size=nfeat)
    fw = {"predict": vals, "low": low, "high": high}
    feats = list(range(nfeat))
    cols = [f"f{i}" for i in range(nfeat)]

    spec_interval = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by="interval",
        ascending=False,
    )
    spec_width = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by="width",
        ascending=False,
    )
    labels_interval = [b.label for b in spec_interval.body.bars]  # type: ignore[union-attr]
    labels_width = [b.label for b in spec_width.body.bars]  # type: ignore[union-attr]
    assert labels_interval == labels_width


def test_plotspec_sorting_abs_means_distance_from_zero():
    # Ensure 'abs' sorts by |value - 0| (distance from zero)
    predict = {"predict": 0.0}
    vals = np.array([-0.05, 0.2, -0.3, 0.1])
    fw = {"predict": vals, "low": vals - 0.01, "high": vals + 0.01}
    feats = list(range(len(vals)))
    cols = [f"f{i}" for i in feats]
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=None,
        interval=True,
        sort_by="abs",
        ascending=False,
    )
    bars = spec.body.bars  # type: ignore[union-attr]
    distances = [abs(b.value - 0.0) for b in bars]
    assert distances == sorted(distances, reverse=True)


def test_plotspec_sorting_width_descending_changes_order():
    # Construct intervals with distinct widths to verify sorting effect
    predict = {"predict": 0.0}
    vals = np.array([0.1, -0.2, 0.05, 0.3])
    # widths: 0.30, 0.10, 0.20, 0.40
    low = np.array([-0.1, -0.15, 0.0, 0.1])
    high = np.array([0.2, -0.05, 0.2, 0.5])
    fw = {"predict": vals, "low": low, "high": high}
    feats = list(range(len(vals)))
    cols = [f"f{i}" for i in feats]
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=None,
        interval=True,
        sort_by="width",
        ascending=False,
    )
    bars = spec.body.bars  # type: ignore[union-attr]
    widths_sorted = [abs(b.interval_high - b.interval_low) for b in bars]  # type: ignore[operator]
    assert widths_sorted == sorted(widths_sorted, reverse=True)


def test_plotspec_sorting_value_ascending():
    predict = {"predict": 0.0}
    vals = np.array([0.3, -0.1, 0.2, -0.4])
    fw = {"predict": vals, "low": vals - 0.05, "high": vals + 0.05}
    feats = list(range(len(vals)))
    cols = [f"f{i}" for i in feats]
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=None,
        interval=True,
        sort_by="value",
        ascending=True,
    )
    bars = spec.body.bars  # type: ignore[union-attr]
    values_sorted = [b.value for b in bars]
    assert values_sorted == sorted(values_sorted)


def test_plotspec_sorting_label_ascending():
    predict = {"predict": 0.0}
    vals = np.array([0.1, 0.1, 0.1])
    fw = {"predict": vals, "low": vals - 0.01, "high": vals + 0.01}
    feats = [2, 0, 1]
    cols = ["b_label", "a_label", "c_label"]
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=None,
        interval=True,
        sort_by="label",
        ascending=True,
    )
    bars = spec.body.bars  # type: ignore[union-attr]
    labels_sorted = [b.label for b in bars]
    assert labels_sorted == sorted(labels_sorted)


def test_plotspec_render_scales_height_with_bar_count(monkeypatch):
    """Adapter-rendered figures should honour the num_to_show-derived height heuristic."""

    nfeat = 8
    predict = {"predict": 0.6, "low": 0.3, "high": 0.9}
    vals = np.linspace(-0.4, 0.4, nfeat)
    fw = {"predict": vals, "low": vals - 0.1, "high": vals + 0.1}
    spec = build_regression_bars_spec(
        title="height",
        predict=predict,
        feature_weights=fw,
        features_to_plot=list(range(nfeat)),
        column_names=[f"f{i}" for i in range(nfeat)],
        instance=np.linspace(0.0, 1.0, nfeat),
        y_minmax=(0.0, 1.0),
        interval=True,
    )

    seen: list[tuple[float, float] | None] = []

    import matplotlib.pyplot as plt

    real_figure = plt.figure

    def spy_figure(*args, **kwargs):
        seen.append(kwargs.get("figsize"))
        return real_figure(*args, **kwargs)

    monkeypatch.setattr("matplotlib.pyplot.figure", spy_figure)

    fig = matplotlib_adapter.render(spec, show=False, save_path=None, return_fig=True)

    assert seen, "render should have created a matplotlib figure"
    width, height = seen[0]
    assert width == pytest.approx(10.0)
    assert height == pytest.approx(0.5 * max(1, nfeat) + 2.0)

    plt.close(fig)


def test_plotspec_probabilistic_interval_requires_idx(monkeypatch):
    explanation = types.SimpleNamespace(
        y_minmax=(0.0, 1.0),
        prediction={"classes": 1},
        get_class_labels=lambda: ["neg", "pos"],
        is_thresholded=lambda: False,
        get_mode=lambda: "classification",
        is_one_sided=lambda: False,
    )
    setattr(explanation, "_get_explainer", lambda: None)

    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("render should not be reached")),
    )

    with pytest.raises(AssertionError):
        plotting._plot_probabilistic(
            explanation,
            instance=np.array([0.1, 0.2]),
            predict={"predict": 0.5, "low": 0.2, "high": 0.8},
            feature_weights={
                "predict": np.array([0.1, -0.1]),
                "low": np.array([0.0, 0.0]),
                "high": np.array([0.2, 0.2]),
            },
            features_to_plot=[0, 1],
            num_to_show=2,
            column_names=["f0", "f1"],
            title="interval",
            path="/tmp/",
            show=True,
            interval=True,
            idx=None,
            save_ext=[],
            use_legacy=False,
        )


def test_plotspec_probabilistic_default_save_ext(monkeypatch, tmp_path):
    sentinel_spec = object()
    builder_args: list[dict] = []

    def _fake_builder(**kwargs):
        builder_args.append(kwargs)
        return sentinel_spec

    render_calls: list[dict] = []

    def _fake_render(spec, **kwargs):
        render_calls.append({"spec": spec, **kwargs})

    formatted: list[tuple[str, str]] = []

    def _format(base, filename):
        formatted.append((base, filename))
        return f"{base}{filename}"

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_probabilistic_bars_spec",
        _fake_builder,
    )
    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        _fake_render,
    )
    monkeypatch.setattr(plotting, "_format_save_path", _format)

    explanation = types.SimpleNamespace(
        y_minmax=(0.0, 1.0),
        prediction={"classes": 1},
        get_class_labels=lambda: ["neg", "pos"],
        is_thresholded=lambda: False,
        get_mode=lambda: "classification",
        is_one_sided=lambda: False,
    )
    setattr(explanation, "_get_explainer", lambda: None)

    base_path = str(tmp_path) + "/"
    plotting._plot_probabilistic(
        explanation,
        instance=np.array([0.1, 0.2]),
        predict={"predict": 0.6, "low": 0.2, "high": 0.8},
        feature_weights=np.array([0.2, -0.1]),
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["f0", "f1"],
        title="prob",
        path=base_path,
        show=False,
        interval=False,
        idx=None,
        save_ext=None,
        use_legacy=False,
    )

    assert builder_args, "Expected builder to be invoked"
    assert [call["save_path"] for call in render_calls] == [
        None,
        f"{base_path}probsvg",
        f"{base_path}probpdf",
        f"{base_path}probpng",
    ]
    assert formatted == [
        (base_path, "probsvg"),
        (base_path, "probpdf"),
        (base_path, "probpng"),
    ]


def test_regression_builder_includes_instance_values():
    predict = {"predict": 0.5, "low": 0.2, "high": 0.8}
    fw = {"predict": [0.2, -0.3], "low": [0.1, -0.4], "high": [0.3, -0.2]}
    feats = [0, 1]
    columns = ["f0", "f1"]
    instance = [1.5, -2.0]
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=columns,
        instance=instance,
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by=None,
        ascending=False,
    )

    assert [bar.instance_value for bar in spec.body.bars] == instance  # type: ignore[union-attr]


def test_regression_builder_rejects_short_instance_vector():
    predict = {"predict": 0.5}
    fw = [0.2, -0.1, 0.05]
    feats = [0, 2]
    with pytest.raises(ValueError):
        build_regression_bars_spec(
            title=None,
            predict=predict,
            feature_weights=fw,
            features_to_plot=feats,
            column_names=["a", "b", "c"],
            instance=[1.0],
            y_minmax=None,
            interval=False,
            sort_by=None,
            ascending=False,
        )


def test_probabilistic_builder_clamps_infinite_bounds():
    from calibrated_explanations.viz import build_probabilistic_bars_spec

    predict = {"predict": 0.6, "low": -np.inf, "high": np.inf}
    fw = {
        "predict": np.array([0.1, -0.2]),
        "low": np.array([0.0, -0.3]),
        "high": np.array([0.2, -0.1]),
    }
    instance = np.array([2.5, -1.2])
    spec = build_probabilistic_bars_spec(
        title="prob",
        predict=predict,
        feature_weights=fw,
        features_to_plot=[0, 1],
        column_names=["f0", "f1"],
        instance=instance,
        y_minmax=(0.0, 1.0),
        interval=True,
    )

    assert spec.header.low == pytest.approx(0.0)  # type: ignore[union-attr]
    assert spec.header.high == pytest.approx(1.0)  # type: ignore[union-attr]
    assert spec.header.xlim == (0.0, 1.0)  # type: ignore[union-attr]


def test_probabilistic_builder_rejects_truncated_labels():
    from calibrated_explanations.viz import build_probabilistic_bars_spec

    predict = {"predict": 0.4}
    with pytest.raises(ValueError):
        build_probabilistic_bars_spec(
            title=None,
            predict=predict,
            feature_weights=[0.1, -0.2],
            features_to_plot=[0, 1],
            column_names=["a"],
            instance=[1.0, 2.0],
            y_minmax=None,
            interval=False,
        )


def test_build_regression_requires_instance_alignment():
    predict = {"predict": 0.5, "low": 0.2, "high": 0.8}
    feature_weights = [0.3, -0.1]
    feats = [0, 1]
    cols = ["a", "b"]

    with pytest.raises(ValueError):
        build_regression_bars_spec(
            title=None,
            predict=predict,
            feature_weights=feature_weights,
            features_to_plot=feats,
            column_names=cols,
            instance=[0.5],
            y_minmax=None,
            interval=False,
        )


def test_matplotlib_adapter_auto_height_tracks_bars():
    nfeat = 6
    predict = {"predict": 0.4, "low": 0.1, "high": 0.7}
    vals = np.linspace(-0.3, 0.4, nfeat)
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights={"predict": vals, "low": vals - 0.1, "high": vals + 0.1},
        features_to_plot=list(range(nfeat)),
        column_names=[f"f{i}" for i in range(nfeat)],
        instance=list(np.linspace(0.0, 1.0, nfeat)),
        y_minmax=(0.0, 1.0),
        interval=True,
    )
    fig = matplotlib_adapter.render(spec, show=False, save_path=None, return_fig=True)
    try:
        height = fig.get_size_inches()[1]
        expected = 0.5 * max(1, nfeat) + 2.0
        assert height == pytest.approx(expected, rel=0.05)
    finally:
        from matplotlib import pyplot as plt

        plt.close(fig)


def test_plot_probabilistic_clamps_infinite_bounds(monkeypatch, tmp_path):
    sentinel_spec = object()
    captured: dict[str, dict] = {}

    def fake_builder(**kwargs):
        captured["predict"] = dict(kwargs["predict"])
        return sentinel_spec

    def fake_render(spec, **kwargs):
        return None

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_probabilistic_bars_spec", fake_builder
    )
    monkeypatch.setattr("calibrated_explanations.viz.matplotlib_adapter.render", fake_render)
    monkeypatch.setattr(plotting, "__require_matplotlib", lambda: None)

    class _Explanation:
        y_minmax = (0.0, 1.0)
        prediction = {"classes": 1}

        def get_class_labels(self):
            return ["neg", "pos"]

    plotting._plot_probabilistic(
        _Explanation(),
        instance=[0.2, 0.3],
        predict={"predict": 0.6, "low": -np.inf, "high": np.inf},
        feature_weights=[0.1, -0.2],
        features_to_plot=[0, 1],
        num_to_show=2,
        column_names=["f0", "f1"],
        title="clamp",
        path=str(tmp_path) + "/",
        show=False,
        interval=False,
        idx=None,
        save_ext=[".png"],
        use_legacy=False,
    )

    assert captured["predict"]["low"] == pytest.approx(0.0)
    assert captured["predict"]["high"] == pytest.approx(1.0)


def test_build_regression_spec_requires_instance_alignment():
    predict = {"predict": 0.5, "low": 0.2, "high": 0.8}
    fw = {
        "predict": np.array([0.1, -0.2]),
        "low": np.array([0.0, -0.1]),
        "high": np.array([0.2, 0.1]),
    }

    with pytest.raises(ValueError):
        build_regression_bars_spec(
            title=None,
            predict=predict,
            feature_weights=fw,
            features_to_plot=[0, 1],
            column_names=["f0", "f1"],
            instance=[0.1],
            y_minmax=(0.0, 1.0),
            interval=True,
        )


def test_plot_alternative_sanitises_non_finite_payloads(monkeypatch):
    from calibrated_explanations import plotting

    recorded: dict[str, dict] = {}
    sentinel = object()

    def fake_builder(**kwargs):
        recorded.update(kwargs)
        return sentinel

    render_calls: list[dict] = []

    def fake_render(spec, **kwargs):
        render_calls.append({"spec": spec, **kwargs})

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_regression_spec", fake_builder
    )
    monkeypatch.setattr("calibrated_explanations.viz.matplotlib_adapter.render", fake_render)

    explanation = types.SimpleNamespace(
        y_minmax=(0.0, 1.0),
        prediction={"classes": 1},
        is_thresholded=lambda: False,
        get_mode=lambda: "regression",
        get_class_labels=lambda: ["neg", "pos"],
    )
    setattr(explanation, "_get_explainer", lambda: None)

    plotting._plot_alternative(
        explanation=explanation,
        instance=np.array([0.1, 0.2, 0.3]),
        predict={"predict": 0.5, "low": -np.inf, "high": np.inf},
        feature_predict={
            "predict": np.array([0.2, -0.1, 0.05]),
            "low": np.array([-np.inf, -0.2, 0.0]),
            "high": np.array([np.inf, 0.1, 0.3]),
        },
        features_to_plot=[0, "1", -5],
        num_to_show=3,
        column_names=["f0", "f1", "f2"],
        title="alt",
        path=None,
        show=True,
        save_ext=None,
        use_legacy=False,
    )

    assert recorded["predict"]["low"] == 0.0
    assert recorded["predict"]["high"] == 1.0
    assert recorded["feature_weights"]["low"][0] == 0.0
    assert recorded["feature_weights"]["high"][0] == 1.0
    assert recorded["features_to_plot"] == [0, 1]
    assert render_calls[0]["spec"] is sentinel
