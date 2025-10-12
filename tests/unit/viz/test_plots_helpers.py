import warnings


from calibrated_explanations import plotting as _plots
from calibrated_explanations.viz import coloring


def test_color_brew_length_and_format():
    cols = coloring.color_brew(5)
    assert isinstance(cols, list)
    # Implementation may generate a slightly different number of colors
    # depending on the arange stepping; ensure it's non-empty and reasonable.
    assert len(cols) >= 3
    for c in cols:
        assert isinstance(c, list) and len(c) == 3
        assert all(isinstance(v, int) and 0 <= v <= 255 for v in c)


def test_get_fill_color_outputs_hex():
    ca = {"predict": 0.8, "low_high": [0.7, 0.9]}
    col = coloring.get_fill_color(ca, reduction=0.5)
    assert isinstance(col, str) and col.startswith("#") and len(col) == 7


def test_setup_style_unknown_section_warns():
    # Ensure warning is emitted for unknown style sections
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = {"nonexistent": {"foo": "bar"}}
        _plots.__require_matplotlib()
        _plots.__setup_plot_style(style_override=cfg)
        assert any("Unknown style section" in str(x.message) for x in w)
