from calibrated_explanations.viz import (
    build_probabilistic_bars_spec,
    matplotlib_adapter as mpl_adapter,
)
from calibrated_explanations.viz.plotspec import SaveBehavior


def test_headless_export_returns_bytes():
    spec = build_probabilistic_bars_spec(
        title="headless",
        predict={"predict": 0.6, "low": 0.5, "high": 0.7},
        feature_weights={"predict": [0.2], "low": [0.1], "high": [0.3]},
        features_to_plot=[0],
        column_names=["f0"],
        rule_labels=None,
        instance=[0],
        y_minmax=(0.0, 1.0),
        interval=True,
    )
    # request headless export (no filesystem path)
    spec.save_behavior = SaveBehavior(path=None, title="headless", default_exts=("png", "svg"))

    result = mpl_adapter.render(spec, show=False)
    assert isinstance(result, dict)
    assert "bytes" in result
    bytes_map = result["bytes"]
    assert (
        "png" in bytes_map
        and isinstance(bytes_map["png"], (bytes, bytearray))
        and len(bytes_map["png"]) > 0
    )
    assert (
        "svg" in bytes_map
        and isinstance(bytes_map["svg"], (bytes, bytearray))
        and len(bytes_map["svg"]) > 0
    )
