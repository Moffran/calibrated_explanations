"""Parity checks between legacy and PlotSpec alternative regression plots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping

import numpy as np
import pytest
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pytest import MonkeyPatch

from calibrated_explanations import plotting as plotspec_plotting
from calibrated_explanations.legacy import plotting as legacy_plotting
from calibrated_explanations.viz.builders import (
    REGRESSION_BASE_COLOR,
    REGRESSION_BAR_COLOR,
)

pytest.importorskip("matplotlib")
pytestmark = pytest.mark.viz


def _confidence_from_percentiles(percentiles: tuple[float, float]) -> float:
    low, high = percentiles
    if math.isinf(high):
        return 100.0 - float(low)
    if math.isinf(low):
        return float(high)
    return float(high) - float(low)


class _StubExplainer:
    """Minimal explainer stub for style resolution and metadata access."""

    _plot_plugin_fallbacks: Mapping[str, tuple[str, ...]] = {"alternative": ()}
    _last_explanation_mode: str = "alternative"

    def is_multiclass(self) -> bool:  # pragma: no cover - constant behaviour
        return False


class _StubCalibrated:
    """Provide confidence metadata expected by both plotting paths."""

    def __init__(self, percentiles: tuple[float, float]) -> None:
        self._percentiles = percentiles
        self._plot_plugin_fallbacks: Mapping[str, tuple[str, ...]] = {"alternative": ()}
        self.calibrated_explainer = _StubExplainer()
        self.low_high_percentiles = percentiles

    def get_confidence(self) -> float:
        return _confidence_from_percentiles(self._percentiles)


class _StubAlternativeExplanation:
    """Simple container emulating the attributes used by the plot helpers."""

    def __init__(
        self,
        *,
        y_minmax: tuple[float, float],
        percentiles: tuple[float, float],
    ) -> None:
        self.y_minmax = y_minmax
        self.low_high_percentiles = percentiles
        self.calibrated_explanations = _StubCalibrated(percentiles)
        self.prediction = {"predict": 0.0, "classes": 1}
        self.y_threshold = None

    def _get_explainer(self) -> _StubExplainer:
        return self.calibrated_explanations.calibrated_explainer

    def get_mode(self) -> str:  # pragma: no cover - constant behaviour
        return "regression"

    def get_class_labels(self):  # pragma: no cover - regression path doesn't require labels
        return None

    def is_thresholded(self) -> bool:  # pragma: no cover - explicit regression branch
        return False


@dataclass(frozen=True)
class RegressionParityCase:
    """Input payload describing a legacy vs PlotSpec comparison scenario."""

    name: str
    predict: Mapping[str, float]
    feature_predict: Mapping[str, Iterable[float]]
    features_to_plot: Iterable[int]
    column_names: Iterable[str]
    instance: Iterable[float]
    y_minmax: tuple[float, float]
    percentiles: tuple[float, float]


CASES: tuple[RegressionParityCase, ...] = (
    RegressionParityCase(
        name="two_sided",
        predict={"predict": 6.0, "low": 4.0, "high": 8.0},
        feature_predict={
            "predict": (7.0, 5.0, 3.0),
            "low": (6.5, 4.5, 2.0),
            "high": (7.5, 5.5, 4.0),
        },
        features_to_plot=(0, 1, 2),
        column_names=("rule0", "rule1", "rule2"),
        instance=(1.0, 2.0, 3.0),
        y_minmax=(0.0, 10.0),
        percentiles=(5.0, 95.0),
    ),
    RegressionParityCase(
        name="lower_unbounded",
        predict={"predict": 6.0, "low": -math.inf, "high": 8.0},
        feature_predict={
            "predict": (7.0, 5.0, 3.0),
            "low": (-math.inf, 4.5, 2.0),
            "high": (7.5, 5.5, 4.0),
        },
        features_to_plot=(0, 1, 2),
        column_names=("rule0", "rule1", "rule2"),
        instance=(1.0, 2.0, 3.0),
        y_minmax=(0.0, 10.0),
        percentiles=(-math.inf, 95.0),
    ),
    RegressionParityCase(
        name="upper_unbounded",
        predict={"predict": 6.0, "low": 4.0, "high": math.inf},
        feature_predict={
            "predict": (7.0, 5.0, 3.0),
            "low": (6.5, 4.5, 2.0),
            "high": (math.inf, math.inf, 4.0),
        },
        features_to_plot=(0, 1, 2),
        column_names=("rule0", "rule1", "rule2"),
        instance=(1.0, 2.0, 3.0),
        y_minmax=(0.0, 10.0),
        percentiles=(5.0, math.inf),
    ),
)


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def _coerce_array(payload: Mapping[str, Iterable[float]]) -> Dict[str, np.ndarray]:
    return {key: np.asarray(list(values), dtype=float) for key, values in payload.items()}


def _collect_legacy_summary(
    *,
    explanation: _StubAlternativeExplanation,
    case: RegressionParityCase,
) -> Dict[str, Dict[str, Dict[str, float | str]]]:
    mp = MonkeyPatch()
    try:
        plt.switch_backend("Agg")
        records: list[tuple[str, tuple, dict]] = []

        def _record(method: str) -> Callable:
            orig = getattr(Axes, method)

            def wrapper(self, *args, **kwargs):
                records.append((method, args, kwargs))
                return orig(self, *args, **kwargs)

            mp.setattr(Axes, method, wrapper)
            return wrapper

        _record("fill_betweenx")
        mp.setattr(Figure, "show", lambda self, *args, **kwargs: None)

        legacy_plotting._plot_alternative(
            explanation=explanation,
            instance=_as_float_list(case.instance),
            predict=dict(case.predict),
            feature_predict=_coerce_array(case.feature_predict),
            features_to_plot=list(case.features_to_plot),
            num_to_show=len(tuple(case.features_to_plot)),
            column_names=list(case.column_names),
            title="legacy",
            path="",
            show=True,
            save_ext=[],
        )

        return _summarise_legacy_records(records, case)
    finally:
        mp.undo()
        plt.close("all")


def _summarise_legacy_records(
    records: list[tuple[str, tuple, dict]],
    case: RegressionParityCase,
) -> Dict[str, Dict[str, Dict[str, float | str]]]:
    base_entries: list[dict] = []
    feature_intervals: Dict[str, dict] = {}
    feature_lines: Dict[str, dict] = {}
    base_line: dict | None = None

    base_hex = mcolors.to_hex(REGRESSION_BASE_COLOR)
    bar_hex = mcolors.to_hex("r")

    for method, args, kwargs in records:
        if method != "fill_betweenx":
            continue
        color_hex = mcolors.to_hex(kwargs.get("color", "k"))
        alpha = float(kwargs.get("alpha", 1.0))
        y_arr = np.asarray(args[0], dtype=float)
        if y_arr.size == 0:
            continue
        y_low = float(y_arr.flat[0])
        y_high = float(y_arr.flat[-1])
        center = 0.5 * (y_low + y_high)
        x1_arr = np.asarray(args[1], dtype=float)
        x2_arr = np.asarray(args[2], dtype=float)
        x_min = float(min(np.min(x1_arr), np.min(x2_arr)))
        x_max = float(max(np.max(x1_arr), np.max(x2_arr)))

        if color_hex == base_hex:
            base_entries.append({"low": x_min, "high": x_max, "alpha": alpha, "color": color_hex})
            continue
        if color_hex != bar_hex:
            continue
        if math.isclose(alpha, 0.3, rel_tol=1e-6) and math.isclose(
            x_min, x_max, rel_tol=1e-9, abs_tol=1e-9
        ):
            if base_line is None:
                base_line = {"value": x_min, "alpha": alpha, "color": color_hex}
            continue
        idx = int(round(center))
        label = list(case.column_names)[idx]
        if math.isclose(alpha, 0.4, rel_tol=1e-6):
            feature_intervals[label] = {
                "low": x_min,
                "high": x_max,
                "alpha": alpha,
                "color": color_hex,
            }
        elif math.isclose(alpha, 1.0, rel_tol=1e-6) and math.isclose(
            x_min, x_max, rel_tol=1e-9, abs_tol=1e-9
        ):
            feature_lines[label] = {"value": x_min, "alpha": alpha, "color": color_hex}

    base_low = min(entry["low"] for entry in base_entries)
    base_high = max(entry["high"] for entry in base_entries)
    base_alpha = base_entries[0]["alpha"] if base_entries else 1.0
    base_color = base_entries[0]["color"] if base_entries else base_hex

    assert base_line is not None, "legacy plot did not emit a baseline marker"

    feature_summary = {}
    for label in case.column_names:
        interval = feature_intervals[label]
        line = feature_lines[label]
        feature_summary[label] = {"interval": interval, "line": line}

    return {
        "base_interval": {
            "low": base_low,
            "high": base_high,
            "alpha": base_alpha,
            "color": base_color,
        },
        "base_line": base_line,
        "features": feature_summary,
    }


def _collect_plotspec_summary(
    *,
    explanation: _StubAlternativeExplanation,
    case: RegressionParityCase,
) -> Dict[str, Dict[str, Dict[str, float | str]]]:
    mp = MonkeyPatch()
    captured: dict = {}
    try:
        mp.setattr(plotspec_plotting, "__require_matplotlib", lambda: None)

        def _capture(spec, **_kwargs):
            captured["spec"] = spec
            return {}

        mp.setattr("calibrated_explanations.viz.matplotlib_adapter.render", _capture)

        plotspec_plotting._plot_alternative(
            explanation=explanation,
            instance=_as_float_list(case.instance),
            predict=dict(case.predict),
            feature_predict=_coerce_array(case.feature_predict),
            features_to_plot=list(case.features_to_plot),
            num_to_show=len(tuple(case.features_to_plot)),
            column_names=list(case.column_names),
            title="plotspec",
            path="",
            show=True,
            save_ext=[],
            use_legacy=False,
        )

        spec = captured["spec"]
        return _summarise_plotspec(spec)
    finally:
        mp.undo()


def _summarise_plotspec(spec) -> Dict[str, Dict[str, Dict[str, float | str]]]:
    body = spec.body
    assert body is not None
    assert body.base_segments is not None and len(body.base_segments) == 1
    segment = body.base_segments[0]
    base_interval = {
        "low": float(segment.low),
        "high": float(segment.high),
        "alpha": float(segment.alpha) if segment.alpha is not None else 1.0,
        "color": mcolors.to_hex(segment.color),
    }

    assert body.base_lines is not None and len(body.base_lines) == 1
    line_value, line_color, line_alpha = body.base_lines[0]
    base_line = {
        "value": float(line_value),
        "alpha": float(line_alpha) if line_alpha is not None else 1.0,
        "color": mcolors.to_hex(line_color),
    }

    features = {}
    for bar in body.bars:
        label = bar.label
        if getattr(bar, "segments", None):
            seg = bar.segments[0]
            interval_summary = {
                "low": float(seg.low),
                "high": float(seg.high),
                "alpha": float(seg.alpha) if seg.alpha is not None else 1.0,
                "color": mcolors.to_hex(seg.color),
            }
        else:
            interval_summary = {
                "low": float(bar.interval_low),
                "high": float(bar.interval_high),
                "alpha": 1.0,
                "color": mcolors.to_hex(bar.color_role or REGRESSION_BAR_COLOR),
            }
        line_summary = {
            "value": float(bar.line if bar.line is not None else bar.value),
            "alpha": float(bar.line_alpha) if bar.line_alpha is not None else 1.0,
            "color": mcolors.to_hex(
                bar.line_color if bar.line_color is not None else REGRESSION_BAR_COLOR
            ),
        }
        features[label] = {"interval": interval_summary, "line": line_summary}

    return {"base_interval": base_interval, "base_line": base_line, "features": features}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_alternative_regression_plotspec_matches_legacy(case: RegressionParityCase) -> None:
    """Ensure PlotSpec rendering matches the legacy alternative regression visuals."""

    explanation = _StubAlternativeExplanation(
        y_minmax=case.y_minmax,
        percentiles=case.percentiles,
    )

    legacy_summary = _collect_legacy_summary(explanation=explanation, case=case)
    plotspec_summary = _collect_plotspec_summary(explanation=explanation, case=case)

    assert plotspec_summary["base_interval"]["color"] == legacy_summary["base_interval"]["color"]
    assert plotspec_summary["base_interval"]["alpha"] == pytest.approx(
        legacy_summary["base_interval"]["alpha"]
    )
    assert plotspec_summary["base_interval"]["low"] == pytest.approx(
        legacy_summary["base_interval"]["low"]
    )
    assert plotspec_summary["base_interval"]["high"] == pytest.approx(
        legacy_summary["base_interval"]["high"]
    )

    assert plotspec_summary["base_line"]["color"] == legacy_summary["base_line"]["color"]
    assert plotspec_summary["base_line"]["alpha"] == pytest.approx(
        legacy_summary["base_line"]["alpha"]
    )
    assert plotspec_summary["base_line"]["value"] == pytest.approx(
        legacy_summary["base_line"]["value"]
    )

    for label in case.column_names:
        legacy_feature = legacy_summary["features"][label]
        spec_feature = plotspec_summary["features"][label]

        assert spec_feature["interval"]["color"] == legacy_feature["interval"]["color"]
        assert spec_feature["interval"]["alpha"] == pytest.approx(
            legacy_feature["interval"]["alpha"]
        )
        assert spec_feature["interval"]["low"] == pytest.approx(legacy_feature["interval"]["low"])
        assert spec_feature["interval"]["high"] == pytest.approx(legacy_feature["interval"]["high"])

        assert spec_feature["line"]["color"] == legacy_feature["line"]["color"]
        assert spec_feature["line"]["alpha"] == pytest.approx(legacy_feature["line"]["alpha"])
        assert spec_feature["line"]["value"] == pytest.approx(legacy_feature["line"]["value"])
