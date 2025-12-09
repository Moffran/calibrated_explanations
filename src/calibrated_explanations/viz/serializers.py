"""PlotSpec serialization and validation helpers.

Provides a small stable envelope for PlotSpec -> dict and back, and a
lightweight validator for the MVP spec. The serialized envelope contains
`plotspec_version` to allow future evolution.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .plotspec import BarHPanelSpec, BarItem, IntervalHeaderSpec, PlotSpec

PLOTSPEC_VERSION = "1.0.0"


def plotspec_to_dict(spec: PlotSpec) -> Dict[str, Any]:
    """Serialize a PlotSpec to a plain JSON-serializable dict envelope.

    Note: instance values are passed through as-is.
    """
    payload: Dict[str, Any] = {"plotspec_version": PLOTSPEC_VERSION}
    if spec.title is not None:
        payload["title"] = spec.title
    if spec.figure_size is not None:
        payload["figure_size"] = tuple(spec.figure_size)

    if spec.header is not None:
        h = spec.header
        payload["header"] = {
            "pred": float(h.pred),
            "low": float(h.low),
            "high": float(h.high),
            "xlim": tuple(h.xlim) if h.xlim is not None else None,
            "xlabel": h.xlabel,
            "ylabel": h.ylabel,
        }

    if spec.body is not None:
        b = spec.body
        bars: List[Dict[str, Any]] = []
        for it in b.bars:
            bars.append(
                {
                    "label": it.label,
                    "value": float(it.value),
                    "interval_low": None if it.interval_low is None else float(it.interval_low),
                    "interval_high": None if it.interval_high is None else float(it.interval_high),
                    "color_role": it.color_role,
                    "instance_value": it.instance_value,
                }
            )
        payload["body"] = {"bars": bars, "xlabel": b.xlabel, "ylabel": b.ylabel}

    return payload


def plotspec_from_dict(obj: Dict[str, Any]) -> PlotSpec:
    """Deserialize a dict envelope to a PlotSpec.

    Will raise ValueError for obviously invalid payloads.
    """
    validate_plotspec(obj)

    title = obj.get("title")
    figure_size = tuple(obj["figure_size"]) if obj.get("figure_size") is not None else None

    header = None
    if obj.get("header") is not None:
        h = obj["header"]
        header = IntervalHeaderSpec(
            pred=float(h.get("pred", 0.0)),
            low=float(h.get("low", 0.0)),
            high=float(h.get("high", 0.0)),
            xlim=tuple(h.get("xlim")) if h.get("xlim") is not None else None,
            xlabel=h.get("xlabel"),
            ylabel=h.get("ylabel"),
        )

    body = None
    if obj.get("body") is not None:
        b = obj["body"]
        bars_list = []
        for r in b.get("bars", []):
            bars_list.append(
                BarItem(
                    label=str(r.get("label", "")),
                    value=float(r.get("value", 0.0)),
                    interval_low=None
                    if r.get("interval_low") is None
                    else float(r.get("interval_low")),
                    interval_high=None
                    if r.get("interval_high") is None
                    else float(r.get("interval_high")),
                    color_role=r.get("color_role"),
                    instance_value=r.get("instance_value"),
                )
            )
        body = BarHPanelSpec(bars=bars_list, xlabel=b.get("xlabel"), ylabel=b.get("ylabel"))

    return PlotSpec(title=title, figure_size=figure_size, header=header, body=body)


def validate_plotspec(obj: Dict[str, Any]) -> None:
    """Lightweight validation for a PlotSpec envelope.

    Raises ValueError when required fields or shapes are missing for the MVP.
    This is intentionally conservative: it checks structural assumptions used by
    the matplotlib adapter and tests.
    """
    if not isinstance(obj, dict):
        from ..core.exceptions import ValidationError

        raise ValidationError(
            "PlotSpec payload must be a dict",
            details={"expected_type": "dict", "actual_type": type(obj).__name__},
        )

    version = obj.get("plotspec_version")
    if version != PLOTSPEC_VERSION:
        from ..core.exceptions import ValidationError

        raise ValidationError(
            f"unsupported or missing plotspec_version: {version}",
            details={"expected_version": PLOTSPEC_VERSION, "actual_version": version},
        )

    # Basic body validation for bar-panel specs
    body = obj.get("body")
    if body is None:
        from ..core.exceptions import ValidationError

        raise ValidationError(
            "PlotSpec body is required for bar plots",
            details={"section": "body", "requirement": "required for bar plots"},
        )
    bars = body.get("bars")
    if not isinstance(bars, list):
        from ..core.exceptions import ValidationError

        raise ValidationError(
            "PlotSpec body.bars must be a list",
            details={
                "field": "body.bars",
                "expected_type": "list",
                "actual_type": type(bars).__name__,
            },
        )
    for i, b in enumerate(bars):
        if "label" not in b or "value" not in b:
            from ..core.exceptions import ValidationError

            raise ValidationError(
                f"bar at index {i} missing required fields 'label' or 'value'",
                details={
                    "bar_index": i,
                    "missing_fields": [f for f in ["label", "value"] if f not in b],
                },
            )


__all__ = ["plotspec_to_dict", "plotspec_from_dict", "validate_plotspec", "PLOTSPEC_VERSION"]
