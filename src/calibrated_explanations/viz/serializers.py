"""PlotSpec serialization and validation helpers.

Provides a small stable envelope for PlotSpec -> dict and back, and a
lightweight validator for the MVP spec. The serialized envelope contains
`plotspec_version` to allow future evolution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from ..utils.exceptions import ValidationError
from .plotspec import (
    BarHPanelSpec,
    BarItem,
    GlobalPlotSpec,
    IntervalHeaderSpec,
    PlotSpec,
    TriangularPlotSpec,
)

PLOTSPEC_VERSION = "1.0.0"


class PlotKindRegistry:
    """Registry of supported PlotSpec kinds and their validation requirements.

    Centralizes kind-aware validation logic for PlotSpec payloads.
    """

    # Supported kinds and their required fields/modes
    _SUPPORTED_KINDS: Dict[str, Dict[str, Any]] = {
        "factual_probabilistic": {
            "modes": {"classification"},
            "requires_body": True,
            "requires_header": True,
            "requires_triangular": False,
            "requires_global_entries": False,
        },
        "factual_regression": {
            "modes": {"regression"},
            "requires_body": True,
            "requires_header": True,
            "requires_triangular": False,
            "requires_global_entries": False,
        },
        "alternative_probabilistic": {
            "modes": {"classification"},
            "requires_body": True,
            "requires_header": False,
            "requires_triangular": False,
            "requires_global_entries": False,
        },
        "alternative_regression": {
            "modes": {"regression"},
            "requires_body": True,
            "requires_header": False,
            "requires_triangular": False,
            "requires_global_entries": False,
        },
        "triangular": {
            "modes": {"classification", "regression"},
            "requires_body": False,
            "requires_header": False,
            "requires_triangular": True,
            "requires_global_entries": False,
        },
        "global_probabilistic": {
            "modes": {"classification"},
            "requires_body": False,
            "requires_header": False,
            "requires_triangular": False,
            "requires_global_entries": True,
        },
        "global_regression": {
            "modes": {"regression"},
            "requires_body": False,
            "requires_header": False,
            "requires_triangular": False,
            "requires_global_entries": True,
        },
    }

    @classmethod
    def supported_kinds(cls) -> Set[str]:
        """Return the set of supported PlotSpec kinds."""
        return set(cls._SUPPORTED_KINDS.keys())

    @classmethod
    def is_supported_kind(cls, kind: str) -> bool:
        """Check if a kind is supported."""
        return kind in cls._SUPPORTED_KINDS

    @classmethod
    def validate_kind_and_mode(cls, kind: str, mode: str) -> None:
        """Validate that kind and mode are compatible."""
        if not cls.is_supported_kind(kind):
            from ..utils.exceptions import ValidationError

            raise ValidationError(
                f"Unsupported PlotSpec kind: {kind}",
                details={"supported_kinds": list(cls.supported_kinds())},
            )
        kind_info = cls._SUPPORTED_KINDS[kind]
        if mode not in kind_info["modes"]:
            from ..utils.exceptions import ValidationError

            raise ValidationError(
                f"Mode '{mode}' not supported for kind '{kind}'",
                details={"supported_modes": list(kind_info["modes"])},
            )

    @classmethod
    def get_kind_requirements(cls, kind: str) -> Dict[str, Any]:
        """Get the validation requirements for a kind."""
        if not cls.is_supported_kind(kind):
            raise ValidationError(f"Unsupported kind: {kind}")
        return cls._SUPPORTED_KINDS[kind].copy()


def plotspec_to_dict(spec: PlotSpec) -> Dict[str, Any]:
    """Serialize a PlotSpec dataclass to a JSON-serializable envelope.

    The envelope has the shape:
    {
        "plotspec_version": <version>,
        "plot_spec": { ... }
    }

    This allows non-panel plot builders to also return the same envelope and
    keeps validation straightforward.
    """
    inner: Dict[str, Any] = {}

    # Basic metadata
    # Only include explicit metadata if provided; otherwise omit to preserve roundtrip equality
    if spec.kind is not None:
        inner["kind"] = spec.kind
    if spec.mode is not None:
        inner["mode"] = spec.mode

    # Header
    if spec.header is not None:
        h = spec.header
        inner["header"] = {
            "pred": float(h.pred),
            "low": float(h.low),
            "high": float(h.high),
            "xlim": list(h.xlim) if h.xlim is not None else None,
            "xlabel": h.xlabel,
            "ylabel": h.ylabel,
            "dual": bool(h.dual),
        }
    else:
        inner["header"] = None

    # Body and feature entries
    inner["body"] = None
    if spec.body is not None:
        entries: List[Dict[str, Any]] = []
        for i, b in enumerate(spec.body.bars):
            e: Dict[str, Any] = {
                "index": i,
                "name": b.label,
                "weight": float(b.value),
                "low": (float(b.interval_low) if b.interval_low is not None else None),
                "high": (float(b.interval_high) if b.interval_high is not None else None),
                "instance_value": b.instance_value,
            }
            entries.append(e)
        inner["body"] = {
            "bars_count": len(entries),
            "xlabel": spec.body.xlabel,
            "ylabel": spec.body.ylabel,
        }
        inner["feature_entries"] = entries
        # prefer the supplied feature_order when present; otherwise omit to
        # preserve roundtrip equality for dataclasses that did not set it
        if spec.feature_order is not None:
            inner["feature_order"] = list(spec.feature_order)
        inner["uncertainty"] = any(
            (b.interval_low is not None and b.interval_high is not None) for b in spec.body.bars
        )
    else:
        inner["feature_order"] = []
        inner["feature_entries"] = None
        inner["uncertainty"] = False

    # Default style metadata required by the plotspec schema; use a
    # conservative default so builders that rely on plotspec dicts pass
    # strict validation when jsonschema is available in tests.
    if inner.get("style") is None:
        inner["style"] = "default"

    # Save behavior and provenance
    if spec.save_behavior is not None:
        sb = spec.save_behavior
        inner["save_behavior"] = {
            "path": sb.path,
            "title": sb.title,
            "default_exts": list(sb.default_exts) if sb.default_exts is not None else None,
        }
    if spec.data_slice_id is not None or spec.rendering_seed is not None:
        inner["provenance"] = {
            "data_slice_id": spec.data_slice_id,
            "rendering_seed": spec.rendering_seed,
        }

    envelope: Dict[str, Any] = {"plotspec_version": PLOTSPEC_VERSION, "plot_spec": inner}
    if spec.title is not None:
        envelope["title"] = spec.title
    if spec.figure_size is not None:
        envelope["figure_size"] = tuple(spec.figure_size)
    # Backwards-compatible top-level fields (legacy callers expect header/body/feature_entries)
    header = inner.get("header")
    if header is not None and header.get("xlim") is not None:
        # Preserve tuple shape when possible for legacy callers
        try:
            header = dict(header)
            header["xlim"] = tuple(header["xlim"])
        except Exception as exc:  # adr002_allow
            import logging

            logging.warning(f"Failed to convert xlim to tuple in header: {exc}")
    envelope["header"] = header

    # Provide a legacy-style body.bars list (if feature_entries exist) for older callers
    feature_entries = inner.get("feature_entries")
    if feature_entries is not None:
        envelope["feature_entries"] = feature_entries
        envelope["body"] = {
            "bars": [
                {
                    "label": e.get("name"),
                    "value": e.get("weight"),
                    "interval_low": e.get("low"),
                    "interval_high": e.get("high"),
                    "instance_value": e.get("instance_value"),
                }
                for e in feature_entries
            ]
        }
    else:
        envelope["feature_entries"] = None
        envelope["body"] = inner.get("body")

    envelope["uncertainty"] = inner.get("uncertainty")
    return envelope


def triangular_plotspec_to_dict(spec: TriangularPlotSpec) -> Dict[str, Any]:
    """Serialize a TriangularPlotSpec to dict."""
    inner: Dict[str, Any] = {
        "kind": spec.kind,
        "mode": spec.mode,
        "triangular": {
            "proba": spec.triangular.proba if spec.triangular else None,
            "uncertainty": spec.triangular.uncertainty if spec.triangular else None,
            "rule_proba": list(spec.triangular.rule_proba)
            if spec.triangular and spec.triangular.rule_proba
            else None,
            "rule_uncertainty": list(spec.triangular.rule_uncertainty)
            if spec.triangular and spec.triangular.rule_uncertainty
            else None,
            "num_to_show": spec.triangular.num_to_show if spec.triangular else 50,
            "is_probabilistic": spec.triangular.is_probabilistic if spec.triangular else True,
        }
        if spec.triangular
        else None,
    }
    if spec.save_behavior is not None:
        sb = spec.save_behavior
        inner["save_behavior"] = {
            "path": sb.path,
            "title": sb.title,
            "default_exts": list(sb.default_exts) if sb.default_exts is not None else None,
        }
    if spec.data_slice_id is not None or spec.rendering_seed is not None:
        inner["provenance"] = {
            "data_slice_id": spec.data_slice_id,
            "rendering_seed": spec.rendering_seed,
        }
    envelope: Dict[str, Any] = {"plotspec_version": PLOTSPEC_VERSION, "plot_spec": inner}
    if spec.title is not None:
        envelope["title"] = spec.title
    if spec.figure_size is not None:
        envelope["figure_size"] = tuple(spec.figure_size)
    return envelope


def global_plotspec_to_dict(spec: GlobalPlotSpec) -> Dict[str, Any]:
    """Serialize a GlobalPlotSpec to dict."""
    inner: Dict[str, Any] = {
        "kind": spec.kind,
        "mode": spec.mode,
        "global_entries": {
            "proba": list(spec.global_entries.proba)
            if spec.global_entries and spec.global_entries.proba is not None
            else None,
            "predict": list(spec.global_entries.predict)
            if spec.global_entries and spec.global_entries.predict is not None
            else None,
            "low": list(spec.global_entries.low)
            if spec.global_entries and spec.global_entries.low is not None
            else None,
            "high": list(spec.global_entries.high)
            if spec.global_entries and spec.global_entries.high is not None
            else None,
            "uncertainty": list(spec.global_entries.uncertainty)
            if spec.global_entries and spec.global_entries.uncertainty is not None
            else None,
            "y_test": list(spec.global_entries.y_test)
            if spec.global_entries and spec.global_entries.y_test is not None
            else None,
        }
        if spec.global_entries
        else None,
    }
    if spec.save_behavior is not None:
        sb = spec.save_behavior
        inner["save_behavior"] = {
            "path": sb.path,
            "title": sb.title,
            "default_exts": list(sb.default_exts) if sb.default_exts is not None else None,
        }
    if spec.data_slice_id is not None or spec.rendering_seed is not None:
        inner["provenance"] = {
            "data_slice_id": spec.data_slice_id,
            "rendering_seed": spec.rendering_seed,
        }
    # Axis hints: provide lightweight suggestions for adapter axes.
    axis_hints: Dict[str, Any] = {}
    ge = spec.global_entries if spec.global_entries is not None else None

    # Helper to safely cast sequence-like values to floats
    def _safe_floats(seq):
        try:
            vals = []
            for v in seq:
                # handle per-entry sequences (multiclass) by picking first element
                if hasattr(v, "__len__") and not isinstance(v, (float, int, str)):
                    vals.append(float(v[0]))
                else:
                    vals.append(float(v))
            return vals
        except Exception:  # adr002_allow
            import logging
            import warnings

            logging.getLogger(__name__).info(
                "Failed to cast sequence to floats for axis hints; skipping hints."
            )
            warnings.warn(
                "Failed to cast sequence to floats for axis hints; skipping hints.",
                UserWarning,
                stacklevel=2,
            )
            return None

    if ge is not None:
        proba_vals = None
        if getattr(ge, "proba", None) is not None:
            proba_vals = _safe_floats(ge.proba)
        if proba_vals:
            axis_hints["xlim"] = [min(proba_vals), max(proba_vals)]
        else:
            # Fall back to predict/low/high when proba not available or invalid
            try_vals = None
            if getattr(ge, "predict", None) is not None:
                try_vals = _safe_floats(ge.predict)
            if try_vals:
                axis_hints["xlim"] = [min(try_vals), max(try_vals)]
        unc_vals = None
        if getattr(ge, "uncertainty", None) is not None:
            unc_vals = _safe_floats(ge.uncertainty)
        if unc_vals:
            axis_hints["ylim"] = [min(unc_vals), max(unc_vals)]

    # Default axis hints for probabilistic global plots
    if not axis_hints.get("xlim") and inner.get("kind", "").startswith("global_probabilistic"):
        axis_hints.setdefault("xlim", [0.0, 1.0])
    if not axis_hints.get("ylim") and inner.get("kind", "").startswith("global_probabilistic"):
        axis_hints.setdefault("ylim", [0.0, 1.0])
    if axis_hints:
        inner["axis_hints"] = axis_hints
    envelope: Dict[str, Any] = {"plotspec_version": PLOTSPEC_VERSION, "plot_spec": inner}
    if spec.title is not None:
        envelope["title"] = spec.title
    if spec.figure_size is not None:
        envelope["figure_size"] = tuple(spec.figure_size)
    return envelope


def triangular_plotspec_from_dict(obj: Dict[str, Any]) -> TriangularPlotSpec:
    """Deserialize a triangular plotspec envelope (or inner dict) to TriangularPlotSpec."""
    if obj.get("plot_spec") is not None:
        version = obj.get("plotspec_version")
        if version is not None and version != PLOTSPEC_VERSION:
            from ..utils.exceptions import ValidationError

            raise ValidationError(
                f"unsupported or missing plotspec_version: {version}",
                details={"expected_version": PLOTSPEC_VERSION, "actual_version": version},
            )
        ps = obj["plot_spec"]
    else:
        ps = obj

    inner = ps.get("triangular") or {}
    tri = None
    if inner:
        tri = {
            "proba": inner.get("proba"),
            "uncertainty": inner.get("uncertainty"),
            "rule_proba": list(inner.get("rule_proba"))
            if inner.get("rule_proba") is not None
            else None,
            "rule_uncertainty": list(inner.get("rule_uncertainty"))
            if inner.get("rule_uncertainty") is not None
            else None,
            "num_to_show": int(inner.get("num_to_show", 50)),
            "is_probabilistic": bool(inner.get("is_probabilistic", True)),
        }
        from .plotspec import TriangularSpec

        tri = TriangularSpec(
            proba=tri["proba"],
            uncertainty=tri["uncertainty"],
            rule_proba=tri["rule_proba"],
            rule_uncertainty=tri["rule_uncertainty"],
            num_to_show=tri["num_to_show"],
            is_probabilistic=tri["is_probabilistic"],
        )

    sb = None
    if ps.get("save_behavior") is not None:
        sb = ps.get("save_behavior")
        from .plotspec import SaveBehavior

        sb = SaveBehavior(
            path=sb.get("path"),
            title=sb.get("title"),
            default_exts=tuple(sb.get("default_exts"))
            if sb.get("default_exts") is not None
            else None,
        )

    prov = ps.get("provenance") or {}

    spec = TriangularPlotSpec(
        title=obj.get("title") if obj.get("title") is not None else ps.get("title"),
        figure_size=tuple(obj.get("figure_size")) if obj.get("figure_size") is not None else None,
        triangular=tri,
        kind=ps.get("kind"),
        mode=ps.get("mode"),
        plotspec_version=obj.get("plotspec_version", PLOTSPEC_VERSION)
        if obj.get("plot_spec") is not None
        else PLOTSPEC_VERSION,
        save_behavior=sb,
        data_slice_id=prov.get("data_slice_id"),
        rendering_seed=prov.get("rendering_seed"),
    )
    return spec


def global_plotspec_from_dict(obj: Dict[str, Any]) -> GlobalPlotSpec:
    """Deserialize a global plotspec envelope (or inner dict) to GlobalPlotSpec."""
    if obj.get("plot_spec") is not None:
        version = obj.get("plotspec_version")
        if version is not None and version != PLOTSPEC_VERSION:
            from ..utils.exceptions import ValidationError

            raise ValidationError(
                f"unsupported or missing plotspec_version: {version}",
                details={"expected_version": PLOTSPEC_VERSION, "actual_version": version},
            )
        ps = obj["plot_spec"]
    else:
        ps = obj

    inner = ps.get("global_entries") or {}
    ge = None
    if inner:
        ge = {
            "proba": list(inner.get("proba")) if inner.get("proba") is not None else None,
            "predict": list(inner.get("predict")) if inner.get("predict") is not None else None,
            "low": list(inner.get("low")) if inner.get("low") is not None else None,
            "high": list(inner.get("high")) if inner.get("high") is not None else None,
            "uncertainty": list(inner.get("uncertainty"))
            if inner.get("uncertainty") is not None
            else None,
            "y_test": list(inner.get("y_test")) if inner.get("y_test") is not None else None,
        }
        from .plotspec import GlobalSpec

        ge = GlobalSpec(
            proba=ge["proba"],
            predict=ge["predict"],
            low=ge["low"],
            high=ge["high"],
            uncertainty=ge["uncertainty"],
            y_test=ge["y_test"],
        )

    sb = None
    if ps.get("save_behavior") is not None:
        sb = ps.get("save_behavior")
        from .plotspec import SaveBehavior

        sb = SaveBehavior(
            path=sb.get("path"),
            title=sb.get("title"),
            default_exts=tuple(sb.get("default_exts"))
            if sb.get("default_exts") is not None
            else None,
        )

    prov = ps.get("provenance") or {}

    spec = GlobalPlotSpec(
        title=obj.get("title") if obj.get("title") is not None else ps.get("title"),
        figure_size=tuple(obj.get("figure_size")) if obj.get("figure_size") is not None else None,
        global_entries=ge,
        kind=ps.get("kind"),
        mode=ps.get("mode"),
        plotspec_version=obj.get("plotspec_version", PLOTSPEC_VERSION)
        if obj.get("plot_spec") is not None
        else PLOTSPEC_VERSION,
        save_behavior=sb,
        data_slice_id=prov.get("data_slice_id"),
        rendering_seed=prov.get("rendering_seed"),
    )
    return spec


def plotspec_from_dict(obj: Dict[str, Any]) -> PlotSpec:
    """Deserialize a dict envelope (or a raw plot_spec dict) to a PlotSpec dataclass.

    Accepts either the envelope with 'plotspec_version' and 'plot_spec' or the
    inner plot_spec dict directly (backwards-compatible builders).
    """
    # Accept either the envelope or the inner dict
    if obj.get("plot_spec") is not None:
        version = obj.get("plotspec_version")
        if version is not None and version != PLOTSPEC_VERSION:
            from ..utils.exceptions import ValidationError

            raise ValidationError(
                f"unsupported or missing plotspec_version: {version}",
                details={"expected_version": PLOTSPEC_VERSION, "actual_version": version},
            )
        ps = obj["plot_spec"]
    else:
        ps = obj

    # Basic metadata
    kind = ps.get("kind")
    mode = ps.get("mode")

    # Title/figure_size may be present either at envelope-level or inner-level
    title = obj.get("title") if obj.get("title") is not None else ps.get("title")
    figure_size = obj.get("figure_size") if obj.get("figure_size") is not None else None

    # Header
    header = None
    if ps.get("header") is not None:
        h = ps["header"]
        header = IntervalHeaderSpec(
            pred=float(h.get("pred", 0.0)),
            low=float(h.get("low", 0.0)),
            high=float(h.get("high", 0.0)),
            xlim=tuple(h.get("xlim")) if h.get("xlim") is not None else None,
            xlabel=h.get("xlabel"),
            ylabel=h.get("ylabel"),
            dual=bool(h.get("dual", False)),
        )

    # Body
    body = None
    # Support both new 'feature_entries' lists and legacy 'body' with 'bars' lists
    if ps.get("feature_entries") is not None:
        b = ps.get("body") or {}
        bars_list = []
        for r in ps.get("feature_entries", []):
            bars_list.append(
                BarItem(
                    label=str(r.get("name", "")),
                    value=float(r.get("weight", 0.0)),
                    interval_low=None if r.get("low") is None else float(r.get("low")),
                    interval_high=None if r.get("high") is None else float(r.get("high")),
                    color_role=None,
                    instance_value=r.get("instance_value"),
                )
            )
        body = BarHPanelSpec(bars=bars_list, xlabel=b.get("xlabel"), ylabel=b.get("ylabel"))
    elif (
        ps.get("body") is not None
        and isinstance(ps.get("body"), dict)
        and ps.get("body").get("bars") is not None
    ):
        # legacy body shape: convert 'bars' entries into BarItem list
        b = ps.get("body")
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

    # Feature order
    feature_order = ps.get("feature_order")

    # Save behavior
    if ps.get("save_behavior") is not None:
        sb = ps["save_behavior"]
        # SaveBehavior is set below

    provenance = ps.get("provenance") or {}

    # Construct PlotSpec
    spec = PlotSpec(
        title=title,
        figure_size=tuple(figure_size) if figure_size is not None else None,
        header=header,
        body=body,
        kind=kind,
        mode=mode,
        feature_order=tuple(feature_order) if isinstance(feature_order, (list, tuple)) else None,
        plotspec_version=obj.get("plotspec_version", PLOTSPEC_VERSION)
        if obj.get("plot_spec") is not None
        else PLOTSPEC_VERSION,
        save_behavior=None,
        data_slice_id=provenance.get("data_slice_id"),
        rendering_seed=provenance.get("rendering_seed"),
    )
    # convert save_behavior dict to SaveBehavior dataclass if present
    if ps.get("save_behavior") is not None:
        from .plotspec import SaveBehavior

        sb = ps["save_behavior"]
        spec.save_behavior = SaveBehavior(
            path=sb.get("path"),
            title=sb.get("title"),
            default_exts=tuple(sb.get("default_exts"))
            if sb.get("default_exts") is not None
            else None,
        )

    return spec


def validate_plotspec(obj: Dict[str, Any]) -> None:
    """Lightweight validation for a PlotSpec envelope or inner plot_spec dict.

    Raises ValidationError for malformed shapes. Accepts either the full envelope
    or the inner plot_spec dict directly (useful for triangular/global builders).
    """
    if not isinstance(obj, dict):
        from ..utils.exceptions import ValidationError

        raise ValidationError(
            "PlotSpec payload must be a dict",
            details={"expected_type": "dict", "actual_type": type(obj).__name__},
        )

    # Accept envelope with 'plotspec_version' and 'plot_spec' keys, or plain inner dict
    if obj.get("plot_spec") is not None:
        version = obj.get("plotspec_version")
        if version != PLOTSPEC_VERSION:
            from ..utils.exceptions import ValidationError

            raise ValidationError(
                f"unsupported or missing plotspec_version: {version}",
                details={"expected_version": PLOTSPEC_VERSION, "actual_version": version},
            )
        ps = obj["plot_spec"]
    else:
        ps = obj

    # Basic metadata checks
    kind = ps.get("kind")
    mode = ps.get("mode")
    if not kind or not mode:
        from ..utils.exceptions import ValidationError

        raise ValidationError(
            "PlotSpec must include 'kind' and 'mode' metadata",
            details={"required": ["kind", "mode"]},
        )

    # Kind-aware validation
    PlotKindRegistry.validate_kind_and_mode(kind, mode)
    requirements = PlotKindRegistry.get_kind_requirements(kind)

    # Check required components based on kind
    if requirements["requires_body"]:
        feature_entries = ps.get("feature_entries")
        if feature_entries is None or not isinstance(feature_entries, list):
            from ..utils.exceptions import ValidationError

            raise ValidationError(
                f"PlotSpec kind '{kind}' requires 'feature_entries' list",
                details={"field": "feature_entries", "expected_type": "list"},
            )
        for i, b in enumerate(feature_entries):
            if "name" not in b or "weight" not in b:
                from ..utils.exceptions import ValidationError

                raise ValidationError(
                    f"feature entry at index {i} missing required fields 'name' or 'weight'",
                    details={
                        "entry_index": i,
                        "missing_fields": [f for f in ["name", "weight"] if f not in b],
                    },
                )

    if requirements["requires_header"] and ps.get("header") is None:
        from ..utils.exceptions import ValidationError

        raise ValidationError(
            f"PlotSpec kind '{kind}' requires 'header'",
            details={"required": ["header"]},
        )

    if requirements["requires_triangular"] and ps.get("triangular") is None:
        from ..utils.exceptions import ValidationError

        raise ValidationError(
            f"PlotSpec kind '{kind}' requires 'triangular' section",
            details={"required": ["triangular"]},
        )

    if requirements["requires_global_entries"] and ps.get("global_entries") is None:
        from ..utils.exceptions import ValidationError

        raise ValidationError(
            f"PlotSpec kind '{kind}' requires 'global_entries' section",
            details={"required": ["global_entries"]},
        )


__all__ = [
    "plotspec_to_dict",
    "plotspec_from_dict",
    "triangular_plotspec_from_dict",
    "global_plotspec_from_dict",
    "validate_plotspec",
    "PLOTSPEC_VERSION",
    "PlotKindRegistry",
]
