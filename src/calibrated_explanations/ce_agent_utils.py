"""CE-First helper utilities for Calibrated Explanations.

This module is intentionally CE-first: every public helper validates that the
calibrated_explanations package is available, uses WrapCalibratedExplainer,
checks .fitted / .calibrated states, and defaults to calibrated outputs.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import logging
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .core.exceptions import (
    ConfigurationError,
    ModelNotSupportedError,
    NotFittedError,
    ValidationError,
)
from .utils.deprecations import deprecate

LOGGER = logging.getLogger(__name__)


CE_FIRST_POLICY: Mapping[str, Any] = {
    "requires_library": "calibrated_explanations",
    "required_class": "WrapCalibratedExplainer",
    "required_methods": [
        "fit",
        "calibrate",
        "predict",
        "predict_proba",
        "explain_factual",
        "explore_alternatives",
        "plot",
    ],
    "required_attributes": ["fitted", "calibrated"],
    "default_options": {
        "calibrated": True,
        "uq_interval": False,
        "low_high_percentiles": (5, 95),
    },
    "failure_messages": {
        "missing_library": "calibrated_explanations is required. Install with: pip install calibrated-explanations",
        "invalid_wrapper": "Wrapper must be a WrapCalibratedExplainer (or subclass) from calibrated_explanations.",
        "not_fitted": "Wrapper must be fitted before calibration or explanation.",
        "not_calibrated": "Wrapper must be calibrated before prediction or explanation.",
    },
}


_NARRATIVE_FORMAT_TO_EXPERTISE: Mapping[str, str] = {
    "short": "beginner",
    "bullet": "intermediate",
    "long": "advanced",
}


@dataclass
class TelemetryEvent:
    """Simple telemetry event payload."""

    name: str
    payload: Mapping[str, Any]


_TELEMETRY_HOOK: Optional[Callable[[TelemetryEvent], None]] = None


def set_telemetry_hook(hook: Optional[Callable[[TelemetryEvent], None]]) -> None:
    """Set a telemetry hook to receive helper events."""
    global _TELEMETRY_HOOK
    _TELEMETRY_HOOK = hook


def _emit(event_name: str, **payload: Any) -> None:
    """Emit telemetry events if a hook is configured."""
    if _TELEMETRY_HOOK is None:
        return
    try:
        _TELEMETRY_HOOK(TelemetryEvent(name=event_name, payload=dict(payload)))
    except Exception as exc:  # pragma: no cover - telemetry must not break runtime  # adr002_allow
        LOGGER.debug("Telemetry hook failed: %s", exc)


def optional_cache(
    enabled: bool = True, maxsize: int = 128
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Apply optional caching to helper functions.

    This uses functools.lru_cache when enabled; otherwise, returns a no-op wrapper.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not enabled:
            return func
        return functools.lru_cache(maxsize=maxsize)(func)

    return decorator


def _require_ce() -> type:
    """Return WrapCalibratedExplainer or raise a CE-first error."""
    try:
        module = importlib.import_module("calibrated_explanations")
    except ImportError as exc:
        raise ConfigurationError(
            CE_FIRST_POLICY["failure_messages"]["missing_library"],
            details={"requirement": "install calibrated-explanations"},
        ) from exc
    try:
        wrap_cls = module.WrapCalibratedExplainer
    except AttributeError as exc:
        raise ConfigurationError(
            CE_FIRST_POLICY["failure_messages"]["missing_library"],
            details={"requirement": "WrapCalibratedExplainer export"},
        ) from exc
    return wrap_cls


def _is_wrapper(obj: Any, wrap_cls: type) -> bool:
    """Return ``True`` when ``obj`` is an instance of the CE wrapper class."""
    return isinstance(obj, wrap_cls)


def policy_as_dict() -> Dict[str, Any]:
    """Return a JSON-serializable copy of the CE-first policy."""
    return dict(CE_FIRST_POLICY)


def serialize_policy() -> str:
    """Serialize the CE-first policy to JSON for registries or docs."""
    return json.dumps(policy_as_dict(), indent=2, sort_keys=True)


def _supports_kwarg(callable_obj: Callable[..., Any], kwarg: str) -> bool:
    """Check whether a callable accepts a named keyword argument."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == kwarg:
            return True
    return False


def _filter_kwargs(callable_obj: Callable[..., Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to those supported by ``callable_obj``."""
    if not kwargs:
        return {}
    if _supports_kwarg(callable_obj, "kwargs"):
        return dict(kwargs)
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


def _ensure_required_methods(wrapper: Any, required: Iterable[str]) -> None:
    """Raise when required wrapper methods are missing."""
    missing = [name for name in required if not hasattr(wrapper, name)]
    if missing:
        raise ModelNotSupportedError(
            f"Wrapper missing required methods: {missing}",
            details={"missing": missing},
        )


def _ensure_required_attrs(wrapper: Any, required: Iterable[str]) -> None:
    """Raise when required wrapper attributes are missing."""
    missing = [name for name in required if not hasattr(wrapper, name)]
    if missing:
        raise ModelNotSupportedError(
            f"Wrapper missing required attributes: {missing}",
            details={"missing": missing},
        )


def _validate_wrapper_state(
    wrapper: Any, *, require_fitted: bool = True, require_calibrated: bool = True
) -> None:
    """Validate fitted/calibrated state flags on the CE wrapper."""
    if require_fitted and not getattr(wrapper, "fitted", False):
        raise NotFittedError(
            CE_FIRST_POLICY["failure_messages"]["not_fitted"],
            details={"requirement": "fit()"},
        )
    if require_calibrated and not getattr(wrapper, "calibrated", False):
        raise ValidationError(
            CE_FIRST_POLICY["failure_messages"]["not_calibrated"],
            details={"requirement": "calibrate()"},
        )


def ensure_ce_first_wrapper(model_or_wrapper: Any) -> Any:
    """Ensure a WrapCalibratedExplainer wrapper and validate CE-first invariants.

    Parameters
    ----------
    model_or_wrapper : Any
        Either a raw model or an instance of WrapCalibratedExplainer.

    Returns
    -------
    WrapCalibratedExplainer
        The validated wrapper.
    """
    wrap_cls = _require_ce()
    if _is_wrapper(model_or_wrapper, wrap_cls):
        _ensure_required_attrs(model_or_wrapper, CE_FIRST_POLICY["required_attributes"])
        _ensure_required_methods(model_or_wrapper, CE_FIRST_POLICY["required_methods"])
        return model_or_wrapper
    wrapper = wrap_cls(model_or_wrapper)
    _ensure_required_attrs(wrapper, CE_FIRST_POLICY["required_attributes"])
    _ensure_required_methods(wrapper, CE_FIRST_POLICY["required_methods"])
    return wrapper


def fit_and_calibrate(
    wrapper: Any,
    x_train: Any,
    y_train: Any,
    x_cal: Any,
    y_cal: Any,
    *,
    learner_kwargs: Optional[Mapping[str, Any]] = None,
    explainer_kwargs: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Fit the learner and calibrate the explainer with CE-first validation.

    learner_kwargs are passed to wrapper.fit (and therefore to learner.fit).
    explainer_kwargs are passed to wrapper.calibrate.

    Extra kwargs can contain "learner" or "explainer" dicts; they are merged.
    """
    _emit("ce.fit_and_calibrate.start")
    wrapper = ensure_ce_first_wrapper(wrapper)
    learner_kwargs = dict(learner_kwargs or {})
    explainer_kwargs = dict(explainer_kwargs or {})
    if "learner" in kwargs:
        learner_kwargs.update(kwargs.get("learner", {}))
    if "explainer" in kwargs:
        explainer_kwargs.update(kwargs.get("explainer", {}))
    wrapper.fit(x_train, y_train, **learner_kwargs)
    if not wrapper.fitted:
        raise NotFittedError(
            CE_FIRST_POLICY["failure_messages"]["not_fitted"],
            details={"stage": "fit"},
        )
    wrapper.calibrate(x_cal, y_cal, **explainer_kwargs)
    if not wrapper.calibrated:
        raise ValidationError(
            CE_FIRST_POLICY["failure_messages"]["not_calibrated"],
            details={"stage": "calibrate"},
        )
    _emit("ce.fit_and_calibrate.end")
    return wrapper


def _ce_strict_call(callable_obj: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke a CE API callable without silently dropping unsupported kwargs.

    Unlike ``_safe_call_with_kwargs``, this function does **not** filter out
    unknown keyword arguments. A ``TypeError`` caused by an unexpected kwarg is
    converted to a ``ValidationError`` so callers learn the CE public API
    instead of receiving a silent no-op.

    Parameters
    ----------
    callable_obj : Callable
        The CE public-API callable to invoke.
    *args : Any
        Positional arguments forwarded unchanged.
    **kwargs : Any
        Keyword arguments forwarded unchanged.

    Returns
    -------
    Any
        Return value from ``callable_obj``.

    Raises
    ------
    ValidationError
        When ``callable_obj`` raises ``TypeError`` due to an unsupported
        keyword argument, surfacing the CE-first policy violation explicitly.
    """
    try:
        return callable_obj(*args, **kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument" in msg or "got an unexpected" in msg:
            func_name = getattr(callable_obj, "__name__", repr(callable_obj))
            raise ValidationError(
                f"CE API call to {func_name!r} received an unsupported keyword argument. "
                "Consult the CE public API for accepted kwargs.",
                details={"cause": msg},
            ) from exc
        raise


def _extract_top_features(explanation: Any, top_k: int = 3) -> List[str]:
    """Extract top-ranked rule texts via the CE public get_rules() API."""
    if explanation is None or not hasattr(explanation, "get_rules"):
        return []
    try:
        rules = explanation.get_rules()
    except Exception:  # pragma: no cover - defensive  # adr002_allow
        return []
    if isinstance(rules, Mapping) and "rule" in rules:
        return list(rules.get("rule", []))[:top_k]
    return []


def _extract_prediction_triplet(prediction: Any) -> Optional[Tuple[Any, Any, Any]]:
    """Extract ``(predict, low, high)`` from prediction mappings when present."""
    if not isinstance(prediction, Mapping):
        return None
    if all(key in prediction for key in ("predict", "low", "high")):
        return (prediction.get("predict"), prediction.get("low"), prediction.get("high"))
    return None


def summarize_explanations(explanations: Any, *, top_k: int = 5) -> Mapping[str, Any]:
    """Summarize CE outputs (rules, conjunctions, and uncertainty metadata).

    This helper is deliberately CE-specific and captures the main CE selling
    points:
    - factual/alternative rule explanations
    - conjunctive (interaction-like) rules
    - uncertainty quantification (intervals for predictions / feature weights)
    - probabilistic regression metadata (thresholded regression)

    Parameters
    ----------
    explanations : Any
        A CE explanation collection (e.g., ``CalibratedExplanations``).
    top_k : int, default=5
        Maximum number of rule strings to include.

    Returns
    -------
    Mapping[str, Any]
        JSON-safe summary dictionary.
    """
    first = explanations[0] if hasattr(explanations, "__getitem__") else None
    prediction = getattr(first, "prediction", None)
    pred_triplet = _extract_prediction_triplet(prediction)

    # get_rules() already returns conjunctive rules when present (CE-first)
    top_rules = _extract_top_features(first, top_k=top_k)
    has_conjunctions = bool(getattr(first, "has_conjunctive_rules", False))

    confidence = None
    if hasattr(explanations, "get_confidence"):
        try:
            confidence = float(explanations.get_confidence())
        except Exception:  # pragma: no cover - defensive  # adr002_allow
            confidence = None

    return {
        "prediction": {"predict": pred_triplet[0], "low": pred_triplet[1], "high": pred_triplet[2]}
        if pred_triplet is not None
        else prediction,
        "top_rules": top_rules,
        "has_conjunctions": has_conjunctions,
        "top_conjunction_rules": top_rules if has_conjunctions else [],
        "low_high_percentiles": getattr(explanations, "low_high_percentiles", None),
        "confidence": confidence,
        "y_threshold": getattr(explanations, "y_threshold", None),
    }


def _coerce_guarded_audit_payload(audit_or_obj: Any) -> Mapping[str, Any]:
    """Normalize guarded audit input to a payload mapping."""
    if isinstance(audit_or_obj, Mapping):
        return audit_or_obj
    getter = getattr(audit_or_obj, "get_guarded_audit", None)
    if callable(getter):
        payload = getter()
        if isinstance(payload, Mapping):
            return payload
    raise ValidationError(
        "Expected guarded audit mapping or object exposing get_guarded_audit().",
        details={"input_type": type(audit_or_obj).__name__},
    )


def format_guarded_audit_table(
    audit_or_obj: Any,
    *,
    max_rows: int = 30,
    bound_decimals: int = 4,
    pvalue_decimals: int = 4,
    include_reason_legend: bool = True,
) -> str:
    """Format guarded audit payload into a compact text table.

    Parameters
    ----------
    audit_or_obj : Any
        A guarded audit payload dict, a guarded explanation, or a guarded collection.
    max_rows : int, default=30
        Maximum number of interval rows to render.
    bound_decimals : int, default=4
        Decimal precision for interval bounds and representative values.
    pvalue_decimals : int, default=4
        Decimal precision for p-values.
    include_reason_legend : bool, default=True
        Append compact reason counts and a legend to aid interpretation.

    Returns
    -------
    str
        A compact multi-line table with summary and interval rows.
    """
    payload = _coerce_guarded_audit_payload(audit_or_obj)
    rows: List[Mapping[str, Any]] = []
    summary: Mapping[str, Any] = payload.get("summary", {})
    n_instances = summary.get("n_instances")

    if isinstance(payload.get("instances"), list):
        for inst in payload["instances"]:
            inst_idx = int(inst.get("instance_index", -1))
            for rec in inst.get("intervals", []) or []:
                rows.append({"instance_index": inst_idx, **rec})
    else:
        inst_idx = int(payload.get("instance_index", -1))
        for rec in payload.get("intervals", []) or []:
            rows.append({"instance_index": inst_idx, **rec})
        n_instances = 1 if n_instances is None else n_instances

    def _fmt_num(value: Any, *, decimals: int) -> str:
        if value is None:
            return ""
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return str(value)
        if np.isnan(fval):
            return "nan"
        if np.isposinf(fval):
            return "inf"
        if np.isneginf(fval):
            return "-inf"
        text = f"{fval:.{max(0, int(decimals))}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text

    def _fmt_interval(lower: Any, upper: Any) -> str:
        lower_str = _fmt_num(lower, decimals=bound_decimals)
        upper_str = _fmt_num(upper, decimals=bound_decimals)
        return f"({lower_str}, {upper_str}]"

    def _interval_bounds(rec: Mapping[str, Any]) -> tuple:
        """Return (lower, upper) to display."""
        return rec.get("lower"), rec.get("upper")

    header = "inst feat name                interval                  p      conf emit mrg  reason"
    divider = "-" * len(header)
    lines = [
        "Guarded Audit Summary",
        (
            f"instances={int(n_instances) if n_instances is not None else '?'} "
            f"tested={summary.get('intervals_tested', 0)} "
            f"conforming={summary.get('intervals_conforming', 0)} "
            f"removed_guard={summary.get('intervals_removed_guard', 0)} "
            f"emitted={summary.get('intervals_emitted', 0)}"
        ),
        header,
        divider,
    ]

    limited = rows[: max(0, int(max_rows))]
    for rec in limited:
        name = str(rec.get("feature_name", ""))[:18]
        disp_lower, disp_upper = _interval_bounds(rec)
        interval = _fmt_interval(disp_lower, disp_upper)[:24]
        p_val = rec.get("p_value", "")
        p_str = _fmt_num(p_val, decimals=pvalue_decimals) if p_val not in ("", None) else ""
        conf = "Y" if rec.get("conforming") else "N"
        emit = "Y" if rec.get("emitted") else "N"
        mrg = "Y" if rec.get("is_merged") else "N"
        reason = str(rec.get("emission_reason", ""))[:18]
        lines.append(
            f"{int(rec.get('instance_index', -1)):>4} "
            f"{int(rec.get('feature', -1)):>4} "
            f"{name:<18} "
            f"{interval:<24} "
            f"{p_str:>6} "
            f"{conf:^4} "
            f"{emit:^4} "
            f"{mrg:^4} "
            f"{reason}"
        )

    if len(rows) > len(limited):
        lines.append(f"... truncated {len(rows) - len(limited)} row(s)")
    if include_reason_legend:
        reason_counts = Counter(str(rec.get("emission_reason", "")) for rec in rows)
        known_order = [
            "emitted",
            "removed_guard",
            "design_excluded",
            "baseline_equal",
            "zero_impact",
            "ignored_feature",
        ]
        counts = [f"{key}={reason_counts[key]}" for key in known_order if reason_counts.get(key, 0)]
        if counts:
            lines.append("reason_counts: " + ", ".join(counts))
        lines.append(
            "legend: emitted=rule kept; removed_guard=non-conforming; "
            "design_excluded=not eligible in this mode; "
            "baseline_equal=no prediction change; "
            "zero_impact=factual equals base; "
            "ignored_feature=explicitly ignored; "
            "mrg=Y: merged adjacent bins; emitted bounds shown"
        )
    return "\n".join(lines)


def print_guarded_audit_table(
    audit_or_obj: Any,
    *,
    max_rows: int = 30,
    bound_decimals: int = 4,
    pvalue_decimals: int = 4,
    include_reason_legend: bool = True,
) -> None:
    """Print a compact guarded audit table."""
    print(
        format_guarded_audit_table(
            audit_or_obj,
            max_rows=max_rows,
            bound_decimals=bound_decimals,
            pvalue_decimals=pvalue_decimals,
            include_reason_legend=include_reason_legend,
        )
    )


def explain_and_narrate(
    wrapper: Any,
    x: Any,
    *,
    mode: str = "factual",
    expertise_level: str = "beginner",
    narrative_format: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, str]:
    """Generate explanations and a CE-first narrative summary.

    Returns the explanations collection and narrative text.

    Parameters
    ----------
    expertise_level : str, default="beginner"
        Narrative detail level: "beginner", "intermediate", or "advanced".
    narrative_format : str, optional
        Deprecated alias for ``expertise_level`` using legacy names
        ("short" → "beginner", "bullet" → "intermediate", "long" → "advanced").
        If provided, takes precedence over ``expertise_level`` with a warning.
    """
    if narrative_format is not None:
        deprecate(
            "narrative_format is deprecated; use expertise_level='beginner'|'intermediate'|'advanced'.",
            key="ce_agent_utils.narrative_format",
            stacklevel=2,
        )
        expertise_level = _NARRATIVE_FORMAT_TO_EXPERTISE.get(narrative_format, expertise_level)
    _emit("ce.explain.start", mode=mode)
    return enforce_ce_first_and_execute(
        _explain_and_narrate_impl,
        wrapper,
        x,
        mode=mode,
        expertise_level=expertise_level,
        **kwargs,
    )


def _explain_and_narrate_impl(
    wrapper: Any,
    x: Any,
    *,
    mode: str,
    expertise_level: str,
    **kwargs: Any,
) -> Tuple[Any, str]:
    explainer = wrapper
    mode_normalized = mode.lower().strip()
    if mode_normalized not in {"factual", "alternatives"}:
        raise ValidationError(
            "mode must be 'factual' or 'alternatives'",
            details={"mode": mode, "allowed": ["factual", "alternatives"]},
        )
    explain_func = (
        explainer.explain_factual
        if mode_normalized == "factual"
        else explainer.explore_alternatives
    )
    explanations = _ce_strict_call(explain_func, x, **kwargs)
    narrative = ""
    if hasattr(explanations, "to_narrative"):
        try:
            narrative = explanations.to_narrative(
                expertise_level=expertise_level,
                output_format="text",
            )
        except Exception:  # pragma: no cover - defensive  # adr002_allow
            narrative = str(explanations)
    _emit("ce.explain.end", mode=mode)
    return explanations, narrative


def explain_and_summarize(
    wrapper: Any,
    x: Any,
    *,
    mode: str = "factual",
    expertise_level: str = "beginner",
    narrative_format: Optional[str] = None,
    add_conjunctions_params: Optional[Mapping[str, Any]] = None,
    uq_interval: bool = True,
    threshold: Optional[Any] = None,
    low_high_percentiles: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> Mapping[str, Any]:
    """CE-first convenience helper for structured explanation output.

    Generates explanations, narrative text, optional conjunctions, and
    calibrated predictions in a single call. This is a post-processing utility
    over the canonical CE-first lifecycle (fit → calibrate → explain); it is
    **not** the primary CE API. Prefer calling ``WrapCalibratedExplainer``
    methods directly when orchestrating CE-first workflows.

    The semantics of ``threshold`` vs ``low_high_percentiles`` follow ADR-021.

    Returns
    -------
    Mapping[str, Any]
        Payload containing ``explanations``, ``narrative``, and a JSON-safe
        ``summary``.
    """
    wrapper = ensure_ce_first_wrapper(wrapper)
    _validate_wrapper_state(wrapper, require_fitted=True, require_calibrated=True)

    explain_kwargs: Dict[str, Any] = dict(kwargs)
    if threshold is not None:
        explain_kwargs["threshold"] = threshold
    if low_high_percentiles is not None:
        explain_kwargs["low_high_percentiles"] = tuple(low_high_percentiles)

    explanations, narrative = explain_and_narrate(
        wrapper,
        x,
        mode=mode,
        expertise_level=expertise_level,
        narrative_format=narrative_format,
        **explain_kwargs,
    )

    if add_conjunctions_params is not None:
        add_conjunctions(explanations, **dict(add_conjunctions_params))

    predictions = get_calibrated_predictions(
        wrapper,
        x,
        calibrated=True,
        uq_interval=uq_interval,
        threshold=threshold,
        low_high_percentiles=tuple(low_high_percentiles)
        if low_high_percentiles is not None
        else None,
    )

    return {
        "wrapper": wrapper,
        "predictions": predictions,
        "explanations": explanations,
        "narrative": narrative,
        "summary": summarize_explanations(explanations),
    }


def add_conjunctions(explanations: Any, **params: Any) -> Any:
    """Add conjunctions to a explanations collection."""
    if not hasattr(explanations, "add_conjunctions"):
        raise ModelNotSupportedError("Explanations object does not support add_conjunctions")
    _emit("ce.conjunctions.add", scope="collection")
    return explanations.add_conjunctions(**params)


def add_conjunctions_to_one(explanations: Any, idx: int, **params: Any) -> Any:
    """Add conjunctions to a single explanation object at index idx."""
    explanation = explanations[idx]
    if not hasattr(explanation, "add_conjunctions"):
        raise ModelNotSupportedError("Explanation object does not support add_conjunctions")
    _emit("ce.conjunctions.add", scope="single", index=idx)
    return explanation.add_conjunctions(**params)


def get_calibrated_predictions(
    wrapper: Any,
    x: Any,
    *,
    calibrated: bool = True,
    uq_interval: bool = False,
    threshold: Optional[Any] = None,
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Return calibrated predictions/probabilities (CE-first by default)."""
    wrapper = ensure_ce_first_wrapper(wrapper)
    if calibrated:
        _validate_wrapper_state(wrapper, require_fitted=True, require_calibrated=True)
    predict_kwargs = dict(kwargs)
    if threshold is not None:
        predict_kwargs["threshold"] = threshold
    if uq_interval:
        predict_kwargs["uq_interval"] = uq_interval
    # For regression, low/high percentiles control conformal prediction intervals.
    if "low_high_percentiles" in kwargs and kwargs.get("low_high_percentiles") is None:
        predict_kwargs.pop("low_high_percentiles", None)
    prediction = _ce_strict_call(wrapper.predict, x, **predict_kwargs)
    proba = None
    if hasattr(wrapper, "predict_proba"):
        try:
            proba = _ce_strict_call(wrapper.predict_proba, x, **predict_kwargs)
        except (ModelNotSupportedError, ValidationError):
            # predict_proba is not applicable (e.g. regression without a threshold)
            proba = None
    return {"prediction": prediction, "probability": proba}


def _safe_call_with_kwargs(callable_obj: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke a callable after dropping unsupported keyword arguments.

    Non-canonical compat helper — silently drops unsupported kwargs.
    Only used by ``get_uncalibrated_predictions`` to forward kwargs to raw
    learner methods that may not accept them. Canonical CE-first paths must
    use ``_ce_strict_call`` instead.
    """
    filtered = _filter_kwargs(callable_obj, kwargs)
    return callable_obj(*args, **filtered)


def get_uncalibrated_predictions(wrapper: Any, x: Any, **kwargs: Any) -> Mapping[str, Any]:
    """Return uncalibrated outputs directly from the learner.

    .. warning::
        **Non-canonical escape hatch.** Raw learner outputs bypass CE
        calibration and interval semantics. A ``UserWarning`` is always emitted
        when this function is called. Prefer ``get_calibrated_predictions`` for
        CE-first workflows.

    Parameters
    ----------
    wrapper : Any
        A fitted ``WrapCalibratedExplainer`` instance.
    x : Any
        Input features.
    **kwargs : Any
        Extra kwargs forwarded to the learner (compat path; silently filtered
        if the learner does not accept them).

    Returns
    -------
    Mapping[str, Any]
        ``{"prediction": ..., "probability": ...}`` from the raw learner.
    """
    warnings.warn(
        "get_uncalibrated_predictions returns raw learner outputs that bypass "
        "CE calibration. This is a non-canonical escape hatch. Use "
        "get_calibrated_predictions for CE-first workflows.",
        UserWarning,
        stacklevel=2,
    )
    LOGGER.info(
        "ce_agent_utils: get_uncalibrated_predictions called; "
        "non-canonical path bypassing CE calibration"
    )
    wrapper = ensure_ce_first_wrapper(wrapper)
    learner = wrapper.learner
    prediction = None
    if hasattr(learner, "predict"):
        prediction = _safe_call_with_kwargs(learner.predict, x, **kwargs)
    proba = None
    if hasattr(learner, "predict_proba"):
        proba = _safe_call_with_kwargs(learner.predict_proba, x, **kwargs)
    return {"prediction": prediction, "probability": proba}


def wrap_and_explain(
    model: Any,
    x_train: Any,
    y_train: Any,
    x_cal: Any,
    y_cal: Any,
    x_test: Any,
    *,
    mode: str = "factual",
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Wrap, fit, calibrate, explain, and narrate.

    This is a post-processing utility, not the canonical CE-first path. Prefer
    calling ``WrapCalibratedExplainer`` methods directly for production
    workflows. Note that ``**kwargs`` are forwarded to both
    ``fit_and_calibrate`` and ``explain_and_narrate``; pass fit-specific and
    explain-specific params via those functions directly if they differ.
    """
    wrapper = ensure_ce_first_wrapper(model)
    wrapper = fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal, **kwargs)
    explanations, narrative = explain_and_narrate(wrapper, x_test, mode=mode, **kwargs)
    explanation = explanations[0] if hasattr(explanations, "__getitem__") else None
    plot = None
    if explanation is not None and hasattr(explanation, "plot"):
        try:
            plot = explanation.plot()
        except Exception:  # pragma: no cover - plotting optional  # adr002_allow
            plot = None
    return {
        "wrapper": wrapper,
        "explanations": explanations,
        "narrative": narrative,
        "plot": plot,
    }


def enforce_ce_first_and_execute(
    action_callable: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Validate CE-first requirements before executing an action."""
    wrap_cls = _require_ce()
    wrapper = None
    for candidate in args:
        if _is_wrapper(candidate, wrap_cls):
            wrapper = candidate
            break
    if wrapper is None and "wrapper" in kwargs:
        wrapper = kwargs.get("wrapper")
    if wrapper is None or not _is_wrapper(wrapper, wrap_cls):
        raise ModelNotSupportedError(CE_FIRST_POLICY["failure_messages"]["invalid_wrapper"])
    _ensure_required_attrs(wrapper, CE_FIRST_POLICY["required_attributes"])
    _ensure_required_methods(wrapper, CE_FIRST_POLICY["required_methods"])
    _validate_wrapper_state(wrapper, require_fitted=True, require_calibrated=True)
    return action_callable(*args, **kwargs)


def probe_optional_features(
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
    find_spec: Optional[Callable[[str], Any]] = None,
) -> Mapping[str, Any]:
    """Probe optional CE features (conditional, difficulty, reject, plugins)."""
    report: Dict[str, Any] = {"available": {}, "warnings": []}
    optional_targets = {
        "crepes.extras.MondrianCategorizer": "conditional/Mondrian categorizer",
        "crepes.extras.DifficultyEstimator": "difficulty estimation",
        "calibrated_explanations.core.reject": "reject policies",
        "calibrated_explanations.plugins": "plugin API",
    }
    for target, label in optional_targets.items():
        try:
            module_name, attr = target.rsplit(".", 1)
            if find_spec is not None and find_spec(module_name) is None:
                raise ImportError(f"Module spec not found for {module_name}")
            module = import_module(module_name)
            _ = getattr(module, attr)
            report["available"][label] = True
        except Exception as exc:  # pragma: no cover - defensive  # adr002_allow
            report["available"][label] = False
            report["warnings"].append(f"Optional feature missing: {label} ({exc})")
    if report["warnings"]:
        warnings.warn("; ".join(report["warnings"]), stacklevel=2)
    return report


__all__ = [
    "CE_FIRST_POLICY",
    "TelemetryEvent",
    "set_telemetry_hook",
    "optional_cache",
    "policy_as_dict",
    "serialize_policy",
    "ensure_ce_first_wrapper",
    "fit_and_calibrate",
    "explain_and_narrate",
    "explain_and_summarize",
    "summarize_explanations",
    "format_guarded_audit_table",
    "print_guarded_audit_table",
    "add_conjunctions",
    "add_conjunctions_to_one",
    "get_calibrated_predictions",
    "get_uncalibrated_predictions",
    "wrap_and_explain",
    "enforce_ce_first_and_execute",
    "probe_optional_features",
]
