"""Deterministic narrative facts for calibrated explanations (ADR-041).

:func:`build_narrative_spec` reads one serialized explanation, the ADR-005
payload emitted by ``to_json()``, and returns an immutable
:class:`NarrativeSpec`. The spec restates facts CE has already computed: the
calibrated prediction and its interval, each rule's weight and prediction
intervals, the problem context from ``metadata.narrative_context`` and a small
set of qualifiers. It performs no new prediction, calibration or explanation
analysis, and it does not render prose.

A correct spec does not make prose generated from it correct; any text built on
a spec still has to be checked against it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..utils import calculate_metrics, median_containment_sides
from ..utils.exceptions import ValidationError

NARRATIVE_SPEC_VERSION = "1.0.0"

_EXPLANATION_TYPES = ("factual", "alternative", "fast")
_PROBLEM_TYPES = {
    "binary_classification": "classification",
    "multiclass_classification": "classification",
    "regression": "regression",
    "probabilistic_regression": "regression",
}
_EPSILON = 1e-9  # floating-point tolerance shared with serialization invariants


@dataclass(frozen=True)
class NarrativeInterval:
    """A reported point value and its interval.

    Values are the reported floats, never rounded. ``low_bounded`` or
    ``high_bounded`` is ``False`` when that endpoint is not a calibrated bound:
    the value is infinite, or it is the ADR-021 placeholder reported for an
    infinite regression percentile. Narration must not present an unbounded
    endpoint as a numeric limit.
    """

    predict: float
    low: float
    high: float
    low_bounded: bool = True
    high_bounded: bool = True


@dataclass(frozen=True)
class NarrativeRuleFact:
    """One explanation rule restated as narrative facts.

    ``counterfactual`` is ``True`` for alternative rules, whose ``prediction``
    describes what CE predicts if the rule's condition held instead of the
    observed value. Factual and FAST rules describe the observed instance.
    """

    fact_id: str
    source: str
    rule: str
    feature: int | tuple[int, ...]
    feature_value: Any
    value_str: str | None
    is_conjunctive: bool
    counterfactual: bool
    weight: NarrativeInterval
    prediction: NarrativeInterval | None
    qualifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class NarrativeSpec:
    """Immutable, versioned narrative facts for one explanation.

    Attributes
    ----------
    spec_version : str
        Version of this contract (:data:`NARRATIVE_SPEC_VERSION`).
    spec_id : str
        SHA-256 of the canonical JSON of every other field. Identical semantic
        input gives an identical ``spec_id``.
    explanation_type : str
        ``"factual"``, ``"alternative"`` or ``"fast"``.
    task : str
        The payload ``task`` (``"classification"`` or ``"regression"``).
    index : int
        The instance index from the payload.
    prediction : NarrativeInterval
        The calibrated prediction for the instance (source ``/prediction``).
    rules : tuple of NarrativeRuleFact
        Every rule, strongest first in the ADR-042 canonical order.
    ranking : Mapping
        The ADR-042 ranking key, ``{"metric": "feature_weight"}`` or
        ``{"metric": "ensured", "weight": w}``.
    context : Mapping
        ``metadata.narrative_context`` as found in the payload; empty when absent.
    qualifiers : tuple of str
        Spec-level qualifiers that narration must not omit.
    """

    spec_version: str
    spec_id: str
    explanation_type: str
    task: str
    index: int
    prediction: NarrativeInterval
    rules: tuple[NarrativeRuleFact, ...]
    ranking: Mapping[str, Any]
    context: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    qualifiers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as plain JSON-compatible Python objects."""
        return _plain(
            {
                "spec_version": self.spec_version,
                "spec_id": self.spec_id,
                "explanation_type": self.explanation_type,
                "task": self.task,
                "index": self.index,
                "prediction": self.prediction,
                "rules": self.rules,
                "ranking": self.ranking,
                "context": self.context,
                "qualifiers": self.qualifiers,
            }
        )

    def to_json(self) -> str:
        """Return the canonical JSON text of the spec.

        Keys are sorted, separators are compact, floats use Python's
        shortest round-trip representation, and infinite values are written as
        the strings ``"inf"`` and ``"-inf"``.
        """
        return _canonical_json(self.to_dict())


def build_narrative_spec(
    explanation: Mapping[str, Any] | Any,
    *,
    rnk_metric: str | None = None,
    rnk_weight: float = 0.5,
) -> NarrativeSpec:
    """Build the :class:`NarrativeSpec` for one serialized explanation.

    Parameters
    ----------
    explanation : Mapping or Explanation
        One entry of ``explanations.to_json()["explanations"]``, or an ADR-008
        domain ``Explanation``, which is serialized first.
    rnk_metric : str or None, default=None
        ADR-042 ranking metric. Factual and FAST explanations accept only
        ``None`` or ``"feature_weight"``. Alternatives accept ``"ensured"``
        (the default), ``"feature_weight"`` and ``"uncertainty"`` (``ensured``
        with weight ``0.0``).
    rnk_weight : float, default=0.5
        The ``ensured`` weight in ``[-1, 1]``, for alternatives only.

    Returns
    -------
    NarrativeSpec
        Facts restated from the payload, rules in canonical order.

    Raises
    ------
    ValidationError
        If required data is missing, non-numeric, ``NaN`` or contradictory, if
        an interval violates the ADR-021 invariant, or if the ranking arguments
        do not apply to the explanation type.

    Notes
    -----
    A payload without ``metadata.narrative_context`` (written before it
    existed) is accepted: the spec gets an empty ``context`` and the
    ``"context_unavailable"`` qualifier. Without it the default ``ensured``
    ranking of a regression alternative is undefined, so that case raises.

    Examples
    --------
    >>> payload = explanations.to_json()["explanations"][0]  # doctest: +SKIP
    >>> spec = build_narrative_spec(payload)  # doctest: +SKIP
    >>> spec.prediction.predict  # doctest: +SKIP
    """
    payload = _as_payload(explanation)
    explanation_type = payload.get("explanation_type")
    if explanation_type not in _EXPLANATION_TYPES:
        raise ValidationError(
            "Narrative spec requires a factual, alternative or fast explanation",
            details={"field": "explanation_type", "value": explanation_type},
        )
    task = payload.get("task")
    if task not in ("classification", "regression"):
        raise ValidationError(
            "Narrative spec requires task 'classification' or 'regression'",
            details={"field": "task", "value": task},
        )
    context = _read_context(payload, task)
    problem_type = context.get("problem_type")
    counterfactual = explanation_type == "alternative"

    # ADR-021 percentile scope: containment holds only on the sides the
    # requested percentiles imply, and an endpoint reported for an infinite
    # percentile is a placeholder rather than a calibrated bound.
    sides, open_sides = (True, True), (False, False)
    percentiles = context.get("low_high_percentiles")
    if percentiles is not None:
        sides = median_containment_sides(percentiles)
        open_sides = (not math.isfinite(percentiles[0]), not math.isfinite(percentiles[1]))
    prediction_sides = (sides, open_sides)
    # A factual or FAST weight is ``predict - perturbed``, so its sides are mirrored.
    weight_sides = prediction_sides if counterfactual else (sides[::-1], open_sides[::-1])

    prediction = _interval(payload.get("prediction"), "/prediction", *prediction_sides)
    rules = []
    for i, raw in enumerate(payload.get("rules") or ()):
        if not isinstance(raw, Mapping):
            raise ValidationError(
                "Narrative spec rule must be an object", details={"source": f"/rules/{i}"}
            )
        weight = _interval(raw.get("rule_weight"), f"/rules/{i}/rule_weight", *weight_sides)
        rule_prediction = None
        if counterfactual or raw.get("rule_prediction"):
            rule_prediction = _interval(
                raw.get("rule_prediction"), f"/rules/{i}/rule_prediction", *prediction_sides
            )
        qualifiers = []
        if (
            not counterfactual
            and (weight.low <= 0.0 or not weight.low_bounded)
            and (weight.high >= 0.0 or not weight.high_bounded)
        ):
            qualifiers.append("weight_interval_contains_zero")
        if not _bounded(weight) or (rule_prediction is not None and not _bounded(rule_prediction)):
            qualifiers.append("unbounded_interval")
        feature = raw.get("feature", i)
        try:
            feature = (
                tuple(int(f) for f in feature)
                if isinstance(feature, (list, tuple))
                else int(feature)
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "Narrative spec rule feature must be an integer index or a list of them",
                details={"source": f"/rules/{i}/feature", "value": repr(feature)},
            ) from exc
        rules.append(
            NarrativeRuleFact(
                fact_id=f"rule/{i}",
                source=f"/rules/{i}",
                rule=str(raw.get("rule", "")),
                feature=feature,
                feature_value=_freeze(raw.get("feature_value")),
                value_str=raw.get("value_str"),
                is_conjunctive=bool(raw.get("is_conjunctive", False)),
                counterfactual=counterfactual,
                weight=weight,
                prediction=rule_prediction,
                qualifiers=tuple(qualifiers),
            )
        )

    ranking, ordered = _canonical_order(
        rules, explanation_type, task, problem_type, prediction, rnk_metric, rnk_weight
    )

    qualifiers = []
    if not context:
        qualifiers.append("context_unavailable")
    if context.get("reject", {}).get("rejected"):
        qualifiers.append("rejected")
    if not _bounded(prediction):
        qualifiers.append("unbounded_interval")

    content = {
        "spec_version": NARRATIVE_SPEC_VERSION,
        "explanation_type": explanation_type,
        "task": task,
        "index": int(payload.get("index", 0)),
        "prediction": prediction,
        "rules": tuple(ordered),
        "ranking": _freeze(ranking),
        "context": _freeze(context),
        "qualifiers": tuple(qualifiers),
    }
    spec_id = hashlib.sha256(_canonical_json(_plain(content)).encode("utf-8")).hexdigest()
    return NarrativeSpec(spec_id=spec_id, **content)


def _as_payload(explanation: Any) -> Mapping[str, Any]:
    from .models import Explanation

    if isinstance(explanation, Explanation):
        from ..serialization import to_json

        return to_json(explanation)
    if isinstance(explanation, Mapping):
        return explanation
    raise ValidationError(
        "Narrative spec requires a serialized explanation payload or a domain Explanation",
        details={"param": "explanation", "type": type(explanation).__name__},
    )


def _read_context(payload: Mapping[str, Any], task: str) -> dict[str, Any]:
    metadata = payload.get("metadata")
    context = metadata.get("narrative_context") if isinstance(metadata, Mapping) else None
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise ValidationError(
            "metadata.narrative_context must be an object",
            details={"source": "/metadata/narrative_context"},
        )
    context = dict(context)
    problem_type = context.get("problem_type")
    if problem_type is not None and _PROBLEM_TYPES.get(problem_type) != task:
        raise ValidationError(
            "narrative_context.problem_type contradicts the payload task",
            details={"problem_type": problem_type, "task": task},
        )
    if (problem_type == "probabilistic_regression") != (context.get("threshold") is not None):
        raise ValidationError(
            "narrative_context threshold is present exactly for probabilistic regression",
            details={"problem_type": problem_type, "threshold": context.get("threshold")},
        )
    percentiles = context.get("low_high_percentiles")
    if percentiles is not None:
        valid = (
            problem_type == "regression"
            and isinstance(percentiles, (list, tuple))
            and len(percentiles) == 2
            and all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in percentiles)
            and not any(math.isnan(p) for p in percentiles)
            and percentiles[0] < percentiles[1]
        )
        if not valid:
            raise ValidationError(
                "narrative_context.low_high_percentiles must be an increasing (low, high) "
                "pair for percentile regression",
                details={"problem_type": problem_type, "low_high_percentiles": percentiles},
            )
    reject = context.get("reject")
    if reject is not None and not (
        isinstance(reject, Mapping) and isinstance(reject.get("rejected"), bool)
    ):
        raise ValidationError(
            "narrative_context.reject must be an object with a boolean 'rejected'",
            details={"source": "/metadata/narrative_context/reject"},
        )
    return context


def _number(value: Any, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            "Narrative spec requires a numeric scalar value",
            details={"source": source, "value": repr(value)},
        )
    number = float(value)
    if math.isnan(number):
        raise ValidationError("Narrative spec rejects NaN values", details={"source": source})
    return number


def _interval(
    raw: Any, source: str, sides: tuple[bool, bool], open_sides: tuple[bool, bool]
) -> NarrativeInterval:
    if not isinstance(raw, Mapping):
        raise ValidationError(
            "Narrative spec requires an interval object with predict, low and high",
            details={"source": source},
        )
    predict, low, high = (_number(raw.get(k), f"{source}/{k}") for k in ("predict", "low", "high"))
    check_low, check_high = sides
    low_bounded = not open_sides[0] and math.isfinite(low)
    high_bounded = not open_sides[1] and math.isfinite(high)
    contained = (not check_low or low - _EPSILON <= predict) and (
        not check_high or predict <= high + _EPSILON
    )
    if low > high or not contained:
        raise ValidationError(
            "Interval violates the ADR-021 invariant low <= predict <= high",
            details={"source": source, "predict": predict, "low": low, "high": high},
        )
    return NarrativeInterval(predict, low, high, low_bounded, high_bounded)


def _bounded(interval: NarrativeInterval) -> bool:
    return interval.low_bounded and interval.high_bounded


def _canonical_order(
    rules: list[NarrativeRuleFact],
    explanation_type: str,
    task: str,
    problem_type: str | None,
    prediction: NarrativeInterval,
    rnk_metric: str | None,
    rnk_weight: float,
) -> tuple[dict[str, Any], list[NarrativeRuleFact]]:
    """Order rules strongest first by the ADR-042 D1, D5 and D6 total order.

    #204 (v1.1.0 T9) makes one internal implementation authoritative for every
    ranking consumer; this function is to delegate to it once that lands.
    """
    if explanation_type != "alternative":
        if rnk_metric not in (None, "feature_weight"):
            raise ValidationError(
                "Factual and FAST explanations are ranked by feature_weight only (ADR-042 D11)",
                details={"param": "rnk_metric", "value": rnk_metric},
            )

        def factual_key(item: tuple[int, NarrativeRuleFact]):
            index, fact = item
            width = abs(fact.weight.high - fact.weight.low)
            return (-abs(fact.weight.predict), _width(width, fact.weight), index)

        ordered = sorted(enumerate(rules), key=factual_key)
        return {"metric": "feature_weight"}, [fact for _, fact in ordered]

    metric = rnk_metric or "ensured"
    if metric not in ("ensured", "feature_weight", "uncertainty"):
        raise ValidationError(
            "Unknown ranking metric for alternative explanations",
            details={"param": "rnk_metric", "value": rnk_metric},
        )

    def uncertainty(fact: NarrativeRuleFact) -> float:
        interval = fact.prediction
        return _width(interval.high - interval.low, interval)

    if metric == "feature_weight":
        ordered = sorted(
            enumerate(rules),
            key=lambda item: (-abs(item[1].weight.predict), uncertainty(item[1]), item[0]),
        )
        return {"metric": "feature_weight"}, [fact for _, fact in ordered]

    weight = 0.0 if metric == "uncertainty" else float(rnk_weight)
    if task == "classification" or problem_type == "probabilistic_regression":
        oriented = prediction.predict > 0.5
    elif problem_type == "regression":
        oriented = None
    else:
        raise ValidationError(
            "The ensured ranking of a regression alternative needs "
            "metadata.narrative_context to know whether it is thresholded",
            details={"param": "rnk_metric", "value": metric},
        )

    def score(fact: NarrativeRuleFact) -> float:
        p = fact.prediction.predict
        if oriented is False:
            p = 1.0 - p
        return float(calculate_metrics(uncertainty(fact), p, w=weight, metric="ensured"))

    def ensured_key(item: tuple[int, NarrativeRuleFact]):
        index, fact = item
        value = score(fact)
        return (math.isnan(value), 0.0 if math.isnan(value) else -value, uncertainty(fact), index)

    ordered = sorted(enumerate(rules), key=ensured_key)
    return {"metric": "ensured", "weight": weight}, [fact for _, fact in ordered]


def _width(width: float, interval: NarrativeInterval) -> float:
    # ADR-042 D6: an unknown or unbounded width ranks as maximally uncertain.
    if not _bounded(interval) or not math.isfinite(width):
        return math.inf
    return width


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, (NarrativeInterval, NarrativeRuleFact)):
        return {k: _plain(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


__all__ = [
    "NARRATIVE_SPEC_VERSION",
    "NarrativeInterval",
    "NarrativeRuleFact",
    "NarrativeSpec",
    "build_narrative_spec",
]
