"""The single authoritative ADR-042 rule ranking (internal).

Every built-in ranking consumer delegates here: ``__repr__``, plot rule
selection, template narration, :func:`build_narrative_spec` and conjunction
candidate selection (ADR-042 D9). The functions order *already computed* rule
values. They never modify them, never drop rules except through an explicit
``top_k`` prefix, and never change the native rule-table order of serialized
payloads (D13).

Rankings are deterministic functions of one explanation's point estimates and
intervals. They carry no statistical guarantee and do not measure global
feature importance (D14).
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

from ..utils.exceptions import ValidationError

_KINDS = frozenset({"factual", "fast", "alternative"})
_ALTERNATIVE_METRICS = frozenset({"ensured", "feature_weight", "uncertainty"})


def validate_top_k(top_k: Any) -> int | None:
    """Return ``top_k`` as an ``int`` (or ``None``) under ADR-042 D8.

    ``None`` selects every rule, and a value at or above the rule count selects
    all of them. Zero selects none. A negative, boolean or non-integer value
    raises :class:`ValidationError`. NumPy integers are accepted.
    """
    if top_k is None:
        return None
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 0:
        raise ValidationError(
            "top_k must be a non-negative integer or None (ADR-042 D8)",
            details={"param": "top_k", "value": repr(top_k)},
        )
    return int(top_k)


def resolve_metric(
    kind: str, rnk_metric: str | None = None, rnk_weight: Any = 0.5
) -> tuple[str, float | None]:
    """Resolve ranking knobs to ``(metric, ensured_weight)`` (ADR-042 D1, D11).

    Factual and FAST explanations accept only ``None`` or ``"feature_weight"``.
    Alternatives default to ``"ensured"`` with ``rnk_weight`` in ``[-1, 1]``;
    ``"uncertainty"`` means ``ensured`` with weight ``0.0``, and
    ``"feature_weight"`` ranks by ``|delta|``. Invalid values raise
    :class:`ValidationError`.
    """
    if kind not in _KINDS:
        raise ValidationError(
            "Unknown explanation kind for ranking",
            details={"param": "kind", "value": kind},
        )
    if kind != "alternative":
        if rnk_metric not in (None, "feature_weight"):
            raise ValidationError(
                "Factual and FAST explanations are ranked by feature_weight only (ADR-042 D11)",
                details={"param": "rnk_metric", "value": rnk_metric},
            )
        return "feature_weight", None

    metric = "ensured" if rnk_metric is None else rnk_metric
    if metric not in _ALTERNATIVE_METRICS:
        raise ValidationError(
            "Unknown ranking metric for alternative explanations",
            details={
                "param": "rnk_metric",
                "value": rnk_metric,
                "allowed_values": sorted(_ALTERNATIVE_METRICS),
            },
        )
    if metric == "feature_weight":
        return "feature_weight", None
    if metric == "uncertainty":
        return "ensured", 0.0
    if isinstance(rnk_weight, bool) or not isinstance(rnk_weight, Real):
        raise ValidationError(
            "rnk_weight must be a number in [-1, 1]",
            details={"param": "rnk_weight", "value": repr(rnk_weight)},
        )
    weight = float(rnk_weight)
    if not -1.0 <= weight <= 1.0:  # also rejects NaN
        raise ValidationError(
            "rnk_weight must be a number in [-1, 1]",
            details={"param": "rnk_weight", "value": repr(rnk_weight)},
        )
    return "ensured", weight


def _number(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            "Ranking requires numeric rule values",
            details={"field": field, "value": repr(value)},
        ) from exc


def _tie_width(value: float) -> float:
    """D6: an unknown or non-finite width is maximally uncertain."""
    value = abs(value)
    return value if math.isfinite(value) else math.inf


def rank_scores(
    scores: Sequence[Any],
    *,
    widths: Sequence[Any] | None = None,
    top_k: int | None = None,
) -> list[int]:
    """Return indices strongest first by the ADR-042 D5/D6 total order.

    The key is: score descending, then width ascending (narrower first), then
    native index ascending. A ``NaN`` score ranks after every other value;
    signed infinities keep their signed order. A ``NaN`` or non-finite width
    loses width ties. ``top_k`` selects a prefix (D8).

    Parameters
    ----------
    scores : sequence of float
        Precomputed primary keys; higher ranks stronger.
    widths : sequence of float, optional
        Tie-break widths, one per score.
    top_k : int or None, default=None
        Number of indices to return; ``None`` returns all.

    Returns
    -------
    list of int
        Strongest-first indices into ``scores``.
    """
    k = validate_top_k(top_k)
    n = len(scores)
    primary = [_number(value, "score") for value in scores]
    if widths is None:
        tie = [0.0] * n
    else:
        if len(widths) != n:
            raise ValidationError(
                "Ranking scores and widths differ in length",
                details={"score_count": n, "width_count": len(widths)},
            )
        tie = [_tie_width(_number(value, "width")) for value in widths]

    def key(index: int) -> tuple[bool, float, float, int]:
        score = primary[index]
        is_nan = math.isnan(score)
        return (is_nan, 0.0 if is_nan else -score, tie[index], index)

    order = sorted(range(n), key=key)
    return order if k is None else order[:k]


def _interval_widths(
    rules: Mapping[str, Sequence[Any]],
    low_key: str,
    high_key: str,
    n: int,
    bounded: Sequence[bool] | None,
) -> list[float]:
    for name in (low_key, high_key):
        if name not in rules or len(rules[name]) != n:
            raise ValidationError(
                "Ranking rule arrays have inconsistent lengths",
                details={"field": name, "rule_count": n},
            )
    if bounded is not None and len(bounded) != n:
        raise ValidationError(
            "Ranking bounded flags have inconsistent lengths",
            details={"field": "bounded", "rule_count": n},
        )
    widths = []
    for i in range(n):
        if bounded is not None and not bounded[i]:
            widths.append(math.inf)
            continue
        width = _number(rules[high_key][i], high_key) - _number(rules[low_key][i], low_key)
        widths.append(_tie_width(width))
    return widths


def rank_rules(
    rules: Mapping[str, Sequence[Any]],
    *,
    kind: str,
    rnk_metric: str | None = None,
    rnk_weight: Any = 0.5,
    base_predict: Any = None,
    probability_task: bool = False,
    top_k: int | None = None,
    weight_bounded: Sequence[bool] | None = None,
    prediction_bounded: Sequence[bool] | None = None,
) -> list[int]:
    """Rank a native rule table strongest first (ADR-042 D1-D9).

    Parameters
    ----------
    rules : Mapping
        A native rule table (``get_rules()`` shape) with ``weight``,
        ``weight_low``, ``weight_high`` and, for alternatives, ``predict``,
        ``predict_low`` and ``predict_high``.
    kind : {"factual", "fast", "alternative"}
        Explanation type; selects the ranked quantity (D1).
    rnk_metric, rnk_weight
        Ranking knobs, resolved by :func:`resolve_metric`.
    base_predict : float, optional
        The explanation's own prediction. Required for probabilistic
        ``ensured`` ranking, which orients ``predict`` to the predicted class.
    probability_task : bool, default=False
        ``True`` for classification and thresholded regression. For plain
        regression the ensured score uses raw predictions and widths in target
        units, unnormalised (D1.3).
    top_k : int or None, default=None
        Prefix size (D8).
    weight_bounded, prediction_bounded : sequence of bool, optional
        Per-rule flags; ``False`` marks an interval with an unbounded endpoint,
        whose width ranks as infinite (D6).

    Returns
    -------
    list of int
        Strongest-first indices into the rule table.
    """
    metric, weight = resolve_metric(kind, rnk_metric, rnk_weight)
    k = validate_top_k(top_k)
    n = len(rules.get("weight", ()) or ())
    if n == 0:
        return []

    if kind != "alternative":
        widths = _interval_widths(rules, "weight_low", "weight_high", n, weight_bounded)
        primary = [abs(_number(value, "weight")) for value in rules["weight"]]
        return rank_scores(primary, widths=widths, top_k=k)

    # Alternatives: the width (u) is the alternative's own prediction interval.
    widths = _interval_widths(rules, "predict_low", "predict_high", n, prediction_bounded)
    if metric == "feature_weight":
        primary = [abs(_number(value, "weight")) for value in rules["weight"]]
        return rank_scores(primary, widths=widths, top_k=k)

    if "predict" not in rules or len(rules["predict"]) != n:
        raise ValidationError(
            "Ensured ranking requires one prediction per alternative rule",
            details={"field": "predict", "rule_count": n},
        )
    flip = False
    if probability_task:
        if base_predict is None:
            raise ValidationError(
                "Probabilistic ensured ranking requires the base prediction",
                details={"param": "base_predict"},
            )
        flip = not _number(base_predict, "base_predict") > 0.5
    coefficient = abs(weight)  # type: ignore[arg-type]
    negate = weight < 0  # type: ignore[operator]
    scores = []
    for index in range(n):
        p = _number(rules["predict"][index], "predict")
        if flip:
            p = 1.0 - p
        if negate:
            p = -p
        # score = (1 - |w|)(1 - u) + |w| p~, as in utils.calculate_metrics. A
        # disabled component contributes 0, never 0 * inf = NaN.
        certainty = 0.0 if coefficient == 1.0 else (1.0 - coefficient) * (1.0 - widths[index])
        prediction = 0.0 if coefficient == 0.0 else coefficient * p
        scores.append(certainty + prediction)
    return rank_scores(scores, widths=widths, top_k=k)


__all__ = ["rank_rules", "rank_scores", "resolve_metric", "validate_top_k"]
