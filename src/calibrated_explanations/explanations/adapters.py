"""Adapters between legacy explanation dicts and internal domain models.

This module provides conversion helpers that allow constructing the
internal domain model from existing legacy dict shapes and emitting a
legacy-shaped dict from the domain model. Public APIs should not import
these directly yet; they are intended for internal use and tests.

Scope: minimal parity with current legacy fields used in tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from . import models


def legacy_to_domain(idx: int, payload: Mapping[str, Any]) -> models.Explanation:
    """Convert a legacy explanation payload to the domain model.

    Parameters
    ----------
    idx : int
        Index of the instance the explanation corresponds to.
    payload : Mapping[str, Any]
        Legacy-shaped explanation payload (parallel arrays under keys such as
        "rules", "feature_weights", "feature_predict", and a top-level
        "prediction" dict).

    Returns
    -------
    Explanation
        Domain model representation of the explanation.
    """
    domain = models.from_legacy_dict(idx, payload)
    provenance = payload.get("provenance")
    metadata = payload.get("metadata")
    if isinstance(provenance, Mapping):
        domain.provenance = dict(provenance)
    else:
        domain.provenance = None
    if isinstance(metadata, Mapping):
        domain.metadata = dict(metadata)
    else:
        domain.metadata = None
    return domain


def domain_to_legacy(exp: models.Explanation) -> Dict[str, Any]:
    """Convert a domain model Explanation into a legacy-shaped dict.

    The resulting structure mirrors the minimal subset used by tests and
    existing code paths:

    - top-level: "task", "prediction"
    - parallel arrays: "rules": {"rule": [...], "feature": [...]}
    - parallel arrays for weights/predictions: "feature_weights", "feature_predict"

    Notes
    -----
    - If no rules exist, arrays are empty lists to keep shapes consistent.
    - Only fields represented in the domain model will be populated; additional
      legacy fields not modeled today are intentionally omitted.
    """
    out: Dict[str, Any] = {
        "task": exp.task,
        "prediction": dict(exp.prediction),
        "provenance": dict(exp.provenance) if isinstance(exp.provenance, Mapping) else None,
        "metadata": dict(exp.metadata) if isinstance(exp.metadata, Mapping) else None,
    }

    # Prepare parallel arrays for rules
    rule_texts: List[str] = []
    rule_features: List[int] = []

    # Aggregate per-rule mappings into parallel arrays keyed by metric name
    weights_acc: Dict[str, List[Any]] = {}
    predicts_acc: Dict[str, List[Any]] = {}

    for fr in exp.rules:
        rule_texts.append(fr.rule)
        rule_features.append(fr.feature)

        for k, v in fr.rule_weight.items():
            weights_acc.setdefault(k, []).append(v)
        for k, v in fr.rule_prediction.items():
            predicts_acc.setdefault(k, []).append(v)

    out["rules"] = {"rule": rule_texts, "feature": rule_features}
    out["feature_weights"] = weights_acc
    out["feature_predict"] = predicts_acc

    return out


__all__ = ["legacy_to_domain", "domain_to_legacy"]
