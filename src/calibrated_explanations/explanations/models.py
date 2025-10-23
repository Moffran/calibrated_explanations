"""Internal domain models for explanations (Phase 2 groundwork).

These dataclasses model Explanation and FeatureRule for clearer internal
reasoning. Public APIs continue to emit legacy dicts; adapters ensure
parity in tests. Do not import these in public signatures yet.

See ADR-008 for rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class FeatureRule:
    """Container describing a single feature rule within an explanation.

    The ``feature`` field captures either a single feature index or a sequence of
    indices to represent conjunctive rules emitted by legacy explainers.
    """

    feature: Any
    rule: str
    weight: Mapping[str, Any]
    prediction: Mapping[str, Any]
    instance_prediction: Mapping[str, Any] | None = None
    feature_value: Any | None = None
    is_conjunctive: bool = False
    value_str: str | None = None
    bin_index: int | None = None


@dataclass
class Explanation:
    """Domain-model representation of a calibrated explanation instance."""

    task: str
    index: int
    prediction: Mapping[str, Any]
    rules: Sequence[FeatureRule]
    provenance: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None


def from_legacy_dict(idx: int, payload: Mapping[str, Any]) -> Explanation:
    """Build an Explanation from the legacy dict shape (adapter).

    This function is intentionally minimal; extend as needed once specific
    call sites are wired. Not used in public API yet.
    """
    # Expect keys similar to the ones used in explanation.py internals
    prediction = payload.get("prediction") or {}
    rules_out: list[FeatureRule] = []
    rules_block = payload.get("rules") or {}
    # Heuristic assembly if vectors are parallel arrays; keep minimal now.
    if isinstance(rules_block, dict) and "rule" in rules_block:
        n = len(rules_block["rule"])  # type: ignore[index]
        for i in range(n):
            # safe feature index -- may be scalar or a list (conjunctive rules)
            features_list = rules_block.get("feature", [])
            raw_feat = features_list[i] if i < len(features_list) else i
            if isinstance(raw_feat, (list, tuple, np.ndarray)):
                feat = list(raw_feat)
            else:
                feat = int(raw_feat)

            # Build weight/prediction dicts with safe indexing: if vectors are shorter
            # than the rules array, fall back to the last available value or None.
            weights_src = payload.get("feature_weights") or {}
            predicts_src = payload.get("feature_predict") or {}

            def _safe_pick(arr, idx):
                try:
                    return arr[idx]
                except Exception:
                    if len(arr) > 0:
                        return arr[-1]
                    return None

            weight_map = {k: _safe_pick(v, i) for k, v in weights_src.items()}
            predict_map = {k: _safe_pick(v, i) for k, v in predicts_src.items()}

            # safe picks from rules_block for optional fields
            def _rb_pick(key, idx):
                arr = rules_block.get(key, [])
                try:
                    return arr[idx]
                except Exception:
                    return arr[-1] if len(arr) > 0 else None

            is_conj = bool(_rb_pick("is_conjunctive", i) or isinstance(feat, list))
            feature_value = _rb_pick("feature_value", i)
            value_str = _rb_pick("value", i)

            fr = FeatureRule(
                feature=feat,
                rule=rules_block["rule"][i],  # type: ignore[index]
                weight=weight_map,
                prediction=predict_map,
                instance_prediction=None,
                feature_value=feature_value,
                is_conjunctive=is_conj,
                value_str=value_str,
            )
            rules_out.append(fr)
    return Explanation(
        task=str(payload.get("task", "unknown")), index=idx, prediction=prediction, rules=rules_out
    )


__all__ = ["Explanation", "FeatureRule", "from_legacy_dict"]
