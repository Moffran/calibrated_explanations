"""Internal domain models for explanations (Phase 2 groundwork).

These dataclasses model Explanation and FeatureRule for clearer internal
reasoning. Public APIs continue to emit legacy dicts; adapters ensure
parity in tests. Do not import these in public signatures yet.

See ADR-008 for rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass
class FeatureRule:
    feature: int
    rule: str
    weight: Mapping[str, Any]
    prediction: Mapping[str, Any]
    instance_prediction: Mapping[str, Any] | None = None
    feature_value: Any | None = None
    is_conjunctive: bool = False
    value_str: str | None = None
    current_bin: int | None = None


@dataclass
class Explanation:
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
            fr = FeatureRule(
                feature=int(rules_block.get("feature", [i])[i]),  # type: ignore[index]
                rule=rules_block["rule"][i],  # type: ignore[index]
                weight={k: v[i] for k, v in (payload.get("feature_weights") or {}).items()},
                prediction={k: v[i] for k, v in (payload.get("feature_predict") or {}).items()},
                instance_prediction=None,
            )
            rules_out.append(fr)
    return Explanation(
        task=str(payload.get("task", "unknown")), index=idx, prediction=prediction, rules=rules_out
    )


__all__ = ["Explanation", "FeatureRule", "from_legacy_dict"]
