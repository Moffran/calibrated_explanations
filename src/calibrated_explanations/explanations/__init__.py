"""Public façade for explanation domain objects (ADR-001 Stage 5).

All sanctioned explanation dataclasses and factories are re-exported from the
package root to discourage deep imports (for example ``explanations.models``).
Callers should treat this module as the stable boundary; plotting helpers and
other experimental utilities remain internal.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .adapters import domain_to_legacy, legacy_to_domain
    from .explanation import (
        AlternativeExplanation,
        CalibratedExplanation,
        FactualExplanation,
        FastExplanation,
    )
    from .explanations import (
        AlternativeExplanations,
        CalibratedExplanations,
        FrozenCalibratedExplainer,
    )
    from .models import Explanation, FeatureRule, from_legacy_dict


__all__ = (
    "CalibratedExplanations",
    "AlternativeExplanations",
    "FrozenCalibratedExplainer",
    "CalibratedExplanation",
    "FactualExplanation",
    "AlternativeExplanation",
    "FastExplanation",
    "Explanation",
    "FeatureRule",
    "from_legacy_dict",
    "legacy_to_domain",
    "domain_to_legacy",
)

_NAME_TO_MODULE = {
    "CalibratedExplanations": ("explanations", "CalibratedExplanations"),
    "AlternativeExplanations": ("explanations", "AlternativeExplanations"),
    "FrozenCalibratedExplainer": ("explanations", "FrozenCalibratedExplainer"),
    "CalibratedExplanation": ("explanation", "CalibratedExplanation"),
    "FactualExplanation": ("explanation", "FactualExplanation"),
    "AlternativeExplanation": ("explanation", "AlternativeExplanation"),
    "FastExplanation": ("explanation", "FastExplanation"),
    "Explanation": ("models", "Explanation"),
    "FeatureRule": ("models", "FeatureRule"),
    "from_legacy_dict": ("models", "from_legacy_dict"),
    "legacy_to_domain": ("adapters", "legacy_to_domain"),
    "domain_to_legacy": ("adapters", "domain_to_legacy"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose the sanctioned explanation API surface."""

    if name not in __all__:
        raise AttributeError(name)

    module_name, attr_name = _NAME_TO_MODULE[name]
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
