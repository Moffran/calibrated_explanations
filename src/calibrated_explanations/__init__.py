"""
Calibrated Explanations (calibrated_explanations).

is a Python package for explaining black-box models.

It is based on the paper "Calibrated Explanations: with Uncertainty Information and Counterfactuals"
by Helena Löfström et al.
"""

import logging as _logging

# Expose viz namespace (internal; subject to change). Avoid importing heavy backends eagerly.
from . import viz  # noqa: F401
from .explanations.explanation import (
    AlternativeExplanation,  # noqa: F401
    FactualExplanation,  # noqa: F401
    FastExplanation,  # noqa: F401
)
from .explanations.explanations import AlternativeExplanations, CalibratedExplanations  # noqa: F401
from .utils.discretizers import (
    BinaryEntropyDiscretizer,  # noqa: F401
    BinaryRegressorDiscretizer,  # noqa: F401
    EntropyDiscretizer,  # noqa: F401
    RegressorDiscretizer,  # noqa: F401
)
from .utils.helper import transform_to_numeric

# Provide a default no-op handler to avoid "No handler" warnings for library users.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__version__ = "v0.9.0"

# Note: core submodules are intentionally not imported here to avoid importing
# large backends and to make deprecation transitions explicit. We still expose
# the public symbols lazily so `from calibrated_explanations import CalibratedExplainer`
# works without triggering an eager import of `calibrated_explanations.core`.
__all__ = [
    "CalibratedExplainer",
    "WrapCalibratedExplainer",
    "transform_to_numeric",
]


def __getattr__(name: str):
    """Lazy import for a small set of public symbols.

    This avoids importing `calibrated_explanations.core` at package import time
    (which would trigger deprecation emissions) while preserving the public API
    surface for users and tests.
    """
    if name == "viz":
        module = importlib.import_module(f"{__name__}.viz")
        globals()[name] = module
        return module
    if name in {
        "BinaryEntropyDiscretizer",
        "BinaryRegressorDiscretizer",
        "EntropyDiscretizer",
        "RegressorDiscretizer",
    }:
        module = importlib.import_module(f"{__name__}.utils.discretizers")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in {
        "AlternativeExplanation",
        "FactualExplanation",
        "FastExplanation",
    }:
        module = importlib.import_module(f"{__name__}.explanations.explanation")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in {
        "AlternativeExplanations",
        "CalibratedExplanations",
    }:
        module = importlib.import_module(f"{__name__}.explanations.explanations")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "CalibratedExplainer":
        from .core.calibrated_explainer import CalibratedExplainer

        globals()[name] = CalibratedExplainer
        return CalibratedExplainer
    if name == "WrapCalibratedExplainer":
        from .core.wrap_explainer import WrapCalibratedExplainer

        globals()[name] = WrapCalibratedExplainer
        return WrapCalibratedExplainer
    if name == "IntervalRegressor":
        from .core.interval_regressor import IntervalRegressor

        globals()[name] = IntervalRegressor
        return IntervalRegressor
    if name == "VennAbers":
        from .core.venn_abers import VennAbers

        globals()[name] = VennAbers
        return VennAbers
    if name == "transform_to_numeric":
        module = importlib.import_module(f"{__name__}.utils.helper")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

