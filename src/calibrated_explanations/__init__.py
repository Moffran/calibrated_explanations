"""
Calibrated Explanations (calibrated_explanations)

is a Python package for explaining black-box models.

It is based on the paper "Calibrated Explanations: with Uncertainty Information and Counterfactuals"
by Helena Löfström et al.
"""

import logging as _logging

from ._interval_regressor import IntervalRegressor  # noqa: F401
from ._VennAbers import VennAbers  # noqa: F401
from .core import CalibratedExplainer, OnlineCalibratedExplainer, WrapCalibratedExplainer
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

__version__ = "v0.5.1"

__all__ = [
    "CalibratedExplainer",
    "WrapCalibratedExplainer",
    "OnlineCalibratedExplainer",
    "transform_to_numeric",
]
