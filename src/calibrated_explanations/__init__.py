# flake8: noqa: E501
"""
Calibrated Explanations (calibrated_explanations)

Python package for explaining black-box models with calibrated uncertainty.
Based on the paper "Calibrated Explanations: with Uncertainty Information and Counterfactuals" 
by Helena Löfström et al.
"""

# NOTE: Avoid importing the deprecated shim `core.py` just to obtain a version string, since
# that would emit a DeprecationWarning on every package import. Maintain the version here
# (kept in sync with pyproject.toml) until a dedicated version module is introduced.
__version__ = "0.5.1"

from .core import (
    CalibratedExplainer,
    WrapCalibratedExplainer,
    OnlineCalibratedExplainer,
)
from .utils.discretizers import BinaryEntropyDiscretizer, EntropyDiscretizer, \
                    RegressorDiscretizer, BinaryRegressorDiscretizer
from .explanations.explanations import CalibratedExplanations, AlternativeExplanations
from .explanations.explanation import FactualExplanation, \
                    AlternativeExplanation, FastExplanation
from ._VennAbers import VennAbers
from ._interval_regressor import IntervalRegressor
from .utils.helper import transform_to_numeric


__all__ = [
    "CalibratedExplainer",
    "WrapCalibratedExplainer",
    "OnlineCalibratedExplainer",
    "transform_to_numeric",
]
