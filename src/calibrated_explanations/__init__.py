# flake8: noqa: E501
"""
Calibrated Explanations (calibrated_explanations)

is a Python package for explaining black-box models.

It is based on the paper "Calibrated Explanations: with Uncertainty Information and Counterfactuals" 
by Helena Löfström et al.
"""
from .core import CalibratedExplainer, __version__
from .core_components import WrapCalibratedExplainer, OnlineCalibratedExplainer
from .utils.discretizers import BinaryEntropyDiscretizer, EntropyDiscretizer, \
                    RegressorDiscretizer, BinaryRegressorDiscretizer
from .explanations.explanations import CalibratedExplanations, AlternativeExplanations
from .explanations.explanation import FactualExplanation, \
                    AlternativeExplanation, FastExplanation
from ._VennAbers import VennAbers
from ._interval_regressor import IntervalRegressor
from .utils.helper import transform_to_numeric


__all__ = [
    'CalibratedExplainer',
    'WrapCalibratedExplainer', 
    'OnlineCalibratedExplainer',
    'transform_to_numeric'
]
