# flake8: noqa: E501
"""
Calibrated Explanations (calibrated_explanations)

is a Python package for explaining black-box models.

It is based on the paper "Calibrated Explanations: with Uncertainty Information and Counterfactuals" 
by Helena Löfström et al.
"""
from .core import CalibratedExplainer, WrapCalibratedExplainer, __version__
from .utils.discretizers import BinaryEntropyDiscretizer, EntropyDiscretizer, \
                    RegressorDiscretizer, BinaryRegressorDiscretizer
from .explanations import CalibratedExplanations, CalibratedExplanation, \
                    FactualExplanation, AlternativeExplanation, PerturbedExplanation
from ._VennAbers import VennAbers
from ._interval_regressor import IntervalRegressor
