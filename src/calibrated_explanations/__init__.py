# flake8: noqa: E501
"""
Calibrated Explanations (calibrated_explanations)

is a Python package for explaining black-box models.

It is based on the paper "Calibrated Explanations: with Uncertainty Information and Counterfactuals" 
by Helena Löfström et al.
"""
from .core import CalibratedExplainer, WrapCalibratedExplainer, __version__
from .wrappers import CalibratedAsLimeTabularExplainer
from ._discretizers import BinaryDiscretizer, BinaryEntropyDiscretizer, \
                    EntropyDiscretizer, QuartileDiscretizer, DecileDiscretizer
from ._explanations import CalibratedExplanations, CalibratedExplanation, \
                    FactualExplanation, CounterfactualExplanation
from .VennAbers import VennAbers
from ._interval_regressor import IntervalRegressor
