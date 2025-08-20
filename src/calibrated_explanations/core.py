"""Calibrated Explanations for Black-Box Predictions (calibrated-explanations).

The calibrated explanations explanation method is based on the paper 
"Calibrated Explanations: with Uncertainty Information and Counterfactuals" 
by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

Calibrated explanations are a way to explain the predictions of a black-box learner 
using Venn-Abers predictors (classification & regression) or 
conformal predictive systems (regression).
"""
# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-lines, too-many-positional-arguments, too-many-public-methods
import warnings as _warnings

# Emit deprecation notice for upcoming core split (Phase 1A)
_warnings.warn(
    "calibrated_explanations.core is scheduled for mechanical split in v0.6.0; "
    "import paths will remain stable until v0.8.0. This warning is informational.",
    DeprecationWarning,
    stacklevel=2,
)
