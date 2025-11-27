# ruff: noqa: N999
# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, fixme
"""Venn-Abers calibration utilities for post-processing model probabilities.

DEPRECATED: This module has been moved to calibrated_explanations.core.calibration.venn_abers.

This module is maintained for backward compatibility. All imports should be updated to use:
    from calibrated_explanations.core.calibration.venn_abers import VennAbers

Part of Phase 6: Refactor Calibration Functionality (ADR-001).
"""

import warnings

import venn_abers as va  # Re-export for test monkeypatching

# Re-export from new location for backward compatibility
# Note: We use filterwarnings to allow pytest to properly configure warning handling
with warnings.catch_warnings():
    warnings.filterwarnings("default", category=DeprecationWarning)
    from .calibration.venn_abers import VennAbers, exponent_scaling_list

__all__ = ["VennAbers", "exponent_scaling_list", "va"]
