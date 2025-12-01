"""Shared utilities used across calibrated explanations.

This module re-exports the public-facing utilities so callers can import
directly from ``calibrated_explanations.utils`` rather than reaching into
individual helper modules.
"""

from .deprecations import _EMITTED, _EMITTED_PER_TEST, _should_raise, deprecate, deprecate_alias
from .discretizers import (
    BinaryEntropyDiscretizer,
    BinaryRegressorDiscretizer,
    EntropyDiscretizer,
    RegressorDiscretizer,
)
from .helper import (
    assert_threshold,
    calculate_metrics,
    check_is_fitted,
    concatenate_thresholds,
    convert_targets_to_numeric,
    immutable_array,
    make_directory,
    is_notebook,
    prepare_for_saving,
    safe_first_element,
    safe_import,
    safe_isinstance,
    safe_mean,
    transform_to_numeric,
)
from .perturbation import (
    categorical_perturbation,
    gaussian_perturbation,
    perturb_dataset,
    uniform_perturbation,
)
from .rng import set_rng_seed

__all__ = [
    "assert_threshold",
    "BinaryEntropyDiscretizer",
    "BinaryRegressorDiscretizer",
    "calculate_metrics",
    "categorical_perturbation",
    "_EMITTED",
    "_EMITTED_PER_TEST",
    "_should_raise",
    "check_is_fitted",
    "concatenate_thresholds",
    "convert_targets_to_numeric",
    "immutable_array",
    "deprecate",
    "deprecate_alias",
    "EntropyDiscretizer",
    "gaussian_perturbation",
    "make_directory",
    "is_notebook",
    "perturb_dataset",
    "prepare_for_saving",
    "RegressorDiscretizer",
    "safe_first_element",
    "safe_import",
    "safe_isinstance",
    "safe_mean",
    "set_rng_seed",
    "transform_to_numeric",
    "uniform_perturbation",
]
