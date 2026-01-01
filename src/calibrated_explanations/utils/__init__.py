"""Shared utilities used across calibrated explanations.

This module re-exports the public-facing utilities so callers can import
directly from ``calibrated_explanations.utils`` rather than reaching into
individual helper modules.
"""

from .deprecation import deprecate_public_api_symbol
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
    is_notebook,
    make_directory,
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


def _ensure_joblib_pool_attribute():
    """Work around joblib ThreadingBackend expecting a missing `pool` attribute."""
    try:
        import joblib._parallel_backends as _joblib_backends
    except Exception:
        return

    if hasattr(_joblib_backends.PoolManagerMixin, "pool"):
        return

    def _get_pool(self):
        return getattr(self, "_pool", None)

    def _set_pool(self, value):
        self._pool = value

    _joblib_backends.PoolManagerMixin.pool = property(_get_pool, _set_pool)


_ensure_joblib_pool_attribute()

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
    "deprecate_public_api_symbol",
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
