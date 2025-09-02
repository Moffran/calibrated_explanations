"""Phase 1B validation module.

Provides input validation helpers for calibrated_explanations. Raises clear
errors early for invalid inputs. Includes a thin wrapper to normalize keyword
arguments to canonical names to reduce alias drift.
"""
from typing import Any, Sequence, Dict, List
import numpy as np
from calibrated_explanations.core.exceptions import ValidationError, DataShapeError
from .param_aliases import canonicalize_params
from ..utils.helper import safe_isinstance


def validate_not_none(value: Any, name: str) -> None:
    if value is None:
        raise ValidationError(f"Argument '{name}' must not be None.")


def validate_type(value: Any, expected_type: type, name: str) -> None:
    if not isinstance(value, expected_type):
        raise DataShapeError(f"Argument '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}.")


def validate_non_empty(value: Any, name: str) -> None:
    if hasattr(value, "__len__") and len(value) == 0:
        raise ValidationError(f"Argument '{name}' must not be empty.")


def validate_inputs(*args: Any, **kwargs: Any) -> None:
    """Basic input validation: checks for None and empty sequences."""
    for idx, arg in enumerate(args):
        validate_not_none(arg, f"arg{idx}")
        if isinstance(arg, (str, Sequence)) and not isinstance(arg, (bytes, bytearray)):
            validate_non_empty(arg, f"arg{idx}")
    for key, value in kwargs.items():
        validate_not_none(value, key)
        if isinstance(value, (str, Sequence)) and not isinstance(value, (bytes, bytearray)):
            validate_non_empty(value, key)

def normalize_kwargs(kwargs: Dict[str, Any], *, raise_on_conflict: bool = False) -> Dict[str, Any]:
    """Return kwargs with parameter names canonicalized.

    This is a convenience wrapper over :func:`canonicalize_params` to be used at
    validation boundaries before delegating to core routines.
    """
    return canonicalize_params(dict(kwargs), raise_on_conflict=raise_on_conflict)


__all__ = [
    "validate_inputs",
    "validate_not_none",
    "validate_type",
    "validate_non_empty",
    "normalize_kwargs",
]


# --------- DataFrame / numeric validation (Phase 1B pave for Phase 2) ---------
def _non_numeric_df_columns(df) -> List[str]:
    """Return a list of non-numeric column names in a pandas DataFrame.

    Avoids importing pandas at module import time; only uses it when needed.
    """
    try:
        import pandas as pd  # type: ignore
        from pandas.api.types import is_numeric_dtype  # type: ignore
    except Exception as exc:  # pragma: no cover - pandas is a dependency, defensive only
        raise ValidationError(
            "Pandas is required to validate DataFrame inputs but is not available."
        ) from exc

    if not isinstance(df, pd.DataFrame):
        return []
    non_numeric = [
        str(col) for col in df.columns if not is_numeric_dtype(df[col].dtype)
    ]
    return non_numeric


def validate_feature_matrix(X: Any, name: str = "X") -> None:
    """Validate that feature matrix is numeric or signal actionable guidance.

    - If a pandas DataFrame is provided and contains non-numeric columns, raise ValidationError
      listing offending columns and suggest using `utils.transform_to_numeric` or passing
      `categorical_features`/`categorical_labels` to the core APIs. Phase 2 will add
      native preprocessing per ADR-009.
    - If a numpy array has non-numeric dtype, raise DataShapeError with guidance.
    - Dictionary-like row inputs (list/array of dict) are allowed and not validated here.
    """
    validate_not_none(X, name)

    # Pandas DataFrame: detect non-numeric columns early
    if safe_isinstance(X, "pandas.core.frame.DataFrame"):
        bad = _non_numeric_df_columns(X)
        if bad:
            cols = ", ".join(bad[:10]) + (" ..." if len(bad) > 10 else "")
            raise ValidationError(
                (
                    f"{name} contains non-numeric columns: {cols}. "
                    "Preprocess to numeric (e.g., utils.transform_to_numeric) or pass "
                    "categorical_features/categorical_labels to CalibratedExplainer. "
                    "Phase 2 will add native preprocessing (see ADR-009)."
                )
            )
        return

    # Numpy arrays should be numeric
    if hasattr(X, "dtype") and hasattr(X, "shape"):
        dtype = getattr(X, "dtype", None)
        if dtype is not None and not np.issubdtype(dtype, np.number):
            raise DataShapeError(
                (
                    f"{name} has non-numeric dtype '{dtype}'. Convert to numeric types "
                    "before calling fit/calibrate/explain."
                )
            )

    # Allow list/iterable inputs; deeper checks happen downstream
    return


__all__.extend(["validate_feature_matrix"])
