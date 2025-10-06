"""Phase 1B validation module.

Provides input validation helpers for calibrated_explanations. Raises clear errors early for invalid inputs.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, cast

import numpy as np
import numpy.typing as npt

from calibrated_explanations.core.exceptions import (
    DataShapeError,
    ModelNotSupportedError,
    NotFittedError,
    ValidationError,
)


def validate_not_none(value: Any, name: str) -> None:
    if value is None:
        raise ValidationError(f"Argument '{name}' must not be None.")


def validate_type(value: Any, expected_type: type, name: str) -> None:
    if not isinstance(value, expected_type):
        raise DataShapeError(
            f"Argument '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}."
        )


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


def infer_task(
    x: Any = None, y: Any = None, model: Any = None
) -> Literal["classification", "regression"]:
    """Infer task type using model capabilities or y dtype.

    Priority: model.predict_proba -> classification; else regression.
    If model is None, use y: non-integer float -> regression, binary/int -> classification (best-effort).
    Fallback to regression.
    """
    if model is not None:
        if hasattr(model, "predict_proba"):
            return "classification"
        return "regression"
    if y is not None:
        y_arr = _as_1d_array(y)
        if np.issubdtype(y_arr.dtype, np.floating):
            return "regression"
        return "classification"
    return "regression"


def _as_2d_array(x: Any) -> npt.NDArray[np.generic]:
    # Accept numpy arrays and pandas DataFrames
    if hasattr(x, "values") and hasattr(x, "shape"):
        try:
            return cast(npt.NDArray[np.generic], np.asarray(x.values))
        except Exception:  # pragma: no cover - fallback
            return cast(npt.NDArray[np.generic], np.asarray(x))
    return cast(npt.NDArray[np.generic], np.asarray(x))


def _as_1d_array(y: Any) -> npt.NDArray[np.generic]:
    if hasattr(y, "values") and not isinstance(y, (np.ndarray,)):
        y = y.values
    arr = cast(npt.NDArray[np.generic], np.asarray(y))
    return cast(npt.NDArray[np.generic], arr.reshape(-1))


def validate_inputs_matrix(
    x: Any,
    y: Any | None = None,
    *,
    task: Literal["auto", "classification", "regression"] = "auto",
    allow_nan: bool = False,
    require_y: bool = False,
    n_features: int | None = None,
    check_finite: bool = True,
) -> None:
    """Validate typical (x, y) input pairs.

    - Ensures x is 2D and y length matches x rows when provided/required.
    - Validates finiteness according to allow_nan/check_finite.
    - Enforces feature count when n_features is provided.
    """
    validate_not_none(x, "x")
    x_arr = _as_2d_array(x)
    if x_arr.ndim != 2:
        raise DataShapeError("Argument 'x' must be 2D (n_samples, n_features).")
    n_samples = x_arr.shape[0]
    if n_features is not None and x_arr.shape[1] != n_features:
        raise DataShapeError(f"Argument 'x' must have {n_features} features, got {x_arr.shape[1]}.")

    if require_y and y is None:
        raise ValidationError("Argument 'y' must be provided when require_y=True.")
    if y is not None:
        y_arr = _as_1d_array(y)
        if y_arr.shape[0] != n_samples:
            raise DataShapeError(
                f"Length of 'y' ({y_arr.shape[0]}) does not match number of samples in x ({n_samples})."
            )
        # Only perform finiteness checks on numeric dtypes to avoid TypeError for object/string labels
        if (
            check_finite
            and not allow_nan
            and np.issubdtype(y_arr.dtype, np.number)
            and not np.isfinite(y_arr).all()
        ):
            raise ValidationError("Argument 'y' contains NaN or infinite values.")

    if check_finite and not allow_nan and not np.isfinite(x_arr).all():
        raise ValidationError("Argument 'x' contains NaN or infinite values.")

    # For now, task inference is unused here, but reserved for future checks
    _ = infer_task(x, y, None) if task == "auto" else task


def validate_model(model: Any) -> None:
    """Validate model protocol minimally.

    - Must have predict.
    - If supports classification (predict_proba present), that's fine; otherwise
      regression is assumed.
    """
    validate_not_none(model, "model")
    if not hasattr(model, "predict"):
        raise ModelNotSupportedError("Model must implement a 'predict' method.")
    # No further checks here to avoid behavior changes; predict_proba is checked at call sites.


def validate_fit_state(obj: Any, *, require: bool = True) -> None:
    """Validate that an object with fitted/calibrated flags is in the right state.

    - If require is True: ensure obj.fitted is True; if obj has calibrated for
      operations requiring it, callers should check that separately.
    """
    if not require:
        return
    if hasattr(obj, "fitted") and not obj.fitted:
        raise NotFittedError("Operation requires a fitted estimator/explainer.")


__all__ = [
    "validate_inputs",
    "validate_not_none",
    "validate_type",
    "validate_non_empty",
    "validate_inputs_matrix",
    "validate_model",
    "validate_fit_state",
    "infer_task",
]
