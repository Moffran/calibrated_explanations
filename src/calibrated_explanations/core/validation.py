"""Validation helpers shared across the core package.

These utilities centralize defensive argument checks while preserving
existing behavior, ensuring future refactors can rely on a consistent
error vocabulary.

ADR-002 compliance: Validation functions use the exception taxonomy
(ValidationError, DataShapeError, NotFittedError, ConfigurationError, etc.)
and accept optional details payloads.
"""

from __future__ import annotations

from typing import Any, Literal, Type, cast

import numpy as np
import numpy.typing as npt

from .exceptions import (
    CalibratedError,
    DataShapeError,
    ModelNotSupportedError,
    NotFittedError,
    ValidationError,
)


def validate_not_none(value: Any, name: str) -> None:
    """Raise ``ValidationError`` when ``value`` is ``None``."""
    if value is None:
        raise ValidationError(f"Argument '{name}' must not be None.")


def validate_type(value: Any, expected_type: type, name: str) -> None:
    """Ensure that ``value`` is an instance of ``expected_type``."""
    if not isinstance(value, expected_type):
        raise DataShapeError(
            f"Argument '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}."
        )


def validate_non_empty(value: Any, name: str) -> None:
    """Ensure that length-aware inputs are not empty."""
    if hasattr(value, "__len__") and len(value) == 0:
        raise ValidationError(f"Argument '{name}' must not be empty.")


def validate_inputs(
    x: Any,
    y: Any | None = None,
    task: Literal["auto", "classification", "regression"] = "auto",
    allow_nan: bool = False,
    require_y: bool = False,
    n_features: int | None = None,
    class_labels: Any | None = None,
    check_finite: bool = True,
) -> None:
    """Validate input features and target for downstream operations.

    This function provides the primary validation entry point per ADR-002,
    accepting feature matrix x and optional target y with comprehensive
    shape, dtype, and value checks.

    Parameters
    ----------
    x : array-like
        Feature matrix of shape (n_samples, n_features). Can be a NumPy array,
        pandas DataFrame, or similar.
    y : array-like, optional
        Target vector of shape (n_samples,). If provided, length must match x.
        Default is None.
    task : {"auto", "classification", "regression"}, default="auto"
        Task type. When "auto", inferred from model capabilities or y dtype.
    allow_nan : bool, default=False
        If False, raises ValidationError when x or y contain NaN values.
    require_y : bool, default=False
        If True, raises ValidationError when y is None.
    n_features : int, optional
        Expected number of features in x. If provided and mismatch occurs,
        raises DataShapeError.
    class_labels : array-like, optional
        Class labels for classification tasks. Stored for later use.
    check_finite : bool, default=True
        If True, checks that x and y contain only finite values (except NaN
        when allow_nan=True).

    Raises
    ------
    ValidationError
        When y is required but None, or when values contain NaN/inf unexpectedly.
    DataShapeError
        When x is not 2D, feature count mismatches, or y length mismatches.

    Examples
    --------
    >>> from calibrated_explanations.core.validation import validate_inputs
    >>> import numpy as np
    >>> x = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> y = np.array([0, 1])
    >>> validate_inputs(x, y, task="classification", require_y=True, n_features=2)
    # Passes silently if valid
    """
    validate_not_none(x, "x")
    x_arr = _as_2d_array(x)
    if x_arr.ndim != 2:
        raise DataShapeError(
            "Argument 'x' must be 2D (n_samples, n_features).",
            details={"param": "x", "ndim": x_arr.ndim, "expected": 2},
        )
    n_samples = x_arr.shape[0]
    if n_features is not None and x_arr.shape[1] != n_features:
        raise DataShapeError(
            f"Argument 'x' must have {n_features} features, got {x_arr.shape[1]}.",
            details={
                "param": "x",
                "expected_features": n_features,
                "actual_features": x_arr.shape[1],
            },
        )

    if require_y and y is None:
        raise ValidationError(
            "Argument 'y' must be provided when require_y=True.",
            details={"param": "y", "requirement": "required", "task": task},
        )

    if y is not None:
        y_arr = _as_1d_array(y)
        if y_arr.shape[0] != n_samples:
            raise DataShapeError(
                f"Length of 'y' ({y_arr.shape[0]}) does not match number of samples in x ({n_samples}).",
                details={
                    "param": "y",
                    "y_length": y_arr.shape[0],
                    "x_samples": n_samples,
                },
            )
        if (
            check_finite
            and not allow_nan
            and np.issubdtype(y_arr.dtype, np.number)
            and not np.isfinite(y_arr).all()
        ):
            raise ValidationError(
                "Argument 'y' contains NaN or infinite values.",
                details={"param": "y", "check": "finitude", "allow_nan": allow_nan},
            )

    if check_finite and not allow_nan and not np.isfinite(x_arr).all():
        raise ValidationError(
            "Argument 'x' contains NaN or infinite values.",
            details={"param": "x", "check": "finitude", "allow_nan": allow_nan},
        )

    # Store class_labels if provided (for metadata tracking)
    if class_labels is not None:
        validate_not_none(class_labels, "class_labels")

    # Reserve task inference for future behavior without changing runtime output.
    _ = infer_task(x, y, None) if task == "auto" else task


def infer_task(
    x: Any = None, y: Any = None, model: Any = None
) -> Literal["classification", "regression"]:
    """Infer the task type using model capabilities or target dtype.

    Priority is given to model capabilities (``predict_proba`` implies
    classification). When a model is unavailable, heuristics based on the
    target dtype are used. Regression is the safe fallback.
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
    """Return ``x`` coerced to a 2D ``ndarray``."""
    if hasattr(x, "values") and hasattr(x, "shape"):
        try:
            return cast(npt.NDArray[np.generic], np.asarray(x.values))
        except Exception:  # pragma: no cover - fallback
            return cast(npt.NDArray[np.generic], np.asarray(x))
    return cast(npt.NDArray[np.generic], np.asarray(x))


def _as_1d_array(y: Any) -> npt.NDArray[np.generic]:
    """Return ``y`` coerced to a flattened 1D ``ndarray``."""
    if hasattr(y, "values") and not isinstance(y, np.ndarray):
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
    """Validate a feature/target matrix pair for downstream operations.

    - Ensure ``x`` is 2D and matches the expected feature count when provided.
    - Confirm that ``y`` has the same number of samples when supplied.
    - Guard against NaN or infinite values unless explicitly allowed.
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
        if (
            check_finite
            and not allow_nan
            and np.issubdtype(y_arr.dtype, np.number)
            and not np.isfinite(y_arr).all()
        ):
            raise ValidationError("Argument 'y' contains NaN or infinite values.")

    if check_finite and not allow_nan and not np.isfinite(x_arr).all():
        raise ValidationError("Argument 'x' contains NaN or infinite values.")

    # Reserve task inference for future behavior without changing runtime output.
    _ = infer_task(x, y, None) if task == "auto" else task


def validate_model(model: Any) -> None:
    """Validate minimal model protocol requirements."""
    validate_not_none(model, "model")
    if not hasattr(model, "predict"):
        raise ModelNotSupportedError("Model must implement a 'predict' method.")


def validate_fit_state(obj: Any, *, require: bool = True) -> None:
    """Validate fit state flags before executing stateful operations."""
    if not require:
        return
    if hasattr(obj, "fitted") and not obj.fitted:
        raise NotFittedError("Operation requires a fitted estimator/explainer.")


def validate(
    condition: bool,
    exc_cls: Type[CalibratedError],
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> None:
    """Conditional validation helper for common patterns.

    Raises an exception when a condition is False, enabling concise guard clauses.

    Parameters
    ----------
    condition : bool
        Condition to check. If False, raises exc_cls.
    exc_cls : Type[CalibratedError]
        Exception class to raise when condition is False.
    message : str
        Error message.
    details : dict, optional
        Structured error details to attach to the exception.

    Raises
    ------
    exc_cls
        If condition is False.

    Example
    -------
    >>> from calibrated_explanations.core.validation import validate
    >>> from calibrated_explanations.core.exceptions import ValidationError
    >>> validate(len(x) > 0, ValidationError, "x must not be empty", details={"param": "x"})
    """
    if not condition:
        raise exc_cls(message, details=details)


__all__ = [
    "validate_inputs",
    "validate_not_none",
    "validate_type",
    "validate_non_empty",
    "validate_inputs_matrix",
    "validate_model",
    "validate_fit_state",
    "infer_task",
    "validate",
]
