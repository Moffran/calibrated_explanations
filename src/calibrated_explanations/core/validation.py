"""Phase 1B validation module.

Provides input validation helpers for calibrated_explanations. Raises clear errors early for invalid inputs.
"""
from typing import Any, Sequence
from calibrated_explanations.core.exceptions import ValidationError, DataShapeError


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

__all__ = ["validate_inputs", "validate_not_none", "validate_type", "validate_non_empty"]
