"""General utility helpers for the test suite."""

import os


def get_env_flag(name: str) -> bool:
    """Return a boolean for environment flags treating common truthy strings as True.

    This treats '1', 'true', 'yes', 'y' (case-insensitive) as True. Everything
    else (including '0', 'false', empty string or unset) is False. Using this
    avoids Python's truthiness where a non-empty string like '0' evaluates to True.
    """
    val = os.getenv(name, "").strip().lower()
    return val in ("1", "true", "yes", "y")


def assert_predictions_match(y_pred1, y_pred2, msg="Predictions don't match"):
    """Verify predictions match exactly."""
    assert len(y_pred1) == len(y_pred2), f"{msg}: Different lengths"
    assert all(y1 == y2 for y1, y2 in zip(y_pred1, y_pred2)), msg


def assert_valid_confidence_bounds(predictions, bounds, msg="Invalid confidence bounds"):
    """Ensure confidence bounds contain predictions."""
    low, high = bounds
    for i, pred in enumerate(predictions):
        assert low[i] <= pred <= high[i], f"{msg} at index {i}"
