"""Utilities for managing difficulty estimator configuration and validation."""

from ..utils.exceptions import NotFittedError


def validate_difficulty_estimator(difficulty_estimator):
    """Validate that a difficulty estimator is properly fitted.

    Parameters
    ----------
    difficulty_estimator : crepes.extras.DifficultyEstimator or None
        A difficulty estimator object from the crepes package, or None.

    Raises
    ------
    NotFittedError
        If the difficulty estimator is not fitted.

    Returns
    -------
    difficulty_estimator
        The validated difficulty estimator (or None).
    """
    if difficulty_estimator is not None:
        try:
            if not difficulty_estimator.fitted:
                raise NotFittedError(
                    "The difficulty estimator is not fitted. Please fit the estimator first."
                )
        except AttributeError as e:
            raise NotFittedError(
                "The difficulty estimator is not fitted. Please fit the estimator first."
            ) from e
    return difficulty_estimator


__all__ = ["validate_difficulty_estimator"]
