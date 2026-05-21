"""Utilities for managing difficulty estimator configuration and validation.

The basic ``validate_difficulty_estimator`` contract remains intentionally small
for backward compatibility: estimators must expose ``fitted=True`` and an
``apply(x)``-compatible interface consumed by CE.

For conformal reject-score normalization workflows, optional provenance checks
can be applied via ``validate_difficulty_estimator_provenance``. These checks are
metadata-driven and permissive by default to avoid breaking third-party
estimators that do not expose provenance attributes.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

from ..utils.exceptions import NotFittedError, ValidationError


@dataclass(frozen=True)
class DifficultyEstimatorProvenanceReport:
    """Result of optional provenance validation for difficulty estimators.

    Attributes
    ----------
    provenance_available : bool
        True when at least one provenance metadata field is present.
    warning_emitted : bool
        True when permissive-mode validation emitted a warning.
    validation_mode : str
        ``"strict"`` or ``"permissive"``.
    fit_source : str or None
        Free-form estimator fit-source metadata when available.
    uses_calibration_labels : bool or None
        Whether estimator metadata says calibration labels/targets were used.
    uses_calibration_residuals : bool or None
        Whether estimator metadata says calibration residuals were used.
    cross_fitted : bool or None
        Whether cross-fitting metadata is present and true.
    """

    provenance_available: bool
    warning_emitted: bool
    validation_mode: str
    fit_source: str | None
    uses_calibration_labels: bool | None
    uses_calibration_residuals: bool | None
    cross_fitted: bool | None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    return None


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def validate_difficulty_estimator_provenance(
    difficulty_estimator: Any,
    *,
    strict: bool = False,
    logger: logging.Logger | None = None,
) -> DifficultyEstimatorProvenanceReport:
    """Optionally validate provenance metadata for conformal reject scoring.

    Provenance metadata is optional and backward compatible. When metadata is
    absent, this function returns a permissive report and does not fail.

    Validation policy
    -----------------
    - OK: fitted on proper training data only.
    - OK: unsupervised feature-only calibration use when explicitly marked.
    - Warn (permissive) or fail (strict): calibration labels/residuals used
      without cross-fitting metadata.

    Parameters
    ----------
    difficulty_estimator : Any
        Difficulty estimator object, or ``None``.
    strict : bool, default=False
        If ``True``, provenance violations raise ``ValidationError``.
        If ``False``, provenance violations emit ``UserWarning`` and continue.
    logger : logging.Logger, optional
        Logger used for INFO visibility of permissive fallbacks.

    Returns
    -------
    DifficultyEstimatorProvenanceReport
        Structured provenance outcome for metadata/audit propagation.

    Raises
    ------
    ValidationError
        If strict validation is enabled and unsafe provenance is detected.
    """
    active_logger = logger or logging.getLogger(__name__)
    mode = "strict" if strict else "permissive"
    if difficulty_estimator is None:
        return DifficultyEstimatorProvenanceReport(
            provenance_available=False,
            warning_emitted=False,
            validation_mode=mode,
            fit_source=None,
            uses_calibration_labels=None,
            uses_calibration_residuals=None,
            cross_fitted=None,
        )

    fit_source_raw = _first_attr(
        difficulty_estimator,
        ("fit_source", "difficulty_fit_source", "provenance_fit_source"),
    )
    fit_source = None if fit_source_raw is None else str(fit_source_raw)
    uses_calibration_labels = _optional_bool(
        _first_attr(
            difficulty_estimator,
            (
                "uses_calibration_labels",
                "uses_calibration_targets",
                "uses_calibration_y",
            ),
        )
    )
    uses_calibration_residuals = _optional_bool(
        _first_attr(
            difficulty_estimator,
            (
                "uses_calibration_residuals",
                "uses_residuals",
                "uses_calibration_errors",
            ),
        )
    )
    cross_fitted = _optional_bool(
        _first_attr(
            difficulty_estimator,
            ("cross_fitted", "is_cross_fitted", "used_cross_fitting"),
        )
    )
    unsupervised_calibration_features = _optional_bool(
        _first_attr(
            difficulty_estimator,
            (
                "unsupervised_calibration_features",
                "uses_calibration_features_unsupervised",
                "allow_unsupervised_calibration_features",
            ),
        )
    )

    fit_source_norm = "" if fit_source is None else fit_source.strip().lower()
    source_mentions_label_leakage = (
        "calibration" in fit_source_norm and "label" in fit_source_norm
    ) or ("calibration" in fit_source_norm and "residual" in fit_source_norm)
    source_mentions_unsupervised_calibration = (
        "calibration" in fit_source_norm
        and "feature" in fit_source_norm
        and "unsuper" in fit_source_norm
    )
    source_mentions_proper_train = (
        "proper" in fit_source_norm
        or "train_only" in fit_source_norm
        or "proper_train" in fit_source_norm
    )

    provenance_available = any(
        value is not None
        for value in (
            fit_source,
            uses_calibration_labels,
            uses_calibration_residuals,
            cross_fitted,
            unsupervised_calibration_features,
        )
    )

    unsafe_without_cross_fit = (
        bool(uses_calibration_labels)
        or bool(uses_calibration_residuals)
        or source_mentions_label_leakage
    ) and (cross_fitted is not True)

    explicitly_safe_unsupervised = (
        ((unsupervised_calibration_features is True) or source_mentions_unsupervised_calibration)
        and not bool(uses_calibration_labels)
        and not bool(uses_calibration_residuals)
    )

    warning_emitted = False
    if unsafe_without_cross_fit and not explicitly_safe_unsupervised:
        message = (
            "Difficulty estimator provenance indicates calibration labels/residuals "
            "without cross-fitting. This may invalidate conformal reject calibration."
        )
        details = {
            "fit_source": fit_source,
            "uses_calibration_labels": uses_calibration_labels,
            "uses_calibration_residuals": uses_calibration_residuals,
            "cross_fitted": cross_fitted,
            "validation_mode": mode,
        }
        if strict:
            raise ValidationError(message, details=details)
        active_logger.info("%s details=%s", message, details)
        warnings.warn(message, UserWarning, stacklevel=3)
        warning_emitted = True
    elif source_mentions_proper_train or explicitly_safe_unsupervised:
        # Explicitly recognized safe provenance path; no warning needed.
        pass

    return DifficultyEstimatorProvenanceReport(
        provenance_available=provenance_available,
        warning_emitted=warning_emitted,
        validation_mode=mode,
        fit_source=fit_source,
        uses_calibration_labels=uses_calibration_labels,
        uses_calibration_residuals=uses_calibration_residuals,
        cross_fitted=cross_fitted,
    )


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


__all__ = [
    "DifficultyEstimatorProvenanceReport",
    "validate_difficulty_estimator",
    "validate_difficulty_estimator_provenance",
]
