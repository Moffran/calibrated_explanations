from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.core.difficulty_estimator_helpers import (
    validate_difficulty_estimator,
    validate_difficulty_estimator_provenance,
)
from calibrated_explanations.utils.exceptions import NotFittedError, ValidationError


class MinimalDifficultyEstimator:
    def __init__(self, fitted: bool = True) -> None:
        self.fitted = fitted

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.ones(len(x), dtype=float)


def test_should_accept_existing_fitted_estimator_without_provenance_metadata() -> None:
    estimator = MinimalDifficultyEstimator(fitted=True)

    validated = validate_difficulty_estimator(estimator)
    report = validate_difficulty_estimator_provenance(estimator, strict=False)

    assert validated is estimator
    assert report.provenance_available is False
    assert report.warning_emitted is False
    assert report.validation_mode == "permissive"


def test_should_fail_for_unfitted_estimator_as_before() -> None:
    estimator = MinimalDifficultyEstimator(fitted=False)

    with pytest.raises(NotFittedError):
        validate_difficulty_estimator(estimator)


def test_should_warn_when_calibration_labels_used_without_cross_fitting_in_permissive_mode() -> (
    None
):
    estimator = MinimalDifficultyEstimator(fitted=True)
    estimator.fit_source = "calibration_labels"
    estimator.uses_calibration_labels = True
    estimator.cross_fitted = False

    with pytest.warns(UserWarning, match="calibration labels/residuals"):
        report = validate_difficulty_estimator_provenance(estimator, strict=False)

    assert report.provenance_available is True
    assert report.warning_emitted is True
    assert report.validation_mode == "permissive"


def test_should_fail_when_calibration_labels_used_without_cross_fitting_in_strict_mode() -> None:
    estimator = MinimalDifficultyEstimator(fitted=True)
    estimator.fit_source = "calibration_labels"
    estimator.uses_calibration_labels = True
    estimator.cross_fitted = False

    with pytest.raises(ValidationError, match="may invalidate conformal reject calibration"):
        validate_difficulty_estimator_provenance(estimator, strict=True)
