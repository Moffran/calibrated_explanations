"""Unit tests for GuardedOptions (ADR-038)."""

from __future__ import annotations

import pytest

from calibrated_explanations import GuardedOptions
from calibrated_explanations.utils.exceptions import ValidationError


def test_should_have_correct_defaults_when_instantiated_without_args():
    opts = GuardedOptions()
    assert opts.confidence == 0.9
    assert opts.n_neighbors == 5
    assert opts.normalize is True
    assert opts.merge_adjacent is False
    assert opts.verbose is False


def test_should_accept_custom_values_when_all_fields_are_valid():
    opts = GuardedOptions(
        confidence=0.8, n_neighbors=3, normalize=False, merge_adjacent=True, verbose=True
    )
    assert opts.confidence == 0.8
    assert opts.n_neighbors == 3
    assert opts.normalize is False
    assert opts.merge_adjacent is True
    assert opts.verbose is True


def test_should_raise_type_error_when_unknown_field_is_passed():
    with pytest.raises(TypeError):
        GuardedOptions(unknown_param=1)  # type: ignore[call-arg]


def test_should_raise_when_field_is_assigned_after_construction():
    opts = GuardedOptions()
    with pytest.raises(Exception):
        opts.confidence = 0.5  # type: ignore[misc]


def test_should_raise_validation_error_when_confidence_is_zero():
    with pytest.raises(ValidationError):
        GuardedOptions(confidence=0.0)


def test_should_raise_validation_error_when_confidence_is_one():
    with pytest.raises(ValidationError):
        GuardedOptions(confidence=1.0)


def test_should_raise_validation_error_when_confidence_is_negative():
    with pytest.raises(ValidationError):
        GuardedOptions(confidence=-0.1)


def test_should_raise_validation_error_when_confidence_is_greater_than_one():
    with pytest.raises(ValidationError):
        GuardedOptions(confidence=1.1)


def test_should_accept_boundary_adjacent_confidence_values():
    lo = GuardedOptions(confidence=0.01)
    hi = GuardedOptions(confidence=0.99)
    assert lo.confidence == pytest.approx(0.01)
    assert hi.confidence == pytest.approx(0.99)


def test_should_raise_validation_error_when_n_neighbors_is_zero():
    with pytest.raises(ValidationError):
        GuardedOptions(n_neighbors=0)


def test_should_raise_validation_error_when_n_neighbors_is_negative():
    with pytest.raises(ValidationError):
        GuardedOptions(n_neighbors=-1)


def test_should_accept_n_neighbors_of_one():
    opts = GuardedOptions(n_neighbors=1)
    assert opts.n_neighbors == 1
