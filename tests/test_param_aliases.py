"""Unit tests for parameter alias canonicalization utilities.

Covers key behaviors:
- simple alias mapping (e.g., percentiles -> low_high_percentiles)
- idempotency on already-canonical kwargs
- conflict detection when both alias and canonical are provided with different values
"""
from __future__ import annotations

import pytest

from calibrated_explanations.core.param_aliases import canonicalize_params
from calibrated_explanations.core.exceptions import ValidationError


def test_simple_alias_mapping():
    params = {
        "percentiles": (10, 90),
        "random_state": 123,
        "fast_mode": True,
    }
    out = canonicalize_params(dict(params))
    # Canonical keys present
    assert out["low_high_percentiles"] == (10, 90)
    assert out["seed"] == 123
    assert out["fast"] is True
    # Aliases removed
    for k in ("percentiles", "random_state", "fast_mode"):
        assert k not in out


def test_idempotent_when_already_canonical():
    params = {
        "low_high_percentiles": (5, 95),
        "seed": 7,
        "fast": False,
    }
    out = canonicalize_params(dict(params))
    assert out == params


def test_conflict_detection_raises():
    params = {
        "low_high_percentiles": (5, 95),
        "percentiles": (10, 90),
    }
    with pytest.raises(ValidationError):
        canonicalize_params(dict(params), raise_on_conflict=True)


def test_unknown_keys_preserved():
    params = {"custom_arg": 42}
    out = canonicalize_params(dict(params))
    assert out["custom_arg"] == 42

