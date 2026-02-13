from __future__ import annotations

import pytest

from calibrated_explanations.api.params import (
    canonicalize_kwargs,
    validate_param_combination,
    warn_on_aliases,
)
from calibrated_explanations.core.exceptions import ConfigurationError


def test_canonicalize_kwargs_maps_alias_without_overwriting() -> None:
    out = canonicalize_kwargs({"alpha": (5, 95), "threshold": 0.4})
    assert out["low_high_percentiles"] == (5, 95)
    assert out["alpha"] == (5, 95)

    out_with_canonical = canonicalize_kwargs({"alpha": (1, 99), "low_high_percentiles": (10, 90)})
    assert out_with_canonical["low_high_percentiles"] == (10, 90)


def test_validate_param_combination_rejects_conflict() -> None:
    with pytest.raises(ConfigurationError):
        validate_param_combination({"threshold": 0.5, "confidence_level": 0.95})


def test_warn_on_aliases_emits_user_warning() -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        warn_on_aliases({"alpha": (5, 95)})
