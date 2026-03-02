from __future__ import annotations

import pytest

from calibrated_explanations.api.params import (
    canonicalize_kwargs,
    reject_removed_aliases,
    validate_param_combination,
    warn_on_aliases,
)
from calibrated_explanations.core.exceptions import ConfigurationError


def test_canonicalize_kwargs_preserves_keys_without_alias_mapping() -> None:
    out = canonicalize_kwargs({"threshold": 0.4})
    assert out == {"threshold": 0.4}


def test_reject_removed_aliases_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="removed in v0.11.0"):
        reject_removed_aliases({"alpha": (5, 95)})


def test_warn_on_aliases_delegates_to_removed_alias_guard() -> None:
    with pytest.raises(ConfigurationError, match="removed in v0.11.0"):
        warn_on_aliases({"n_jobs": 2})


def test_validate_param_combination_rejects_conflict() -> None:
    with pytest.raises(ConfigurationError):
        validate_param_combination({"threshold": 0.5, "confidence_level": 0.95})
