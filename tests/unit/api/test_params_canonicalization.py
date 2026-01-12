import pytest
import warnings

from calibrated_explanations.api.params import (
    ALIAS_MAP,
    canonicalize_kwargs,
    validate_param_combination,
    warn_on_aliases,
)
from calibrated_explanations.utils.exceptions import ConfigurationError


def test_alias_mapping_adds_canonical_when_missing():
    kwargs = {"alpha": (5, 95)}
    out = canonicalize_kwargs(kwargs)
    assert out["low_high_percentiles"] == (5, 95)
    # Original alias is preserved
    assert out["alpha"] == (5, 95)


def test_alias_does_not_override_existing_canonical():
    kwargs = {"alpha": (10, 90), "low_high_percentiles": (5, 95)}
    out = canonicalize_kwargs(kwargs)
    # Canonical value remains
    assert out["low_high_percentiles"] == (5, 95)
    # Alias preserved
    assert out["alpha"] == (10, 90)


def test_unknown_keys_preserved_and_validate_noop():
    kwargs = {"unknown_param": 123, "n_jobs": 4}
    out = canonicalize_kwargs(kwargs)
    assert out["unknown_param"] == 123
    # n_jobs alias mapped to parallel_workers
    assert out["parallel_workers"] == 4

    validate_param_combination(out)


def test_alias_map_contains_expected_minimal_keys():
    # Guard minimal surface; can expand later
    for k in ("alpha", "alphas", "n_jobs"):
        assert k in ALIAS_MAP


def test_validate_param_combination_raises_on_conflict():
    kwargs = {"threshold": 0.5, "confidence_level": 0.95}
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        validate_param_combination(kwargs)


def test_warn_on_aliases_emits_warning():
    kwargs = {"alpha": (5, 95)}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_aliases(kwargs)
        # Check that warnings were issued
        assert len(w) >= 1
        # Check that at least one is UserWarning or DeprecationWarning
        categories = [warn.category for warn in w]
        assert UserWarning in categories or DeprecationWarning in categories
        # Check message
        messages = [str(warn.message) for warn in w]
        assert any("alpha" in msg and "deprecated" in msg.lower() for msg in messages)
