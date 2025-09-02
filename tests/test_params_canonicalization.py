from calibrated_explanations.api.params import (
    ALIAS_MAP,
    canonicalize_kwargs,
    validate_param_combination,
)


def test_alias_mapping_adds_canonical_when_missing():
    kwargs = {"alpha": (5, 95)}
    out = canonicalize_kwargs(kwargs)
    assert out["low_high_percentiles"] == (5, 95)
    # Original alias is preserved in Phase 1B
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

    # No-op in Phase 1B
    validate_param_combination(out)


def test_alias_map_contains_expected_minimal_keys():
    # Guard minimal surface; can expand later
    for k in ("alpha", "alphas", "n_jobs"):
        assert k in ALIAS_MAP
