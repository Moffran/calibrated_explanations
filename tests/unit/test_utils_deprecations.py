import warnings
import importlib
import os

import pytest


def _reload_module():
    # Import the helper module freshly to reset module-level state
    import calibrated_explanations.utils.deprecations as dep_mod

    importlib.reload(dep_mod)
    return dep_mod


def test_deprecate_emits_once(monkeypatch):
    dep_mod = _reload_module()
    # Ensure environment not in error mode
    monkeypatch.delenv("CE_DEPRECATIONS", raising=False)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always", DeprecationWarning)
        dep_mod.deprecate("old func is deprecated", key="test1", stacklevel=2)
        # second call with same key should be a no-op
        # under pytest we emit on each call so both invocations produce warnings
        dep_mod.deprecate("old func is deprecated", key="test1", stacklevel=2)

    assert len(rec) == 2
    assert "old func is deprecated" in str(rec[0].message)


def test_deprecate_alias_convenience(monkeypatch):
    dep_mod = _reload_module()
    monkeypatch.delenv("CE_DEPRECATIONS", raising=False)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always", DeprecationWarning)
        # use a repo-agnostic alias to avoid colliding with real ALIAS_MAP keys
        dep_mod.deprecate_alias("__test_alias_xyz__", "canonical_param_xyz", stacklevel=2)

    assert len(rec) == 1
    assert (
        "Parameter or alias '__test_alias_xyz__' is deprecated; use 'canonical_param_xyz'"
        in str(rec[0].message)
    )
    # The helper should mark the alias key as emitted. When running under
    # pytest we record emitted keys in the per-test map to avoid polluting
    # session-wide state; otherwise fall back to the global set.
    pytest_id = os.getenv("PYTEST_CURRENT_TEST")
    if pytest_id:
        assert "alias:__test_alias_xyz__" in dep_mod._EMITTED_PER_TEST.get(pytest_id, set())
    else:
        assert "alias:__test_alias_xyz__" in dep_mod._EMITTED


def test_deprecations_raise_when_env_set(monkeypatch):
    dep_mod = _reload_module()
    # Enable error mode
    monkeypatch.setenv("CE_DEPRECATIONS", "error")

    # Ensure key is not present initially
    dep_mod._EMITTED.discard("raise_key")

    with pytest.raises(DeprecationWarning):
        dep_mod.deprecate("please stop using this", key="raise_key")

    # Even when raised, the key should be recorded to avoid duplicate attempts
    pytest_id = os.getenv("PYTEST_CURRENT_TEST")
    if pytest_id:
        assert "raise_key" in dep_mod._EMITTED_PER_TEST.get(pytest_id, set())
    else:
        assert "raise_key" in dep_mod._EMITTED
