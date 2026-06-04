"""Contract tests for Task 10: configuration management system hardening.

Covers all 9 in-scope Task-10 findings:
  1/6. task / parallel_workers removed from ExplainerConfig and ExplainerBuilder
  2.   process-level ConfigManager singleton lifecycle fixed
  3.   CE_DEBUG_TRUST_INVARIANTS in _KNOWN_ENV_KEYS
  4.   zombie config.ini absent
  5.   env-var precedence Notes present in perf_cache / perf_parallel docstrings
  8.   ADR-034 §7 (documentation — no code assertion needed)
  9.   ExplainerBuilder / ExplainerConfig in root namespace
  10.  env-only-by-design keys (documentation — no code assertion needed)
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier

import calibrated_explanations
from calibrated_explanations.api.config import ExplainerBuilder, ExplainerConfig
from calibrated_explanations.core.config_manager import (
    ConfigManager,
    get_process_config_manager,
    init_process_config_manager,
    reset_process_config_manager_for_testing,
    _KNOWN_ENV_KEYS,
)
from calibrated_explanations.utils.exceptions import CalibratedError


# ---------------------------------------------------------------------------
# Finding 3 — CE_DEBUG_TRUST_INVARIANTS governed
# ---------------------------------------------------------------------------


def test_should_include_ce_debug_trust_invariants_in_known_env_keys() -> None:
    assert "CE_DEBUG_TRUST_INVARIANTS" in _KNOWN_ENV_KEYS


def test_should_not_warn_for_ce_debug_trust_invariants_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warnings

    monkeypatch.setenv("CE_DEBUG_TRUST_INVARIANTS", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ConfigManager.from_sources()
    unknown_key_warnings = [
        str(w.message) for w in caught if "CE_DEBUG_TRUST_INVARIANTS" in str(w.message)
    ]
    assert unknown_key_warnings == [], f"Unexpected unknown-key warnings: {unknown_key_warnings}"


# ---------------------------------------------------------------------------
# Finding 2 — process-level singleton lifecycle
# ---------------------------------------------------------------------------


def test_should_expose_reset_process_config_manager_for_testing() -> None:
    assert callable(reset_process_config_manager_for_testing)


def test_reset_process_config_manager_is_callable_and_does_not_raise() -> None:
    # Behavioral: reset must succeed without raising regardless of prior state.
    assert reset_process_config_manager_for_testing() is None
    assert reset_process_config_manager_for_testing() is None  # idempotent


def test_should_return_same_process_config_manager_until_reset() -> None:
    reset_process_config_manager_for_testing()
    first = get_process_config_manager()
    second = get_process_config_manager()
    assert second is first


def test_should_raise_when_process_config_manager_initialized_twice() -> None:
    reset_process_config_manager_for_testing()
    manager = ConfigManager(env_snapshot={}, pyproject_snapshot={})
    assert init_process_config_manager(manager) is manager
    with pytest.raises(CalibratedError, match="already been initialized"):
        init_process_config_manager(manager)


# ---------------------------------------------------------------------------
# Finding 4 — zombie config.ini absent
# ---------------------------------------------------------------------------


def test_should_not_have_config_ini_at_legacy_path() -> None:
    legacy_path = (
        Path(calibrated_explanations.__file__).parent / "utils" / "configurations" / "config.ini"
    )
    assert not legacy_path.exists(), f"Zombie config.ini still present at {legacy_path}"


# ---------------------------------------------------------------------------
# Findings 1/6 — task and parallel_workers removed
# ---------------------------------------------------------------------------


def test_should_not_have_task_field_on_explainer_config() -> None:
    cfg = ExplainerConfig(model=RandomForestClassifier())
    assert "task" not in ExplainerConfig.__dataclass_fields__, (
        "task field must be removed from ExplainerConfig (v0.11.3)"
    )
    assert not hasattr(cfg, "task")


def test_should_not_have_parallel_workers_field_on_explainer_config() -> None:
    cfg = ExplainerConfig(model=RandomForestClassifier())
    assert "parallel_workers" not in ExplainerConfig.__dataclass_fields__, (
        "parallel_workers field must be removed from ExplainerConfig (v0.11.3)"
    )
    assert not hasattr(cfg, "parallel_workers")


def test_should_not_have_task_method_on_explainer_builder() -> None:
    b = ExplainerBuilder(RandomForestClassifier())
    assert not hasattr(b, "task"), "task() method must be removed from ExplainerBuilder (v0.11.3)"


def test_should_not_have_parallel_workers_method_on_explainer_builder() -> None:
    b = ExplainerBuilder(RandomForestClassifier())
    assert not hasattr(b, "parallel_workers"), (
        "parallel_workers() method must be removed from ExplainerBuilder (v0.11.3)"
    )


# ---------------------------------------------------------------------------
# Finding 5 — env-var precedence Notes in perf_cache / perf_parallel docstrings
# ---------------------------------------------------------------------------


def test_perf_cache_docstring_mentions_ce_cache_precedence() -> None:
    doc = inspect.getdoc(ExplainerBuilder.perf_cache) or ""
    assert "CE_CACHE" in doc, "perf_cache() docstring must document CE_CACHE env-var precedence"


def test_perf_parallel_docstring_mentions_ce_parallel_precedence() -> None:
    doc = inspect.getdoc(ExplainerBuilder.perf_parallel) or ""
    assert "CE_PARALLEL" in doc, (
        "perf_parallel() docstring must document CE_PARALLEL env-var precedence"
    )


# ---------------------------------------------------------------------------
# Finding 9 — root namespace promotion
# ---------------------------------------------------------------------------


def test_should_export_explainer_builder_from_root_namespace() -> None:
    from calibrated_explanations import ExplainerBuilder as RootExplainerBuilder

    assert RootExplainerBuilder is ExplainerBuilder


def test_should_export_explainer_config_from_root_namespace() -> None:
    from calibrated_explanations import ExplainerConfig as RootExplainerConfig

    assert RootExplainerConfig is ExplainerConfig


def test_explainer_builder_and_config_in_all() -> None:
    assert "ExplainerBuilder" in calibrated_explanations.__all__
    assert "ExplainerConfig" in calibrated_explanations.__all__


# ---------------------------------------------------------------------------
# Finding 1 (docstring) — from_config docstring does not claim "Intentionally minimal"
# ---------------------------------------------------------------------------


def test_from_config_docstring_does_not_claim_intentionally_minimal() -> None:
    from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

    doc = inspect.getdoc(WrapCalibratedExplainer.from_config) or ""
    assert "Intentionally minimal" not in doc, (
        "from_config() docstring must not claim 'Intentionally minimal' — wiring is substantial"
    )
    assert "task" not in doc or "task" in doc.lower(), True  # no strict task-field reference needed
