from __future__ import annotations

import json

import pytest

from calibrated_explanations.core.config_manager import ConfigManager, ConfigSpec
from calibrated_explanations.core.config_manager import _KNOWN_ENV_KEYS
from calibrated_explanations.utils.exceptions import ConfigurationError


def test_should_capture_env_snapshot_when_env_changes_after_construction(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CE_PLOT_STYLE", "legacy")
    manager = ConfigManager.from_sources()
    monkeypatch.setenv("CE_PLOT_STYLE", "plot_spec.default")
    assert manager.env("CE_PLOT_STYLE") == "legacy"
    reconstructed = ConfigManager.from_sources()
    assert reconstructed.env("CE_PLOT_STYLE") == "plot_spec.default"


def test_should_export_snapshot_with_metadata() -> None:
    manager = ConfigManager(
        env_snapshot={"CE_PLOT_STYLE": "legacy"},
        pyproject_snapshot={
            "plugins": {},
            "explanations": {},
            "intervals": {},
            "plots": {"style": "legacy"},
            "telemetry": {},
        },
    )
    snapshot = manager.export_effective()
    assert snapshot.profile_id == "default"
    assert snapshot.schema_version == "1"
    assert snapshot.values["env.CE_PLOT_STYLE"] == "legacy"
    assert snapshot.sources["env.CE_PLOT_STYLE"] == "env"
    assert snapshot.values["effective.CE_PLOT_STYLE"] == "legacy"
    assert snapshot.sources["effective.CE_PLOT_STYLE"] == "env"


def test_should_raise_when_unknown_pyproject_keys_under_strict_validation() -> None:
    with pytest.raises(ConfigurationError):
        ConfigManager(
            env_snapshot={},
            pyproject_snapshot={
                "plugins": {"unknown_key": "x"},
                "explanations": {},
                "intervals": {},
                "plots": {},
                "telemetry": {},
            },
        )


def test_should_return_empty_unknown_section_when_non_strict() -> None:
    manager = ConfigManager(
        env_snapshot={},
        pyproject_snapshot={
            "plugins": {},
            "explanations": {},
            "intervals": {},
            "plots": {},
            "telemetry": {},
        },
        strict=False,
    )
    assert manager.pyproject_section("not_a_section") == {}


def test_should_capture_validation_issues_when_non_strict() -> None:
    manager = ConfigManager(
        env_snapshot={},
        pyproject_snapshot={
            "plugins": {"unknown_key": "x"},
            "explanations": {},
            "intervals": {},
            "plots": {},
            "telemetry": {},
            "unexpected_section": {},
        },
        strict=False,
    )
    report = manager.validation_report()
    assert report.has_errors is True
    assert len(report.issues) == 2


def test_should_raise_when_supported_key_has_invalid_type_under_strict_validation() -> None:
    with pytest.raises(ConfigurationError):
        ConfigManager(
            env_snapshot={},
            pyproject_snapshot={
                "plugins": {"trusted": 123},
                "explanations": {},
                "intervals": {},
                "plots": {},
                "telemetry": {},
            },
        )


def test_should_raise_when_supported_key_has_invalid_value_under_strict_validation() -> None:
    with pytest.raises(ConfigurationError):
        ConfigManager(
            env_snapshot={},
            pyproject_snapshot={
                "plugins": {},
                "explanations": {},
                "intervals": {},
                "plots": {},
                "telemetry": {"diagnostic_mode": "definitely"},
            },
        )


def test_should_capture_supported_key_validation_issue_when_non_strict() -> None:
    manager = ConfigManager(
        env_snapshot={},
        pyproject_snapshot={
            "plugins": {},
            "explanations": {"factual": ""},
            "intervals": {},
            "plots": {},
            "telemetry": {},
        },
        strict=False,
    )
    report = manager.validation_report()
    assert report.has_errors is True
    assert any(issue.location == "pyproject.explanations.factual" for issue in report.issues)


def test_should_warn_for_unknown_ce_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("CE_UNKNOWN_SETTING", "x")
    with pytest.warns(UserWarning, match=r"Unknown CE_\* environment variables detected"):
        ConfigManager.from_sources()


def test_should_not_include_duplicate_known_env_keys() -> None:
    assert len(set(_KNOWN_ENV_KEYS)) == len(_KNOWN_ENV_KEYS)


def test_should_merge_config_specs_while_preserving_base_namespace() -> None:
    base = ConfigSpec(
        known_env_keys=("CE_BASE",),
        section_schema={"base": ("enabled",)},
        resolution_spec={"CE_BASE": ("base", "enabled", False)},
        value_validators={},
        pyproject_tool_namespace=("tool", "base"),
    )
    extension = ConfigSpec(
        known_env_keys=("CE_BASE", "CE_EXTENSION"),
        section_schema={"base": ("mode",), "extension": ("enabled",)},
        resolution_spec={"CE_EXTENSION": ("extension", "enabled", True)},
        value_validators={
            ("extension", "enabled"): (lambda value: isinstance(value, bool), "bool")
        },
        pyproject_tool_namespace=("tool", "extension"),
    )

    merged = base.merged_with(extension)

    assert merged.known_env_keys == ("CE_BASE", "CE_EXTENSION")
    assert merged.section_schema["base"] == ("enabled", "mode")
    assert merged.resolution_spec["CE_EXTENSION"] == ("extension", "enabled", True)
    assert merged.pyproject_tool_namespace == ("tool", "base")


_EMPTY_PYPROJECT: dict = {
    "plugins": {},
    "explanations": {},
    "intervals": {},
    "plots": {},
    "telemetry": {},
}


def test_telemetry_diagnostic_mode_env_takes_precedence_over_pyproject() -> None:
    manager = ConfigManager(
        env_snapshot={"CE_TELEMETRY_DIAGNOSTIC_MODE": "1"},
        pyproject_snapshot={**_EMPTY_PYPROJECT, "telemetry": {"diagnostic_mode": False}},
    )
    assert manager.telemetry_diagnostic_mode() is True


def test_env_default_returned_when_key_absent() -> None:
    manager = ConfigManager(env_snapshot={}, pyproject_snapshot=_EMPTY_PYPROJECT)
    assert manager.env("CE_CACHE", "sentinel") == "sentinel"


def test_export_values_are_immutable() -> None:
    manager = ConfigManager(
        env_snapshot={"CE_PLOT_STYLE": "x"},
        pyproject_snapshot=_EMPTY_PYPROJECT,
    )
    snapshot = manager.export_effective()
    with pytest.raises(TypeError):
        snapshot.values["injected"] = "bad"  # type: ignore[index]


def test_effective_export_uses_pyproject_when_env_absent() -> None:
    manager = ConfigManager(
        env_snapshot={},
        pyproject_snapshot={
            "plugins": {},
            "explanations": {},
            "intervals": {},
            "plots": {"style": "plot_spec.default"},
            "telemetry": {},
        },
    )
    snapshot = manager.export_effective()
    assert snapshot.values["effective.CE_PLOT_STYLE"] == "plot_spec.default"
    assert snapshot.sources["effective.CE_PLOT_STYLE"] == "pyproject"


def test_effective_export_uses_default_profile_when_no_env_or_pyproject() -> None:
    manager = ConfigManager(env_snapshot={}, pyproject_snapshot=_EMPTY_PYPROJECT)
    snapshot = manager.export_effective()
    assert snapshot.values["effective.CE_PLOT_RENDERER"] is None
    assert snapshot.sources["effective.CE_PLOT_RENDERER"] == "default_profile"


def test_should_snapshot_raw_plugin_config_without_live_env_reads(monkeypatch) -> None:
    monkeypatch.setenv(
        "CE_PLUGIN_CONFIG_JSON",
        json.dumps({"example.plugin": {"threshold": 0.7, "labels": ["a", "b"]}}),
    )
    manager = ConfigManager.from_sources()
    monkeypatch.setenv(
        "CE_PLUGIN_CONFIG_JSON",
        json.dumps({"example.plugin": {"threshold": 0.1}}),
    )

    config = manager.plugin_config("example.plugin")

    assert config["threshold"] == 0.7
    assert config["labels"] == ("a", "b")


def test_should_preserve_plugin_config_source_attribution_and_redact_secrets() -> None:
    manager = ConfigManager(
        env_snapshot={
            "CE_PLUGIN_CONFIG_JSON": json.dumps(
                {"example.plugin": {"threshold": 0.7, "api_token": "env-secret"}}
            )
        },
        pyproject_snapshot={
            **_EMPTY_PYPROJECT,
            "plugin_configs": {
                "example.plugin": {
                    "threshold": 0.2,
                    "api_token": "py-secret",
                    "mode": "pyproject",
                }
            },
        },
    )

    assert manager.plugin_config("example.plugin")["threshold"] == 0.7
    assert manager.plugin_config_sources("example.plugin") == {
        "threshold": "env",
        "api_token": "env",
        "mode": "pyproject",
    }

    snapshot = manager.export_effective(
        plugin_config_schemas={
            "example.plugin": {
                "version": 1,
                "keys": {"api_token": {"type": "str", "sensitive": True}},
            }
        }
    )

    assert snapshot.values["diagnostic.plugin_config_export_schema_version"] == "provisional-1"
    assert snapshot.values["effective.plugin_config.example.plugin"]["api_token"] == "<redacted>"
    assert snapshot.sources["effective.plugin_config.example.plugin.threshold"] == "env"
    assert snapshot.sources["effective.plugin_config.example.plugin.mode"] == "pyproject"


def test_should_fail_clearly_when_plugin_config_json_is_malformed() -> None:
    with pytest.raises(ConfigurationError, match="Malformed CE_PLUGIN_CONFIG_JSON"):
        ConfigManager(
            env_snapshot={"CE_PLUGIN_CONFIG_JSON": "{not-json"},
            pyproject_snapshot=_EMPTY_PYPROJECT,
        )


def test_should_validate_unselected_plugin_config_with_strict_or_permissive_behavior() -> None:
    manager = ConfigManager(
        env_snapshot={},
        pyproject_snapshot={
            **_EMPTY_PYPROJECT,
            "plugin_configs": {"configured.plugin": {"enabled": True}},
        },
    )

    with pytest.raises(ConfigurationError, match="unselected plugin"):
        manager.validate_plugin_config_selection(("selected.plugin",), strict=True)

    with pytest.warns(UserWarning, match="unselected plugin"):
        report = manager.validate_plugin_config_selection(("selected.plugin",), strict=False)

    assert report.has_errors is True


def test_should_reject_malformed_pyproject_plugin_config_shape() -> None:
    with pytest.raises(ConfigurationError, match="Plugin config values must be mappings"):
        ConfigManager(
            env_snapshot={},
            pyproject_snapshot={
                **_EMPTY_PYPROJECT,
                "plugin_configs": {"example.plugin": "not-a-mapping"},
            },
        )


# ---------------------------------------------------------------------------
# Phase B: module-specific singleton ownership assertions (ADR-034 Task 1)
# ---------------------------------------------------------------------------


def test_should_instantiate_orchestrator_without_errors() -> None:
    """PredictionOrchestrator.__init__ must complete successfully (proves ConfigManager.from_sources() wiring is correct)."""
    from unittest.mock import MagicMock

    from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator

    fake_explainer = MagicMock()
    fake_explainer.plugin_manager.initialize_chains.return_value = None
    orch = PredictionOrchestrator(fake_explainer)
    assert orch is not None


def test_should_snapshot_ce_cache_env_at_construction_time() -> None:
    """CacheConfig.from_env reads CE_CACHE from ConfigManager snapshot, not live env."""
    from calibrated_explanations.cache.cache import CacheConfig

    # Build manager with CE_CACHE enabled.
    mgr = ConfigManager(env_snapshot={"CE_CACHE": "on"}, pyproject_snapshot=_EMPTY_PYPROJECT)
    cfg = CacheConfig.from_env(config_manager=mgr)
    assert cfg.enabled is True

    # A manager with CE_CACHE absent returns disabled (default).
    mgr_off = ConfigManager(env_snapshot={}, pyproject_snapshot=_EMPTY_PYPROJECT)
    cfg_off = CacheConfig.from_env(config_manager=mgr_off)
    assert cfg_off.enabled is False


def test_should_snapshot_ce_parallel_env_at_construction_time() -> None:
    """ParallelConfig.from_env reads CE_PARALLEL from ConfigManager snapshot, not live env."""
    from calibrated_explanations.parallel.parallel import ParallelConfig

    mgr = ConfigManager(env_snapshot={"CE_PARALLEL": "on"}, pyproject_snapshot=_EMPTY_PYPROJECT)
    cfg = ParallelConfig.from_env(config_manager=mgr)
    assert cfg.enabled is True

    mgr_off = ConfigManager(env_snapshot={}, pyproject_snapshot=_EMPTY_PYPROJECT)
    cfg_off = ParallelConfig.from_env(config_manager=mgr_off)
    assert cfg_off.enabled is False
