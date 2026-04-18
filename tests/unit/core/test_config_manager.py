from __future__ import annotations

import pytest

from calibrated_explanations.core.config_manager import ConfigManager
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
