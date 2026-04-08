"""Centralized runtime configuration management (ADR-034)."""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..utils.exceptions import ConfigurationError
from .config_helpers import read_pyproject_section

_PROFILE_ID = "default"
_SCHEMA_VERSION = "1"
_LOGGER = logging.getLogger(__name__)
_KNOWN_ENV_KEYS = (
    "CE_EXPLANATION_PLUGIN",
    "CE_EXPLANATION_PLUGIN_FACTUAL",
    "CE_EXPLANATION_PLUGIN_ALTERNATIVE",
    "CE_EXPLANATION_PLUGIN_FAST",
    "CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS",
    "CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS",
    "CE_EXPLANATION_PLUGIN_FAST_FALLBACKS",
    "CE_INTERVAL_PLUGIN",
    "CE_INTERVAL_PLUGIN_FAST",
    "CE_INTERVAL_PLUGIN_FALLBACKS",
    "CE_INTERVAL_PLUGIN_FAST_FALLBACKS",
    "CE_PLOT_STYLE",
    "CE_PLOT_STYLE_FALLBACKS",
    "CE_PLOT_RENDERER",
    "CE_TRUST_PLUGIN",
    "CE_DENY_PLUGIN",
    "CE_TELEMETRY_DIAGNOSTIC_MODE",
    "CE_CACHE",
    "CE_PARALLEL",
    "CE_PARALLEL_MIN_BATCH_SIZE",
    "CE_FEATURE_FILTER",
    "CE_STRICT_OBSERVABILITY",
    "CI",
    "GITHUB_ACTIONS",
)

_SECTION_SCHEMA: dict[str, tuple[str, ...]] = {
    "plugins": ("trusted",),
    "explanations": (
        "factual",
        "alternative",
        "fast",
        "factual_fallbacks",
        "alternative_fallbacks",
        "fast_fallbacks",
    ),
    "intervals": ("default", "fast", "default_fallbacks", "fast_fallbacks"),
    "plots": ("style", "fallbacks", "style_fallbacks", "renderer"),
    "telemetry": ("diagnostic_mode",),
}

_RESOLUTION_SPEC: dict[str, tuple[str | None, str | None, Any]] = {
    "CE_EXPLANATION_PLUGIN": ("explanations", "factual", None),
    "CE_EXPLANATION_PLUGIN_FACTUAL": ("explanations", "factual", None),
    "CE_EXPLANATION_PLUGIN_ALTERNATIVE": ("explanations", "alternative", None),
    "CE_EXPLANATION_PLUGIN_FAST": ("explanations", "fast", None),
    "CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS": ("explanations", "factual_fallbacks", ()),
    "CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS": (
        "explanations",
        "alternative_fallbacks",
        (),
    ),
    "CE_EXPLANATION_PLUGIN_FAST_FALLBACKS": ("explanations", "fast_fallbacks", ()),
    "CE_INTERVAL_PLUGIN": ("intervals", "default", None),
    "CE_INTERVAL_PLUGIN_FAST": ("intervals", "fast", None),
    "CE_INTERVAL_PLUGIN_FALLBACKS": ("intervals", "default_fallbacks", ()),
    "CE_INTERVAL_PLUGIN_FAST_FALLBACKS": ("intervals", "fast_fallbacks", ()),
    "CE_PLOT_STYLE": ("plots", "style", None),
    "CE_PLOT_STYLE_FALLBACKS": ("plots", "style_fallbacks", ()),
    "CE_PLOT_RENDERER": ("plots", "renderer", None),
    "CE_TRUST_PLUGIN": ("plugins", "trusted", ()),
    "CE_DENY_PLUGIN": (None, None, ()),
    "CE_TELEMETRY_DIAGNOSTIC_MODE": ("telemetry", "diagnostic_mode", False),
    "CE_CACHE": (None, None, None),
    "CE_PARALLEL": (None, None, None),
    "CE_PARALLEL_MIN_BATCH_SIZE": (None, None, None),
    "CE_FEATURE_FILTER": (None, None, None),
    "CE_STRICT_OBSERVABILITY": (None, None, False),
    "CI": (None, None, None),
    "GITHUB_ACTIONS": (None, None, None),
}


@dataclass(frozen=True)
class ResolvedConfigSnapshot:
    """Frozen snapshot of effective configuration values and source attribution."""

    values: Mapping[str, Any]
    sources: Mapping[str, str]
    profile_id: str = _PROFILE_ID
    schema_version: str = _SCHEMA_VERSION


@dataclass(frozen=True)
class ConfigValidationIssue:
    """Represents one validation error captured during config ingestion."""

    location: str
    message: str


@dataclass(frozen=True)
class ConfigValidationReport:
    """Validation report for strict=False config ingestion."""

    issues: tuple[ConfigValidationIssue, ...] = ()

    @property
    def has_errors(self) -> bool:
        """Return True when any validation issue has been captured."""
        return bool(self.issues)


def _coerce_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enable"}


_BOOL_LABELS = {
    "1",
    "0",
    "true",
    "false",
    "yes",
    "no",
    "on",
    "off",
    "enable",
    "disable",
}


def _is_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return value.strip().lower() in _BOOL_LABELS
    return False


def _is_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_identifier_list_or_scalar(value: Any) -> bool:
    if _is_identifier(value):
        return True
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        entries = [str(item).strip() for item in value]
        return bool(entries) and all(entries)
    return False


_VALUE_VALIDATORS: dict[tuple[str, str], tuple[Callable[[Any], bool], str]] = {
    ("plugins", "trusted"): (
        _is_identifier_list_or_scalar,
        "must be a non-empty string or a non-empty sequence of non-empty strings",
    ),
    ("explanations", "factual"): (_is_identifier, "must be a non-empty string"),
    ("explanations", "alternative"): (_is_identifier, "must be a non-empty string"),
    ("explanations", "fast"): (_is_identifier, "must be a non-empty string"),
    ("explanations", "factual_fallbacks"): (
        _is_identifier_list_or_scalar,
        "must be a non-empty string or a non-empty sequence of non-empty strings",
    ),
    ("explanations", "alternative_fallbacks"): (
        _is_identifier_list_or_scalar,
        "must be a non-empty string or a non-empty sequence of non-empty strings",
    ),
    ("explanations", "fast_fallbacks"): (
        _is_identifier_list_or_scalar,
        "must be a non-empty string or a non-empty sequence of non-empty strings",
    ),
    ("intervals", "default"): (_is_identifier, "must be a non-empty string"),
    ("intervals", "fast"): (_is_identifier, "must be a non-empty string"),
    ("intervals", "default_fallbacks"): (
        _is_identifier_list_or_scalar,
        "must be a non-empty string or a non-empty sequence of non-empty strings",
    ),
    ("intervals", "fast_fallbacks"): (
        _is_identifier_list_or_scalar,
        "must be a non-empty string or a non-empty sequence of non-empty strings",
    ),
    ("plots", "style"): (_is_identifier, "must be a non-empty string"),
    ("plots", "fallbacks"): (
        _is_identifier_list_or_scalar,
        "must be a non-empty string or a non-empty sequence of non-empty strings",
    ),
    ("plots", "style_fallbacks"): (
        _is_identifier_list_or_scalar,
        "must be a non-empty string or a non-empty sequence of non-empty strings",
    ),
    ("plots", "renderer"): (_is_identifier, "must be a non-empty string"),
    ("telemetry", "diagnostic_mode"): (_is_bool_like, "must be bool or bool-like string"),
}


class ConfigManager:
    """Authoritative runtime configuration resolver with snapshot semantics."""

    def __init__(
        self,
        *,
        env_snapshot: Mapping[str, str],
        pyproject_snapshot: Mapping[str, Mapping[str, Any]],
        profile_id: str = _PROFILE_ID,
        schema_version: str = _SCHEMA_VERSION,
        strict: bool = True,
    ) -> None:
        self._env_snapshot = dict(env_snapshot)
        self._pyproject_snapshot = {
            section: dict(values) for section, values in pyproject_snapshot.items()
        }
        self._profile_id = profile_id
        self._schema_version = schema_version
        self._strict = strict
        self._source_count = self._compute_source_count(
            env_snapshot=self._env_snapshot,
            pyproject_snapshot=self._pyproject_snapshot,
        )
        self._validation_report = ConfigValidationReport()
        self._validate_sections()

    @classmethod
    def from_sources(cls, *, strict: bool = True) -> ConfigManager:
        """Capture environment and pyproject snapshots once."""
        unknown_env_keys = sorted(
            key for key in os.environ if key.startswith("CE_") and key not in _KNOWN_ENV_KEYS
        )
        if unknown_env_keys:
            message = (
                "Unknown CE_* environment variables detected: "
                + ", ".join(unknown_env_keys)
                + ". They are ignored by ConfigManager."
            )
            _LOGGER.info(message)
            warnings.warn(message, UserWarning, stacklevel=2)

        env_snapshot = {
            key: value
            for key, value in os.environ.items()
            if key in _KNOWN_ENV_KEYS and isinstance(value, str)
        }
        pyproject_snapshot = {
            "plugins": read_pyproject_section(("tool", "calibrated_explanations", "plugins")),
            "explanations": read_pyproject_section(
                ("tool", "calibrated_explanations", "explanations")
            ),
            "intervals": read_pyproject_section(("tool", "calibrated_explanations", "intervals")),
            "plots": read_pyproject_section(("tool", "calibrated_explanations", "plots")),
            "telemetry": read_pyproject_section(("tool", "calibrated_explanations", "telemetry")),
        }
        manager = cls(
            env_snapshot=env_snapshot, pyproject_snapshot=pyproject_snapshot, strict=strict
        )
        manager._emit_config_governance_event(
            event_type="resolve",
            validation_issue_count=0,
        )
        return manager

    @staticmethod
    def _compute_source_count(
        *,
        env_snapshot: Mapping[str, str],
        pyproject_snapshot: Mapping[str, Mapping[str, Any]],
    ) -> int:
        return len(env_snapshot) + sum(len(values) for values in pyproject_snapshot.values())

    def _emit_config_governance_event(
        self,
        *,
        event_type: str,
        validation_issue_count: int,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        # Import lazily to avoid module-import cycles between logging/config/governance.
        from ..governance.events import emit_config_governance_event

        emit_config_governance_event(
            event_type=event_type,
            profile_id=self._profile_id,
            config_schema_version=self._schema_version,
            strict=self._strict,
            source_count=self._source_count,
            validation_issue_count=validation_issue_count,
            details=details,
        )

    def _validate_sections(self) -> None:
        issues: list[ConfigValidationIssue] = []
        for section in self._pyproject_snapshot:
            if section not in _SECTION_SCHEMA:
                issues.append(
                    ConfigValidationIssue(
                        location=f"pyproject.{section}",
                        message=f"Unknown configuration section: {section}",
                    )
                )
                continue

            allowed = set(_SECTION_SCHEMA[section])
            unknown_keys = set(self._pyproject_snapshot.get(section, {}).keys()) - allowed
            if unknown_keys:
                issues.append(
                    ConfigValidationIssue(
                        location=f"pyproject.{section}",
                        message=f"Unknown key(s) in [{section}] configuration: {sorted(unknown_keys)}",
                    )
                )
                continue

            section_values = self._pyproject_snapshot.get(section, {})
            for key, value in section_values.items():
                validator_entry = _VALUE_VALIDATORS.get((section, key))
                if validator_entry is None:
                    continue
                validator, expectation = validator_entry
                if validator(value):
                    continue
                issues.append(
                    ConfigValidationIssue(
                        location=f"pyproject.{section}.{key}",
                        message=f"Invalid value for [{section}].{key}: {value!r}; {expectation}",
                    )
                )

        self._validation_report = ConfigValidationReport(tuple(issues))
        if not issues:
            return
        details = {"location": issues[0].location, "issue_count": len(issues)}
        self._emit_config_governance_event(
            event_type="validation_failure",
            validation_issue_count=len(issues),
            details=details,
        )
        if self._strict:
            raise ConfigurationError(issues[0].message)

        summary = "; ".join(f"{issue.location}: {issue.message}" for issue in issues)
        _LOGGER.info("Config validation issues captured with strict=False: %s", summary)
        warnings.warn(
            f"Config validation issues captured with strict=False: {summary}",
            UserWarning,
            stacklevel=2,
        )

    def validation_report(self) -> ConfigValidationReport:
        """Return the validation report captured during manager construction."""
        return self._validation_report

    def env(self, key: str, default: str | None = None) -> str | None:
        """Return the captured environment value for ``key``."""
        return self._env_snapshot.get(key, default)

    def pyproject_section(self, section: str) -> dict[str, Any]:
        """Return a copy of a captured pyproject section.

        Raises ConfigurationError for unknown sections when strict=True.
        Returns an empty dict when strict=False.
        """
        if section not in self._pyproject_snapshot:
            if self._strict:
                raise ConfigurationError(f"Unknown pyproject section: {section}")
            return {}
        return dict(self._pyproject_snapshot[section])

    def telemetry_diagnostic_mode(self) -> bool:
        """Resolve telemetry diagnostic mode using ADR-034 precedence."""
        env_value = self.env("CE_TELEMETRY_DIAGNOSTIC_MODE")
        if env_value is not None:
            return _coerce_bool(env_value)
        telemetry = self.pyproject_section("telemetry")
        return _coerce_bool(telemetry.get("diagnostic_mode"))

    def export_effective(self) -> ResolvedConfigSnapshot:
        """Export an immutable snapshot of effective config values and sources."""
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}

        # Preserve source snapshots for diagnostics/debugging.
        for key, value in self._env_snapshot.items():
            values[f"env.{key}"] = value
            sources[f"env.{key}"] = "env"
        for section in ("plugins", "explanations", "intervals", "plots", "telemetry"):
            values[f"pyproject.{section}"] = dict(self._pyproject_snapshot.get(section, {}))
            sources[f"pyproject.{section}"] = "pyproject"

        # Export fully resolved effective values with per-key source attribution.
        for key in _KNOWN_ENV_KEYS:
            resolved_value, resolved_source = self._resolve_effective_key(key)
            values[f"effective.{key}"] = resolved_value
            sources[f"effective.{key}"] = resolved_source

        snapshot = ResolvedConfigSnapshot(
            values=MappingProxyType(values),
            sources=MappingProxyType(sources),
            profile_id=self._profile_id,
            schema_version=self._schema_version,
        )
        self._emit_config_governance_event(
            event_type="export",
            validation_issue_count=0,
            details={"diagnostic_only": True},
        )
        return snapshot

    def _resolve_effective_key(
        self,
        key: str,
        *,
        override: Mapping[str, Any] | None = None,
    ) -> tuple[Any, str]:
        """Resolve one effective key following ADR-034 precedence."""
        if override and key in override:
            return override[key], "override"

        env_value = self._env_snapshot.get(key)
        if env_value is not None:
            return env_value, "env"

        section, py_key, default = _RESOLUTION_SPEC.get(key, (None, None, None))
        if section and py_key:
            py_section = self._pyproject_snapshot.get(section, {})
            if py_key in py_section:
                return py_section[py_key], "pyproject"
        return default, "default_profile"
