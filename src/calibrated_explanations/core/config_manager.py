"""Centralized runtime configuration management (ADR-034)."""

from __future__ import annotations

import json
import logging
import os
import threading
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping

from ..utils.exceptions import CalibratedError, ConfigurationError
from .config_helpers import read_pyproject_section

_PROFILE_ID = "default"
_SCHEMA_VERSION = "1"
_PLUGIN_CONFIG_EXPORT_SCHEMA_VERSION = "provisional-1"
_LOGGER = logging.getLogger(__name__)
_PYPROJECT_TOOL_NAMESPACE = ("tool", "calibrated_explanations")
_PLUGIN_CONFIG_SECTION = "plugin_configs"
_PLUGIN_CONFIG_ENV_KEY = "CE_PLUGIN_CONFIG_JSON"
_DEFAULT_KNOWN_ENV_KEYS = (
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
    _PLUGIN_CONFIG_ENV_KEY,
    "CE_TELEMETRY_DIAGNOSTIC_MODE",
    "CE_CACHE",
    "CE_PARALLEL",
    "CE_PARALLEL_MIN_BATCH_SIZE",
    "CE_FEATURE_FILTER",
    "CE_STRICT_OBSERVABILITY",
    "CE_DEBUG_TRUST_INVARIANTS",
    "CI",
    "GITHUB_ACTIONS",
)

_DEFAULT_SECTION_SCHEMA: dict[str, tuple[str, ...]] = {
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
    # Dynamic plugin IDs live under this provisional section. ConfigManager validates
    # only the raw shape here; schema binding happens after plugin trust resolution.
    _PLUGIN_CONFIG_SECTION: (),
    "telemetry": ("diagnostic_mode",),
}

_DEFAULT_RESOLUTION_SPEC: dict[str, tuple[str | None, str | None, Any]] = {
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
    _PLUGIN_CONFIG_ENV_KEY: (None, None, {}),
    "CE_TELEMETRY_DIAGNOSTIC_MODE": ("telemetry", "diagnostic_mode", False),
    "CE_CACHE": (None, None, None),
    "CE_PARALLEL": (None, None, None),
    "CE_PARALLEL_MIN_BATCH_SIZE": (None, None, None),
    "CE_FEATURE_FILTER": (None, None, None),
    "CE_STRICT_OBSERVABILITY": (None, None, False),
    # Sanctioned direct os.getenv read in plugins/_trust.py (routing through ConfigManager
    # risks a circular import via plugins/registry.py). Listed here for governance visibility
    # and export_effective() inclusion only — see ADR-034 §7.
    "CE_DEBUG_TRUST_INVARIANTS": (None, None, None),
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


def _freeze_config_value(value: Any) -> Any:
    """Return a deeply immutable copy of a config value."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_config_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_config_value(item) for item in value), key=repr))
    return value


_SECRET_KEY_FRAGMENTS = (
    "secret",
    "token",
    "password",
    "credential",
    "api_key",
    "apikey",
    "private_key",
    "license_key",
    "access_key",
)


def _is_secret_like_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS)


def _redact_mapping(
    value: Mapping[str, Any],
    *,
    sensitive_keys: Iterable[str] = (),
) -> dict[str, Any]:
    sensitive = {key.lower() for key in sensitive_keys}
    redacted: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if key_text.lower() in sensitive or _is_secret_like_key(key_text):
            redacted[key_text] = "<redacted>"
        elif isinstance(item, Mapping):
            redacted[key_text] = _redact_mapping(item)
        elif isinstance(item, (list, tuple)):
            redacted[key_text] = [
                _redact_mapping(entry) if isinstance(entry, Mapping) else entry for entry in item
            ]
        else:
            redacted[key_text] = item
    return redacted


def _schema_sensitive_keys(schema: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(schema, Mapping):
        return ()
    raw_keys = schema.get("keys", schema.get("properties", {}))
    if not isinstance(raw_keys, Mapping):
        return ()
    return tuple(
        str(key)
        for key, value in raw_keys.items()
        if isinstance(value, Mapping) and bool(value.get("sensitive", False))
    )


def _is_plugin_config_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


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


_DEFAULT_VALUE_VALIDATORS: dict[tuple[str, str], tuple[Callable[[Any], bool], str]] = {
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


@dataclass(frozen=True)
class ConfigSpec:
    """Declarative runtime configuration schema used by ``ConfigManager``.

    Parameters
    ----------
    known_env_keys : tuple of str
        Environment variable names recognized by the manager.
    section_schema : Mapping[str, tuple[str, ...]]
        Supported pyproject.toml sections and allowed keys.
    resolution_spec : Mapping[str, tuple[str | None, str | None, Any]]
        Mapping from environment key to ``(section, key, default)`` resolution metadata.
    value_validators : Mapping[tuple[str, str], tuple[Callable, str]]
        Per-section validation callbacks and expectation text.
    pyproject_tool_namespace : tuple of str
        Root TOML path used when reading pyproject sections.
    """

    known_env_keys: tuple[str, ...]
    section_schema: Mapping[str, tuple[str, ...]]
    resolution_spec: Mapping[str, tuple[str | None, str | None, Any]]
    value_validators: Mapping[tuple[str, str], tuple[Callable[[Any], bool], str]]
    pyproject_tool_namespace: tuple[str, ...] = _PYPROJECT_TOOL_NAMESPACE

    def merged_with(self, other: "ConfigSpec") -> "ConfigSpec":
        """Return a spec extended with entries from ``other``.

        The base spec's ``pyproject_tool_namespace`` is retained intentionally so a
        subclass can choose the namespace it owns and merge in shared keys without
        inheriting another manager's TOML root.
        """
        known_env_keys = tuple(dict.fromkeys((*self.known_env_keys, *other.known_env_keys)))
        section_schema = dict(self.section_schema)
        for section, keys in other.section_schema.items():
            existing = section_schema.get(section, ())
            section_schema[section] = tuple(dict.fromkeys((*existing, *keys)))

        resolution_spec = dict(self.resolution_spec)
        resolution_spec.update(other.resolution_spec)

        value_validators = dict(self.value_validators)
        value_validators.update(other.value_validators)

        return ConfigSpec(
            known_env_keys=known_env_keys,
            section_schema=section_schema,
            resolution_spec=resolution_spec,
            value_validators=value_validators,
            pyproject_tool_namespace=self.pyproject_tool_namespace,
        )


def _build_default_spec() -> ConfigSpec:
    """Build the default CE runtime configuration spec."""
    return ConfigSpec(
        known_env_keys=_DEFAULT_KNOWN_ENV_KEYS,
        section_schema=dict(_DEFAULT_SECTION_SCHEMA),
        resolution_spec=dict(_DEFAULT_RESOLUTION_SPEC),
        value_validators=dict(_DEFAULT_VALUE_VALIDATORS),
        pyproject_tool_namespace=_PYPROJECT_TOOL_NAMESPACE,
    )


class ConfigManager:
    """Authoritative runtime configuration resolver with snapshot semantics."""

    _spec: ClassVar[ConfigSpec] = _build_default_spec()
    _KNOWN_ENV_KEYS: ClassVar[tuple[str, ...]] = _spec.known_env_keys
    _SECTION_SCHEMA: ClassVar[Mapping[str, tuple[str, ...]]] = _spec.section_schema
    _RESOLUTION_SPEC: ClassVar[Mapping[str, tuple[str | None, str | None, Any]]] = (
        _spec.resolution_spec
    )
    _VALUE_VALIDATORS: ClassVar[Mapping[tuple[str, str], tuple[Callable[[Any], bool], str]]] = (
        _spec.value_validators
    )
    _pyproject_tool_namespace: ClassVar[tuple[str, ...]] = _spec.pyproject_tool_namespace

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Keep compatibility class attributes synchronized for subclasses."""
        super().__init_subclass__(**kwargs)
        cls._KNOWN_ENV_KEYS = cls._spec.known_env_keys
        cls._SECTION_SCHEMA = cls._spec.section_schema
        cls._RESOLUTION_SPEC = cls._spec.resolution_spec
        cls._VALUE_VALIDATORS = cls._spec.value_validators
        cls._pyproject_tool_namespace = cls._spec.pyproject_tool_namespace

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
        self._plugin_config_env_snapshot: dict[str, dict[str, Any]] = {}
        self._source_count = self._compute_source_count(
            env_snapshot=self._env_snapshot,
            pyproject_snapshot=self._pyproject_snapshot,
        )
        self._validation_report = ConfigValidationReport()
        self._validate_sections()

    @classmethod
    def from_sources(cls, *, strict: bool = True) -> ConfigManager:
        """Capture environment and pyproject snapshots once."""
        spec = cls._spec
        unknown_env_keys = sorted(
            key for key in os.environ if key.startswith("CE_") and key not in spec.known_env_keys
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
            if key in spec.known_env_keys and isinstance(value, str)
        }
        pyproject_snapshot = {
            section: read_pyproject_section((*spec.pyproject_tool_namespace, section))
            for section in spec.section_schema
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
        spec = type(self)._spec
        for section in self._pyproject_snapshot:
            if section not in spec.section_schema:
                issues.append(
                    ConfigValidationIssue(
                        location=f"pyproject.{section}",
                        message=f"Unknown configuration section: {section}",
                    )
                )
                continue

            if section == _PLUGIN_CONFIG_SECTION:
                for plugin_id, plugin_values in self._pyproject_snapshot.get(section, {}).items():
                    if not isinstance(plugin_id, str) or not plugin_id.strip():
                        issues.append(
                            ConfigValidationIssue(
                                location=f"pyproject.{section}",
                                message="Plugin config IDs must be non-empty strings",
                            )
                        )
                        continue
                    if not _is_plugin_config_mapping(plugin_values):
                        issues.append(
                            ConfigValidationIssue(
                                location=f"pyproject.{section}.{plugin_id}",
                                message="Plugin config values must be mappings with string keys",
                            )
                        )
                continue

            allowed = set(spec.section_schema[section])
            unknown_keys = set(self._pyproject_snapshot.get(section, {}).keys()) - allowed
            if unknown_keys:
                issues.append(
                    ConfigValidationIssue(
                        location=f"pyproject.{section}",
                        message=(
                            f"Unknown key(s) in [{section}] configuration: "
                            f"{sorted(unknown_keys)}"
                        ),
                    )
                )
                continue

            section_values = self._pyproject_snapshot.get(section, {})
            for key, value in section_values.items():
                validator_entry = spec.value_validators.get((section, key))
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

        self._plugin_config_env_snapshot, env_issues = self._parse_plugin_config_env_snapshot()
        issues.extend(env_issues)
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

    def _parse_plugin_config_env_snapshot(
        self,
    ) -> tuple[dict[str, dict[str, Any]], list[ConfigValidationIssue]]:
        raw_value = self._env_snapshot.get(_PLUGIN_CONFIG_ENV_KEY)
        if raw_value is None:
            return {}, []
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            return (
                {},
                [
                    ConfigValidationIssue(
                        location=f"env.{_PLUGIN_CONFIG_ENV_KEY}",
                        message=f"Malformed {_PLUGIN_CONFIG_ENV_KEY}: {exc.msg}",
                    )
                ],
            )
        if not isinstance(parsed, Mapping):
            return (
                {},
                [
                    ConfigValidationIssue(
                        location=f"env.{_PLUGIN_CONFIG_ENV_KEY}",
                        message=f"{_PLUGIN_CONFIG_ENV_KEY} must decode to a mapping",
                    )
                ],
            )

        plugin_configs: dict[str, dict[str, Any]] = {}
        issues: list[ConfigValidationIssue] = []
        for plugin_id, plugin_values in parsed.items():
            if not isinstance(plugin_id, str) or not plugin_id.strip():
                issues.append(
                    ConfigValidationIssue(
                        location=f"env.{_PLUGIN_CONFIG_ENV_KEY}",
                        message="Plugin config IDs must be non-empty strings",
                    )
                )
                continue
            if not _is_plugin_config_mapping(plugin_values):
                issues.append(
                    ConfigValidationIssue(
                        location=f"env.{_PLUGIN_CONFIG_ENV_KEY}.{plugin_id}",
                        message="Plugin config values must be mappings with string keys",
                    )
                )
                continue
            plugin_configs[plugin_id] = dict(plugin_values)
        return plugin_configs, issues

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

    def configured_plugin_ids(self) -> tuple[str, ...]:
        """Return plugin IDs that have raw config in captured sources."""
        pyproject_configs = self._pyproject_snapshot.get(_PLUGIN_CONFIG_SECTION, {})
        configured = {
            *(key for key in pyproject_configs if isinstance(key, str)),
            *self._plugin_config_env_snapshot.keys(),
        }
        return tuple(sorted(configured))

    def plugin_config(
        self,
        plugin_id: str,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Return provisional raw plugin config from the captured snapshot.

        This method intentionally performs no plugin-owned schema validation. Trusted
        registry code must bind and validate the returned values only after plugin
        selection, trust resolution, and metadata availability.
        """
        values, _sources = self._resolve_raw_plugin_config(plugin_id, overrides=overrides)
        return _freeze_config_value(values)

    def plugin_config_sources(
        self,
        plugin_id: str,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> Mapping[str, str]:
        """Return per-key source attribution for provisional raw plugin config."""
        _values, sources = self._resolve_raw_plugin_config(plugin_id, overrides=overrides)
        return MappingProxyType(dict(sources))

    def validate_plugin_config_selection(
        self,
        selected_plugin_ids: Iterable[str],
        *,
        strict: bool = True,
    ) -> ConfigValidationReport:
        """Validate whether configured plugin IDs are selected for this process.

        This provides explicit strict/permissive behavior for unknown or unselected
        raw plugin config without loading plugin metadata.
        """
        selected = {plugin_id for plugin_id in selected_plugin_ids if isinstance(plugin_id, str)}
        unselected = sorted(set(self.configured_plugin_ids()) - selected)
        if not unselected:
            return ConfigValidationReport()
        issue = ConfigValidationIssue(
            location="plugin_configs",
            message="Plugin config provided for unselected plugin(s): " + ", ".join(unselected),
        )
        report = ConfigValidationReport((issue,))
        if strict:
            raise ConfigurationError(issue.message)
        _LOGGER.info("Config validation issues captured with strict=False: %s", issue.message)
        warnings.warn(
            "Config validation issues captured with strict=False: "
            f"{issue.location}: {issue.message}",
            UserWarning,
            stacklevel=2,
        )
        return report

    def _resolve_raw_plugin_config(
        self,
        plugin_id: str,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if not isinstance(plugin_id, str) or not plugin_id.strip():
            raise ConfigurationError("Plugin config ID must be a non-empty string")

        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        pyproject_configs = self._pyproject_snapshot.get(_PLUGIN_CONFIG_SECTION, {})
        pyproject_values = pyproject_configs.get(plugin_id, {})
        if isinstance(pyproject_values, Mapping):
            for key, value in pyproject_values.items():
                values[str(key)] = value
                sources[str(key)] = "pyproject"

        env_values = self._plugin_config_env_snapshot.get(plugin_id, {})
        for key, value in env_values.items():
            values[str(key)] = value
            sources[str(key)] = "env"

        if overrides:
            for key, value in overrides.items():
                values[str(key)] = value
                sources[str(key)] = "override"
        return values, sources

    def telemetry_diagnostic_mode(self) -> bool:
        """Resolve telemetry diagnostic mode using ADR-034 precedence."""
        env_value = self.env("CE_TELEMETRY_DIAGNOSTIC_MODE")
        if env_value is not None:
            return _coerce_bool(env_value)
        telemetry = self.pyproject_section("telemetry")
        return _coerce_bool(telemetry.get("diagnostic_mode"))

    def export_effective(
        self,
        *,
        plugin_config_schemas: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> ResolvedConfigSnapshot:
        """Export an immutable snapshot of effective config values and sources."""
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        plugin_config_schemas = plugin_config_schemas or {}

        # Preserve source snapshots for diagnostics/debugging.
        for key, value in self._env_snapshot.items():
            if key == _PLUGIN_CONFIG_ENV_KEY:
                values[f"env.{key}"] = _redact_mapping(self._plugin_config_env_snapshot)
            elif _is_secret_like_key(key):
                values[f"env.{key}"] = "<redacted>"
            else:
                values[f"env.{key}"] = value
            sources[f"env.{key}"] = "env"
        for section in self._pyproject_snapshot:
            if section == _PLUGIN_CONFIG_SECTION:
                values[f"pyproject.{section}"] = _redact_mapping(
                    self._pyproject_snapshot.get(section, {})
                )
            else:
                values[f"pyproject.{section}"] = dict(self._pyproject_snapshot.get(section, {}))
            sources[f"pyproject.{section}"] = "pyproject"
        values["diagnostic.plugin_config_export_schema_version"] = (
            _PLUGIN_CONFIG_EXPORT_SCHEMA_VERSION
        )
        sources["diagnostic.plugin_config_export_schema_version"] = "default_profile"

        # Export fully resolved effective values with per-key source attribution.
        for key in type(self)._spec.known_env_keys:
            resolved_value, resolved_source = self._resolve_effective_key(key)
            if key == _PLUGIN_CONFIG_ENV_KEY:
                resolved_value = _redact_mapping(self._plugin_config_env_snapshot)
            elif _is_secret_like_key(key):
                resolved_value = "<redacted>"
            values[f"effective.{key}"] = resolved_value
            sources[f"effective.{key}"] = resolved_source

        for plugin_id in self.configured_plugin_ids():
            raw_config, raw_sources = self._resolve_raw_plugin_config(plugin_id)
            values[f"effective.plugin_config.{plugin_id}"] = _redact_mapping(
                raw_config,
                sensitive_keys=_schema_sensitive_keys(plugin_config_schemas.get(plugin_id)),
            )
            unique_sources = set(raw_sources.values())
            sources[f"effective.plugin_config.{plugin_id}"] = (
                "mixed" if len(unique_sources) > 1 else next(iter(unique_sources), "default")
            )
            for key, source in raw_sources.items():
                sources[f"effective.plugin_config.{plugin_id}.{key}"] = source

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

        section, py_key, default = type(self)._spec.resolution_spec.get(key, (None, None, None))
        if section and py_key:
            py_section = self._pyproject_snapshot.get(section, {})
            if py_key in py_section:
                return py_section[py_key], "pyproject"
        return default, "default_profile"


# Module-level alias kept for test imports that reference _KNOWN_ENV_KEYS directly.
_KNOWN_ENV_KEYS = ConfigManager._spec.known_env_keys

_PROCESS_CONFIG_MANAGER: ConfigManager | None = None
_PROCESS_CONFIG_MANAGER_LOCK = threading.RLock()


def get_process_config_manager() -> ConfigManager:
    """Return the process-level ``ConfigManager`` singleton.

    Returns
    -------
    ConfigManager
        The manager initialized for this process, constructing the default
        snapshot on first use.
    """
    global _PROCESS_CONFIG_MANAGER
    if _PROCESS_CONFIG_MANAGER is None:
        with _PROCESS_CONFIG_MANAGER_LOCK:
            if _PROCESS_CONFIG_MANAGER is None:
                _PROCESS_CONFIG_MANAGER = ConfigManager.from_sources()
    return _PROCESS_CONFIG_MANAGER


def init_process_config_manager(
    config_manager: ConfigManager | None = None,
    *,
    strict: bool = True,
) -> ConfigManager:
    """Initialize the process-level ``ConfigManager`` exactly once.

    Parameters
    ----------
    config_manager : ConfigManager, optional
        Explicit manager to install. When omitted, a new manager is constructed
        from the current environment and pyproject.toml snapshot.
    strict : bool, default=True
        Strict validation setting used only when ``config_manager`` is omitted.

    Returns
    -------
    ConfigManager
        The initialized process-level manager.

    Raises
    ------
    CalibratedError
        If the process manager has already been initialized.
    """
    global _PROCESS_CONFIG_MANAGER
    with _PROCESS_CONFIG_MANAGER_LOCK:
        if _PROCESS_CONFIG_MANAGER is not None:
            raise CalibratedError("Process ConfigManager has already been initialized.")
        _PROCESS_CONFIG_MANAGER = (
            config_manager
            if config_manager is not None
            else ConfigManager.from_sources(strict=strict)
        )
        return _PROCESS_CONFIG_MANAGER


def reset_process_config_manager_for_testing() -> None:
    """Reset the process-level ``ConfigManager`` singleton for tests.

    Notes
    -----
    This helper is intentionally test-scoped. Production code should initialize
    once at the process boundary or use ``get_process_config_manager()``.
    """
    global _PROCESS_CONFIG_MANAGER
    with _PROCESS_CONFIG_MANAGER_LOCK:
        _PROCESS_CONFIG_MANAGER = None
