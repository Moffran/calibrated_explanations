"""Base plugin protocols and metadata validation helpers (ADR-006)."""

from __future__ import annotations

import logging
import re
import warnings
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence

from ..utils.exceptions import ValidationError

try:  # Python < 3.10 compatibility
    from typing import TypeAlias
except ImportError:  # pragma: no cover - fallback when TypeAlias is unavailable
    TypeAlias = object  # type: ignore[assignment]

PluginMeta: TypeAlias = Mapping[str, Any]
_RUNTIME_PLUGIN_API_MAJOR = 1
_RUNTIME_PLUGIN_API_MINOR = 0
_RUNTIME_PLUGIN_API_PATCH = 0
_SEMVER_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?$")
_GOVERNANCE_LOGGER = logging.getLogger("calibrated_explanations.governance.plugins")
_CANONICAL_MODALITIES = {
    "tabular",
    "vision",
    "audio",
    "text",
    "multimodal",
}
_MODALITY_ALIASES = {
    "image": "vision",
    "images": "vision",
    "img": "vision",
    "multi-modal": "multimodal",
    "multi_modal": "multimodal",
}
_PROVISIONAL_CONFIG_SCHEMA_VERSION = 1
_ALLOWED_PLOT_KINDS: frozenset[str] = frozenset({"instance", "collection", "global"})
_ALLOWED_PLOT_MODES: frozenset[str] = frozenset({"factual", "alternative", "fast", "any"})
_DEFAULT_PLOT_KINDS: tuple[str, ...] = ("instance", "collection", "global")
_DEFAULT_PLOT_MODES: tuple[str, ...] = ("factual", "alternative", "fast")
_CONFIG_SCHEMA_TYPES = {
    "str",
    "int",
    "float",
    "bool",
    "list",
    "list[str]",
    "mapping",
}


class ExplainerPlugin(Protocol):
    """Protocol describing the minimal explainer plugin contract."""

    plugin_meta: PluginMeta

    def supports(self, model: Any) -> bool:  # pragma: no cover - protocol
        """Return whether this plugin can operate on the supplied model."""
        ...

    def explain(self, model: Any, x: Any, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        """Produce an explanation for ``model`` and feature matrix ``x``."""
        ...


def _ensure_sequence_of_strings(value: Any, *, key: str) -> Sequence[str]:
    """Return *value* as a sequence of strings or raise ``ValidationError``."""
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValidationError(f"plugin_meta[{key!r}] must be a sequence of strings")

    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValidationError(f"plugin_meta[{key!r}] must contain only string values")
        if not item:
            raise ValidationError(f"plugin_meta[{key!r}] must contain non-empty string values")
        result.append(item)
    if not result:
        raise ValidationError(f"plugin_meta[{key!r}] must not be empty")
    return tuple(result)


def _parse_plugin_api_version(raw: Any, *, plugin_name: str | None = None) -> str:
    """Parse and validate plugin API version string."""
    if not isinstance(raw, str) or not raw:
        raise ValidationError("plugin_meta['plugin_api_version'] must be a non-empty string")
    if not _SEMVER_RE.match(raw):
        raise ValidationError(
            "plugin_meta['plugin_api_version'] must match MAJOR.MINOR or MAJOR.MINOR.PATCH"
        )
    major = int(raw.split(".", maxsplit=1)[0])
    if major != _RUNTIME_PLUGIN_API_MAJOR:
        raise ValidationError(
            "plugin_meta['plugin_api_version'] major is incompatible with runtime"
        )

    parts = [int(part) for part in raw.split(".")]
    minor = parts[1]
    patch = parts[2] if len(parts) > 2 else 0
    runtime_minor_patch = (_RUNTIME_PLUGIN_API_MINOR, _RUNTIME_PLUGIN_API_PATCH)
    declared_minor_patch = (minor, patch)
    if declared_minor_patch > runtime_minor_patch:
        warnings.warn(
            "plugin_meta['plugin_api_version'] declares a newer minor/patch than runtime "
            f"{_RUNTIME_PLUGIN_API_MAJOR}.{_RUNTIME_PLUGIN_API_MINOR}.{_RUNTIME_PLUGIN_API_PATCH}; "
            "accepting with forward-compatibility risk.",
            UserWarning,
            stacklevel=3,
        )
        _GOVERNANCE_LOGGER.info(
            "Accepted plugin with newer plugin_api_version minor/patch",
            extra={
                "plugin_name": plugin_name,
                "runtime_plugin_api_version": (
                    f"{_RUNTIME_PLUGIN_API_MAJOR}.{_RUNTIME_PLUGIN_API_MINOR}."
                    f"{_RUNTIME_PLUGIN_API_PATCH}"
                ),
                "declared_plugin_api_version": raw,
                "compatibility_policy": "major-hard/minor-soft",
            },
        )
    return raw


def _normalise_modality(token: str) -> str:
    """Normalize a modality token to canonical form."""
    value = token.strip().lower()
    if not value:
        raise ValidationError("plugin_meta['data_modalities'] must contain non-empty string values")
    if value in _MODALITY_ALIASES:
        value = _MODALITY_ALIASES[value]
    if value in _CANONICAL_MODALITIES:
        return value
    if value.startswith("x-") and len(value) > 2:
        return value
    raise ValidationError("plugin_meta['data_modalities'] contains unsupported modality: " + value)


def _normalise_data_modalities(value: Any) -> Sequence[str]:
    """Validate and normalize data modalities metadata."""
    items = _ensure_sequence_of_strings(value, key="data_modalities")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        modality = _normalise_modality(item)
        if modality not in seen:
            seen.add(modality)
            normalized.append(modality)
    if not normalized:
        raise ValidationError("plugin_meta['data_modalities'] must not be empty")
    return tuple(normalized)


def freeze_plugin_config(value: Any) -> Any:
    """Return a deeply immutable plugin config value."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): freeze_plugin_config(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_plugin_config(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((freeze_plugin_config(item) for item in value), key=repr))
    return value


def thaw_plugin_config(value: Any) -> Any:
    """Recursively convert MappingProxyType to plain dict so the result is picklable.

    This is the inverse of :func:`freeze_plugin_config` for the purpose of
    ``__getstate__`` implementations.  Tuples produced by freeze are walked
    recursively; all other values are returned unchanged.
    """
    if isinstance(value, MappingProxyType):
        return {k: thaw_plugin_config(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(thaw_plugin_config(item) for item in value)
    return value


def _schema_keys(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    keys = schema.get("keys", schema.get("properties", {}))
    if not isinstance(keys, Mapping):
        raise ValidationError("plugin_meta['config_schema']['keys'] must be a mapping")
    return keys


def validate_plugin_config_schema(schema: Mapping[str, Any]) -> None:
    """Validate the provisional plugin config schema shape.

    The schema is a hardened integration surface, not a compatibility-frozen API.
    """
    if not isinstance(schema, Mapping):
        raise ValidationError("plugin_meta['config_schema'] must be a mapping")
    version = schema.get("version", _PROVISIONAL_CONFIG_SCHEMA_VERSION)
    if version != _PROVISIONAL_CONFIG_SCHEMA_VERSION:
        raise ValidationError("plugin_meta['config_schema']['version'] must be 1")
    additional = schema.get("additional_properties", False)
    if not isinstance(additional, bool):
        raise ValidationError(
            "plugin_meta['config_schema']['additional_properties'] must be a boolean"
        )
    for key, entry in _schema_keys(schema).items():
        if not isinstance(key, str) or not key:
            raise ValidationError("plugin_meta['config_schema']['keys'] names must be non-empty")
        if not isinstance(entry, Mapping):
            raise ValidationError(
                f"plugin_meta['config_schema']['keys'][{key!r}] must be a mapping"
            )
        raw_type = entry.get("type")
        if raw_type not in _CONFIG_SCHEMA_TYPES:
            raise ValidationError(
                f"plugin_meta['config_schema']['keys'][{key!r}]['type'] must be one of "
                f"{sorted(_CONFIG_SCHEMA_TYPES)}"
            )
        if "required" in entry and not isinstance(entry["required"], bool):
            raise ValidationError(
                f"plugin_meta['config_schema']['keys'][{key!r}]['required'] must be a boolean"
            )
        if "sensitive" in entry and not isinstance(entry["sensitive"], bool):
            raise ValidationError(
                f"plugin_meta['config_schema']['keys'][{key!r}]['sensitive'] must be a boolean"
            )
        choices = entry.get("choices")
        if choices is not None:
            if isinstance(choices, str) or not isinstance(choices, Iterable):
                raise ValidationError(
                    f"plugin_meta['config_schema']['keys'][{key!r}]['choices'] "
                    "must be a sequence"
                )
            if not tuple(choices):
                raise ValidationError(
                    f"plugin_meta['config_schema']['keys'][{key!r}]['choices'] " "must not be empty"
                )
        if "default" in entry and not _config_value_matches_type(entry["default"], str(raw_type)):
            raise ValidationError(
                f"plugin_meta['config_schema']['keys'][{key!r}]['default'] "
                f"does not match type {raw_type!r}"
            )


def _config_value_matches_type(value: Any, raw_type: str) -> bool:
    if raw_type == "str":
        return isinstance(value, str)
    if raw_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if raw_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if raw_type == "bool":
        return isinstance(value, bool)
    if raw_type == "list":
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    if raw_type == "list[str]":
        return (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and all(isinstance(item, str) for item in value)
        )
    if raw_type == "mapping":
        return isinstance(value, Mapping)
    return False


def validate_plugin_config(
    *,
    plugin_id: str,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate raw config for a selected trusted plugin using its provisional schema."""
    validate_plugin_config_schema(schema)
    if not isinstance(config, Mapping):
        raise ValidationError(f"Plugin config for {plugin_id!r} must be a mapping")

    schema_entries = _schema_keys(schema)
    additional_allowed = bool(schema.get("additional_properties", False))
    unknown_keys = sorted(set(config) - set(schema_entries))
    if unknown_keys and not additional_allowed:
        raise ValidationError(
            f"Plugin config for {plugin_id!r} contains unknown key(s): {unknown_keys}"
        )

    resolved = {
        key: entry["default"]
        for key, entry in schema_entries.items()
        if isinstance(entry, Mapping) and "default" in entry
    }
    resolved.update(dict(config))

    for key, entry in schema_entries.items():
        if not isinstance(entry, Mapping):  # pragma: no cover - guarded by schema validation
            continue
        required = bool(entry.get("required", False))
        if required and key not in resolved:
            raise ValidationError(f"Plugin config for {plugin_id!r} missing required key: {key}")
        if key not in resolved:
            continue
        raw_type = str(entry["type"])
        value = resolved[key]
        if not _config_value_matches_type(value, raw_type):
            raise ValidationError(f"Plugin config for {plugin_id!r} key {key!r} must be {raw_type}")
        choices = entry.get("choices")
        if choices is not None and value not in tuple(choices):
            raise ValidationError(
                f"Plugin config for {plugin_id!r} key {key!r} must be one of {tuple(choices)!r}"
            )
    return freeze_plugin_config(resolved)


def validate_plugin_meta(meta: Dict[str, Any]) -> None:
    """Validate minimal plugin metadata required by ADR-006."""
    if not isinstance(meta, dict):
        raise ValidationError("plugin_meta must be a dict")

    required_scalars = (
        ("schema_version", int),
        ("name", str),
        ("version", str),
        ("provider", str),
    )
    for key, typ in required_scalars:
        if key not in meta:
            raise ValidationError(f"plugin_meta missing required key: {key}")
        value = meta[key]
        if not isinstance(value, typ) or (isinstance(value, str) and not value):
            raise ValidationError(f"plugin_meta[{key!r}] must be a non-empty {typ.__name__}")

    capabilities = meta.get("capabilities")
    if capabilities is None:
        raise ValidationError("plugin_meta missing required key: capabilities")
    meta["capabilities"] = _ensure_sequence_of_strings(capabilities, key="capabilities")

    checksum = meta.get("checksum")
    if checksum is not None and not isinstance(checksum, (str, Mapping)):
        raise ValidationError("plugin_meta['checksum'] must be a string or mapping")

    if "trusted" in meta:
        trusted_value = meta["trusted"]
        if not isinstance(trusted_value, bool):
            raise ValidationError("plugin_meta['trusted'] must be a boolean")
    elif "trust" in meta:
        # Backwards compatibility with earlier drafts that exposed ``trust``.
        trust_value = meta["trust"]
        if isinstance(trust_value, Mapping) and "trusted" in trust_value:
            meta["trusted"] = bool(trust_value["trusted"])
        else:
            meta["trusted"] = bool(trust_value)
    else:
        # Default to False for clarity; registry callers can still override.
        meta["trusted"] = False

    # ADR-033: metadata compatibility defaults for legacy plugins.
    meta["plugin_api_version"] = _parse_plugin_api_version(
        meta.get("plugin_api_version", "1.0"), plugin_name=meta.get("name")
    )
    # ADR-033 §6.2: plugins must declare data_modalities explicitly; default-fallback removed in v0.11.4.
    if "data_modalities" not in meta:
        raise ValidationError(
            "plugin_meta missing required key: data_modalities. "
            "Declare e.g. data_modalities=['tabular'] per ADR-033 §6.2."
        )
    meta["data_modalities"] = _normalise_data_modalities(meta["data_modalities"])
    if "config_schema" in meta:
        validate_plugin_config_schema(meta["config_schema"])

    # supports_guarded: boolean, defaults to False; only valid for explanation plugins.
    if "supports_guarded" in meta:
        sg = meta["supports_guarded"]
        if not isinstance(sg, bool):
            raise ValidationError("plugin_meta['supports_guarded'] must be a boolean")
        capabilities = meta.get("capabilities", ())
        has_explanation_cap = any(
            isinstance(c, str) and c.startswith("explanation:") for c in capabilities
        )
        if sg and not has_explanation_cap:
            raise ValidationError(
                "plugin_meta['supports_guarded']=True is only valid for explanation plugins "
                "(capabilities must include at least one 'explanation:*' entry)"
            )
    else:
        meta["supports_guarded"] = False

    # ADR-037 §4: plot extensions must declare supported plot kinds and modes.
    # Defaults to the full allowed sets for backward compatibility.
    _caps = meta.get("capabilities", ())
    if any(isinstance(c, str) and c.startswith("plot:") for c in _caps):
        if "plot_kinds" in meta:
            kinds = _ensure_sequence_of_strings(meta["plot_kinds"], key="plot_kinds")
            invalid = [k for k in kinds if k not in _ALLOWED_PLOT_KINDS]
            if invalid:
                raise ValidationError(
                    f"plugin_meta['plot_kinds'] contains invalid values: {sorted(invalid)}; "
                    f"allowed: {sorted(_ALLOWED_PLOT_KINDS)}"
                )
            meta["plot_kinds"] = kinds
        else:
            meta["plot_kinds"] = _DEFAULT_PLOT_KINDS

        if "plot_modes" in meta:
            modes = _ensure_sequence_of_strings(meta["plot_modes"], key="plot_modes")
            invalid = [m for m in modes if m not in _ALLOWED_PLOT_MODES]
            if invalid:
                raise ValidationError(
                    f"plugin_meta['plot_modes'] contains invalid values: {sorted(invalid)}; "
                    f"allowed: {sorted(_ALLOWED_PLOT_MODES)}"
                )
            meta["plot_modes"] = modes
        else:
            meta["plot_modes"] = _DEFAULT_PLOT_MODES


__all__ = [
    "ExplainerPlugin",
    "freeze_plugin_config",
    "thaw_plugin_config",
    "validate_plugin_config",
    "validate_plugin_config_schema",
    "validate_plugin_meta",
]
