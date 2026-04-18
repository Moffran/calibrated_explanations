"""Base plugin protocols and metadata validation helpers (ADR-006)."""

from __future__ import annotations

import logging
import re
import warnings
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
    meta["data_modalities"] = _normalise_data_modalities(meta.get("data_modalities", ("tabular",)))


__all__ = ["ExplainerPlugin", "validate_plugin_meta"]
