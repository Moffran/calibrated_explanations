"""Plugin registry (ADR-006 minimal, opt-in).

Explicit, in-process registry for explainer plugins. Users must call
``register`` to add plugins or request discovery through the entry-point loader.
Discovery uses the ``calibrated_explanations.plugins`` entry-point group but
only trusted plugins are instantiated automatically, mirroring ADR-006's
conservative trust model.

This module now also exposes identifier-based registries for explanation,
interval, and plot plugins as outlined by ADR-013/ADR-014/ADR-015. The legacy
list-based helpers remain available for the interim so callers can migrate
incrementally.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import inspect
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .. import __version__ as package_version
from ..core.config_helpers import coerce_string_tuple, read_pyproject_section
from ..logging import ensure_logging_context_filter, logging_context
from ..utils.exceptions import ValidationError
from .. import __version__ as package_version
from .base import ExplainerPlugin, validate_plugin_meta

_REGISTRY: List[ExplainerPlugin] = []

# Minimal trust store: only plugins explicitly trusted by the user are allowed
# to be returned by discovery helpers when trust is requested.
_TRUSTED: List[ExplainerPlugin] = []

_ENTRYPOINT_GROUP = "calibrated_explanations.plugins"
_ENV_TRUST_CACHE: set[str] | None = None
_PYPROJECT_TRUST_CACHE: set[str] | None = None
_WARNED_UNTRUSTED: set[str] = set()
_LAST_DISCOVERY_REPORT: "PluginDiscoveryReport | None" = None

_LOGGER = logging.getLogger("calibrated_explanations.governance.registry")
ensure_logging_context_filter()


def _freeze_meta(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable copy of plugin metadata."""
    return MappingProxyType(dict(meta))


def _normalise_trust(meta: Mapping[str, Any]) -> bool:
    """Extract the trusted-by-default flag from metadata."""
    if "trusted" in meta:
        return bool(meta["trusted"])

    trust = meta.get("trust", False)
    if isinstance(trust, Mapping):
        # Accept a couple of common patterns without committing to a schema
        if "trusted" in trust:
            return bool(trust["trusted"])
        if "default" in trust:
            return bool(trust["default"])
    return bool(trust)


def _env_trusted_names() -> set[str]:
    """Return plugin identifiers trusted via ``CE_TRUST_PLUGIN``."""
    global _ENV_TRUST_CACHE
    if _ENV_TRUST_CACHE is not None:
        return set(_ENV_TRUST_CACHE)

    raw = os.getenv("CE_TRUST_PLUGIN", "")
    names: set[str] = set()
    for chunk in raw.replace(";", ",").split(","):
        name = chunk.strip()
        if name:
            names.add(name)
    _ENV_TRUST_CACHE = names
    return set(names)


def _pyproject_trusted_identifiers() -> set[str]:
    """Return plugin identifiers trusted via pyproject.toml."""
    global _PYPROJECT_TRUST_CACHE
    if _PYPROJECT_TRUST_CACHE is not None:
        return set(_PYPROJECT_TRUST_CACHE)

    config = read_pyproject_section(("tool", "calibrated_explanations", "plugins"))
    trusted = coerce_string_tuple(config.get("trusted"))
    _PYPROJECT_TRUST_CACHE = set(trusted)
    return set(_PYPROJECT_TRUST_CACHE)


def _trusted_identifiers() -> set[str]:
    """Return identifiers explicitly trusted by the operator."""
    return _env_trusted_names() | _pyproject_trusted_identifiers()


def _env_denylist() -> set[str]:
    """Return plugin identifiers blocked via ``CE_DENY_PLUGIN``."""
    raw = os.getenv("CE_DENY_PLUGIN", "")
    names: set[str] = set()
    for chunk in raw.replace(";", ",").split(","):
        name = chunk.strip()
        if name:
            names.add(name)
    return names


def is_identifier_denied(identifier: str) -> bool:
    """Return ``True`` when *identifier* appears in the denylist environment toggle."""
    denied = _env_denylist()
    return identifier in denied


def _should_trust(meta: Mapping[str, Any], *, identifier: str, source: str) -> bool:
    """Return whether *identifier* should be trusted by default."""
    # Builtin plugins are trusted by definition.
    if source == "builtin":
        return True

    # All non-builtin plugins (including entry-point and external/manual
    # registrations) must be explicitly allowed by the operator. The
    # operator-provided allowlist is sourced from CE_TRUST_PLUGIN and
    # the pyproject.toml trusted list.
    trusted_ids = _trusted_identifiers()
    return identifier in trusted_ids


def _update_trust_keys(meta: Dict[str, Any], trusted: bool) -> None:
    """Ensure ``trusted``/``trust`` keys reflect *trusted*."""
    meta["trusted"] = bool(trusted)
    if "trust" in meta and isinstance(meta["trust"], Mapping):
        updated = dict(meta["trust"])
        updated["trusted"] = bool(trusted)
        meta["trust"] = updated
    else:
        meta["trust"] = bool(trusted)


def _propagate_trust_metadata(plugin: Any, meta: Mapping[str, Any]) -> None:
    """Best-effort propagation of trust metadata back onto *plugin*."""
    raw_meta = getattr(plugin, "plugin_meta", None)
    if raw_meta is None:
        return

    trusted_value = meta.get("trusted", bool(meta))
    trust_value = meta.get("trust", trusted_value)

    if isinstance(raw_meta, dict):
        raw_meta["trusted"] = trusted_value
        raw_meta["trust"] = trust_value
        return

    setter = getattr(raw_meta, "__setitem__", None)
    if setter is None:
        return

    try:
        raw_meta["trusted"] = trusted_value
        raw_meta["trust"] = trust_value
    except Exception:  # ADR002_ALLOW: metadata propagation is best-effort.  # pragma: no cover
        # pragma: no cover - defensive
        _LOGGER.debug(
            "Failed to propagate trust metadata for plugin %r",
            plugin,
            exc_info=True,
        )


def _warn_untrusted_plugin(meta: Mapping[str, Any], *, source: str) -> None:
    """Emit a warning about an untrusted plugin once."""
    name = meta.get("name", "<unknown>")
    if name in _WARNED_UNTRUSTED:
        return
    provider = meta.get("provider", "<unknown provider>")
    warnings.warn(
        "Skipping untrusted plugin '%s' from %s discovered via %s. "
        "Set CE_TRUST_PLUGIN, add it to [tool.calibrated_explanations.plugins].trusted, "
        "or call trust_plugin('%s') to load it." % (name, provider, source, name),
        UserWarning,
        stacklevel=3,
    )
    # Governance log for plugin trust decision
    governance_logger = logging.getLogger("calibrated_explanations.governance.plugins")
    ensure_logging_context_filter("calibrated_explanations.governance.plugins")
    with logging_context(plugin_identifier=name):
        governance_logger.info(
            "Plugin trust decision: skipped untrusted plugin",
            extra={
                "provider": provider,
                "source": source,
                "decision": "skipped_untrusted",
            },
        )
    _WARNED_UNTRUSTED.add(name)


# Public testing helpers (temporary; used during Category A remediation).
def normalise_trust(meta: Mapping[str, Any]) -> bool:
    """Public wrapper around internal trust normalisation used by tests."""
    return _normalise_trust(meta)


def env_trusted_names() -> set[str]:
    """Return identifiers trusted via CE_TRUST_PLUGIN (public wrapper)."""
    return _env_trusted_names()


def should_trust(meta: Mapping[str, Any], *, identifier: str, source: str) -> bool:
    """Public wrapper around internal trust decision helper."""
    return _should_trust(meta, identifier=identifier, source=source)


def propagate_trust_metadata(plugin: Any, meta: Mapping[str, Any]) -> None:
    """Public wrapper for best-effort propagation of trust metadata."""
    return _propagate_trust_metadata(plugin, meta)


def update_trust_keys(meta: dict, trusted: bool) -> None:
    """Public wrapper for synchronising trust keys in metadata (testing helper)."""
    return _update_trust_keys(meta, trusted)


def resolve_plugin_module_file(plugin: ExplainerPlugin) -> Path | None:
    """Public wrapper for module file resolution (used in tests)."""
    return _resolve_plugin_module_file(plugin)


def verify_plugin_checksum(plugin: ExplainerPlugin, meta: Mapping[str, Any]) -> None:
    """Public wrapper for checksum verification used by tests."""
    return _verify_plugin_checksum(plugin, meta)


def clear_env_trust_cache() -> None:
    """Clear the environment-derived trust cache (testing helper)."""
    global _ENV_TRUST_CACHE, _PYPROJECT_TRUST_CACHE
    _ENV_TRUST_CACHE = None
    _PYPROJECT_TRUST_CACHE = None


def set_pyproject_trust_cache_for_testing(trusted: Iterable[str] | None) -> None:
    """Set the pyproject trust cache for tests."""
    global _PYPROJECT_TRUST_CACHE
    _PYPROJECT_TRUST_CACHE = None if trusted is None else set(trusted)


def clear_trust_warnings() -> None:
    """Clear the warned-untrusted set (testing helper)."""
    _WARNED_UNTRUSTED.clear()


# Plot/registry accessors for tests (temporary)
def get_entrypoint_group() -> str:
    """Return the entrypoint group used for discovery."""
    return _ENTRYPOINT_GROUP


def plot_styles() -> Dict[str, Any]:
    """Return the internal plot styles mapping (shallow copy)."""
    return dict(_PLOT_STYLES)


def set_plot_style(identifier: str, descriptor: Any) -> None:
    """Set a plot style descriptor in the registry (testing helper)."""
    _PLOT_STYLES[identifier] = descriptor


def clear_plot_styles() -> None:
    """Clear the registered plot styles (testing helper)."""
    _PLOT_STYLES.clear()


def plot_builders() -> Dict[str, Any]:
    """Return the internal plot builders mapping (shallow copy)."""
    return dict(_PLOT_BUILDERS)


def set_plot_builder(identifier: str, descriptor: Any, *, trusted: bool = False) -> None:
    """Set a plot builder descriptor and optionally mark trusted."""
    _PLOT_BUILDERS[identifier] = descriptor
    if trusted:
        _TRUSTED_PLOT_BUILDERS.add(identifier)
    else:
        _TRUSTED_PLOT_BUILDERS.discard(identifier)


def clear_plot_builders() -> None:
    _PLOT_BUILDERS.clear()
    _TRUSTED_PLOT_BUILDERS.clear()


def plot_renderers() -> Dict[str, Any]:
    """Return the internal plot renderers mapping (shallow copy)."""
    return dict(_PLOT_RENDERERS)


def set_plot_renderer(identifier: str, descriptor: Any, *, trusted: bool = False) -> None:
    """Set a plot renderer descriptor and optionally mark trusted."""
    _PLOT_RENDERERS[identifier] = descriptor
    if trusted:
        _TRUSTED_PLOT_RENDERERS.add(identifier)
    else:
        _TRUSTED_PLOT_RENDERERS.discard(identifier)


def clear_plot_renderers() -> None:
    _PLOT_RENDERERS.clear()
    _TRUSTED_PLOT_RENDERERS.clear()


def registry_snapshot() -> Tuple[ExplainerPlugin, ...]:
    """Return a snapshot of the internal registry list for tests."""
    return tuple(_REGISTRY)


def append_to_registry(plugin: ExplainerPlugin) -> None:
    """Append a plugin to the internal registry without validation (test helper)."""
    if plugin not in _REGISTRY:
        _REGISTRY.append(plugin)


def remove_from_registry(plugin: ExplainerPlugin) -> None:
    """Remove a plugin from the internal registry if present."""
    with contextlib.suppress(ValueError):
        _REGISTRY.remove(plugin)


def resolve_plugin_from_name(name: str) -> ExplainerPlugin:
    """Public wrapper resolving a plugin by human-readable name."""
    return _resolve_plugin_from_name(name)


def safe_supports(plugin: ExplainerPlugin, model: Any) -> bool:
    """Public wrapper for safe support-checking used by tests."""
    return _safe_supports(plugin, model)


def warn_untrusted_plugin(meta: Mapping[str, Any], *, source: str) -> None:
    """Public wrapper to emit the single-shot untrusted-plugin warning."""
    return _warn_untrusted_plugin(meta, source=source)


# Additional validation/testing wrappers
def ensure_sequence(
    meta: Mapping[str, Any],
    key: str,
    *,
    allowed: Iterable[str] | None = None,
    allow_empty: bool = False,
) -> Tuple[str, ...]:
    """Public wrapper for sequence validation used by tests."""
    return _ensure_sequence(meta, key, allowed=allowed, allow_empty=allow_empty)


def coerce_string_collection(value: Any, *, key: str | None = None):
    """Public wrapper for coercing string collections."""
    return _coerce_string_collection(value, key=key)


def normalise_dependency_field(
    value: Any, key: str, *, optional: bool = False, allow_empty: bool = False
):
    """Public wrapper for normalising dependency metadata.

    This wrapper accepts the optional parameters used by the internal
    implementation and forwards them through to :func:`_normalise_dependency_field`.
    """
    return _normalise_dependency_field(value, key, optional=optional, allow_empty=allow_empty)


def normalise_tasks(value: Any):
    """Public wrapper for normalising tasks metadata."""
    return _normalise_tasks(value)


def ensure_bool(value: Mapping[str, Any], key: str) -> bool:
    return _ensure_bool(value, key)


def ensure_string(value: Mapping[str, Any], key: str) -> str:
    return _ensure_string(value, key)


def _resolve_plugin_module_file(plugin: ExplainerPlugin) -> Path | None:
    """Attempt to resolve the module file for checksum verification."""
    module_name: str | None
    if inspect.ismodule(plugin):  # pragma: no cover - defensive
        module_name = getattr(plugin, "__name__", None)
    else:
        module_name = getattr(plugin, "__module__", None)
    if not module_name:
        return None
    module = sys.modules.get(module_name)
    if module is None:
        with contextlib.suppress(Exception):
            module = importlib.import_module(module_name)
    if module is None:
        return None
    file = getattr(module, "__file__", None)
    if not file:
        return None
    return Path(file)


def _verify_plugin_checksum(plugin: ExplainerPlugin, meta: Mapping[str, Any]) -> None:
    """Best-effort checksum verification for plugins."""
    checksum = meta.get("checksum")
    if not checksum:
        return

    checksum_value = checksum.get("sha256") if isinstance(checksum, Mapping) else checksum

    if not isinstance(checksum_value, str):
        raise ValidationError(
            "plugin_meta['checksum'] must be a string or mapping with 'sha256'",
            details={
                "param": "checksum",
                "expected_type": "str | Mapping with 'sha256' key",
                "actual_type": type(checksum_value).__name__,
            },
        )

    checksum_value = checksum_value.lower()
    # Use the public resolver to allow tests to monkeypatch via the public API.
    module_file = resolve_plugin_module_file(plugin)
    if module_file is None or not module_file.exists():
        warnings.warn(
            "Cannot verify checksum for plugin '%s'; module file missing."
            % meta.get("name", "<unknown>"),
            UserWarning,
            stacklevel=3,
        )
        return

    try:
        data = module_file.read_bytes()
    except OSError:
        warnings.warn(
            "Cannot read module for checksum verification for plugin '%s'."
            % meta.get("name", "<unknown>"),
            UserWarning,
            stacklevel=3,
        )
        return

    digest = hashlib.sha256(data).hexdigest()
    if digest != checksum_value:
        raise ValidationError(
            f"Checksum mismatch for plugin '{meta.get('name', '<unknown>')}'.",
            details={
                "param": "checksum",
                "expected": checksum_value,
                "actual": digest,
                "plugin": str(meta.get("name", "<unknown>")),
            },
        )


_EXPLANATION_PROTOCOL_VERSION = 1
EXPLANATION_PROTOCOL_VERSION = _EXPLANATION_PROTOCOL_VERSION

_EXPLANATION_MODE_ALIASES = {
    "explanation:factual": "factual",
    "explanation:alternative": "alternative",
    "explanation:fast": "fast",
}

_EXPLANATION_VALID_MODES = {"factual", "alternative", "fast"}


def _ensure_sequence(
    meta: Mapping[str, Any],
    key: str,
    *,
    allowed: Iterable[str] | None = None,
    allow_empty: bool = False,
) -> Tuple[str, ...]:
    """Validate a metadata field is a sequence of strings."""
    if key not in meta:
        raise ValidationError(
            f"plugin_meta missing required key: {key}",
            details={"param": key, "section": "plugin_meta"},
        )
    value = meta[key]
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValidationError(
            f"plugin_meta[{key!r}] must be a sequence of strings",
            details={
                "param": key,
                "expected_type": "Iterable[str]",
                "actual_type": type(value).__name__,
            },
        )

    collected: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValidationError(
                f"plugin_meta[{key!r}] must contain only string values",
                details={
                    "param": key,
                    "invalid_item_type": type(item).__name__,
                    "expected": "str",
                },
            )
        collected.append(item)

    if not collected and not allow_empty:
        raise ValidationError(
            f"plugin_meta[{key!r}] must not be empty",
            details={"param": key, "allow_empty": False},
        )

    if allowed is not None:
        allowed_set = set(allowed)
        unknown = sorted(set(collected) - allowed_set)
        if unknown:
            raise ValidationError(
                f"plugin_meta[{key!r}] has unsupported values: {', '.join(unknown)}",
                details={
                    "param": key,
                    "allowed_values": sorted(allowed_set),
                    "unsupported_values": unknown,
                },
            )

    return tuple(collected)


def _validate_dependencies(meta: Mapping[str, Any]) -> Tuple[str, ...]:
    """Validate dependency metadata as a sequence of identifiers."""
    return _ensure_sequence(meta, "dependencies", allow_empty=True)


def _coerce_string_collection(
    value: Any,
    *,
    key: str,
    allow_empty: bool = False,
) -> Tuple[str, ...]:
    """Coerce *value* to a tuple of strings."""
    if isinstance(value, str):
        result = (value,)
    elif isinstance(value, Iterable):
        collected: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValidationError(
                    f"plugin_meta[{key!r}] must contain only string values",
                    details={
                        "param": key,
                        "invalid_item_type": type(item).__name__,
                        "expected": "str",
                    },
                )
            collected.append(item)
        result = tuple(collected)
    else:
        raise ValidationError(
            f"plugin_meta[{key!r}] must be a string or sequence of strings",
            details={
                "param": key,
                "expected_types": "str | Iterable[str]",
                "actual_type": type(value).__name__,
            },
        )

    if not result and not allow_empty:
        raise ValidationError(
            f"plugin_meta[{key!r}] must not be empty",
            details={"param": key, "allow_empty": False},
        )
    return result


def _normalise_dependency_field(
    meta: Dict[str, Any],
    key: str,
    *,
    optional: bool = False,
    allow_empty: bool = False,
) -> Tuple[str, ...] | None:
    """Validate dependency style metadata fields."""
    if key not in meta:
        if optional:
            return None
        raise ValidationError(
            f"plugin_meta missing required key: {key}",
            details={"param": key, "section": "plugin_meta", "optional": optional},
        )

    value = meta[key]
    normalised = _coerce_string_collection(value, key=key, allow_empty=allow_empty)
    meta[key] = normalised
    return normalised


def _normalise_tasks(meta: Dict[str, Any]) -> Tuple[str, ...]:
    """Validate the tasks field for explanation plugins."""
    allowed_tasks = {"classification", "regression", "both"}
    if "tasks" not in meta:
        raise ValidationError(
            "plugin_meta missing required key: tasks",
            details={"param": "tasks", "section": "plugin_meta"},
        )
    tasks_value = meta["tasks"]
    tasks = _coerce_string_collection(tasks_value, key="tasks")
    unknown = sorted(set(tasks) - allowed_tasks)
    if unknown:
        raise ValidationError(
            "plugin_meta['tasks'] has unsupported values: " + ", ".join(unknown),
            details={
                "param": "tasks",
                "allowed_values": sorted(allowed_tasks),
                "unsupported_values": unknown,
            },
        )
    meta["tasks"] = tasks
    return tasks


def validate_explanation_metadata(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate ADR-015 metadata requirements for explanation plugins."""
    if not isinstance(meta, dict):
        meta = dict(meta)
    schema_version = meta.get("schema_version")
    if isinstance(schema_version, int) and schema_version > _EXPLANATION_PROTOCOL_VERSION:
        raise ValidationError(
            "explanation plugin declares unsupported schema_version "
            f"{schema_version}; runtime supports {_EXPLANATION_PROTOCOL_VERSION}",
            details={
                "param": "schema_version",
                "plugin_declares": schema_version,
                "runtime_supports": _EXPLANATION_PROTOCOL_VERSION,
            },
        )

    allowed_modes = set(_EXPLANATION_VALID_MODES) | set(_EXPLANATION_MODE_ALIASES)
    raw_modes = _ensure_sequence(meta, "modes", allowed=allowed_modes)
    normalised_modes: List[str] = []
    seen: set[str] = set()
    from ..utils import deprecate

    for mode in raw_modes:
        canonical = _EXPLANATION_MODE_ALIASES.get(mode, mode)
        if mode in _EXPLANATION_MODE_ALIASES:
            deprecate(
                "explanation mode alias '" + mode + "' is deprecated; use '" + canonical + "'",
                key=f"mode_alias:{mode}",
                stacklevel=3,
            )
        if canonical not in _EXPLANATION_VALID_MODES:
            raise ValidationError(
                f"plugin_meta['modes'] has unsupported values: {canonical}",
                details={
                    "param": "modes",
                    "allowed_values": sorted(_EXPLANATION_VALID_MODES),
                    "unsupported_value": canonical,
                },
            )
        if canonical not in seen:
            seen.add(canonical)
            normalised_modes.append(canonical)
    if not normalised_modes:
        raise ValidationError(
            "explanation plugin must declare at least one mode",
            details={"param": "modes", "required": True},
        )
    meta["modes"] = tuple(normalised_modes)

    meta["capabilities"] = _ensure_sequence(meta, "capabilities", allow_empty=False)
    meta["dependencies"] = _validate_dependencies(meta)
    _normalise_tasks(meta)
    _normalise_dependency_field(meta, "interval_dependency", optional=True)
    _normalise_dependency_field(meta, "plot_dependency", optional=True)
    _normalise_dependency_field(meta, "fallbacks", optional=True, allow_empty=True)
    # Trust flags can be bool or mapping; ensure the key exists for explicitness
    if "trust" not in meta:
        if "trusted" in meta:
            meta["trust"] = meta["trusted"]
        else:
            raise ValidationError(
                "plugin_meta missing required key: trust",
                details={"param": "trust", "section": "plugin_meta"},
            )
    return meta


def _ensure_bool(meta: Mapping[str, Any], key: str) -> bool:
    """Return *key* from *meta* ensuring it is a boolean."""
    if key not in meta:
        raise ValidationError(
            f"plugin_meta missing required key: {key}",
            details={"param": key, "section": "plugin_meta"},
        )
    value = meta[key]
    if isinstance(value, bool):
        return value

    raise ValidationError(
        f"plugin_meta[{key!r}] must be a boolean",
        details={"param": key, "expected_type": "bool", "actual_type": type(value).__name__},
    )


def _ensure_string(meta: Mapping[str, Any], key: str) -> str:
    """Return *key* from *meta* ensuring it is a string."""
    if key not in meta:
        raise ValidationError(
            f"plugin_meta missing required key: {key}",
            details={"param": key, "section": "plugin_meta"},
        )
    value = meta[key]
    if isinstance(value, str) and value:
        return value

    raise ValidationError(
        f"plugin_meta[{key!r}] must be a non-empty string",
        details={
            "param": key,
            "expected_type": "str",
            "expected_empty": False,
            "actual_type": type(value).__name__,
        },
    )


def validate_interval_metadata(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate ADR-013 metadata requirements for interval plugins."""
    modes = _ensure_sequence(
        meta,
        "modes",
        allowed={"classification", "regression"},
    )
    if not modes:
        raise ValidationError(
            "interval plugin must declare at least one mode",
            details={"param": "modes", "required": True},
        )

    _ensure_sequence(meta, "capabilities", allow_empty=False)
    _validate_dependencies(meta)
    _ensure_bool(meta, "fast_compatible")
    _ensure_bool(meta, "requires_bins")
    _ensure_string(meta, "confidence_source")
    if "legacy_compatible" in meta:
        _ensure_bool(meta, "legacy_compatible")
    # Trust must be explicitly present; interval metadata should not infer it
    # solely from 'trusted' when the 'trust' key is missing.
    if "trust" not in meta:
        raise ValidationError(
            "plugin_meta missing required key: trust",
            details={"param": "trust", "section": "plugin_meta"},
        )
    # Reconcile trust/trusted to be consistent, mutating in place when possible
    declared_trust = _normalise_trust(meta)
    if isinstance(meta, dict):
        _update_trust_keys(meta, declared_trust)  # type: ignore[arg-type]
        return meta
    meta_copy: Dict[str, Any] = dict(meta)
    _update_trust_keys(meta_copy, declared_trust)
    return meta_copy


def validate_plot_builder_metadata(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate ADR-014 metadata requirements for plot builders."""
    _ensure_string(meta, "style")
    _ensure_sequence(meta, "capabilities", allow_empty=False)
    _validate_dependencies(meta)
    _ensure_bool(meta, "legacy_compatible")
    _ensure_sequence(meta, "output_formats", allow_empty=False)
    # Optional default renderer identifier for builders that recommend a renderer
    if "default_renderer" in meta:
        _ensure_string(meta, "default_renderer")
    declared_trust = _normalise_trust(meta)
    if isinstance(meta, dict):
        _update_trust_keys(meta, declared_trust)  # type: ignore[arg-type]
        return meta
    meta_copy: Dict[str, Any] = dict(meta)
    _update_trust_keys(meta_copy, declared_trust)
    return meta_copy


def validate_plot_renderer_metadata(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate ADR-014 metadata requirements for plot renderers."""
    _ensure_sequence(meta, "capabilities", allow_empty=False)
    _validate_dependencies(meta)
    _ensure_sequence(meta, "output_formats", allow_empty=False)
    _ensure_bool(meta, "supports_interactive")
    declared_trust = _normalise_trust(meta)
    if isinstance(meta, dict):
        _update_trust_keys(meta, declared_trust)  # type: ignore[arg-type]
        return meta
    meta_copy: Dict[str, Any] = dict(meta)
    _update_trust_keys(meta_copy, declared_trust)
    return meta_copy


def validate_plot_style_metadata(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate metadata for plot style registrations."""
    _ensure_string(meta, "style")
    builder = _ensure_string(meta, "builder_id")
    renderer = _ensure_string(meta, "renderer_id")
    if builder == renderer:
        # no restriction; ensure they are non-empty strings only
        pass
    fallbacks = meta.get("fallbacks", ())
    if fallbacks:
        if isinstance(fallbacks, str):
            fallbacks = (fallbacks,)
        elif isinstance(fallbacks, Iterable):
            normalised: list[str] = []
            for item in fallbacks:
                if not isinstance(item, str) or not item:
                    raise ValidationError(
                        "plugin_meta['fallbacks'] must contain non-empty strings",
                        details={
                            "param": "fallbacks",
                            "requirement": "non-empty strings",
                            "invalid_item_type": type(item).__name__,
                        },
                    )
                normalised.append(item)
            fallbacks = tuple(normalised)
        else:
            raise ValidationError(
                "plugin_meta['fallbacks'] must be a string or sequence of strings",
                details={
                    "param": "fallbacks",
                    "expected_types": "str | Iterable[str]",
                    "actual_type": type(fallbacks).__name__,
                },
            )
    else:
        fallbacks = ()
    meta = dict(meta)
    meta["fallbacks"] = fallbacks
    if "is_default" in meta:
        meta["is_default"] = _ensure_bool(meta, "is_default")
    else:
        meta["is_default"] = False
    if "legacy_compatible" in meta:
        meta["legacy_compatible"] = _ensure_bool(meta, "legacy_compatible")
    default_for = meta.get("default_for")
    if default_for is not None:
        meta["default_for"] = _coerce_string_collection(
            default_for, key="default_for", allow_empty=True
        )
    else:
        meta["default_for"] = ()
    return meta


@dataclass(frozen=True)
class ExplanationPluginDescriptor:
    """Descriptor for explanation plugins keyed by identifier."""

    identifier: str
    plugin: ExplainerPlugin
    metadata: Mapping[str, Any] = field(repr=False)
    trusted: bool = False
    source: str = "manual"

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        """Freeze the metadata mapping for immutability."""
        object.__setattr__(self, "metadata", _freeze_meta(self.metadata))

    def __getstate__(self):
        """Get state for pickling.

        Returns
        -------
        dict
            The state dictionary.
        """
        # Convert mappingproxy to dict for pickling
        return dict(self.__dict__)


@dataclass(frozen=True)
class IntervalPluginDescriptor:
    """Descriptor for interval calibrator plugins."""

    identifier: str
    plugin: Any
    metadata: Mapping[str, Any] = field(repr=False)
    trusted: bool = False
    source: str = "manual"

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        """Freeze the metadata mapping for immutability."""
        object.__setattr__(self, "metadata", _freeze_meta(self.metadata))

    def __getstate__(self):
        """Get state for pickling.

        Returns
        -------
        dict
            The state dictionary.
        """
        # Convert mappingproxy to dict for pickling
        return dict(self.__dict__)


_EXPLANATION_PLUGINS: Dict[str, ExplanationPluginDescriptor] = {}
_TRUSTED_EXPLANATIONS: set[str] = set()

_INTERVAL_PLUGINS: Dict[str, IntervalPluginDescriptor] = {}
_TRUSTED_INTERVALS: set[str] = set()


@dataclass(frozen=True)
class PlotBuilderDescriptor:
    """Descriptor for plot builders."""

    identifier: str
    builder: Any
    metadata: Mapping[str, Any] = field(repr=False)
    trusted: bool = False
    source: str = "manual"

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        """Freeze the metadata mapping for immutability."""
        object.__setattr__(self, "metadata", _freeze_meta(self.metadata))

    def __getstate__(self):
        """Get state for pickling.

        Returns
        -------
        dict
            The state dictionary.
        """
        # Convert mappingproxy to dict for pickling
        return dict(self.__dict__)


@dataclass(frozen=True)
class PlotRendererDescriptor:
    """Descriptor for plot renderers."""

    identifier: str
    renderer: Any
    metadata: Mapping[str, Any] = field(repr=False)
    trusted: bool = False
    source: str = "manual"

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        """Freeze the metadata mapping for immutability."""
        object.__setattr__(self, "metadata", _freeze_meta(self.metadata))

    def __getstate__(self):
        """Get state for pickling.

        Returns
        -------
        dict
            The state dictionary.
        """
        # Convert mappingproxy to dict for pickling
        return dict(self.__dict__)


@dataclass(frozen=True)
class PlotStyleDescriptor:
    """Descriptor mapping styles to builders and renderers."""

    identifier: str
    metadata: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        """Freeze the metadata mapping for immutability."""
        object.__setattr__(self, "metadata", _freeze_meta(self.metadata))

    def __getstate__(self):
        """Get state for pickling.

        Returns
        -------
        dict
            The state dictionary.
        """
        # Convert mappingproxy to dict for pickling
        return dict(self.__dict__)


_PLOT_BUILDERS: Dict[str, PlotBuilderDescriptor] = {}
_TRUSTED_PLOT_BUILDERS: set[str] = set()

_PLOT_RENDERERS: Dict[str, PlotRendererDescriptor] = {}
_TRUSTED_PLOT_RENDERERS: set[str] = set()

_PLOT_STYLES: Dict[str, PlotStyleDescriptor] = {}


def clear_explanation_plugins() -> None:
    """Clear explanation plugin descriptors (testing helper)."""
    _EXPLANATION_PLUGINS.clear()
    _TRUSTED_EXPLANATIONS.clear()


def clear_interval_plugins() -> None:
    """Clear interval plugin descriptors (testing helper)."""
    _INTERVAL_PLUGINS.clear()
    _TRUSTED_INTERVALS.clear()


def clear_plot_plugins() -> None:
    """Clear plot plugin descriptors (testing helper)."""
    _PLOT_BUILDERS.clear()
    _TRUSTED_PLOT_BUILDERS.clear()
    _PLOT_RENDERERS.clear()
    _TRUSTED_PLOT_RENDERERS.clear()
    _PLOT_STYLES.clear()


def ensure_builtin_plugins() -> None:
    """Re-register in-tree plugins when registries have been cleared."""
    expected_explanations = {
        "core.explanation.factual",
        "core.explanation.alternative",
        "core.explanation.fast",
    }
    expected_intervals = {"core.interval.legacy", "core.interval.fast"}
    expected_plot_builders = {"core.plot.legacy", "core.plot.plot_spec.default"}
    expected_plot_renderers = {"core.plot.legacy", "core.plot.plot_spec.default"}
    expected_plot_styles = {"legacy", "plot_spec.default"}

    missing = any(identifier not in _EXPLANATION_PLUGINS for identifier in expected_explanations)
    missing = missing or any(
        identifier not in _INTERVAL_PLUGINS for identifier in expected_intervals
    )
    missing = missing or any(
        identifier not in _PLOT_BUILDERS for identifier in expected_plot_builders
    )
    missing = missing or any(
        identifier not in _PLOT_RENDERERS for identifier in expected_plot_renderers
    )
    missing = missing or any(identifier not in _PLOT_STYLES for identifier in expected_plot_styles)

    if not missing:
        return

    from . import builtins as _builtins  # Local import avoids circular dependency

    _builtins.register_builtins()


def register_explanation_plugin(
    identifier: str,
    plugin: ExplainerPlugin,
    *,
    metadata: Mapping[str, Any] | None = None,
    source: str = "manual",
) -> ExplanationPluginDescriptor:
    """Register an explanation plugin under the given identifier."""
    with logging_context(plugin_identifier=identifier):
        if not isinstance(identifier, str) or not identifier:
            raise ValidationError(
                "identifier must be a non-empty string",
                details={
                    "param": "identifier",
                    "expected_type": "str",
                    "expected_empty": False,
                    "actual_type": type(identifier).__name__,
                },
            )
        if is_identifier_denied(identifier):
            raise ValidationError(
                f"Plugin '{identifier}' is denied via CE_DENY_PLUGIN",
                details={"param": "identifier", "identifier": identifier},
            )
        raw_meta = metadata or getattr(plugin, "plugin_meta", None)
        if raw_meta is None:
            raise ValidationError(
                "plugin must expose plugin_meta metadata",
                details={"param": "plugin", "required_attribute": "plugin_meta"},
            )
        meta: Dict[str, Any] = dict(raw_meta)
        validate_plugin_meta(meta)
        meta = validate_explanation_metadata(meta)
        trusted = _should_trust(meta, identifier=identifier, source=source)
        _update_trust_keys(meta, trusted)
        _verify_plugin_checksum(plugin, meta)
        if "checksum" in meta:
            trusted = True
            _update_trust_keys(meta, trusted)
        if isinstance(raw_meta, dict):
            raw_meta["trusted"] = meta["trusted"]
            raw_meta["trust"] = meta["trust"]

        descriptor = ExplanationPluginDescriptor(
            identifier=identifier,
            plugin=plugin,
            metadata=meta,
            trusted=trusted,
            source=source,
        )
        _EXPLANATION_PLUGINS[identifier] = descriptor
        if trusted:
            _TRUSTED_EXPLANATIONS.add(identifier)
        else:
            _TRUSTED_EXPLANATIONS.discard(identifier)

        # Maintain backwards compatibility with the legacy list registry.
        register(plugin, source=source, identifier=identifier)
        if trusted:
            trust_plugin(plugin)

        return descriptor


def find_explanation_descriptor(identifier: str) -> ExplanationPluginDescriptor | None:
    """Return the explanation plugin descriptor for *identifier* if present."""
    return _EXPLANATION_PLUGINS.get(identifier)


def find_explanation_plugin(identifier: str) -> ExplainerPlugin | None:
    """Return the explanation plugin instance for *identifier* if present."""
    descriptor = find_explanation_descriptor(identifier)
    return descriptor.plugin if descriptor else None


def find_explanation_plugin_trusted(identifier: str) -> ExplainerPlugin | None:
    """Return the trusted explanation plugin instance for *identifier* if any."""
    descriptor = find_explanation_descriptor(identifier)
    if descriptor and descriptor.trusted:
        return descriptor.plugin
    return None


def register_interval_plugin(
    identifier: str,
    plugin: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    source: str = "manual",
) -> IntervalPluginDescriptor:
    """Register an interval plugin descriptor."""
    with logging_context(plugin_identifier=identifier):
        if not isinstance(identifier, str) or not identifier:
            raise ValidationError(
                "identifier must be a non-empty string",
                details={
                    "param": "identifier",
                    "expected_type": "str",
                    "expected_empty": False,
                    "actual_type": type(identifier).__name__,
                },
            )
        if is_identifier_denied(identifier):
            raise ValidationError(
                f"Plugin '{identifier}' is denied via CE_DENY_PLUGIN",
                details={"param": "identifier", "identifier": identifier},
            )
        raw_meta = metadata or getattr(plugin, "plugin_meta", None)
        if raw_meta is None:
            raise ValidationError(
                "plugin must expose plugin_meta metadata",
                details={"param": "plugin", "required_attribute": "plugin_meta"},
            )
        meta: Dict[str, Any] = dict(raw_meta)
        validate_plugin_meta(meta)
        validate_interval_metadata(meta)
        trusted = _should_trust(meta, identifier=identifier, source=source)
        _update_trust_keys(meta, trusted)
        _verify_plugin_checksum(plugin, meta)
        if isinstance(raw_meta, dict):
            raw_meta["trusted"] = meta["trusted"]
            raw_meta["trust"] = meta["trust"]

        descriptor = IntervalPluginDescriptor(
            identifier=identifier,
            plugin=plugin,
            metadata=meta,
            trusted=trusted,
            source=source,
        )
        _INTERVAL_PLUGINS[identifier] = descriptor
        if trusted:
            _TRUSTED_INTERVALS.add(identifier)
        else:
            _TRUSTED_INTERVALS.discard(identifier)
        return descriptor


def find_interval_descriptor(identifier: str) -> IntervalPluginDescriptor | None:
    """Return the descriptor for an interval plugin by identifier."""
    return _INTERVAL_PLUGINS.get(identifier)


def find_interval_plugin(identifier: str) -> Any | None:
    """Return the interval plugin instance for *identifier* if registered."""
    descriptor = find_interval_descriptor(identifier)
    return descriptor.plugin if descriptor else None


def find_interval_plugin_trusted(identifier: str) -> Any | None:
    """Return the trusted interval plugin instance when available."""
    descriptor = find_interval_descriptor(identifier)
    if descriptor and descriptor.trusted:
        return descriptor.plugin
    return None


def register_plot_builder(
    identifier: str,
    builder: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    source: str = "manual",
) -> PlotBuilderDescriptor:
    """Register a plot builder under *identifier*."""
    if not isinstance(identifier, str) or not identifier:
        raise ValidationError(
            "identifier must be a non-empty string",
            details={
                "param": "identifier",
                "expected_type": "str",
                "expected_empty": False,
                "actual_type": type(identifier).__name__,
            },
        )
    if is_identifier_denied(identifier):
        raise ValidationError(
            f"Plugin '{identifier}' is denied via CE_DENY_PLUGIN",
            details={"param": "identifier", "identifier": identifier},
        )
    raw_meta = metadata or getattr(builder, "plugin_meta", None)
    if raw_meta is None:
        raise ValidationError(
            "builder must expose plugin_meta metadata",
            details={"param": "builder", "required_attribute": "plugin_meta"},
        )
    meta: Dict[str, Any] = dict(raw_meta)
    validate_plugin_meta(meta)
    validate_plot_builder_metadata(meta)
    trusted = _should_trust(meta, identifier=identifier, source=source)
    _update_trust_keys(meta, trusted)
    _verify_plugin_checksum(builder, meta)
    if isinstance(raw_meta, dict):
        raw_meta["trusted"] = meta["trusted"]
        raw_meta["trust"] = meta["trust"]

    descriptor = PlotBuilderDescriptor(
        identifier=identifier,
        builder=builder,
        metadata=meta,
        trusted=trusted,
        source=source,
    )
    _PLOT_BUILDERS[identifier] = descriptor
    if trusted:
        _TRUSTED_PLOT_BUILDERS.add(identifier)
    else:
        _TRUSTED_PLOT_BUILDERS.discard(identifier)
    return descriptor


def register_plot_renderer(
    identifier: str,
    renderer: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    source: str = "manual",
) -> PlotRendererDescriptor:
    """Register a plot renderer under *identifier*."""
    if not isinstance(identifier, str) or not identifier:
        raise ValidationError(
            "identifier must be a non-empty string",
            details={
                "param": "identifier",
                "expected_type": "str",
                "expected_empty": False,
                "actual_type": type(identifier).__name__,
            },
        )
    if is_identifier_denied(identifier):
        raise ValidationError(
            f"Plugin '{identifier}' is denied via CE_DENY_PLUGIN",
            details={"param": "identifier", "identifier": identifier},
        )
    raw_meta = metadata or getattr(renderer, "plugin_meta", None)
    if raw_meta is None:
        raise ValidationError(
            "renderer must expose plugin_meta metadata",
            details={"param": "renderer", "required_attribute": "plugin_meta"},
        )
    meta: Dict[str, Any] = dict(raw_meta)
    validate_plugin_meta(meta)
    validate_plot_renderer_metadata(meta)
    trusted = _should_trust(meta, identifier=identifier, source=source)
    _update_trust_keys(meta, trusted)
    _verify_plugin_checksum(renderer, meta)
    if isinstance(raw_meta, dict):
        raw_meta["trusted"] = meta["trusted"]
        raw_meta["trust"] = meta["trust"]

    descriptor = PlotRendererDescriptor(
        identifier=identifier,
        renderer=renderer,
        metadata=meta,
        trusted=trusted,
        source=source,
    )
    _PLOT_RENDERERS[identifier] = descriptor
    if trusted:
        _TRUSTED_PLOT_RENDERERS.add(identifier)
    else:
        _TRUSTED_PLOT_RENDERERS.discard(identifier)
    return descriptor


def register_plot_style(
    identifier: str,
    *,
    metadata: Mapping[str, Any],
) -> PlotStyleDescriptor:
    """Register a style entry that maps to builder and renderer identifiers."""
    if not isinstance(identifier, str) or not identifier:
        raise ValidationError(
            "identifier must be a non-empty string",
            details={
                "param": "identifier",
                "expected_type": "str",
                "expected_empty": False,
                "actual_type": type(identifier).__name__,
            },
        )
    if metadata is None:
        raise ValidationError(
            "metadata is required for style registration",
            details={"param": "metadata", "required": True},
        )
    meta: Dict[str, Any] = dict(metadata)
    validate_plot_style_metadata(meta)
    if meta.get("style") != identifier:
        meta.setdefault("style", identifier)
    descriptor = PlotStyleDescriptor(identifier=identifier, metadata=meta)
    _PLOT_STYLES[identifier] = descriptor
    return descriptor


def find_plot_builder_descriptor(
    identifier: str,
) -> PlotBuilderDescriptor | None:
    """Return the builder descriptor for *identifier* if present."""
    return _PLOT_BUILDERS.get(identifier)


def find_plot_builder(identifier: str) -> Any | None:
    """Return the registered plot builder for *identifier* if any."""
    descriptor = find_plot_builder_descriptor(identifier)
    return descriptor.builder if descriptor else None


def find_plot_renderer_descriptor(
    identifier: str,
) -> PlotRendererDescriptor | None:
    """Return the renderer descriptor for *identifier* if present."""
    return _PLOT_RENDERERS.get(identifier)


def find_plot_renderer(identifier: str) -> Any | None:
    """Return the registered plot renderer for *identifier* if any."""
    descriptor = find_plot_renderer_descriptor(identifier)
    return descriptor.renderer if descriptor else None


def find_plot_renderer_trusted(identifier: str) -> PlotRendererDescriptor | None:
    """Return the renderer descriptor for *identifier* if it is trusted."""
    if identifier in _TRUSTED_PLOT_RENDERERS:
        return _PLOT_RENDERERS.get(identifier)
    return None


def find_plot_style_descriptor(identifier: str) -> PlotStyleDescriptor | None:
    """Return the style descriptor for *identifier* if present."""
    return _PLOT_STYLES.get(identifier)


def find_plot_plugin(identifier: str) -> Any | None:
    """Return a combined plot plugin for the given style identifier."""
    style_descriptor = find_plot_style_descriptor(identifier)
    if style_descriptor is None:
        return None

    builder_id = style_descriptor.metadata.get("builder_id")
    renderer_id = style_descriptor.metadata.get("renderer_id")

    if not builder_id or not renderer_id:
        return None

    builder = find_plot_builder(builder_id)
    renderer = find_plot_renderer(renderer_id)

    if builder is None or renderer is None:
        return None

    # Use the named wrapper class to combine builder and renderer.
    from .plots import CombinedPlotPlugin

    return CombinedPlotPlugin(builder, renderer)


def find_plot_plugin_trusted(identifier: str) -> Any | None:
    """Return a trusted combined plot plugin for the given style identifier."""
    style_descriptor = find_plot_style_descriptor(identifier)
    if style_descriptor is None:
        return None

    builder_id = style_descriptor.metadata.get("builder_id")
    renderer_id = style_descriptor.metadata.get("renderer_id")

    if not builder_id or not renderer_id:
        return None

    # Check if both builder and renderer are trusted
    builder_descriptor = find_plot_builder_descriptor(builder_id)
    renderer_descriptor = find_plot_renderer_descriptor(renderer_id)

    if (
        builder_descriptor is None
        or not builder_descriptor.trusted
        or renderer_descriptor is None
        or not renderer_descriptor.trusted
    ):
        return None

    builder = builder_descriptor.builder
    renderer = renderer_descriptor.renderer

    from .plots import CombinedPlotPlugin

    return CombinedPlotPlugin(builder, renderer)


def list_plot_builder_descriptors(
    *, trusted_only: bool = False
) -> Tuple[PlotBuilderDescriptor, ...]:
    """Return registered plot builder descriptors."""
    ensure_builtin_plugins()
    return _list_descriptors(
        _PLOT_BUILDERS,
        trusted_only,
        _TRUSTED_PLOT_BUILDERS,
    )


def list_plot_renderer_descriptors(
    *, trusted_only: bool = False
) -> Tuple[PlotRendererDescriptor, ...]:
    """Return registered plot renderer descriptors."""
    ensure_builtin_plugins()
    return _list_descriptors(
        _PLOT_RENDERERS,
        trusted_only,
        _TRUSTED_PLOT_RENDERERS,
    )


def list_plot_style_descriptors() -> Tuple[PlotStyleDescriptor, ...]:
    """Return registered plot style descriptors."""
    ensure_builtin_plugins()
    identifiers = sorted(_PLOT_STYLES.keys())
    return tuple(_PLOT_STYLES[identifier] for identifier in identifiers)


def list_plot_descriptors(include_untrusted=True):
    """Return registered plot style descriptors."""
    ensure_builtin_plugins()
    return list(_PLOT_STYLES.values())


@dataclass(frozen=True)
class PluginDiscoveryRecord:
    """Record capturing the outcome for a discovered plugin."""

    identifier: str
    provider: str | None
    source: str
    metadata: Mapping[str, Any] | None = field(default=None, repr=False)
    details: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __getstate__(self):
        """Get state for pickling.

        Returns
        -------
        dict
            The state dictionary.
        """
        # Convert mappingproxy to dict for pickling
        return dict(self.__dict__)


@dataclass
class PluginDiscoveryReport:
    """Summarize plugin discovery decisions for diagnostics."""

    skipped_untrusted: list[PluginDiscoveryRecord] = field(default_factory=list)
    skipped_denied: list[PluginDiscoveryRecord] = field(default_factory=list)
    checksum_failures: list[PluginDiscoveryRecord] = field(default_factory=list)
    accepted: list[PluginDiscoveryRecord] = field(default_factory=list)


def get_last_discovery_report() -> PluginDiscoveryReport | None:
    """Return the most recent discovery report, if any."""
    return _LAST_DISCOVERY_REPORT


def get_discovery_report() -> PluginDiscoveryReport:
    """Return a thread-safe snapshot of the most recent discovery report.

    This returns an independent copy so callers can inspect the report
    without risk of concurrent mutation by later discovery runs.
    """
    if _LAST_DISCOVERY_REPORT is None:
        return PluginDiscoveryReport()

    # Create a shallow copy of each list to create a snapshot.
    return PluginDiscoveryReport(
        skipped_untrusted=list(_LAST_DISCOVERY_REPORT.skipped_untrusted),
        skipped_denied=list(_LAST_DISCOVERY_REPORT.skipped_denied),
        checksum_failures=list(_LAST_DISCOVERY_REPORT.checksum_failures),
        accepted=list(_LAST_DISCOVERY_REPORT.accepted),
    )


def load_entrypoint_plugins(*, include_untrusted: bool = False) -> Tuple[ExplainerPlugin, ...]:
    """Discover plugins advertised via entry points."""
    ensure_logging_context_filter()
    loaded: list[ExplainerPlugin] = []
    report = PluginDiscoveryReport()
    global _LAST_DISCOVERY_REPORT
    try:
        entry_points = importlib_metadata.entry_points()
    except (
        Exception
    ) as exc:  # ADR002_ALLOW: entrypoint discovery is best-effort.  # pragma: no cover
        # pragma: no cover - defensive
        warnings.warn(
            f"Failed to enumerate plugin entry points: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return ()

    try:
        candidates = entry_points.select(group=_ENTRYPOINT_GROUP)
    except AttributeError:  # pragma: no cover - Python <3.10 fallback
        candidates = [ep for ep in entry_points if getattr(ep, "group", None) == _ENTRYPOINT_GROUP]

    for entry_point in candidates:
        identifier = (
            f"{entry_point.module}:{entry_point.attr}"
            if getattr(entry_point, "attr", None)
            else entry_point.name
        )
        provider = None
        dist = getattr(entry_point, "dist", None)
        if dist is not None:
            provider = getattr(dist, "name", None) or str(dist)
        if is_identifier_denied(identifier):
            report.skipped_denied.append(
                PluginDiscoveryRecord(
                    identifier=identifier,
                    provider=provider,
                    source="entrypoint",
                    details={"deny_source": "CE_DENY_PLUGIN"},
                )
            )
            warnings.warn(
                f"Skipping denied plugin {identifier!r} from {provider or 'unknown provider'} "
                "due to CE_DENY_PLUGIN.",
                UserWarning,
                stacklevel=2,
            )
            continue
        plugin = None
        try:
            plugin = entry_point.load()
        except (
            Exception
        ) as exc:  # ADR002_ALLOW: keep discovery resilient to plugin failures.  # pragma: no cover
            # Attempt best-effort alternative loaders that some test harnesses
            # or legacy entrypoint shims may provide (e.g. attributes named
            # '_loader' or 'loader'). If those exist and are callable, use
            # them before giving up.
            alt_loader = getattr(entry_point, "_loader", None) or getattr(
                entry_point, "loader", None
            )
            if callable(alt_loader):
                try:
                    plugin = alt_loader()
                except Exception as exc_alt:  # adr002_allow  # pragma: no cover - best-effort
                    warnings.warn(
                        f"Failed to load plugin entry point {identifier!r}: {exc_alt}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
            else:
                warnings.warn(
                    f"Failed to load plugin entry point {identifier!r}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
        raw_meta = getattr(plugin, "plugin_meta", None)
        if raw_meta is None:
            warnings.warn(
                f"Plugin {identifier!r} does not expose plugin_meta; skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        meta: Dict[str, Any] = dict(raw_meta)
        try:
            validate_plugin_meta(meta)
        except (
            ValueError,
            ValidationError,
        ) as exc:  # ADR002_ALLOW: warn and skip invalid metadata.  # pragma: no cover
            warnings.warn(
                f"Invalid metadata for plugin {identifier!r}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue

        declared_trust = _normalise_trust(meta)
        meta_name = meta.get("name")
        if isinstance(meta_name, str) and is_identifier_denied(meta_name):
            report.skipped_denied.append(
                PluginDiscoveryRecord(
                    identifier=identifier,
                    provider=meta.get("provider", provider),
                    source="entrypoint",
                    metadata=meta,
                    details={"deny_source": "CE_DENY_PLUGIN", "denied_identifier": meta_name},
                )
            )
            warnings.warn(
                f"Skipping denied plugin {identifier!r} ({meta_name!r}) "
                f"from {meta.get('provider', provider) or 'unknown provider'} "
                "due to CE_DENY_PLUGIN.",
                UserWarning,
                stacklevel=2,
            )
            # Governance log for plugin deny decision
            governance_logger = logging.getLogger("calibrated_explanations.governance.plugins")
            ensure_logging_context_filter("calibrated_explanations.governance.plugins")
            with logging_context(plugin_identifier=meta_name):
                governance_logger.info(
                    "Plugin trust decision: skipped denied plugin",
                    extra={
                        "provider": meta.get("provider", provider),
                        "source": "entrypoint",
                        "decision": "skipped_denied",
                        "deny_source": "CE_DENY_PLUGIN",
                    },
                )
            continue

        trusted = _should_trust(meta, identifier=identifier, source="entrypoint")
        _update_trust_keys(meta, trusted)

        if not trusted and not include_untrusted:
            _warn_untrusted_plugin(meta, source="entry point")
            report.skipped_untrusted.append(
                PluginDiscoveryRecord(
                    identifier=identifier,
                    provider=meta.get("provider", provider),
                    source="entrypoint",
                    metadata=meta,
                    details={"declared_trusted": declared_trust},
                )
            )
            continue

        try:
            _verify_plugin_checksum(plugin, meta)
        except ValidationError as exc:
            report.checksum_failures.append(
                PluginDiscoveryRecord(
                    identifier=identifier,
                    provider=meta.get("provider", provider),
                    source="entrypoint",
                    metadata=meta,
                    details={"error": str(exc)},
                )
            )
            warnings.warn(
                f"Checksum validation failed for plugin {identifier!r}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue
        register(plugin, source="entrypoint", identifier=identifier)
        if trusted:
            trust_plugin(plugin)
        loaded.append(plugin)
        report.accepted.append(
            PluginDiscoveryRecord(
                identifier=identifier,
                provider=meta.get("provider", provider),
                source="entrypoint",
                metadata=meta,
                details={"trusted": trusted},
            )
        )

    _LAST_DISCOVERY_REPORT = report
    return tuple(loaded)


def register_plot_plugin(
    identifier: str,
    plugin: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    source: str = "manual",
) -> PlotBuilderDescriptor:
    """Compatibility shim registering *plugin* as both builder and renderer."""
    from ..utils import deprecate

    deprecate(
        "register_plot_plugin is deprecated; use register_plot_builder/register_plot_renderer",
        key="register_plot_plugin",
        stacklevel=3,
    )
    descriptor = register_plot_builder(identifier, plugin, metadata=metadata, source=source)
    register_plot_renderer(identifier, plugin, metadata=metadata, source=source)
    register_plot_style(
        identifier,
        metadata={
            "style": identifier,
            "builder_id": identifier,
            "renderer_id": identifier,
            "fallbacks": (),
        },
    )
    return descriptor


def _list_descriptors(
    store: Dict[str, Any],
    trusted_only: bool,
    trusted_set: set[str],
    include_untrusted: bool = False,
) -> Tuple[Any, ...]:
    """Return descriptors from *store* with optional trust filtering."""
    if trusted_only:
        identifiers = sorted(identifier for identifier in trusted_set if identifier in store)
    else:
        identifiers = sorted(store.keys())
    return tuple(store[identifier] for identifier in identifiers)


def list_explanation_descriptors(
    *, trusted_only: bool = False, include_untrusted: bool = False
) -> Tuple[ExplanationPluginDescriptor, ...]:
    """Return registered explanation plugin descriptors."""
    ensure_builtin_plugins()
    return _list_descriptors(
        _EXPLANATION_PLUGINS,
        trusted_only,
        _TRUSTED_EXPLANATIONS,
    )


def list_interval_descriptors(
    *, trusted_only: bool = False
) -> Tuple[IntervalPluginDescriptor, ...]:
    """Return registered interval plugin descriptors."""
    ensure_builtin_plugins()
    return _list_descriptors(_INTERVAL_PLUGINS, trusted_only, _TRUSTED_INTERVALS)


def _refresh_descriptor_trust(identifier: str, *, trusted: bool) -> ExplanationPluginDescriptor:
    """Return descriptor with updated trust metadata stored in registries."""
    descriptor = find_explanation_descriptor(identifier)
    if descriptor is None:
        raise KeyError(f"Explanation plugin '{identifier}' is not registered")
    updated_meta = dict(descriptor.metadata)
    _update_trust_keys(updated_meta, trusted)
    updated = ExplanationPluginDescriptor(
        identifier=descriptor.identifier,
        plugin=descriptor.plugin,
        metadata=updated_meta,
        trusted=trusted,
        source=descriptor.source,
    )
    _EXPLANATION_PLUGINS[identifier] = updated
    if trusted:
        _TRUSTED_EXPLANATIONS.add(identifier)
    else:
        _TRUSTED_EXPLANATIONS.discard(identifier)
    return updated


def mark_explanation_trusted(identifier: str) -> ExplanationPluginDescriptor:
    """Mark the explanation plugin *identifier* as trusted."""
    descriptor = _refresh_descriptor_trust(identifier, trusted=True)
    trust_plugin(descriptor.plugin)
    return descriptor


def mark_explanation_untrusted(identifier: str) -> ExplanationPluginDescriptor:
    """Remove the explanation plugin *identifier* from the trusted set."""
    descriptor = _refresh_descriptor_trust(identifier, trusted=False)
    untrust_plugin(descriptor.plugin)
    return descriptor


def _refresh_interval_descriptor_trust(
    identifier: str, *, trusted: bool
) -> IntervalPluginDescriptor:
    """Return interval descriptor with updated trust state."""
    descriptor = find_interval_descriptor(identifier)
    if descriptor is None:
        raise KeyError(f"Interval plugin '{identifier}' is not registered")

    updated_meta = dict(descriptor.metadata)
    _update_trust_keys(updated_meta, trusted)

    updated = IntervalPluginDescriptor(
        identifier=descriptor.identifier,
        plugin=descriptor.plugin,
        metadata=updated_meta,
        trusted=trusted,
        source=descriptor.source,
    )

    _INTERVAL_PLUGINS[identifier] = updated
    if trusted:
        _TRUSTED_INTERVALS.add(identifier)
    else:
        _TRUSTED_INTERVALS.discard(identifier)

    _propagate_trust_metadata(descriptor.plugin, updated_meta)
    return updated


def mark_interval_trusted(identifier: str) -> IntervalPluginDescriptor:
    """Mark the interval plugin *identifier* as trusted."""
    return _refresh_interval_descriptor_trust(identifier, trusted=True)


def mark_interval_untrusted(identifier: str) -> IntervalPluginDescriptor:
    """Remove the interval plugin *identifier* from the trusted set."""
    return _refresh_interval_descriptor_trust(identifier, trusted=False)


def _refresh_plot_builder_trust(identifier: str, *, trusted: bool) -> PlotBuilderDescriptor:
    """Return builder descriptor with updated trust metadata."""
    descriptor = find_plot_builder_descriptor(identifier)
    if descriptor is None:
        raise KeyError(f"Plot builder '{identifier}' is not registered")

    updated_meta = dict(descriptor.metadata)
    _update_trust_keys(updated_meta, trusted)

    updated = PlotBuilderDescriptor(
        identifier=descriptor.identifier,
        builder=descriptor.builder,
        metadata=updated_meta,
        trusted=trusted,
        source=descriptor.source,
    )

    _PLOT_BUILDERS[identifier] = updated
    if trusted:
        _TRUSTED_PLOT_BUILDERS.add(identifier)
    else:
        _TRUSTED_PLOT_BUILDERS.discard(identifier)

    _propagate_trust_metadata(descriptor.builder, updated_meta)
    return updated


def mark_plot_builder_trusted(identifier: str) -> PlotBuilderDescriptor:
    """Mark the plot builder *identifier* as trusted."""
    return _refresh_plot_builder_trust(identifier, trusted=True)


def mark_plot_builder_untrusted(identifier: str) -> PlotBuilderDescriptor:
    """Remove the plot builder *identifier* from the trusted set."""
    return _refresh_plot_builder_trust(identifier, trusted=False)


def _refresh_plot_renderer_trust(identifier: str, *, trusted: bool) -> PlotRendererDescriptor:
    """Return renderer descriptor with updated trust metadata."""
    descriptor = find_plot_renderer_descriptor(identifier)
    if descriptor is None:
        raise KeyError(f"Plot renderer '{identifier}' is not registered")

    updated_meta = dict(descriptor.metadata)
    _update_trust_keys(updated_meta, trusted)

    updated = PlotRendererDescriptor(
        identifier=descriptor.identifier,
        renderer=descriptor.renderer,
        metadata=updated_meta,
        trusted=trusted,
        source=descriptor.source,
    )

    _PLOT_RENDERERS[identifier] = updated
    if trusted:
        _TRUSTED_PLOT_RENDERERS.add(identifier)
    else:
        _TRUSTED_PLOT_RENDERERS.discard(identifier)

    _propagate_trust_metadata(descriptor.renderer, updated_meta)
    return updated


def mark_plot_renderer_trusted(identifier: str) -> PlotRendererDescriptor:
    """Mark the plot renderer *identifier* as trusted."""
    return _refresh_plot_renderer_trust(identifier, trusted=True)


def mark_plot_renderer_untrusted(identifier: str) -> PlotRendererDescriptor:
    """Remove the plot renderer *identifier* from the trusted set."""
    return _refresh_plot_renderer_trust(identifier, trusted=False)


def register(
    plugin: ExplainerPlugin,
    *,
    source: str = "manual",
    identifier: str | None = None,
) -> None:
    """Register a plugin after minimal metadata validation.

    Notes: Registering a plugin executes third-party code at import-time.
    Only register trusted plugins.
    """
    raw_meta = getattr(plugin, "plugin_meta", None)
    if raw_meta is None:
        raise ValidationError(
            "plugin must expose plugin_meta metadata",
            details={"param": "plugin", "required_attribute": "plugin_meta"},
        )
    meta: Dict[str, Any] = dict(raw_meta)
    validate_plugin_meta(meta)
    identifier = identifier or meta.get("name")
    if not isinstance(identifier, str) or not identifier:
        raise ValidationError(
            "plugin_meta['name'] must be a non-empty string",
            details={"param": "name", "expected_type": "str", "source": source},
        )
    if is_identifier_denied(identifier):
        raise ValidationError(
            f"Plugin '{identifier}' is denied via CE_DENY_PLUGIN",
            details={"param": "identifier", "identifier": identifier},
        )
    trusted = _should_trust(meta, identifier=identifier, source=source)
    _update_trust_keys(meta, trusted)
    _verify_plugin_checksum(plugin, meta)
    if isinstance(raw_meta, dict):
        raw_meta.setdefault("version", meta.get("version", package_version))
        raw_meta.setdefault("provider", meta.get("provider"))
        raw_meta["trusted"] = meta["trusted"]
        raw_meta["trust"] = meta["trust"]
    elif hasattr(raw_meta, "__setitem__"):
        try:
            raw_meta["trusted"] = meta["trusted"]
            raw_meta["trust"] = meta["trust"]
        except Exception:  # ADR002_ALLOW: metadata propagation is best-effort.  # pragma: no cover
            # pragma: no cover - defensive
            _LOGGER.debug(
                "Failed to propagate trust metadata for plugin %r",
                plugin,
                exc_info=True,
            )
    if plugin in _REGISTRY:
        if trusted and plugin not in _TRUSTED:
            _TRUSTED.append(plugin)
        return
    _REGISTRY.append(plugin)
    if trusted and plugin not in _TRUSTED:
        _TRUSTED.append(plugin)


def unregister(plugin: ExplainerPlugin) -> None:
    """Remove a plugin if present."""
    with contextlib.suppress(ValueError):
        _REGISTRY.remove(plugin)
    with contextlib.suppress(ValueError):
        _TRUSTED.remove(plugin)


def clear() -> None:
    """Clear all registered plugins (testing convenience)."""
    _REGISTRY.clear()
    _TRUSTED.clear()


def list_plugins(*, include_untrusted: bool = True) -> Tuple[ExplainerPlugin, ...]:
    """Return a snapshot of registered plugins."""
    if include_untrusted:
        return tuple(_REGISTRY)
    trusted_set = set(_TRUSTED)
    return tuple(plugin for plugin in _REGISTRY if plugin in trusted_set)


def _resolve_plugin_from_name(name: str) -> ExplainerPlugin:
    """Resolve a registered plugin or descriptor by its human-readable name."""
    for plugin in _REGISTRY:
        meta = getattr(plugin, "plugin_meta", {})
        getter = getattr(meta, "get", None)
        if getter is None:
            continue
        try:
            plugin_name = getter("name")
        except (
            Exception
        ):  # ADR002_ALLOW: continue enumerating if metadata misbehaves.  # pragma: no cover
            # pragma: no cover - defensive
            _LOGGER.debug(
                "Failed to read plugin name for %r",
                plugin,
                exc_info=True,
            )
            plugin_name = None
        if plugin_name == name:
            return plugin
    for descriptor in _EXPLANATION_PLUGINS.values():
        if descriptor.metadata.get("name") == name:
            return descriptor.plugin
    raise KeyError(f"Plugin '{name}' is not registered")


def trust_plugin(plugin: ExplainerPlugin | str) -> None:
    """Mark an already-registered plugin as trusted.

    Trust is an explicit, opt-in operation. The function validates metadata
    before adding to the trusted list. Only trusted plugins will be returned
    by :func:`find_for` when `trusted_only=True` is passed.
    """
    if isinstance(plugin, str):
        plugin = _resolve_plugin_from_name(plugin)
    if plugin not in _REGISTRY:
        raise ValidationError(
            "Plugin must be registered before it can be trusted",
            details={"param": "plugin", "requirement": "must be registered"},
        )
    raw_meta = getattr(plugin, "plugin_meta", None)
    meta: Dict[str, Any] = dict(raw_meta)
    validate_plugin_meta(meta)
    _update_trust_keys(meta, True)
    if isinstance(raw_meta, dict):
        raw_meta["trusted"] = True
        raw_meta["trust"] = True
    if plugin in _TRUSTED:
        return
    _TRUSTED.append(plugin)


def untrust_plugin(plugin: ExplainerPlugin | str) -> None:
    """Remove a plugin from the trusted set if present."""
    if isinstance(plugin, str):
        plugin = _resolve_plugin_from_name(plugin)
    with contextlib.suppress(ValueError):
        _TRUSTED.remove(plugin)
    raw_meta = getattr(plugin, "plugin_meta", None)
    if isinstance(raw_meta, dict):
        raw_meta["trusted"] = False
        raw_meta["trust"] = False


def find_for(model: Any) -> Tuple[ExplainerPlugin, ...]:
    """Find plugins that declare support for the given model."""
    return tuple(p for p in _REGISTRY if _safe_supports(p, model))


def find_for_trusted(model: Any) -> Tuple[ExplainerPlugin, ...]:
    """Find trusted plugins that declare support for the given model."""
    return tuple(p for p in _TRUSTED if _safe_supports(p, model))


def _safe_supports(plugin: ExplainerPlugin, model: Any) -> bool:
    """Return True when a plugin reports support for *model* without raising."""
    try:
        return bool(plugin.supports(model))
    except Exception:  # ADR002_ALLOW: treat plugin errors as lack of support.  # pragma: no cover
        return False


__all__ = [
    "ExplanationPluginDescriptor",
    "IntervalPluginDescriptor",
    "PluginDiscoveryRecord",
    "PluginDiscoveryReport",
    "PlotBuilderDescriptor",
    "PlotRendererDescriptor",
    "PlotStyleDescriptor",
    "EXPLANATION_PROTOCOL_VERSION",
    "validate_explanation_metadata",
    "validate_interval_metadata",
    "validate_plot_builder_metadata",
    "validate_plot_renderer_metadata",
    "validate_plot_style_metadata",
    "clear_explanation_plugins",
    "clear_interval_plugins",
    "clear_plot_plugins",
    "ensure_builtin_plugins",
    "is_identifier_denied",
    "register_explanation_plugin",
    "register_interval_plugin",
    "register_plot_builder",
    "register_plot_renderer",
    "register_plot_style",
    "register_plot_plugin",
    "find_explanation_descriptor",
    "find_interval_descriptor",
    "find_plot_builder_descriptor",
    "find_plot_renderer_descriptor",
    "find_plot_style_descriptor",
    "find_explanation_plugin",
    "find_interval_plugin",
    "find_plot_builder",
    "find_plot_renderer",
    "find_plot_plugin",
    "find_plot_plugin_trusted",
    "get_last_discovery_report",
    "get_discovery_report",
    "list_explanation_descriptors",
    "list_interval_descriptors",
    "list_plot_builder_descriptors",
    "list_plot_renderer_descriptors",
    "list_plot_style_descriptors",
    "load_entrypoint_plugins",
    "mark_explanation_trusted",
    "mark_explanation_untrusted",
    "mark_interval_trusted",
    "mark_interval_untrusted",
    "mark_plot_builder_trusted",
    "mark_plot_builder_untrusted",
    "mark_plot_renderer_trusted",
    "mark_plot_renderer_untrusted",
    "register",
    "unregister",
    "clear",
    "list_plugins",
    "trust_plugin",
    "untrust_plugin",
    "find_for",
    "find_for_trusted",
]
