from __future__ import annotations

from typing import Any, Mapping

from ..config_helpers import coerce_string_tuple


def check_interval_runtime_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    identifier: str | None,
    fast: bool,
    mode: str,
    bins: Any = None,
) -> str | None:
    """Validate interval plugin metadata for the current execution.

    Parameters
    ----------
    metadata : Mapping[str, Any] | None
        The metadata dictionary from the plugin.
    identifier : str | None
        The identifier of the plugin (for error messages).
    fast : bool
        Whether the execution is in fast mode.
    mode : str
        The operating mode of the explainer ('classification' or 'regression').
    bins : Any, optional
        The bins configuration of the explainer.

    Returns
    -------
    str | None
        An error message if validation fails, else None.
    """
    prefix = identifier or str((metadata or {}).get("name") or "<anonymous>")
    if metadata is None:
        return f"{prefix}: interval metadata unavailable"

    schema_version = metadata.get("schema_version")
    if schema_version not in (None, 1):
        return f"{prefix}: unsupported interval schema_version {schema_version}"

    modes = coerce_string_tuple(metadata.get("modes"))
    if not modes:
        return f"{prefix}: plugin metadata missing modes declaration"

    required_mode = "regression" if "regression" in mode else "classification"
    if required_mode not in modes:
        declared = ", ".join(modes)
        return f"{prefix}: does not support mode '{required_mode}' (modes: {declared})"

    capabilities = set(coerce_string_tuple(metadata.get("capabilities")))
    required_cap = "interval:regression" if "regression" in mode else "interval:classification"
    if required_cap not in capabilities:
        declared = ", ".join(sorted(capabilities)) or "<none>"
        return f"{prefix}: missing capability '{required_cap}' (capabilities: {declared})"

    if fast and not bool(metadata.get("fast_compatible")):
        return f"{prefix}: not marked fast_compatible"
    if metadata.get("requires_bins") and bins is None:
        return f"{prefix}: requires bins but explainer has none configured"
    return None
