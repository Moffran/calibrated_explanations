"""Content checksums for JSON-safe calibrator primitives (ADR-031).

The checksum covers exactly the persisted content of a schema v2 calibrator
primitive (``schema_version``, ``calibrator_type``, ``parameters`` and
``fields``) serialized as canonical JSON. It is therefore reproducible across
processes and can be recomputed and verified when the primitive is loaded.

Primitives written by calibrated-explanations 1.0.1 and earlier carry a digest
of a pickle of the in-memory calibrator, without an ``algorithm`` tag. That
digest cannot be recomputed from the persisted content, so such primitives load
unverified with a ``UserWarning``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from typing import Any, Mapping

from ..utils.exceptions import IncompatibleStateError, SerializationError

_LOGGER = logging.getLogger(__name__)

CHECKSUM_ALGORITHM = "sha256-canonical-json-v1"
_CHECKSUMMED_KEYS = ("schema_version", "calibrator_type", "parameters", "fields")


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars to their Python equivalents for canonical JSON."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise SerializationError(
        "Calibrator primitive contains a value that is not JSON serializable.",
        details={"value_type": type(value).__name__},
    )


def compute_primitive_checksum(primitive: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 content digest of a calibrator primitive.

    Parameters
    ----------
    primitive : Mapping[str, Any]
        Schema v2 calibrator primitive.

    Returns
    -------
    str
        Hex digest over the canonical JSON of the checksummed keys.
    """
    body = {key: primitive.get(key) for key in _CHECKSUMMED_KEYS}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def primitive_checksums(primitive: Mapping[str, Any]) -> dict[str, str]:
    """Return the ``checksums`` entry for a calibrator primitive.

    Parameters
    ----------
    primitive : Mapping[str, Any]
        Schema v2 calibrator primitive without its ``checksums`` entry.

    Returns
    -------
    dict[str, str]
        Mapping with the ``algorithm`` tag and the ``sha256`` digest.
    """
    return {"algorithm": CHECKSUM_ALGORITHM, "sha256": compute_primitive_checksum(primitive)}


def verify_primitive_checksum(primitive: Mapping[str, Any]) -> None:
    """Verify the content checksum of a calibrator primitive before it is loaded.

    Parameters
    ----------
    primitive : Mapping[str, Any]
        Schema v2 calibrator primitive.

    Raises
    ------
    IncompatibleStateError
        If the primitive declares the canonical checksum algorithm and its
        content does not match the recorded digest, or if it declares an
        unknown checksum algorithm.

    Warns
    -----
    UserWarning
        If the primitive carries no verifiable checksum (written by
        calibrated-explanations 1.0.1 or earlier). It is loaded unverified.
    """
    calibrator_type = primitive.get("calibrator_type")
    checksums = primitive.get("checksums")
    if not isinstance(checksums, Mapping) or checksums.get("algorithm") is None:
        _LOGGER.info(
            "Calibrator primitive %r has no verifiable checksum; it is loaded without "
            "integrity verification. Re-save it with save_state() to add one.",
            calibrator_type,
        )
        warnings.warn(
            "Calibrator primitive %r has no verifiable checksum; it is loaded without "
            "integrity verification. Re-save it with save_state() to add one. Primitives "
            "without a verifiable checksum will be rejected by a future schema version."
            % (calibrator_type,),
            UserWarning,
            stacklevel=3,
        )
        return
    algorithm = checksums["algorithm"]
    if algorithm != CHECKSUM_ALGORITHM:
        raise IncompatibleStateError(
            "Calibrator primitive declares an unsupported checksum algorithm.",
            details={
                "calibrator_type": calibrator_type,
                "algorithm": algorithm,
                "supported_algorithms": [CHECKSUM_ALGORITHM],
            },
        )
    expected = checksums.get("sha256")
    actual = compute_primitive_checksum(primitive)
    if expected != actual:
        raise IncompatibleStateError(
            "Calibrator primitive checksum validation failed: its content was modified "
            "after it was written.",
            details={
                "calibrator_type": calibrator_type,
                "expected_sha256": expected,
                "actual_sha256": actual,
            },
        )


__all__ = [
    "CHECKSUM_ALGORITHM",
    "compute_primitive_checksum",
    "primitive_checksums",
    "verify_primitive_checksum",
]
