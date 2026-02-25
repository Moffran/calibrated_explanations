---
name: ce-serializer-impl
description: >
  Implement ADR-031 serializer and persistence APIs with versioned primitives and
  compatible load behavior.
---

# CE Serializer Implementation

You are implementing calibrator serialization or explainer state persistence
per ADR-031. The contract is: versioned JSON-safe primitives + fail-fast on
schema incompatibility.

---

## Part 1 — Calibrator `to_primitive` / `from_primitive`

All built-in calibrators must implement this pair.

```python
from __future__ import annotations

from typing import Any, Mapping

from calibrated_explanations.utils.exceptions import IncompatibleSchemaError

_SCHEMA_VERSION = 1


class MyCalibrator:
    """Custom calibrator with ADR-031 serialization.

    Parameters
    ----------
    alpha : float
        Calibration parameter.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._fitted_values: list[float] = []

    def to_primitive(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict.

        Returns
        -------
        dict[str, Any]
            Primitive payload. Contains a mandatory ``schema_version`` key
            and calibrator-specific fields.

        Notes
        -----
        All values must be JSON-safe (str, int, float, bool, list, dict, None).
        Do NOT include numpy arrays — convert to ``list`` first.
        """
        return {
            "schema_version": _SCHEMA_VERSION,
            "alpha": float(self.alpha),
            "fitted_values": [float(v) for v in self._fitted_values],
        }

    @classmethod
    def from_primitive(cls, payload: Mapping[str, object]) -> "MyCalibrator":
        """Reconstruct from a serialised primitive.

        Parameters
        ----------
        payload : Mapping[str, object]
            Output of ``to_primitive()``.

        Returns
        -------
        MyCalibrator
            Reconstructed calibrator instance.

        Raises
        ------
        IncompatibleSchemaError
            If ``schema_version`` is missing or unsupported.
        """
        version = payload.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise IncompatibleSchemaError(
                f"MyCalibrator expects schema_version={_SCHEMA_VERSION}, "
                f"got {version!r}. Supported versions: [{_SCHEMA_VERSION}]. "
                "See docs/migration/ for upgrade guidance."
            )
        instance = cls(alpha=float(payload["alpha"]))
        instance._fitted_values = list(payload.get("fitted_values", []))
        return instance
```

---

## Part 2 — Explainer `save_state` / `load_state`

```python
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Union

from calibrated_explanations import __version__


def save_state(self, path_or_fileobj: Union[str, Path, IO[bytes]]) -> None:
    """Persist explainer state to a file or file-like object.

    Parameters
    ----------
    path_or_fileobj : str, Path, or file-like
        Destination for the serialised state.

    Notes
    -----
    The on-disk format is a JSON document with a top-level manifest and
    nested calibrator primitives. All primitives must be JSON-safe per ADR-031.
    """
    state = self._build_state_dict()
    payload = json.dumps(state, indent=2, sort_keys=True).encode()

    if isinstance(path_or_fileobj, (str, Path)):
        Path(path_or_fileobj).write_bytes(payload)
    else:
        path_or_fileobj.write(payload)


def _build_state_dict(self) -> dict:
    payload = json.dumps({
        "classification_calibrator": (
            self._calibrator.to_primitive() if self._calibrator else None
        ),
        "regression_calibrator": (
            self._reg_calibrator.to_primitive() if self._reg_calibrator else None
        ),
        "preprocessing_mappings": self._mappings.export() if self._mappings else None,
        "plugin_identifiers": self._plugin_identifiers,
        "rng_seed": getattr(self, "_rng_seed", None),
    }, sort_keys=True)
    checksum = hashlib.sha256(payload.encode()).hexdigest()

    return {
        "manifest": {
            "schema_version": 1,
            "ce_version": __version__,
            "serialized_at": datetime.now(tz=timezone.utc).isoformat(),
            "checksum": checksum,
        },
        **json.loads(payload),
    }


@classmethod
def load_state(cls, path_or_fileobj: Union[str, Path, IO[bytes]]):
    """Restore explainer state from a file or file-like object.

    Parameters
    ----------
    path_or_fileobj : str, Path, or file-like
        Source of the serialised state.

    Returns
    -------
    WrapCalibratedExplainer
        Restored explainer instance.

    Raises
    ------
    IncompatibleSchemaError
        If the manifest schema version is unsupported.
    """
    if isinstance(path_or_fileobj, (str, Path)):
        raw = Path(path_or_fileobj).read_bytes()
    else:
        raw = path_or_fileobj.read()

    state = json.loads(raw)
    manifest = state.get("manifest", {})
    version = manifest.get("schema_version")
    if version != 1:
        raise IncompatibleSchemaError(
            f"save_state schema_version={version!r} is not supported. "
            "Supported: [1]."
        )
    # Reconstruct calibrators, mappings, plugins...
    ...
```

---

## Round-trip invariant (ADR-031 §4)

After a save/load round-trip, the restored calibrator must produce identical outputs:

```python
import numpy as np

# Reference
original = MyCalibrator(alpha=0.3)
original.fit(X_cal, y_cal)
ref_proba = original.predict_proba(X_query)

# Round-trip
restored = MyCalibrator.from_primitive(original.to_primitive())
rt_proba = restored.predict_proba(X_query)

assert np.allclose(ref_proba, rt_proba, atol=1e-9), "Round-trip invariant violated"
```

---

## Test template

```python
# tests/unit/test_my_calibrator_serialization.py
import pytest
import numpy as np
from calibrated_explanations.utils.exceptions import IncompatibleSchemaError


def test_should_round_trip_when_primitive_is_valid():
    """Restored calibrator must produce identical outputs to the original."""
    from mypkg.calibrators import MyCalibrator
    cal = MyCalibrator(alpha=0.7)
    # ... fit cal ...

    primitive = cal.to_primitive()
    restored = MyCalibrator.from_primitive(primitive)

    # Verify: same predictions
    np.testing.assert_allclose(
        cal.predict_proba(X_test), restored.predict_proba(X_test), atol=1e-9
    )


def test_should_raise_incompatible_schema_when_version_mismatch():
    """from_primitive must fail fast on unknown schema_version."""
    from mypkg.calibrators import MyCalibrator
    with pytest.raises(IncompatibleSchemaError, match="schema_version"):
        MyCalibrator.from_primitive({"schema_version": 99, "alpha": 0.5})


def test_should_be_json_safe_when_serialising():
    """to_primitive must return only JSON-safe types."""
    import json
    from mypkg.calibrators import MyCalibrator
    cal = MyCalibrator()
    json.dumps(cal.to_primitive())  # must not raise
```

---

## Out of Scope

- Pickle-based serialization — only JSON-safe primitives per ADR-031.
- Third-party calibrator schema management — external packages own their schema versions.

## Evaluation Checklist

- [ ] `to_primitive()` returns a dict with `schema_version` as the first key.
- [ ] All values are JSON-safe (no numpy arrays, no non-serialisable objects).
- [ ] `from_primitive()` raises `IncompatibleSchemaError` on version mismatch with guidance.
- [ ] Round-trip test verifies identical predictions (not just no-exception).
- [ ] `json.dumps(primitive)` test verifies JSON-safety.
- [ ] `save_state` / `load_state` include a manifest with `schema_version`, `ce_version`, `checksum`.
