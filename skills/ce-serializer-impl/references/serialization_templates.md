# Serialization Code Templates (ADR-031, schema v3 security hardening)

> **Trust boundary reminder:** a state artifact is untrusted input. A SHA-256
> checksum recorded *inside* the artifact proves only that the artifact is
> internally self-consistent, never that it came from a trustworthy source.
> Never gate a `pickle.loads`/`pickle.load`/`joblib.load`/`cloudpickle.loads`
> call on a checksum computed from bytes stored alongside it -- that pattern
> is exactly the vulnerability this schema version fixed (see ADR-031's
> "Trust boundary (schema v3)" section and its 2026-07-22 security addendum).
> The templates below are the safe pattern; do not reintroduce pickle on the
> `to_primitive`/`from_primitive` or `save_state`/`load_state` call paths.

## Calibrator `to_primitive` / `from_primitive`

### Actual pattern (JSON-safe fields only)

The built-in calibrators (`VennAbers`, `IntervalRegressor`) serialise their
restorable state as plain JSON-safe fields (lists via `.tolist()`, dicts with
string keys, primitives) under a `"fields"` key. `from_primitive` rebuilds the
object directly from those fields -- it never calls `pickle.loads` anywhere.
(They also stash a `checksums.sha256` computed from a pickle of internal
state, but that pickle blob is *not stored* and *never decoded*; it is a
vestigial corruption-detection value, not part of restoration. Do not model a
new calibrator on that leftover detail -- model it on the `fields` pattern.)

See `src/calibrated_explanations/calibration/venn_abers.py` and
`src/calibrated_explanations/calibration/interval_regressor.py` for the real
implementations.

```python
from typing import Any, Mapping

import numpy as np

from calibrated_explanations.utils.exceptions import ConfigurationError


class MyCalibrator:
    def to_primitive(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict of plain fields -- no pickle."""

        def _as_list(value: Any) -> Any:
            if value is None:
                return None
            return value.tolist() if hasattr(value, "tolist") else list(value)

        return {
            "schema_version": 1,
            "calibrator_type": "my_calibrator",
            "parameters": {
                # Lightweight metadata for inspection without deserialising
                "is_multiclass": bool(self.is_multiclass()),
            },
            "fields": {
                "alpha": float(self.alpha),
                "x_cal": _as_list(self.x_cal),
                "y_cal": _as_list(self.y_cal),
            },
        }

    @classmethod
    def from_primitive(cls, payload: Mapping[str, object]) -> "MyCalibrator":
        """Rehydrate purely from JSON-safe fields -- never unpickle anything."""
        schema_version = payload.get("schema_version")
        if schema_version != 1:
            raise ConfigurationError(
                "Unsupported MyCalibrator schema_version. Supported versions: [1].",
                details={"schema_version": schema_version, "supported_versions": [1]},
            )
        calibrator_type = payload.get("calibrator_type")
        if calibrator_type != "my_calibrator":
            raise ConfigurationError(
                "Invalid calibrator_type for MyCalibrator payload.",
                details={"calibrator_type": calibrator_type, "expected": "my_calibrator"},
            )
        fields = payload.get("fields")
        if not isinstance(fields, Mapping):
            raise ConfigurationError(
                "MyCalibrator primitive is missing 'fields' mapping.",
                details={"field": "fields"},
            )
        required = ("alpha", "x_cal", "y_cal")
        missing = [name for name in required if fields.get(name) is None]
        if missing:
            raise ConfigurationError(
                "MyCalibrator primitive missing required field(s): " + ", ".join(missing),
                details={"fields": missing},
            )

        obj = cls.__new__(cls)
        obj.alpha = float(fields["alpha"])
        obj.x_cal = np.asarray(fields["x_cal"])
        obj.y_cal = np.asarray(fields["y_cal"])
        return obj
```

Register `MyCalibrator` in the fixed dispatch table used by
`WrapCalibratedExplainer._restore_calibrator_from_primitive` (a per-type
`if calibrator_type == "..."` branch with a local, non-dynamic import) so
restoration never resolves a calibrator type via an artifact-provided module
path.

If a calibrator genuinely cannot expose a JSON-safe `fields` representation,
do **not** fall back to pickling it. Let
`WrapCalibratedExplainer._calibrator_to_primitive` raise `SerializationError`
for it (the default behavior when `to_primitive` is absent/non-callable) so
`save_state()` fails closed instead of silently writing an executable payload.

---

## Explainer `save_state` / `load_state`

The actual implementation persists a **directory artifact** containing only
JSON files, with per-file sha256 checksums in the manifest. There is no
pickle payload anywhere in the artifact.

See `src/calibrated_explanations/core/wrap_explainer.py` for the real
implementation (`WrapCalibratedExplainer.save_state` / `load_state`).

### Artifact directory structure

```
my_explainer_state/
  manifest.json               # schema_version (3), created_at_utc, artifact_type, files
  explainer_state.json        # wrapper/learner-identity/calibration config (JSON-safe)
  calibrator_primitive.json   # calibrator.to_primitive() output (if fitted)
  preprocessing_mapping.json  # feature mappings (if present)
```

There is intentionally **no `wrapper.pkl`**. A generic fitted learner (and an
arbitrary custom preprocessor) cannot be safely reconstructed from bytes, so
`load_state()` requires the caller to supply those live objects again instead
of persisting them.

### save_state (returns Path, no pickle)

```python
def save_state(self, path_or_fileobj: Any) -> Path:
    """Persist wrapper state using a schema v3 (JSON-only) manifest + checksums."""
    target = self._state_path(path_or_fileobj)
    # ... build explainer_state.json / calibrator_primitive.json /
    #     preprocessing_mapping.json as JSON, write to temp dir, compute
    #     per-file sha256 checksums ...

    manifest = {
        "schema_version": self._STATE_SCHEMA_VERSION,  # 3
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_type": "wrap_calibrated_explainer_state",
        "files": checksums,  # {"explainer_state.json": "<sha256>", ...}
    }
    # Atomic rename temp_dir -> target
    return target
```

### load_state (rejects legacy pickle artifacts, validates before reconstructing)

```python
from calibrated_explanations.utils.exceptions import IncompatibleStateError, ValidationError

@classmethod
def load_state(
    cls,
    path_or_fileobj: Any,
    *,
    learner: Any | None = None,
    preprocessor: Any | None = None,
    difficulty_estimator: Any | None = None,
    mc: Any | None = None,
) -> WrapCalibratedExplainer:
    """Load wrapper state from a safe (schema v3) ADR-031 artifact."""
    path = ...
    manifest = cls._read_manifest_json(path)  # validates JSON top-level type first

    # Legacy (pickle-based) schema versions are refused unconditionally,
    # before any other file is even opened -- a matching checksum does not
    # make a legacy artifact trustworthy.
    schema_version = manifest.get("schema_version")
    if schema_version in cls._LEGACY_STATE_SCHEMA_VERSIONS:
        raise IncompatibleStateError("... legacy artifact rejected ...", details={...})
    if schema_version != cls._STATE_SCHEMA_VERSION:
        raise IncompatibleStateError("Unsupported state schema_version.", details={...})

    # Validate the file inventory (allow-listed filenames only, no absolute
    # paths / traversal / symlink escape / undeclared files) and verify every
    # checksum before any file's contents are treated as anything but bytes.
    validated_files = cls._validate_and_verify_manifest_files(manifest, path)

    state = cls._read_json_object(path / "explainer_state.json")

    # The caller must supply runtime objects that cannot be safely
    # reconstructed from bytes; validate them against persisted metadata.
    cls._validate_supplied_learner(learner, state.get("learner") or {})
    if state.get("calibration", {}).get("difficulty_estimator_required") and difficulty_estimator is None:
        raise ValidationError("difficulty_estimator required for restoration", details={...})

    # Reconstruct the CalibratedExplainer directly from JSON-safe calibration
    # config + the caller-supplied learner, then overwrite interval_learner
    # with the restored (JSON-safe) calibrator primitive if present.
    ...
```

---

## Round-trip invariant (ADR-031 §4)

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
import json
import pytest
import numpy as np
from calibrated_explanations.utils.exceptions import ConfigurationError


def test_should_round_trip_when_primitive_is_valid(fitted_calibrator, X_test):
    """Restored calibrator must produce identical outputs to the original."""
    primitive = fitted_calibrator.to_primitive()
    restored = type(fitted_calibrator).from_primitive(primitive)

    np.testing.assert_allclose(
        fitted_calibrator.predict_proba(X_test),
        restored.predict_proba(X_test),
        atol=1e-9,
    )


def test_should_raise_configuration_error_when_version_mismatch():
    """from_primitive must fail fast on unknown schema_version."""
    from calibrated_explanations.calibration.venn_abers import VennAbers

    with pytest.raises(ConfigurationError, match="schema_version"):
        VennAbers.from_primitive({"schema_version": 99, "calibrator_type": "venn_abers"})


def test_should_raise_configuration_error_when_calibrator_type_mismatch():
    """from_primitive must reject wrong calibrator_type."""
    from calibrated_explanations.calibration.venn_abers import VennAbers

    with pytest.raises(ConfigurationError, match="calibrator_type"):
        VennAbers.from_primitive({"schema_version": 2, "calibrator_type": "wrong"})


def test_should_be_json_safe_when_serialising(fitted_calibrator):
    """to_primitive must return only JSON-safe types."""
    json.dumps(fitted_calibrator.to_primitive())  # must not raise


def test_load_state_rejects_python_pickle_primitive_without_decoding():
    """A calibrator_type='python_pickle' payload, even with a correctly
    recomputed checksum, must be rejected before any base64 decode or
    pickle.loads() call -- see
    tests/unit/core/test_wrap_explainer_persistence_security.py for the full
    exploit-regression and static AST-guard tests this pattern must satisfy.
    """
```
