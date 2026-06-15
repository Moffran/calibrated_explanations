# Serialization Code Templates (ADR-031)

## Calibrator `to_primitive` / `from_primitive`

### Actual pattern (pickle + base64 + checksum)

The built-in calibrators (`VennAbers`, `IntervalRegressor`) serialise complex
internal state using pickle, base64-encode it, and include a sha256 checksum
for integrity verification. The JSON-safe dict wraps this with metadata.

See `src/calibrated_explanations/calibration/venn_abers.py` and
`src/calibrated_explanations/calibration/interval_regressor.py` for the real
implementations.

```python
import base64
import hashlib
import pickle
from typing import Any, Mapping

from calibrated_explanations.utils.exceptions import ConfigurationError


class MyCalibrator:
    def to_primitive(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict with pickle+b64 payload."""
        payload_bytes = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
        return {
            "schema_version": 1,
            "calibrator_type": "my_calibrator",
            "parameters": {
                # Lightweight metadata for inspection without deserialising
                "is_multiclass": bool(self.is_multiclass()),
            },
            "checksums": {
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            },
            "payload": {
                "pickle_b64": payload_b64,
            },
        }

    @classmethod
    def from_primitive(cls, payload: Mapping[str, object]) -> "MyCalibrator":
        """Rehydrate from a primitive payload with checksum validation."""
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
        payload_section = payload.get("payload")
        if not isinstance(payload_section, Mapping):
            raise ConfigurationError(
                "MyCalibrator primitive is missing 'payload' mapping.",
                details={"field": "payload"},
            )
        pickle_b64 = payload_section.get("pickle_b64")
        if not isinstance(pickle_b64, str):
            raise ConfigurationError(
                "MyCalibrator primitive is missing 'pickle_b64'.",
                details={"field": "payload.pickle_b64"},
            )
        payload_bytes = base64.b64decode(pickle_b64.encode("ascii"))

        # Checksum validation
        checksums = payload.get("checksums")
        if not isinstance(checksums, Mapping):
            raise ConfigurationError(
                "MyCalibrator primitive is missing checksum metadata.",
                details={"field": "checksums"},
            )
        expected_sha = checksums.get("sha256")
        actual_sha = hashlib.sha256(payload_bytes).hexdigest()
        if not isinstance(expected_sha, str) or expected_sha != actual_sha:
            raise ConfigurationError(
                "MyCalibrator primitive checksum validation failed.",
                details={"expected_sha256": expected_sha, "actual_sha256": actual_sha},
            )

        restored = pickle.loads(payload_bytes)  # noqa: S301
        if not isinstance(restored, cls):
            raise ConfigurationError(
                "MyCalibrator primitive restored unexpected object type.",
                details={"restored_type": type(restored).__name__},
            )
        return restored
```

---

## Explainer `save_state` / `load_state`

The actual implementation persists a **directory artifact** containing multiple
files with per-file sha256 checksums in the manifest.

See `src/calibrated_explanations/core/wrap_explainer.py` for the real
implementation (`WrapCalibratedExplainer.save_state` / `load_state`).

### Artifact directory structure

```
my_explainer_state/
  manifest.json               # schema_version, created_at_utc, artifact_type, files
  wrapper.pkl                 # pickle of the full WrapCalibratedExplainer
  calibrator_primitive.json   # calibrator.to_primitive() output (if fitted)
  preprocessing_mapping.json  # feature mappings (if present)
  explainer_config.json       # explainer configuration payload
```

### save_state (returns Path)

```python
def save_state(self, path_or_fileobj: Any) -> Path:
    """Persist wrapper state using an ADR-031 manifest + checksums."""
    target = self._state_path(path_or_fileobj)
    # ... write files to temp dir, compute checksums ...

    manifest = {
        "schema_version": self._STATE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_type": "wrap_calibrated_explainer_state",
        "files": checksums,  # {"wrapper.pkl": "<sha256>", ...}
    }
    # Atomic rename temp_dir -> target
    return target
```

### load_state (validates manifest + checksums)

```python
from calibrated_explanations.utils.exceptions import IncompatibleStateError

@classmethod
def load_state(cls, path_or_fileobj: Any) -> WrapCalibratedExplainer:
    """Load wrapper state from an ADR-031 persisted artifact."""
    path = ...
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    # Fail fast on unsupported schema
    schema_version = manifest.get("schema_version")
    if schema_version != cls._STATE_SCHEMA_VERSION:
        raise IncompatibleStateError(
            "Unsupported state schema_version.",
            details={
                "schema_version": schema_version,
                "supported_versions": [cls._STATE_SCHEMA_VERSION],
            },
        )

    # Validate per-file checksums
    files = manifest.get("files")
    for file_name, expected_sha in files.items():
        actual_sha = cls._sha256_file(path / file_name)
        if actual_sha != expected_sha:
            raise IncompatibleStateError(
                "State checksum validation failed.",
                details={"file": file_name, ...},
            )

    # Deserialise wrapper from wrapper.pkl
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
        VennAbers.from_primitive({"schema_version": 1, "calibrator_type": "wrong"})


def test_should_raise_configuration_error_when_checksum_invalid(fitted_calibrator):
    """from_primitive must reject tampered payloads."""
    primitive = fitted_calibrator.to_primitive()
    primitive["checksums"]["sha256"] = "0" * 64  # tampered

    with pytest.raises(ConfigurationError, match="checksum"):
        type(fitted_calibrator).from_primitive(primitive)


def test_should_be_json_safe_when_serialising(fitted_calibrator):
    """to_primitive must return only JSON-safe types."""
    json.dumps(fitted_calibrator.to_primitive())  # must not raise
```
