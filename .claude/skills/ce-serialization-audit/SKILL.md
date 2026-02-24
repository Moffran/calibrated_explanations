---
name: ce-serialization-audit
description: >
  Audit serialization code for conformance with ADR-031 (calibrator serialization
  and state persistence). Use when asked to 'audit serialization code', 'check
  save/load contract', 'verify round-trip', 'review to_primitive', 'check
  schema_version', 'serialization contract review', 'JSON-safe audit',
  'IncompatibleSchemaError check'. Covers all ADR-031 requirements and the
  invariant checks from ADR-021.
---

# CE Serialization Audit

You are auditing serialization code for conformance with ADR-031. Work through
each dimension below and report findings. The round-trip invariant and
schema-version enforcement are blocking (must fix before merge).

---

## Dimension 1 — `to_primitive()` contract

```python
primitive = calibrator.to_primitive()
```

| Check | Requirement | Pass / Fail |
|---|---|---|
| Returns a `dict` | Yes — not a custom object | |
| `schema_version` present | Must be present and a positive `int` | |
| `schema_version` is first key | Convention for readability | |
| All values are JSON-safe | `str`, `int`, `float`, `bool`, `list`, `dict`, `None` only | |
| No `numpy.ndarray` values | Must be converted to `list` | |
| No `datetime` or custom objects | Must be serialised to `str` or primitives | |

```python
import json

primitive = calibrator.to_primitive()
assert "schema_version" in primitive, "schema_version missing"
assert isinstance(primitive["schema_version"], int), "schema_version must be int"
json.dumps(primitive)   # raises TypeError if not JSON-safe
```

---

## Dimension 2 — `from_primitive()` error handling

```python
# Must raise IncompatibleSchemaError — not ValueError, not KeyError
with pytest.raises(IncompatibleSchemaError):
    MyCalibrator.from_primitive({"schema_version": 9999})

# Must raise on missing schema_version (not produce silent garbage)
with pytest.raises(IncompatibleSchemaError):
    MyCalibrator.from_primitive({})
```

| Check | Requirement |
|---|---|
| Unknown `schema_version` → `IncompatibleSchemaError` | Must fail fast |
| Missing `schema_version` → `IncompatibleSchemaError` | Must fail fast |
| Error message names supported versions | Must aid migration |
| Error message links to migration guide | Strongly encouraged |

---

## Dimension 3 — Round-trip invariant (ADR-031 §4 + ADR-021)

The semantics defined in ADR-021 (probability bounds, interval ordering, monotonicity)
must be preserved across a save/load round-trip:

```python
import numpy as np

original = MyCalibrator()
# ... fit ...
prim = original.to_primitive()
restored = MyCalibrator.from_primitive(prim)

# Prediction identical to within floating-point
np.testing.assert_allclose(
    original.predict_proba(X_test),
    restored.predict_proba(X_test),
    atol=1e-9,
)

# Interval invariant preserved
pred = restored.predict_proba(X_test, output_interval=True)
assert np.all(pred[..., 1] <= pred[..., 0]), "low > predict violates ADR-021"
assert np.all(pred[..., 0] <= pred[..., 2]), "predict > high violates ADR-021"
```

---

## Dimension 4 — `save_state` / `load_state` manifest

If the code serialises explainer state:

| Check | Requirement |
|---|---|
| `manifest.schema_version` present | Must be int |
| `manifest.ce_version` present | CE package version string |
| `manifest.serialized_at` present | ISO timestamp |
| `manifest.checksum` present | SHA-256 of the payload (or equivalent) |
| Load validates manifest version before deserialising | Fail fast |
| Load raises `IncompatibleSchemaError` on unsupported version | Not silent |

---

## Dimension 5 — Migration guidance

When `schema_version` changes in a calibrator:

- [ ] New version added to the supported-version list in `from_primitive()`.
- [ ] Old version retained for the minimum migration window (usually one minor release).
- [ ] Migration notes added to `docs/migration/`.
- [ ] `RELEASE_PLAN_v1.md` updated with the new schema version note.

---

## Dimension 6 — Convenience helpers (optional but expected)

```python
# In-memory round-trip helpers simplify tests and tool usage
save_state_to_bytes() -> bytes
load_state_from_bytes(b: bytes) -> WrapCalibratedExplainer
```

---

## Audit Report Template

```
Serialization Audit: <module/class>
=====================================
to_primitive():
  schema_version present:         PASS / FAIL
  all values JSON-safe:           PASS / FAIL
  numpy arrays converted:         PASS / FAIL

from_primitive() error handling:
  unknown version → error:        PASS / FAIL
  missing version → error:        PASS / FAIL
  error message useful:           PASS / FAIL

Round-trip invariant (ADR-021):
  identical outputs:              PASS / FAIL
  interval ordering preserved:    PASS / FAIL

save_state / load_state (if present):
  manifest.schema_version:        PASS / FAIL / N_A
  manifest.checksum:              PASS / FAIL / N_A
  load validates version first:   PASS / FAIL / N_A

Migration guidance:
  docs/migration/ entry:          PRESENT / MISSING / N_A
  RELEASE_PLAN_v1.md updated:     YES / NO / N_A

Overall: CONFORMANT / NON-CONFORMANT (<N> issues)
```

---

## Evaluation Checklist

- [ ] `to_primitive()` passes `json.dumps()` without error.
- [ ] `from_primitive()` raises `IncompatibleSchemaError` on version mismatch.
- [ ] Round-trip test verifies identical `predict_proba` outputs (not just no-exception).
- [ ] Interval invariant verified on restored calibrator.
- [ ] Manifest fields present in `save_state` output.
- [ ] Migration guide entry present when schema version incremented.
