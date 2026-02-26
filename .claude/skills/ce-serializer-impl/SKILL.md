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

Load `references/serialization_templates.md` for full code templates.

---

## Key contracts

### Calibrator `to_primitive` / `from_primitive`

All built-in calibrators must implement this pair:
- `to_primitive()` returns a dict with `schema_version` as the first key
  and only JSON-safe values (no numpy arrays — convert to `list`).
- `from_primitive(payload)` raises `IncompatibleSchemaError` on version
  mismatch with guidance message.

### Explainer `save_state` / `load_state`

- On-disk format: JSON document with top-level manifest (`schema_version`,
  `ce_version`, `serialized_at`, `checksum`) and nested calibrator primitives.
- `save_state` accepts `str | Path | IO[bytes]`.
- `load_state` validates manifest schema version and fails fast if unsupported.

### Round-trip invariant (ADR-031 §4)

After a save/load round-trip, the restored calibrator must produce **identical**
outputs (`np.allclose(ref, restored, atol=1e-9)`).

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
