---
name: ce-serializer-impl
description: >
  Implement ADR-031 serializer and persistence APIs with versioned primitives and
  compatible load behavior.
---

# CE Serializer Implementation

You are implementing calibrator serialization or explainer state persistence
per ADR-031. The contract is: versioned JSON-safe primitives, fail-fast on
schema incompatibility, and -- as of the schema v3 security hardening --
**no pickle bytes anywhere on the `load_state()` call path.** A SHA-256
checksum recorded inside the same artifact proves internal consistency only;
it never proves who produced the artifact, so it must never gate whether
pickle-like bytes get deserialized. If you are extending this area, assume
the artifact is untrusted input and re-read ADR-031's "Trust boundary
(schema v3)" section before writing code.

Load `references/serialization_templates.md` for full code templates.

---

## Key contracts

### Calibrator `to_primitive` / `from_primitive`

All built-in calibrators must implement this pair:
- `to_primitive()` returns a dict with `schema_version` as the first key
  and only JSON-safe values (lists/numbers/strings from `.tolist()` etc.).
  `VennAbers`/`IntervalRegressor` also stash a `checksums.sha256` field
  computed from a pickle of internal state, but that pickle blob is never
  stored in the primitive and never fed to `pickle.loads()` -- it is a
  vestigial corruption-detection artifact, not part of the trust boundary.
  Do not add a calibrator that pickles its *restorable* payload.
- `from_primitive(payload)` validates `schema_version` and `calibrator_type`,
  and reconstructs the object purely from JSON-safe `fields`. Raises
  `ConfigurationError` on any mismatch with a guidance message.
- Calibrators that cannot implement a JSON-safe `to_primitive()` must **not**
  be silently pickled by `WrapCalibratedExplainer.save_state()`. It raises
  `SerializationError` instead (see `_calibrator_to_primitive` in
  `wrap_explainer.py`). Restoration (`_restore_calibrator_from_primitive`)
  dispatches `calibrator_type` against a fixed, explicit set of trusted
  restorers only -- never a dynamic import of an artifact-provided module
  path -- and rejects the legacy `calibrator_type="python_pickle"` value
  unconditionally, before any base64 decode or checksum check.

### Explainer `save_state` / `load_state` (schema v3)

- On-disk format: directory artifact with `manifest.json` plus
  `explainer_state.json` (JSON-safe wrapper/learner-identity/calibration
  config), optional `calibrator_primitive.json`, optional
  `preprocessing_mapping.json`. **No `wrapper.pkl` or any other pickle
  payload is ever written.**
- Manifest fields include `schema_version` (currently `3`), `created_at_utc`,
  `artifact_type`, and `files` (mapping `filename -> sha256`).
- `save_state(path)` writes the artifact directory and returns `Path`.
- `load_state(path, *, learner=None, preprocessor=None,
  difficulty_estimator=None, mc=None)` validates the manifest and every
  file's checksum, **then** reconstructs the wrapper from JSON alone plus the
  caller-supplied runtime objects (a generic fitted learner/custom
  preprocessor cannot be safely reconstructed from bytes, so the caller must
  supply them again; they are validated against persisted identity/shape
  metadata before use). Raises `IncompatibleStateError` for schema/manifest
  problems (including unconditional rejection of legacy schema v1/v2
  artifacts, before any pickle byte is read) and `ValidationError` for
  missing/incompatible caller-supplied objects.

### Round-trip invariant (ADR-031 §4)

After a save/load round-trip, the restored calibrator must produce **identical**
outputs (`np.allclose(ref, restored, atol=1e-9)`).

---

## Additional compatibility expectations

- Wrapper object round-trips via direct `pickle.dump/load` and
  `joblib.dump/load` (i.e. the caller pickling the Python object themselves,
  *not* going through `save_state()`/`load_state()`) should remain
  functional -- that is a different, well-understood trust boundary (the
  caller already trusts whatever produced the pickle bytes) than the
  ADR-031 state-artifact format.
- Explanation collection objects should remain pickleable.

## Out of Scope

- Third-party calibrator schema management - external packages own their schema versions.

## Evaluation Checklist

- [ ] `to_primitive()` returns a dict with `schema_version` as the first key.
- [ ] All values are JSON-safe (no numpy arrays, no non-serialisable objects).
- [ ] `from_primitive()` validates `schema_version`, `calibrator_type`, and checksum.
- [ ] `from_primitive()` raises `ConfigurationError` on any validation failure.
- [ ] Round-trip test verifies identical predictions (not just no-exception).
- [ ] `json.dumps(primitive)` test verifies JSON-safety.
- [ ] Checksum tamper test verifies `from_primitive()` rejects corrupted payloads.
- [ ] `save_state` / `load_state` manifest includes `schema_version`, `created_at_utc`, `artifact_type`, and `files`.
- [ ] `load_state` validates per-file sha256 checksums before deserialising.
- [ ] Unsupported state schema version raises `IncompatibleStateError`.
- [ ] Legacy (pre-v3) pickle-based schema versions are rejected unconditionally, before any pickle byte is read -- never accepted merely because a checksum matches.
- [ ] Unsupported calibrators fail closed (`SerializationError`) at save time instead of falling back to pickling.
- [ ] `load_state()` never calls `pickle.loads`/`pickle.load`/`joblib.load`/`cloudpickle.loads` on artifact-provided bytes; an AST-based guard test enforces this.
- [ ] Wrapper pickle and joblib round-trips are covered in integration tests (direct `pickle.dump/load`, not via `save_state`/`load_state`).
