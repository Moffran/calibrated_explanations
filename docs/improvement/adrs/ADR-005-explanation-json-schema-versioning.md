> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-005: Explanation JSON Schema Versioning

Status: Accepted (revised for v0.10.1 scope)
Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Calibrated explanations are serialized to JSON for persistence, interoperability, and
visual tooling. The current implementation already ships a stable, versioned payload
schema for instance-level explanations, but the ADR previously specified a broader,
more generic envelope than the codebase supports today. This revision narrows the
scope to what is actively used and validated, while leaving room for future extension.

## Decision

Adopt a **calibrated-explanations-first** JSON schema contract for instance-level
feature attribution explanations, with explicit schema versioning and minimal required
metadata. The v1 contract describes the **payload** shape used by the library and
fixtures; additional envelope layers are deferred until there are concrete runtime
producers and consumers.

### Envelope (deferred, optional)

A top-level envelope (e.g., `{schema_version, type, generator, meta, payload}`) is not
required for v1.0.0 and is treated as a **future extension**. Implementations MAY wrap
payloads in an envelope for external tooling, but the core library will continue to
validate and round-trip the payload itself.

### Payload (v1.0.0, required)

The v1 payload is the authoritative JSON contract for calibrated explanations and is
aligned with the current serialization and fixtures:

```json
{
  "schema_version": "1.0.0",
  "task": "classification",
  "index": 0,
  "explanation_type": "factual",
  "prediction": {"predict": 0.8, "low": 0.7, "high": 0.9},
  "rules": [
    {
      "feature": 3,
      "rule": "feature <= 1.2",
      "rule_weight": {"predict": 0.12, "low": 0.08, "high": 0.16},
      "rule_prediction": {"predict": 0.81, "low": 0.73, "high": 0.88}
    }
  ],
  "provenance": null,
  "metadata": null
}
```

Rules:

- `schema_version` follows semver for the **payload schema** (independent from library
  version) starting at 1.0.0.
- Minor increments allow additive, backwards-compatible fields (consumers ignore unknown
  keys).
- Major increments indicate breaking structural changes.
- `explanation_type` is required and must be `"factual"` or `"alternative"` as defined
  in ADR-008; this is the only explanation-type discriminator in v1.
- `prediction` and rule-level `rule_weight` / `rule_prediction` carry calibrated
  predictions with uncertainty intervals as required by the CE papers.
- `provenance` and `metadata` are optional extension points for library/caller context
  (e.g., version, timestamps, hashes); there is no dedicated `generator` block in v1.

### Operational metadata guidance (optional)

To support auditability and operational governance without changing the payload
contract, v1.0.0 treats `provenance` and `metadata` as the **official extension
surface** for runtime context. This keeps the CE-First principle intact while allowing
wrappers and downstream systems to attach audit context.

Recommended (optional) fields for runtime tooling:

- `provenance.library_version`: the runtime version used to produce the payload.
- `provenance.calibration_version`: a deterministic identifier for the calibration
  state (e.g., checkpoint/rollback ID).
- `provenance.parameters_hash`: a stable hash of non-sensitive configuration inputs
  needed for reproducibility (model ID, bins, thresholds), without leaking data.
- `metadata.audit`: audit metadata (correlation ID, policy version, evidence pack
  reference).
- `metadata.tenant_id`: tenant/customer identifier for multi-tenant deployments.
- `metadata.security`: encryption or key-management hints (e.g., key alias) when
  operating in restricted environments.

These fields are **optional** and must never change the mathematical explanation
outputs. They are intended for audit trails, compliance evidence, and observability.

### Types and registries (deferred)

A multi-type registry (`feature_attribution`, `interval`, `global_importance`,
`calibration_diagnostics`) is **not** part of v1.0.0. The only supported payload is the
instance-level feature attribution explanation described above. Any future additions
must come with:

- explicit consumers and fixtures in the codebase,
- per-type schema files,
- and a corresponding validation strategy.

## Validation

- Provide `validate_payload(obj)` that checks required fields via JSON Schema when
  `jsonschema` is installed.
- Enforce interval invariants (`low <= predict <= high`) during serialization for
  top-level predictions and rule-level predictions.
- Envelope validation is **out of scope** for v1.0.0.

## Alternatives Considered

1. **Full envelope now** — Rejected for v1.0.0 because it is not implemented in the
   current codebase and would add complexity without concrete producers/consumers.
2. **No schema/versioning** — Rejected because downstream tooling needs a stable,
   forward-compatible contract.
3. **Protocol Buffers / Avro** — Rejected for initial adoption; JSON is sufficient for
   the current portability needs.

## Consequences

Positive:

- Schema matches the actual calibrated explanation payload used by serialization and
  fixtures.
- Reduced implementation scope makes v0.10.1 delivery more tractable.
- Clear, explicit extension points (provenance/metadata) without enforcing unused
  envelope fields.

Negative / Risks:

- External tooling wanting a richer envelope must add its own wrapper for now.
- Future addition of multiple payload types will require a new schema revision.

## Adoption & Migration

- v1.0.0 formalizes the existing payload as the stable contract.
- Legacy flat dicts should be mapped into this payload structure via adapters.
- If an envelope is introduced later, it must not invalidate the v1 payload contract
  and must preserve round-trip serialization for existing fixtures.

## Open Questions

- When concrete non-feature-attribution payloads exist, should they be a new major
  schema version or a separate registry with its own versioning?
- Should provenance include a standardized hash for reproducibility (e.g., parameters
  hash), or is it better left to downstream tooling?

## Decision Notes

Revisit after the first external consumer feedback cycle and after v0.10.1 ships with
payload-level validation and fixtures aligned to the v1 schema.
