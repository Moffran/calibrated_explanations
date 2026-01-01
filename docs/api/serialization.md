# Serialization & schema reference

Calibrated Explanations serializes explanations to a JSON payload governed by
an explicit schema contract.

## Explanation payload schema

- **Schema file**: `src/calibrated_explanations/schemas/explanation_schema_v1.json`
- **Reference doc**: `docs/schema_v1.md`
- **Validator**: `calibrated_explanations.schema.validate_payload`

The v1 payload schema is the canonical contract. Optional `provenance` and
`metadata` fields allow runtime context to be attached without breaking the
schema.

## PlotSpec serialization

Plot specifications are serialized via the PlotSpec dataclasses in
`src/calibrated_explanations/viz/plotspec.py` and the JSON helpers in
`src/calibrated_explanations/viz/serializers.py`.

## Related ADRs

- ADR-005 (explanation payload schema versioning)
- ADR-007 (visualization abstraction)
- ADR-016 (PlotSpec separation)
