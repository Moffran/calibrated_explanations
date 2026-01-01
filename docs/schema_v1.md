# Explanation Schema v1

This page summarizes the stable JSON contract for serialized explanations in v1.0.0 (ADR-005).

- Schema file: `calibrated_explanations/schemas/explanation_schema_v1.json`
- Version: `schema_version` (string, optional; recommended)
- Purpose: round-trip Explanation domain objects to a portable JSON payload.

## Top-level fields

- `schema_version` (string, optional; recommended): a suggested schema version string (e.g. `"1.0.0"`). The schema does not require a fixed literal; including the version is recommended for interoperability.
- `task` (string): e.g., `classification` or `regression`.
- `index` (integer): item index the explanation refers to.
- `explanation_type` (string): either `"factual"` or `"alternative"`.
- `prediction` (object): calibrated prediction values from the underlying model, e.g., `{predict, low, high}` with uncertainty interval. This is always the factual calibrated prediction for reference.
- `rules` (array of objects): feature rules composing the explanation.
- `provenance` (object|null): optional metadata (library version, timestamp, etc.). Recommended minimal keys: `library_version`, `created_at` (ISO8601), and `generator` (e.g., plugin or pipeline id). Note: these are conventions only and not validated by the schema.
- `metadata` (object|null): additional caller-provided info.

`provenance` and `metadata` are the stable extension surface for runtime context; the v1 contract is payload-only and does not require an outer envelope.

## Rule fields (per entry in `rules`)

- `feature` (integer): feature index.
- `rule` (string): human-readable rule string representing the condition (factual for factual explanations, alternative for alternative explanations).
- `rule_weight` (object):
  - **Factual explanations** – calibrated feature weight summary with uncertainty bounds, always exposing `{predict, low, high}` to match the CE paper definition. The rule represents a factual condition covering the feature’s instance value.
  - **Alternative explanations** – optional metadata capturing the delta from the factual baseline for ranking/metadata; consumers should rely on the `rule_prediction` field for decision making.
- `rule_prediction` (object):
  - **Factual explanations** – optional metadata used in legacy exports. The calibrated prediction for the instance lives at the top level.
  - **Alternative explanations** – calibrated prediction estimate plus uncertainty interval for the alternative condition. This is the primary quantity mandated by the CE papers. The rule represents an alternative condition covering alternative instance values for the feature.
- `instance_prediction` (object|null): instance-specific prediction (optional).
- `feature_value` (any): the instance feature value (optional).
- `is_conjunctive` (boolean): whether this rule is part of a conjunction.
- `value_str` (string|null): human-readable value.
- `bin_index` (integer|null): Index of the discretization bin (used when creating the explanation) that contains the instance’s feature value (when available).

## Validation

If `jsonschema` is installed, `calibrated_explanations.schema.validate_payload(obj)` can validate a payload against the schema. Note that semantic invariants such as the inclusive interval requirement (`low <= predict <= high`) are enforced by the library at serialization time (see `calibrated_explanations.serialization._validate_invariants`).

## Examples

See the tests for round-trip examples:

- `tests/integration/core/test_golden_fixture.py::test_golden_fixture_roundtrip_and_validate`
