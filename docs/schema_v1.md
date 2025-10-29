# Explanation Schema v1

This page summarizes the stable JSON contract for serialized explanations in v0.6.0 (ADR-005).

- Schema file: `calibrated_explanations/schemas/explanation_schema_v1.json`
- Version: `schema_version = "1.0.0"`
- Purpose: round-trip Explanation domain objects to a portable JSON payload.

## Top-level fields

- `schema_version` (string, optional; recommended): fixed `"1.0.0"`.
- `task` (string): e.g., `classification` or `regression`.
- `index` (integer): item index the explanation refers to.
- `prediction` (object): calibrated prediction values, e.g., `{predict, low, high}`.
- `rules` (array of objects): feature rules composing the explanation.
- `provenance` (object|null): optional metadata (library version, timestamp, etc.).
- `metadata` (object|null): additional caller-provided info.

## Rule fields (per entry in `rules`)

- `feature` (integer): feature index.
- `rule` (string): human-readable rule string.
- `weight` (object): feature weight summary (typically `{predict, low, high}`).
- `prediction` (object): rule-level prediction summary (same shape as top-level prediction).
- `instance_prediction` (object|null): instance-specific prediction (optional).
- `feature_value` (any): the instance feature value (optional).
- `is_conjunctive` (boolean): whether this rule is part of a conjunction.
- `value_str` (string|null): human-readable value.
- `bin_index` (integer|null): Index of the discretization bin (used when creating the explanation) that contains the instance’s feature value (when available).

## Validation

If `jsonschema` is installed, `calibrated_explanations.serialization.validate_payload(obj)` can validate a payload against the schema.

## Examples

See the unit tests for round-trip examples:

- `tests/test_serialization_and_quick.py::test_domain_json_round_trip_and_schema_validation`
- `tests/test_serialization_and_quick.py::test_adapter_legacy_to_json_round_trip`
