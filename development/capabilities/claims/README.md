# Capability Claims

This directory contains CE capability claim files for the `calibrated_explanations` library.

Each claim file documents a user-visible statement about what CE provides, following
the verification chain defined in `development/README.md`:

```text
Capability claim
    -> requirement
    -> verification case
    -> evidence record
```

## Location authority

This directory is the canonical location for CE capability claims.
See `development/README.md` for the full location map.

## File naming

Claim files use the prefix `CE-CAP-` and are stored as YAML:

```
CE-CAP-<AREA>-<NNN>.yaml
```

Examples: `CE-CAP-EXPL-001.yaml`, `CE-CAP-PRED-001.yaml`

## Claim schema (illustrative)

```yaml
claim_id: CE-CAP-EXPL-001
claim_type: capability
owner: calibrated_explanations
status: current
claim_text: >
  One-sentence statement of the user-visible behavior.
public_api:
  - WrapCalibratedExplainer.explain_factual
requirements:
  - CE-REQ-EXPL-API-001
verification:
  proves:
    - api_contract
evidence_required:
  - commit_sha
  - package_version
  - test_id
  - result
```

## Rules

1. Claims are not requirements. Do not write acceptance criteria in a claim.
2. Every claim must have an `owner` and at least one requirement ID.
3. Do not duplicate definitions that already exist in another claim.
4. Statistical claims must state their assumptions (calibration data,
   exchangeability, task-type scope, empirical vs theoretical boundary).
5. Do not mark roadmap or unsupported behavior as `status: current`.
6. Claims describe existing CE behavior only — they do not introduce new functionality.

---

## Structuring guide: when one operation spans multiple object types

### Rule C-1 — Claims describe capabilities, not implementations

A claim covers the full capability regardless of which concrete class exposes it.
Write ONE claim per conceptual capability group, even if the method exists on multiple
classes (e.g., collection and individual, factual and alternative).

```
WRONG: CE-CAP-EXPL-CONJ-FAC-001  (conjunctions — factual collection only)
       CE-CAP-EXPL-CONJ-ALT-001  (conjunctions — alternative collection only)

RIGHT: CE-CAP-EXPL-CONJ-001      (conjunctions — all applicable types)
```

### Rule C-2 — public_api lists all first-class entry points

List every public_api entry point users are expected to call directly.
Do not list implementation helpers that are only called transitively.

### Rule C-3 — requirements list must be exhaustive

The `requirements` list in a claim must reference every requirement that
derives from it. Requirements are separated by **operation** (see R-2), not by
object level (collection vs individual) — a single requirement covers all
applicable object levels and declares them in its `applicable_on` field.

```yaml
requirements:
  - CE-REQ-EXPL-CONJ-001   # collection and individual (see applicable_on in requirement)
```

### Rule C-4 — operation families share one claim

When several operations share the same conceptual purpose and similar parameter
signatures (e.g., super/semi/counter/ensured/pareto all filter alternative
explanations by different criteria), use ONE claim for the family. The
requirements list then contains one entry per distinct operation.

### Rule C-5 — use probabilistic_regression for threshold-based regression queries

`probabilistic_regression` is a distinct task type: a regression model is queried
with `threshold=` to return P(Y > threshold | X) rather than a point estimate or
interval. Valid task type values and when to use them:

| Task type | When to use |
|---|---|
| `binary_classification` | Two-class classification model |
| `multiclass_classification` | More than two classes |
| `regression` | Continuous output model: point estimates, UQ intervals |
| `probabilistic_regression` | Regression model queried with `threshold=` for P(Y > threshold) |

List `probabilistic_regression` explicitly when a capability applies to regression
models in threshold mode. A capability that applies to all regression modes must list
BOTH `regression` AND `probabilistic_regression`. Do NOT list `probabilistic_regression`
for capabilities that are specific to non-threshold regression (e.g., UQ intervals via
`predict(X, uq_interval=True)`) or to classification.

## Related locations

| Material | Location |
|---|---|
| Requirements derived from claims | `development/capabilities/requirements/` |
| Verification scenarios and helpers | `development/capabilities/verification/` |
| Pytest capability tests | `tests/capabilities/` |
| Generated verification run outputs | `reports/verification/` |
| Curated capability evidence summaries | `development/capabilities/evidence/` |
