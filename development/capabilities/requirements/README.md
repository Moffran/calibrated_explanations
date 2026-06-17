# Capability Requirements

This directory contains CE requirement files derived from capability claims in
`development/capabilities/claims/`.

Each requirement file translates one or more capability claims into specific,
testable obligations on the CE public API, following the verification chain
defined in `development/README.md`:

```text
Capability claim
    -> requirement
    -> verification case
    -> evidence record
```

## Location authority

This directory is the canonical location for CE requirements.
See `development/README.md` for the full location map.

## File naming

Requirement files use the prefix `CE-REQ-` and are stored as Markdown:

```
CE-REQ-<AREA>-<FACET>-<NNN>.md
```

Examples: `CE-REQ-EXPL-API-001.md`, `CE-REQ-PRED-API-001.md`

## Requirement structure

Each requirement file must state:

- **requirement_id**: unique CE-REQ-... identifier
- **obligation_type**: one of `api_contract`, `payload_schema`, `numerical_behavior`,
  `statistical_method_alignment`, `documentation_boundary`, `visualization_behavior`,
  `plugin_behavior`, `empirical_smoke`
- **claim_refs**: which CE-CAP-... claims this requirement serves
- **scope**: public API surface, task type, and workflow applicable
- **observable_behavior**: what must be true when the requirement is satisfied
- **acceptance_criterion**: the measurable or checkable condition
- **verification_method**: how the criterion is checked (automated test, structural
  check, analytical review, etc.)
- **test_refs**: which tests in `tests/capabilities/` (or linked nearby tests) verify
  this requirement
- **evidence_required**: metadata that a passing evidence record must include

## Rules

1. Requirements are not tests. Do not embed test code in requirement files.
2. Every requirement must have at least one `claim_ref`.
3. Every requirement must have a stated `verification_method` and `acceptance_criterion`.
4. Acceptance criteria must be visible here, not hidden inside test code only.
5. Statistical obligations must state their assumptions explicitly.

---

## Structuring guide: when one operation spans multiple object types

### Rule R-1 — One requirement per OPERATION, not per class

Requirements decompose by **operation** (what is being called), not by the concrete
class it is called on. A single requirement covers the same operation on all object
types listed in its scope.

```
WRONG: CE-REQ-EXPL-FILTER-SUPER-COL-001  (super on collection only)
       CE-REQ-EXPL-FILTER-SUPER-IND-001  (super on individual only)

RIGHT: CE-REQ-EXPL-FILTER-SUPER-001      (super on both collection and individual)
       — acceptance criterion has separate entries for each object type
```

Exception: if the operation has **materially different contracts** on different object
types (different return types, different preconditions, or different failure modes that
users must handle separately), then separate requirements are appropriate.

### Rule R-2 — Always separate requirements for SEPARATE OPERATIONS

Operations that are semantically distinct (e.g., `super_explanations`, `semi_explanations`,
`counter_explanations`, `ensured_explanations`, `pareto_explanations`) must be separate
requirements even when they share parameters and return types.

```
WRONG: CE-REQ-EXPL-FILTER-001  (lumps super + semi + counter + ensured + pareto)

RIGHT: CE-REQ-EXPL-FILTER-SUPER-001
       CE-REQ-EXPL-FILTER-SEMI-001
       CE-REQ-EXPL-FILTER-COUNTER-001
       CE-REQ-EXPL-FILTER-ENSURED-001
       CE-REQ-EXPL-FILTER-PARETO-001
```

### Rule R-3 — State applicable object level; do not split requirements on it

When a method exists on both a collection type and an individual explanation type,
use ONE requirement that covers both. State which object levels apply in the
`applicable_on` field of the Metadata table:

```markdown
| applicable_on | collection (CalibratedExplanations, AlternativeExplanations) and individual (FactualExplanation, AlternativeExplanation) |
```

The acceptance criterion must contain separate sub-entries for collection and
individual to ensure each is independently verifiable. Tests can be separate
functions within the same test file — one per object level.

Do NOT create separate requirements just because the same operation is callable
on both a collection and an individual.

```
WRONG: CE-REQ-EXPL-CONJ-COL-001  (add_conjunctions on collection only)
       CE-REQ-EXPL-CONJ-IND-001  (add_conjunctions on individual only)

RIGHT: CE-REQ-EXPL-CONJ-001      (add_conjunctions; applicable_on: collection and individual)
```

Exception: if the operation has **materially different contracts** on different
object types (different return types, different preconditions, or different failure
modes that users must handle separately), then separate requirements are appropriate.

### Rule R-4 — Aliases do not need separate requirements

When a short-form alias delegates directly to the canonical method (e.g., `.super()`
delegates to `.super_explanations()`), one requirement covers both. State the alias
explicitly in scope and note "alias delegator — verified by the canonical test."

### Rule R-5 — Parameter variants: assertions within the same requirement

A parameter that selects a **different code path** (e.g., `max_rule_size=1` disables
conjunction generation; `max_rule_size=2` enables pairs) requires coverage in the
requirement's acceptance criterion and tests. Use `pytest.mark.parametrize` to cover
meaningful parameter values **within the SAME requirement**. Do NOT create a separate
requirement just because a test uses a different parameter value.

A parameter that only changes the **count or size of output** (e.g., `n_top_features`
controlling how many features are considered) does not require a separate requirement.

### Rule R-6 — Each requirement must have at least one test

Every requirement file must reference at least one named test in `tests/capabilities/`.
Tests for a family of related requirements can live in one test file, but each
requirement must have its own named test function.

## Related locations

| Material | Location |
|---|---|
| Capability claims that generate requirements | `development/capabilities/claims/` |
| Verification scenarios and helpers | `development/capabilities/verification/` |
| Pytest capability tests | `tests/capabilities/` |
| Generated verification run outputs | `reports/verification/` |
| Curated capability evidence summaries | `development/capabilities/evidence/` |
