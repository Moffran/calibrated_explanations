> **Status note (2026-11-04):** Last edited 2026-11-04 · Archive after: Retain indefinitely as architectural record · Implementation window: v1.1.0 (#227-#230, #232).

# ADR-041: Minimal Narrative Semantic Contract

Status: Accepted
Date: 2026-11-02
Deciders: Core maintainers
Reviewers: Core maintainers
Supersedes: None
Superseded-by: None
Related: ADR-002-validation-and-exception-design, ADR-005-explanation-json-schema-versioning, ADR-008-explanation-domain-model-and-compat, ADR-020-legacy-user-api-stability, ADR-021-calibrated-interval-semantics, ADR-029-reject-integration-strategy, ADR-037-visualization-extension-and-rendering-governance, ADR-040-capability-verification-framework, ADR-042-canonical-rule-and-feature-ranking

## Context

CE narrates explanations offline: `to_narrative()` (with the aliases
`to_dataframe()` and `narrate()`, and `plot(style="narrative")`) renders a
packaged YAML template through `core/narrative_generator.py` and
`viz/narrative_plugin.py`.

Describing an explanation correctly needs the calibrated prediction and its
interval, each rule's weight, interval and kind (factual or counterfactual),
the problem type, the class a probability refers to, the regression threshold,
the requested percentiles and the reject state. These facts are read ad hoc
from live objects, and the ADR-005 payload omits several of them, so a stored
explanation cannot be described without the explainer that produced it. There
is no deterministic record of what is true about an explanation against which
narration can be checked.

## Decision

### D1. Explanation results are authoritative

Computed explanation results (ADR-005 payload, ADR-008 domain model) are the
only source of narrative facts. Narrative code MUST NOT recompute prediction,
calibration or explanation analysis. It MAY copy values, compare a value with
an interval endpoint or zero, and order rules by ADR-042.

### D2. `NarrativeSpec` is an immutable, versioned fact record

`calibrated_explanations.explanations.NarrativeSpec` (frozen dataclasses in
`explanations/narrative_spec.py`) is built by
`build_narrative_spec(payload, *, rnk_metric=None, rnk_weight=0.5)` and
exposes:

- the calibrated prediction and its interval;
- one fact per rule: rule text, feature index or indices, feature value,
  weight interval, rule prediction interval, and whether the rule is
  counterfactual (alternative) or factual (factual and FAST);
- a JSON-pointer source reference into the payload for every fact;
- `context` (the payload's `metadata.narrative_context`) and `ranking`
  (the ADR-042 key);
- qualifiers narration must not omit: `rejected`, `unbounded_interval`,
  `context_unavailable` (spec level); `weight_interval_contains_zero`,
  `unbounded_interval` (rule level);
- `spec_version` (`1.0.0`) and `spec_id`.

Identical semantic input MUST produce identical canonical JSON
(`NarrativeSpec.to_json()`: sorted keys, compact separators, shortest
round-trip floats, infinities as `"inf"`/`"-inf"`); `spec_id` is the SHA-256
of that JSON without `spec_id`. Numbers are not rounded, and equal integers and
floats are the same input. Any change to fields or their meaning bumps
`spec_version`.

The builder raises `ValidationError` (ADR-002) for missing, non-numeric, `NaN`
or contradictory data: unknown explanation type or task, `low > high`, a value
outside its interval (subject to D4), `problem_type` contradicting `task`, a
threshold present outside probabilistic regression or absent within it,
invalid percentiles, a malformed reject record, or ranking arguments that do
not apply.

### D3. `metadata.narrative_context` extends ADR-005 additively

`to_json()` and `to_json_stream()` MUST emit `metadata.narrative_context` on
every explanation for which CE knows at least one of these facts:

| Key | When present | Source |
|---|---|---|
| `problem_type` | mode known | explainer mode, multiclass flag, threshold |
| `class_index`, `class_label` | classification | `prediction["classes"]`, class labels |
| `threshold` | probabilistic regression | `y_threshold` |
| `low_high_percentiles` | percentile regression | requested percentiles |
| `reject` | reject policy active | `RejectContext` |

Unavailable keys are omitted, never inferred. No existing field changes and no
v2 schema cycle is opened. Emission is unconditional so stored payloads are
self-describing. Older payloads remain valid: the builder records
`context_unavailable` and raises only when a needed fact is undefined without
context (the default `ensured` ranking of a regression alternative).

### D4. Interval semantics follow ADR-021

The builder checks `low <= predict <= high` for the prediction, every weight
interval and every rule prediction, with the serialization tolerance. For
percentile regression it applies the ADR-021 §4 percentile scope, mirrored for
factual and FAST weights (`predict - perturbed`). An infinite endpoint, or the
`min(y_cal)`/`max(y_cal)` placeholder for an infinite percentile, is marked
`low_bounded=False` or `high_bounded=False` with its value unchanged.
Narration MUST NOT present an unbounded endpoint as a numeric limit.

### D5. ADR-042 governs rule order

`NarrativeSpec.rules` is in ADR-042 canonical order (D1, D5, D6), strongest
first, and drops no rules. `ranking` records `feature_weight` for factual and
FAST explanations, and `ensured` (with its weight) or `feature_weight` for
alternatives; `uncertainty` is recorded as `ensured` with weight `0.0`. The
builder delegates to the single ADR-042 ranking implementation once #204
provides it.

### D6. Offline narration and its public API are preserved

`to_narrative()`, `to_dataframe()`, `narrate()` and `plot(style="narrative")`
keep their signatures, defaults, return types and formats, and work offline.
Prose is not contracted byte-for-byte (ADR-020 does not list `to_narrative`),
but wording or order changes MUST be documented and tested. ADR-042 D16 owns
changes to narrative rule selection and order. The template engine MAY later
read a `NarrativeSpec` if that causes no regression.

### D7. No providers or external services

Narration MUST NOT add a plugin family, provider registry, dispatch, trust
orchestration, credentials, network configuration or provider-facing API. It
remains a plot style under ADR-037 with no new dependency or network access.

### D8. No correctness guarantee for prose

A correct `NarrativeSpec` does not make text built from it correct. CE MUST NOT
describe narrative prose as calibrated or verified: ADR-021 guarantees attach
to predictions and intervals, and ranks carry no guarantee (ADR-042 D14).

## Governed claims

- `CE-CAP-NARR-001` — offline template narration keeps its public API and formats (D6) and makes no correctness claim about prose (D8).

## Alternatives Considered

1. **Narrative provider plugin family.** Rejected: provider dispatch, trust
   and prose verification serve no need of offline narration and add a public
   surface to keep stable.
2. **No spec; keep reading live objects.** Rejected: stored payloads cannot be
   described, and narration has nothing deterministic to be tested against.
3. **ADR-005 v2 schema cycle.** Rejected: the v1 freeze permits additive
   `metadata` keys.
4. **Opt-in `narrative_context`.** Rejected: adds API surface and leaves
   payloads that can never be narrated correctly.
5. **Rewrite placeholder endpoints to `±inf`.** Rejected: changes reported
   values; a bounded flag records the ADR-021 meaning instead.

## Consequences

### Positive

- Stored payloads carry the context needed to describe them.
- One deterministic, versioned, source-referenced fact record for narration
  and tests.
- No new dependency, plugin family, network path or provider API.

### Negative / Risks

- `metadata.narrative_context` and `NarrativeSpec` v1 are stable surfaces that
  must be versioned when changed.
- Until #204, `NarrativeSpec.rules` and the template narration may list rules
  in different orders.
- Off-centre one-sided regression queries (e.g. percentiles `(-inf, 40)`)
  fail the serialization invariant check, so no spec can be built for them
  (#237).

## Adoption & Migration

- v1.1.0: #228 (`narrative_context`), #229 and #230 (`NarrativeSpec`), #232
  (offline narration compatibility).
- #204 makes the builder delegate ranking and changes template narration per
  ADR-042 D16.
- Additive only; no deprecation or migration. Changes are recorded in
  `CHANGELOG.md`.
- Verification: `tests/unit/explanations/test_narrative_spec.py` and
  `tests/capabilities/test_narrative_contracts.py`.

## Open Questions

None.
