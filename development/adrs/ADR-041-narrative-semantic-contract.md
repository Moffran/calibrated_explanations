---
id: "ADR-041"
title: "Minimal Narrative Semantic Contract"
status: "accepted"
description: "Governing the minimal semantic contract for offline explanation narration without a separate narrative fact model."
supersedes: []
superseded_by: []
---
> **Status note (2026-11-25):** Last edited 2026-11-25 (revised: no separate `NarrativeSpec` contract; canonical selection via ADR-042 D17; #227) · Archive after: Retain indefinitely as architectural record · Implementation window: v1.1.0 (#227-#230, #232).

# ADR-041: Minimal Narrative Semantic Contract

Date: 2026-11-02 (revised 2026-11-25)
Deciders: Core maintainers
Reviewers: Core maintainers
Related: ADR-002-validation-and-exception-design, ADR-005-explanation-json-schema-versioning, ADR-008-explanation-domain-model-and-compat, ADR-020-legacy-user-api-stability, ADR-021-calibrated-interval-semantics, ADR-029-reject-integration-strategy, ADR-037-visualization-extension-and-rendering-governance, ADR-040-capability-verification-framework, ADR-042-canonical-rule-and-feature-ranking

## Context

CE narrates explanations offline: `to_narrative()` (with the aliases
`to_dataframe()` and `narrate()`, and `plot(style="narrative")`) renders a
packaged YAML template through `core/narrative_generator.py` and
`viz/narrative_plugin.py`.

Describing an explanation correctly needs the calibrated prediction and its
interval, each rule's weight and interval, the explanation type, the problem
type, the class a probability refers to, the regression threshold, the
requested percentiles and the reject state. Before v1.1.0 the ADR-005 payload
omitted several of these facts, so a stored explanation could not be described
without the explainer that produced it.

### 2026-11-25 reassessment (#227)

The 2026-11-02 version of this ADR also introduced `NarrativeSpec`, an
immutable, versioned fact record with its own canonical JSON, `spec_id`
(SHA-256), qualifiers, bounded flags and builder `build_narrative_spec`. The
v1.1.0 scope review reopened that decision. An inventory of actual consumers
on `main` (2026-11-25) found:

| Candidate consumer | Uses `NarrativeSpec`? |
|---|---|
| Template narration (`core/narrative_generator.py`, `viz/narrative_plugin.py`) | No; it reads explanation objects and ADR-042 ranking directly |
| Plots, `__repr__`, conjunction selection | No; they use the ADR-042 ranking (`explanations/_ranking.py`) |
| Serialization (`to_json`, `from_json`, invariant check) | No; `narrative_context` and ADR-021 invariants are handled there |
| Capability claims (`CE-CAP-NARR-001`, `CE-REQ-NARR-API-001`) | No; the claim covers `to_narrative()` only |
| Released packages (v1.0.0-v1.0.2 tags) | Not present; never published |
| Downstream code (public GitHub code search) | None besides forks of this repository |
| Remaining references | Its own unit tests, one assertion in a capability test, two docs sections, the unreleased `CHANGELOG.md` entry |

Each field of the spec is either already in the payload or derivable from it
without CE-specific logic:

- prediction, weight and rule-prediction intervals: copied verbatim from the
  payload;
- `context`: `metadata.narrative_context` itself;
- bounded flags: since #237 (T10) open one-sided endpoints are serialized as
  `-inf`/`inf`, so "unbounded" is `not math.isfinite(value)`; the
  `min/max(y_cal)` placeholder the flags compensated for no longer exists;
- qualifiers: `rejected` is `narrative_context.reject.rejected`,
  `context_unavailable` is the absence of the key, and the zero/unbounded
  qualifiers are one comparison each;
- `source`: the JSON pointer is the rule's position in the payload;
- interval validation: `to_json()` already enforces ADR-021 containment with
  the percentile scope (`serialization._validate_invariants`), so the builder
  duplicated it;
- `spec_version`, `spec_id`, canonical JSON: no consumer compares or stores
  them.

The only behaviour not otherwise available is ADR-042 canonical selection
and order for a *stored* payload: payloads keep native rule order (ADR-042
D13), and CE's only public ranking API, `rank_features`, works on
caller-built arrays and leaves the `ensured` score, its probability
orientation and the base-identical filter to the caller. `from_json()` returns the ADR-008
`models.Explanation` dataclass, which has no ranking, plot or narrative
method, and the private `_rules_with_impact` needs a live explanation bound to
its explainer. An earlier draft of this revision said the order could be
obtained from the deserialized domain model; that is not true.

The maintainers decided on 2026-11-25 that external consumers MUST be able to
obtain exactly the selection and order CE's own narratives and plots use,
without private API. That need is met by a ranking interface over payload rule
indices (ADR-042 D17), not by a narrative record: it needs none of the spec's
other fields, and keeping the spec would freeze a second schema, version and
hash identity plus a duplicate validator.

The same review found that `to_json()` takes each rule's `rule_weight` and
`rule_prediction` from feature-indexed arrays rather than from the rule
itself, so for factual and alternative explanations, and for conjunctive
rules, the serialized values can belong to another feature (#262, present in
v1.0.2). The payload can only be authoritative (D1) once that is fixed.

## Decision

### D1. Explanation results are authoritative

Computed explanation results (ADR-005 payload, ADR-008 domain model) are the
only source of narrative facts. Narrative code MUST NOT recompute prediction,
calibration or explanation analysis. It MAY copy values, compare a value with
an interval endpoint or zero, and select and order rules by ADR-042.

The payload is authoritative only if it restates the explanation faithfully:
each serialized rule's values MUST be those of the same rule-table entry as its
`rule` text, the entry that plots and narratives use (#262).

### D2. No separate narrative fact contract

*Revised 2026-11-25 (#227).* CE MUST NOT ship a separate narrative fact
record. The self-describing ADR-005 payload (with D3) is the narrative
semantic interface for stored explanations, and the live explanation objects
(`to_narrative()`, `plot()`) are the interface for live ones. The unreleased `NarrativeSpec`, `NarrativeRuleFact`,
`NarrativeInterval`, `NARRATIVE_SPEC_VERSION` and `build_narrative_spec`, with
their canonical JSON, `spec_id`, qualifiers and builder-side validation, are
removed before v1.1.0 (#229, #230). Because they never appeared in a release,
removal needs no ADR-011 deprecation.

Canonical rule selection and order for a payload is provided by ADR-042 D17
as indices into the payload's `rules` array; it is a ranking interface, not a
narrative record, and carries no narrative fields.

A narrative fact record or adapter MAY be proposed later only with a concrete
consumer whose requirement the payload plus the existing ADR-021 validation
and the ADR-042 ranking, including D17, cannot meet. Such a proposal
MUST state that requirement, carry only the fields it needs, reuse the
authoritative validation and ranking, and add a version or hash identity only
if the consumer needs one.

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
self-describing. Payloads without `narrative_context` (written before v1.1.0)
remain valid and load unchanged. `narrative_context` is a stable additive
ADR-005 surface: changing a key's meaning requires an ADR-005 schema change.

### D4. Interval semantics follow ADR-021

Narrative facts carry ADR-021 interval semantics unchanged. `to_json()`
enforces `low <= predict <= high` with the ADR-021 §4 percentile scope taken
from `narrative_context.low_high_percentiles`; no second validator is added for
narration. An open one-sided endpoint is serialized as `-inf` or `inf`
(#237) and is not a calibrated finite bound.

### D5. ADR-042 governs rule selection and order

Narration selects and orders rules through the single ADR-042 ranking
implementation (ADR-042 D9, D16). Serialized payloads keep native rule order
(ADR-042 D13). A consumer of a stored payload, including one that writes its
own narrative text, obtains the canonical selection and order with
`canonical_rule_indices` (ADR-042 D17) and MUST NOT re-rank (ADR-042 D15). The
deserialized ADR-008 domain model does not provide a ranking.

*Corrected 2026-11-25:* an earlier wording of this revision said that order is
obtained from the domain model.

### D6. Offline narration and its public API are preserved

`to_narrative()`, `to_dataframe()`, `narrate()` and `plot(style="narrative")`
keep their signatures, defaults, return types and formats, and work offline.
Prose is not contracted byte-for-byte (ADR-020 does not list `to_narrative`),
but wording or order changes MUST be documented and tested. ADR-042 D16 owns
changes to narrative rule selection and order. The template engine is not
rewritten.

### D7. No providers or external services

Narration MUST NOT add a plugin family, provider registry, dispatch, trust
orchestration, credentials, network configuration or provider-facing API. It
remains a plot style under ADR-037 with no new dependency or network access.

### D8. No correctness guarantee for prose

Correct facts do not make text built from them correct. CE MUST NOT describe
narrative prose as calibrated or verified: ADR-021 guarantees attach to
predictions and intervals, and ranks carry no guarantee (ADR-042 D14).

## Governed claims

- `CE-CAP-NARR-001` — offline template narration keeps its public API and formats (D6) and makes no correctness claim about prose (D8).

## Alternatives Considered

1. **Narrative provider plugin family.** Rejected: provider dispatch, trust
   and prose verification serve no need of offline narration and add a public
   surface to keep stable.
2. **Keep `NarrativeSpec` as a stable, versioned, hashed fact record (the
   2026-11-02 decision).** Rejected on 2026-11-25: no supported consumer uses
   it; every field is in the payload or a one-line derivation; it duplicates
   the serialization invariant check; and it would freeze a second schema,
   version and hash identity.
3. **Reduce `NarrativeSpec` to a minimal passive adapter.** Rejected: the one
   demonstrated requirement, canonical selection for stored payloads, is met
   by ADR-042 D17 as rule indices; no requirement justifies the other fields,
   so they would be speculative public surface.
4. **No `narrative_context`; keep reading live objects.** Rejected: stored
   payloads could not be described without the explainer.
5. **ADR-005 v2 schema cycle.** Rejected: the v1 freeze permits additive
   `metadata` keys.
6. **Opt-in `narrative_context`.** Rejected: adds API surface and leaves
   payloads that can never be described correctly.

## Consequences

### Positive

- Stored payloads carry the context needed to describe them.
- One semantic source (the payload), one interval validator (serialization)
  and one ranking implementation (ADR-042), exposed publicly only as rule
  indices (ADR-042 D17); no parallel schema to version.
- No new dependency, plugin family, network path or provider API.

### Negative / Risks

- `metadata.narrative_context` is a stable surface that must be versioned
  under ADR-005 when changed.
- Narrative correctness for stored payloads depends on the #262 fix;
  payloads written before it carry misaligned rule values for factual and
  alternative explanations, and ADR-042 D17 cannot restore plot parity for
  them.
- Code that used `build_narrative_spec` from `main` between 2026-11-02 and
  its removal breaks; it was never released.

## Adoption & Migration

- v1.1.0: #228 (`narrative_context`, done), #232 (offline narration
  compatibility, done).
- #229 removes the `NarrativeSpec` types, `NARRATIVE_SPEC_VERSION`, their
  exports, API docs, how-to section and `CHANGELOG.md` wording.
- #262 makes serialized rule values come from the rule itself (D1).
- #204 adds `canonical_rule_indices` (ADR-042 D17) and moves the spec's
  ranking, tie, unbounded-endpoint and missing-context tests onto it.
- #230 then removes `build_narrative_spec` and its validation, ranking glue
  and canonical-JSON helpers, deletes the spec-only tests and updates the
  T9/T10 task-verification lists.
- No deprecation: the removed names were never released.
- Verification: `tests/unit/explanations/test_narrative_spec.py` (its
  `narrative_context` tests, relocated as #230 decides) and
  `tests/capabilities/test_narrative_contracts.py`.

## Open Questions

None.
