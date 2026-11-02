> **Status note (2026-11-02):** Last edited 2026-11-02 · Archive after: Retain indefinitely as architectural record · Implementation window: v1.1.0 (#227-#230, #232).

# ADR-041: Minimal Narrative Semantic Contract

Status: Accepted
Date: 2026-11-02
Deciders: Core maintainers
Reviewers: Core maintainers
Supersedes: None
Superseded-by: None
Related: ADR-002-validation-and-exception-design, ADR-005-explanation-json-schema-versioning, ADR-008-explanation-domain-model-and-compat, ADR-020-legacy-user-api-stability, ADR-021-calibrated-interval-semantics, ADR-029-reject-integration-strategy, ADR-037-visualization-extension-and-rendering-governance, ADR-040-capability-verification-framework, ADR-042-canonical-rule-and-feature-ranking

## Context

CE ships offline template narration. `CalibratedExplanation.to_narrative()` and
`CalibratedExplanations.to_narrative()` (with the `to_dataframe()` and
`narrate()` aliases, and `plot(style="narrative")`) drive
`core/narrative_generator.py` over a packaged YAML template through
`viz/narrative_plugin.py`. Output formats are `dataframe`, `text`, `html`,
`dict` and `markdown`.

Describing an explanation in words requires the same facts every time: the
calibrated prediction and its interval, each rule's weight and interval, whether
a rule is counterfactual, the problem type, the class a probability refers to,
the threshold of a probabilistic regression, the requested percentiles, and
whether a reject policy flagged the instance. Today these are read ad hoc from
live explanation objects. The serialized ADR-005 payload omits several of them
(problem type, class, threshold, percentiles, reject state), so a stored
explanation cannot be described correctly without the explainer that produced
it. There is also no single, deterministic statement of "what is true about
this explanation" that narration, tests or any text built on CE can be checked
against.

An earlier draft of this ADR proposed a narrative provider plugin family with
provider registration, dispatch, trust orchestration, host verification of
provider prose, provider configuration objects and a public provider API. CE's
own need, describing already computed predictions, intervals and explanations
correctly, deterministically and consistently, does not require any of that.
This ADR replaces that draft with a minimal contract.

## Decision

### D1. Explanation results are authoritative

The explanation results CE has computed, with their documented semantics, are
the only source of narrative facts. The ADR-005 payload and the ADR-008 domain
model are the primary inputs. Narrative code MUST NOT recompute, re-derive or
duplicate prediction, calibration or explanation analysis. It MAY copy values,
compare a value with an interval endpoint or with zero, and order rules by
ADR-042.

### D2. `NarrativeSpec` is a minimal, immutable, versioned fact record

CE provides `calibrated_explanations.explanations.NarrativeSpec` (frozen
dataclasses in `explanations/narrative_spec.py`) and the builder
`build_narrative_spec(payload, *, rnk_metric=None, rnk_weight=0.5)`. A spec
exposes:

- the calibrated prediction and its interval;
- one fact per rule: rule text, feature index or indices, feature value,
  weight interval, rule prediction interval, and whether it is counterfactual
  (alternative rules) or describes the observed instance (factual and FAST
  rules);
- a source reference (a JSON pointer into the payload, e.g. `/rules/3`) for
  every fact;
- `metadata.narrative_context` as found in the payload (`context`);
- the ADR-042 ranking key (`ranking`);
- qualifiers from a closed vocabulary that narration must not omit:
  `rejected`, `unbounded_interval`, `context_unavailable` (spec level) and
  `weight_interval_contains_zero`, `unbounded_interval` (rule level);
- `spec_version` (`1.0.0`) and `spec_id`.

A spec MUST be deterministic: identical semantic input MUST produce identical
canonical JSON (`NarrativeSpec.to_json()`: sorted keys, compact separators,
shortest round-trip floats, infinities as `"inf"`/`"-inf"`), and `spec_id` is
the SHA-256 of that JSON without `spec_id`. Numbers are copied without
rounding; integers and floats of equal value are the same input. A change to
the fields or their meaning MUST bump `spec_version`.

The builder MUST raise `ValidationError` (ADR-002) for missing, non-numeric,
`NaN` or contradictory required data: an unknown explanation type or task, an
interval with `low > high` or a contained value outside it (subject to D4), a
`problem_type` that contradicts `task`, a threshold present for anything but
probabilistic regression (or absent for it), invalid percentiles, a malformed
reject record, or ranking arguments that do not apply to the explanation type.
No new exception hierarchy is introduced.

FAST explanations are covered; their rules are ranked and qualified as factual
rules (ADR-042 D1.1).

### D3. `metadata.narrative_context` is an additive ADR-005 extension

`CalibratedExplanations.to_json()` and `to_json_stream()` MUST emit
`metadata.narrative_context` on every explanation for which CE has established
at least one of these facts, and only these:

| Key | When present | Source |
|---|---|---|
| `problem_type` | always when the mode is known | explainer mode, multiclass flag, threshold |
| `class_index`, `class_label` | classification | `prediction["classes"]`, explainer class labels |
| `threshold` | probabilistic regression | the explanation's `y_threshold` |
| `low_high_percentiles` | percentile regression | the collection's requested percentiles |
| `reject` | a reject policy is active | the explanation's `RejectContext` |

A key whose source is unavailable MUST be omitted, never inferred. No existing
payload field is renamed, removed or reinterpreted, and no v2 schema cycle is
opened (ADR-005 v1 freeze statement). Payloads written before this extension
remain valid: `build_narrative_spec` accepts them, records
`context_unavailable`, and raises only where a fact it needs is undefined
without context (the default `ensured` ranking of a regression alternative).

Emission is unconditional rather than opt-in so that a stored payload is
self-describing; the added size is a handful of scalar keys per explanation.

### D4. Interval semantics follow ADR-021

The builder MUST check `low <= predict <= high` for the prediction, every
weight interval and every rule prediction, with the floating-point tolerance
used by serialization. For percentile regression it MUST apply the ADR-021 §4
percentile scope: containment is checked only on the sides the requested
percentiles imply, mirrored for factual and FAST weights (a factual weight is
`predict - perturbed`). An endpoint that is infinite, or that is the
`min(y_cal)`/`max(y_cal)` placeholder reported for an infinite percentile, is
marked `low_bounded=False` or `high_bounded=False`; its reported value is kept
unchanged. Narration MUST NOT present an unbounded endpoint as a numeric limit.
When #237 represents open endpoints as infinite, the same flags apply.

### D5. ADR-042 governs rule order

`NarrativeSpec.rules` MUST be in the ADR-042 canonical order (D1, D5, D6),
strongest first, and `ranking` MUST record the key: `feature_weight` for
factual and FAST explanations; `ensured` with its weight, or `feature_weight`,
for alternatives (`uncertainty` is recorded as `ensured` with weight `0.0`).
The spec drops no rules; filtering for display stays a consumer step (ADR-042
D7). Until #204 provides the single authoritative ranking implementation
(ADR-042 D9), the builder carries its own implementation of the D5 total order
and reuses `calculate_metrics` for the `ensured` score; #204 MUST make it
delegate.

### D6. Existing offline narration and public API are preserved

`to_narrative()`, `to_dataframe()`, `narrate()` and `plot(style="narrative")`
keep their signatures, defaults, return types and formats, and continue to work
offline with the packaged template. This ADR does not change the template
engine or its output; ADR-042 D16 (#204) owns the planned changes to narrative
rule selection and order, and those are recorded in `CHANGELOG.md` when they
land. Byte-for-byte prose is not contracted (ADR-020 does not list
`to_narrative`), but any wording or order change MUST be documented and tested.
The template engine MAY later read a `NarrativeSpec` instead of live objects if
that causes no regression; it is not required.

### D7. No providers, plugin family or external services

CE MUST NOT add, for narration, a plugin family, provider registry or dispatch,
trust orchestration, credentials, network configuration, provider configuration
objects or a provider-facing public API. Narration remains a plot style under
ADR-037 and adds no runtime dependency and no network access.

### D8. No correctness guarantee for prose

A correct `NarrativeSpec` does not make text built from it correct. CE MUST NOT
describe narrative prose as calibrated or verified. The calibration guarantees
of ADR-021 attach to predictions and intervals, not to sentences, and ranks
carry no guarantee (ADR-042 D14). Text built on a spec must still be checked
against it by whoever produces that text.

## Governed claims

This ADR constrains how narratives describe explanations. It adds no new claim;
`NarrativeSpec` is verified by unit tests until a claim is warranted.

- `CE-CAP-NARR-001` — narratives; offline template narration keeps its public API and formats (D6), and makes no correctness claim about prose (D8).

## Alternatives Considered

1. **Narrative provider plugin family with host verification (the earlier
   draft).** Rejected: provider dispatch, trust, verification of free-form
   prose and a provider API serve no need of CE's own offline narration, and
   would add a sixth hand-rolled family to `plugins/registry.py` plus a public
   surface that has to be kept stable.
2. **Keep reading live explanation objects, no spec.** Rejected: stored payloads
   cannot be described correctly, and there is no deterministic fact record to
   test narration or other text against.
3. **Open an ADR-005 v2 schema cycle for the missing context.** Rejected: the v1
   freeze permits additive `metadata` sub-keys, which is sufficient.
4. **Emit `narrative_context` only on request.** Rejected: an opt-in flag adds
   API surface, and payloads written without it could never be narrated
   correctly later.
5. **Rewrite placeholder endpoints to `±inf` in the spec.** Rejected: it would
   change reported values before #237 decides their representation; a bounded
   flag records the ADR-021 meaning without altering numbers.
6. **Re-express the template engine over `NarrativeSpec` now.** Rejected for
   this release: the engine works and is covered by tests, and the rewrite
   would change output without a user-visible benefit.

## Consequences

### Positive

- Stored explanation payloads carry the context needed to describe them.
- One deterministic, versioned fact record that tests and any narration can be
  checked against, with source references to the payload.
- Interval and reject semantics are stated once, consistently with ADR-021,
  ADR-029 and ADR-042.
- No new dependency, plugin family, network path or provider API.

### Negative / Risks

- `metadata.narrative_context` and `NarrativeSpec` v1 become documented
  surfaces that must stay stable or be versioned.
- Until #204 lands, the builder's ADR-042 ordering coexists with the legacy,
  non-canonical orderings used by `__repr__`, plots and the template engine;
  narrative text and `NarrativeSpec.rules` can therefore list rules in
  different orders.
- `to_json()` of an off-centre one-sided regression query (for example
  percentiles `(-inf, 40)`) still fails its serialization invariant check, so
  no payload, and hence no spec, can be produced for it; this is tracked with
  #237.

## Adoption & Migration

- v1.1.0: #227 (this ADR), #228 (`narrative_context`), #229 and #230
  (`NarrativeSpec` and builder), #232 (compatibility of offline narration).
  #231, #233 and #234 (plugin family, provider API, provider capability chain)
  are closed as not planned.
- #204 (ADR-042 implementation) makes the builder delegate to the single
  ranking implementation and changes template narration as ADR-042 D16
  requires.
- #237 (unbounded endpoints) keeps the D4 flags meaningful; no spec change is
  expected.
- No deprecation or migration is required. The additions are additive and
  `CHANGELOG.md` records them.
- Verification: `tests/unit/explanations/test_narrative_spec.py` and
  `tests/capabilities/test_narrative_contracts.py`. No new capability claim is
  introduced; `CE-CAP-NARR-001` links this ADR.

## Open Questions

None.
