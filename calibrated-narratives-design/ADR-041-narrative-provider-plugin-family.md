> **Active scope:** Governing architectural decision for the narrative provider plugin
> family, the `NarrativeSpec` semantic contract, and host-side narrative verification.
> Remains active as long as these contracts govern CE narration; superseded when the
> narration contract is revised.

> **Status note (2026-09-20):** Last edited 2026-09-20 · Archive after: Retain
> indefinitely as architectural record · Implementation window: next minor release
> (not v1.0.1, which excludes new plugin categories).

# ADR-041: Narrative Provider Plugin Family

Status: Draft
Date: 2026-09-20
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None
Related: ADR-001-core-decomposition-boundaries,
ADR-002-validation-and-exception-design,
ADR-005-explanation-json-schema-versioning,
ADR-006-plugin-registry-trust-model,
ADR-008-explanation-domain-model-and-compat,
ADR-010-core-vs-evaluation-split-and-distribution,
ADR-011-deprecation-and-migration-policy,
ADR-020-legacy-user-api-stability,
ADR-021-calibrated-interval-semantics,
ADR-028-logging-and-governance-observability,
ADR-029-reject-integration-strategy,
ADR-030-test-quality-priorities-and-enforcement,
ADR-033-modality-extension-plugin-contract-and-packaging,
ADR-037-visualization-extension-and-rendering-governance,
ADR-038-call-time-configuration-taxonomy,
ADR-040-capability-verification-framework,
Standard-001-nomenclature-standardization,
Standard-005-logging-and-observability-standard

## Context

CE ships deterministic template narration today. `CalibratedExplanation.to_narrative()`
and `CalibratedExplanations.to_narrative()` delegate to `NarrativePlotPlugin`, which
drives `NarrativeGenerator` over a YAML template. `plot(style="narrative")` reaches the
same code through the plot-style registry. The capability is claimed by
`CE-CAP-NARR-001` and verified by `CE-REQ-NARR-API-001` / `CE-TIF-NARR-001` at the
api_contract level only.

Three structural problems block any second narrator, including an LLM-backed one.

1. **Narration is modelled as a plot style.** ADR-037 §6 forbids runtime plot-kind
   extension and its `plot_kinds` vocabulary is purely visual. A prose generator has
   different inputs, failure modes, provenance, and safety requirements than a
   renderer. The current placement cannot be extended without violating ADR-037.
2. **There is no semantic handover contract.** `NarrativeGenerator.generate_narrative`
   accepts a live explanation object and reads `get_rules()`, `reject_context`,
   `get_class_labels()`, and `y_threshold`. Any second narrator would re-derive
   interval, sign, threshold, and reject semantics independently, which defeats
   controlled uncertainty-preserving narration.
3. **There is no verification boundary.** Nothing distinguishes "the source
   explanation is calibrated" from "this prose faithfully represents it".

A networked LLM narrator is desirable but must not enter CE. CE would acquire provider
SDKs, credentials, prompts, network policy, and fast-moving research code, which
ADR-010 explicitly keeps in companion repositories.

## Decision

### D1. Narrative providers are a distinct plugin family

CE MUST define a `narrative:provider` capability tag and a narrative plugin family
alongside the existing explanation, interval, and plot families.

Narration MUST NOT be expressed as a `PlotBuilder`, `PlotRenderer`, or new
`plot_kinds` value. ADR-037 §§2–3 bind builders and renderers to canonical PlotSpec;
ADR-037 §6 forbids runtime kind extension. This ADR adds a sibling family and does not
amend ADR-037.

The existing `plot(style="narrative")` path and the `narrative` entry in
`_BUILTIN_INSTANCE_PLOT_STYLES` are retained unchanged as a legacy alias that routes to
the built-in narrative provider. No new plot style, kind, or mode is registered.

### D2. CE owns the contract; providers own the prose

CE MUST own, as passive provider-neutral types:

- `NarrativeSpec` — the deterministic semantic plan (facts, mandatory facts,
  prohibited claims, source binding);
- `CandidateNarrative` — what a provider returns;
- `NarrativeResult` — what a user receives;
- `VerificationReport` — what the host checked.

`NarrativeSpec` is an intermediate representation, not a configuration surface. It is
named by direct analogy to `PlotSpec`, which ADR-038 §5 does not govern because §5
addresses configuration objects. Narrative *configuration* follows ADR-038 in D6.

CE MUST NOT own provider SDKs, prompts, credentials, network retries, model-specific
parsing, or vendor telemetry.

### D3. The semantic source is the explanation, never PlotSpec

For local narration the builder MUST derive `NarrativeSpec` from the ADR-005
explanation payload plus the ADR-008 domain object. It MUST NOT read PlotSpec.

The frozen ADR-005 v1 payload does not carry problem-type framing, class labels,
decision thresholds, or reject state. CE MUST supply these through the documented
additive extension surface as `metadata.narrative_context`, which is permitted by the
ADR-005 v1 freeze statement ("additive additions to `metadata` and `provenance` in
patch releases are permitted"). No existing field is renamed, removed, or
reinterpreted, and no v2 schema cycle is opened.

`metadata.narrative_context` MUST carry only presentation-neutral semantics already
established elsewhere in CE: problem type, class labels, threshold definition, and
reject/defer state per ADR-029. It MUST NOT carry new analysis.

### D4. Builders are deterministic and preserve interval semantics

`NarrativeSpec` builders MUST:

- preserve raw numeric values and interval bounds without rounding;
- uphold the ADR-021 invariant `low <= predict <= high` and fail closed on violation,
  consistent with ADR-026 §2 treating violations as hard failures;
- derive sign, zero-crossing, threshold relation, and reject/defer state
  deterministically;
- attach a source reference to every fact;
- select mandatory uncertainty qualifiers before any provider is invoked;
- default causal and actionable claim permissions to false;
- raise `NarrativeSpecError` on missing or ambiguous required semantics.

Two builds from the same payload MUST produce byte-identical canonical JSON.

### D5. Numeric claims are bound by the host, not asserted by the provider

A provider MUST express every numeric, sign, direction, and interval claim as a
placeholder token referencing a `NarrativeSpec` fact field, never as literal text. CE
substitutes the values from `NarrativeSpec` after verification.

This makes numeric fidelity true by construction rather than by checking provider
arithmetic. A candidate containing an unbound numeric literal in a `factual` or
`qualifier` claim MUST fail host verification.

### D6. Configuration follows the ADR-038 taxonomy

Narration configuration MUST use the ADR-038 four-tier taxonomy:

| Surface | Tier | Type | Parameter |
|---|---|---|---|
| Which provider and verification mode | Strategy | `NarrativeProviderSpec` | `narrative_provider=` |
| Timeouts, attempt limits, size caps | Tuning (grouped) | `NarrativeOptions` | `narrative_options=` |
| Audience level | Tuning (single) | qualified kwarg | `expertise_level=` (existing) |
| Output format | Existing public kwarg | `str` | `output_format=` (existing) |

`NarrativeProviderSpec` MUST NOT contain numeric thresholds, per ADR-038 §2b.
No new `**kwargs` surface is introduced; `reject_unsupported_narrative_kwargs` continues
to fail fast on unknown names per the ADR-038 2026-07-08 addendum.

### D7. Trust, discovery, and metadata reuse existing mechanisms

Narrative providers MUST be discovered through the existing
`calibrated_explanations.plugins` entry-point group and MUST satisfy
`validate_plugin_meta`, including the ADR-033 `data_modalities` and
`plugin_api_version` requirements.

Third-party narrative providers MUST be untrusted on installation and MUST NOT be
*dispatched* until explicitly trusted under ADR-006. ADR-006 documents that CE performs
no sandboxing, so entry-point module import may still execute provider code; this ADR
does not claim otherwise and MUST NOT be described as preventing execution.

Providers MUST declare `network_access` and a supported `NarrativeSpec` range. The
range MUST be validated with CE's existing major/minor parsing. CE MUST NOT add
`packaging` or any other new runtime dependency for this purpose.

A networked provider MUST NOT become the default through entry-point order. Resolution
order is explicit spec > configured default > built-in template provider.

### D8. Host verification is mandatory and its scope is recorded

CE MUST run deterministic host verification after every provider call. A provider MUST
NOT set its own final status. An optional provider-supplied verifier MAY add findings
but MUST NOT clear a host failure or widen claim permissions.

`NarrativeResult.status` MUST be one of `checked`, `fallback`, `abstained`, or
`unchecked`. The result MUST record `verification.binding` as `substituted` when all
numeric claims were host-bound per D5, or `structural` otherwise.

CE MUST NOT describe narrative prose as calibrated. `source_calibrated` and the
verification fields are separate and MUST NOT be conflated, consistent with ADR-040 D8,
which already states that narrative generation tests do not prove explanation quality.

### D9. Fallback is always visible

Fallback from a selected provider MUST emit a `UserWarning` and an INFO log, and MUST
be recorded in `NarrativeResult`, per the repository fallback-visibility policy and
Standard-005.

Provider trust and deny decisions are governance events and MUST be emitted through
`emit_plugin_governance_event` on `calibrated_explanations.governance.plugins`, per
ADR-028 §2. Fallback and provider-selection events are operational and MUST stay in the
`calibrated_explanations.plugins.*` domain.

Per ADR-028 §6, narrative text, prompts, vendor responses, feature values, and feature
labels MUST NOT be logged by default.

### D10. No new analysis in narration

Narrative generation MUST NOT perform new explanation analysis. Global feature
aggregation, causal inference, feasibility analysis, and out-of-distribution detection
belong to explanation or analysis plugins upstream of narration.

Global narration is out of scope for this ADR. CE has no presentation-neutral global
summary today; `GlobalPlotSpec` is a visualization artifact and D3 forbids using it as a
narrative source. A later ADR may add a global summary and extend this family to it.

### D11. Existing behaviour is preserved without a byte-for-byte freeze

`to_narrative()`, `to_dataframe()`, `narrate()`, and `plot(style="narrative")` called
without the new keyword arguments MUST continue to return the documented formats
(`dataframe`, `text`, `html`, `dict`, `markdown`) and MUST continue to satisfy
`CE-REQ-NARR-API-001`.

Byte-for-byte output equality is explicitly **not** contracted. `to_narrative` does not
appear in the ADR-020 legacy user API contract, so it is not major-gated, and
`CE-CAP-NARR-001` claims only non-`None` output with a non-empty string for
`output_format='text'`. Freezing exact prose would prevent the built-in provider from
sharing the `NarrativeSpec` path with third-party providers and would duplicate the
uncertainty rules in two places.

Template wording changes remain additive and documented in `CHANGELOG.md`. No
deprecation is introduced, so the ADR-011 zero-deprecation posture is unaffected.

## Plugin shape

```python
class NarrativeProvider(Protocol):
    plugin_meta: Mapping[str, object]

    def generate(
        self,
        spec: NarrativeSpec,
        request: NarrativeRequest,
    ) -> CandidateNarrative: ...
```

A separate `narrative:verifier` capability is deliberately **not** defined in v1. An
optional additive verifier is an ordinary method on a provider package and is revisited
only if two independent providers need it.

## Module placement

Placement follows ADR-001 and mirrors the existing plugin families.

| Material | Location |
|---|---|
| Passive contracts and builders | `src/calibrated_explanations/explanations/narrative/` |
| Provider protocol | `src/calibrated_explanations/plugins/narratives.py` |
| Built-in provider registration | `src/calibrated_explanations/plugins/builtins.py` |
| Registry descriptor and trust helpers | `src/calibrated_explanations/plugins/registry.py` |
| Typed errors | `src/calibrated_explanations/utils/exceptions.py` |
| Template engine (unchanged) | `src/calibrated_explanations/core/narrative_generator.py` |

`plugins/narratives.py` parallels the existing `plugins/explanations.py`,
`plugins/intervals.py`, and `plugins/plots.py`. Errors live in `utils/exceptions.py`
alongside `PlotPluginError`; `core/exceptions.py` is a re-export shim only.

Identifiers use dot-delimited lowercase paths per Standard-001 §7. The built-in
provider is `core.narrative.template`.

## Errors

Added to `utils/exceptions.py`, all inheriting `CalibratedError` per ADR-002:

```text
NarrativeError(CalibratedError)
├── NarrativeSpecError
├── NarrativeProviderError
└── NarrativeVerificationError
```

Unsupported keyword arguments continue to raise `ConfigurationError` through the
existing `reject_unsupported_narrative_kwargs` path.

## Governed claims

- `CE-CAP-NARR-001` — extended from template-only narration to provider-dispatched
  narration; existing acceptance criteria retained unchanged.
- `CE-CAP-NARR-002` — new: narrative providers are dispatched through a trusted plugin
  family with mandatory host verification and visible fallback.

Requirements, TIF interfaces, and evidence follow ADR-040 D1–D8. See
`verification-chain.md`.

## Alternatives Considered

### A. Register prose as a `PlotRenderer`

Rejected. ADR-037 §3 binds renderers to validated canonical PlotSpec and §6 forbids
runtime kind extension. A renderer promises visual realization.

### B. Put the whole narration subsystem in CE, including LLM providers

Rejected. Provider clients, prompts, credentials, and network policy would enlarge CE's
dependency and governance surface, contrary to ADR-010's companion-repository model.

### C. Put the whole narration subsystem, including the contract, in the companion repo

Rejected as the target architecture. CE would neither define nor invoke the contract,
so the result is middleware around CE rather than a CE plugin. It would also duplicate
the public API and let each provider reinterpret interval semantics.

Retained as a staging prototype: the contract may be prototyped externally before CE
adopts it. See `implementation-plan.md` Stage 0.

### D. Let providers consume explanation objects directly

Rejected. Providers would independently interpret intervals, signs, thresholds, and
reject states, defeating controlled uncertainty-preserving narration and bypassing the
ADR-021 invariant check.

### E. Open a v2 explanation schema cycle for narrative fields

Rejected. The ADR-005 v1 freeze permits additive `metadata` sub-keys, which is
sufficient. A v2 cycle would be disproportionate.

### F. Freeze legacy narrative output byte-for-byte

Rejected. See D11.

## Consequences

Positive:

- One CE API covers deterministic and third-party narration.
- CE controls semantic grounding, uncertainty qualification, trust, and provenance.
- Numeric fidelity is structural, not probabilistic, because of D5.
- External providers evolve without adding CE dependencies.
- PlotSpec remains purely a visualization contract.

Negative / Risks:

- CE maintains a new plugin family, which means a sixth parallel descriptor, validator,
  registration, trust-marking, and reset path in an already large `registry.py`.
- The `NarrativeSpec` contract must be versioned conservatively.
- The built-in provider is re-expressed over `NarrativeSpec`, so exact legacy prose may
  shift within the documented formats.
- `metadata.narrative_context` becomes a documented additive surface that must be kept
  stable.

## Adoption & Migration

1. Prototype the contract externally against the current public API (Stage 0). Do not
   change CE.
2. Land `metadata.narrative_context` as an additive ADR-005 extension with schema docs.
3. Add passive contracts, deterministic builders, and typed errors.
4. Add the narrative family to the registry with trust and discovery.
5. Re-express the built-in template generator as `core.narrative.template`.
6. Add `NarrativeProviderSpec` / `NarrativeOptions` and `output_format="result"`.
7. Extend the ADR-040 chain: claims, requirements, TIF, evidence.
8. Publish the external provider against the released contract.

This work targets a milestone after v1.0.1. The active v1.0.1 plan explicitly excludes
new public API and plugin categories.

## Open Questions

1. Should `metadata.narrative_context` be emitted by default in `to_json()`, or only
   when a narrative build requests it? Default emission enlarges every payload.
2. Should the built-in provider keep its YAML template engine, or move to the same
   placeholder-binding mechanism used by external providers?
3. Does the registry warrant a generic plugin-kind abstraction before adding a sixth
   hand-rolled family, as a separate refactor?
4. Should `NarrativeSpec` v1 cover `FastExplanation`, or only factual and alternative?
