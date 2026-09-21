# Calibrated Narratives — Narrative Provider Plugin Family

Status: **Private design draft, revision 4**
Date: 2026-09-20
Scope: design only. No code, commit, publication, or external provider call.

## Decision

CE owns the narrative **contract and dispatch**. Concrete networked LLM narrators live
in `kristinebergs/calibrated-explanations-plugins` as third-party CE plugins.

- CE defines a `narrative:provider` plugin family, a deterministic `NarrativeSpec`
  builder, mandatory host verification, and one public API.
- CE ships `core.narrative.template`, the offline built-in provider, as the default.
- CE never imports a provider, adds no LLM SDK, and makes no network call.
- Numeric claims are **host-bound**: providers emit placeholders, CE substitutes the
  values, so numeric fidelity holds by construction.
- Narration is **not** a plot renderer, builder, kind, or mode. ADR-037 §6 forbids
  runtime plot-kind extension.

| Concern | CE | Companion repo |
|---|---|---|
| Plugin family, capability tag, dispatch, trust | **owns** | implements against |
| `NarrativeSpec` / `CandidateNarrative` / `NarrativeResult` | **owns** | consumes |
| Deterministic builders, mandatory verification | **owns** | cannot bypass |
| Offline template provider | **owns** | no duplicate |
| LLM client, prompts, credentials, retries | never | **owns** |
| Research fixtures, metrics, human studies | never | `…-studies` repo |

## Files

Five documents plus schemas. Read in this order.

| File | Purpose |
|---|---|
| [ADR-041-narrative-provider-plugin-family.md](ADR-041-narrative-provider-plugin-family.md) | The governing decision, ready to copy into `development/adrs/` |
| [contract.md](contract.md) | `NarrativeSpec`, provider protocol, host checks, registry, trust, public API |
| [implementation-plan.md](implementation-plan.md) | Staged work, ADR-040 capability chain, tests, gates |
| [plugins-repository-design.md](plugins-repository-design.md) | The external provider package |
| [schemas/](schemas/) | Draft interchange schemas |

Revision 3 spread this across 15 documents; revision 4 consolidates them into the five
above. Architecture, spec, API, provider contracts, and security live in `contract.md`;
validation and the ADR-040 chain in `implementation-plan.md`; ADR alignment, deferred
scope, and the red-team record in this README; external-provider controls and tests in
`plugins-repository-design.md`.

## ADR alignment

Verified against the repository on 2026-09-20. Superseded ADRs (007, 014, 016, 022, 024,
025) are not cited; revision 3 cited ADR-014 and ADR-016, both superseded by ADR-036/037.

| ADR / Standard | Constraint | Satisfied by |
|---|---|---|
| ADR-001 | Sibling packages talk only through domain models or explicit interfaces | `plugins/narratives.py` beside `explanations.py`/`intervals.py`/`plots.py`; no new top-level package |
| ADR-002 | Inherit `CalibratedError`; canonical module is `utils/exceptions.py` | `NarrativeError` + 3 subclasses beside the `PlotPluginError` precedent |
| ADR-005 | v1 frozen; additive `metadata` sub-keys permitted | `metadata.narrative_context`; no v2 cycle |
| ADR-006 | Third-party untrusted by default; no sandboxing | Reuses entry-point group and `validate_plugin_meta`; "not dispatched", never "cannot execute" |
| ADR-008 | Canonical domain model | Builders read the domain object and payload |
| ADR-010 | Core independently installable; official plugins in the companion repo | No SDK, credentials, or network in CE |
| ADR-011 | Two-minor window; zero deprecations at v1.0.0 | No deprecation introduced; all changes additive |
| ADR-020 | `legacy_user_api_contract.md` is the source of truth | `to_narrative` has 0 hits there, so it is not major-gated |
| ADR-021 / ADR-026 | `low <= predict <= high`; violations are hard failures | Builders fail closed, never coerce |
| ADR-028 / STD-005 | Governance vs operational domains; data minimisation | Trust events via `emit_plugin_governance_event`; no text, prompts, or labels logged |
| ADR-029 | Reject state in the envelope | Carried in `narrative_context`; mandatory when active |
| ADR-030 / STD-003 | 90%+ coverage; `test_should_…_when_…` | All test names follow the pattern |
| ADR-033 | `data_modalities` + `plugin_api_version` required | Both declared; CE's existing version parser reused |
| ADR-036 / ADR-037 | PlotSpec authority; no runtime kind extension | Narration is a sibling family; `plot(style="narrative")` kept as a legacy alias |
| ADR-038 | `*Config`/`*Spec`/`*Options`; no unvalidated `**kwargs` | `NarrativeProviderSpec` + `NarrativeOptions` replace 7 loose kwargs |
| ADR-040 | Claim → requirement → TIF → test → evidence | Full chain in the implementation plan |
| STD-001 | Dotted lowercase plugin IDs | `core.narrative.template` |

## Non-goals

- Treating prose as a plot renderer, or adding a `plot_kinds` value.
- Adding an LLM dependency, prompt, client, credential handling, or retry engine to CE.
- Global narration, global feature aggregation, or a `narrative:verifier` capability.
- Claiming causal or actionable meaning absent from the source explanation.
- Calling narrative prose calibrated.
- Byte-for-byte freezing of existing narrative output.

## Deferred, with reasons

| Item | Why not now |
|---|---|
| **Global narration** | CE has only `GlobalPlotSpec`, a scatter IR disqualified by ADR-037. Needs a presentation-neutral `GlobalExplanationSummary` first — its own feature and ADR. Revision 3 hard-coded three global kinds into a schema it also declared immutable. |
| **Global feature aggregation** | An explanation method, not narration (ADR-041 D10). Belongs in a separate explanation-family plugin, usable with no narrator. |
| **`narrative:verifier` capability** | No second provider exists yet; an additive check works as an ordinary method; a second capability doubles registry surface. |
| **Generic plugin-kind abstraction** | `registry.py` is 2,519 lines with five hand-rolled families; ADR-041 adds a sixth the same way. The refactor is larger and riskier — ADR-041 Open Question 3. |
| **Non-tabular narration** | v1 is `data_modalities = ("tabular",)`; goes through ADR-033. |
| **Locale / translation** | Host checks cannot currently answer whether mandatory qualifiers survive translation. |

## What changed from revision 3

A QA pass against the actual repository found six blocking conflicts.

| # | Revision 3 | Revision 4 |
|---|---|---|
| 1 | `format=`, `audience`, `detail`, `locale`, `verification`, `failure_policy`, `provider_options` — all would raise `ConfigurationError` today | `NarrativeProviderSpec` + `NarrativeOptions`; existing `output_format`/`expertise_level` kept |
| 2 | "Canonical payload" as the builder source, but frozen ADR-005 v1 has no problem type, class labels, threshold, or reject state | `metadata.narrative_context`, additive under the v1 freeze |
| 3 | Built-in provider both byte-for-byte legacy **and** a `NarrativeSpec` consumer — impossible | Byte-for-byte declined; `to_narrative` is absent from the ADR-020 contract |
| 4 | Registry work understated; `SpecifierSet` needs a dependency CE lacks; "untrusted cannot execute" | Sixth family specified; existing version parser; "not dispatched" |
| 5 | Superseded ADR-014/016 cited; ADR-036/037/038/040 omitted; no ADR number; no ADR-040 chain; targeted a release excluding plugin categories | ADR-041; alignment table above; full chain; later milestone |
| 6 | Global kinds hard-coded into a v1.0.0 schema declared immutable | All global kinds deferred |

Soundness fixes: placeholder binding replaces an overclaiming `verified` status; per-fact
payload schemas replace `additionalProperties: true`; `TaskDescriptor` keeps both
`task_type` and `problem_type`; `spec_id` defined as a content hash; `NarrativePolicy`
split per ADR-038; Stage 0 gates the whole thing on a faithfulness experiment.

The companion-repo plan was rewritten against the real monorepo: fixed families,
hatchling, `[tool.ce_plugin_repo]`, the plugins repo's ADR-P001 lifecycle policy, the plugin-intake path, and
research assets in `…-studies`.

## Residual risks

1. A sixth hand-rolled registry family adds real maintenance weight.
2. Binding makes numbers exact and claims traceable. It does **not** prove that a fluent
   sentence built from true facts conveys a true impression. That is the honest limit.
3. Persuasive prose may raise confidence without improving decisions — a falsifying
   criterion, tested with humans, not in CI.
4. `metadata.narrative_context` is a second home for explanation semantics and must not
   drift from the domain object.
5. No sandboxing: trust gates dispatch, not import. Inherited from ADR-006 and must reach
   user documentation.

## Open decisions for maintainers

1. Accept the no-byte-for-byte-freeze position, or require an exact freeze and lose the
   single-contract property?
2. Add a `narrative` family to the companion repo (needs an ADR-P001 amendment, since
   `PLUGIN_FAMILIES` is hard-coded to three), or place the provider under `explanation`?
3. Refactor `registry.py` to a generic kind abstraction before adding the sixth family?
4. Emit `metadata.narrative_context` from `to_json()` always, or only on request?

## Prerequisite

Stage 0 must pass before any CE change. If structured, bound narration is not measurably
more faithful than direct prompting, ADR-041 is not written.
