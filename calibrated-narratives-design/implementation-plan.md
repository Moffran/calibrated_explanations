# Implementation Plan

Design only. No code is committed, pushed, or published from this package.

## Release targeting

`development/current-work/v1.0.1_plan.md` excludes "New public API or plugin categories"
and broad plugin architecture redesign, so this targets a later milestone. Planning is
issue- and milestone-based since v1.0.1 T1, so the entry point is a GitHub milestone;
`ce-release-planner` produces the `vX.Y.Z_plan.md`.

## Stage 0 — external prototype, no CE change

Find out whether structured, host-verified narration is measurably more faithful than
direct prompting, **before** CE commits to a contract. Runs entirely in the companion
repositories.

1. Build a `NarrativeSpec` equivalent outside CE from `explanation.to_json()` plus a
   side-channel for problem type, class labels, threshold, and reject state.
2. Implement placeholder binding and the host checks as library code.
3. Run the faithfulness comparison in `calibrated-explanations-studies`.

**Exit criterion:** structured, bound narration beats direct explanation-to-LLM prompting
on fact precision/recall, numeric exactness, and uncertainty-qualification recall. If it
does not, stop — ADR-041 is not written and CE keeps template-only narration.

This is the staging prototype ADR-041 Alternative C retains.

## Stage 1 — CE foundations

| ID | Work | Gate |
|---|---|---|
| N1 | ADR-041 reviewed and accepted; release-plan entry added | ADR accepted |
| N2 | `metadata.narrative_context` as an additive ADR-005 extension, documented in `docs/schema_v1.md` | no existing field changed; round-trip passes |
| N3 | `NarrativeError` hierarchy in `utils/exceptions.py` | ADR-002 taxonomy honoured |
| N4 | Passive contracts in `explanations/narrative/contracts.py` | frozen dataclasses; no vendor types |
| N5 | Deterministic builders for factual, alternative, fast | identical `spec_id` on rebuild; ADR-021 invariant enforced |
| N6 | Host verifier and placeholder binder | unbound literal fails; mandatory omission fails |

**Gate:** existing narration tests, `CE-TIF-NARR-001`, and `make local-checks-pr` pass
with no behaviour change, because nothing is dispatched differently yet.

## Stage 2 — the plugin family

| ID | Work | Gate |
|---|---|---|
| N7 | `NarrativeProvider` protocol and metadata validation in `plugins/narratives.py` | `validate_plugin_meta` accepts; ADR-033 fields required |
| N8 | Registry descriptor, register/find/list, trust helpers, catalog reset | joins `_assert_catalog_trust_consistency` |
| N9 | Entry-point routing on `narrative:provider` | untrusted provider not dispatched |
| N10 | `core.narrative.template` wrapping the existing YAML engine | documented formats preserved |
| N11 | Fixture provider under `tests/fixtures/` | no network, no subprocess |
| N12 | Fix unescaped HTML in the narrative formatter | markup in a feature label is escaped |

**Gate:** built-in and fixture providers run through one contract; untrusted providers are
not dispatched; no third-party import in CE; base install gains no dependency.

## Stage 3 — public API and verification chain

| ID | Work | Gate |
|---|---|---|
| N13 | `NarrativeProviderSpec` and `NarrativeOptions` | ADR-038 taxonomy; `*Spec` carries no numerics |
| N14 | Keyword-only `narrative_provider`/`narrative_options`; `output_format="result"` | existing calls unchanged; unknown kwargs still fail fast |
| N15 | `build_narrative_spec` / `render_narrative` facade | offline replay works |
| N16 | ADR-040 chain (below) | `make capability-chain-check` passes |
| N17 | RTD docs, including the ADR-006 no-sandbox risk | `ce-rtd-auditor` clean |

**Gate:** `make local-checks`, `make capability-chain-check`, and
`make capability-evidence-refresh` pass; `CHANGELOG.md` records the additive API and any
template wording change.

## Stage 4 — external provider

Runs in the companion repository against the released contract. See
`plugins-repository-design.md`. Requires a plugin-intake issue on the Moffran repository
first; the package starts experimental and source-only.

---

# ADR-040 capability chain

ADR-040 D1 makes claim → requirement → TIF → test → evidence canonical. The previous
revision had no chain and would have failed `make capability-chain-check`.

## Existing chain, must keep passing unchanged

`CE-CAP-NARR-001` → `CE-REQ-NARR-API-001` → `CE-TIF-NARR-001` →
`tests/capabilities/test_narrative_contracts.py`. This is the concrete meaning of
"existing behaviour is preserved" in ADR-041 D11.

## Changes to CE-CAP-NARR-001 (additive)

Add `ADR-041` to `adr_links`; extend `claim_text` to say narration is dispatched through
a provider defaulting to the built-in; add `CE-REQ-NARR-DEFAULT-001`; keep every existing
assumption including "narrative quality is NOT verified".

Rewrite `atomic_rationale`: it currently argues one requirement suffices, but with a
provider family the claim decomposes further, and ADR-040 D3 forbids using atomic
rationale to avoid decomposition.

Two existing assumptions are factually wrong and should be corrected while the file is
open: the claim names Jinja2 twice, but `NarrativeGenerator` uses PyYAML plus
`str.replace`, and the `narrative` extra declares `pyyaml`. `CE-TIF-NARR-001` already
states PyYAML correctly.

## New claim CE-CAP-NARR-002

Narration is dispatched through a trusted provider family: an untrusted third-party
provider is not dispatched, a provider cannot set its own verification status, and any
fallback is visible through `UserWarning`, INFO, and `NarrativeResult`.

Mandatory assumption boundary (ADR-040 D8): quality and fluency are not verified; host
verification proves structural and numeric-binding consistency, not linguistic
faithfulness; ADR-006 provides no sandboxing, so trust gates dispatch, not import;
network behaviour is not exercised, as CI uses a fixture provider.

## New requirements

| Requirement | Type | Observable behaviour |
|---|---|---|
| `CE-REQ-NARR-DEFAULT-001` | api_contract | With no provider arguments, `to_narrative()` returns the documented format from the built-in provider |
| `CE-REQ-NARR-TRUST-001` | behavioral | An installed but untrusted provider is not dispatched; the built-in is used and the skip reported |
| `CE-REQ-NARR-VERIFY-001` | behavioral | A candidate omitting a mandatory fact, or citing an unknown fact, never reaches the user as `checked` |
| `CE-REQ-NARR-BIND-001` | behavioral | Numeric values in returned prose equal spec values; a provider-authored literal fails |
| `CE-REQ-NARR-FALLBACK-001` | behavioral | Provider failure emits `UserWarning` + INFO and sets `status="fallback"` |
| `CE-REQ-NARR-SPEC-001` | behavioral | Same payload yields identical `spec_id`; expertise level does not change it |
| `CE-REQ-NARR-ISOLATION-001` | repository policy | Base install imports no LLM SDK and makes no network call; removing every provider leaves narration working offline |

Each file carries the ADR-040 D4 fields. `CE-REQ-NARR-ISOLATION-001` is the single TIF
exemption, under the D6 "repository policy" class, because it is an import-graph and
packaging property rather than a public-API behaviour; the exemption must say so.

## New TIF interfaces

`CE-TIF-NARR-002` (dispatch, trust, fallback) and `CE-TIF-NARR-003` (spec determinism and
binding). Both follow ADR-040 D5: stimulate through `WrapCalibratedExplainer`, use the
public `fit → calibrate → explain` lifecycle, return a dataclass of observations, use no
private members, construct no explanation objects directly, and hold no final assertions.

`CE-TIF-NARR-002` observations: `provider_dispatched`,
`untrusted_provider_dispatched`, `warning_emitted`, `info_logged`, `result_status`,
`verification_binding`.

The fixture provider registers through a patched `importlib.metadata.entry_points`,
mirroring `tests/unit/plugins/test_adr033_packaging_smoke.py` — no install, subprocess, or
network.

Raw evidence goes to `reports/verification/`, curated evidence to
`development/capabilities/evidence/`. Both are ADR-040 D2 locations; no new location is
introduced.

---

# Tests

ADR-030 naming `test_should_<behavior>_when_<condition>`; 90%+ coverage per ADR-030 and
STD-003; fallback tests use the `enable_fallbacks` fixture and assert `UserWarning`.

```text
test_should_preserve_documented_formats_when_no_provider_argument_supplied
test_should_raise_configuration_error_when_unknown_narrative_kwarg_supplied
test_should_produce_identical_spec_id_when_same_payload_built_twice
test_should_not_change_spec_id_when_expertise_level_changes
test_should_raise_validation_error_when_interval_invariant_violated
test_should_mark_direction_unsupported_when_effect_interval_crosses_zero
test_should_add_mandatory_qualifier_when_prediction_interval_crosses_threshold
test_should_include_reject_state_as_mandatory_fact_when_reject_active
test_should_not_dispatch_provider_when_provider_untrusted
test_should_emit_warning_and_info_when_provider_fails
test_should_set_status_fallback_when_provider_times_out
test_should_fail_verification_when_mandatory_fact_omitted
test_should_fail_verification_when_candidate_cites_unknown_fact
test_should_fail_verification_when_factual_claim_contains_numeric_literal
test_should_bind_placeholders_from_spec_when_candidate_verified
test_should_ignore_provider_status_when_provider_claims_verified
test_should_escape_markup_when_feature_label_contains_html
test_should_reject_provider_when_narrative_spec_major_version_unknown
```

## Property and metamorphic

- Widening an effect interval across zero forces `direction_supported=False` and adds a
  mandatory qualifier.
- Crossing a decision threshold changes the threshold relation and required wording.
- Activating reject/defer makes it mandatory content.
- Expertise level changes wording only — never facts, permissions, qualifiers, or
  `spec_id`.
- Display-label changes affect wording only, never canonical IDs or numerics.
- A provider cannot introduce a feature, class, or threshold absent from the spec.
- Numeric output equals spec values exactly for every bound placeholder.

## Golden cases

Confident and threshold-crossing binary classification; multiclass per-class; regression
with narrow and wide intervals; probabilistic regression with scalar and interval
thresholds; positive, negative, and zero-crossing effects; factual and alternative;
conjunctive rules; reject and defer active; untrusted, incompatible, timed-out,
malformed, and adversarial providers.

Each stores the payload, expected spec, mandatory and prohibited claims, the built-in
reference narrative, candidate variants, and the expected verification outcome.

## Metrics

Fact precision and recall, numeric exactness, sign and direction fidelity, interval and
threshold fidelity, uncertainty-qualification recall, unsupported-claim rate, fallback
and abstention rate, latency. Report separately; never collapse into one score.

With binding in place numeric exactness is 1.0 by construction. Below 1.0 means the
binder is broken, not that the model is imprecise.

## Falsifying criteria

The design fails if, after reasonable engineering: structured bound narration is no more
faithful than direct prompting (the Stage 0 exit criterion); mandatory uncertainty
planning does not improve comprehension or appropriate reliance; host verification
accepts materially misleading paraphrases too often; expertise level changes epistemic
content; users become more confident without deciding better; or maintaining the contract
and the sixth registry family costs more than provider interchangeability is worth.

Human validation runs in `calibrated-explanations-studies`, not CE.

## Rollback

The built-in provider is always available. A faulty external provider is untrusted or
denied via `CE_DENY_PLUGIN` without changing CE. Removing every provider leaves CE fully
functional offline. Released schema meanings are immutable; an incompatible correction
requires a major `NarrativeSpec` version and migration guidance.

## Completion checklist

- [ ] Stage 0 falsification experiment run and passed.
- [ ] ADR-041 accepted with a release-plan entry.
- [ ] `metadata.narrative_context` documented as additive; no v2 schema cycle.
- [ ] Existing narration tests and `CE-TIF-NARR-001` pass unchanged.
- [ ] `NarrativeSpec` never reads PlotSpec; no global kinds shipped.
- [ ] Built-in and fixture providers share one protocol.
- [ ] Untrusted providers not dispatched; no-sandbox limit documented.
- [ ] Host verification cannot be bypassed; numeric claims host-bound.
- [ ] Base install has no SDK, credential, prompt, network, or `packaging` dependency.
- [ ] ADR-040 chain complete; `capability-chain-check` green.
- [ ] HTML escaping defect fixed; no deprecation introduced.
- [ ] Nothing committed, pushed, or published without a later explicit request.
