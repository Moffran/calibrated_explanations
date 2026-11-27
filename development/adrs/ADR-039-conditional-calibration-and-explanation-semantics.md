---
id: "ADR-039"
title: "Conditional (Mondrian) Calibration and Explanation Semantics"
status: "accepted"
description: "Governing conditional calibration and Mondrian-category semantics across calibration, prediction, explanation and persistence paths."
superseded_by: []
---
> **Status note (2026-07-07):** Last edited 2026-07-07 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.11.5 (pre-RC hardening milestone inserted 2026-07-07; see `development/current-work/v0.11.5_plan.md`).

# ADR-039: Conditional (Mondrian) Calibration and Explanation Semantics

Date: 2026-07-07
Deciders: Core maintainers
Reviewers: Core maintainers
Related: ADR-013-interval-calibrator-plugin-strategy, ADR-021-calibrated-interval-semantics, ADR-031-calibrator-serialization-and-state-persistence, ADR-002-validation-and-exception-design, ADR-011-deprecation-and-migration-policy, ADR-020-legacy-user-api-stability, ADR-038-call-time-configuration-taxonomy

## Context

Conditional (Mondrian) calibration has been supported since the library's inception
and is the published basis for subgroup-aware uncertainty and fairness analysis
(Löfström et al., *Conditional Calibrated Explanations*, xAI 2024). Users activate it
through two channels:

1. **Inline `bins`** — an array of per-instance category labels passed to
   `WrapCalibratedExplainer.calibrate(..., bins=bins_cal)` (forwarded to
   `CalibratedExplainer(bins=...)`) and again to every inference call
   (`predict`, `predict_proba`, `explain_factual`, `explore_alternatives`,
   `explain_fast`, `plot`) as `bins=bins_test`.
2. **`mc` categorizer** — a `crepes.extras.MondrianCategorizer` or arbitrary callable
   passed to `calibrate(..., mc=...)`. The wrapper stores the categorizer and applies
   it to every input matrix at both calibration and inference time, deriving bins
   automatically.

Downstream, bins are threaded through the prediction orchestrator
(`core/prediction/orchestrator.py`) into the interval calibrators: `VennAbers` fits
one Venn-Abers model per calibration bin label (per class when multiclass);
`IntervalRegressor` forwards bins to crepes' `ConformalPredictiveSystem` and to the
internal Venn-Abers split for thresholded regression. The explanation pipeline
(`core/explain/_computation.py`) correctly propagates each test instance's bin to all
perturbed variants of that instance. ADR-013 already requires interval plugins to
declare `requires_bins` metadata and expose `is_mondrian()`; ADR-021 mentions
conditioning on Mondrian bins. No ADR codifies the end-to-end user-facing contract.

The architecture lacks a single answer to five contract questions: channel
precedence and combinability, the inference-time obligation once an explainer is
conditionally calibrated, the treatment of out-of-vocabulary category labels, the
lifecycle of conditional state across re-calibration, and what survives
serialization. Behavior in these uncodified areas is implementation-defined and
diverges across entry points, task modes, and the persistence boundary; a 2026-07-07
audit confirmed the divergence, and its defect inventory and remediation tasks are
tracked in `development/current-work/v0.11.5_plan.md` (archived to
`development/finished-work/` on milestone closure). The term "bins" is
additionally overloaded in the codebase (Mondrian category labels, discretizer bin
edges, `crepes.extras.binning` outputs), and "conditional" collides with the
unrelated `condition_source` rule parameter. A governing contract is needed before
the v1.0.0 API freeze.

## Decision

### D1. Terminology (normative for code, docs, and messages)

- **Conditional calibration** (synonym: **Mondrian calibration**) is the partitioning
  of calibration data by a per-instance **category label** so each category receives
  its own calibrator. The public parameter for category labels is `bins`; the public
  parameter for a category-deriving object or callable is `mc`.
- Docs and error messages MUST NOT use "bins" for discretizer bin edges or
  `crepes.extras.binning` thresholds without qualification, and MUST NOT describe
  Mondrian `bins` as "discretization bins".
- `condition_source` (rule-condition anchoring, "observed" vs "prediction") is a
  distinct concept and MUST NOT be documented as part of conditional calibration.

### D2. Specification channels and precedence

- `calibrate` MUST accept at most one conditional channel per call: `bins=` (labels
  aligned with `x_calibration`, `len(bins) == n_cal`), `mc=` (categorizer or
  callable), or `reuse_conditional=True` (D4). Supplying more than one in the same
  call MUST raise `ValidationError`.
- When `mc` is configured, inference-time bins are derived by applying `mc` to the
  (preprocessed) input. Passing an explicit `bins=` argument to an inference method
  while `mc` is configured MUST raise `ConfigurationError`: the call is ambiguous
  between two category sources, and an explicit argument MUST never be silently
  ignored.
- A callable `mc` MUST be applied as `mc(x)`; a `MondrianCategorizer` via
  `mc.apply(x)`. The derived labels MUST be validated like explicit `bins`
  (D3).

### D3. Inference-time consistency contract

- If the explainer was calibrated conditionally via inline `bins`, every inference
  call MUST receive `bins` (length `n_test`, label space seen at calibration).
  Omission MUST raise `ValidationError` with a message naming the requirement and
  the available options. Calibration-time bins MUST NOT be substituted for an
  inference input implicitly: category labels are positional per instance, so a
  stored calibration assignment is meaningful only for the calibration set itself.
- If the explainer was calibrated globally (no `bins`, no `mc`), passing `bins` to
  an inference call MUST raise `ConfigurationError`: no per-category calibrators
  exist for the labels to select, and accepting the argument would misrepresent
  global output as conditional. The rule MUST hold uniformly across task modes
  (classification, percentile regression, thresholded regression) and across all
  inference entry points.
- Test-time bin labels outside the calibration-time label space MUST raise
  `ValidationError` listing the unknown labels. An out-of-vocabulary category has no
  fitted calibrator; no default, nearest-neighbor, or global substitute may be
  applied implicitly.
- Structural validation (`len(bins) == len(x)` and label-space membership) MUST
  occur at the public entry boundary (wrapper and `CalibratedExplainer` methods) and
  MUST surface as CE exceptions per ADR-002; errors raised by calibrator internals
  or third-party libraries MUST NOT be the user-visible failure mode for
  misconfigured conditional calls.

### D4. Lifecycle and state

- Conditional configuration is **per calibration**. Each `calibrate` call MUST derive
  conditional state exclusively from its own arguments: a `calibrate` call without
  `bins`/`mc` on a previously conditional wrapper MUST reset `self.mc` and produce a
  globally calibrated explainer. Reusing the previous explainer's bins at calibrate
  time MUST NOT happen implicitly.
- **Opt-in persistence.** Users who intend to stay conditional across re-calibration
  MAY opt in explicitly: `calibrate(..., reuse_conditional=True)` reuses the stored
  `mc` by applying it to the new calibration data. It MUST raise `ValidationError`
  when no `mc` is stored — an inline-`bins` calibration cannot transfer, because the
  previous labels are aligned with the previous calibration set; the caller must pass
  fresh `bins` instead. It MUST also raise `ValidationError` when combined with
  `bins=` or `mc=` in the same call (D2 exclusivity). The parameter name is subject
  to ADR-038 taxonomy review before implementation (a boolean per-call toggle on an
  existing method).
- `is_mondrian()` on the explainer and calibrators remains the discovery predicate
  and MUST reflect the active calibration only.

### D5. Serialization (with ADR-031)

- Calibrator-level bins MUST round-trip through pickle and `save_state`/`load_state`
  (guaranteed via the ADR-031 calibrator primitives).
- `mc` is not guaranteed picklable. Until a portable representation exists,
  persistence MUST be loud: `save_state`/`__getstate__` MUST emit a `UserWarning`
  (and INFO log, per the fallback-visibility policy) when dropping a configured `mc`,
  and a loaded conditional wrapper without `mc` MUST behave as bins-calibrated
  (D3: explicit `bins` required, clear error when omitted). Silent post-load behavior
  change MUST NOT occur.

### D6. Capability and verification chain

- `CE-CAP-MOND-001` MUST be extended (or companion claims added) to cover: inline
  `bins` calibration, regression task types (percentile and thresholded), and the
  inference-consistency contract in D3. Requirements MUST include negative-path
  obligations (omitted bins, unknown labels, bins on global calibration) verified via
  TIF scenarios through `WrapCalibratedExplainer`.

### D7. Documentation contract

- One canonical how-to MUST document both channels with runnable examples that
  configure the conditional channel **at calibrate time first**, and state the
  calibrate/inference consistency rule (D3), the per-bin minimum sample-size
  guidance, and the serialization caveat (D5).
- Every documentation surface that demonstrates conditional usage (playbooks, task
  pages, agent guides, skills) MUST show calibrate-time channel configuration before
  any inference-time `bins`, MUST pass category-label arrays (never categorizer
  objects) as `bins`, and MUST use the categorizer API of the pinned crepes version.

## Alternatives Considered

1. **Implementation-defined leniency (status quo).** Accept omitted, extraneous, or
   out-of-vocabulary bins and resolve them heuristically per code path. Rejected:
   every heuristic resolution either misassigns categories or degrades to
   non-conditional output while remaining indistinguishable from correct conditional
   output to the caller; violates the repository fallback-visibility policy (no
   silent fallbacks).
2. **Silent fallback to global calibration when bins are omitted at inference.**
   Rejected: defeats the purpose of conditional calibration exactly where it matters
   (subgroup validity), and a silent fallback in a fairness-sensitive path is the
   worst failure mode; also violates fallback-visibility policy.
3. **Require `mc` for all conditional use (drop inline `bins`).** Rejected: breaks
   the legacy stable API (ADR-020), the published usage in `demo_conditional.ipynb`,
   and workflows where labels come from external attributes not derivable from `x`
   (e.g., protected attributes held outside the feature matrix).
4. **Explicit `bins` at inference overrides a configured `mc` (instead of erroring).**
   Rejected for now: silently diverging from the calibrated category scheme risks
   label-space mismatch that cannot be validated cheaply; fail-fast is preferred.
   Revisit if a concrete override use case emerges (Open Questions).
5. **Typed call-time object (`ConditionalSpec`) per ADR-038 replacing `bins`/`mc`.**
   Deferred, not rejected: a Strategy-tier spec object is the natural ADR-038 shape,
   but introducing a new typed configuration surface immediately before the v1.0.0
   freeze would add unvetted API to the contract being frozen. Recorded as a
   post-1.0 candidate follow-up.

## Consequences

### Positive

- Conditional misconfiguration fails fast at the public boundary with
  self-explanatory CE exceptions; conditional output can be trusted to actually be
  conditional — the property fairness and compliance workflows depend on.
- A single documented precedence and consistency contract that docs, skills,
  capability claims, and plugins (ADR-013 `requires_bins`) can align to.
- Serialization behavior becomes predictable and observable.

### Negative / Risks

- Strict validation is a behavior change relative to prior releases: calls that
  previously returned results in the uncodified areas will raise. Mitigated by
  landing pre-RC (v0.11.5) with CHANGELOG migration notes; the superseded behaviors
  are classified as correctness defects, not supported API. No new deprecation
  cycles are introduced, because the v1.0.0-rc gate requires an empty
  active-deprecation ledger.
- Unseen-label validation adds a per-call `np.unique`/set check; cost is O(n) and
  negligible relative to calibrator inference.
- `calibrate`-resets-conditional-state (D4) changes re-calibration semantics for
  users who expected `mc` to persist across calibrations; mitigated by the explicit
  `reuse_conditional` opt-in and a clear CHANGELOG migration note.

## Adoption & Migration

1. **v0.11.5 (dedicated pre-RC hardening milestone, inserted 2026-07-07):** implement
   D2–D5 and land the D7 documentation alignment across all conditional
   documentation surfaces in the same milestone, before the v1.0.0-rc API freeze.
   Task breakdown, defect inventory, and affected-surface inventory:
   `development/current-work/v0.11.5_plan.md`.
2. **No new deprecation cycles.** The v1.0.0-rc gate requires an empty
   active-deprecation ledger, so v0.11.5 MUST NOT introduce `DeprecationWarning`
   cycles for this remediation. All D2–D5 changes ship as fail-fast corrections with
   CHANGELOG migration notes; the superseded behaviors are classified as correctness
   defects (see the v0.11.5 plan), not supported API. Migration guidance: callers
   performing inference over the calibration set itself pass its bins explicitly;
   callers re-calibrating conditionally use `reuse_conditional=True` (D4).
3. **Verification:** extend `tests/capabilities/test_mondrian_contracts.py` and the
   TIF layer per D6; each defect class inventoried in the v0.11.5 plan gets a
   regression test that fails before and passes after remediation.
4. **Consolidation:** bins resolution and validation MUST have a single shared
   implementation (one source of truth for the D2/D3 precedence and validation
   rules) used by both the wrapper and the core explainer entry points; per-path
   reimplementations are prohibited.

## Governed claims

- `CE-CAP-MOND-001`

## Open Questions

- Explicit `bins=` overriding a configured `mc` is deferred until a concrete
  post-v1.0 Strategy-tier use case is proposed. v0.11.5 rejects the combination
  with `ConfigurationError`.
- Portable `mc` persistence is deferred post-v1.0. v0.11.5 adopts the D5
  loud-drop contract: `UserWarning` plus INFO log, with explicit `bins=` required
  after load.
- A typed ADR-038 `ConditionalSpec`/`MondrianOptions` surface is deferred
  post-v1.0. The intended shape is Strategy-tier, not Session `*Config`.
- Per-bin minimum-size warnings remain documentation-only for v0.11.5; a future
  warning policy would need a separate threshold and evidence decision.
