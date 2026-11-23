> **Active scope:** Governing architectural decision for the `*Spec`/`*Options`/`*Config` naming taxonomy for call-time configuration surfaces. The `**kwargs` graduation gate (Gap 1, v1.0.0-rc) is an implementation milestone within this ADR's lifecycle. The unknown-kwarg policy for closed/stable public surfaces (§3) and the multi-label spelling question were resolved 2026-07-08 (v0.11.6 Task 5 series): unknown and cross-method keyword arguments now fail fast with `ConfigurationError` on both `WrapCalibratedExplainer` and `CalibratedExplainer`, validated per method against allow-lists with a single source of truth — see the Addendum below.

> **Status note (2026-07-08):** Last edited 2026-07-08 (Addendum: fail-fast call-time kwarg validation, v0.11.6 Task 5 series) · Archive after: Retain indefinitely as architectural record · Implementation window: v0.11.3 (base decision), v0.11.6 (fail-fast addendum). Establishes the call-time configuration taxonomy and naming conventions. Canonical examples are `RejectPolicySpec` (Strategy) and `GuardedOptions` (Tuning).

# ADR-038: Call-time Configuration Taxonomy and Naming Conventions

Status: Accepted
Date: 2026-06-12
Deciders: Core maintainers
Reviewers: API, Plugin, and Governance maintainers
Supersedes: None
Superseded-by: None
Related: ADR-034-centralized-configuration-management, ADR-006-plugin-registry-trust-model, ADR-011-deprecation-and-migration-policy, ADR-020-legacy-user-api-stability, ADR-029-reject-integration-strategy, ADR-032-guarded-explanation-semantics

## Context

CE has four distinct configuration tiers, but only the deployment tier (ADR-034,
`ConfigManager`) has a governing document. The remaining three tiers — session-level
setup, per-call strategy selection, and per-call numeric tuning — have grown
organically without a shared vocabulary or naming convention. The result is that
`RejectPolicySpec` (strategy) and `ExplainerConfig` (session) already use different
suffixes by coincidence rather than contract, `**kwargs` is used for per-call tuning
with no runtime validation, and plugin authors have no canonical pattern to follow.

The immediate trigger is the `significance` / `confidence` ambiguity across the
guarded-explanation and reject paths: two parameters that are mathematical inverses
(`significance = 1 − confidence`) were independently named, are used in overlapping
call contexts, and have no governing document that would have prevented the drift.
Formalizing the taxonomy provides the rule that would have caught this inconsistency
at design time.

## Decision

### 1. Four-tier taxonomy

| Tier | When set | Governed by | Controls |
|---|---|---|---|
| **Deployment** | Process startup | ADR-034 (`ConfigManager`) | Env vars, `pyproject.toml`, plugin selection, cache/parallel deployment |
| **Session** | Explainer construction | This ADR — `*Config` | Per-explainer behavioral settings (parallelism, cache policy, feature-filter) |
| **Strategy** | Per call; or at construction as an explainer-wide default | This ADR — `*Spec` | Which algorithm or behavior path is activated |
| **Tuning** | Per call | This ADR — `*Options` / qualified kwarg | Numeric thresholds and operational parameters for the chosen path |

Each tier is strictly independent. A higher tier sets structural constraints; a lower
tier cannot override them. Session configuration is immutable after construction.
Strategy and Tuning objects are immutable after construction and must not be mutated
by the callee.

### 2. Naming conventions

#### 2a. Session configuration — `*Config`

A `*Config` dataclass governs per-explainer behavioral settings set once at
construction time and retained for the session lifetime. Its fields represent the
behavioral envelope available to all calls made through this explainer instance.

- Suffix: `Config`
- Mutability: frozen after the owning object is constructed
- Instantiation: at `CalibratedExplainer` or `ExplainerBuilder` construction
- Canonical examples: `ExplainerConfig`, `ParallelConfig`, `CacheConfig`

#### 2b. Strategy / policy selection — `*Spec`

A `*Spec` dataclass bundles the choices that determine *which algorithm or behavior
variant* is activated for a call. It selects the behavior path; it does not set
numeric thresholds. Two calls using the same `*Spec` and the same inputs must produce
semantically equivalent results regardless of tuning values.

- Suffix: `Spec`
- Mutability: frozen after construction
- Instantiation: by the caller; supplied per call or at `CalibratedExplainer` / `ExplainerBuilder` construction
- Canonical example: `RejectPolicySpec` (which NCF mode, which reject policy variant)
- Rule: a `*Spec` MUST NOT contain floating-point coverage thresholds or numeric
  tuning values. Those belong in Tuning.

A `*Spec` supports two lifecycle placements:

| Placement | Scope | Example |
|---|---|---|
| Method argument | That invocation only | `explain_factual(x, reject_policy=RejectPolicySpec.flag())` |
| Constructor argument | All calls through this explainer instance | `CalibratedExplainer(..., reject_policy=RejectPolicySpec.flag())` |

When a per-call `*Spec` is provided it takes precedence over any explainer-wide
default for that invocation. The `*Spec` type itself does not encode lifecycle; the
caller determines granularity by where they supply it. This distinguishes `*Spec`
from `*Config` (which is exclusively session-level) and from `*Options` (which is
exclusively per-call).

#### 2c. Per-call tuning — `*Options` dataclass (3+ parameters)

A `*Options` dataclass bundles three or more related numeric or operational
parameters that all tune the same internal object or algorithm instance. The
threshold for bundling is three parameters: fewer than three related tuning
parameters do not warrant a bundle object.

- Suffix: `Options`
- Mutability: frozen after construction
- Instantiation: by the caller, passed per call
- Typo safety: dataclass `__init__` rejects unknown fields at construction time —
  no silent `**kwargs` swallowing
- Canonical example: `GuardedOptions` (tunes `InDistributionGuard`: `confidence`,
  `n_neighbors`, `normalize`, `merge_adjacent`, `verbose`)

#### 2d. Per-call tuning — qualified flat kwarg (fewer than 3 parameters)

When a call path has only one or two numeric tuning parameters, they are passed
as explicit keyword arguments with a qualifier prefix that identifies the path
they belong to: `<context>_<name>`.

- Pattern: `<context>_<name>` (e.g., `reject_confidence`)
- Mutability: not applicable (scalar)
- Canonical example: `reject_confidence` (the coverage level for the reject path)
- Rule: the qualifier MUST match the functional context of the parameter, not the
  internal implementation detail. `reject_confidence` is correct; `orchestrator_confidence`
  is not.

### 3. `**kwargs` in public API signatures

Unvalidated `**kwargs` in stable public API method signatures is non-compliant for
new surfaces as of this ADR. Existing `**kwargs` uses are legacy exceptions tracked
by the CI deprecation script.

When a stable public method needs to accept call-time tuning parameters:

- If 3+ parameters govern the same internal object: define a `*Options` dataclass
  and accept it as a single named argument.
- If fewer than 3: add them as explicit keyword-only arguments.
- `**kwargs` reserved for internal orchestration boundaries (not public API) where
  forwarding to private functions is required.

**Exception — experimental surfaces under active development:**

A surface explicitly marked as experimental MAY use `**kwargs` while its parameter
contract is being settled, subject to all three of the following conditions:

1. **Explicit experimental marker.** The method carries a `@experimental` decorator,
   a `[EXPERIMENTAL]` tag in its docstring, or a name prefix that signals instability
   to callers. Absence of a marker disqualifies the exception.
2. **Unknown-kwarg handling.** The `**kwargs` path MUST either validate against a
   known-valid set and emit a warning on unknowns, or document in the experimental
   tag that unknown arguments are silently ignored and callers should expect noise.
   Silent discard with no signal is non-compliant even in experimental surfaces.
   This is the floor for *experimental* surfaces only; closed/stable enumerable
   surfaces (e.g. the per-method kwarg allow-lists on `WrapCalibratedExplainer` and
   `CalibratedExplainer`) are held to the stricter fail-fast policy in the
   2026-07-08 Addendum below.
3. **Graduation gate.** `**kwargs` MUST be replaced with explicit typed arguments
   before the surface transitions out of experimental status. This is a hard gate,
   not a soft intention.

The exception exists because the cost of ADR-011 deprecation cycles during parameter
design exploration is disproportionate. The guard rails ensure callers are never
silently misled: they know the surface is unstable, and unknown arguments produce
some observable signal.

### 4. Canonical summary of CE call-time configuration surfaces

| Surface | Tier | Type | Parameter name |
|---|---|---|---|
| Reject algorithm selection | Strategy | `RejectPolicySpec` | `reject_policy=` |
| Reject coverage threshold | Tuning (single) | qualified kwarg | `reject_confidence=` |
| Guard tuning bundle | Tuning (grouped) | `GuardedOptions` | `guarded_options=` |
| `guarded=True` flag | Removed | boolean → `guarded_options=GuardedOptions()` | removed in v0.11.5 |
| `significance=` kwarg | Removed | kwarg → `GuardedOptions(confidence=...)` | removed in v0.11.5 |

### 5. Plugin compliance

Plugin authors MUST follow the same taxonomy for any configuration surface exposed
through the plugin contract:

- Plugin algorithm or mode selection: `*Spec` dataclass
- Plugin grouped call-time tuning (3+ params for one internal object): `*Options` dataclass
- Plugin single scalar tuning: qualified flat kwarg with the plugin context as qualifier
- Plugin session setup: `*Config` dataclass
- Unvalidated `**kwargs` at the plugin public boundary: non-compliant

Plugin configuration that flows through `ConfigManager` (deployment tier) is
governed by ADR-034 §8 and is separate from this taxonomy.

### 6. Semantic inversion rule for `*Options` replacing `significance`

When a `*Options` field replaces a parameter whose numeric convention was expressed
as an alpha/significance level (e.g., `significance=0.1`), the replacement field
MUST express the same concept as a coverage/confidence level (`confidence = 1 − significance`)
so that higher values always mean "more inclusive" or "stricter requirement" in the
coverage sense, consistent with the established `reject_confidence` convention.

`GuardedOptions.confidence = 0.9` is the canonical replacement for `significance = 0.1`.
Both express the same conformity threshold; the coverage convention makes the
relationship with `reject_confidence` immediately readable.


## Governed claims

- `CE-CAP-CONFIG-001` — Runtime and call-time configuration use governed tiers, centralized reads, and suffix conventions for Config, Spec, and Options objects.
- `CE-CAP-REJECT-001` — calibrated_explanations supports reject/defer policies: passing a RejectPolicySpec   to WrapCalibratedExplainer.explain_factual or explore_alternatives tags each   instance's explanation envelope with a rejection status (FLAG, ONLY_REJECTED,   ONLY_ACCEPTED) based on the calibrated uncertainty score.
- `CE-CAP-GUARD-001` — calibrated_explanations supports guarded explanations: passing a GuardedOptions   instance to WrapCalibratedExplainer.explain_factual or explore_alternatives restricts   explanation output to instances that lie within the support of the calibration data,   returning a CalibratedExplanations collection.

## Alternatives Considered

1. **Use `*Config` for all grouped parameter bundles (session and per-call alike).**
   Rejected: `*Config` already carries the meaning "session-level, set at construction."
   Reusing it for per-call bundles destroys the tier signal the name provides.

2. **Use `**kwargs` with a runtime allowlist for per-call tuning.**
   Rejected: allowlists are maintenance burden and silently fail on typos until the
   allowlist is consulted. A dataclass rejects unknown fields at the `__init__` call
   site with a clear `TypeError`.

3. **Single flat `*Spec` for both strategy and tuning.**
   Rejected: mixing algorithm selection and numeric thresholds in one object makes
   the strategy non-reusable across different confidence levels and creates a new
   confusable surface (strategy choices vs numeric values in the same object).

4. **Rename `reject_confidence` to `RejectOptions(confidence=...)` for symmetry.**
   Rejected: one scalar does not warrant a bundle object. The qualified-kwarg pattern
   is proportionate and sufficient for a single tuning value.

## Consequences

**Positive:**

- Readers of any CE call site can immediately identify the configuration tier from
  the suffix: `*Spec` = algorithm/strategy choice (per-call or explainer-wide default),
  `*Options` = per-call numeric tuning, `*Config` = exclusively session-level setup.
- Typo safety at `*Options` construction is automatic and immediate (dataclass `TypeError`).
- Plugin authors have a canonical pattern that prevents the `significance`/`confidence`
  class of ambiguity in new plugin surfaces.
- The `significance` / `confidence` naming inconsistency is resolved by the convention:
  both surfaces now express coverage thresholds (`reject_confidence`, `GuardedOptions.confidence`).

**Negative / Risks:**

- `guarded=True` boolean flag and `significance=` kwarg completed their ADR-011
  deprecation cycle and were removed in v0.11.5.
- Existing code using `significance=0.1` must migrate to `GuardedOptions(confidence=0.9)` —
  numerically inverted, not just renamed; migration notes are mandatory.
- Plugin authors must learn a three-suffix convention rather than a single `Config` pattern.

## Delivery Governance

Implementation is governed by the v0.11.3 release plan (Task 17). ADR-011 deprecation
process applies to all renamed or replaced public surfaces. The canonical examples
(`GuardedOptions`, `reject_confidence`) must be present in the root namespace per
ADR-020 before the task is marked complete.

## Addendum (2026-07-08): Fail-fast call-time kwarg validation (v0.11.6 Task 5)

Delivered across v0.11.6 Task 5 and its follow-ups 5A–5D; recorded here as a single
consolidated decision.

### Decision

**D3 — Unknown-kwarg policy for closed/stable public surfaces is fail-fast.**
Unrecognized keyword arguments on a closed, enumerable public surface (a `**kwargs`
path validated against a fixed known-name set, as opposed to an experimental
plugin-forwarding seam under the §3 exception) MUST raise `ConfigurationError`, not
emit a warning. This resolves the ambiguity left open by §3's "MUST either validate
... and emit a warning on unknowns, or document ... silently ignored" language —
which governs the *experimental-surface* floor only (see the note appended to §3
condition 2 above) — and supersedes the warn-and-forward behavior introduced in
v0.11.4 Task 15.

The policy is realized through five subsidiary rules:

1. **Per-method allow-lists.** Every gated method validates against its own allowed
   set, not a shared flat one. A name recognized on another method but not valid on
   the called method raises `ConfigurationError` ("recognized on another method but
   not valid here") instead of being silently accepted and inert.
2. **Single source of truth.** The per-surface sets are defined once, in
   `core/calibrated_explainer.py`; `core/wrap_explainer.py` *derives* its gates from
   them, never re-enumerating a surface. Invariant: anything accepted by
   `CalibratedExplainer` on a surface must also be accepted by the wrapper's
   corresponding method; the wrapper only subtracts the internal `_ce_skip_reject`
   escape hatch and adds wrapper-only names (`reuse_conditional`).
3. **Both public entry classes are covered.** The gates apply to
   `WrapCalibratedExplainer` (`calibrate`, `explain_factual`, `explore_alternatives`,
   `explain_fast`, `predict`, `predict_proba`) and to `CalibratedExplainer` used
   directly. On `CalibratedExplainer`, `__init__`/`predict`/`predict_proba` are fully
   closed surfaces; `explain_factual`/`explore_alternatives` reject only names known
   on a closed surface but not valid here, while genuinely unrecognized names still
   pass through to explanation plugins — the §3 experimental exception
   (`multi_labels_enabled`, `interval_summary`, arbitrary plugin-forwarded kwargs)
   is preserved. `explain_fast` is already fully typed and needs no gate.
4. **`normalize` is a removed alias, not a parameter.** It moved out of the
   allow-list into `REMOVED_NORMALIZATION_KWARG_MAP` /
   `reject_removed_normalization_kwarg()`, raising `ConfigurationError` at the
   public gate with `normalization=NormalizationStrategy.<MEMBER>` migration
   guidance. All four removed-alias check families (removed public aliases, removed
   guarded kwargs, removed reject kwargs, removed `normalize`) are applied
   consistently on every gated method of both classes.
5. **`mondrian_categorizer=` is an intentional, documented alias for `mc=`** on
   `calibrate()` (short form kept canonical, descriptive form for discoverability);
   supplying both raises `ConfigurationError`.

**D4 — `multi_labels_enabled` is confirmed the sole multi-label spelling.** A
repo-wide search of `src/` and `tests/` (2026-07-08) found no alias (`multi_label`,
`multilabel`, `multi_class_labels`, etc.) in use anywhere. No code change was
required; this closes D4 as compliant-as-is.

Collateral decisions required to make fail-fast safe to ship:

- `calibrate()` flips `self.calibrated = False` only after every validation gate has
  passed, immediately before constructing the replacement explainer. A rejected
  `calibrate()` call (typo'd or cross-method kwarg) leaves the previous calibration
  fully usable.
- The legacy `plot_global` path forwards only prediction-relevant kwargs (`bins`,
  plus `low_high_percentiles` on the regression `predict` branch) into the gated
  prediction methods, mirroring the PlotSpec path; plot-only keys never reach the
  prediction gates.
- `perf_cache`/`perf_parallel` are public at `calibrate()` and forwarded with
  `kwargs.setdefault(...)`, so a call-time value overrides the wrapper-level
  attribute.
- Unknown-kwarg errors report the per-method surface
  (`WrapCalibratedExplainer.calibrate received unknown keyword arguments: ...`),
  not just the class name.

### Rationale

D3 was inherited as an open binary choice from the v1.0.0-rc plan re-baseline
(`development/current-work/v0.11.6_plan.md`, decisions table). The Global Rules of
that plan already implied fail-fast for removed/unknown kwargs under
`ConfigurationError`, and Tasks 2–4 of the same milestone had implemented fail-fast
for every sibling silent-kwarg-sink defect (removed guarded kwargs, removed reject
kwargs, coercer fallback resolution). Warn-and-forward — the pre-v0.11.6 behavior —
left typos and inert parameters producing no actionable signal; see the CHANGELOG
entry under `## [Unreleased]` for the user-facing reversal notice.

**Why per-method allow-lists and a single source of truth.** The first fail-fast
implementation validated only "is this name known *anywhere*," never "is this name
valid *for this call*": a name meaningful on one method was silently
accepted-but-inert on the other five (confirmed concretely: `calibrate(x, y,
guarded_options=GuardedOptions())` passed the gate, then vanished into unconsumed
constructor kwargs with no error and no effect). Splitting the list per method
fixed that, but duplicating the resulting sets across the wrapper and
`CalibratedExplainer` caused drift within the same release: the wrapper wrongly
rejected `reject_confidence` on the explain methods, `interval_summary` on
`predict()`, and eight live `__init__` parameters on `calibrate()` (`noise_type`,
`scale_factor`, `severity`, `sample_percentiles`, `suppress_crepes_errors`,
`reject`, `perf_cache`, `perf_parallel` — including the FAST-mode tuning knobs
while `fast=True` itself was allowed). Deriving the wrapper's gates from the
`CalibratedExplainer` definitions makes that class of drift structurally
impossible, and the invariant is additionally enforced by tests (below).

**Why `normalize` is handled as a removed alias.** `normalize`/`normalization`
looked like accidental synonyms but are not: `normalize` is a legacy passthrough
removed in v0.11.5 that always raises when set. While allow-listed it looked valid,
then raised the wrong exception class (`ValidationError` instead of the
`ConfigurationError` every other removed alias raises) three calls deep inside
`VennAbers.predict_proba`, well past the public boundary — while `normalization=`
itself was silently inert on `calibrate()` (the cross-method bug again, using the
exact pair flagged as confusing). `VennAbers.predict_proba`'s own direct-call
`ValidationError` behavior (Task 4) is untouched; that boundary still validates its
own parameter values when the wrapper is bypassed. By contrast,
`mc`/`mondrian_categorizer` is a genuinely intentional alias pair and is documented
as such rather than eliminated.

**Why `CalibratedExplainer` is covered directly.** It is a fully documented,
supported direct-use class (its own docstring demonstrates
`CalibratedExplainer(learner, X_cal, y_cal, mode=...)`); "recommended use is
`WrapCalibratedExplainer`" does not make direct use unsupported. Before this
addendum a user on that path got none of the fail-fast protections. The surface
also carries higher blast radius than the wrapper, because plugin/viz code forwards
*the same kwargs dict* across sibling methods: `CalibratedExplainer.plot()`
re-injects `style_override` into the kwargs it hands `plotting.plot_global()`,
which forwards them (plus `show=` if passed) into `predict()`/`predict_proba()`.
`predict()` therefore gained the same defensive `show`/`style_override` strip
`predict_proba()` already had, and the legacy `plot_global` path was changed to
forward only prediction-relevant kwargs — naive fail-fast would otherwise have
broken every documented `plot(x, use_legacy=True)` call and the automatic
PlotSpec→legacy fallback (and briefly did, before the whitelist fix landed in the
same milestone).

**Why the allow-list contents were audited rather than inherited.** The original
list (v0.11.4 Task 15, commit `b1a49d33`) was a broad enumeration of kwarg-shaped
names observed at the time, not a verified inventory of real public parameters.
Tracing every name to its actual consumer found three kinds of defect:

- **Missing real names.** `categorical_labels`, `class_labels`,
  `features_to_ignore`, `oob`, and the plugin-selection kwargs (`factual_plugin`,
  `alternative_plugin`, `fast_plugin`, `interval_plugin`, `fast_interval_plugin`,
  `plot_style`) were documented/notebook-used but absent — raising on them would
  itself have been a regression. All were added before the switch was flipped.
- **Dead names.** `condition`, `condition_label`, `condition_labels`, and
  `include_reject_details` have no consumer anywhere in `src/` and no ADR
  reference; they now raise. Alternatives rejected: implementing the implied
  features (no spec exists to build from), documenting them as reserved no-ops
  (perpetuates the confusing surface), defensively popping them (reintroduces
  silent kwarg-swallowing).
- **Internal-only names.** `output_interval` and `y_threshold` are set internally
  by `CalibratedExplainer.predict_proba` for its own interval-learner calls; a
  caller passing either would clear the public gate, then hit `TypeError: got
  multiple values for keyword argument` several calls deeper — the exact confusing
  failure mode fail-fast exists to eliminate. Both were removed from the public
  surface; the internal call sites are unchanged.
- Also dropped: `mc`/`uq_interval` (always captured by an explicit formal parameter
  wherever meaningful, never reaching `**kwargs`) and `show`/`style_override` (only
  meaningful to `plot()`, a seventh wrapper surface this gate does not cover —
  noted for a future task). A genuinely dead `perturb=True` kwarg was also removed
  from a pre-existing integration test.

**Verification lessons baked into the tests.** Two rounds of near-misses showed
that restricting a surface requires auditing real usage and running the full suite,
not just the tests written for the change: a first pass wrongly excluded
`reject_confidence` from the explain methods (two pre-existing tests proved it is
live whenever `reject_policy` is set), and the legacy-plot breakage escaped an
earlier verification round because its regression test used `show=False`, which
silently no-ops on the legacy path whenever matplotlib has not yet been loaded into
the compat module — an order-dependent vacuity. The contract suite (below)
therefore exercises every allow-listed name end-to-end and tests the legacy plot
path with `show=True` under the Agg backend.

**Alternative rejected: full explicit-parameter promotion now.** `threshold`,
`low_high_percentiles`, `classes`, `feature`, `bins`, and `reject_policy` were
confirmed as good candidates for promotion to explicit typed parameters (the
`CalibratedExplainer`/orchestrator layer already treats them as stable, typed,
explicit parameters). That promotion remains the Gap 1 graduation gate
(v1.0.0-rc); per-method scoping closes the cross-method inconsistency now, without
a signature change, at lower risk.

### Implementation

- `api/params.py`: `reject_unknown_public_kwargs()` (fail-fast against an allowed
  set), `reject_cross_surface_kwargs()` (rejects names present in a
  `closed_surface_names` reference set but absent from `allowed`; a name in neither
  passes through untouched — the §3 exception), and
  `REMOVED_NORMALIZATION_KWARG_MAP` / `reject_removed_normalization_kwarg()`,
  mirroring the existing `reject_removed_guarded_kwargs` /
  `reject_removed_reject_kwargs` helpers.
- `core/calibrated_explainer.py`: canonical per-surface frozensets
  (`_INIT_KWARGS`, `_PREDICT_KWARGS`, `_PREDICT_PROBA_KWARGS`, `_EXPLAIN_KWARGS`),
  the `_INIT_EXPLICIT_PARAMS` reference set (the explicit `__init__` formals
  besides `learner`/`x_cal`/`y_cal`), and `_CLOSED_SURFACE_KWARGS` (their union
  plus the explicit formal names of the closed methods, so names that never reach
  `**kwargs` on their own methods are still recognized as "known elsewhere" by the
  cross-surface check). All gated methods apply the four removed-alias checks plus
  the appropriate rejection helper; `predict()` gains the `show`/`style_override`
  defensive strip.
- `core/wrap_explainer.py`: derived gates — `_CALIBRATE_KWARGS = _INIT_KWARGS |
  _INIT_EXPLICIT_PARAMS | {"reuse_conditional"}`, `_EXPLAIN_KWARGS` taken
  verbatim, `_PREDICT_KWARGS`/`_PREDICT_PROBA_KWARGS` minus `_ce_skip_reject`;
  `_KNOWN_PUBLIC_KWARGS` remains only as the union used for the first-pass "known
  at all" check. `_normalize_public_kwargs()` raises `ConfigurationError` both for
  unknown names and for known-elsewhere names, labelled with the per-method
  `surface=`. `calibrate()` validates before invalidating, gains the
  `mondrian_categorizer=` keyword-only alias, and forwards
  `perf_cache`/`perf_parallel` with `setdefault` semantics.
- `viz/_matplotlib_compat.py`: `plot_global` builds a `prediction_kwargs`
  whitelist instead of forwarding `**kwargs` into `predict`/`predict_proba`.
- Experimental surfaces are unaffected: `multi_labels_enabled` /
  `interval_summary` on the explain methods remain `[EXPERIMENTAL]`-tagged plugin
  passthroughs under the §3 exception. Gap 1 (replacing that `**kwargs` with
  explicit typed arguments) remains targeted at v1.0.0-rc and is unchanged by this
  addendum.

### Testing

- `tests/unit/core/test_parameter_surface_contracts.py` (new) guards the invariant
  three ways: structural contracts (wrapper set ⊇ explainer set per surface;
  `_INIT_EXPLICIT_PARAMS`/`_EXPLAIN_FAST_KWARGS` verified against the live
  signatures via `inspect`), acceptance matrices (every allow-listed name on every
  gated method exercised end-to-end with a benign value, with a completeness
  assertion so a newly allow-listed name cannot land without an acceptance test),
  and explicit regressions for the legacy plot path run with `show=True` under the
  Agg backend.
- `tests/unit/core/test_wrap_explainer_core.py`: `ConfigurationError` (not
  `UserWarning`) on unknown kwargs; acceptance of each documented name; rejection
  of the dead and internal-only names; the removed `normalize` alias; the
  `mondrian_categorizer` alias and its `mc` conflict error; cross-method rejection
  parametrized across all six wrapper methods.
- `tests/unit/core/test_calibrated_explainer_more_paths.py`: unknown kwarg on
  `__init__`; cross-surface rejection on `predict`/`predict_proba` (full
  fail-fast) and on the explain methods (narrow, experimental-preserving); a
  genuinely-unknown kwarg still passes through the explain methods untouched;
  removed-alias consistency across methods; the `_ce_skip_reject` internal escape
  hatch still works; `.plot()` still works end-to-end.
- Full unit and integration suites pass, with viz-marked, slow-marked,
  plugins/legacy, and capabilities/docs suites re-run explicitly given the
  plot-path blast radius (one pre-existing unrelated failure predates these
  changes).

### Delivery Governance

Delivered in v0.11.6 Task 5 and its follow-ups 5A–5D
(`development/current-work/v0.11.6_plan.md`, §5–§5D).

## Addendum (2026-11-23): Graduation of `multi_labels_enabled` / `interval_summary` (v1.1.0 T12, #206)

The §3 graduation gate is met for both explain-time controls:

- `CalibratedExplainer.explain_factual`/`explore_alternatives` and the matching
  `WrapCalibratedExplainer` methods declare `multi_labels_enabled: bool = False`
  and `interval_summary: IntervalSummary | str | None = None` as explicit
  keyword-only parameters. Neither travels through `**kwargs` any more, and
  neither is `[EXPERIMENTAL]`.
- A plain `bool` is enough for `multi_labels_enabled`. Fewer than three
  parameters govern multi-label output, so §3 calls for flat keyword-only
  arguments, not a `MultiLabelOptions` object.
- Validation happens once, at `CalibratedExplainer` (the wrapper forwards the
  same values): `validate_bool_parameter()` and `coerce_interval_summary()`,
  both raising `ValidationError` (ADR-002). `interval_summary=None` means "no
  call-time override", so the explainer-level strategy applies.
  `multi_labels_enabled=True` on a regression explainer raises
  `ValidationError`.
- The explanation orchestrator receives both values as named arguments. The
  plugin receives the canonical `IntervalSummary` through
  `ExplanationRequest.interval_summary` only. Both names are removed from the
  orchestrator's framework-known extras set, and the plugin extension seam for
  other kwargs is unchanged (Addendum D3 rule 3).
- Both names stay in `_EXPLAIN_KWARGS`, so the wrapper gate derived from it
  (D3 rule 2) and the cross-surface check still recognise them.
- The guarded path (ADR-032) does not consume either control. Passing one with
  `guarded_options` emits a `UserWarning` plus an INFO log instead of being
  silently inert.
