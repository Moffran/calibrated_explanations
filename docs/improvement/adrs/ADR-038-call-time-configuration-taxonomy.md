> **Status note (2026-06-12):** Last edited 2026-06-12 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.11.3. Establishes the call-time configuration taxonomy and naming conventions. Canonical examples are `RejectPolicySpec` (Strategy) and `GuardedOptions` (Tuning).

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
| `guarded=True` flag | (deprecated) | boolean → `guarded_options=GuardedOptions()` | deprecated |
| `significance=` kwarg | (deprecated) | kwarg → `GuardedOptions(confidence=...)` | deprecated |

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

- `guarded=True` boolean flag and `significance=` kwarg require ADR-011 deprecation
  cycles before removal.
- Existing code using `significance=0.1` must migrate to `GuardedOptions(confidence=0.9)` —
  numerically inverted, not just renamed; migration notes are mandatory.
- Plugin authors must learn a three-suffix convention rather than a single `Config` pattern.

## Delivery Governance

Implementation is governed by the v0.11.3 release plan (Task 17). ADR-011 deprecation
process applies to all renamed or replaced public surfaces. The canonical examples
(`GuardedOptions`, `reject_confidence`) must be present in the root namespace per
ADR-020 before the task is marked complete.
