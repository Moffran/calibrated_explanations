> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-026 — Explanation Plugin Semantics and Legacy Contracts

Status: Draft
Date: 2025-10-18
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None
Related: ADR-006, ADR-013, ADR-015, ADR-021

## Context

ADR-015 establishes the architecture that lets `CalibratedExplainer` outsource
explanation assembly to pluggable strategies. Contributors building new plugins
have asked for guidance comparable to ADR-021 (which documents interval
semantics) so they can reproduce the expected behaviour without reverse-
engineering the monolithic legacy implementation. While ADR-015 explains the
protocols and registry hooks, it does not capture the runtime guarantees that
legacy call paths rely upon: how requests are seeded, which calibrated
predictions must be available, how explanation batches interact with interval
and plot plugins, and which metadata a plugin is expected to surface.

This ADR codifies those semantics so that new plugins can be authored against a
clear contract without duplicating the in-tree implementation details.

## Decision

### 1. Orchestrator lifecycle and plugin resolution

* `CalibratedExplainer.explain*` delegates all work to `_invoke_explanation_plugin`,
  which resolves or instantiates a plugin per mode, initialises it once with an
  `ExplanationContext`, and reuses the instance for subsequent calls. Resolution
  draws from (in priority order) keyword overrides, environment variables,
  pyproject configuration, descriptor-declared fallbacks, and the default
  built-ins before giving up, so authors can override behaviour without touching
  code.【F:src/calibrated_explanations/core/calibrated_explainer.py†L421-L459】【F:src/calibrated_explanations/core/calibrated_explainer.py†L894-L959】
* Runtime metadata is validated before a plugin is accepted. The resolver checks
  schema versions, declared modes/tasks, and capability tags (`explain`,
  `explanation:{mode}`, and the active task or `task:both`) so misconfigured
  plugins fail fast instead of producing partially calibrated batches.【F:src/calibrated_explanations/core/calibrated_explainer.py†L571-L623】
* Once a plugin wins the resolution race, `_ensure_explanation_plugin` records
  any interval dependency hints advertised in its metadata, builds the immutable
  execution context, and invokes `plugin.initialize(context)`. Initialisation
  failures surface as configuration errors, matching the behaviour of the
  legacy adapters.【F:src/calibrated_explanations/core/calibrated_explainer.py†L968-L1006】
* `ExplanationContext` exposes frozen references to the task, mode,
  discretiser, feature metadata, helper handles (including the explainer under
  the `"explainer"` key), interval hints, and the calibrated
  `_PredictBridgeMonitor` that guards bridge usage. Plugins **must not** mutate
  any of these fields; helper handles are provided strictly for read-only
  compatibility.【F:src/calibrated_explanations/plugins/explanations.py†L28-L41】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1008-L1040】

### 2. Request payload and calibrated prediction usage

* Each invocation constructs an immutable `ExplanationRequest` that normalises
  per-call inputs: thresholds, percentile pairs (converted to tuples when
  present), Mondrian bins, the ignored-feature mask (always exposed as a tuple),
  and a shallow copy of any extras supplied by the caller.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1081-L1101】【F:src/calibrated_explanations/plugins/explanations.py†L43-L51】
* Legacy semantics still apply. When regression callers omit a threshold, the
  prediction helpers attach the calibrated percentile pair to the collection so
  downstream utilities can recover the same bounds the legacy code produced.
  Plugins that bypass the helpers must mirror this bookkeeping to stay
  compatible.【F:src/calibrated_explanations/core/prediction_helpers.py†L82-L110】
* Plugins **must** call at least one of the calibrated prediction bridge methods
  (`predict`, `predict_interval`, or `predict_proba`) for every batch. The
  `_PredictBridgeMonitor` resets before each invocation and raises a
  configuration error if the bridge is never exercised, keeping interval
  semantics aligned with ADR-013/ADR-021.【F:src/calibrated_explanations/core/calibrated_explainer.py†L138-L179】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1102-L1155】
* The bridge additionally validates that every returned triple satisfies the
  inclusive bounds invariant (`low <= predict <= high`). Any violation—whether
  detected by the bridge, the plugin, or downstream consumers of feature-level
  intervals—**must** be treated as a hard failure rather than coerced or
  truncated output. `[low, high]` pairs that do not contain their `predict`
  component are nonsensical in this system, and silently accepting them
  reintroduces the very calibration drift these ADRs forbid.
* The built-in adapters demonstrate the contract by invoking the bridge before
  delegating to the legacy `CalibratedExplainer` methods. New implementations may
  discard the return value but must perform a calibrated call so telemetry and
  interval tracking remain consistent.【F:src/calibrated_explanations/plugins/builtins.py†L321-L348】

### 3. Expected batch structure and collection materialisation

* `explain_batch` must return an `ExplanationBatch` whose `container_cls`
  inherits from `CalibratedExplanations`, whose `explanation_cls` derives from
  `CalibratedExplanation`, whose `instances` form a sequence of mappings, and
  whose metadata is mutable. The validator enforces all of these properties
  immediately after the plugin returns.【F:src/calibrated_explanations/plugins/explanations.py†L83-L155】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1102-L1120】
* Legacy adapters serialise each explanation as `{"explanation": instance}` and
  embed the already-instantiated collection under `collection_metadata["container"]`.
  `CalibratedExplanations.from_batch` simply unwraps that object, so custom
  plugins that rely on the default materialiser must supply the same entry (or a
  drop-in replacement) to preserve cached telemetry and helper behaviour.【F:src/calibrated_explanations/plugins/builtins.py†L107-L124】【F:src/calibrated_explanations/explanations/explanations.py†L151-L176】
* Runtime metadata seeding adds cross-cutting details such as the active task,
  resolved interval source, dependency hints, preprocessing snapshot, and plot
  fallback chain. Plugins are free to extend the mapping with their own keys but
  should avoid deleting runtime-provided values so downstream tooling observes a
  stable schema.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1121-L1150】
* Collections materialised from the batch must expose the same caches and
  attributes (`prediction_interval`, `feature_names`, `class_labels`, cached
  predictions/probabilities, etc.) that the legacy `CalibratedExplanations`
  supplies, otherwise helpers and JSON/plot exports will diverge.【F:src/calibrated_explanations/explanations/explanations.py†L24-L249】

### 4. Interaction with interval and plot plugins

* Explanation metadata seeds interval fallback hints. `_ensure_explanation_plugin`
  captures `interval_dependency` entries declared on plugin metadata, and the
  resolver aggregates factual/alternative hints (while keeping FAST hints
  separate) before attempting interval resolution. This preserves the legacy
  pairing of FAST explanations with FAST calibrators unless callers override the
  chain.【F:src/calibrated_explanations/core/calibrated_explainer.py†L968-L995】【F:src/calibrated_explanations/core/calibrated_explainer.py†L644-L655】【F:src/calibrated_explanations/plugins/builtins.py†L353-L442】
* The same metadata drives plot resolution. `_build_explanation_context` records
  the derived plot fallback chain in the immutable context and on the runtime
  state so plots triggered after an explanation run reuse the plugin-preferred
  style.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1014-L1059】
* Once a batch is accepted, the explainer mirrors the resolved interval source,
  dependency hints, and plot chain into both the batch metadata and the emitted
  telemetry payload, so downstream tooling can audit which calibrators and plot
  plugins participated in the run.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1121-L1150】

### 5. Plugin metadata and capability declarations

* Runtime validation requires capability tags `"explain"`, the mode-specific
  `"explanation:{mode}"` (or legacy `"mode:{mode}"`), and either
  `"task:{classification|regression}"` or `"task:both"`. Metadata must also
  declare supported tasks and modes; omissions cause the resolver to reject the
  plugin before any batch executes.【F:src/calibrated_explanations/core/calibrated_explainer.py†L571-L623】
* Built-in plugins demonstrate the expected declarations, including explicit
  `interval_dependency` and `plot_dependency` hints so the resolver can compose
  interval/plot fallbacks without additional configuration. Third-party plugins
  should model their metadata after these examples to avoid subtle resolution
  failures.【F:src/calibrated_explanations/plugins/builtins.py†L353-L442】
* Additional metadata fields (such as `dependencies` or `trusted`) are primarily
  advisory. The runtime currently inspects `interval_dependency`,
  `plot_dependency`, and trust settings to enforce ADR-006/ADR-013/ADR-014, so
  authors should keep those aligned with the calibrators and plotters they
  expect to use.

### 6. Legacy compatibility guarantees

* Built-in plugins preserve byte-for-byte outputs by delegating to
  `CalibratedExplainer` methods, disabling the plugin recursion flag, and
  wrapping the resulting collection in an `ExplanationBatch` that embeds the
  original container. New plugins targeting legacy compatibility may follow the
  same pattern until a fully plugin-native pathway is introduced.【F:src/calibrated_explanations/plugins/builtins.py†L336-L348】
* Containers expose cached predictions, probability cubes, percentile metadata,
  thresholds, ignored-feature state, and helper handles used by telemetry,
  plotting, and JSON export. Plugins that produce alternative containers must
  populate equivalent attributes so downstream utilities continue to operate
  without change.【F:src/calibrated_explanations/explanations/explanations.py†L24-L249】

## Consequences

Positive:

* Contributors have a single reference describing what an explanation plugin is
  expected to do, which metadata it must provide, and how it coordinates with the
  interval and plot resolvers.
* Telemetry and downstream tooling remain stable because plugins still route
  through the calibrated prediction bridge and expose the same collection
  surface.
* The ADR clarifies which parts of the legacy implementation are contractual vs
  implementation details, easing future refactors (for example, streaming
  batches).

Negative / Risks:

* The ADR documents current behaviour; deviating from these semantics will
  require revisions to keep the contract authoritative.
* Legacy adapters remain complex, and the ADR points new authors to them for
  examples. Without additional helper utilities, crafting a new plugin still
  requires familiarity with the existing collections.

## Status & Follow-up

* Adopt this ADR as the canonical explanation of explanation plugin semantics.
* Link developer-facing documentation (e.g., `docs/plugins.md`) to this ADR when
  referencing explanation behaviour or requesting new plugins.
* Future enhancements (streaming batches, non-CalibratedExplanations containers)
  must either honour these semantics or include updates to this ADR that explain
  the new expectations.
