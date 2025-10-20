# ADR-026 — Explanation Plugin Semantics and Legacy Contracts

Status: Draft  
Date: 2025-10-10  
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
  `ExplanationContext`, and reuses the instance for subsequent calls. The
  resolver enforces registry metadata, per-task capability tags, and user or
  configuration overrides before the plugin is considered active.【F:src/calibrated_explanations/core/calibrated_explainer.py†L894-L1006】
* The context exposes frozen references to the task, mode, discretiser, feature
  metadata, helper handles (including the explainer itself), interval hints, and
  the calibrated `PredictBridge` monitor that guards bridge usage.【F:src/calibrated_explanations/plugins/explanations.py†L24-L56】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1008-L1040】
* Plugins **must not** mutate context attributes. Helper handles are provided
  strictly for read-only compatibility, and new plugins should treat the
  `CalibratedExplainer` reference as an escape hatch for legacy behaviour only.

### 2. Request payload and calibrated prediction usage

* Each invocation constructs an `ExplanationRequest` containing the threshold,
  percentile pair, bins, ignored feature indices, and a shallow copy of any
  extras supplied by the caller.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1081-L1101】【F:src/calibrated_explanations/plugins/explanations.py†L58-L75】
* Plugins **must** call at least one of the calibrated prediction bridge methods
  (`predict`, `predict_interval`, or `predict_proba`) for every batch. A monitor
  raises a configuration error if the bridge is never touched, ensuring interval
  semantics from ADR-013/ADR-021 are honoured even when plugins generate their
  own perturbations.【F:src/calibrated_explanations/core/calibrated_explainer.py†L138-L179】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1149-L1155】
  The bridge additionally validates that every returned triple satisfies the
  inclusive bounds invariant (`low <= predict <= high`). Any violation—whether
  detected by the bridge, the plugin, or downstream consumers of feature-level
  intervals—**must** be treated as a hard failure rather than coerced or
  truncated output. `[low, high]` pairs that do not contain their `predict`
  component are nonsensical in this system, and silently accepting them
  reintroduces the very calibration drift these ADRs forbid.
* Legacy adapters demonstrate the contract by invoking the bridge before calling
  the underlying `CalibratedExplainer` method. New implementations may ignore the
  return value but must exercise the bridge to keep telemetry and interval
  tracking consistent.【F:src/calibrated_explanations/plugins/builtins.py†L321-L335】

### 3. Expected batch structure and collection materialisation

* `explain_batch` returns an `ExplanationBatch` whose `container_cls` inherits
  from `CalibratedExplanations` and whose `explanation_cls` extends
  `CalibratedExplanation`. The runtime validates these properties immediately
  after the plugin returns.【F:src/calibrated_explanations/plugins/explanations.py†L77-L150】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1102-L1120】
* `batch.instances` is a sequence of mappings. Each mapping should contain the
  payload needed to instantiate a single explanation, even if the container later
  hydrates objects lazily. Plugins may embed additional keys, but they must not
  rely on positional ordering outside the sequence contract.
* `batch.collection_metadata` is a mutable mapping seeded by the runtime with
  task, interval source, interval dependency hints, preprocessor snapshot, plot
  fallback chain, and telemetry fields. Plugins should merge their own metadata
  (e.g., perturbation parameters, timing information) without deleting keys set
  by the runtime.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1121-L1148】
* The default materialisation path calls `container_cls.from_batch`. In-tree
  containers expect the original collection to be embedded in metadata (the
  legacy adapters simply round-trip the `CalibratedExplanations` instance). New
  containers may implement `from_batch` however they choose, but the result must
  provide the historical APIs consumed by downstream tooling.【F:src/calibrated_explanations/explanations/explanations.py†L151-L166】

### 4. Interaction with interval and plot plugins

* Explanation metadata seeds interval fallback hints. Plugins declare a primary
  interval dependency via `plugin_meta["interval_dependency"]`; the explainer
  aggregates these hints and passes them to the interval resolver so FAST and
  factual paths can share calibrators when appropriate.【F:src/calibrated_explanations/core/calibrated_explainer.py†L968-L977】【F:src/calibrated_explanations/plugins/builtins.py†L353-L404】
* The orchestrator mirrors the same pattern for plots using
  `plugin_meta["plot_dependency"]`. Returned metadata includes the resolved plot
  source and fallback chain so downstream renderers can select a compatible
  builder and renderer combination.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1043-L1138】
* Interval plugins expect calibrated predictions to flow through the bridge so
  the telemetry field `interval_source` always reflects the actual calibrator in
  use. Explanation plugins must therefore avoid short-circuiting the bridge even
  when they reuse cached collections or fast-path logic from the legacy
  explainer.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1121-L1144】

### 5. Plugin metadata and capability declarations

* Runtime validation requires capability tags `"explain"`, the mode-specific
  `"explanation:<mode>"`, and a task capability (`"task:classification"`,
  `"task:regression"`, or `"task:both"`). Plugins that omit these fields will be
  rejected during resolution. Declared modes and tasks must include the current
  explainer configuration.【F:src/calibrated_explanations/core/calibrated_explainer.py†L594-L623】【F:src/calibrated_explanations/plugins/builtins.py†L353-L420】
* Metadata may include `dependencies` for human reference, but the runtime only
  inspects `interval_dependency`, `plot_dependency`, `trusted`, and `trust` to
  enforce registry rules defined in ADR-006/ADR-013/ADR-014. Authors should keep
  these aligned with the actual calibrators and plot styles returned by their
  plugin to avoid configuration drift.

### 6. Legacy compatibility guarantees

* Built-in plugins preserve byte-for-byte outputs by delegating to
  `CalibratedExplainer` methods, disabling the plugin recursion flag, and
  wrapping the resulting collection in an `ExplanationBatch` that embeds the
  original container. New plugins targeting legacy compatibility may follow the
  same pattern until a fully plugin-native pathway is introduced.【F:src/calibrated_explanations/plugins/builtins.py†L338-L348】
* Containers expose cached predictions, probability cubes, percentile metadata,
  and thresholds used for regression probability intervals. Plugins that produce
  alternative containers must populate equivalent attributes so downstream
  utilities (serialization, telemetry, plots) continue to operate.【F:src/calibrated_explanations/explanations/explanations.py†L24-L249】

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
