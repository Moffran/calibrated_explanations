> **Status note (2026-01-03):** Last edited 2026-01-03 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-027 — Internal FAST-Based Feature Filtering

Status: Accepted

Date: 2026-01-03

Authors: Core maintainers

Supersedes: N/A

Related: ADR-003-caching-key-and-eviction, ADR-004-parallel-backend-abstraction, ADR-006-plugin-registry-trust-model, ADR-015-explanation-plugin, ADR-020-legacy-user-api-stability, ADR-026-explanation-plugin-semantics

## Context

The explanation runtime now includes an internal “FAST-based feature filtering”
step to reduce compute for factual and alternative explanations. The flow runs
an internal FAST pass over the same batch, uses the resulting per-instance
feature weights to derive a filtered ignore set, and then executes the more
expensive explanation path with the reduced feature set. The implementation is
internal (no new public CalibratedExplainer API) and is surfaced via
configuration (`ExplainerConfig` / `ExplainerBuilder`) plus an environment
variable override. The filter logic is implemented in
`core.explain._feature_filter`, and is invoked from the execution plugin
wrapper in `plugins.builtins`.

This ADR formalizes the architectural decision, its scope, and the guardrails
for configuration, failure behavior, and observability.

## Decision

### 1. Scope and visibility

- FAST-based feature filtering is an **internal** optimization that does not
  change the public `CalibratedExplainer` API surface.
- The feature is **opt-in** and remains **experimental** in configuration
  semantics (builder and environment variable). The intent is to provide a
  controlled, explicit toggle while preserving the ability to refine behavior
  without a breaking-change burden.
- The filtering applies **only** to the factual and alternative modes. FAST
  explanations remain unchanged.

### 2. Algorithm and stability

- The current algorithm (per-instance absolute FAST weights → per-instance
  keep set → batch-level global ignore set) is the **reference implementation**.
- Future refinements must preserve the externally observable invariants:
  - No change to factual/alternative semantics beyond skipping unimportant
    features.
  - Baseline ignore set (explainer + user-provided) is always respected.
  - Per-instance masks can be attached without altering rule semantics.

### 3. Failure and fallback behavior

- The system **fails open**: if the FAST pass or filtering fails for any
  reason, the explainer proceeds without filtering.
- Observability policy:
  - Emit **telemetry events** (when telemetry is configured) for
    `filter_enabled`, `filter_skipped`, and `filter_error`.
  - Log at **debug** level by default.
  - Emit **warnings** only when an explicit “strict observability” mode is
    enabled (future config hook) to avoid noisy user-facing behavior.

### 4. Per-instance mask exposure

- Per-instance ignore masks are attached to the resulting explanation
  collection **as best-effort metadata** to support transparency and
  debugging.
- This metadata is documented for advanced users but does not constitute a
  hard public API contract.

### 5. Configuration precedence and auditability

- Configuration resolution follows explicit precedence rules, with audit
  visibility of the final effective config:
  1. Call-site configuration (builder/config object).
  2. Environment variable overrides.
- When telemetry is configured, the resolved feature-filter configuration is
  emitted as a single “effective config” event to support traceability.

## Consequences

### Positive

- Reduces compute for large feature spaces while preserving per-instance
  explanation fidelity.
- Maintains public API stability and avoids introducing new user-facing
  explain methods.
- Preserves deterministic behavior by default and allows controlled, explicit
  opt-in to the filtering path.

### Negative / Trade-offs

- Experimental configuration semantics require careful documentation to avoid
  being perceived as stable public API.
- Failure policy requires consistent telemetry wiring to avoid silent
  behavior changes in production.

### Follow-ups

- Align runtime logging with the observability policy (debug by default;
  warnings only in strict mode).
- Document metadata exposure for per-instance ignore masks and provide
  examples in performance tuning documentation.

## Out of Scope / Future Considerations

- Pluginizing feature filtering, strategy injection, or providing alternative
  built-in filtering strategies are explicitly deferred. These remain
  candidates for a future ADR.
