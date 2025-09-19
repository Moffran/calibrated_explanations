# Development Plan and Release Strategy

Created: September 19, 2025
Owners: Core maintainers
Scope: High‑level roadmap focused on releases and ADR alignment (concise)

## Release Strategy

- v0.6.x (stabilization)
  - Maintain contract‑first guarantees (Schema v1, legacy output compatibility).
  - Land non‑breaking groundwork behind opt‑in flags: PlotSpec MVP, perf foundations, minimal plugin registry.

- v0.7.0 (performance + docs + PlotSpec routing)
  - Enable feature‑flagged caching/parallel in selected internal paths with guardrails and micro‑bench checks (ADR‑003/ADR‑004).
  - Publish PlotSpec usage; introduce opt‑in routing from `Explanation.plot()` via the matplotlib adapter (ADR‑014).
  - Expand docs (API, gallery) and CI linkcheck.

- v0.8.0 (extensibility + viz by default)
  - Finalize plugin paths: interval calibrator plugins and explanation plugins (ADR‑013/ADR‑015).
  - Route a subset of plots through PlotSpec by default; keep legacy plotting path available (ADR‑014).
  - Document plugin trust model and example plugins.

## Release Gates (readiness)

- v0.7.0 gates
  - Micro‑bench checker wired with thresholds; caching/parallel demonstrably opt‑in with example.
  - PlotSpec opt‑in routing available with one classification/probabilistic plot converted and parity tests.
  - Docs refreshed and linkcheck green.

- v0.8.0 gates
  - Interval calibrator protocols and registry helpers shipped; example interval plugin passes contract tests.
  - Explanation plugin contract and finalize/predict bridges shipped; example explanation plugin passes integration tests.
  - PlotSpec default for selected plots; legacy path remains supported.

## Near‑Term Milestones (6–8 weeks)

- Target v0.7.0 scope: PlotSpec opt‑in routing (one plot), perf example + CI guard, docs refresh.
  - Unify style config mapping between legacy plots and adapter.
  - Add a short “Using caching/parallel safely” docs section.

## Risk and Compatibility

- Backward compatibility: keep legacy outputs and plotting path unchanged by default until v0.9.
- Security/trust: plugin discovery is explicit and in‑process (ADR‑006); ship clear risk documentation and “trusted only” helpers.
