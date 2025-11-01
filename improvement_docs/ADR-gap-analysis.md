# ADR Gap Analysis (Unified Severity Model)

This document consolidates the individual ADR findings into a single reference that
uses a unified two-axis severity scale. Every gap is ranked within its ADR so the
highest-impact remediation items appear first.

## Unified Severity Scales

* **Violation impact (1–5)** – How seriously the observed gap violates the ADR.
  * 5 – Directly contradicts the ADR and blocks its intended outcome.
  * 4 – Major erosion of the ADR goal; functionality works but is fragile or
    inconsistent with the decision.
  * 3 – Noticeable drift that weakens guarantees or increases medium-term risk.
  * 2 – Minor divergence, largely cosmetic or limited in blast radius.
  * 1 – Informational observation; no required action.
* **Code scope (1–5)** – Breadth of code affected by the gap.
  * 5 – Cross-cutting impact spanning multiple top-level packages or the
    majority of runtime paths.
  * 4 – Multiple modules or a critical hub within a top-level package.
  * 3 – Confined to a single package or subsystem.
  * 2 – A couple of modules or helper utilities.
  * 1 – Single file or narrowly scoped helper.

**Unified severity** is the product of the two scores. Entries are sorted in
descending order of this product within each ADR.

---

## ADR-001 – Package and Boundary Layout

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Calibration layer remains embedded in `core` | 5 | 4 | 20 | Calibration helpers (`calibration_helpers.py`, `venn_abers.py`) still live under `core`, preventing the dedicated `calibration` package the ADR mandates. |
| 2 | Core imports downstream siblings directly | 5 | 4 | 20 | `core.calibrated_explainer` pulls in explanations, plugins, perf, and API helpers, violating the "no sibling cross-talk" rule. |
| 3 | Cache and parallel boundaries not split | 4 | 3 | 12 | Cache and parallel primitives continue to live under `perf/`, blocking independent evolution of the two packages. |
| 4 | Schema validation package missing | 3 | 2 | 6 | Validation helpers sit in `serialization.py` rather than a standalone `schema` package, so consumers must import serialization to access schemas. |
| 5 | Public API surface overly broad | 3 | 2 | 6 | `calibrated_explanations.__init__` lazily exports explanations, discretizers, and visualization namespaces beyond the sanctioned façade. |
| 6 | Extra top-level namespaces lack ADR coverage | 3 | 2 | 6 | Packages such as `api`, `legacy`, `perf`, and `plotting.py` persist without documented boundary rationale. |

## ADR-002 – Exception Taxonomy and Validation Contract

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Legacy `ValueError`/`RuntimeError` usage in core and plugins | 5 | 4 | 20 | Critical prediction, calibration, and plugin paths still raise generic exceptions instead of the ADR taxonomy. |
| 2 | Validation API contract not implemented | 4 | 4 | 16 | `validate_inputs` exposes a generic signature and duplicates logic, preventing reuse across wrappers and entry points. |
| 3 | Structured error payload helpers absent | 4 | 3 | 12 | Missing `validate(...)` helper and `explain_exception` utility block the promised diagnostics pipeline. |
| 4 | `validate_param_combination` is a no-op | 3 | 3 | 9 | Guardrails for parameter consistency remain unimplemented, weakening validation coverage. |
| 5 | Fit-state and alias handling inconsistent | 3 | 2 | 6 | Wrapper normalisation diverges from `canonicalize_kwargs`, risking behavioural drift. |

## ADR-003 – Caching Strategy

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Automatic invalidation & flush hooks missing | 5 | 4 | 20 | Cache versions do not track package revisions, and operators lack a manual flush API. |
| 2 | Required artefacts not cached | 4 | 4 | 16 | Only `_predict` leverages the cache; calibration summaries and explanation tensors recompute each run. |
| 3 | Governance & documentation (STRATEGY_REV) absent | 4 | 3 | 12 | Release checklist lacks strategy revision tracking and public rollout collateral. |
| 4 | Telemetry integration incomplete | 3 | 3 | 9 | Hit/miss counters are not surfaced through runtime telemetry, reducing observability. |
| 5 | Backend diverges from cachetools + pympler stack | 3 | 3 | 9 | Custom LRU and sizing heuristics differ from the mandated library combination. |

## ADR-004 – Parallel Execution Framework

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Workload-aware auto strategy absent | 5 | 4 | 20 | `_auto_strategy` ignores task hints and payload size, preventing the promised intelligent backend selection. |
| 2 | Telemetry lacks timings and utilisation metrics | 5 | 4 | 20 | Metrics record counts only, undermining ADR observability commitments and cache telemetry integration. |
| 3 | Context management & cancellation missing | 4 | 4 | 16 | `ParallelExecutor` lacks context manager support and cancellation APIs, limiting safe lifecycle management. |
| 4 | Configuration surface incomplete | 4 | 3 | 12 | `ParallelConfig` omits `task_size_hint_bytes`, `force_serial_on_failure`, and strategy injection hooks. |
| 5 | Resource guardrails ignore cgroup/CI limits | 4 | 3 | 12 | Worker selection relies on CPU count without container-aware caps. |
| 6 | Fallback warnings not emitted | 4 | 2 | 8 | Serial fallbacks occur silently, contrary to ADR guidance for structured warnings. |
| 7 | Testing and benchmarking coverage limited | 3 | 3 | 9 | Spawn lifecycle and throughput benchmarks remain manual or untested. |
| 8 | Documentation for strategies & troubleshooting lacking | 3 | 2 | 6 | No platform-specific matrices or plugin interoperability notes accompany the rollout. |

## ADR-005 – Explanation Envelope & Schema

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | ADR-compliant envelope absent | 5 | 4 | 20 | Serializer emits flat payloads without `type`, `generator`, `meta`, or nested `payload` blocks. |
| 2 | Enumerated type registry missing | 5 | 3 | 15 | No discriminant `type` field or per-type schema files exist. |
| 3 | Generator provenance (`parameters_hash`) missing | 4 | 3 | 12 | Exports lack provenance metadata required for reproducibility. |
| 4 | Validation helper misaligned | 4 | 3 | 12 | `validate_payload` skips semantic checks and does not enforce the envelope contract. |
| 5 | Schema version optional | 3 | 3 | 9 | Callers can omit `schema_version`, weakening compatibility guarantees. |
| 6 | Documentation & fixtures out of date | 3 | 2 | 6 | Docs and tests still describe the legacy flat schema, impeding migration. |

## ADR-006 – Plugin Trust Model

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Trust flag from third-party metadata auto-enables plugins | 4 | 4 | 16 | Registry honours `trusted=True` from packages without operator approval, bypassing opt-in controls. |
| 2 | Deny list not enforced during discovery | 3 | 3 | 9 | `CE_DENY_PLUGIN` is ignored when loading entry points, allowing denied plugins to register. |
| 3 | Untrusted entry-point metadata unavailable for diagnostics | 3 | 2 | 6 | Skipped plugins leave no audit trail, limiting operator visibility. |
| 4 | “No sandbox” warning undocumented | 2 | 2 | 4 | User-facing docs omit the ADR-required reminder that plugins execute without isolation. |

## ADR-007 – PlotSpec Abstraction

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | PlotSpec schema lacks kind/encoding/version fields | 5 | 3 | 15 | Dataclass cannot represent ADR-required structure, blocking new plot families. |
| 2 | Backend dispatcher & registry missing | 4 | 3 | 12 | Rendering hard-codes the matplotlib adapter with no extensible registry. |
| 3 | Plugin extensibility hooks absent | 4 | 3 | 12 | Plugins cannot register kinds/default renderers as ADR envisioned. |
| 4 | Kind-aware validation incomplete | 3 | 3 | 9 | `validate_plotspec` only understands bar bodies, leaving other kinds unchecked. |
| 5 | JSON round-trip inconsistent for non-bar plots | 3 | 2 | 6 | Triangular/global builders emit dicts outside the versioned serializer contract. |
| 6 | Headless export support missing | 2 | 2 | 4 | Adapter lacks byte-returning interfaces for remote rendering use cases. |

## ADR-008 – Explanation Domain Model

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Domain model not authoritative source | 5 | 4 | 20 | Core workflows still operate on legacy dicts, with domain objects produced only at serialization boundaries. |
| 2 | Legacy-to-domain round-trip fails for conjunctive rules | 4 | 3 | 12 | `domain_to_legacy` casts features to scalars, breaking conjunction support. |
| 3 | Structured model/calibration metadata absent | 4 | 3 | 12 | Explanation dataclass lacks dedicated fields for calibration parameters and model descriptors. |
| 4 | Golden fixture parity tests missing | 3 | 2 | 6 | Absence of byte-level fixtures weakens regression detection for adapters. |
| 5 | `_safe_pick` silently duplicates data | 3 | 2 | 6 | Interval helper duplicates endpoints rather than flagging inconsistencies, risking misreported uncertainty. |

## ADR-009 – Preprocessing Pipeline

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Automatic encoding pathway unimplemented | 5 | 4 | 20 | `auto_encode='auto'` has no effect; built-in encoder never runs. |
| 2 | Unseen-category policy ignored | 4 | 3 | 12 | Configuration captures the flag but no enforcement occurs during inference. |
| 3 | DataFrame/dtype validation incomplete | 3 | 3 | 9 | Validators coerce to NumPy without inspecting categorical columns, missing ADR-required diagnostics. |
| 4 | Telemetry docs mismatch emitted fields | 2 | 2 | 4 | Documentation references `identifier` while runtime payload exposes `transformer_id`. |

## ADR-010 – Optional Dependency Split

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Core dependency list still heavy | 5 | 4 | 20 | Mandatory dependencies include `ipython`, `lime`, and `matplotlib`, contradicting the lean-core mandate. |
| 2 | Evaluation extra incomplete | 4 | 3 | 12 | `[eval]` extra omits packages (`xgboost`, `lime`) required by evaluation scripts and docs. |
| 3 | Visualization tests not auto-skipped without extras | 4 | 3 | 12 | `pytest.mark.viz` cases run regardless of matplotlib availability, causing failures on core installs. |
| 4 | Evaluation environment lockfile missing | 3 | 2 | 6 | No `environment.yml`/requirements file accompanies the evaluation README. |
| 5 | Extras documentation inaccurate | 3 | 2 | 6 | README and researcher docs promise packages not actually bundled in extras. |
| 6 | Contributor guidance on extras absent | 2 | 2 | 4 | CONTRIBUTING lacks instructions for working with lean core vs. extras. |

## ADR-011 – Deprecation Policy

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Central `deprecate()` helper missing | 5 | 3 | 15 | Public APIs call `warnings.warn` directly, so deprecation messages repeat and lack keying. |
| 2 | Migration guide absent | 5 | 3 | 15 | CHANGELOG references a guide that has not been authored, leaving adopters without instructions. |
| 3 | Release plan lacks status table | 4 | 3 | 12 | `RELEASE_PLAN_v1.md` has no structured tracking for deprecation windows. |
| 4 | CI gates for deprecation policy missing | 4 | 3 | 12 | No automated enforcement of the "two minor releases" window or migration-note presence. |

## ADR-014 – Visualization Plugin Architecture

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Legacy fallback builder/renderer inert | 5 | 3 | 15 | Legacy style fallback returns empty results, breaking guaranteed `.plot()` behaviour. |
| 2 | Helper base classes (`viz/plugins.py`) missing | 4 | 3 | 12 | Third-party authors lack shared lifecycle/validation helpers. |
| 3 | Metadata lacks `default_renderer` | 4 | 3 | 12 | Registry cannot chain builders to preferred renderers. |
| 4 | Renderer override resolution incomplete | 4 | 3 | 12 | Environment variables and explicit overrides are ignored when selecting renderers. |
| 5 | Dedicated `PlotPluginError` absent | 3 | 2 | 6 | Failures bubble up as generic configuration errors instead of plot-specific diagnostics. |
| 6 | Default renderer skips `validate_plotspec` | 3 | 2 | 6 | Invalid specs can pass through unchecked before rendering. |
| 7 | CLI helpers not implemented | 3 | 2 | 6 | Required commands (`ce.plugins list --plots`, `validate-plot`, `set-default`) are missing. |
| 8 | Documentation for plot plugins lacking | 2 | 2 | 4 | No authoring guide or migration notes accompany the new plugin system. |

## ADR-015 – Explanation Plugin Integration

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | In-tree FAST plugin missing | 5 | 3 | 15 | Default `explain_fast` path fails without external plugins, contradicting ADR defaults. |
| 2 | Collection reconstruction bypassed | 4 | 3 | 12 | `CalibratedExplanations.from_batch` returns plugin-provided containers instead of rebuilding collections with metadata. |
| 3 | Trust enforcement during resolution lax | 4 | 3 | 12 | Resolver activates untrusted plugins unless the operator intervenes, bypassing ADR safeguards. |
| 4 | Predict bridge omits interval invariants | 4 | 3 | 12 | `LegacyPredictBridge` does not enforce `low ≤ predict ≤ high`, risking incorrect outputs. |
| 5 | Environment variable names diverge | 3 | 2 | 6 | Resolver expects mode-specific keys instead of `CE_EXPLANATION_PLUGIN[_FAST]` documented in the ADR. |
| 6 | Helper handles expose mutable explainer | 3 | 2 | 6 | Plugins receive direct access to the explainer instance, undermining the intended immutable context. |

