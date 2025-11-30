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
| 1 | ParallelFacade (conservative chooser) missing | 4 | 3 | 12 | A small, conservative facade that centralizes selection heuristics and emits compact decision telemetry (v0.9.1 scope) has not been implemented. Implementing this facade is the recommended low-risk first step before a full strategy rollout. |
| 2 | Workload-aware auto strategy absent | 5 | 4 | 20 | `_auto_strategy` (full automatic strategy) remains unimplemented and is deferred to v0.10; the facade will collect field evidence to inform whether the full `_auto_strategy` is justified. |
| 3 | Telemetry lacks timings and utilisation metrics | 5 | 4 | 20 | Runtime timings and worker utilisation remain uninstrumented; v0.9.1 should require compact decision telemetry (decision, reason, n_instances, n_features, bytes_hint, platform, executor_type) while full per-task metrics are deferred to v0.10. |
| 4 | Context management & cancellation missing | 4 | 4 | 16 | `ParallelExecutor` lacks context manager support and cancellation APIs, limiting safe lifecycle management; cooperative cancellation is out of scope for the v0.9.1 facade and belongs to v0.10. |
| 5 | Configuration surface incomplete | 4 | 3 | 12 | `ParallelConfig` omits `task_size_hint_bytes`, `force_serial_on_failure`, and strategy injection hooks. v0.9.1 should add a small conservative config surface (min_instances_for_parallel, min_features_for_parallel, task_size_hint_bytes). |
| 6 | Resource guardrails ignore cgroup/CI limits | 4 | 3 | 12 | Worker selection relies on CPU count without container-aware caps; full cgroup/container detection is deferred to v0.10 but the facade should use conservative defaults to avoid oversubscription. |
| 7 | Fallback warnings not emitted | 4 | 2 | 8 | Serial fallbacks occur silently; the facade should emit structured decision telemetry and a warning when it forces a serial fallback (v0.9.1). |
| 8 | Testing and benchmarking coverage limited | 3 | 3 | 9 | Spawn lifecycle and throughput benchmarks remain manual or untested; v0.9.1 should add unit tests for the facade decision logic and a micro-benchmark harness to gather evidence. |
| 9 | Documentation for strategies & troubleshooting lacking | 3 | 2 | 6 | No platform-specific matrices or plugin interoperability notes accompany the rollout; v0.9.1 should ship minimal guidance for the facade and record the v0.10 plan in ADR notes. |

Alignment note: Parallel is treated as a shared service with domain-specific runtime wrappers (e.g., explain) expected to wrap
the shared `ParallelExecutor` so heuristics and chunking remain co-located with domain executors while respecting ADR-001
boundaries. ADR-004 now documents this expectation.

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
| 3 | Structured model/calibration metadata absent | 4 | 3 | 12 | Explanation dataclass lacks dedicated fields for calibration parameters and model descriptors. **STATUS 2025-11-04: Factual/alternative payload structures clarified with formal definitions in ADR-008; implementation work remains.** |
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

## ADR-012 – Documentation & Gallery Build Policy

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Notebooks never rendered in docs CI | 5 | 4 | 20 | Docs workflow skips sphinx-gallery/nbconvert execution, so notebook breakage ships undetected. |
| 2 | Docs build ignores `[viz]`/`[notebooks]` extras | 4 | 3 | 12 | CI installs bespoke requirements instead of project extras, letting dependency drift go unnoticed. |
| 3 | Example runtime ceiling unenforced | 3 | 3 | 9 | No automation times notebooks/examples, so the <30 s headless contract can regress silently. |
| 4 | Gallery tooling decision undocumented | 2 | 2 | 4 | ADR-required choice between sphinx-gallery/nbconvert is not recorded, leaving contributors without guidance. |

## ADR-013 – Interval Calibrator Plugin Strategy

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Runtime skips protocol validation for calibrators | 5 | 4 | 20 | Resolved plugins are not checked against classification/regression protocols or interval invariants before use. |
| 2 | FAST plugin returns non-protocol collections | 5 | 3 | 15 | `FastIntervalCalibratorPlugin.create` yields lists instead of protocol objects, breaking ADR guarantees. |
| 3 | Interval context remains mutable | 4 | 3 | 12 | Context builders hand plugins plain dicts, allowing mutation despite the read-only requirement. **STATUS 2025-11-04: Read-only contract documented in ADR-013 interval propagation section; enforcement remains.** |
| 4 | Legacy default plugin rebuilds calibrators | 3 | 3 | 9 | Default plugin reinstantiates calibrators rather than returning frozen instances as mandated. **STATUS 2025-11-04: Frozen instance requirement clarified in ADR-013; cache/registry integration remains.** |
| 5 | CLI interval validation commands missing | 2 | 2 | 4 | `ce.plugins explain-interval` and related commands are absent, reducing observability of runtime validation. |

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
| 4 | Predict bridge omits interval invariants | 4 | 3 | 12 | `LegacyPredictBridge` does not enforce `low ≤ predict ≤ high`, risking incorrect outputs. **STATUS 2025-11-04: Bridge contract and invariant enforcement requirements documented in ADR-026 subsections 2a/2b; implementation remains.** |
| 5 | Environment variable names diverge | 3 | 2 | 6 | Resolver expects mode-specific keys instead of `CE_EXPLANATION_PLUGIN[_FAST]` documented in the ADR. |
| 6 | Helper handles expose mutable explainer | 3 | 2 | 6 | Plugins receive direct access to the explainer instance, undermining the intended immutable context. **STATUS 2025-11-04: Immutability requirement clarified in ADR-026; enforcement remains.** |

## ADR-016 – PlotSpec Separation and Schema

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | PlotSpec dataclass lacks `kind`/`mode`/`feature_order` | 5 | 3 | 15 | Core schema from ADR-016 never landed, so adapters must guess fundamental metadata. |
| 2 | Feature indices discarded during dict conversion | 4 | 3 | 12 | Builders renumber `feature_order`, breaking parity with caller-supplied indices. |
| 3 | Validator still enforces legacy envelope | 4 | 3 | 12 | `viz.serializers.validate_plotspec` ignores ADR-016 invariants and is never invoked by builders. |
| 4 | Builders skip validation hooks | 3 | 3 | 9 | PlotSpec payloads return without structural checks, letting malformed specs leak to adapters. |
| 5 | `save_behavior` metadata unimplemented | 3 | 2 | 6 | Save hints stay in ad hoc dict manipulation instead of declared dataclass fields. |

## ADR-017 – Nomenclature Standardization

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Double-underscore fields still mutated outside legacy | 5 | 4 | 20 | Core helpers touch `__` attributes directly, contravening the ADR’s hard ban. |
| 2 | Naming guardrails lack automated enforcement | 4 | 4 | 16 | Ruff/pre-commit configuration does not fail on new snake-case or `__` violations, leaving the policy unenforced. |
| 3 | Kitchen-sink `utils/helper.py` persists | 3 | 3 | 9 | Monolithic helper resists the topic-focused split mandated for Phase 1. |
| 4 | Telemetry for lint drift missing | 3 | 3 | 9 | No runtime metrics capture naming debt, undermining the ADR’s governance plan. |
| 5 | Transitional shims remain first-class | 3 | 2 | 6 | `_legacy_explain` and similar helpers live alongside active modules instead of isolated legacy shims. |

## ADR-018 – Documentation Standardisation

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Wrapper public APIs lack full numpydoc blocks | 4 | 3 | 12 | `WrapCalibratedExplainer` methods omit Parameters/Returns/Raises despite being the stable user surface. |
| 2 | `IntervalRegressor.__init__` docstring outdated | 4 | 2 | 8 | Documented parameters no longer exist, misleading users of the calibrator. |
| 3 | `IntervalRegressor.bins` setter undocumented | 3 | 2 | 6 | Public mutator ships without a summary or type guidance. |
| 4 | Guard helpers missing summaries | 2 | 2 | 4 | `_assert_fitted`/`_assert_calibrated` lack the mandated one-line docstrings. |
| 5 | Nested combined-plot plugin classes undocumented | 2 | 2 | 4 | Dynamically returned classes expose blank docstrings, hurting plugin discoverability. |

## ADR-019 – Test Coverage Standard

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Coverage floor still enforced at 88 % | 5 | 4 | 20 | CI, local tooling, and the release checklist gate at 88 %, missing the ADR-mandated 90 % threshold. |
| 2 | Critical modules below 95 % without gates | 5 | 3 | 15 | Prediction helpers, serialization, and registry lack per-path enforcement and sit under target coverage. |
| 3 | Codecov patch gate optional | 4 | 4 | 16 | Patch status stays informational, so sub-88 % diffs can merge contrary to ADR policy. |
| 4 | Public API packages under-tested | 4 | 3 | 12 | `__init__` shims and gateway modules remain far from the guardrail, risking unnoticed regressions. |
| 5 | Exemptions lack expiry metadata | 3 | 2 | 6 | `.coveragerc` omissions omit review dates, weakening waiver governance. |

## ADR-020 – Legacy User API Stability

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Release checklist omits legacy API gate | 4 | 3 | 12 | No checkbox ensures contract/tests stay in sync before shipping. |
| 2 | Wrapper regression tests miss parity on key methods | 4 | 3 | 12 | `explain_factual`/`explore_alternatives` signatures and normalisation aren’t asserted, risking drift. |
| 3 | Contributor workflow ignores contract document | 3 | 3 | 9 | CONTRIBUTING never directs authors to update the canonical contract alongside code changes. |
| 4 | Notebook audit process undefined | 3 | 2 | 6 | ADR’s periodic notebook review step has no script or checklist entry. |

## ADR-021 – Calibrated Interval Semantics

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Interval invariants never enforced | 5 | 4 | 20 | Prediction bridges return payloads without checking `low ≤ predict ≤ high`, undermining safety guarantees. **STATUS 2025-11-04 (CRITICAL): Invariant contract clarified uniformly across three levels (prediction, feature-weight, scenario) in ADR-021 subsections 4a-4b; enforcement remains for v0.10.0.** |
| 2 | FAST explanations drop probability cubes | 4 | 3 | 12 | `explain_fast` omits `__full_probabilities__`, so ADR-promised metadata is missing for FAST runs. |
| 3 | JSON export stores live callables | 3 | 2 | 6 | `_collection_metadata` serializes `assign_threshold` functions, breaking downstream tooling expectations. |

## ADR-022 – Documentation Information Architecture (Superseded)

*Superseded by ADR-027. See ADR-027 section for active gaps.*

## ADR-023 – Matplotlib Coverage Exemption

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Visualization tests never run in CI | 5 | 4 | 20 | Workflows skip the promised `pytest --no-cov -m viz` job, so regressions go undetected. |
| 2 | Pytest ignores block viz suite entirely | 5 | 3 | 15 | `pytest.ini` `--ignore` entries prevent even manual runs, nullifying the safety valve. |
| 3 | Coverage threshold messaging inconsistent | 3 | 4 | 12 | ADR cites an 85 % floor but tooling enforces 88 %, confusing contributors and waiver reviews. |

## ADR-024 – Legacy Plot Input Contracts

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | `_plot_global` ignores `show=False` | 5 | 3 | 15 | Helper always calls `plt.show()`, violating the headless contract and breaking CI/headless runs. |
| 2 | `_plot_global` lacks save parameters | 4 | 3 | 12 | Helper cannot honour ADR-shared save semantics, leaving plots unsaveable through the documented interface. |
| 3 | Save-path concatenation drift undocumented | 2 | 2 | 4 | Helper now normalises directories, diverging from ADR guidance without updated docs. |

## ADR-025 – Legacy Plot Rendering Semantics

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Matplotlib guard allows silent skips | 4 | 3 | 12 | Helpers return early without requiring Matplotlib even when file output is requested, hiding failures. |
| 2 | Regression axis not forced symmetric | 4 | 3 | 12 | `_plot_regression` sets raw min/max limits instead of the symmetric range promised by the ADR. |
| 3 | Interval backdrop disabled | 3 | 3 | 9 | Commented-out `fill_betweenx` leaves regression interval visuals inconsistent with documented design. |
| 4 | One-sided interval warning untested | 3 | 2 | 6 | Guard exists but lacks coverage, so regressions could ship unnoticed. |

## ADR-026 – Explanation Plugin Semantics

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | `explain` method remains public | 5 | 4 | 20 | `CalibratedExplainer.explain` is exposed as a public method, but ADR-026 defines it as an internal orchestration primitive that must not be invoked directly. |
| 2 | Predict bridge skips interval invariant checks | 5 | 3 | 15 | `_PredictBridgeMonitor` never enforces `low ≤ predict ≤ high`, letting malformed intervals through. **STATUS 2025-11-04 (CRITICAL): Calibration contract and validation requirements clarified in ADR-026 subsections 2a/2b/3a/3b; enforcement remains for v0.10.0.** |
| 3 | Explanation context exposes mutable dicts | 4 | 3 | 12 | Context builder embeds plain dicts despite the frozen contract, enabling plugin-side mutation. **STATUS 2025-11-04: Frozen context requirement clarified in ADR-026 subsection 1; enforcement remains.** |
| 4 | Telemetry omits interval dependency hints | 3 | 2 | 6 | Batch telemetry drops `interval_dependencies`, reducing observability. |
| 5 | Mondrian bins left mutable in requests | 2 | 2 | 4 | `ExplanationRequest` stores caller-supplied bins verbatim, violating the immutability promise. |

## ADR-027 – Documentation Standard (Audience Hubs)

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Audience-based navigation structure not implemented | 5 | 4 | 20 | The "Getting Started / Practitioner / Researcher / Contributor" hub structure (ADR-027) is not yet implemented in the Sphinx toctree. |
| 2 | PR template lacks parity review gate | 5 | 3 | 15 | Adoption Step 5 requires reviewers to confirm classification/regression parity, but no checklist enforces it. |
| 3 | “Task API comparison” reference missing | 3 | 3 | 9 | Get Started hub omits the mandated comparison link, weakening practitioner onboarding. |
| 4 | Telemetry concept page lacks substance | 4 | 2 | 8 | Flesh out telemetry concept content (required by ADR-027 advanced tracks). |
| 5 | Researcher future-work ledger absent | 3 | 2 | 6 | Researcher advanced hub lacks the promised roadmap tied to literature references. |
