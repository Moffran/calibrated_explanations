# Calibrated Explanations Repository Improvement Action Plan

**Created:** August 16, 2025
**Repository:** calibrated_explanations
**Current Version:** v0.5.1
**Target Releases:** v0.6.0 (Foundational Refactor & Stability), v0.7.0 (Performance & Robustness), v0.8.0 (Extensibility & Advanced UX)

---

## 0. Executive Summary

This plan focuses on: (1) introducing architectural baselining before refactors, (2) moving validation & exception work earlier to avoid duplicate fixes, (3) adding deprecation & migration strategy, (4) formalizing performance/memory baselines with regression guards, (5) tightening quality gates (lint, typing, doc coverage), and (6) defining stable plugin / visualization contracts. Refactors are split into mechanical and semantic stages to minimize risk.

---

## 1. Current State Assessment (Condensed)

**Strengths:** Strong academic basis, rich examples, packaging + tests exist.
**Pain Points:** Monolithic `core.py` (2600+ LOC), heavy pylint disables, inconsistent API, unclear validation, limited performance profiling, no formal plugin interface, scattered visualization patterns.

---

## 2. Guiding Principles

1. Behavior parity first; enhancements second.
2. Mechanical refactor isolated from logic changes.
3. Introduce safety nets (tests, benchmarks, lint, type checks) before deep changes.
4. Explicit deprecations (warnings + migration helpers) instead of silent breaks.
5. Performance & memory are tracked, not assumed.
6. Public surface area stays minimal + documented.
7. Extensibility via stable, versioned contracts (plugin, visualization, explanation data schema).

---

## 3. Architecture & Baselines (Pre-Phase Gates)

Before Phase 1 tasks begin:

- Produce high-level component diagram (Explain Flow: Training → Calibration → Explanation → Visualization).
- Extract Architecture Decision Records (ADRs) for: calibration strategy abstraction, plugin boundaries, caching layer, parallel backend choice, explanation data schema.
- Capture baseline metrics script (`scripts/collect_baseline.py`):
  - Import time of `calibrated_explanations`.
  - Median / p95 explanation time (small & medium dataset).
  - Peak RSS memory for representative tasks.
  - (Test runtime & coverage no longer part of baseline collection; retained in later CI reporting.)
- Store JSON baseline under `benchmarks/baseline_<date>.json` and commit.

---

## 4. Phase Overview (Re-ordered)

| Phase | Weeks | Focus | Key Deliverable Tag |
|-------|-------|-------|---------------------|
| 0 | 0 | Baselines, ADRs, Tooling | `baseline` |
| 1A | 1-2 | Mechanical Core Split + Deprecation Shims | `core-split` |
| 1B | 3 | Exceptions + Validation + Type Core | `stability` |
| 2 | 4-6 | API Simplification & Config Normalization | `api-simplify` |
| 3 | 7-10 | Performance (Caching, Parallel, Memory) | `perf` |
| 4 | 11-14 | Testing Expansion + Docs + Migration Guides | `robust-docs` |
| 5 | 15-18 | Plugin Architecture + Data Schema + Visualization Abstraction | `extensibility` |
| 6 | 19-20 | Advanced Visualization & Dashboards | `viz-adv` |

Release Alignment:

- v0.6.0 → End of Phase 2 (stability + simplified API).
- v0.7.0 → End of Phase 4 (performance + robust testing/docs).
- v0.8.0 → End of Phase 6 (plugins + advanced visualization).

---

## 5. Phase 0: Baselines, Tooling, Policy (Week 0)

**Goals:** Create safety net so later refactors are measurable & reversible.

**Tasks:**

- Benchmark harness: integrate `pytest-benchmark` (tag perf tests) + memory probe (tracemalloc / psutil).
- Pre-commit: `ruff` (lint + formatting), `isort` (if not using ruff-format), `mypy`, `bandit`, `pip-audit`, `codespell` (optional).
- Add `CODEOWNERS`, `CONTRIBUTING` update with quality checklist.
- Add `scripts/` utilities: baseline collector, API symbol list generator (compare drift).
- Introduce semantic version & changelog automation (e.g., `python-semantic-release` or `towncrier`).
- Decide environment management (e.g., `uv` or `pip-tools`) and produce lock file for CI reproducibility.
- Establish logging policy (structured via `logging` + optional user-provided handler).

**Deliverables:** Baseline JSON, ADRs committed, CI updated, pre-commit enforced, lock file, logging guidelines.

---

## 6. Phase 1A: Mechanical Core Decomposition (Weeks 1-2)

**Rule:** No logic or signature change; only move & re-export.

**New Structure:**

```text
src/calibrated_explanations/core/
  __init__.py
  calibrated_explainer.py
  wrapper_explainer.py
  online_explainer.py
  fast_explainer.py
  calibration.py
  prediction.py
  validation_stub.py (temporary)
  _legacy_core_shim.py
```

**Deprecation Strategy:**

- Keep `core.py` as thin shim re-exporting moved symbols; emit `DeprecationWarning` on import.
- Provide `from calibrated_explanations.core import CalibratedExplainer` unchanged for v0.6.x.

**Validation:** Snapshot test counts & benchmarks must match baseline ±5% runtime, ±0.5% memory.

**Deliverables:** Files split, tests green, import path compatibility layer, ADR updated if drift.

---

## 7. Phase 1B: Exceptions, Validation, Typing Core (Week 3)

(Moved earlier from original Week 3+5-6 to avoid duplicate edits.)

**Tasks:**

- Introduce `exceptions.py` (custom hierarchy).
- Implement `validation.py` replacing `validation_stub.py`.
- Central argument normalization helper (`api/params.py`): alias mapping + canonicalization.
- Add minimal type hints to core public APIs; enable `mypy --strict` gradually (start permissive, escalate).
- Replace generic exceptions across refactored modules.

**Deliverables:** 80%+ reduction of pylint disables tied to naming/args; typed stubs for main classes; validation unit tests; doc examples for error handling.

---

## 8. Phase 2: API Simplification & Configuration (Weeks 4-6)

**Goals:** Reduce cognitive load; consolidate configuration.

**Enhancements:**

- `ExplainerConfig` dataclass (model, calibration params, thresholds, parallel settings).
- Builder (`ExplainerBuilder`) returns configured wrapper; ensures validation pre-fit.
- High-level convenience `quick_explain(...)`.
- Parameter alias resolution early (e.g., `alpha → low_high_percentiles`).
- Add deprecation warnings for old parameter names (scheduled removal v0.8.0).
- Auto-migration script: scans notebooks/scripts and rewrites deprecated args optionally.

**Deliverables:** Updated docs with old vs new side-by-side, migration guide draft (v0.5.x → v0.6.0), CLI snippet for migration script, increased docstring coverage.

---

## 9. Phase 3: Performance (Weeks 7-10)

**Focus Areas:** Caching, parallelism abstraction, memory management, lazy compute.

**Design Decisions (ADRs):**

- Parallel backend interface (`ParallelBackend`): default `joblib`, future pluggable (Ray/Dask).
- Cache policy: LRU keyed by deterministic hash (data bytes + params) with size accounting; allow user-supplied cache store (in-memory / disk).
- Memory Manager: pre-operation estimate + soft cap warnings; optional chunked iteration for large arrays.
- Lazy wrapper `LazyExplanation` returning materialized structure on demand.

**Instrumentation:** Collect before/after benchmark deltas; fail CI if regression beyond thresholds.

**Deliverables:** Cache module + tests (hit ratio test), parallel interface, memory warnings, updated benchmarks doc, p95 latency improvement target (≥30% vs baseline where achievable).

---

## 10. Phase 4: Testing & Documentation Hardening (Weeks 11-14)

**Testing Expansion:**

- Property-based tests (Hypothesis) for calibration invariants (probabilities within [0,1], monotonic cumulative).
- Edge cases: single row, single feature, extreme values, missing values.
- Performance tests marked `@pytest.mark.perf` excluded by default, nightly in CI.
- Mutation testing (optional stretch: `mutmut` or `cosmic-ray`) for critical modules.

**Docs:**

- Sphinx: auto API from docstrings (enforce style via `pydocstyle`).
- Sphinx-gallery converting notebooks → gallery examples to prevent drift.
- Progressive migration guide updates (finalized here).
- Best Practices: memory, parallel, caching usage, error handling patterns.

**Deliverables:** >90% coverage, docstring coverage metric (≥85%), API reference generated, failing CI on broken links, performance dashboard markdown updated.

---

## 11. Phase 5: Extensibility Foundations (Weeks 15-18)

**Plugin System:**

- `plugins/base.py` abstract interface: `supports(model)`, `explain(model, X, **kwargs)`.
- `plugins/registry.py` with registration + discovery; versioned capability metadata (`plugin_meta = {"schema_version": 1, "capabilities": [..]}`).
- Security note: warn users about third-party code execution; optional whitelist config.

**Explanation Data Schema:**

- Versioned JSON spec (`schemas/explanation_schema_v1.json`) capturing: inputs, model metadata, calibration params, feature attributions, uncertainty intervals, provenance (library versions, timestamp, git commit).
- Serialization utilities (`serialization.py`) + round-trip tests.

**Visualization Abstraction:**

- Renderer-agnostic data layer: transform Explanation → canonical PlotSpec JSON; adapters for matplotlib/plotly.

**Deliverables:** Plugin examples (LIME, SHAP adaptors), schema doc, serialization tests, initial viz abstraction.

---

## 12. Phase 6: Advanced Visualization & Dashboards (Weeks 19-20)

**Features:**

- Interactive plots (plotly; optional bokeh backend).
- Exporters: HTML (self-contained), JSON (schema v1), PDF (via WeasyPrint/reportlab) – PDF optional if time.
- Prototype dashboard (Streamlit) reading Explanation JSON & rendering plots; add Jupyter widget wrapper.

**Deliverables:** Export APIs documented, dashboard example, usage tutorial, performance note (client-side payload size tracking).

---

## 13. Cross-Cutting Quality Gates

| Category | Gate |
|----------|------|
| Lint/Style | ruff passes (no new ignores), black or ruff-format enforced |
| Types | mypy incremental; by v0.7.0: no errors in `core/` & `api/` |
| Tests | Coverage >90%, critical paths property-tested |
| Performance | No p95 latency regression >10% vs previous release; improved targets logged |
| Memory | Peak memory not >10% regression; mem snapshot diff stored |
| Docs | Build passes, linkcheck clean, docstring coverage ≥85% |
| Deprecations | All have warning, timeline, migration note |
| Security | bandit & pip-audit clean (or documented justifications) |

---

## 14. Deprecation & Migration Policy

- Each deprecated symbol emits `DeprecationWarning` (once per session).
- Public removal only after 2 minor releases (e.g., introduced in v0.6.0 → removed v0.8.0).
- Migration guide sections include: Old signature, New signature, Rationale, Automated rewrite example.
- Provide `scripts/list_public_api.py` to diff exported symbols between versions.

---

## 15. Metrics & Monitoring

**Performance:** median & p95 explanation latency, throughput (explanations/sec), import time, cache hit ratio, parallel speedup scaling chart.
**Quality:** pylint score, cyclomatic complexity (radon) delta, open issues tagged `tech-debt` burndown.
**Usability:** time-to-first-explanation (measured via scripted notebook), doc page view anomalies (manual if analytics later).
**Extensibility:** plugin count, external plugin adoption (manual tracking initially).

---

## 16. Risk Management (Updated)

| Risk | Mitigation | Trigger Action |
|------|------------|----------------|
| Hidden behavior changes during split | Golden tests + baseline hash of exemplar outputs | Rollback PR; isolate diff |
| Performance regression | CI perf guard thresholds | Optimize or feature flag |
| Plugin API churn | Versioned schema + semantic version gating | Provide compatibility adapter |
| Cache memory bloat | Size accounting & eviction policy tests | Reduce default, doc tuning |
| Third-party plugin security | Warning + opt-in registry trust list | Document risk & sandbox ideas |

Rollback Checklist: revert feature branch, restore baseline JSON, issue hotfix tag.

---

## 17. Release Strategy (Adjusted)

- v0.6.0: Phases 0-2 complete, deprecations active, migration guide (draft), no performance degradation.
- v0.7.0: Phases 3-4; performance boosts, finalized migration, stable schema v1.
- v0.8.0: Phases 5-6; plugin ecosystem baseline, advanced visualization, potential schema v1.1 if non-breaking enhancements.

---

## 18. Communication Plan

- Weekly CHANGELOG fragment via newsfiles (towncrier) to prevent large doc merges.
- Biweekly progress summary (metrics delta + risk updates).
- Early adopter channel for plugin API feedback after Phase 5 start.

---

## 19. Next Immediate Steps

1. Create `baseline` branch; implement Phase 0 tasks.
2. Record and commit baseline metrics & ADRs.
3. Open tracking issues / GitHub Project board columns: Backlog, In Progress, Review, Done, Risks.
4. Begin mechanical split PR (Phase 1A) with golden test suite.

---

## 20. Appendix

**ADR Topics (Initial List):**

- ADR-001: Core decomposition boundaries
- ADR-002: Validation & exception design
- ADR-003: Caching key & eviction strategy
- ADR-004: Parallel backend abstraction
- ADR-005: Explanation JSON schema versioning
- ADR-006: Plugin registry trust model
- ADR-007: Visualization abstraction layer

**Glossary:**

- Mechanical Refactor: Code relocation without semantic change.
- Golden Test: Asserts serialized output equality of canonical examples.
- PlotSpec: Backend-agnostic structure for visualization adapters.

---

This plan emphasizes measurable, low-risk progression toward a modular, performant, and extensible library while preserving user trust through explicit deprecation and robust migration support.
