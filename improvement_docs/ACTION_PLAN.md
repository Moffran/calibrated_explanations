# Calibrated Explanations Improvement Plan (Contract-first)

Created: August 16, 2025
Last Updated: September 4, 2025 (post-perf scaffolding tests)
Repository: calibrated_explanations
Current Version: v0.6.0
Target Releases: v0.6.0 (Stable contracts & config), v0.7.0 (Perf foundations & docs CI), v0.8.0 (Extensibility & viz)

---

## Executive summary

Shift to contract-first delivery: freeze public data contracts (schema v1), preprocessing policy, and plugin trust model; keep performance behind feature flags; harden tests/docs. Maintain legacy output compatibility via adapters while introducing a clear internal domain model.

---

## Current repo state (observed)

Based on the v0.6.0 release and code review, the repo has completed Phase 2 (contract-first) and Phase 2S (optional extras + CI) milestones.

- Phase 1A/1B: Done
  - Validation/exception types are present at boundaries.
  - Parameter aliasing/canonicalization exists in the API layer.
- Phase 2 core: Complete in v0.6.0
  - Config primitives and wrapper wiring present; config defaults flow through wrapper paths.
  - Preprocessing and mapping persistence policy (ADR-009): user-supplied preprocessor supported in wrapper with deterministic reuse; unseen-category policy field present; tests and docs in place.
  - Explanation domain model (ADR-008): implemented with `Explanation`/`FeatureRule` and legacy adapters; golden outputs preserved.
  - Explanation JSON Schema v1 (ADR-005): frozen and shipped as `schemas/explanation_schema_v1.json`; serialization helpers with optional validation and round-trip tests added.
  - Convenience API `quick_explain(...)` added; parity tests present.
  - Deprecation warnings for parameter aliases wired at public boundaries.
- Phase 2S structural: Complete (initial scope)
  - Optional extras groups declared (`viz`, `lime`, `notebooks`, `dev`, `eval`).
  - Lazy plotting import; tests marked `@pytest.mark.viz` and skipped when extras are absent; CI includes a core-only job.
  - Docs build and linkcheck wired in CI; API reference and new pages added (schema v1, migration guide).
- Later phases:
  - Visualization abstraction (PlotSpec) not yet implemented (plot style/config landed separately).
  - Caching/parallel backends not yet implemented (planned behind flags).

Implication: with contracts stable in v0.6.0, prioritize visualization abstraction (PlotSpec), then performance foundations behind flags, and a minimal plugin trust baseline.

---

## Prioritization (what’s next)

1) Accept ADR-007 and ship a minimal PlotSpec with a matplotlib adapter for 1–2 plots; keep optional deps and preserve current plotting behavior by default.
2) Implement ADR-003/ADR-004 behind feature flags with micro-benchmarks; integrate with existing perf guard; disabled by default.
3) Finalize ADR-006 (plugin trust) with minimal capability metadata and a registry stub; document risks and opt-in usage.
4) Harden deprecation/migration policy (ADR-011): ensure alias warnings are consistent; expand migration guide with examples and an optional script.
5) Expand docs/gallery pipeline (ADR-012): add more example renders; maintain linkcheck and API ref gates.

Near-term (2–3 weeks): PlotSpec MVP + adapter, baseline caching/parallel behind flags with micro-benchmarks, and plugin trust minimal wiring.

---

## Phase A: Validation, Exceptions, and Parameter Canonicalization

Status: Done/In progress (minor gaps)

- Verify ADR-002 is Accepted and applied across public entry points.
- Ensure `core.exceptions` are consistently raised; replace any remaining ad-hoc exceptions while preserving message compatibility.
- Confirm `api/params` canonicalization is used end-to-end (builder, wrapper, CLI/examples); add any missing combination checks.

Acceptance

- Tests cover error typing and messages at boundaries.
- Error types consistent across main flows.
- No change to successful behavior or serialized outputs.

---

## Phase B: Explanation Schema v1 and Domain Model

Status: Done in v0.6.0

- ADR-005 Accepted; schema v1 frozen and shipped; serialization helpers and round-trip tests added.
- ADR-008 Implemented; internal domain model with adapters preserves legacy dict outputs and golden tests.

Acceptance

- Schema v1 round-trip passes on reference fixtures. [Done]
- Legacy public outputs preserved via adapter where required. [Done]

---

## Phase C: Visualization Abstraction (PlotSpec)

Status: In progress (MVP landed)

- Move ADR-007 to Accepted.
- Introduce a minimal, backend-agnostic PlotSpec and convert 1–2 existing plots.
- Keep matplotlib adapter as default example; ensure import without heavy optional deps remains error-free.

Acceptance

- PlotSpec renders via the matplotlib adapter for selected examples. [Done: regression bars]
- Public API docs updated with PlotSpec usage examples. [Done: docs/viz_plotspec.md, linked in index]

---

## Phase D: Preprocessing and Mapping Persistence

Status: Done in v0.6.0 (initial scope)

- ADR-009 Accepted; wrapper supports user-supplied preprocessor with deterministic mapping reuse and unseen-category policy field (default conservative behavior). Tests and docs added.
- Cookbook examples and DataFrame tests present; numeric-only paths preserved.

Acceptance

- Deterministic mappings across fit/predict; tests cover unseen categories behavior. [Done]

---

## Phase E: Performance Foundations (Caching & Parallel)

Status: Started (scaffolding added; unit tests passing)

- Implement baseline ADR-003 (Caching) and ADR-004 (Parallel Backend) behind feature flags.
- Provide a small LRU cache and a thin parallel map abstraction with sensible defaults. [Done: `perf.cache.LRUCache`, `perf.parallel.JoblibBackend`]
- Export surface finalized for this phase (`make_key`, `LRUCache`, `JoblibBackend`, `sequential_map`); tests import from `calibrated_explanations.perf` and pass. [Done]
- Add micro-benchmarks and integrate with existing perf-guard job. [Done: `scripts/micro_bench_perf.py`, `scripts/check_perf_micro.py`; CI wired in `.github/workflows/test.yml` and baseline updater simplified]

Acceptance

- Feature-flagged; disabled by default; no behavior change when off. [Maintained]

Notes

- New tests: `tests/test_perf_foundations.py` validate cache eviction, deterministic keys, and the joblib backend fallback; green locally.
- Micro benchmark script prints JSON with import time and sequential vs joblib timings for future CI perf guard.

---

## Phase F: Testing & Documentation Hardening

Status: In progress (docs pipeline active)

- Expand tests: property-based checks for calibration invariants; edge cases; light performance smoke tests.
- Documentation updates:
  - API reference generated; strengthen docstring coverage.
  - nbconvert/sphinx-gallery examples rendered to HTML (expand coverage).
  - Linkcheck active in CI.

Acceptance

- Tests remain green; doc build & linkcheck pass in CI. [Docs CI: Done]

---

## Phase G: Deprecation & Migration Policy Activation

Status: In progress

- Deprecation warnings for alias keys are active at public boundaries.
- Migration guide (0.5.x → 0.6.0) drafted; extend with code rewrite helper.

Acceptance

- Deprecation warnings appear once; migration guide pages updated with concrete examples; optional script available.

---

## ADR changes and priorities

Must finalize (move to Accepted; implement)

- ADR-002 Validation & Exception Design (verify coverage; minor tidy-ups)
- ADR-003 Caching Key & Eviction Strategy (baseline, feature-flagged)
- ADR-004 Parallel Backend Abstraction (baseline, feature-flagged)
- ADR-005 Explanation JSON Schema v1 (Accepted; implemented)
- ADR-006 Plugin Registry Trust Model (minimal metadata; registry wiring)
- ADR-007 Visualization Abstraction (PlotSpec v1)
- ADR-008 Explanation Domain Model & Legacy Compatibility (Implemented)
- ADR-009 Input Preprocessing & Mapping Persistence Policy (Accepted; implemented)

New ADRs to add (public, process-focused)

- ADR-011: Deprecation & Migration Policy
  - Scope: warning semantics, removal timelines, API snapshot tooling, migration guide structure.
- ADR-012: Documentation & Gallery Build Policy
  - Scope: doc build gates (API ref, linkcheck), example rendering pipeline, contribution guidelines for examples.

Note: ADR-010 is already used for Core vs Evaluation split in this repo; numbering is adjusted accordingly.

Prioritization (sequence aligned to current state)

1) ADR-007 (PlotSpec MVP) and docs/gallery enrichment (ADR-012)
2) ADR-003, ADR-004 (performance enablers behind flags)
3) ADR-006 (plugin trust minimal)
4) ADR-011 (deprecation governance)

---

## Issue templates (suggested)

- Adopt Validation & Exceptions at Public Boundaries
  - Replace any remaining ad-hoc exceptions; add tests; no behavior change.
- Freeze Explanation Schema v1 and Round-Trip Tests
  - Finalize fields; add domain model; adapters for legacy dict; round-trip tests; document usage.
- Preprocessing Mapping Persistence (Categorical/Text)
  - Finalize hooks; deterministic mappings; unseen category policy; tests & docs.
- Plugin Registry Trust Model (Minimal)
  - Finalize ADR-006; document capability metadata; smoke tests for registry import.
- PlotSpec MVP and Adapter Conversion
  - Implement PlotSpec; convert two plots; update examples.
- Baseline Caching & Parallel (Feature-Flagged)
  - Implement minimal cache and parallel map; add micro-benchmarks; disabled by default.
- Docs & Gallery build in CI
  - API ref build, example HTML rendering, linkcheck; contribution guide.
- Deprecation Policy ADR and Migration Notes
  - Add ADR-011; wire warnings; write migration page.

---

## Quality gates (public repo)


- Lint/style: ruff clean (no new ignores); markdownlint for docs.
- Types: mypy permissive initially; stricter per-module as code stabilizes.
- Tests: unit + targeted property tests; keep CI under reasonable time budgets.
- Docs: build + linkcheck must pass; examples render to HTML.

---

## Communication

- Use CHANGELOG fragments (e.g., towncrier) for incremental releases.
- Reference ADRs in PR descriptions and keep a short “status” table in the action plan.
- Post periodic progress summaries (what changed, tests/docs added, any contract decisions).

---

## Near-term focus (next 2–3 weeks)

- Freeze ADR-005 (schema v1) and ADR-009 (preprocessing policy) and implement minimal domain model + adapters.
- Land docs CI (API ref + nbconvert + linkcheck) to make examples first-class.
- Publish issues above with checklists and label priorities.


- `ExplainerConfig` dataclass (model, calibration params, thresholds, parallel settings).
- Builder (`ExplainerBuilder`) returns configured wrapper; ensures validation pre-fit.
- High-level convenience `quick_explain(...)`.
- Parameter alias resolution early (e.g., `alpha → low_high_percentiles`).
- Add deprecation warnings for old parameter names (scheduled removal v0.8.0).
- Auto-migration script: scans notebooks/scripts and rewrites deprecated args optionally.

- Reported Issue (Explanations storage redesign):
  - Introduce an internal Explanation domain model (e.g., `explanations/models.py`) with `Explanation` and `FeatureRule` types, structuring rules as an explicit list. Maintain the public legacy dict shape via adapters to preserve golden outputs and API behavior. This resolves the “mixed singleton vs. list” storage complexity and simplifies filtering/iteration.
  - Add iteration/filter helpers that operate on the new model; public APIs continue to emit legacy dicts until schema v1 is adopted (Phase 5).

- Reported Issue (Native non-numeric input support):
  - In `wrap_explainer.py`, add `auto_encode=True|False|'auto'` and support a user-supplied `preprocessor` (e.g., ColumnTransformer/Pipeline). Default behavior preserves existing numeric path; when enabled, apply `utils.helper.transform_to_numeric` or user transformer. Persist mappings and attach to provenance for reproducibility.
  - Extend `ExplainerConfig` with `preprocessor`, `auto_encode`, and unseen-category policy. Ensure predict/online paths reuse the same mapping deterministically.

**Deliverables:** Updated docs with old vs new side-by-side, migration guide draft (v0.5.x → v0.6.0), CLI snippet for migration script, increased docstring coverage.

Additional Deliverables (Phase 2, tied to reported issues):

- Internal domain model implemented with adapter parity tests against golden fixtures; no public change yet.
- Preprocessing path integrated in wrapper with round-trip tests on categorical/text DataFrames; numeric-only paths remain unchanged.

### Phase 2 status (2025-09-02)

Done:

- Configuration primitives `ExplainerConfig`, `ExplainerBuilder` added.
- Private `WrapCalibratedExplainer._from_config` and propagation of config defaults to common methods.
- Controlled user-supplied preprocessor wiring (fit/transform) in `wrap_explainer.py`; helpers `_pre_fit_preprocess`, `_pre_transform`; inference path reuse; tests added and passing.
- Docs updated (getting_started + API reference) describing config-driven preprocessing and defaults.

Remaining:

- Internal Explanation domain model + adapters preserving legacy dict shape; add parity tests vs golden fixtures.
- Wire alias deprecation warnings at public boundaries (use `warn_on_aliases`); keep behavior unchanged; document deprecation policy.
- Add `quick_explain(...)` convenience.
- Draft migration guide (v0.5.x → v0.6.0) with examples; optional notebook/script auto-rewrite helper.
- Optional: expose a public `from_config` constructor once stabilized (today’s `_from_config` stays private).

Exit criteria update:

- Replace “planned” with “complete” for config + preprocessor wiring + docs; keep domain model and migration items open.

---

## 8.1 Phase 2S: Packaging Split (Core vs Evaluation) & Optional Extras (Week 6)

Purpose: streamline the core package and make evaluation/research assets clearly optional, improving install footprint and comprehension.

Scope:

- Adopt ADR-010. No runtime behavior changes besides optional visualization gating.
- Add optional dependency groups in `pyproject.toml`:
  - `viz`: plotting stack (matplotlib and related);
  - `notebooks`: IPython/Jupyter authoring;
  - `dev`: lint/test/type tools plus `viz`;
  - `eval`: dependencies required by `evaluation/` scripts.
- Make plotting imports lazy in library code; when missing, raise a clear error instructing to `pip install calibrated-explanations[viz]`.
- Tag plotting tests with `@pytest.mark.viz` and skip if `viz` extras are unavailable.
- Add brief `evaluation/README.md` clarifying scope and how to set up the eval environment.

Acceptance criteria:

1. `pyproject.toml` updated with `[project.optional-dependencies]` groups. [Done: `viz`, `lime`, `notebooks`, `dev`, `eval`]
2. README installation section documents extras and examples. [Done]
3. Plotting code path performs lazy import; missing-backend error message covered by tests. [Done]
4. Tests marked and conditionally skipped without `viz`. CI matrix includes one job without `viz` to ensure core remains independent. [Done]
5. ADR-010 status updated to Accepted (initial scope) with adoption progress. [Done]

Notes:

- This sub-phase complements ADR-007 (Visualization Abstraction) by decoupling dependencies now, ahead of introducing a PlotSpec later.

### Phase 2S status (2025-09-04)

Done:

- Optional-deps groups present (`viz`, `lime`, `notebooks`, `dev`, `eval`).
- Lazy plotting import with actionable error suggesting `pip install "calibrated_explanations[viz]"`.
- README and getting started mention extras.
- Viz tests marked and skipped without extras; CI includes a core-only job.
- Docs CI (HTML + linkcheck) active.

Notes:

- Heavy viz deps remain in core install for compatibility; plan decoupling in v0.7.0 once downstream impact is evaluated.

---

## 9. Phase 3: Performance (Weeks 7-10)

**Focus Areas:** Caching, parallelism abstraction, memory management, lazy compute.

**Design Decisions (ADRs):**

- Parallel backend interface (`ParallelBackend`): default `joblib`, future pluggable (Ray/Dask).
- Cache policy: LRU keyed by deterministic hash (data bytes + params) with size accounting; allow user-supplied cache store (in-memory / disk).
- Memory Manager: pre-operation estimate + soft cap warnings; optional chunked iteration for large arrays.
- Lazy wrapper `LazyExplanation` returning materialized structure on demand.

**Instrumentation:** Collect before/after benchmark deltas; fail CI if regression beyond thresholds.

**Deliverables:** Cache module + tests (hit ratio test), parallel interface, memory warnings, updated benchmarks doc, CI perf guard integration, p95 latency improvement target (≥30% vs baseline where achievable).

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
- `plugins/__init__.py` exports protocol and registry; tests added (`tests/test_plugin_registry.py`).
- Security note: warn users about third-party code execution; optional whitelist config.

**Explanation Data Schema:**

- Versioned JSON spec (`schemas/explanation_schema_v1.json`) capturing: inputs, model metadata, calibration params, feature attributions, uncertainty intervals, provenance (library versions, timestamp, git commit).
- Serialization utilities (`serialization.py`) + round-trip tests.
- Align the schema with the internal domain model introduced in Phase 2 by representing feature rules as `rules: []`. Provide a migration tool from legacy dicts. This directly addresses the earlier reported issue about explanation storage structure.

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

## 13. Cross-Cutting Quality Gates (Scoped)

| Category | Phase 0–2 (Minimal) | Later Target (Phase 3–4) |
|----------|---------------------|--------------------------|
| Lint/Style | Ruff clean | No new ignores; stable formatting |
| Types | (Deferred) | `core/` & new API mypy-clean by v0.7.0 |
| Tests | Existing suite green | >90% coverage + property tests |
| Performance | Import + micro runtime guard | Broader p95 + memory dashboards |
| Memory | Basic RSS snapshot | ≤10% regression guard by Phase 3 |
| Docs | Build succeeds | Linkcheck + docstring ≥85% |
| Deprecations | Warnings present | Migration guide finalized Phase 4 |
| Security | Manual review | bandit + audit integrated Phase 3 |

Justification: reduce early overhead; delay heavier gates until after mechanical refactors stabilize.

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


- v0.6.0: Phases 0–2 (+2S) complete (released 2025-09-04), deprecations active, migration guide draft, no behavior changes to serialized outputs.
- v0.7.0: Phases 3–4; performance boosts (feature-flagged), PlotSpec MVP, finalized migration, stable schema v1.
- v0.8.0: Phases 5–6; plugin ecosystem baseline, advanced visualization, potential schema v1.1 (non-breaking enhancements only).

---

## 18. Communication Plan

- Weekly CHANGELOG fragment via newsfiles (towncrier) to prevent large doc merges.
- Biweekly progress summary (metrics delta + risk updates).
- Early adopter channel for plugin API feedback after Phase 5 start.

---

## 19. Next Immediate Steps

1. Open feature branch `viz-plotspec` and implement PlotSpec MVP with matplotlib adapter; convert 1–2 plots; update docs/examples. [In progress; regression bars done]
2. Open feature branch `perf-foundations`; add baseline LRU cache and parallel map behind flags; wire micro-benchmarks and extend perf guard. [Scaffolding added; benchmarks pending]
3. Finalize ADR-006 and add `plugins/registry.py` skeleton with minimal metadata; document opt-in risks.
4. Expand migration guide with concrete alias examples and add an optional rewrite helper script.
5. Evaluate decoupling heavy viz deps from core install path in v0.7.0; document trade-offs and transition plan.

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
- ADR-008: Explanation domain model & legacy-compatibility strategy
- ADR-009: Input preprocessing & mapping persistence policy
- ADR-010: Core vs Evaluation split & distribution strategy

**Glossary:**

- Mechanical Refactor: Code relocation without semantic change.
- Golden Test: Asserts serialized output equality of canonical examples.
- PlotSpec: Backend-agnostic structure for visualization adapters.

---

This plan emphasizes measurable, low-risk progression toward a modular, performant, and extensible library while preserving user trust through explicit deprecation and robust migration support.
