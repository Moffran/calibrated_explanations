---
applyTo: '**'
---
# Calibrated Explanations Improvement Plan (Contract-first)

Created: August 16, 2025
Last Updated: September 3, 2025
Repository: calibrated_explanations
Current Version: v0.5.1
Target Releases: v0.6.0 (Stable contracts & config), v0.7.0 (Perf foundations & docs CI), v0.8.0 (Extensibility & viz)

---

## Executive summary

Shift to contract-first delivery: freeze public data contracts (schema v1), preprocessing policy, and plugin trust model; keep performance behind feature flags; harden tests/docs. Maintain legacy output compatibility via adapters while introducing a clear internal domain model.

---

## Current repo state (observed)

Based on public plan and code review, the repo is mid-Phase 2 with some 2S structural work landed.

- Phase 1A/1B: Done
  - Validation/exception types are present at boundaries.
  - Parameter aliasing/canonicalization exists in the API layer.
- Phase 2 core: In progress
  - Config normalization and wrapper improvements present; coverage expanding.
  - Preprocessing and mapping persistence policy (ADR-009) has hooks; needs tests/docs hardening.
  - Domain model for explanations (ADR-008) not yet implemented; legacy dict outputs dominate.
  - Explanation JSON Schema v1 (ADR-005) not yet frozen/implemented.
- Phase 2S structural:
  - Plugin registry and capability metadata exist minimally; trust model ADR pending finalization.
- Later phases:
  - Visualization abstraction (PlotSpec) not yet implemented.
  - Caching/parallel backends not yet implemented (planned behind flags).
  - Docs pipeline (API ref, example HTML, linkcheck) not yet wired in CI.

Implication: prioritize contract-freezing steps (schema v1, domain model, preprocessing policy), then visualization MVP, while keeping performance features behind flags.

---

## Prioritization (what’s next)

1) Freeze ADR-005 (schema v1) and ADR-009 (preprocessing policy), finalize ADR-006 (plugin trust) and adopt.
2) Implement ADR-008 (domain model) with legacy adapter to preserve current outputs.
3) Accept ADR-007 and ship a minimal PlotSpec with a matplotlib adapter for 1–2 plots; keep optional deps.
4) Implement ADR-003/ADR-004 behind feature flags with micro-benchmarks; disabled by default.
5) Add policy ADRs for deprecation and docs/gallery CI; activate deprecation warnings for aliases.

Near-term (2–3 weeks): Freeze schema v1 + preprocessing policy, land domain model + adapters, and wire docs CI (API ref + nbconvert + linkcheck).

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

Status: Not started (schema/domain), legacy outputs in use

- Move ADR-005 to Accepted; freeze v1 fields and semantics.
- Implement the internal domain model (ADR-008) and adapters to maintain legacy dict compatibility.
- Add round-trip serialization tests (domain -> JSON -> domain) and basic size/perf notes.

Acceptance

- Schema v1 round-trip passes on reference fixtures.
- Legacy public outputs preserved via adapter where required.

---

## Phase C: Visualization Abstraction (PlotSpec)

Status: Not started

- Move ADR-007 to Accepted.
- Introduce a minimal, backend-agnostic PlotSpec and convert 1–2 existing plots.
- Keep matplotlib adapter as default example; ensure import without heavy optional deps remains error-free.

Acceptance

- PlotSpec renders via the matplotlib adapter for selected examples.
- Public API docs updated with PlotSpec usage examples.

---

## Phase D: Preprocessing and Mapping Persistence

Status: In progress (hooks exist; needs tests/docs hardening)

- Move ADR-009 to Accepted (already); finalize tests and docs.
- Finalize preprocessor hooks in wrappers with deterministic mapping persistence and an unseen category policy.
- Provide cookbook examples and tests with pandas DataFrame inputs (categorical/text).

Acceptance

- Deterministic mappings across fit/predict; tests cover unseen categories behavior.

---

## Phase E: Performance Foundations (Caching & Parallel)

Status: Not started

- Implement baseline ADR-003 (Caching) and ADR-004 (Parallel Backend) behind feature flags.
- Provide a small LRU cache and a thin parallel map abstraction with sensible defaults.
- Add micro-benchmarks and perf guards where feasible (import time, p50/p95 on small datasets).

Acceptance

- Feature-flagged; disabled by default; no behavior change when off.

---

## Phase F: Testing & Documentation Hardening

Status: In progress (unit tests present; docs pipeline pending)

- Expand tests: property-based checks for calibration invariants; edge cases; light performance smoke tests.
- Documentation updates:
  - Generate API reference; strengthen docstring coverage.
  - Add sphinx-gallery or nbconvert pipeline for examples rendered as HTML.
  - Linkcheck in CI.

Acceptance

- Tests remain green; doc build & linkcheck pass in CI.

---

## Phase G: Deprecation & Migration Policy Activation

Status: Not started

- Document and enforce the public deprecation policy (two minor releases before removal).
- Provide migration notes and a simple guide for parameter alias changes; optional script.

Acceptance

- Deprecation warnings appear once; migration guide pages updated.

---

## ADR changes and priorities

Must finalize (move to Accepted; implement)

- ADR-002 Validation & Exception Design (verify coverage; minor tidy-ups)
- ADR-003 Caching Key & Eviction Strategy (baseline, feature-flagged)
- ADR-004 Parallel Backend Abstraction (baseline, feature-flagged)
- ADR-005 Explanation JSON Schema v1 (freeze and implement)
- ADR-006 Plugin Registry Trust Model (minimal metadata; registry wiring)
- ADR-007 Visualization Abstraction (PlotSpec v1)
- ADR-008 Explanation Domain Model & Legacy Compatibility
- ADR-009 Input Preprocessing & Mapping Persistence Policy (finalize tests/docs)

New ADRs to add (public, process-focused)

- ADR-011: Deprecation & Migration Policy
  - Scope: warning semantics, removal timelines, API snapshot tooling, migration guide structure.
- ADR-012: Documentation & Gallery Build Policy
  - Scope: doc build gates (API ref, linkcheck), example rendering pipeline, contribution guidelines for examples.

Note: ADR-010 is already used for Core vs Evaluation split in this repo; numbering is adjusted accordingly.

Prioritization (sequence aligned to current state)

1) ADR-005, ADR-009, ADR-006 (freeze contracts used by users and extensions)
2) ADR-008 (implement domain model with legacy adapter)
3) ADR-007 (PlotSpec MVP) and docs pipeline (ADR-012)
4) ADR-003, ADR-004 (performance enablers behind flags)
5) ADR-011 (deprecation governance)

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

1. `pyproject.toml` updated with `[project.optional-dependencies]` groups. [Partially Done: `viz`, `lime` present; `notebooks`, `dev`, `eval` pending]
2. README installation section documents extras and examples. [Pending; getting_started updated with viz note]
3. Plotting code path performs lazy import; missing-backend error message covered by tests. [Done: lazy import + clear error; tests still require matplotlib in current matrix]
4. Tests marked and conditionally skipped without `viz`. CI matrix includes one job without `viz` to ensure core remains independent. [Pending]
5. ADR-010 status updated to Accepted (initial scope) with adoption progress. [Done]

Notes:

- This sub-phase complements ADR-007 (Visualization Abstraction) by decoupling dependencies now, ahead of introducing a PlotSpec later.

### Phase 2S status (2025-09-02)

Done/Partial:

- Added optional-deps groups `viz`, `lime` in `pyproject.toml`.
- Implemented lazy plotting import with actionable error suggesting `pip install "calibrated_explanations[viz]"`.
- Getting started mentions the viz extra.

Remaining next actions:

- Add extras: `notebooks`, `dev`, `eval` (finalize package lists).
- Update README install section with extras matrix; add `evaluation/README.md` and an eval environment file.
- Tag viz tests (e.g., `@pytest.mark.viz`) and skip when extras not installed; add CI job without viz to guarantee core independence.
- Optional: separate CI workflow for evaluation that installs `[eval]` and exercises benchmarks.

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


- v0.6.0: Phases 0-2 (+2S) complete, deprecations active, migration guide (draft), no performance degradation.
- v0.7.0: Phases 3-4; performance boosts, finalized migration, stable schema v1.
- v0.8.0: Phases 5-6; plugin ecosystem baseline, advanced visualization, potential schema v1.1 if non-breaking enhancements.

---

## 18. Communication Plan

- Weekly CHANGELOG fragment via newsfiles (towncrier) to prevent large doc merges.
- Biweekly progress summary (metrics delta + risk updates).
- Early adopter channel for plugin API feedback after Phase 5 start.

---

## 19. Next Immediate Steps

1. (Completed) Phase 0 tasks & baselines committed.
2. (Action) Open feature branch `core-split`.
3. (Action) Extraction Wave 1: Introduce `prediction_helpers.py`; relocate pure helper functions; update imports; run tests.
4. (Action) Extraction Wave 2: Introduce `calibration_helpers.py`; move interval learner initialization/update logic; run tests & golden checks.
5. (Action) Extraction Wave 3: Introduce `fast_explainer.py` & migrate `fast` path code; add delegations.
6. (Action) Add `validation_stub.py` with no-op placeholders used by `calibrated_explainer.py`.
7. (Action) Prune `calibrated_explainer.py` retaining only orchestrating methods & datamodel; ensure exports unchanged.
8. (Action) Re-run baseline metrics; store new JSON (naming convention preserved) and verify performance guard (±5% median/p95, ±0.5% RSS). Document in CHANGELOG fragment.
9. (Action) Optionally refine deprecation warning text (non-blocking).
10. (Action) Open PR referencing Phase 1A checklist; include diffstat and baseline comparison summary.
11. (Deferred Action, Phase 2/3) Draft `FastCalibratedExplainer` class design (constructor, predict/explain interface, compatibility with existing tests), then implement and swap delegations.

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
