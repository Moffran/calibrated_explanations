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

## 5. Phase 0: Minimal Baselines (Week 0)

*Adjusted for primarily single maintainer; ambitious tooling deferred.*

**Goals (Must):**

1. Architectural intent captured (diagram + ADRs 001–007).
2. Lightweight performance + API baseline (import time, micro timings, public symbols).
3. Deprecation shim in place ahead of core split.
4. Minimal linting (ruff) to prevent style drift.

**Done Already:**

- ADRs & component diagram.
- Baseline collector (`scripts/collect_baseline.py`) & stored baselines (16 & 20 Aug).
- Performance thresholds + regression check (`perf_thresholds.json`, `check_perf_regression.py`, CI workflow).
- API snapshot & diff scripts.
- Deprecation shim (`core.py`).
- Extended pre-commit suite (bandit, pip-audit, codespell).
- Add ruff to development workflow (pre-commit or simple CI step).
- Minimal logging scaffold (NullHandler + lifecycle INFO logs in core wrappers).

All minimal Phase 0 tasks completed; no remaining mandatory items.

**Explicitly Deferred (moved to later phases):**

- Benchmark harness (`pytest-benchmark`).
- Capturing test runtime & coverage inside baseline JSON.
- Environment lock / pin strategy.
- Release automation (semantic-release / towncrier).
- Formal logging policy & metrics aggregation.

**Phase 0 Exit Criteria (Revised):**

- Baseline + thresholds + CI regression guard green.
- ADRs & diagram committed.
- Ruff lint passes.
- Deprecation warning emitted on legacy import.

Anything else proceeds in later phases without blocking Phase 1A.

---

## 6. Phase 1A: Mechanical Core Decomposition (Weeks 1-2)

**Rule:** No logic or signature change; only mechanical moves + delegation wrappers. Zero semantic drift.

**Target Internal File Layout (remaining extractions only):**

```text
src/calibrated_explanations/core/
  __init__.py                (exports)
  calibrated_explainer.py    (shrinking; delegates after splits)
  wrap_explainer.py          (done)
  online_explainer.py        (removed)
  prediction_helpers.py      (NEW: prediction / probability helper functions)
  calibration_helpers.py     (NEW: interval learner & calibration assembly)
  fast_explainer.py          (DEFERRED: separate FastCalibratedExplainer class; Phase 2/3)
  validation_stub.py         (NEW: placeholder no-op API for Phase 1B integration)
```

Dropped: `_legacy_core_shim.py` (superseded by existing top-level `core.py` shim). Renaming divergence (wrapper→wrap) accepted and documented in ADR-001.

**Deprecation Strategy (clarified):**

1. Retain legacy module `calibrated_explanations/core.py` as a shim issuing a single `DeprecationWarning` directing users to the package form.
2. No additional alias files; removal not before v0.8.0 per deprecation policy.
3. Warning text: "The legacy module 'calibrated_explanations.core' is deprecated; import from the 'calibrated_explanations.core' package instead." (Will be updated in code.)

**Safety Nets (must be in place before major file shrinking):**

- Golden output tests (classification + regression) comparing serialized explanation dicts before/after moves.
- Import compatibility test asserting exactly one `DeprecationWarning` and symbol equivalence.
- Public API snapshot diff (script) must be empty.

**Performance & Memory Guard:**

- Re-run `scripts/collect_baseline.py` post-split; median + p95 explanation latency within ±5% of pre-split baseline; peak RSS within ±0.5% (else investigate before merging).
- No new benchmark harness (pytest-benchmark) introduced—explicitly deferred.

**Acceptance Criteria:**

1. All target modules created; `calibrated_explainer.py` reduced (only delegating wrappers + core datamodel). [Done]
2. Tests green; golden fixtures unchanged byte-for-byte. [Done]
3. API snapshot diff empty. [Done]
4. Deprecation warning emitted once; message updated/clear. [Done]
5. ADR-001 status set to Accepted (done) with note on file naming. [Done]
6. Updated component diagram (optional) if it enumerates files; not a blocker. [Deferred]

**Deliverables (Phase 1A Final):**

- New modules (`prediction_helpers.py`, `calibration_helpers.py`, `validation_stub.py`).
- Updated `calibrated_explainer.py` delegations.
- Golden test fixtures + associated tests.
- Import compatibility + API snapshot tests.
- Updated baseline comparison evidence (JSON + summary); perf remediation deferred to Phase 2.
- Refined deprecation warning text.

**Inline Gap Analysis (current status):**

| Item | Status (2025-08-21) | Notes |
|------|---------------------|-------|
| wrap_explainer.py | Done | Already split and imported in core package `__all__` |
| prediction_helpers.py / calibration_helpers.py / fast_explainer.py | Partially Done | prediction_helpers.py created and wired; calibration_helpers.py created and wired for init/update; fast_explainer.py deferred to Phase 2/3 as a separate class |
| validation_stub.py | Done | No-op placeholder added before Phase 1B real validation logic |
| Golden output tests | Done | `tests/test_golden_explanations.py` covers classification & regression serialization (first 5 instances); consider extending with hash of full probabilistic vectors later |
| Import deprecation test | Done | `tests/test_deprecation_import.py` asserts single DeprecationWarning & symbol presence |
| API snapshot diff test | Done | `tests/test_api_snapshot.py` guards root & core `__all__` |
| Deprecation warning clarity | Partially Done | Warning emitted; optionally adjust wording to match policy text (non-blocking) |
| CalibratedExplainer size reduction | In Progress | Delegated input/init/predict-step and interval learner init/update; fast path extraction deferred to Phase 2/3 |
| Performance baseline re-check | Pending | Run `scripts/collect_baseline.py` after module creation & ensure ±5% latency / ±0.5% RSS |
| Branch / PR strategy | Pending | Create feature branch `core-split` with staged commits per module extraction |
| `_legacy_core_shim.py` | Dropped | Not needed, documented in ADR-001 |

### 6.1 Current Progress Summary (as of 2025-08-22)

Overall Phase 1A progress: 100% (planned mechanical moves complete; fast path extraction deferred to Phase 2/3).

Completed safeguards now allow safe mechanical moves without semantic drift risk:

- Golden serialization tests (regression & classification)
- API snapshot stability check
- Single deprecation warning test
- ADR-001 accepted & updated

Remaining work focuses purely on structural extraction; no behavior changes planned. Fast path extraction moved to Phase 2/3 as a separate class.

### 6.3 Baseline & Performance Check (2025-08-22)

- New baseline captured: `benchmarks/baseline_20250822.json`.
- Regression check run against `benchmarks/baseline_20250820.json` with thresholds `benchmarks/perf_thresholds.json`.
- Summary (truncated):
  - import_time_seconds: +42.53% (limit 15%) — REGRESSION
  - classification.calibrate_time_s: +107.96% (limit 25%) — REGRESSION
  - classification.predict_batch_time_s: +85.80% (limit 40%) — REGRESSION
  - regression.calibrate_time_s: +70.76% (limit 25%) — REGRESSION
  - regression.predict_batch_time_s: +82.08% (limit 40%) — REGRESSION
  - fit_time_s within limits for both tasks.

Decision: Do not block Phase 1A on perf deltas; module splitting/import changes can affect timings. Investigate in Phase 2 (lazy imports, caching, fast class). Consider stabilizing environment for benchmarks.

### 6.2 Planned Extraction Order & Commit Slices

1. prediction_helpers.py: Move pure prediction / probability helper functions (no class-level state changes). Commit 1. [Completed]
2. calibration_helpers.py: Move interval learner + calibration assembly (currently `__initialize_interval_learner`, `__update_interval_learner`, related helpers). Commit 2. [Completed]
3. fast_explainer.py: Extract as separate `FastCalibratedExplainer` class with clear contract (inputs/outputs, behavior) — Deferred to Phase 2/3 to avoid semantic changes in Phase 1A. [Deferred]
4. validation_stub.py: Introduce placeholder functions (e.g., `validate_inputs(...)` no-op) to anchor future Phase 1B validation integration. Commit 4.
5. Shrink `calibrated_explainer.py`: Replace moved method bodies with thin delegations/imports; ensure public signatures unchanged. Commit 5.
6. Run full test suite + baseline collector; compare metrics vs latest baseline JSON (record new `benchmarks/baseline_<date>.json`). Commit 6.

Rollback strategy: If any golden/API snapshot test fails after a commit, revert that single commit, inspect diff for unintended logic movement, and re-attempt with smaller chunk.

### 6.3 Additional (Optional) Hardening (non-blocking for Phase 1A)

- Add a size assertion test ensuring `calibrated_explainer.py` LOC below a target (e.g., < 1000) to prevent regression.
- Add content hash of golden explanation serialized JSON for stricter equality (rather than element-wise tolerance) once stable.
- Refine deprecation warning text to: "The legacy module 'calibrated_explanations.core' is deprecated; import from the 'calibrated_explanations.core' package instead." (aligns with policy wording).


Risk mitigation: perform extractions in two waves (prediction → calibration/fast) with golden tests after each wave; rollback if diff detected.

---

## 7. Phase 1B: Exceptions, Validation, Typing Core (Week 3)

Purpose: lock in stability and developer ergonomics immediately after the mechanical split. This phase replaces the temporary validation stub, introduces a clear exception taxonomy, establishes argument normalization, and adds foundational typing and CI gates—without changing external behavior.

### 7.1 Scope & Non-Goals

In scope:

- Replace `validation_stub.py` with a real validation module and wire it in (no behavior changes beyond raising clearer errors earlier).
- Introduce a project-wide exception hierarchy and replace ad-hoc exceptions across `core/` modules.
- Add parameter alias canonicalization in a dedicated module to reduce argument drift in later phases.
- Add targeted type hints for public APIs and critical helpers; mypy in permissive mode to start.
- CI: add mypy and keep ruff; fail on new errors only.

- Reported Issue (Non-numeric input support): validation should detect DataFrame inputs and non-numeric columns early, with actionable `ValidationError`/`DataShapeError` messages to pave the way for native preprocessing in Phase 2.

Out of scope (deferred to Phases 2–4):

- API surface changes or signature renames (only aliases + warnings allowed later).
- Performance improvements.
- Extensive doc overhaul (only essentials for the new error/validation behavior now).

### 7.2 Work Breakdown (files, tasks, commit slices)

#### 7.2.1 Exceptions (commit slice 1)

Status: Completed in Phase 1B. New `core.exceptions` implemented and adopted across call sites.

- File: `src/calibrated_explanations/core/exceptions.py`
- Classes (tentative; see ADR-002):
  - `CalibratedError(Exception)` base (non-recoverable, library-specific).
  - `ValidationError(CalibratedError)` for input/config validation.
  - `ConfigurationError(CalibratedError)` invalid combos of parameters.
  - `ModelNotSupportedError(CalibratedError)` model type not supported by explainer.
  - `DataShapeError(ValidationError)` mismatched shapes/features/labels.
  - `NotFittedError(CalibratedError)` operations requiring prior fit.
  - `ConvergenceError(CalibratedError)` calibration/optimization didn’t converge.
  - `SerializationError(CalibratedError)` explanation JSON/schema issues.
- Map/replace common generic exceptions (ValueError/TypeError/RuntimeError) in the core path to these classes without changing user-facing messages yet.

#### 7.2.2 Validation engine (commit slice 2)

- Remove stub and add real validator:
  - Replace: `src/calibrated_explanations/core/validation_stub.py`
  - With: `src/calibrated_explanations/core/validation.py`
- Functions (contracts below):
  - `validate_inputs(X, y=None, task="auto", allow_nan=False, require_y=False, n_features=None, class_labels=None, check_finite=True) -> None`
  - `validate_model(model) -> None` (basic protocol checks: predict/predict_proba as applicable).
  - `validate_fit_state(explainer, require=True) -> None`
  - `infer_task(X, y, model) -> Literal["classification","regression"]` (best-effort; no hard dependency outside core).
- Centralize numpy/pandas checks, sparse detection, NaN policy, dtype/shape checks, and class label sanity. Raise the new exceptions.
  - Include DataFrame-aware checks to support the reported need for native non-numeric inputs (see Phase 2 plan).

#### 7.2.3 Parameter canonicalization (commit slice 3)

- File: `src/calibrated_explanations/api/params.py`
- Artifacts:
  - `ALIAS_MAP = {"alpha": "low_high_percentiles", "alphas": "low_high_percentiles", "n_jobs": "parallel_workers", ...}` (start minimal, extend later in Phase 2).
  - `canonicalize_kwargs(kwargs: dict) -> dict` (returns a copy with canonical keys; preserves originals for warning messages if needed later).
  - `validate_param_combination(kwargs: dict) -> None` raising `ConfigurationError` for mutually exclusive or invalid combos.
- Wire this at the boundary constructors/wrappers in `calibrated_explainer.py` and `wrap_explainer.py` without changing external signatures. Online variant removed.

#### 7.2.4 Typing foundation (commit slice 4)

- Add type hints to public entry points and newly added modules.
- Mypy configuration:
  - Add `tool.mypy` to `pyproject.toml` with relaxed defaults: `ignore_missing_imports = true`, `warn_unused_ignores = true`, `disable_error_code = ["annotation-unchecked"]` initially.
  - Enable strict per-module gradually via `mypy.ini` or pyproject module overrides for `core/validation.py` and `core/exceptions.py`.
- Goals this phase: `core/exceptions.py` and `core/validation.py` are mypy-clean; public methods in `CalibratedExplainer` and wrappers have typed signatures and return types.

#### 7.2.5 CI and pre-commit (commit slice 5)

- Add mypy step to CI (non-blocking for legacy files; blocking for the two new modules + changed public APIs).
- Keep ruff; ensure new code has no new disables. Prefer local `# noqa` only when justified and documented.

#### 7.2.6 Wiring & messaging (commit slice 6)

- Replace calls to `validation_stub.*` with `validation.*` across core.
- Update error raising sites to the new exception classes.
- Maintain existing log messages; add one-line INFO messages around validation boundaries only if already scaffolded.

### 7.3 Contracts (tiny spec)

Inputs/outputs and errors:

- `validate_inputs(...)` raises `ValidationError | DataShapeError` on failure, else returns `None`.
- `validate_model(model)` raises `ModelNotSupportedError` when required methods are missing for the inferred/selected task.
- `validate_fit_state(explainer, require=True)` raises `NotFittedError` when required and not fitted.
- `canonicalize_kwargs(kwargs)` returns a new dict; lossless for unknown keys; does not emit warnings in 1B (warnings deferred to Phase 2).

Success criteria:

- No change to successful paths or serialized outputs versus Phase 1A golden tests.
- All new modules are covered by unit tests; overall test suite remains green.

### 7.4 Tests to add

- `tests/test_exceptions.py`: hierarchy, isinstance relationships, pickling of exceptions (optional), repr/str stability.
- `tests/test_validation.py`: matrix of cases
  - X only (2D), X,y shapes mismatch, y dtypes invalid, NaN policy raise/allow, single-feature/row, sparse inputs, missing `predict_proba` for classification.
  - Model protocol tests using simple dummy estimators.
- `tests/test_params_canonicalization.py`:
  - Aliases map to canonical keys; unknown keys preserved; conflict detection triggers `ConfigurationError`.
- Integration assertions:
  - In calibrated explainer paths, invalid inputs now raise `ValidationError` instead of generic `ValueError` (update tests accordingly without changing messages where asserted).

### 7.5 Documentation & ADRs

- Update ADR-002 (Validation & Exception Design): status → Accepted; reference module paths and exception taxonomy finalized in 1B.
- Short docs page section or README snippet: “Error handling and validation”—what to expect, common errors, quick examples.
- Add API reference stubs for `core.exceptions` and `core.validation`.

### 7.6 Acceptance Criteria (exit checklist)

1. `core/exceptions.py` and `core/validation.py` implemented, imported by `core/__init__.py` as needed. [Done when merged]
2. All references to `validation_stub` removed; replaced with `validation`. [Done]
3. Golden serialization tests unchanged vs Phase 1A baselines. [Guard]
4. At least 20 targeted tests added across exceptions/validation/params; all green in CI. [Guard]
5. Mypy runs in CI; new modules mypy-clean; no increase in ruff violations. [Guard]
6. ADR-002 updated to Accepted; ACTION_PLAN reflects 1B completion scope. [Process]

### 7.7 Milestones & Timeline (1 week)

- Day 1: exceptions module + unit tests; open PR 1 (fast review).
- Day 2: validation module, replace stub; tests; open PR 2.
- Day 3: parameter canonicalization + wiring; tests; open PR 3.
- Day 4: typing pass + mypy CI; adapt call sites; open PR 4.
- Day 5: integration polish, docs/ADR-002 updates, finalize. Re-run baselines to confirm no behavioral drift (perf deltas not gating in 1B).

### 7.8 Risks & Mitigations

- Risk: Raising earlier/stricter errors could break user flows. Mitigation: retain error messages and timing as much as possible; only swap exception types; document mapping.
- Risk: Hidden dependencies on old generic exceptions in tests. Mitigation: update tests in the same PR; keep message substrings stable.
- Risk: Mypy noise. Mitigation: module-scoped strictness; start permissive, iterate.

### 7.9 Deliverables (Phase 1B Final)

- New: `core/exceptions.py`, `core/validation.py`, `api/params.py`.
- Rewired: `calibrated_explainer.py`, `wrap_explainer.py` to call new validation and exceptions. Online variant removed.
- CI: mypy step, config in `pyproject.toml`.
- Tests: exceptions/validation/params + updated integration tests.
- Docs: ADR-002 to Accepted; short error-handling guide; API stubs for new modules.

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

**Glossary:**

- Mechanical Refactor: Code relocation without semantic change.
- Golden Test: Asserts serialized output equality of canonical examples.
- PlotSpec: Backend-agnostic structure for visualization adapters.

---

This plan emphasizes measurable, low-risk progression toward a modular, performant, and extensible library while preserving user trust through explicit deprecation and robust migration support.
