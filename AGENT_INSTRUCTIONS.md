# Agent Instructions for `calibrated_explanations`

> **Canonical source of truth** for all AI agent platforms (GitHub Copilot, Codex,
> Claude Code, Google Gemini). Platform-specific files (`AGENTS.md`, `CLAUDE.md`,
> `GEMINI.md`, `.github/copilot-instructions.md`) reference this file and **only**
> add what is unique to each platform.
>
> **Update cadence:** Update this file whenever the public API changes, an ADR is
> closed, or the feedback log identifies a recurring error. Commit the update in the
> same PR as the code change.

---

## 1. CE-First Policy (mandatory for all agents)

All agent workflows **must** follow this checklist before producing any prediction
or explanation output.

1. **Library presence** – If `calibrated_explanations` is not importable, fail fast:
   ```bash
   pip install calibrated-explanations
   ```
2. **Wrapper** – Use **`WrapCalibratedExplainer`** (or a verified subclass). Never
   invent a new wrapper class.
3. **Fit** – `explainer.fit(x_proper, y_proper)` → assert `explainer.fitted is True`.
4. **Calibrate** – `explainer.calibrate(x_cal, y_cal)` → assert `explainer.calibrated is True`.
5. **Explain** – Use `explainer.explain_factual(X)` or `explainer.explore_alternatives(X)`.
    For higher security / in-distribution filtering, use `explainer.explain_guarded_factual(X)`
    or `explainer.explore_guarded_alternatives(X)` instead.
6. **Calibrated by default** – Do not return uncalibrated outputs unless explicitly
   requested.
7. **Conjunctions** – `explanations.add_conjunctions(...)` or
   `explanations[idx].add_conjunctions(...)`.
8. **Narratives & plots** – `.to_narrative(output_format=...)` and `.plot(...)`.
9. **Probabilistic regression** – `threshold=` for probabilistic intervals;
   `low_high_percentiles=` for conformal.

Helper utilities that enforce these invariants programmatically live in
`src/calibrated_explanations/ce_agent_utils.py`. Use them instead of writing
ad-hoc checks.

---

## 2. Architecture

| Package | Purpose | Rules |
|---|---|---|
| `core/` | `CalibratedExplainer`, `WrapCalibratedExplainer`, `core.exceptions` | Never import `plugins/` here (ADR-001) |
| `plugins/` | Calibrators, plotters, explanation plugins | Register via plugin registry; never hard-code in `core/` |
| `calibration/` | Venn-Abers & Conformal Prediction primitives | Stateless helpers only |
| `utils/` | `deprecate`, logging, serialization | Shared across packages |

Design patterns:
- **Lazy Loading**: `matplotlib`, `pandas`, `joblib` are imported lazily. Never
  add top-level imports of heavy libs in any module reachable from the package root.
- **Plugin-First**: New functionality goes into `plugins/`, not `core/`.
- **Exception hierarchy**: Use `core.exceptions` (ADR-002). No bare `Exception` or
  `ValueError` unless documented.

---

## 3. Coding Standards

- **Future annotations**: `from __future__ import annotations` at the top of every
  source file.
- **Docstrings**: **Numpy style** (`Parameters`, `Returns`, `Raises`, `Examples`).
- **Type hints**: Comprehensive. Avoid `Any` without a documented reason.
- **Private members**: Prefix with `_`; omit from `__all__`.
- **Circular imports**: Use `if TYPE_CHECKING:` blocks for type-only imports.
- **Deprecation**: Use `utils.deprecate` for legacy features (ADR-011).

---

## 4. Testing Standards

> Full guidance: `.github/tests-guidance.md`

- Framework: **pytest** + **pytest-mock**. No alternative frameworks.
- Naming: `should_<behavior>_when_<condition>`.
- Structure: Arrange–Act–Assert; one logical assertion block per behavior.
- Determinism: no real network, clock, or randomness; seed RNG; freeze time.
- File policy: extend the nearest existing test file first; new files require a
  "Why a new test file?" justification in the PR.
- Coverage gate: `pytest --cov=src/calibrated_explanations --cov-config=.coveragerc --cov-fail-under=90`.
- Fallback tests: tests that rely on a fallback **must** use the `enable_fallbacks`
  fixture and assert a `UserWarning` is raised.

---

## 5. Fallback Visibility Policy (mandatory)

Every fallback must be visible to users. No silent fallbacks.

- Emit `warnings.warn("<message>", UserWarning)`.
- Emit an `INFO` log entry summarising the fallback decision.
- Applies to: parallel → sequential execution, plugin fallback chains, cache backend
  fallbacks, visualization simplifications.
- When introducing a new fallback, update `CHANGELOG.md`.

---

## 6. Key Files & Directories

| Path | Purpose |
|---|---|
| `src/calibrated_explanations/core/__init__.py` | Public API surface |
| `src/calibrated_explanations/plugins/` | Plugin implementations |
| `src/calibrated_explanations/ce_agent_utils.py` | CE-first runtime guardrails for agents |
| `docs/get-started/ce_first_agent_guide.md` | Runnable CE-first guide |
| `docs/improvement/RELEASE_PLAN_v1.md` | Active release plan and milestone gates |
| `docs/improvement/adrs/` | Architectural Decision Records |
| `docs/standards/` | Engineering Standards (STD-001 through STD-005) |
| `docs/improvement/test-quality-method/` | Test-quality agent team definitions |
| `.github/tests-guidance.md` | Authoritative test guidance |
| `CHANGELOG.md` | Changelog; update under `## [Unreleased]` for every change |
| `Makefile` | Entry points: `make test`, `make ci-local` |

---

## 7. Development Workflow

```bash
# Install (editable, with dev extras)
pip install -e .[dev]

# Run tests
make test
# or
pytest --cov=src/calibrated_explanations --cov-config=.coveragerc --cov-fail-under=90

# Run full local CI (lint + type-check + tests)
make ci-local

# Pre-commit hooks
pre-commit install && pre-commit run --all-files
```

Before any implementation work:
1. Read `docs/improvement/RELEASE_PLAN_v1.md` to identify the active milestone.
2. Check `docs/improvement/adrs/` for ADRs governing the area being changed.

---

## 8. Design Patterns & TDD

**TDD workflow:**
1. **Red** – Write a failing test first (follow `.github/tests-guidance.md`).
2. **Green** – Implement the minimal code to pass the test.
3. **Refactor** – Clean up while keeping tests green.

**Required patterns:**
- **Plugin/Strategy**: New functionality → `plugins/` using `typing.Protocol`.
- **Adapter**: scikit-learn compatibility via `WrapCalibratedExplainer`.
- **Lazy Loading**: no eager imports of heavy libs in `__init__.py`.

**Anti-patterns to avoid:**
- Deep inheritance – prefer composition and Protocols.
- Heavy DI containers – use the existing plugin registry.

---

## 9. ADR and Standards Reference Map

Always consult the relevant ADR or Standard before making design or implementation
decisions. The ADR takes precedence over any plan document.

### Architectural Decision Records (ADRs)

| ADR | Title | When to consult |
|---|---|---|
| ADR-001 | Core Decomposition Boundaries | Any change to `core/` ↔ `plugins/` boundary |
| ADR-002 | Validation and Exception Design | Adding or changing error/validation logic |
| ADR-003 | Caching Key and Eviction | Cache behavior or LRU/TTL changes |
| ADR-004 | Parallel Backend Abstraction | Changes to parallel execution strategy |
| ADR-005 | Explanation JSON Schema Versioning | Explanation payload structure changes |
| ADR-006 | Plugin Registry Trust Model | Adding or modifying a plugin registration |
| ADR-007 | Visualization Abstraction Layer | PlotSpec IR or headless export changes |
| ADR-008 | Explanation Domain Model | Explanation domain object changes |
| ADR-009 | Input Preprocessing and Mapping | Input encoding, unseen-category policy |
| ADR-010 | Core vs Evaluation Split | Import-time dependencies or extras flags |
| ADR-011 | Deprecation and Migration Policy | Deprecating or removing any API |
| ADR-012 | Documentation & Gallery Build | Docs/notebook build policy |
| ADR-013 | Interval Calibrator Plugin Strategy | Calibrator plugin changes |
| ADR-014 | Plot Plugin Strategy | Plot plugin changes |
| ADR-015 | Explanation Plugin Integration | Explanation plugin changes |
| ADR-016 | PlotSpec Separation and Schema | PlotSpec schema/validation changes |
| ADR-020 | Legacy User API Stability | Any public API change or removal |
| ADR-021 | Calibrated Interval Semantics | Interval/probability semantics |
| ADR-023 | Matplotlib Coverage Exemption | Coverage targets for viz modules |
| ADR-026 | Explanation Plugin Semantics | Explanation plugin invariants |
| ADR-027 | FAST-Based Feature Filtering | Feature filtering behavior |
| ADR-028 | Logging and Governance Observability | Logging or telemetry changes |
| ADR-029 | Reject Integration Strategy | Reject/guard mode integration changes |
| ADR-030 | Test Quality Priorities and Enforcement | Any new or modified test |
| ADR-031 | Calibrator Serialization | Calibrator save/load contracts |
| ADR-032 | Guarded Explanation Semantics | In-distribution / guarded mode |

> **Note:** ADR-017, ADR-018, and ADR-019 were reclassified as engineering standards
> (STD-001, STD-002, STD-003 respectively); the original ADR files were removed and
> replaced by the standards files in `docs/standards/`. ADR-022, ADR-024, and ADR-025
> are superseded and retained in `docs/improvement/adrs/` with a `superseded` prefix.
> See `docs/improvement/adrs/` for the full list.

All active ADRs live in `docs/improvement/adrs/`.

### Engineering Standards (STDs)

| Standard | Title | When to consult |
|---|---|---|
| STD-001 | Nomenclature Standardization | Naming new modules, classes, or functions |
| STD-002 | Code Documentation Standard | Writing or reviewing docstrings |
| STD-003 | Test Coverage Standard | Coverage targets and per-module gates |
| STD-004 | Documentation Audience Standard | Writing or restructuring docs |
| STD-005 | Logging and Observability Standard | Adding log statements or telemetry |

All standards live in `docs/standards/`.

---

## 10. ADR Conformance Workflow

When making architectural or design decisions:
1. Read the relevant ADRs in `docs/improvement/adrs/`.
2. If an ADR governs the area, the ADR takes precedence over any plan document.
3. Record the ADR reference in inline code comments for future agents.
4. If a conflict arises, request clarification rather than guessing.

---

## 11. Python Scripting Rules

Never use the heredoc Python construct:

```bash
# ❌ WRONG – does not work in this repo
python - <<PY
print("hello")
PY
```

Always use `-c`:

```bash
# ✅ CORRECT
python -c "import calibrated_explanations as ce; print(ce.__version__)"
```

---

## 12. Feedback Persistence Pattern

AI agents have no cross-session memory. To make feedback durable:

1. Convert the feedback into a concrete change in a versioned file:
   - A new bullet in this file or a platform-specific instruction file.
   - A test that reproduces the mistake.
   - A helper update in `ce_agent_utils.py`.
2. Commit the change in the same PR as the fix.
3. Record a dated entry in `.github/copilot-feedback-log.md` (see format there).

This is the **only** reliable way to make an agent "learn" across sessions.

---

## 13. Test-Quality Improvement Method

This repository uses a **team-of-agents** approach for test quality improvement
governed by ADR-030 and STD-003. Before writing, modifying, or removing tests,
agents should understand this method.

### Method overview

The full method lives in `docs/improvement/test-quality-method/README.md`. Three
usage modes:

- **Option A – Test-Focused Cycle**: Run per-test coverage pipeline → prune redundant
  tests → close coverage gaps with high-signal behavioral tests.
- **Option B – Code-Focused Cycle**: Run code-quality gates → refactor and clean up
  dead code. (Faster; skip heavy per-test pipeline unless pruning is also needed.)
- **Option C – Full Cycle**: Both A and B in sequence. Recommended for large changes.

### Quick start

```bash
# Full per-test coverage pipeline (slow; produces all analysis data)
python scripts/over_testing/run_over_testing_pipeline.py

# Per-module coverage gate check
python scripts/quality/check_coverage_gates.py
```

All outputs land in `reports/over_testing/`.

### Agent team (`test-quality-improvement`)

| Agent | File | Mission |
|---|---|---|
| **test-creator** | `docs/improvement/test-quality-method/test_creator.md` | Analyze coverage gaps; design high-value tests to close them |
| **pruner** | `docs/improvement/test-quality-method/pruner.md` | Remove/consolidate redundant tests (zero-unique-lines candidates) |
| **anti-pattern-auditor** | `docs/improvement/test-quality-method/anti_pattern_auditor.md` | Detect test quality violations (private members, weak assertions, non-determinism) |
| **code-quality-auditor** | `docs/improvement/test-quality-method/code_quality_auditor.md` | Audit source-code quality gates (exception taxonomy, imports, docstrings) |
| **deadcode-hunter** | `docs/improvement/test-quality-method/deadcode_hunter.md` | Identify source code that is dead or covered only incidentally |
| **process-architect** | `docs/improvement/test-quality-method/process_architect.md` | Design and improve the test-quality enforcement workflow |
| **devils-advocate** | `docs/improvement/test-quality-method/devils_advocate.md` | Critically review every other agent's proposals before implementation |
| **implementer** | `docs/improvement/test-quality-method/implementer.md` | Consolidate specialist proposals into a final remedy plan and execute |

**Recommended workflow:**
1. `test-creator` → produces prioritized coverage-gap analysis and test designs.
2. `pruner` → identifies zero-unique-lines tests.
3. `anti-pattern-auditor` + `code-quality-auditor` + `deadcode-hunter` → produce
   independent quality reports.
4. `devils-advocate` → challenges all proposals.
5. `implementer` → merges approved changes; runs verification gates.

### Key quality rules (ADR-030)

1. **Determinism** – no wall-clock time, nondeterministic RNG, network I/O, or test order.
2. **Public-contract testing** – validate observable public behavior; do not access `_private` members unless listed in `.github/private_member_allowlist.json`.
3. **Strong assertions** – every test must assert specific values that would fail for plausible regressions.
4. **No zero-unique-lines tests** – a test that adds zero unique coverage lines must be removed or parameterized.
5. **No identical fingerprints** – tests with the same coverage fingerprint must be consolidated with `pytest.mark.parametrize`.
6. **Fallback opt-in** – use the `enable_fallbacks` fixture and assert `UserWarning` whenever a test exercises a fallback path.

### Critical coverage targets (STD-003)

| Module | Gate |
|---|---|
| `core/calibrated_explainer.py` | ≥ 95% |
| `utils/serialization.py` | ≥ 95% |
| `plugins/registry.py` | ≥ 95% |
| `calibration/interval_regressor.py` | ≥ 95% |
| Package-wide | ≥ 90% |
