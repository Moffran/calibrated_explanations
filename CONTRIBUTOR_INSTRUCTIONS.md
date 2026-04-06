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

> Full guidance: `tests/README.md`

- Framework: **pytest** + **pytest-mock**. No alternative frameworks.
- Naming: `test_should_<behavior>_when_<condition>` (pytest collection-safe).
- Structure: Arrange–Act–Assert; one logical assertion block per behavior.
- Determinism: no real network, clock, or randomness; seed RNG; freeze time.
- File policy: extend the nearest existing test file first; new files require a
  "Why a new test file?" justification in the PR.
- Coverage gate: `pytest --cov=src/calibrated_explanations --cov-config=pyproject.toml --cov-fail-under=90`.
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
| `tests/README.md` | Authoritative test guidance |
| `CHANGELOG.md` | Changelog; update under `## [Unreleased]` for every change |
| `Makefile` | Entry points: `make test`, `make ci-local` |

---

## 6A. Shared Skill Registry (Cross-Agent)

All agent platforms (Codex, GitHub Copilot, Claude Code, Gemini, and others)
must treat `.claude/skills/` as the canonical repository skill catalog.

### Skill discovery and use

1. Discover skills from `./.claude/skills/*/SKILL.md`.
2. Select and apply the minimum skill set matching the task intent.
3. Follow each selected skill's `SKILL.md` workflow before writing ad-hoc logic.
4. If a skill conflicts with CE-first policy, CE-first policy wins.

### Current shared skills

| Skill | Path | Primary use |
|---|---|---|
| `ce-adr-author` | `.claude/skills/ce-adr-author/SKILL.md` | Author or update ADR documents |
| `ce-adr-consult` | `.claude/skills/ce-adr-consult/SKILL.md` | Identify and apply relevant ADRs for a task |
| `ce-adr-gap-analyzer` | `.claude/skills/ce-adr-gap-analyzer/SKILL.md` | Analyze ADR compliance by verifying implementation and RTD against ADR intent |
| `ce-alternatives-explore` | `.claude/skills/ce-alternatives-explore/SKILL.md` | Alternative/counterfactual exploration workflows |
| `ce-calibrated-predict` | `.claude/skills/ce-calibrated-predict/SKILL.md` | Generate calibrated predictions without explanations |
| `ce-classification` | `.claude/skills/ce-classification/SKILL.md` | Calibrated Explanations for binary and multiclass tasks |
| `ce-code-quality-auditor` | `.claude/skills/ce-code-quality-auditor/SKILL.md` | Identify quality risks and anti-patterns per ADR-030 |
| `ce-code-review` | `.claude/skills/ce-code-review/SKILL.md` | Perform CE-focused code reviews |
| `ce-data-preparation` | `.claude/skills/ce-data-preparation/SKILL.md` | Validate and preprocess input data for CE pipelines |
| `ce-deadcode-hunter` | `.claude/skills/ce-deadcode-hunter/SKILL.md` | Identify and clean up unreachable or non-contributing code |
| `ce-deprecation` | `.claude/skills/ce-deprecation/SKILL.md` | Implement ADR-011 compliant deprecations |
| `ce-devils-advocate` | `.claude/skills/ce-devils-advocate/SKILL.md` | Rigorously review agent proposals for risks and blind spots |
| `ce-docstring-author` | `.claude/skills/ce-docstring-author/SKILL.md` | Write or fix numpy-style docstrings |
| `ce-explain-interact` | `.claude/skills/ce-explain-interact/SKILL.md` | Interactive explanation workflows |
| `ce-factual-explain` | `.claude/skills/ce-factual-explain/SKILL.md` | Factual explanation workflows |
| `ce-fallback-impl` | `.claude/skills/ce-fallback-impl/SKILL.md` | Implement visible, policy-compliant fallback paths |
| `ce-fallback-test` | `.claude/skills/ce-fallback-test/SKILL.md` | Fallback visibility and warning tests |
| `ce-integration-compare` | `.claude/skills/ce-integration-compare/SKILL.md` | Guide CE integration with SHAP and LIME |
| `ce-logging-observability` | `.claude/skills/ce-logging-observability/SKILL.md` | Manage logging, governance, and audit context (ADR-028) |
| `ce-modality-extension` | `.claude/skills/ce-modality-extension/SKILL.md` | Extend CE support to new data modalities |
| `ce-mondrian-conditional` | `.claude/skills/ce-mondrian-conditional/SKILL.md` | Configure conditional/Mondrian calibration flows |
| `ce-notebook-audit` | `.claude/skills/ce-notebook-audit/SKILL.md` | Audit notebooks for API/policy compliance |
| `ce-onboard` | `.claude/skills/ce-onboard/SKILL.md` | Session-start CE orientation and skill routing |
| `ce-payload-governance` | `.claude/skills/ce-payload-governance/SKILL.md` | Manage and validate explanation payloads (ADR-005) |
| `ce-pipeline-builder` | `.claude/skills/ce-pipeline-builder/SKILL.md` | CE-first pipeline construction tasks |
| `ce-plot-review` | `.claude/skills/ce-plot-review/SKILL.md` | Review visualization and plotting changes |
| `ce-plotspec-author` | `.claude/skills/ce-plotspec-author/SKILL.md` | Author or extend PlotSpec-driven visualizations |
| `ce-plugin-audit` | `.claude/skills/ce-plugin-audit/SKILL.md` | Audit plugin implementations and contracts |
| `ce-performance-tuning` | `.claude/skills/ce-performance-tuning/SKILL.md` | Configure caching, parallelism, and batch-size tuning |
| `ce-plugin-scaffold` | `.claude/skills/ce-plugin-scaffold/SKILL.md` | Scaffold new plugin implementations |
| `ce-regression-intervals` | `.claude/skills/ce-regression-intervals/SKILL.md` | Probabilistic/conformal regression interval workflows |
| `ce-regulatory-compliance` | `.claude/skills/ce-regulatory-compliance/SKILL.md` | Map CE capabilities to EU AI Act, GDPR, and liability directive obligations |
| `ce-reject-policy` | `.claude/skills/ce-reject-policy/SKILL.md` | Configure reject/defer decision policies |
| `ce-release-check` | `.claude/skills/ce-release-check/SKILL.md` | Select and validate next release tasks |
| `ce-release-finalize` | `.claude/skills/ce-release-finalize/SKILL.md` | Execute the PyPI release checklist |
| `ce-release-planner` | `.claude/skills/ce-release-planner/SKILL.md` | Analyze release plan and produce vX.Y.Z implementation plans |
| `ce-release-task` | `.claude/skills/ce-release-task/SKILL.md` | Identify, implement, and verify individual release tasks |
| `ce-rtd-auditor` | `.claude/skills/ce-rtd-auditor/SKILL.md` | Audit RTD docs for structure, accuracy, and governance alignment |
| `ce-rtd-writer` | `.claude/skills/ce-rtd-writer/SKILL.md` | Author or revise RTD docs with audience-aware structure |
| `ce-serialization-audit` | `.claude/skills/ce-serialization-audit/SKILL.md` | Audit serialization behavior and coverage |
| `ce-serializer-impl` | `.claude/skills/ce-serializer-impl/SKILL.md` | Implement serialization and persistence support |
| `ce-skill-audit` | `.claude/skills/ce-skill-audit/SKILL.md` | Audit repository skills against Claude authoring guidance |
| `ce-skill-creator` | `.claude/skills/ce-skill-creator/SKILL.md` | Create/refactor skills and enforce structure/trigger quality |
| `ce-skill-registry-sync` | `.claude/skills/ce-skill-registry-sync/SKILL.md` | Enforce synchronization of all skill registries |
| `ce-standards-gap-analyzer` | `.claude/skills/ce-standards-gap-analyzer/SKILL.md` | Analyze STD compliance by verifying implementation and RTD against STD intent |
| `ce-test-audit` | `.claude/skills/ce-test-audit/SKILL.md` | Test quality/compliance audit workflows |
| `ce-test-author` | `.claude/skills/ce-test-author/SKILL.md` | High-signal CE test authoring workflows |
| `ce-test-creator` | `.claude/skills/ce-test-creator/SKILL.md` | Design high-efficiency tests to close coverage gaps |
| `ce-test-pruning-expert` | `.claude/skills/ce-test-pruning-expert/SKILL.md` | Identify and remove redundant or low-value tests |
| `ce-test-quality-method` | `.claude/skills/ce-test-quality-method/SKILL.md` | Coordinate the full Test Quality Method (ADR-030) |

### Maintenance rule

Whenever a skill is added, renamed, removed, or moved under `.claude/skills/`,
or when a user requests a skill listing update, agents must invoke
`ce-skill-registry-sync` and update both this section and
`.claude/skills/ce-onboard/SKILL.md` section 4 and
`docs/contributor/agent_skills.md` in the same PR. All listings must include
every existing skill directory under `.claude/skills/`.

---

## 7. Development Workflow

```bash
# Install (editable, with dev extras)
pip install -e .[dev]

# Run tests
make test
# or
pytest --cov=src/calibrated_explanations --cov-config=pyproject.toml --cov-fail-under=90

# Run full local CI (lint + type-check + tests)
make ci-local

# Pre-commit hooks
pre-commit install && pre-commit run --all-files
```

Before any implementation work:
1. Read `docs/improvement/RELEASE_PLAN_v1.md` to identify the active milestone.
2. Check `docs/improvement/adrs/` for ADRs governing the area being changed.

### 7A. Engineering planning hierarchy authority (mandatory)

The repository planning/control hierarchy is authoritative and must be preserved:

1. `docs/improvement/RELEASE_PLAN_v1.md` (release-level plan and control gates)
2. Concrete implementation plans in `vX.Y.Z_plan.md`
3. Governance via ADRs (`docs/improvement/adrs/`) and Standards (`docs/standards/`)

Do not redesign, replace, or create a parallel planning hierarchy unless one of these
is true:

- The current documents directly contradict each other.
- A required engineering rule is missing and safe execution is impossible without it.
- The current structure causes a concrete execution failure (not a stylistic concern).
- A repository-wide instruction for all agents and human contributors has no suitable
  existing home elsewhere.

When a shared instruction must be added or updated, `CONTRIBUTOR_INSTRUCTIONS.md`
is the primary location. Platform-specific files (`AGENTS.md`, `CLAUDE.md`,
`GEMINI.md`, `.github/copilot-instructions.md`) must not become the main source of
repository-wide engineering rules.

---

## 8. Design Patterns & TDD

**TDD workflow:**
1. **Red** – Write a failing test first (follow `tests/README.md`).
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
| ADR-008 | Explanation Domain Model | Explanation domain object changes |
| ADR-009 | Input Preprocessing and Mapping | Input encoding, unseen-category policy |
| ADR-010 | Core vs Evaluation Split | Import-time dependencies or extras flags |
| ADR-011 | Deprecation and Migration Policy | Deprecating or removing any API |
| ADR-012 | Documentation & Gallery Build | Docs/notebook build policy |
| ADR-013 | Interval Calibrator Plugin Strategy | Calibrator plugin changes |
| ADR-015 | Explanation Plugin Integration | Explanation plugin changes |
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
| ADR-033 | Modality Extension Plugin Contract and Packaging Strategy | Modality metadata, plugin API compatibility, resolver modality selection, or extension packaging/shim policy |
| ADR-034 | Centralized Configuration Management | Runtime config reads, env/`pyproject.toml` precedence, strict config validation, or config export/governance events |
| ADR-035 | CI Workflow Governance | Changes to `.github/workflows/**`, `.github/actions/ci-policy/**`, CI merge gates, or `scripts/local_checks.py` parity rules |
| ADR-036 | PlotSpec Canonical Contract and Validation Boundary | PlotSpec canonical model, serialization boundaries, and validation contract changes |
| ADR-037 | Visualization Extension and Rendering Governance | Plot builder/renderer governance and visualization extension changes |

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
3. Record a dated entry in `.github/copilot-feedback-log.md` using this schema:
   - `**Feedback:**` what the agent got wrong
   - `**Root cause:**` why the error happened
   - `**Durable fix:**` exact files/tests/scripts updated
   - `**Verification:**` command(s) proving the fix
   - `**Status:**` `open | ✅ incorporated`

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
7. **No production test-helper wrapper exports** – do not expose test-helper scaffolding via `__all__` in `src/`; CI blocks this via `scripts/quality/check_no_test_helper_exports.py`.

### Critical coverage targets (STD-003)

| Module | Gate |
|---|---|
| `core/calibrated_explainer.py` | ≥ 95% |
| `utils/serialization.py` | ≥ 95% |
| `plugins/registry.py` | ≥ 95% |
| `calibration/interval_regressor.py` | ≥ 95% |
| Package-wide | ≥ 90% |

---

## 14. Feedback-Driven Agent Optimization (Shared Canonical Loop)

To maximize agent and Copilot efficiency for calibrated_explanations:

- **Centralize feedback:** Log all agent/Copilot errors, misses, and improvement notes in `.github/copilot-feedback-log.md` after each PR/session.
- **Update instructions:** After each feedback entry, update `CONTRIBUTOR_INSTRUCTIONS.md` and platform-specific instruction files with new rules, anti-patterns, and best practices.
- **Strict CE-first enforcement:** Always prime agents with the latest canonical instructions; fail fast or warn for non-canonical patterns.
- **Continuous improvement:** Review feedback log and instructions after each release/major PR; communicate changes to all agent platforms.
- **Verification:** Ensure feedback log and instructions are updated and referenced in PRs; validate agent suggestions follow latest CE-first guardrails.

This loop ensures all agents and Copilot benefit from improvements and become more expert and reliable over time.
