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
    For in-distribution filtering (guarded explanations), use `guarded_options=GuardedOptions()`:
    ```python
    from calibrated_explanations.explanations.guarded_options import GuardedOptions
    explainer.explain_factual(X, guarded_options=GuardedOptions())
    explainer.explore_alternatives(X, guarded_options=GuardedOptions())
    ```
    `GuardedOptions` fields: `confidence` (default 0.9), `n_neighbors`, `normalize`,
    `merge_adjacent`. Do NOT use the REMOVED methods `explain_guarded_factual(X)` /
    `explore_guarded_alternatives(X)` (deleted v0.11.3) or the deprecated `guarded=True`
    boolean kwarg (emits `DeprecationWarning`; removed in v1.0.0).
6. **Calibrated by default** – Do not return uncalibrated outputs unless explicitly
   requested.
7. **Conjunctions** – `explanations.add_conjunctions(...)` or
   `explanations[idx].add_conjunctions(...)`.
8. **Narratives & plots** – `.to_narrative(output_format=...)` and `.plot(...)`.
9. **Probabilistic regression** – `threshold=` for probabilistic intervals;
   `low_high_percentiles=` for conformal.

Agents must not use `calibrated_explanations.ce_agent_utils` as their
implementation shortcut. Use the public API directly. `ce_agent_utils.py` is
retained for backward compatibility and as a legacy example only.

---

## 2. Architecture

| Package | Purpose | Rules |
|---|---|---|
| `core/` | `CalibratedExplainer`, `WrapCalibratedExplainer`, `core.exceptions` | Never import `plugins/` here (ADR-001) |
| `core/config_manager.py` | Runtime configuration authority (`ConfigManager`) | Only boundary module for env/pyproject reads (ADR-034) |
| `plugins/` | Calibrators, plotters, explanation plugins | Register via plugin registry; never hard-code in `core/` |
| `calibration/` | Venn-Abers & Conformal Prediction primitives | Stateless helpers only |
| `utils/` | `deprecate`, logging, serialization | Shared across packages |

Design patterns:
- **Lazy Loading**: `matplotlib`, `pandas`, `joblib` are imported lazily. Never
  add top-level imports of heavy libs in any module reachable from the package root.
- **Plugin-First**: New functionality goes into `plugins/`, not `core/`.
- **Exception hierarchy**: Use `core.exceptions` (ADR-002). No bare `Exception` or
  `ValueError` unless documented.
- **Config authority**: All runtime configuration reads go through
  `ConfigManager` (ADR-034). Do not call `os.getenv` or parse `pyproject.toml`
  directly outside `core/config_manager.py` and `core/config_helpers.py`. Use
  `get_process_config_manager()` to access the process-level singleton.

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

### 4A. Capability-Based Verification

Externally visible CE capability claims are product/library claims about
`calibrated_explanations` behavior. Treat them as verifiable engineering
contracts, not as verified facts merely because they appear in documentation.

Use this chain for release confidence:

```text
Capability claim
    -> requirement
    -> verification case
    -> evidence record
```

Definitions:

- **Capability claim**: a user-visible or release-visible statement about what
  CE provides.
- **Requirement**: a scoped, testable obligation derived from one or more
  capability claims.
- **Verification case**: an automated, manual, analytical, or review-based
  check mapped to a requirement.
- **Evidence**: a durable record of the verification result, including enough
  metadata to reconstruct the context.

Rules:

1. Capability claims are not requirements.
2. Requirements are not tests.
3. Tests are not evidence.
4. Evidence is the recorded result of running a defined verification against a
   specific version and configuration.
5. Capability claims must be decomposed into explicit requirements before they
   are used for release confidence.
6. Requirements must be testable, scoped, and linked to concrete public
   behavior.
7. Tests must verify requirements, not vague promotional language.
8. Evidence should record enough information to reconstruct what was checked,
   including the version, data/configuration, verification identifier, and
   result.

Each requirement should distinguish the obligation type it expresses, for
example:

- API contract
- payload/schema contract
- numerical behavior
- statistical-method assumption
- documentation boundary
- visualization behavior
- plugin or extension behavior

Statistical claims must state their assumptions explicitly, especially
calibration-data assumptions, exchangeability assumptions, task-type
constraints, and empirical-vs-theoretical verification boundaries.

Avoid overclaiming:

- Empirical tests do not prove finite-sample theoretical guarantees by
  themselves.
- A test that checks output shape does not prove calibration validity.
- A visual rendering test does not prove the scientific meaning of an
  explanation.
- Documentation claims must be scoped to what the implementation and tests
  actually support.

Anti-patterns:

- Treating vague documentation prose as a testable requirement without
  normalization.
- Hiding acceptance criteria inside test code only.
- Claiming that empirical smoke tests prove theoretical guarantees.
- Adding tests with no requirement link.
- Adding requirements with no verification strategy.
- Adding claims with no owner.
- Duplicating capability definitions in multiple places.
- Using vague claims as release evidence.

Future claim IDs and requirement IDs should use CE-native prefixes, for example
`CE-CAP-...` and `CE-REQ-...`.

Canonical CE locations for capability verification material:

| Material | Location |
|---|---|
| Capability claims | `development/capabilities/claims/` |
| Requirements | `development/capabilities/requirements/` |
| Verification scenarios and helpers | `verification/capabilities/` |
| Pytest verification | `tests/capabilities/` for new capability-contract tests; existing nearby unit or integration tests may be linked from requirements when appropriate |
| Evidence run outputs | `reports/verification/` |
| Curated release or closure evidence summaries | `development/finished-work/` |

These paths define the CE verification layout even if the directories do not yet
exist. Do not create additional locations for the same material. This instruction
defines the layout only; do not add claim catalogs, requirement catalogs,
schemas, verification code, tests, or evidence records unless the task explicitly
requires that work.

Illustrative example only, not a binding schema:

```yaml
claim_id: CE-PRED-REG-001
claim_type: capability
claim_text: >
  CE supports calibrated prediction intervals for supported regression workflows.
requirements:
  - CE-REQ-PRED-REG-API-001
  - CE-REQ-PRED-REG-BOUNDS-001
  - CE-REQ-PRED-REG-ASSUMPTIONS-001
verification:
  proves:
    - api_contract
    - interval_bounds_contract
    - documented_assumption_boundary
evidence_required:
  - commit_sha
  - package_version
  - test_id
  - dataset_id
  - random_seed
  - result
```

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
| `src/calibrated_explanations/core/config_manager.py` | Runtime configuration authority: `ConfigManager`, `get_process_config_manager`, `init_process_config_manager` |
| `src/calibrated_explanations/plugins/` | Plugin implementations |
| `src/calibrated_explanations/ce_agent_utils.py` | Legacy compatibility module — backward-compat and example only, not the recommended agent interface |
| `development/README.md` | Canonical development documentation map |
| `development/current-work/` | Active release plans, status tracking, and execution checklists |
| `development/future-work/` | Forward-looking plans that are not active execution state |
| `development/finished-work/` | Closed plans and curated closure evidence summaries |
| `development/adrs/` | Architectural Decision Records after migration |
| `development/standards/` | Engineering Standards after migration |
| `development/capabilities/claims/` | Capability claim catalog when introduced |
| `development/capabilities/requirements/` | Requirement catalog when introduced |
| `verification/capabilities/` | Capability verification scenarios and helpers when introduced |
| `docs/get-started/ce_first_agent_guide.md` | Runnable CE-first guide |
| `docs/foundations/how-to/configure_runtime.md` | How-to guide: ConfigManager, env vars, pyproject.toml sections, export diagnostics |
| `docs/improvement/` | Legacy development-planning area; existing files remain valid until migrated, but no new planning or verification-governance files should be added there |
| `docs/standards/` | Legacy Engineering Standards location until migrated |
| `docs/improvement/test-quality-method/` | Legacy test-quality agent team definitions until migrated |
| `tests/README.md` | Authoritative test guidance |
| `CHANGELOG.md` | Changelog; update under `## [Unreleased]` for every change |
| `Makefile` | Entry points: `make test`, `make ci-local` |

### Root-directory policy

**No new root-level directories may be created** without explicit maintainer approval.
`development/` is the approved exception for maintainer planning, engineering
governance, capability claim/requirement catalogs, and curated closure evidence
summaries.
Artifact and report outputs must be placed under an existing top-level directory:
`reports/`, `artifacts/`, `docs/`, `scripts/`, `tests/`, or `development/`
according to the location map above.
Proposing a new root directory requires a PR rationale and an update to this file.

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

### First-time setup (required before running any checks)

```bash
# Install the package in editable mode with all dev dependencies
pip install -e .[dev] -c constraints.txt

# Optional: faster install if uv is available
uv pip install -e .[dev] -c constraints.txt

# Install pre-commit hooks (run once per clone)
pre-commit install
```

Verify the environment is healthy:

```bash
python -m ruff --version    # must be present
python -m mypy --version    # must be present
python -m pytest --version  # must be present
```

### Routine local validation

```bash
# Fast PR-scope checks (lint + type-check + core tests + policy scanners)
# This is the primary local validation path — run before every commit.
make local-checks-pr

# Full local CI including main-branch gates (coverage, perf, over-testing)
make local-checks

# Run tests only
make test

# Run only non-viz tests (faster; avoids matplotlib import)
make test-core
```

### Performance profiling

See `docs/contributor/performance_harness.md` for a guide to the available
profiling scripts (`scripts/perf/`) and how to run baseline snapshots and
regression checks.

### Governance status artifact

`reports/governance/governance_status.json` is a CI-derived artifact.
The `lint` fields (`ruff`, `mypy`) show **"unavailable"** when generated
without running lint tools. Use the local target to populate them:

```bash
# Run ruff + mypy locally, write results into the artifact
make governance-status-local
```

`local_checks_pr` always shows **"unavailable"** locally — only CI can
set it after the full test suite passes. The `schema_checks` fields are
populated from local report files and reflect the last time those scripts ran.

`make local-checks-pr` calls `make governance-status-local` internally,
so after a full local checks run the artifact will have real ruff/mypy results.

Before any implementation work:
1. Read `development/README.md` to identify the current development map and
   transition rules.
2. Read the active release plan. During migration, this may still be
   `docs/improvement/RELEASE_PLAN_v1.md`; after migration it belongs under
   `development/current-work/`.
3. Check the governing ADRs and Standards. During migration, active records may
   still live in `docs/improvement/adrs/` and `docs/standards/`; after migration
   they belong under `development/adrs/` and `development/standards/`.

### 7A. Engineering planning hierarchy authority (mandatory)

The repository planning/control hierarchy is authoritative and must be preserved:

1. `development/README.md` (development documentation map and location authority)
2. Active release plan in `development/current-work/` after migration; during
   migration, `docs/improvement/RELEASE_PLAN_v1.md` remains valid
3. Concrete implementation plans in `development/current-work/` or
   `development/future-work/`
4. Governance via ADRs and Standards in `development/adrs/` and
   `development/standards/` after migration; during migration, existing records
   in `docs/improvement/adrs/` and `docs/standards/` remain valid

Conflict rule: ADRs and Standards govern design, behavior, architecture, and
engineering standards. If any plan text conflicts with an ADR/Standard, the
ADR/Standard is authoritative and must win.

Do not redesign, replace, or create another planning hierarchy unless one of these
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

`docs/improvement/` is a legacy planning area during migration. Do not add new
planning, ADR, Standard, claim, requirement, verification-framework, or curated
evidence files there. Existing files may be edited in place only when moving them
would be outside the scope of the current task. When active material from
`docs/improvement/` is already being substantially changed, move it to the
appropriate `development/` location in the same change.

Default execution posture for plan/instruction edits:

- If no concrete contradiction, missing required rule, or execution blocker
  exists, make no structural documentation changes.
- Apply minimum-diff edits only; preserve existing headings, numbering,
  milestone-plan structure, and future-milestone detail unless a concrete
  conflict requires local clarification.
- Do not create new plan/policy/checklist files by default; prefer resolving
  shared guidance in this file with short cross-references from plans only when
  necessary.

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

Current ADRs live in `docs/improvement/adrs/` until migrated. New ADRs belong
under `development/adrs/` unless the task is explicitly scoped to maintaining an
existing legacy file in place.

### Engineering Standards (STDs)

| Standard | Title | When to consult |
|---|---|---|
| STD-001 | Nomenclature Standardization | Naming new modules, classes, or functions |
| STD-002 | Code Documentation Standard | Writing or reviewing docstrings |
| STD-003 | Test Coverage Standard | Coverage targets and per-module gates |
| STD-004 | Documentation Audience Standard | Writing or restructuring docs |
| STD-005 | Logging and Observability Standard | Adding log statements or telemetry |

Current standards live in `docs/standards/` until migrated. New standards belong
under `development/standards/` unless the task is explicitly scoped to
maintaining an existing legacy file in place.

---

## 10. ADR Conformance Workflow

When making architectural or design decisions:
1. Read `development/README.md` to identify the current authoritative locations.
2. Read the relevant ADRs. During migration, existing ADRs may still live in
   `docs/improvement/adrs/`; new ADRs belong under `development/adrs/`.
3. If an ADR governs the area, the ADR takes precedence over any plan document.
4. Record the ADR reference in inline code comments for future agents.
5. If a conflict arises, request clarification rather than guessing.

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

The full method currently lives in
`docs/improvement/test-quality-method/README.md` until migrated. Three usage
modes:

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
