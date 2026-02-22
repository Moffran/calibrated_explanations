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
6. **Calibrated by default** – Do not return uncalibrated outputs unless explicitly
   requested.
7. **Conjunctions** – `explanations.add_conjunctions(...)` or
   `explanations[idx].add_conjunctions(...)`.
8. **Narratives & plots** – `.to_narrative(format=...)` and `.plot(...)`.
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

## 9. ADR Conformance

When making architectural or design decisions:
1. Read the relevant ADRs in `docs/improvement/adrs/`.
2. If an ADR governs the area, the ADR takes precedence over any plan document.
3. Record the ADR reference in inline code comments for future agents.
4. If a conflict arises, request clarification rather than guessing.

---

## 10. Python Scripting Rules

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

## 11. Feedback Persistence Pattern

AI agents have no cross-session memory. To make feedback durable:

1. Convert the feedback into a concrete change in a versioned file:
   - A new bullet in this file or a platform-specific instruction file.
   - A test that reproduces the mistake.
   - A helper update in `ce_agent_utils.py`.
2. Commit the change in the same PR as the fix.
3. Record a dated entry in `.github/copilot-feedback-log.md` (see format there).

This is the **only** reliable way to make an agent "learn" across sessions.
