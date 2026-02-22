# GitHub Copilot Instructions for `calibrated_explanations`

> **Canonical agent instructions:** `AGENT_INSTRUCTIONS.md` — the single source of
> truth shared by all agent platforms (Copilot, Codex, Claude Code, Gemini). Read
> that first. This file adds **only** GitHub Copilot-specific context.
> **Setup guide:** `docs/get-started/copilot-setup.md` — VS Code setup, prompt files, feedback loop.
> **Feedback log:** `.github/copilot-feedback-log.md` — dated corrections; run `/refresh-ce-context feedback="…"` to add an entry.

---

## 1. Project Context & Architecture
- **Core Purpose:** Extract calibrated explanations (factual rules, alternatives, prediction intervals) from scikit-learn compatible models.
- **Key Components:**
  - `core/`: Main logic (`CalibratedExplainer`, `WrapCalibratedExplainer`). **Strictly separated** from plugins (ADR-001).
  - `plugins/`: Extensible architecture for calibrators, plotters, and explanations (ADR-006, ADR-013, ADR-014).
  - `calibration/`: Underlying calibration logic (Venn-Abers, Conformal Prediction).
- **Design Patterns:**
  - **Lazy Loading:** Heavy dependencies (matplotlib, etc.) are imported lazily in `__init__.py` and `__getattr__` to minimize startup time.
  - **Plugin-First:** New functionality should often be implemented as a plugin rather than modifying core logic directly.
  - **Validation:** Use the unified exception hierarchy in `core.exceptions` (ADR-002).

## 2. Development Workflow
- **Build & Test:**
  - Run tests: `make test` or `pytest`.
  - Run local CI: `make ci-local` (runs linting, type checking, and tests).
  - **Strict Gates:** The project enforces high standards for coverage (90%+), docstrings, and linting.
- **Release Process:**
  - Follow `docs/improvement/RELEASE_PLAN_v1.md`.
  - Check `docs/improvement/adrs/` for architectural constraints before major changes.

## 3. Coding Standards
- **Type Hinting:** Use `from __future__ import annotations` and comprehensive type hints.
- **Docstrings:** Strict **Numpy style** docstrings are required (ADR-018).
- **Imports:** Avoid circular imports. Use `if TYPE_CHECKING:` for type-only imports.
- **Deprecation:** Use `utils.deprecate` for legacy features (ADR-011).

## 4. Testing Guidance (Critical)
> **Ref:** `tests/README.md`

1. **Modify existing files first.** Locate the nearest test file for the SUT and extend it.
2. **Respect scope + naming.** Use `tests/unit|integration|e2e/...` paths with `test_<module>.py` naming.
3. **Content rubric.** Output deterministic, AAA-structured pytest tests named `test_should_<behavior>_when_<condition>`.
4. **Mocking & snapshots.** Mock only to avoid slow I/O. Assert behaviors, not implementation details.
5. **Coverage context.** Keep an eye on `pytest --cov=... --cov-fail-under=90`.

## 5. Key Files & Directories
- `src/calibrated_explanations/core/__init__.py`: Public API surface (lazy loaded).
- `src/calibrated_explanations/plugins/`: Plugin implementations.
- `src/calibrated_explanations/ce_agent_utils.py`: CE-first runtime guardrails for agents.
- `docs/improvement/adrs/`: Architectural Decision Records (Read these!).
- `Makefile`: Entry points for build and test tasks.

## 6. Design Patterns & TDD
**Prefer Test-Driven Development (TDD):**
1.  **Red:** Write a failing test case first (following the Testing Guidance).
2.  **Green:** Implement the minimal code to pass the test.
3.  **Refactor:** Clean up the code while keeping tests green.

**Required Patterns:**
-   **Plugin/Strategy:** Implement new functionality (calibrators, plotters) as plugins in `plugins/` using `typing.Protocol`. Do not modify `core/` logic unless necessary.
-   **Adapter:** Maintain scikit-learn compatibility via `WrapCalibratedExplainer`.
-   **Lazy Loading:** **CRITICAL:** Do not add eager imports for heavy libs (matplotlib, pandas) in `__init__.py`. Use `__getattr__` or local imports.

**Nice-to-Haves (Recommended):**
-   **Facade:** `CalibratedExplainer` acts as a facade. When adding complex subsystems, hide them behind a simple facade method.
-   **Factory:** The registry acts as a factory. Use factory methods for creating complex `Explanation` objects.

**Anti-Patterns (Overhead):**
-   **Deep Inheritance:** Prefer composition and Protocols.
-   **Heavy DI:** Use the existing registry; do not introduce DI containers.

## 7. Fallback Visibility Policy (MANDATORY)

All runtime, plugin, visualization, and utility fallbacks must be visible to users.

- **Log Info:** Emit an `INFO` log summarizing the fallback decision (what failed, what path is chosen).
- **Raise Warning:** Emit a `warnings.warn(..., UserWarning)` with a clear message. No silent fallbacks.
- **Where:** Apply to execution strategy selection (parallel → sequential), plugin execution fallbacks (legacy path), cache backend fallbacks, visualization simplifications, and perturbation fallbacks.
- **Tests:**
  - If a test depends on a fallback, assert that a warning is raised using `pytest.warns(UserWarning)`.
  - If a test does not depend on a fallback, then the fallback chain must be explicitly made empty to avoid unexpected fallbacks.
- **Docs:** When introducing a new fallback, update `docs/improvement/RELEASE_PLAN_v1.md` and `CHANGELOG.md` with a short note.

This policy enforces traceability and observability across all fallback decisions and is required for PR approval.

---

## 8. Python Scripting

Never use the construct:

```bash
python - <<PY
# Your Python code here
PY
```

It does not work!

Prefer using a python -c command, like:

```pwsh
python -c "import calibrated_explanations as ce; print(ce.__version__)"
```

---

## 9. Copilot-Specific: Instruction Files (auto-injected context)

The following files are automatically loaded into Copilot's context based on the
file you are editing (via `applyTo` frontmatter). You do not need to paste them
into the chat.

| File | Scope | Purpose |
|---|---|---|
| `.github/instructions/source-code.instructions.md` | `src/**/*.py` | Module layout, import rules, docstring style, error handling |
| `.github/instructions/tests.instructions.md` | `tests/**`, `*test*` | Test framework, naming, structure, coverage gate |
| `.github/instructions/execution plan.instructions.md` | all files | Release plan, ADR conformance, changelog policy |

---

## 10. Copilot-Specific: Prompt Slash Commands

Type `/` in Copilot Chat to invoke these CE workflows:

| Command | Use when |
|---|---|
| `/generate-tests-strict` | Writing new tests for any CE module |
| `/implement-plugin` | Scaffolding a new calibrator, plot, or explanation plugin |
| `/fix-issue` | Diagnosing and fixing a bug or failing test |
| `/refresh-ce-context` | Updating instructions after an API change or to record feedback |

---

## 11. Copilot-Specific: Chat Tips

- Use `@workspace` for questions about CE internals — it indexes local files.
- Use `#file:AGENT_INSTRUCTIONS.md` to explicitly pin the canonical instructions in chat.
- Commit instruction-file updates in the same PR as the code change so history stays in sync.
- Persist feedback in `.github/copilot-feedback-log.md` with `Feedback`, `Root cause`, `Durable fix`, `Verification`, and `Status` fields.
