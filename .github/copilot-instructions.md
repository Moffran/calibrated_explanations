# GitHub Copilot Instructions for `calibrated_explanations`

> **Scope:** Repository-wide defaults for Copilot. **Always** pair these notes with `.github/tests-guidance.md` and `docs/improvement/adrs/` for authoritative rules.

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
> **Ref:** `.github/tests-guidance.md`

1. **Modify existing files first.** Locate the nearest test file for the SUT and extend it.
2. **Respect scope + naming.** Use `tests/unit|integration|e2e/...` paths with `test_<module>.py` naming.
3. **Content rubric.** Output deterministic, AAA-structured pytest tests named `should_<behavior>_when_<condition>`.
4. **Mocking & snapshots.** Mock only to avoid slow I/O. Assert behaviors, not implementation details.
5. **Coverage context.** Keep an eye on `pytest --cov=... --cov-fail-under=90`.

## 5. Key Files & Directories
- `src/calibrated_explanations/core/__init__.py`: Public API surface (lazy loaded).
- `src/calibrated_explanations/plugins/`: Plugin implementations.
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
