---
applyTo:
  - "src/**/*.py"
priority: 90
---

## CE Source-Code Rules (applies to all files under `src/`)

### Module layout
- `core/` — `CalibratedExplainer`, `WrapCalibratedExplainer`, shared exceptions (`core.exceptions`).
  Do **not** import plugins here.
- `plugins/` — calibrators, plotters, explanation plugins.  Each plugin registers itself via the
  plugin registry; never hard-code plugin logic inside `core/`.
- `calibration/` — Venn-Abers and Conformal Prediction primitives.  These are stateless helpers.
- `utils/` — shared helpers (`deprecate`, logging, serialization).

### Strict import rules
- Heavy optional libs (`matplotlib`, `pandas`, `joblib`) **must** be imported lazily:
  either inside the function body or via `__getattr__` in `__init__.py`.
- Use `if TYPE_CHECKING:` blocks for type-only imports to avoid circular dependencies.
- Never add top-level `import matplotlib` or `import pandas` to any module that is
  reachable from the package root without an extras flag.

### Coding style
- All files: `from __future__ import annotations` at the top.
- Docstrings: **Numpy style** (`Parameters`, `Returns`, `Raises`, `Examples`).
- Type hints: comprehensive; avoid `Any` unless there is a documented reason.
- Private members: prefix with `_`; do not expose them in `__all__`.

### Error handling
- Use the unified exception hierarchy from `core.exceptions` (ADR-002).
- Do not raise bare `Exception` or `ValueError` unless the callers already expect them
  as part of the documented contract.

### Fallback visibility (mandatory – see `copilot-instructions.md` §7)
- Every fallback must emit `warnings.warn(..., UserWarning)` **and** an `INFO` log.
- Parallel → sequential fallbacks must be announced before execution starts.
- No silent fallbacks anywhere in `src/`.
