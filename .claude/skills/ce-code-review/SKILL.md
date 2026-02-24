---
name: ce-code-review
description: >
  Review code for conformance with calibrated_explanations coding standards.
  Use when asked to 'review this code', 'does this follow CE standards',
  'check my PR', 'code quality review', 'coding standards check', 'pre-commit
  checks', 'ADR conformance', 'check imports', 'check docstrings', 'CE code
  review', 'check for anti-patterns'. Covers all CE source-code standards from
  CONTRIBUTOR_INSTRUCTIONS.md, source-code.instructions.md, and the active ADRs.
---

# CE Code Review

You are reviewing code for conformance with calibrated_explanations standards.
Work through each review dimension below and produce a finding per violation.

---

## Dimension 1 — Module boundary (ADR-001) 🔴 CRITICAL

```python
# FAIL: core/ importing from plugins/
# grep -r "from.*plugins" src/calibrated_explanations/core/

# PASS: delegation pattern
class CalibratedExplainer:
    def explain_factual(self, X):
        return self._plugin_manager.explain(X, mode="factual")
```

Checklist:
- `core/` must never import `plugins/` implementation details.
- `plugins/` must never import `core/` implementation details (only protocols/exceptions are OK).
- Circular imports → use `if TYPE_CHECKING:` for type-only imports.

---

## Dimension 2 — Lazy imports 🔴 CRITICAL

```python
# FAIL — any top-level heavy import in a module reachable from the package root
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# PASS — function-scoped import
def plot(self, ...):
    import matplotlib.pyplot as plt  # inside the function body
    ...
```

Heavy libs that must always be lazy: `matplotlib`, `pandas`, `joblib`.

Scan command:
```bash
grep -rn "^import matplotlib\|^import pandas\|^import joblib\|^from matplotlib\|^from pandas\|^from joblib" \
    src/calibrated_explanations/
```

---

## Dimension 3 — Future annotations 🟡 REQUIRED

```python
# Every .py file must start with:
from __future__ import annotations
```

---

## Dimension 4 — Docstrings (Numpy style) 🟡 REQUIRED

```python
# FAIL — missing or wrong style
def predict(self, X):
    """Predict outputs."""
    ...

# PASS — Numpy style
def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict calibrated outputs.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Calibrated probability estimates.

    Raises
    ------
    ValidationError
        If the explainer is not calibrated.

    Examples
    --------
    >>> explainer.predict(X_test)
    array([0.62, 0.41, 0.87])
    """
```

Sections: `Parameters` → `Returns` → `Raises` → `Notes` → `Examples`. All optional
except for public API methods where `Parameters` and `Returns` are expected.

---

## Dimension 5 — Exception handling (ADR-002) 🟡 REQUIRED

```python
# FAIL — bare ValueError or Exception
raise ValueError("something went wrong")
raise Exception("unexpected")

# PASS — CE exception hierarchy
from calibrated_explanations.utils.exceptions import ValidationError, ConfigurationError
raise ValidationError("X_query must be 2-D; got shape ...")
```

Allowed exceptions: `ValidationError`, `ConfigurationError`, `IncompatibleSchemaError`,
`MissingExtensionError`, and other classes from `calibrated_explanations.utils.exceptions`.

---

## Dimension 6 — Fallback visibility (mandatory §7 copilot-instructions.md) 🔴 CRITICAL

```python
# FAIL — silent fallback
if parallel_failed:
    use_sequential()

# PASS — visible fallback
if parallel_failed:
    msg = "Parallel execution failed; falling back to sequential."
    _LOGGER.info(msg)
    warnings.warn(msg, UserWarning, stacklevel=2)
    use_sequential()
```

Every fallback decision must have both `_LOGGER.info(...)` and `warnings.warn(..., UserWarning)`.

---

## Dimension 7 — Type hints 🟡 REQUIRED

```python
# FAIL
def process(data, config):
    ...

# PASS
def process(data: np.ndarray, config: Mapping[str, Any]) -> CalibratedExplanations:
    ...
```

- Avoid `Any` unless a documented reason exists.
- Private members: prefix with `_`; exclude from `__all__`.

---

## Dimension 8 — Deprecation (ADR-011) 🟡 REQUIRED

```python
# FAIL — removing an old parameter without warning
def explain(self, X):  # old_param silently dropped
    ...

# PASS — using the CE deprecate helper
from calibrated_explanations.utils.deprecate import deprecate

def explain(self, X, *, old_param=None):
    if old_param is not None:
        deprecate(
            "old_param is deprecated and will be removed in v0.13.0. "
            "Use new_param instead.",
            once_key="explain_old_param_deprecation",
        )
```

Timeline: minimum 2 minor releases before removal.

---

## Dimension 9 — CE-First compliance

Public explanation-producing methods must:
1. Return a `CalibratedExplanations` or `FactualExplanation` / `AlternativeExplanation`.
2. Never return uncalibrated predictions unless explicitly documented.
3. Assert `explainer.fitted` and `explainer.calibrated` before generating outputs.

---

## Quick-check command

```bash
# Run all local CE checks (pre-commit + tests + coverage)
make local-checks-pr               # fast required checks
make local-checks                  # full checks (only needed for main-branch gates)
pre-commit run --all-files         # linting, ruff, mypy subset
```

---

## Review Report Template

```
CE Code Review: <module/PR name>
=================================
ADR-001 module boundary:    PASS / FAIL
  violations: <list file:line>

Lazy imports:               PASS / FAIL
  eager heavy imports:      <list>

Future annotations:         PASS / FAIL
  missing in:               <list>

Docstrings (numpy style):   PASS / FAIL
  missing sections in:      <list fn:section>

Exception handling:         PASS / FAIL
  bare exceptions at:       <list>

Fallback visibility:        PASS / FAIL
  missing warn()/log() at:  <list>

Type hints:                 PASS / FAIL
  untyped parameters:       <list>

Deprecation (ADR-011):      PASS / FAIL / N_A

CE-First compliance:        PASS / FAIL

Overall: CONFORMANT / NON-CONFORMANT (<N> issues)
```

## Evaluation Checklist

- [ ] All 9 dimensions checked.
- [ ] ADR-001 boundary violations are blocking (must fix before merge).
- [ ] Fallback visibility violations are blocking.
- [ ] Lazy-import violations are blocking.
- [ ] Report produced with file:line references for each issue.
