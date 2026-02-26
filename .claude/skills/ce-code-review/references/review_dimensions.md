# CE Code Review Dimensions

## Dimension 1 — Module boundary (ADR-001) CRITICAL

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

## Dimension 2 — Lazy imports CRITICAL

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

## Dimension 3 — Future annotations REQUIRED

```python
# Every .py file must start with:
from __future__ import annotations
```

---

## Dimension 4 — Docstrings (Numpy style) REQUIRED

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

## Dimension 5 — Exception handling (ADR-002) REQUIRED

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

## Dimension 6 — Fallback visibility (mandatory §7 copilot-instructions.md) CRITICAL

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

## Dimension 7 — Type hints REQUIRED

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

## Dimension 8 — Deprecation (ADR-011) REQUIRED

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
