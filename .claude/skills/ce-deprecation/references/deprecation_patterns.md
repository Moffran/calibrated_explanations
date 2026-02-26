# Deprecation Code Patterns (ADR-011)

## The `deprecate()` helper

```python
from calibrated_explanations.utils.deprecate import deprecate

# Basic usage — emits DeprecationWarning once per session per once_key
deprecate(
    "old_param is deprecated and will be removed no earlier than v0.13.0. "
    "Use new_param instead.",
    once_key="module.function.old_param_deprecation",
)
```

- Warning type: `DeprecationWarning` (not `UserWarning`).
- Emitted **once per session** per `once_key` (no warning fatigue).
- `once_key` format: `<module>.<class_or_function>.<symbol>_deprecation` (reverse-DNS style).

---

## Deprecating a function parameter

```python
def explain_factual(self, X, *, n_top: int | None = None, top_features: int | None = None):
    """Explain factual predictions.

    Parameters
    ----------
    X : np.ndarray
        Query instances.
    n_top : int, optional
        .. deprecated:: 0.11.0
            Use ``top_features`` instead. Will be removed in v0.13.0.
    top_features : int, optional
        Maximum number of features to return.
    """
    if n_top is not None:
        from calibrated_explanations.utils.deprecate import deprecate
        deprecate(
            "'n_top' is deprecated since v0.11.0 and will be removed in v0.13.0. "
            "Use 'top_features' instead.",
            once_key="CalibratedExplainer.explain_factual.n_top_deprecation",
        )
        if top_features is None:
            top_features = n_top
    ...
```

---

## Deprecating a function / class

```python
def old_explain(self, X):
    """Old explanation method.

    .. deprecated:: 0.11.0
        Use `explain_factual` instead. Will be removed in v0.13.0.
    """
    from calibrated_explanations.utils.deprecate import deprecate
    deprecate(
        "'old_explain()' is deprecated since v0.11.0. Use 'explain_factual()' instead.",
        once_key="CalibratedExplainer.old_explain_deprecation",
    )
    return self.explain_factual(X)
```

---

## Deprecating a module import path

```python
# src/calibrated_explanations/legacy_module.py
"""Legacy module path — deprecated since v0.11.0.

.. deprecated:: 0.11.0
    Import from ``calibrated_explanations.core`` instead.
"""
from __future__ import annotations

from calibrated_explanations.utils.deprecate import deprecate

deprecate(
    "'calibrated_explanations.legacy_module' is deprecated since v0.11.0. "
    "Import from 'calibrated_explanations.core' instead.",
    once_key="legacy_module_import_path_deprecation",
)

# Re-export everything so existing code keeps working
from calibrated_explanations.core import *  # noqa: F401, F403
```

---

## Testing deprecations

```python
import pytest
import warnings


def test_should_emit_deprecation_warning_when_old_param_used():
    """DeprecationWarning must fire exactly once per session per key."""
    from calibrated_explanations import WrapCalibratedExplainer
    explainer = ...  # fit + calibrate

    with pytest.deprecated_call():
        explainer.explain_factual(X_query, n_top=5)


def test_should_not_emit_for_new_param_when_using_top_features():
    """No deprecation warning when using the new parameter."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        explainer.explain_factual(X_query, top_features=5)
        # no exception → test passes
```
