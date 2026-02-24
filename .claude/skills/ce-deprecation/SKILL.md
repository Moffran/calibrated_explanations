---
name: ce-deprecation
description: >
  Deprecate or remove a feature, parameter, or module in calibrated_explanations.
  Use when asked to 'deprecate this feature', 'deprecate a parameter', 'mark as
  deprecated', 'migration path', 'remove old param', 'ADR-011',
  'utils.deprecate', 'DeprecationWarning', 'once_key', 'two minor release rule',
  'deprecation timeline', 'CE_DEPRECATIONS=error', 'mitigation guide'. Covers
  the ADR-011 deprecation policy, the mitigation guide
  (docs/migration/deprecations.md) updates, symbol removal lifecycle, and
  searching commit history for deprecation origins.
---

# CE Deprecation & Mitigation

You are deprecating a public symbol, parameter, module path, or serialized output,
or removing a previously deprecated symbol. Follow ADR-011 strictly.
The mitigation guide (`docs/migration/deprecations.md`) is the single source of
truth for all deprecated and removed features.

---

## The Mitigation Guide (`docs/migration/deprecations.md`)

Whenever you encounter a deprecated symbol or implement a new deprecation,
maintain consistency with the **Mitigation Guide**:

1.  **Check if existing**: Every symbol marked with a `.. deprecated::` docstring MUST be listed in the status table of `docs/migration/deprecations.md`.
2.  **Add if missing**: If you find a deprecation in code that isn't in the guide, add it immediately.
3.  **Removal Status**: Check the "Removal ETA" column to determine if a symbol is eligible for removal.

---

## Removing Deprecated Symbols

Before removing a symbol, verify it meets the ADR-011 "Two Minor Release" rule:
- A symbol deprecated in `v0.10.x` is only eligible for removal in `v0.12.x` or later.
- If the current codebase version is ≥ the removal version listed in the Mitigation Guide, perform the removal.

**Steps for removal**:
1.  Delete the implementation, deprecated parameters, or module shims.
2.  Update `docs/migration/deprecations.md`: Move the entry from the "Status table" to a "Removed" section or update its status to "Removed in vX.Y.Z".
3.  Remove associated deprecation tests.
4.  Update `docs/improvement/RELEASE_PLAN_v1.md` status table.

---

## Historical Research (Missing Versions)

If a symbol's docstring contains a `.. deprecated::` note but lacks a version
(e.g., `.. deprecated:: (unknown version)`), you must determine the origin:

1.  **Search commit history**: Use `git log -S ".. deprecated::"` or `git blame <file>` on the docstring lines.
2.  **Find the release**: Identify the earliest version/tag containing that change.
3.  **Update code + guide**: Add the found version to the docstring and the Mitigation Guide entry.

---

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

## Timeline (ADR-011)

```
v0.X.0  — introduce new API; add deprecation warning for old API
v0.X+1  — still warning (minimum: 2 minor releases before removal)
v0.X+2  — remove old API (earliest)
```

Example:
- Introduced deprecation in `v0.11.0` → earliest removal is `v0.13.0`.

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

## Docstring annotation

Always add a `.. deprecated::` directive to the Numpy docstring:

```
.. deprecated:: <version>
    <one-line reason and migration pointer>.
```

---

## Migration guide entry

When deprecating a public symbol, add an entry to the status table in
`docs/migration/deprecations.md` (the "mitigation guide"):

```markdown
| Deprecated symbol | Replacement | Introduced | Removal ETA | Notes |
|---|---|---:|---:|---|
| `old_param` | `top_features` | v0.11.0 | v0.13.0 | Uses `deprecate()` in `explain_factual`. |
```

Also update the status table in `docs/improvement/RELEASE_PLAN_v1.md`.

---

## CI strict mode

Users can opt into treating deprecation warnings as errors:
```bash
CE_DEPRECATIONS=error pytest
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

---

## Out of Scope

- Legacy User API (ADR-020) — governed by "Major Release Only" lifecycle, not ADR-011.
- Plugin removal — plugins follow their own version lifecycle but still use `deprecate()`.

## Evaluation Checklist

- [ ] `deprecate()` called with descriptive message naming old and new symbol.
- [ ] `once_key` is unique and follows the `<module>.<symbol>_deprecation` pattern.
- [ ] `.. deprecated:: <version>` added to docstring.
- [ ] If version is unknown, research commit history to find deprecation origin.
- [ ] Removal version is at least 2 minor releases after the deprecation release.
- [ ] Mitigation Guide (`docs/migration/deprecations.md`) status table updated.
- [ ] `RELEASE_PLAN_v1.md` status table updated.
- [ ] Test uses `pytest.deprecated_call()` to assert the warning fires.
