# Deprecation follow-up checklist

This file lists remaining `DeprecationWarning` occurrences that should be migrated
to the centralized helpers (`deprecate()` / `deprecate_alias()` / `deprecate_public_api_symbol`).
The goal is to ensure consistent messaging and CI-enforceable `CE_DEPRECATIONS=error` behaviour.

For each entry below provide a small patch replacing `warnings.warn(..., DeprecationWarning, ...)`
with the appropriate helper, add a stable `key=` value, and run focused tests.

Suggested pattern:

```py
from calibrated_explanations.utils.deprecations import deprecate

deprecate("message...", key="module:path:identifier", stacklevel=2)
```

---

Remaining locations found (scan run):

- `src/calibrated_explanations/__init__.py` (top-level deprecated public symbols)
  - Action: already uses `deprecate_public_api_symbol()` in many places; review other deprecated entries and ensure all use `deprecate_public_api_symbol()`.

- `src/calibrated_explanations/core/__init__.py` (module-level note about normal DeprecationWarning)
  - Action: inspect and, where programmatic warnings are emitted, route through `deprecate_public_api_symbol()`.

- `src/calibrated_explanations/core/wrap_explainer.py` (alias-kwargs deprecation guidance at ~L912)
  - Action: replace direct `warnings.warn(DeprecationWarning)` in alias handling with `deprecate_alias()`.

- `src/calibrated_explanations/core/reject.py` and `src/calibrated_explanations/core/reject/policy.py`
  - Action: migrate `DeprecationWarning` emits to `deprecate()` / `deprecate_alias()` as appropriate.

- `src/calibrated_explanations/core/explain/__init__.py` (deprecated exports)
  - Action: ensure `deprecate_public_api_symbol()` use for exported deprecated names.

- `src/calibrated_explanations/core/calibration/__init__.py`, `venn_abers.py`, `summaries.py`
  - Action: replace any direct `DeprecationWarning` emits with `deprecate()` or `deprecate_public_api_symbol()` depending on context.

- Note: some occurrences are docstrings/notes referencing DeprecationWarning; only real `warnings.warn()` or `warnings.warn(..., DeprecationWarning)` callsites require migration.

---

Recommended next steps

1. Triage the listed files and create small PRs (1-3 files each) migrating to helpers.
2. For each PR, run focused tests that touch the changed modules with `pytest -q -o addopts=--no-cov <tests>`.
3. After migration, run a repository scan to ensure no remaining callsites and update this checklist.

If you want, I can prepare an initial PR patchset migrating the remaining callsites now (in batches).
