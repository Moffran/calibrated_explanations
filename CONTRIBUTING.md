# Contributing — extras and test workflows

This project separates a lean core install from optional extras for visualization, notebooks and evaluation. The guidelines below explain how to work with the extras and run tests with or without them.

- Core install (recommended for development of core features):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

- Install visualization extras (required to run viz tests and examples):

```powershell
pip install -e .[viz]
```

- Install notebook extras (for notebook development):

```powershell
pip install -e .[notebooks]
```

- Install evaluation extras (for reproducing experiments):

```powershell
pip install -r evaluation/requirements.txt
# or using the project extras:
pip install -e .[eval]
```

Running tests
-------------

- Core-only test run (will skip `viz` tests when `matplotlib` isn't installed):

```powershell
pytest
```

- Run full test suite with visualization tests enabled:

```powershell
pip install -e .[viz]
pytest
```

CLI and plugin configuration
----------------------------

- The plugin CLI entry point is available as `ce.plugins` (packaged via
  `pyproject.toml`). Use it to inspect registered plugins and trust state:

```powershell
ce.plugins list all
```

- Configure plugins via environment variables or explainer kwargs:
  - Env vars: `CE_EXPLANATION_PLUGIN`, `CE_INTERVAL_PLUGIN`,
    `CE_INTERVAL_PLUGIN_FAST`, `CE_PLOT_STYLE`, plus their `*_FALLBACKS`
    counterparts.
  - Explainer kwargs: `factual_plugin`, `alternative_plugin`, `fast_plugin`,
    `interval_plugin`, `fast_interval_plugin`, and `plot_style`.
    - Preferred FAST activation: pass `fast=True` to `CalibratedExplainer` to
      enable FAST-mode execution. `fast_plugin` remains available when you need
      to target a specific FAST implementation, but `fast=True` is the primary
      activation mechanism.

Style guardrails
----------------

- Naming and documentation conventions are enforced in CI (Ruff naming + pydocstyle).
- Review the quick-reference checklist in `.github/CONTRIBUTING.md` before
  submitting changes that touch public APIs or new modules.

Logging and Observability
-------------------------

We follow [Standard-005](docs/standards/STD-005-logging-and-observability-standard.md) for logging. When adding new features:
1. Use the appropriate logger domain (e.g. `calibrated_explanations.core.*`, `calibrated_explanations.plugins.*`).
2. Use `calibrated_explanations.logging.logging_context` to propagate identifiers like `explainer_id` or `plugin_identifier`.
3. Consult [ADR-028](docs/improvement/adrs/ADR-028-logging-and-governance-observability.md) for architecture details.

Notes
-----
- The test suite automatically skips tests marked with `@pytest.mark.viz` when
  `matplotlib` cannot be imported. This makes local development faster and
  avoids false failures on minimal installs.
- If you need to run only viz tests, install the `viz` extras and run
  `pytest -m viz`.

If you add or remove optional dependencies, please update `pyproject.toml`,
`evaluation/requirements.txt`, and `evaluation/environment.yml` accordingly.
