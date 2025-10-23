# Maintainer Sync Notes – v0.8.0 Plot Routing & Telemetry

**Date:** 2025-10-11
**Audience:** Core maintainers, release managers

## Highlights

- **PlotSpec Default Routing:** `CalibratedExplanation.plot` now resolves
  through the new `calibrated_explanations.plotting` module. Legacy helpers
  live under `legacy/plotting.py`; `viz/plots.py`/`legacy/_plots_legacy.py` emit
  `DeprecationWarning` and proxy to the new modules. Import hygiene (tests,
  CLI, viz adapters) has been updated accordingly.
- **Telemetry Guardrails:** Runtime telemetry and README/docs now showcase the
  PlotSpec-first path (`plot_source`, `plot_fallbacks`) plus the ADR-009
  preprocessor snapshot. `ce.plugins list --plots` surfaces `is_default` /
  `legacy_compatible` metadata so operators can verify guardrails from the CLI.

## Action Items

1. **Docs/Release Notes:** CHANGELOG and README entries drafted—reference them
   when announcing the v0.8.0 cut. Point maintainers to the updated quickstart
   snippet and plugins doc section that documents telemetry fields.
2. **Deprecation Follow-up:** The `_plots*` shims should remain until the first
  post-v0.8 minor release. Track downstream usage; plan removal once warning
  period lapses.
3. **CI/Tooling:** Ensure upcoming PRs install `pydocstyle` to satisfy
  ADR-018 checks (packages `explanations/`, `perf/`, `plugins/` now carry
  docstrings). Coverage guardrail bump (`--cov-fail-under=85`) still pending in
  CI configuration.

## Communication

- Announce the rename + telemetry guardrails in the next maintainer meeting and
  slack digest. Emphasise the new module import path, telemetry payload shape,
  and CLI discoverability so downstream teams know where to look.
- Highlight the deprecation warning emitted by `_plots` modules to encourage
  migration.
