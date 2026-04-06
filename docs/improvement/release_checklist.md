# Release Checklist

This checklist tracks compliance with release gates defined in `docs/improvement/RELEASE_PLAN_v1.md`.
It must be reviewed and completed for every pull request that touches the user-facing API or documentation.

Boundary policy: keep checklist structure stable within a milestone; adjust gates and wording at milestone boundaries to reduce process overhead.

## Legacy Stability (ADR-020)

- **Legacy contract touched?**
  - [ ] No
  - [ ] Yes (requires verification against `docs/improvement/legacy_user_api_contract.md`)
- **Notebooks audited?**
  - [ ] Yes (run `python scripts/quality/audit_notebook_api.py notebooks --check`)
  - [ ] No (only allowed if notebooks are untouched)
- **API snapshot updated?**
  - [ ] N/A (no public API changes)
  - [ ] Yes (run `python scripts/quality/snapshot_public_api.py`)

## Documentation & Gallery (ADR-012) (run `cd docs && sphinx-build -b html . _build/html`)

- **Documentation builds?**
  - [ ] Yes - without warnings
- **Gallery examples execute? (within time ceilings)**
  - [ ] Yes

## Test Quality (ADR-030) (run `python scripts/anti-pattern-analysis/detect_test_anti_patterns.py --check --baseline .github/test-quality-baseline.json`)

- **Anti-patterns check**
  - [ ] Passed

## v0.11.0 Gate Closure (ADR-033/006/026 + Pattern 1)

- **ADR-033 metadata contract enforced (`plugin_api_version`, `data_modalities`)**
  - [ ] Passed
- **ADR-033 ambiguity handling covered by tests**
  - [ ] Passed
- **PluginTrustPolicy protocol exists and is registry-wired**
  - [ ] Passed
- **Dead registry `include_untrusted` descriptor params removed**
  - [ ] Passed
- **Pattern 1: test-helper wrapper exports/re-exports blocked**
  - [ ] Passed
- **ADR-026 immutable context + telemetry assertions covered**
  - [ ] Passed

## Formatting & Standards (run `pre-commit run --all-files`)

- **Linting**
  - [ ] Ruff passes
- **Typing**
  - [ ] MyPy passes
