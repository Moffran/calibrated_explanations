> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as an engineering standard · Implementation window: Per Standard status (see Decision).

# Standard-001: Internal nomenclature standardisation

Formerly ADR-017. Reclassified as an engineering standard to keep ADRs scoped to
architectural or contract decisions.

Status: Active (2025-10-06)

## Context

Repeated refactors left the package with a patchwork of naming schemes. The canonical calibration modules now live in `core/venn_abers.py`, `core/interval_regressor.py`, and the plotting surface in `viz/plots.py`. Earlier compatibility wrappers lived under `legacy/_*.py` and emitted `DeprecationWarning` during the migration; they have now been retired, leaving `legacy/plotting.py` as the lone legacy surface while the public API remains unchanged. Transitional shims such as `core.py` still exist but are clearly marked, letting contributors infer intent without losing the public API guarantees documented in the README.【F:src/calibrated_explanations/core/venn_abers.py†L1-L120】【F:src/calibrated_explanations/core/interval_regressor.py†L1-L120】【F:src/calibrated_explanations/viz/plots.py†L1-L20】【F:src/calibrated_explanations/legacy/plotting.py†L1-L120】

## Decision

Adopt the following naming standards for all **non-public** code paths (tests and tooling included) while keeping the public API stable, explicitly preserving the WrapCalibratedExplainer contract (fit/calibrate/explain/predict routines, plotting helpers, and uncertainty/threshold options) without renames or deprecation notices:

1. **Module and package names** must use `snake_case`. Leading underscores are reserved for private transitional modules that are scheduled for removal. CamelCase file names are prohibited.
2. **Class names** use `PascalCase`; helper classes intended for local use should include a suffix clarifying scope (e.g. `...Helper`, `...Mixin`).
3. **Functions, methods, and module-level variables** use `snake_case`. Names signalling booleans should be prefixed with verbs such as `is_`, `has_`, or `should_`.
4. **Private attributes** use a single leading underscore. Double-underscore name mangling is limited to legacy code; new code MUST NOT introduce it outside of dataclass field defaults required by language constraints.
5. **Transitional shims** (e.g. deprecated modules kept for import compatibility) must include a `deprecated_` prefix or live in a `legacy` subpackage so that their status is self-evident.
6. **Utility modules** should be split by responsibility (e.g. `fs_utils`, `type_utils`) with names describing the primary concern. Cross-cutting helpers belong in a dedicated module with descriptive prefixes (`convert_`, `ensure_`, etc.).
7. **Configuration and schema identifiers** (plugin IDs, registry keys) follow dot-delimited lowercase paths (`core.explanation.factual`). Aliases should be documented in one location and deprecated with clear suffixes (e.g. `.legacy`).
8. **Documentation and ADRs** must reference canonical names and call out deprecated identifiers explicitly to avoid drift between prose and code.

## Consequences

- **Pros**: Predictable naming reduces onboarding friction, clarifies module ownership, and enables mechanical refactors (e.g. automated module moves) without repeated human review of naming choices.
- **Cons**: Renaming legacy files will touch many imports; coordinated updates and deprecation warnings are necessary to avoid churn for downstream users reading stack traces.
- **Neutral**: The public API remains untouched, but internal contributors must adhere to the new conventions and update style guides accordingly.

## Implementation notes

- Update the contributor documentation with a quick-reference table describing the naming rules.
- Introduce lint checks (Ruff `N8` rules or custom scripts) that fail when modules deviate from snake_case or when new double-underscore attributes appear.
- Ensure deprecation shims emit warnings explaining the new module names and pointing to migration guides.
- Plan renames in batches (per subpackage) to keep pull requests reviewable while moving toward full compliance.

## Status tracking

- 2025-10-06 – Ratified as **Accepted** following maintainer sign-off with
  preparatory guardrails targeting the v0.7.0 milestone.
- v0.7.0 – Ruff naming checks and double-underscore detectors run in
  non-blocking mode, CONTRIBUTING.md gains the quick-reference guide, and
  telemetry captures naming lint drift to size the remaining debt.
- v0.8.0 – Phase 2 renames move priority packages to canonical names with
  shims captured under `legacy/`, and CI surfaces warnings when new
  aliases appear outside the allowed namespace. Completed via the
  `core.interval_regressor`, `core.venn_abers`, and `viz.plots` moves in
  this release.
- v0.9.0 – Deprecated shims scheduled for removal are pruned, naming lint
  promotion to blocking status is completed on the release branch, and
  release notes document the surviving compatibility shims.
- v1.0.0-rc – Remaining transitional imports are removed, and the release
  candidate checklist verifies lint parity between `main` and the RC
  branch before freeze.
- v1.0.0 – Post-tag monitoring keeps the naming lint suite blocking and
  schedules quarterly audits of the legacy namespace to guard against
  regressions.
