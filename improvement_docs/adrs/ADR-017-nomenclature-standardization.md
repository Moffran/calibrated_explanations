# ADR-017: Internal nomenclature standardisation

Status: Accepted (2025-10-06)

## Context

Repeated refactors have left the package with a patchwork of naming schemes. Mixed module casings (`_VennAbers.py` vs `_interval_regressor.py`), transitional shims (`core.py` vs the `core` package), and helpers that reach into mangled private attributes make it difficult for contributors to infer intent from names alone. The public API exposed in the README must remain unchanged, yet internal clarity is critical to finish ongoing modularisation work. 【F:src/calibrated_explanations/_VennAbers.py†L1-L144】【F:src/calibrated_explanations/_interval_regressor.py†L1-L40】【F:src/calibrated_explanations/core.py†L1-L14】【F:src/calibrated_explanations/core/calibration_helpers.py†L19-L115】

## Decision

Adopt the following naming standards for all **non-public** code paths (tests and tooling included) while keeping the public API stable:

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

- 2025-10-06 – Ratified as **Accepted** following maintainer sign-off; preparatory
  guardrails scheduled for the v0.7.0 release to begin enforcing the conventions.
- Upcoming – Initial lint guardrails (Ruff module-naming checks, double underscore
  detectors) and contributor style excerpts land alongside documentation updates
  so that new contributions adhere to the standard by default.
- Acceptance notes recorded for v0.7.0 staging: contributor quick
  reference published in `CONTRIBUTING.md` and CI now surfaces Ruff naming
  warnings to track adoption without blocking merges.
