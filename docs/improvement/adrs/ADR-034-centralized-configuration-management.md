> **Status note (2026-03-03):** Last edited 2026-03-03
> Archive after: Retain indefinitely as architectural record
> Implementation window: v0.11.1–v1.0.0

# ADR-034: Centralized Configuration Management

Status: Proposed
Date: 2026-03-03
Deciders: Core maintainers
Reviewers: Runtime, Plugin, and Governance maintainers
Supersedes: None
Superseded-by: None
Related: ADR-003, ADR-004, ADR-006, ADR-011, ADR-015, ADR-020, ADR-028, ADR-029

## Context

Runtime configuration is currently distributed across multiple modules
(`plugins/manager.py`, `plugins/registry.py`, `logging.py`, and cache/parallel helpers),
each reading environment variables and `pyproject.toml` independently.
This causes inconsistent precedence handling across modules, scattered validation,
and no single point for inspecting effective configuration.

## Decision

### 1. Single runtime authority

`ConfigManager` is the canonical entry point for all runtime configuration reads.
Production code MUST NOT call `os.getenv` or parse `pyproject.toml` directly outside
the approved boundary modules (`core/config_manager.py`, `core/config_helpers.py`).
Existing out-of-boundary reads are temporary exceptions tracked in a CI allowlist
and must be migrated (see Adoption & Migration).

### 2. Deterministic precedence

Each configuration key resolves in this order:

1. Call-site override (highest)
2. Environment variable
3. `pyproject.toml` (`[tool.calibrated_explanations.*]`)
4. Versioned default profile (lowest)

`ConfigManager.env(key)` returns the value captured at construction time, not the live
process environment. Code that depends on specific env values must set them before
constructing ConfigManager.

Callers who require stable, reproducible configuration across library upgrades SHOULD
explicitly pass `profile_id` to `ConfigManager.from_sources()` rather than relying
on the default. The default profile is `"v1"`.

### 3. Strict validation by default

When `strict=True` (default), unknown configuration keys at any nesting level and
unrecognised `pyproject.toml` sections raise `ConfigurationError`.

`strict=False` is an explicit escape hatch: validation errors are collected in
`ConfigValidationReport` without raising. The caller accepts responsibility for the
unvalidated configuration.

### 4. Diagnostic export surface

`ConfigManager.export_effective()` and the CLI commands `ce config show` /
`ce config export` expose the fully resolved configuration for debugging and support.
The export includes profile ID, schema version, effective values, and per-key
source attribution.

## Alternatives Considered

1. **Keep distributed readers.** Rejected: preserves precedence drift and scattered
   validation with no audit point.
2. **Centralize in documentation only.** Rejected: policy without an implementation
   point will always regress.
3. **Third-party config library (e.g., dynaconf, pydantic-settings).** Rejected: adds
   a mandatory dependency with semantics that do not map cleanly onto the CE precedence
   model or plugin trust surface; overkill for the problem at hand.

## Consequences

**Positive:**

- Consistent precedence and validation across all runtime surfaces.
- Single location to inspect or export effective configuration.
- Governance log events (`calibrated_explanations.governance.config`) for resolve,
  export, and validation-failure lifecycle points, supporting audit requirements.

**Negative / Risks:**

- Migration requires touchpoints across several modules in phases.
- Strict validation will surface latent misconfigurations in user `pyproject.toml` files.
- Sensitive values (e.g., plugin identifiers containing secret-like names) currently
  appear in plain text in exports and governance logs. Redaction is deferred to v1.0
  (see Open Items).

## Adoption & Migration

- **Phase A (v0.11.1 — complete):** `ConfigManager` introduced; plugin manager and
  registry fully migrated; CI allowlist scanner wired to PR flow.
- **Phase B (v0.11.2 target):** Migrate `cache/cache.py`, `parallel/parallel.py`,
  `core/explain/_feature_filter.py`, `core/prediction/orchestrator.py`.
  Done when all four files are removed from the CI allowlist.
- **Phase C (v0.11.3 target):** Migrate any remaining readers; shrink or close allowlist.

Use ADR-011 deprecation process for any public API changes during migration.

## Open Items

1. **Sensitive-value redaction (v1.0):** Governance logs and exports should not leak
   secret-like values. A hybrid redaction policy (pattern-based deny + safe-key allowlist)
   is the intended approach; both the allowlist contents and the implementation are
   deferred to v1.0.

2. **Export payload schema contract (v1.0):** The `export_effective()` payload structure
   is not yet versioned for consumers. A stable schema contract with version-gating is
   needed before the export surface can be relied on by external tooling.
