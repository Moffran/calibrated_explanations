> **Status note (2026-04-07):** Last edited 2026-04-07 to update status to Accepted.
> Archive after: Retain indefinitely as architectural record
> Implementation window: v0.11.1–v1.0.0

# ADR-034: Centralized Configuration Management

Status: Accepted
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

`core/config_helpers.py` is a sanctioned low-level ingestion boundary only. It may
load raw configuration sources for `ConfigManager`, CLI plumbing, or migration-time
compatibility helpers, but migrated runtime consumer modules MUST NOT call it
directly.

### 2. Deterministic precedence

Each configuration key resolves in this order:

1. Call-site override (highest)
2. Environment variable
3. `pyproject.toml` (`[tool.calibrated_explanations.*]`)
4. Versioned default profile (lowest)

`ConfigManager` is snapshot-based by design. `ConfigManager.from_sources()` captures
the effective environment and `pyproject.toml` content at construction time, and all
subsequent lookups resolve against that captured snapshot. `ConfigManager.env(key)`
returns the value captured at construction time, not the live process environment.
Existing runtime objects therefore do not observe later environment changes.

Code that depends on changed environment values must reconstruct `ConfigManager` or
reconstruct the owning runtime object that holds it. Re-reading environment variables
on each lookup is non-compliant because it defeats snapshot semantics and undermines
deterministic configuration resolution.

`ConfigManager` resolves against a versioned default profile. Callers who require
stable, reproducible behavior across library upgrades SHOULD explicitly pass
`profile_id` to `ConfigManager.from_sources()` rather than relying on the
implementation-selected default. The current implementation-selected default
profile is documented in user-facing configuration reference and code, and must be
synchronized when changed.

### 3. Ownership and lifecycle

`ConfigManager` must be owned at a top-level execution boundary. A migrated
runtime consumer either:

1. receives a `ConfigManager` instance via injection, or
2. constructs one once during its own initialization and retains it for its lifetime.

Long-lived runtime components must receive a `ConfigManager` via injection or
construct one once during initialization and retain it for their own lifetime.
CLI commands may construct one `ConfigManager` per invocation. Helper functions and
lower-level utilities must not construct a fresh `ConfigManager` per lookup.

Constructing a fresh `ConfigManager` on every lookup is non-compliant because it
quietly reintroduces live-read behavior through repeated snapshot recreation.

### 4. Strict validation by default

When `strict=True` (default), unknown configuration keys in supported
`pyproject.toml` sections and type/value validation failures in supported
configuration keys raise `ConfigurationError`.

`strict=False` is an explicit escape hatch: validation errors are collected in
`ConfigValidationReport` without raising. The caller accepts responsibility for the
unvalidated structured configuration, but observability is not suppressed.

Unknown environment variables in the package-defined configuration namespace should
emit a warning rather than a hard failure because environment namespaces are
operationally noisy and may contain stale or mistyped settings.

### 5. Diagnostic export surface

`ConfigManager.export_effective()` and the CLI commands `ce config show` /
`ce config export` expose the fully resolved configuration for debugging and support.
These commands construct a manager for the current CLI invocation and report that
invocation's effective snapshot; they do not introspect already-instantiated in-memory
runtime objects. The export includes profile ID, schema version, effective values,
and per-key source attribution. `ce config show` and `ce config export` are
diagnostic/operator commands for the effective snapshot created for that invocation.
Until export schema versioning is introduced, CLI/export output is diagnostic rather
than a stable external automation contract.

### 6. Enforcement-oriented boundary rule

Allowed direct configuration-source ingestion is restricted to sanctioned boundary
modules only. In the OSS package, sanctioned boundary modules are
`core/config_manager.py` and `core/config_helpers.py`.

`core/config_helpers.py` may ingest raw environment and `pyproject.toml` data for
manager construction, CLI support, or migration-time compatibility boundaries. All
other runtime modules must resolve configuration through `ConfigManager`.

CI enforcement must fail new direct runtime reads outside sanctioned boundaries.
Temporary exceptions must be explicitly allowlisted with justification and
target-release ownership.

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
- Centralized configuration provides the basis for
  `calibrated_explanations.governance.config` lifecycle observability.

**Negative / Risks:**

- Migration requires touchpoints across several modules in phases.
- Strict validation will surface latent misconfigurations in user `pyproject.toml` files.
- Sensitive values (e.g., plugin identifiers containing secret-like names) currently
  appear in plain text in exports and governance logs. Redaction is deferred to v1.0
  (see Open Items).
- Schema-backed event emission is governed separately and closes through the release
  plan.
- Centralized snapshot-based configuration changes the effective behavior of some
  legacy ad hoc env-read paths that previously behaved like live reads. If
  public-facing workflows rely on in-process env mutation, migration notes are
  required through ADR-011/ADR-020-governed release documentation.

## Delivery Governance

ADR-034 defines the target architecture and operational contract for centralized
configuration management. Delivery sequencing, milestone scope, migration ownership,
and release gating are governed by the versioned release plans.

Use ADR-011 deprecation process for any public API changes during implementation.

## Open Items

1. **Sensitive-value redaction (v1.0):** Governance logs and exports should not leak
   secret-like values. A hybrid redaction policy (pattern-based deny + safe-key allowlist)
   is the intended approach; both the allowlist contents and the implementation are
   deferred to v1.0.

2. **Export payload schema contract (v1.0):** The `export_effective()` payload structure
   is not yet versioned for consumers. A stable schema contract with version-gating is
   needed before the export surface can be relied on by external tooling.
