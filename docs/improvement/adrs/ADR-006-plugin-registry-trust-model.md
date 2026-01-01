> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-006: Plugin Registry Trust Model

Status: Accepted
Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Planned plugin system will allow third parties to register calibration or explanation strategies and visual components. Executing arbitrary plugin code poses supply-chain and integrity risks. Need a minimal trust model and user controls before opening the surface.

## Decision

Implement a conservative, opt-in plugin registry with explicit trust policy hooks:

- Discovery via entry points group `calibrated_explanations.plugins` (setuptools) or explicit `register_plugin()` call.
- Registry stores metadata: name, version, provider, capabilities, checksum or signature metadata (optional), `trusted` status.
- By default, only built-in plugins auto-load. Third-party plugins require explicit trust action: environment variable `CE_TRUST_PLUGIN=<name>` or programmatic `trust_plugin(name)`.
- Support allowlist/denylist policy controls (`CE_TRUST_PLUGIN`, `CE_DENY_PLUGIN`) and a `PluginTrustPolicy` interface that can be overridden by integrators.
- On detection of untrusted or denied plugins, emit a warning with guidance and skip load.
- Provide `list_plugins(include_untrusted=True)` API for diagnostics.
- Integrity checks: allow authors to supply SHA256 or a signed metadata blob. If present, the registry verifies against the supplied hash/signature and records the result (best-effort, non-blocking unless policy requires it).
- Activation logging: every plugin load or rejection emits a structured audit event via the existing logging/telemetry hook (name, version, provider, decision, reason).
- Isolation: no sandboxing initially (document risk); future ADR may explore subprocess / WASM.
- **Delegation & Ownership:** The `PluginManager` serves as the single source of truth for plugin resolution and defaults. `CalibratedExplainer` delegates all explanation requests to this manager, ensuring that trust and opt-in rules are consistently enforced regardless of the entry point.

## Alternatives Considered

1. Auto-load all entry points (simpler, higher risk).
2. Hard-coded allowlist only (limits ecosystem growth, manual overhead).
3. Full sandbox (too heavy early, increases complexity significantly).

## Consequences

Positive:

- Reduces accidental execution of unknown code.
- Transparent listing aids auditing & debugging.
- Scales to eventual signed metadata without breaking API.

Negative / Risks:

- False sense of security (trust flag is coarse; code still executes in-process).
- Added friction for quick experimentation (must trust explicitly).
- Users may ignore warnings; need clear docs.
- Optional signature verification depends on producers supplying correct metadata.

## Adoption & Migration

Phase B (v0.6.0): Land minimal registry data structures and API; load built-ins only; document trust workflow.
Phase E–F (v0.7.0): Enable third-party registration & trust workflow behind explicit opt-in; publish docs + examples.

## Open Questions

- Provide structured plugin capability descriptors for filtering? (Yes, minimal early.)
- Expose plugin health/validation checks (schema compliance, version compatibility)?

## Decision Notes

Revisit after first external plugin submissions to evaluate need for stronger isolation.
