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

Implement a conservative, opt-in plugin registry:

- Discovery via entry points group `calibrated_explanations.plugins` (setuptools) or explicit `register_plugin()` call.
- Registry stores metadata: name, version, provider, capabilities, checksum (optional), `trusted` flag.
- By default, only built-in plugins auto-load. Third-party plugins require explicit trust action: environment variable `CE_TRUST_PLUGIN=<name>` or programmatic `trust_plugin(name)`.
- On first detection of untrusted plugin, emit warning with guidance and skip load.
- Provide `list_plugins(include_untrusted=True)` API for diagnostics.
- Optional integrity field: author may supply SHA256 of source dist; library can verify if hash file present (best-effort, not security grade initially).
- Isolation: no sandboxing initially (document risk); future ADR may explore subprocess / WASM.

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

## Adoption & Migration

Phase B (v0.6.0): Land minimal registry data structures and API; load built-ins only; document trust workflow.
Phase E–F (v0.7.0): Enable third-party registration & trust workflow behind explicit opt-in; publish docs + examples.

## Open Questions

- Should we support a denylist environment variable? (Probably yes for defense-in-depth.)
- Provide structured plugin capability descriptors for filtering? (Yes, minimal early.)
- Expose plugin health/validation checks (schema compliance, version compatibility)?

## Decision Notes

Revisit after first external plugin submissions to evaluate need for stronger isolation.
