> **Status note (2026-06-04):** Last edited 2026-06-04 to add §8 provisional plugin config hardening: raw snapshotting, source attribution, redaction, trusted binding after plugin trust/metadata resolution, and provisional export diagnostics. The plugin config surface is a hardened implementation milestone, not a compatibility-frozen contract.
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
The payload carries schema markers for diagnostic consumers. Plugin configuration
diagnostics are explicitly provisional until a separate stabilization review freezes
the schema shape and compatibility rules.

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
- Config diagnostics redact secret-like keys and schema-marked sensitive plugin
  values. Wider governance-log redaction remains tracked separately (see Open
  Items).
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

### Implementation summary (v0.11.1-v0.11.2)

- Phase A (v0.11.1): centralized authority and governance-schema closure landed for runtime ConfigManager adoption and `governance.config` lifecycle-event schema alignment.
- Phase B (v0.11.2): completed migration of `cache/cache.py`, `parallel/parallel.py`, `core/explain/_feature_filter.py`, and `core/prediction/orchestrator.py` to ConfigManager-owned reads.
- CI enforcement: `scripts/quality/check_config_manager_usage.py --scope targeted --report reports/config_manager_usage_report.json` reports zero violations as of 2026-04-20.
- Remaining deferred items: wider governance-log redaction and any compatibility-frozen export contract beyond the current diagnostic schema markers (see Open Items).
- Evidence commands:
   - `python scripts/quality/check_config_manager_usage.py --scope targeted --report reports/config_manager_usage_report.json`
   - `python -m pytest -q tests/scripts/test_check_config_manager_usage.py -o addopts= --no-cov`
   - `make local-checks-pr`

## §7 Configuration scope boundaries (v0.11.3 addendum)

### Capabilities governed by ConfigManager

`ConfigManager` governs behavioral and deployment configuration: plugin
selection (explanation, interval, plot, trust/deny), telemetry diagnostic
mode, cache settings, parallel settings, feature-filter settings, strict
observability mode, and CI-environment markers.

### Env-only-by-design keys (v0.11.x)

The following keys resolve `(None, None, None)` in `_RESOLUTION_SPEC` —
they are intentionally env-or-programmatic-only with no pyproject.toml
mapping in v0.11.x. This is a deliberate design choice, not an oversight:

| Key | Reason env-only in v0.11.x |
| --- | --- |
| `CE_CACHE` | Deployment toggle; pyproject.toml wiring deferred |
| `CE_PARALLEL` | Deployment toggle; pyproject.toml wiring deferred |
| `CE_PARALLEL_MIN_BATCH_SIZE` | Tuning knob; env-sufficient for v0.11.x |
| `CE_FEATURE_FILTER` | Experimental; pyproject.toml wiring deferred |
| `CE_STRICT_OBSERVABILITY` | Ops flag; env-sufficient for v0.11.x |
| `CI` / `GITHUB_ACTIONS` | Read-only environment markers; no user configuration |

### Sanctioned direct env read: CE_DEBUG_TRUST_INVARIANTS

`plugins/_trust.py` reads `CE_DEBUG_TRUST_INVARIANTS` directly via
`os.getenv()` rather than through `ConfigManager`. This is a sanctioned
exception: routing it through `ConfigManager` would require `_trust.py` to
import `config_manager`, creating a circular import via `plugins/registry.py`
(which already imports both modules). The key is present in `_KNOWN_ENV_KEYS`
for governance visibility and appears in `export_effective()` output; its
runtime behavior is unaffected by the ConfigManager registration.

### ExplainerBuilder / env-var precedence rule

For cache and parallel settings, env vars take precedence over
`ExplainerBuilder` settings set via `perf_cache()` / `perf_parallel()`.
This is because `CacheConfig.from_env()` and `ParallelConfig.from_env()` are
applied inside `_build_perf_factory()` after builder construction, overriding
any builder-supplied values. The rule is documented at the call site in each
method's docstring.

### Intentional two-system plot configuration design

`ConfigManager` governs plugin/style selection (which renderer and style
plugin to use). Aesthetic settings — fonts, DPI, colors — are governed by a
separate `plot_config.ini` read by `plotting.py:load_plot_config()` via
`configparser`. These serve different concerns and different update frequencies.
Bridging them into a single system would add complexity without benefit.
This separation is intentional and is not planned to change in v1.0.0.

### Root namespace exports (v0.11.3)

`ExplainerBuilder` and `ExplainerConfig` are promoted to the root
`calibrated_explanations` namespace as of v0.11.3. `ConfigManager` is
intentionally not promoted — it is an infrastructure primitive; its stable
import path is `calibrated_explanations.core.config_manager`.

### Process-level lifecycle API (v0.11.3)

`get_process_config_manager()` is the default process-level singleton for
migrated runtime consumers that do not receive an explicit `ConfigManager`
through injection. `init_process_config_manager()` may be called once by a
top-level process boundary before runtime consumers initialize. A second
initialization raises `CalibratedError` because double initialization is a
programming error that would make snapshot ownership ambiguous.

`reset_process_config_manager_for_testing()` is test-only and exists so tests
that mutate environment variables can reset the singleton between test cases.
Production code should not reset the process manager after initialization.
The process singleton is protected by a lock so concurrent first callers share
one constructed snapshot.

### ConfigSpec extension surface (v0.11.3)

`ConfigManager` owns a class-level `ConfigSpec` describing known environment
keys, pyproject sections, resolution metadata, validators, and the pyproject
tool namespace. The legacy module-level aliases (`_KNOWN_ENV_KEYS`,
`_SECTION_SCHEMA`, `_RESOLUTION_SPEC`, `_VALUE_VALIDATORS`) remain for import
compatibility, but internal resolution uses the class-level spec so subclasses
can override or merge configuration schemas without rewriting resolution logic.

## §8 Provisional plugin config hardening (2026-06-04 addendum)

This addendum hardens plugin configuration behavior without freezing the
plugin-facing contract. `ConfigManager.plugin_config(...)`,
`CE_PLUGIN_CONFIG_JSON`,
`[tool.calibrated_explanations.plugin_configs."<plugin_id>"]`,
`plugin_meta["config_schema"]`, `context.plugin_config`, and the plugin config
portion of `export_effective()` are provisional hardening surfaces. They are
available for integration across OSS CE and official plugins, but their
API shape, schema shape, export shape, and environment-variable interface remain
subject to stabilization after cross-repository validation.

### Raw config snapshotting

OSS `ConfigManager` owns raw plugin config snapshotting. It captures the process
environment and `pyproject.toml` content once, resolves plugin config with the
standard precedence model, records per-key source attribution, validates only the
raw shape, and redacts sensitive diagnostics. It does not load plugin metadata or
plugin modules during process initialization.

Malformed `CE_PLUGIN_CONFIG_JSON` fails clearly when present. Unknown or
unselected plugin config is handled through an explicit strict/permissive
selection check so deployments can choose whether configured-but-unselected
plugin entries are fatal.

`validate_plugin_config_selection(...)` is intentionally not called by
`bind_plugin_config(...)`, which validates one selected trusted plugin at a
time. The owner that knows the complete selected-plugin set for a runtime
boundary — for example `CalibratedExplainer`/`PluginManager` orchestration or a
server startup bridge — is responsible for calling it when strict
configured-but-unselected behavior is required. Diagnostic or migration
workflows may call it with `strict=False`.

### Trusted plugin config binding

Plugin config validation happens after plugin selection, trust resolution, and
metadata availability through the approved registry path. The plugin owns its
schema and semantic interpretation. OSS may validate the provisional schema shape
and apply defaults, but it must not imply that raw configuration can be fully
validated independently of plugin trust and metadata resolution.

Runtime plugin config delivered through context objects is deeply immutable,
including list-like values, so trusted plugins observe a deterministic snapshot
and cannot mutate shared config state.

### Export diagnostics

`export_effective()` includes source attribution and redacted plugin config
diagnostics with a plugin config export schema marker. The plugin config export
shape is diagnostic and provisional; external consumers must not treat it as
compatibility-frozen until a future stabilization gate explicitly freezes the
schema and release notes identify the supported contract.

### Future stabilization gate

The plugin config surface may be stabilized only after OSS CE and the
official plugin repository have passing integration evidence for snapshotting,
precedence, validation routing, redaction, strict/permissive behavior, runtime
immutability, no untrusted plugin loading, plugin template/schema validation,
and a documented selected/trusted/installed/configured plugin state matrix.

## Open Items

1. **Governance-log redaction:** Config diagnostics now redact secret-like keys and
   schema-marked sensitive values. Wider governance logs should receive the same
   redaction posture before they are treated as safe for external support bundles.

2. **Export payload schema contract:** `export_effective()` carries diagnostic
   schema markers, including a provisional plugin config export marker. A
   compatibility-frozen schema contract with version-gating is needed before
   external tooling can rely on the full export surface.
