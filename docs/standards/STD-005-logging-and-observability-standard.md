> **Status note (2026-01-05):** Last edited 2026-01-05 · Archive after: Retain indefinitely as an engineering standard · Implementation window: v0.10.2 and onwards.

# Standard-005: Logging and Observability

Formerly part of an ad-hoc logging practice. Reclassified and consolidated as an
engineering standard so ADRs can focus on architectural and contract decisions.

Status: Draft (2026-01-05) — to be enforced starting with the v0.10.2 milestone.

## 1. Purpose

This standard defines how logging and observability are used across the project.
It complements architectural decision records (ADRs) that cover the logging
architecture and governance model, but focuses on day-to-day contributor and
operator guidance.

Goals:

- Ensure logs are predictable, parsable, and useful for debugging, observability,
  and governance.
- Separate operational logs from governance/audit logs while keeping
  configuration consistent.
- Minimize the risk of leaking sensitive data in logs.
- Provide a clear, testable contract for contributors adding or modifying
  logging.

## 2. Relationship to ADRs

- Architectural choices about logger hierarchy, governance integration, and
  context propagation are specified in `ADR-028-logging-and-governance-observability.md`
  in `docs/improvement/adrs/`.
- This standard operationalizes those decisions by defining naming, level usage,
  context fields, and PII rules.
- If behaviour conflicts with ADR-028, the ADR prevails; this standard should
  then be updated accordingly.

## 3. Logger Hierarchy and Naming

### 3.1 Root and domains

- All project loggers MUST live under the `calibrated_explanations` root.
- The following domain subtrees are reserved and SHOULD be used consistently:
  - `calibrated_explanations.core.*` – core runtime, explain/predict orchestration,
    cache, parallel, calibration.
  - `calibrated_explanations.plugins.*` – builtin and external plugins
    (interval, explanation, plot, etc.).
  - `calibrated_explanations.telemetry.*` – metrics/telemetry shims that emit
    structured payloads via logging.
  - `calibrated_explanations.governance.*` – governance/audit events (checkpoints,
    rollbacks, trust decisions).

### 3.2 Module-level loggers

- Library modules MUST create loggers using one of the following patterns:
  - `logger = logging.getLogger(__name__)` when the module already lives under
    the correct domain path; or
  - `logger = logging.getLogger("calibrated_explanations.<domain>.<component>")`
    when a more specific domain is required.
- Tests MAY use `__name__` directly but SHOULD NOT install or modify handlers.
- Contributors MUST NOT call `logging.basicConfig` in library code;
  configuration is the responsibility of the host application or top-level
  scripts.

## 4. Log Levels and Message Conventions

### 4.1 Level semantics

- `DEBUG` – Detailed internal behaviour, decisions, and diagnostic information.
  Safe to disable in production.
- `INFO` – High-level state changes, configuration summaries, and successful
  operations that are useful for operators.
- `WARNING` – Degraded behaviour with fallbacks (e.g. disabled features, skipped
  plugins) that still preserves correctness.
- `ERROR` – Failures that prevent completing the requested operation but do not
  necessarily terminate the process.
- `CRITICAL` – Irrecoverable failures that typically precede process
  termination.

Contributors MUST:

- Use `WARNING` instead of `INFO` when behaviour deviates from the expected
  contract (e.g. filters/intervals disabled, plugin trust decisions overridden).
- Prefer `logging.exception` or `exc_info=True` for unexpected exceptions where
  stack traces are necessary.
- Avoid `DEBUG` logs in tight per-instance loops unless guarded by an explicit
  debug/trace flag or configuration toggle.

### 4.2 Message content

- Messages MUST be short, descriptive, and stable enough to support operational
  playbooks (avoid including volatile values in the free-text portion where
  possible).
- When including identifiers (explainer IDs, plugin identifiers, checkpoint
  IDs, tenant IDs), contributors SHOULD prefer structured fields (see Section 5)
  over concatenating them into the message string.

## 5. Structured Fields and Context Propagation

### 5.1 Structured fields

- Log records that describe domain events (e.g. plugin resolution, checkpoint
  operations, drift alerts) SHOULD attach structured fields via the `extra`
  argument rather than encoding this information in message text.
- At minimum, contributors SHOULD include the following keys where available:
  - `explainer_id`
  - `plugin_identifier`
  - `mode` (e.g. `factual`, `alternative`, `fast`)
  - `checkpoint_id`
  - `tenant_id` (when provided by the host environment)

### 5.2 Context helper

- The runtime MUST provide a shared logging context helper (as defined in
  ADR-028) that:
  - Stores domain context (request IDs, tenant IDs, explainer/checkpoint IDs)
    in a thread- or task-local store.
  - Installs a logging `Filter` that injects this context into all project log
    records.
- Contributors SHOULD prefer using the context helper over manually attaching
  `extra` fields when the same context applies to multiple log calls within a
  request or batch.

## 6. Governance vs Operational Logs

- Governance/audit events MUST be emitted via `calibrated_explanations.governance.*`
  loggers.
- Typical governance events include (non-exhaustive):
  - Checkpoint creation, promotion, and rollback operations.
  - Plugin trust/deny decisions and configuration changes that impact runtime
    behaviour.
  - Policy and configuration changes that affect calibration or governance
    guarantees.
- Operational logs (performance, cache behaviour, plugin loading, plotting
  fallbacks) MUST remain under the `core.*`, `plugins.*`, or `telemetry.*`
  domains.
- Release checklists and governance evidence workflows SHOULD treat governance
  logs as part of the audit trail.

## 7. Data Minimisation and PII

- Logs MUST NOT include raw feature vectors, labels, or personally identifying
  data unless explicitly gated behind a tightly controlled diagnostic mode.
- By default, logs SHOULD focus on identifiers and configuration (e.g. plugin
  IDs, modes, checkpoint IDs), not concrete input values or predictions.
- If a diagnostic mode is implemented, it MUST:
  - Be clearly documented as unsafe for production use with sensitive data.
  - Be disabled by default and controlled via configuration (environment
    variables or `pyproject.toml` settings).

## 8. Configuration and Formats

- Library code MUST NOT implicitly configure handlers or formatters; host
  applications own global logging configuration.
- The project SHOULD provide:
  - A helper function (for example,
    `calibrated_explanations.logging.configure_logging(...)`) for simple setups,
    implemented according to ADR-028.
  - Example configurations for both human-readable text logging and JSON /
    structured logging.
- Configuration knobs for logging (for example, formats, minimum levels, strict
  observability toggles) SHOULD be defined in a central location
  (`pyproject.toml` or documented environment variables) and kept in sync with
  ADR-028.

## 9. Testing and Review

- New or changed logging in core paths SHOULD be covered by tests that exercise:
  - Log emission under expected conditions (e.g. plugin deny decisions, filter
    disablement, checkpoint creation/rollback).
  - Correct domain/logger selection (e.g. governance vs core domains).
- Code review SHOULD treat logging changes as part of the public operational
  interface, particularly for governance logs and telemetry shims.

## 10. Backwards Compatibility

- Existing loggers and messages MAY remain until migrated, but new code MUST
  follow this standard.
- When changing logger names or message formats that are likely to be scraped
  or monitored, changes SHOULD be documented in the release notes and, where
  feasible, be introduced with a deprecation window or compatibility shim
  (e.g. forwarding governance events to both old and new logger names for one
  minor release).


