> **Status note (2026-01-05):** Last edited 2026-01-05 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.10.2–v1.0.0.

# ADR-028: Logging and Governance Observability

Status: Draft
Date: 2026-01-05
Deciders: Runtime & Governance maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Existing logging in the runtime focuses on fallbacks, configuration failures,
and optional telemetry. It uses the standard Python `logging` module with a
`NullHandler` at the package root and module-level loggers in many modules. This
is appropriate for a library but lacks a cohesive architectural model for:

- Separating operational logs from governance/audit logs.
- Providing consistent, machine-readable context (explainer IDs, plugin
  identifiers, checkpoint IDs, tenant IDs) for observability and evidence.
- Integrating logging with telemetry and governance features such as
  checkpoints, rollback, and plugin trust decisions.

As the project moves toward more formal governance and observability guarantees,
logging behaviour becomes part of the architectural contract, not just an
implementation detail. We need a clear, shared decision for how loggers are
structured, how context flows, and how logging interacts with governance and
telemetry. Standard-005 captures contributor-level rules; this ADR defines the
underlying architecture.

## Decision

We adopt the following architectural model for logging and governance
observability:

1. **Logger hierarchy and domains.**
   - All loggers live under the `calibrated_explanations` root.
   - We define four primary domains:
     - `calibrated_explanations.core.*` — core runtime (explain/predict
       orchestration, cache, parallel, calibration).
     - `calibrated_explanations.plugins.*` — builtin and external plugins
       (interval, explanation, plot, etc.).
     - `calibrated_explanations.telemetry.*` — telemetry and metrics integration
       layers that emit structured payloads via logging.
     - `calibrated_explanations.governance.*` — governance and audit events
       (checkpoints, rollbacks, trust/deny decisions, policy changes).
   - Module-level loggers SHOULD use `logging.getLogger(__name__)` where package
     layout already reflects the correct domain, otherwise they SHOULD compute a
     fully-qualified domain path under the root.

2. **Governance vs operational separation.**
   - Governance/audit events (for example, checkpoint creation/promotion,
     rollback operations, plugin trust/deny decisions, configuration/policy
     changes that impact guarantees) MUST be emitted via
     `calibrated_explanations.governance.*` loggers.
   - Operational events (performance, cache activity, plugin loading,
     visualization fallbacks, feature filter behaviour) MUST remain in the
     `core.*`, `plugins.*`, or `telemetry.*` domains.
   - Governance logs are treated as part of the audit evidence surface and are
     expected to feed into Evidence Pack or similar governance workflows.

3. **Context propagation.**
   - We introduce a shared logging context helper in the runtime that:
     - Stores contextual identifiers such as `request_id`, `tenant_id`,
       `explainer_id`, `checkpoint_id`, and `plugin_identifier` in a thread- or
       task-local structure (e.g. via `contextvars`).
     - Installs a project-wide logging `Filter` that injects this context into
       all `calibrated_explanations.*` records.
   - Core entry points (explain/predict, checkpoint management, plugin
     resolution, governance operations) MUST set or update context when they
     begin work and clear or restore it on exit.
   - External correlation IDs (e.g. from an API gateway) MAY be bridged into
     this context by host applications but are not required for correctness.

4. **Structured logging compatibility.**
   - Log records emitted by the runtime MUST be compatible with both human-
     readable text formatting and JSON/structured formatting. The architecture
     assumes that:
     - Message text is short, human-readable context.
     - Structured fields (e.g. `explainer_id`, `plugin_identifier`, `mode`,
       `checkpoint_id`, `tenant_id`) are attached as record attributes (via
       `extra` or the context filter).
   - The library itself does not enforce a particular formatter; host
     applications choose between text and JSON formatting. Standard-005
     documents recommended configurations.

5. **Configuration boundaries.**
   - The library does not implicitly install global handlers or formatters
     beyond adding a `NullHandler` at import time. Global configuration remains
     the responsibility of host applications or top-level scripts.
   - The runtime MAY provide a helper (for example,
     `calibrated_explanations.logging.configure_logging(...)`) that installs
     a minimal, best-practice configuration for simple deployments
     (e.g. stdout handler, text or JSON formatter) but such helpers MUST be
     opt-in and MUST NOT be invoked automatically on import.
   - Logging-related configuration knobs (for example, strict observability
     toggles, log levels, formatter choices) SHOULD be expressed via
     `pyproject.toml` or documented environment variables and are considered
     part of the configuration surface of the runtime.

6. **Data minimisation.**
   - By architectural default, logging focuses on identifiers, configuration and
     status, not raw input data or model outputs.
   - Diagnostic modes that log more detailed data MAY exist but are considered
     opt-in and environment-specific; they must be explicitly gated and are not
     required for core correctness.

Standard-005, *Logging and Observability*, captures the contributor-facing rules
that implement this decision (logger naming, level usage, data minimisation,
testing expectations).

## Alternatives Considered

1. **Ad-hoc, module-local logging without domains or context helpers.**
   - Simpler in the short term but makes it difficult to route governance logs
     separately, carry consistent identifiers, or generate machine-readable
     evidence.

2. **Single root logger with no domains, using only structured fields.**
   - Operators would have to rely exclusively on message content or structured
     fields to differentiate subsystems, complicating configuration for
     environments where logger hierarchies are the primary control surface.

3. **Tight coupling to a third-party structured logging framework.**
   - Would simplify structured logging integration, but reduce portability and
     increase the burden on downstream users who already have logging
     infrastructure based on the standard library.

4. **Separate audit ledger without any governance logging.**
   - Strong for auditability, but removes a natural integration point between
     observability and governance and forces every consumer to integrate a new
     storage path.

We rejected these alternatives in favour of a standard-library-based design
with explicit domains and context propagation, which provides a good balance
between compatibility and governance/observability needs.

## Consequences

Positive:

- Operators can configure routing, retention, and formatting separately for
  `core.*`, `plugins.*`, `telemetry.*`, and `governance.*` logs without changes
  to library code.
- Governance events (checkpoints, rollbacks, plugin trust/deny, configuration
  changes) become first-class, machine-readable artefacts suitable for audit
  and Evidence Pack generation.
- Context (request/tenant/explainer/checkpoint/plugin IDs) becomes consistently
  available for both logs and downstream observability tools.
- The design remains compatible with the standard `logging` module and
  existing deployment environments.

Negative / Risks:

- Introducing domains and a shared context helper requires plumbing context into
  core entry points and plugin orchestration, which may touch many call sites.
- Misuse of governance domains (e.g. emitting high-volume operational events
  under `governance.*`) could inflate audit logs and increase storage costs.
- Host applications that previously relied on implicit behaviour may need to
  update their logging configuration to take full advantage of the new
  structure.

Neutral:

- This ADR does not mandate a particular log transport (files, stdout,
  centralised logging), but it encourages structured logging and domain-based
  routing through configuration.

## Adoption & Migration

We adopt the following phased migration plan, aligned with release milestones:

- **v0.10.2 (initial enforcement):**
  - Introduce the domain-based logger hierarchy and shared logging context
    helper, focusing on core explain/predict and plugin orchestration paths.
  - Classify existing logging call sites into `core.*`, `plugins.*`,
    `telemetry.*`, and `governance.*` domains, adjusting logger names where
    necessary.
  - Wire governance-relevant events (e.g. plugin trust/deny, filter strict
    observability warnings that impact governance) to `governance.*` loggers
    where appropriate.
  - Ratify Standard-005 and update contributor documentation to reference this
    ADR.

- **v0.11.0 and v1.0.0-rc:**
- Extend context propagation to checkpoints, rollback, and any additional
  governance surfaces introduced by downstream extensions or governance-related
  features.
  - Finalise helper configuration APIs and example logging configurations (text
    vs JSON) for production deployments.
  - Update release gates so that governance logs are treated as part of the
    audit evidence for relevant ADRs and standards.

Migration guidelines:

- New code MUST follow the hierarchy and context rules described here and in
  Standard-005.
- Existing loggers MAY be left as-is initially but should be migrated when
  touched for other work or as part of targeted uplift tasks.
- When logger names or message formats used by operators are changed, changes
  SHOULD be documented in release notes with a short migration note.

## Open Questions

- Should the project provide a default JSON logging configuration for governance
  logs (e.g. a recommended formatter) or leave all formatting decisions to host
  applications, with only examples in documentation?
- How much of the governance log stream should be mirrored into any future
  dedicated audit ledger, and what consistency guarantees will that ledger
  need?
- Should diagnostic modes that log more detailed data (e.g. summaries of input
  distributions) be standardised now, or kept as internal tooling until
  governance requirements demand formalisation?

## Related Documents

- `docs/standards/STD-005-logging-and-observability-standard.md` — Contributor
  and operator standard that implements this ADR’s architectural decisions.
- `docs/foundations/governance/optional_telemetry.md` — Existing telemetry
  guidance; should be updated to reference the new logging hierarchy and
  context helper where appropriate.
- `docs/improvement/ignore/Logging_Analysis.md` — Detailed analysis of current
  logging usage and recommendations that informed this ADR.
