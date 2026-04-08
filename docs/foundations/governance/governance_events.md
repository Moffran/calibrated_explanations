# Governance Events (Plugin + Config Lifecycle)

This page defines the machine-checkable governance event contract used for
plugin registration/discovery decisions and ConfigManager lifecycle events.

## Scope

The contracts apply to runtime governance paths under
`calibrated_explanations.governance.*` loggers.

Plugin decision events (`calibrated_explanations.governance.plugins`) cover:

- `accepted_registration`
- `skipped_untrusted`
- `skipped_denied`
- `checksum_failure`
- `denied_registration`

Config lifecycle events (`calibrated_explanations.governance.config`) cover:

- `resolve`
- `export`
- `validation_failure`

## Event envelope (v1)

Plugin schema file:
`src/calibrated_explanations/schemas/governance_event_schema_v1.json`

Required fields:

- `schema_version`
- `event_id`
- `event_name`
- `decision`
- `identifier`
- `source`
- `trusted`
- `actor`
- `timestamp`

Optional audit/context fields:

- `provider`
- `reason_code`
- `reason`
- `invocation_id`
- `request_id`
- `tenant_id`
- `plugin_identifier`
- `details`

Config schema file:
`src/calibrated_explanations/schemas/governance_config_event_schema_v1.json`

Required config fields:

- `schema_version`
- `event_id`
- `event_name` (`config.lifecycle`)
- `event_type` (`resolve` | `export` | `validation_failure`)
- `profile_id`
- `config_schema_version`
- `strict`
- `source_count`
- `validation_issue_count`
- `timestamp`

Optional config field:

- `details`

## Example payload

```json
{
  "schema_version": "1.0",
  "event_id": "4f35b20f-3558-4f31-b6fc-04c70f9975f0",
  "event_name": "plugin.registration.decision",
  "decision": "skipped_denied",
  "identifier": "third.party.interval",
  "provider": "third-party-plugin",
  "source": "entrypoint",
  "trusted": false,
  "actor": "load_entrypoint_plugins",
  "reason_code": "denylist",
  "reason": "Plugin identifier is denied via CE_DENY_PLUGIN",
  "timestamp": "2026-03-19T12:34:56.789012+00:00"
}
```

## Example config lifecycle payloads

```json
{
  "schema_version": "1.0",
  "event_id": "6106f0aa-2e5d-4a01-a8e4-68a3d4bcbac8",
  "event_name": "config.lifecycle",
  "event_type": "resolve",
  "profile_id": "default",
  "config_schema_version": "1",
  "strict": true,
  "source_count": 4,
  "validation_issue_count": 0,
  "timestamp": "2026-04-08T10:00:00+00:00",
  "details": null
}
```

```json
{
  "schema_version": "1.0",
  "event_id": "ca594651-8d37-4ddd-a071-bb2bd6f74087",
  "event_name": "config.lifecycle",
  "event_type": "export",
  "profile_id": "default",
  "config_schema_version": "1",
  "strict": true,
  "source_count": 4,
  "validation_issue_count": 0,
  "timestamp": "2026-04-08T10:00:05+00:00",
  "details": {
    "diagnostic_only": true
  }
}
```

```json
{
  "schema_version": "1.0",
  "event_id": "1ea42ce4-fb09-4d40-8ca8-129ca109c6ac",
  "event_name": "config.lifecycle",
  "event_type": "validation_failure",
  "profile_id": "default",
  "config_schema_version": "1",
  "strict": false,
  "source_count": 2,
  "validation_issue_count": 1,
  "timestamp": "2026-04-08T10:00:09+00:00",
  "details": {
    "location": "pyproject.plugins",
    "issue_count": 1
  }
}
```

## Enterprise ingest mapping

For enterprise governance pipelines that use `ce.*` namespaced telemetry fields,
the following mapping is recommended:

- `decision` -> `ce.governance.decision`
- `identifier` -> `ce.governance.plugin_identifier`
- `source` -> `ce.governance.source`
- `actor` -> `ce.governance.actor`
- `reason_code` -> `ce.governance.reason_code`
- `event_id` -> `ce.governance.event_id`
- `timestamp` -> `ce.governance.timestamp`

## Validation and checks

- Runtime event payloads are validated by
  `calibrated_explanations.governance.events.validate_governance_event`.
- Config lifecycle payloads are validated by
  `calibrated_explanations.governance.events.validate_config_governance_event`.
- CI/local quality gate:
  `python scripts/quality/check_governance_event_schema.py`.

## Config lifecycle notes

- `resolve` events are emitted by `ConfigManager.from_sources()` only.
- Direct `ConfigManager(...)` construction for injection/testing is silent for
  `resolve` events.
- `export` events are diagnostic-only until ADR-034 Open Item 2 is closed.
- `validation_failure` events are emitted in both strict and non-strict
  validation paths.

## Data minimization constraints

Governance events must not include raw feature vectors, labels, predictions, or
full plugin metadata blobs. Keep payloads limited to identifiers, decision
metadata, and reason fields needed for audit reconstruction.

Config lifecycle payloads must not include raw config values, environment
strings, or raw pyproject payload blobs. In particular, `validation_failure`
details are restricted to location and issue-count metadata (see ADR-034 Open
Item 1 in `docs/improvement/adrs/ADR-034-centralized-configuration-management.md`).

Entry-point tier: Tier 3
