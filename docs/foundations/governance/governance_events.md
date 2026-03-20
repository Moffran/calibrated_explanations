# Governance Events (Plugin Decisions)

This page defines the machine-checkable governance event contract used for
plugin registration and discovery decisions.

## Scope

The event contract applies to runtime plugin decision paths under
`calibrated_explanations.governance.*` loggers and covers:

- `accepted_registration`
- `skipped_untrusted`
- `skipped_denied`
- `checksum_failure`
- `denied_registration`

## Event envelope (v1)

Schema file:
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
- CI/local quality gate:
  `python scripts/quality/check_governance_event_schema.py`.

## Data minimization constraints

Governance events must not include raw feature vectors, labels, predictions, or
full plugin metadata blobs. Keep payloads limited to identifiers, decision
metadata, and reason fields needed for audit reconstruction.

Entry-point tier: Tier 3
