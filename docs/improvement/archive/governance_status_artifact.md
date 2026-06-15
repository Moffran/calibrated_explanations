# Governance Status Artifact

## Overview

`reports/governance/governance_status.json` is a **derived CI artifact** that
aggregates quality check results from multiple gate scripts into a single
machine-readable summary. It is produced by
`scripts/quality/build_governance_status_artifact.py` and validated against
`docs/improvement/schemas/governance_status_schema_v1.json`.

> **Important:** This is NOT a runtime governance event schema. It does not
> replace `governance_event_schema_v1.json` or
> `governance_config_event_schema_v1.json` in `src/calibrated_explanations/schemas/`.

---

## Payload structure

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-04-22T10:00:00+00:00",
  "run": {
    "workflow": null,
    "run_id": null,
    "commit": null
  },
  "lint": {
    "local_checks_pr": "unavailable",
    "mypy": "unavailable",
    "ruff": "unavailable"
  },
  "schema_checks": {
    "governance_event_schema": "passed",
    "config_manager_usage": "passed",
    "logging_domains": "passed",
    "no_local_paths": "passed"
  }
}
```

### Fields

| Field | Type | Description |
| --- | --- | --- |
| `schema_version` | `string` | Always `"1.0"`. |
| `generated_at` | `string` | ISO-8601 UTC timestamp of artifact generation. |
| `run.workflow` | `string\|null` | GitHub Actions workflow name; `null` when run locally. |
| `run.run_id` | `string\|null` | GitHub Actions run ID; `null` when run locally. |
| `run.commit` | `string\|null` | Git commit SHA; `null` when run locally. |
| `lint.*` | `"passed"\|"failed"\|"unavailable"` | Lint gate results passed via CLI flags. |
| `schema_checks.*` | `"passed"\|"failed"\|"unavailable"` | Status derived from quality report files. |

### `schema_checks` input sources

| Key | Source report file | Status derivation |
| --- | --- | --- |
| `governance_event_schema` | `reports/quality/governance_event_schema_report.json` | `ok` field |
| `config_manager_usage` | `reports/config_manager_usage_report.json` | `total_violations == 0` |
| `logging_domains` | `reports/quality/logging_domain_report.json` | `ok` or `total_violations == 0` |
| `no_local_paths` | `reports/quality/no_local_paths_report.json` | `ok` field |

If a report file is absent, the corresponding key is set to `"unavailable"`.

---

## Local generation

```bash
python scripts/quality/build_governance_status_artifact.py \
    --output reports/governance/governance_status.json --validate
```

The `schema_checks` section is populated from the quality report files produced
by previous quality check steps. Run the gate scripts first to get up-to-date
report files:

```bash
python scripts/quality/check_governance_event_schema.py
python scripts/quality/check_config_manager_usage.py --scope targeted --report reports/config_manager_usage_report.json
python scripts/quality/check_logging_domains.py --root src/calibrated_explanations --report reports/quality/logging_domain_report.json
python scripts/quality/check_no_local_paths_in_reports.py --check --report reports/quality/no_local_paths_report.json
```

---

## CI wiring (manual step)

To publish the artifact from GitHub Actions, add the following step to
`.github/workflows/ci.yml` **after** the lint and test steps:

```yaml
- name: Build governance status artifact
  run: |
    python scripts/quality/build_governance_status_artifact.py \
      --output reports/governance/governance_status.json \
      --validate \
      --lint-local-checks-pr ${{ steps.lint.outcome == 'success' && 'passed' || 'failed' }} \
      --lint-mypy ${{ steps.mypy.outcome == 'success' && 'passed' || 'failed' }} \
      --lint-ruff ${{ steps.ruff.outcome == 'success' && 'passed' || 'failed' }}

- name: Upload governance status artifact
  uses: actions/upload-artifact@v4
  with:
    name: governance-status-${{ github.run_id }}
    path: reports/governance/governance_status.json
    retention-days: 30
  if: always()
```

> **Note:** Step IDs (`steps.lint`, `steps.mypy`, `steps.ruff`) must match your
> actual workflow step `id:` fields. Adjust accordingly.

---

## Schema validation

The artifact schema is at `docs/improvement/schemas/governance_status_schema_v1.json`.
It uses JSON Schema draft 2020-12 with `additionalProperties: false` at all
object boundaries for governance hardening.

Validate locally:

```bash
python scripts/quality/build_governance_status_artifact.py --validate
```

---

## Governing references

- ADR-028: Logging and Governance Observability (evidence surface)
- ADR-034: Centralized Configuration Management (config_manager_usage source)
- ADR-035: CI Workflow Governance (artifact publication rules)
