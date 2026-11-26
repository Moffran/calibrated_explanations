# CE-REQ-DOCS-GOV-001 - Documentation Build Policy Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-DOCS-GOV-001 |
| obligation_type | quality_gate |
| claim_refs | CE-CAP-DOCS-001 |
| adr_refs | ADR-012 |
| status | active |
| verification_status | verified |

## Scope

ADR-012 documentation and gallery policy for maintained navigation, upgrade docs, agent guide correctness, and notebook execution driver behavior.

## Observable behavior

- Top-level documentation toctree targets exist.
- Upgrade and maintenance documentation pages referenced by navigation exist.
- Active agent instructions, skills, user documentation, notebooks, examples and CE source do not reference the deprecated `ce_agent_utils` module outside formal deprecation, historical and temporary compatibility records.
- Notebook execution policy emits deterministic reports and fails in blocking mode for execution errors.

## Acceptance criterion

- Documentation navigation tests pass for top-level, maintenance, and upgrade pages.
- The active-guidance scan finds no `ce_agent_utils` reference outside its explicit allow-list, and every allow-list entry still exists.
- Notebook driver tests pass for successful execution, blocking-mode failures, deterministic reporting, and timeout/error reporting.

## Verification method

Automated pytest tests for documentation navigation, CE-first guide policy, and notebook execution policy.

## Verification targets

- pytest: tests/docs/test_navigation.py::test_top_level_toctree_targets_exist
- pytest: tests/docs/test_navigation.py::test_upgrade_docs_exist
- pytest: tests/docs/get_started/test_no_ce_agent_utils_recommendation.py::test_active_guidance_and_workflows_do_not_reference_deprecated_module
- pytest: tests/docs/test_notebook_driver.py::TestRunNotebooksReport::test_should_emit_report_with_all_required_fields

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies executable docs policy checks. It does not prove prose quality or every rendered ReadTheDocs page outside the checked navigation and notebook-policy surfaces.
