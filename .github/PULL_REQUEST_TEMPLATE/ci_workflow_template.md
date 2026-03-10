# CI Workflow Change PR

## Summary

- What CI behavior is changing and why?

## CI Governance Checklist (ADR-035)

- [ ] I used one of the approved reusable workflows OR marked this workflow `experimental: true` with expiry YYYY-MM-DD.
- [ ] Pip installs in workflow use `-c constraints.txt` (or explanation attached).
- [ ] Job `permissions` default to `contents: read` and any write escalation is limited to maintenance workflows with justification.
- [ ] Heavy job(s) are path-gated, scheduled, or `workflow_dispatch` present.
- [ ] `scripts/local_checks.py` / `Makefile` updated for local reproduction where required.
- [ ] I ran `make local-checks-pr` and recorded results.
- [ ] PR contains an audit block: `owner`, `reason`, `expected_impact`, `local_repro_steps`, `expiry/next_review`.
- [ ] I added `ci:workflow` or `ci:cleanup` label as appropriate and requested CODEOWNER review.

## Audit Metadata

- owner:
- reason:
- expected_impact:
- local_repro_steps:
- expiry/next_review:
