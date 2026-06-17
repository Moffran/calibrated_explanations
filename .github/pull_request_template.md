# Pull Request

## Summary

- What does this PR change? Why?

## Checklist

- [ ] Reviewed the `## Release preparation` section in the active `development/current-work/vX.Y.Z_plan.md` (replaces standalone release checklist)
- [ ] Aligned with current phase in `development/current-work/RELEASE_PLAN_v1.md` (link section)
- [ ] Referenced relevant ADR(s) in `development/adrs/` (IDs)
- [ ] Added/updated tests for new or changed behavior
- [ ] Coverage gate passes (`pytest --cov=src/calibrated_explanations --cov-config=pyproject.toml --cov-fail-under=90`)
- [ ] Coverage waiver requested (if needed) with linked issue: <!-- paste issue URL or write N/A -->
- [ ] Legacy API parity verified (if wrapper/API changed) - ref ADR-020
- [ ] mypy passes for touched modules (and strict for new core modules)
- [ ] Ruff and Markdown lint pass locally
- [ ] Updated docs/README if public behavior or user flows changed
- [ ] Considered backward compatibility and deprecation notes
- [ ] DCO sign-off added to commits (`git commit -s`)
- [ ] Reviewed CODE_OF_CONDUCT.md / SECURITY.md / GOVERNANCE.md as needed

## Screenshots/Notes (optional)

- Any UI/log/output snippets that help reviewers

## Breaking changes (if any)

- [ ] Documented in CHANGELOG with migration notes
