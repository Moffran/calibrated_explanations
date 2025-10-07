# Pull Request

## Summary

- What does this PR change? Why?

## Checklist

- [ ] Aligned with current phase in `improvement_docs/ACTION_PLAN.md` (link section)
- [ ] Referenced relevant ADR(s) in `improvement_docs/adrs/` (IDs)
- [ ] Added/updated tests for new or changed behavior
- [ ] Coverage gate passes (`pytest --cov=src/calibrated_explanations --cov-config=.coveragerc --cov-fail-under=80`)
- [ ] Coverage waiver requested (if needed) with linked issue: <!-- paste issue URL or write N/A -->
- [ ] mypy passes for touched modules (and strict for new core modules)
- [ ] Ruff and Markdown lint pass locally
- [ ] Updated docs/README if public behavior or user flows changed
- [ ] Considered backward compatibility and deprecation notes

## Screenshots/Notes (optional)

- Any UI/log/output snippets that help reviewers

## Breaking changes (if any)

- [ ] Documented in CHANGELOG with migration notes
