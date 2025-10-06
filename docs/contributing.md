# Contributing and Roadmap

Contributions are welcome. Please see the repository’s `CONTRIBUTING.md` for the full guide.

## Roadmap-driven development

We are updating the package according to a written release plan and ADRs:

- Release Plan: `improvement_docs/RELEASE_PLAN_v1.md` tracks milestone scope and readiness gates on the way to v1.0.0.
- ADRs: `improvement_docs/adrs/` contains accepted and proposed architectural decisions.

When opening a PR, please align with the active milestone in the release plan and reference the relevant ADRs.

## Quality gates (summary)

- Type checks: mypy for new/modified core modules.
- Linting: ruff and markdownlint.
- Tests: add unit tests for new behavior; keep runtime reasonable.
- Docs: update README/docs when public behavior changes.

## Style excerpts (ADR-017 & ADR-018)

Naming and documentation guidelines originate from the accepted ADRs and are
summarised here for quick reference. Refer to the ADRs for detailed rationale
and future migration phases.【F:improvement_docs/adrs/ADR-017-nomenclature-standardization.md†L1-L37】【F:improvement_docs/adrs/ADR-018-code-documentation-standard.md†L1-L62】

### Naming essentials

- Modules/packages: snake_case filenames; transitional shims should either live
  under `calibrated_explanations.legacy` or use a `deprecated_*.py` prefix.
- Classes: PascalCase with descriptive suffixes when scope is not obvious.
- Functions/attributes: snake_case and verb-prefixed booleans (`is_`, `has_`).
- Plugin identifiers & schema keys: dot-separated lowercase strings
  (`core.explanation.fast`). Keep deprecation notices alongside aliases.

### Documentation essentials

- Public callables: numpydoc-formatted docstrings with `Parameters`, `Returns`,
  and `Raises` (when applicable) sections.
- Modules: leading summary paragraph describing responsibilities and legacy
  compatibility shims that remain in place.
- Examples: prefer runnable snippets and link to helper APIs rather than
  repeating logic inline.
- Coverage: new modules should ship with docstrings so coverage tooling can
  report progress and CI lint jobs can highlight regressions.
