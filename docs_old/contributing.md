# Contributing and Roadmap

Contributions are welcome. This page distills the day-to-day workflow, tooling,
and governance guardrails so you can get productive quickly. For the complete
policy and code of conduct, read the repository’s {file}`../CONTRIBUTING.md`.

## Local development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pip install -r docs/requirements-doc.txt
```

The `dev` extra installs pytest, mypy, ruff, nbqa, and other quality gates used
in continuous integration. The docs requirements enable `make -C docs html`
without missing extensions.

## Required checks before you push

Run the same commands that CI executes:

```bash
pytest                        # unit and integration tests
ruff check .                  # linting and import order
mypy src tests                # type checks (targeting Python 3.11)
nbqa ruff notebooks           # optional, keeps notebooks linted
make -C docs html             # optional, confirms Sphinx builds cleanly
```

Document any skipped steps in the pull-request description so reviewers can
triage quickly.

## Roadmap-driven development

We are updating the package according to a written release plan and ADRs:

- Release Plan: `improvement_docs/RELEASE_PLAN_v1.md` tracks milestone scope and readiness gates on the way to v1.0.0.
- ADRs: `improvement_docs/adrs/` contains accepted and proposed architectural decisions.

When opening a PR, please align with the active milestone in the release plan and reference the relevant ADRs.

## Quality gates (summary)

- Type checks: `mypy src tests` for new/modified modules.
- Linting: `ruff check .` for Python, `markdownlint` for docs (see
  {file}`../.markdownlint.json`).
- Tests: add unit tests for new behaviour; keep runtime reasonable and note any
  slow suites.
- Docs: update README/docs when public behaviour changes and rebuild Sphinx to
  confirm there are no warnings.

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

## Plugin guardrails & denylist controls

- Review `docs/plugins.md` before touching the registry. It links directly to
  ADR-024/025/026 and provides the minimal calibrated plugin scaffold.
- Use `CE_DENY_PLUGIN` during development to block plugin identifiers without
  modifying code. CLI discovery labels denied plugins so you can confirm the
  toggle took effect.
- External plugins (for example FAST explanations) live under
  `external_plugins/` and are installed via `pip install
  "calibrated-explanations[external-plugins]"` followed by an explicit
  `register()` call.
