# How to contribute

I'm really glad you're reading this, because we need volunteer developers to help this project come to fruition.

Since we are now open to contributions, we welcome your feedback, suggestions, and contributions. You can contribute by opening a pull request, filing an issue, or initiating discussions in our [discussion forum](https://github.com/Moffran/calibrated_explanations/discussions). We value your input and look forward to collaborating with you.

## Roadmap and ADR-driven development

This project follows a written release plan and Architecture Decision Records (ADRs):

- Release Plan: see `improvement_docs/RELEASE_PLAN_v1.md`. It defines the remaining milestones and gates on the path to v1.0.0. Please align PRs with the current milestone and its scope.
- ADRs: see `improvement_docs/adrs/`. If your change affects architecture, public API, serialization schema, or cross-cutting behavior, add/update an ADR (status `Proposed` → `Accepted` on merge).

Current highlights coming from reported issues and the release plan:

- Explanation storage redesign (internal domain model with rule objects), tracked in ADR-008.
- Native non-numeric input support in the wrapper (preprocessing + mapping persistence), tracked in ADR-009.

Prefer small, focused PRs that map to the plan’s daily/weekly slices (e.g., `feat/1b-exceptions`, `feat/1b-validation`).


## Feature requests

File a new feature request by opening an issue and using the feature request template. Please make sure that the feature request is not already listed in the [enhancement issues](https://github.com/Moffran/calibrated_explanations/labels/enhancement).


## Bug reports

File a new bug report by opening an issue and using the bug report template. Please make sure that the bug is not already listed in the [bug issues](https://github.com/Moffran/calibrated_explanations/labels/bug).


## Pull requests

Please send pull requests through the
[PR tracker on GitHub](https://github.com/Moffran/calibrated_explanations/pulls).
Include tests to ensure your contribution is compatible with the tested use cases.
We have CI set up,
so watch out for the automated test results.

PR expectations:

- Keep changes scoped to a single slice/milestone. Write a brief checklist in the PR description referencing the relevant release-plan milestone.
- Add/adjust tests: unit tests for new modules/paths, and keep golden/API snapshot tests unchanged unless the release plan explicitly calls for a public change (then update snapshots intentionally).
- Quality gates should pass: ruff lint/format, pytest, and mypy. New modules may be subject to stricter mypy settings (see `pyproject.toml`).
- If touching performance-sensitive paths, run or reference the perf guard and baseline scripts in `benchmarks/` and `scripts/`.
- For architectural/public changes, include/modify an ADR in `improvement_docs/adrs/` and link it in the PR.


## Testing and Code Coverage

We use pytest as our testing framework and aim to achieve a code coverage of about 90% in our tests. This ensures that our code is thoroughly tested and helps identify any potential issues or bugs. We encourage contributors to write comprehensive tests and strive for high code coverage. Code coverage tests are added and monitored at [Codecov](https://app.codecov.io/github/Moffran/calibrated_explanations).

Additional checks:

- Linting via ruff (style and simple correctness rules).
- Type checking via mypy. During Phase 1B, new core modules (e.g., `core/exceptions.py`, `core/validation.py`) are checked with stricter settings.
- Performance guard: see `scripts/check_perf_regression.py` and `benchmarks/perf_thresholds.json`.

## Naming and documentation style quick reference

ADR-017 and ADR-018 define the internal style rules that keep the plugin-first
architecture consistent. The cheat-sheet below summarises what reviewers expect
in day-to-day contributions; consult the ADRs for the full context and
motivation.【F:improvement_docs/adrs/ADR-017-nomenclature-standardization.md†L1-L37】【F:improvement_docs/adrs/ADR-018-code-documentation-standard.md†L1-L62】

### Naming conventions (ADR-017)

- **Modules and packages**: use `snake_case` filenames (e.g. `calibration_helpers.py`).
  Transitional import shims live under a `legacy/` namespace or carry a
  `deprecated_*.py` prefix.
- **Classes**: keep `PascalCase` and add suffixes that clarify scope when useful
  (e.g. `...Helper`, `...Mixin`).
- **Functions and attributes**: `snake_case`, with boolean helpers prefixed by
  verbs such as `is_`/`has_`. Avoid introducing new double-underscore attributes
  outside compatibility shims.
- **Plugin identifiers and schema keys**: prefer dot-delimited lowercase names
  (`core.explanation.factual`). Document aliases and flag deprecations clearly.

### Documentation conventions (ADR-018)

- **Docstrings**: adopt numpydoc sections (`Parameters`, `Returns`, `Raises`)
  for public functions, methods, and classes. Summaries must start with an
  imperative sentence and fit on one line where possible.
- **Module docs**: each module should start with a short prose summary outlining
  its primary responsibility and noteworthy shims or legacy behaviours.
- **Examples and references**: prefer runnable snippets; link to canonical
  helper utilities instead of duplicating logic in prose.
- **Coverage expectations**: when adding new modules, include docstrings with
  at least the sections above so automated docstring coverage tools can report
  progress accurately.

## Plugin tooling quickstart

Plugin development relies on the shared registry introduced in ADR-006/ADR-013/ADR-015.
Two practical helpers when working on plugins:

- Use the `ce.plugins` console script (packaged via `pyproject.toml`) to inspect
  registered explanation/interval/plot plugins and their trust state:
  `ce.plugins list all`.
- Inline smoke tests live under `tests/integration/plugins/test_cli_smoke.py`.
  Keep them green when adding new commands or metadata fields so the CLI output
  remains stable for operators.
