# How to contribute

I'm really glad you're reading this, because we need volunteer developers to help this project come to fruition.

Since we are now open to contributions, we welcome your feedback, suggestions, and contributions. You can contribute by opening a pull request, filing an issue, or initiating discussions in our [discussion forum](https://github.com/Moffran/calibrated_explanations/discussions). We value your input and look forward to collaborating with you.

## Community health references

- Code of Conduct: see `CODE_OF_CONDUCT.md`.
- Security reporting: see `SECURITY.md` for private reporting channels.
- Governance and support: see `GOVERNANCE.md` and `SUPPORT.md`.

## Roadmap and ADR-driven development

- Roadmap: see `ROADMAP.md` for a high-level summary of the current release pillars and the implementation plan that supports them.
- Release Plan: see `docs/improvement/RELEASE_PLAN_v1.md`. It defines the remaining milestones and gates on the path to v1.0.0. Please align PRs with the current milestone and its scope.
- Milestone Plan: see `vx.y.z_plan.md` (e.g., `v0.10.1_plan.md`) for detailed checklists that map to each release plan milestone.
- ADRs: see `docs/improvement/adrs/`. If your change affects architecture, public API, serialization schema, or cross-cutting behavior, add/update an ADR (status `Proposed` → `Accepted` on merge).
- Documentation: Follow [STD-027](docs/standards/STD-027-documentation-audience-standard.md) for all documentation structure and audience guidelines.

Current highlights coming from reported issues and the release plan:

- Explanation storage redesign (internal domain model with rule objects), tracked in ADR-008.
- Native non-numeric input support in the wrapper (preprocessing + mapping persistence), tracked in ADR-009.

Prefer small, focused PRs that map to the plan’s daily/weekly slices (e.g., `feat/1b-exceptions`, `feat/1b-validation`).

## Contribution licensing (DCO)

We use the Developer Certificate of Origin (DCO) to confirm that contributions
are licensed under the project's BSD-3-Clause license. By contributing, you
agree that your work is original or that you have the right to submit it.

Please add a sign-off line to every commit:

```\nSigned-off-by: Your Name <your.email@example.com>\n```

You can add this automatically with `git commit -s`.


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
- If touching performance-sensitive paths, run or reference the perf guard and baseline scripts in `tests/benchmarks/` and `scripts/`.
- For architectural/public changes, include/modify an ADR in `docs/improvement/adrs/` and link it in the PR.
- **Legacy API compatibility**: Changes to `WrapCalibratedExplainer`, `CalibratedExplainer`, or explanation collection methods must preserve the documented contract in `docs/improvement/legacy_user_api_contract.md`. The PR template includes a parity review checkpoint; regression tests in `tests/unit/api/test_legacy_user_api_contract.py` enforce this guardrail (ADR-020).

### Dependency constraints

CI and local dev workflows use `constraints.txt` to pin key dependencies. When
installing dev or doc dependencies locally, add the constraint file:

```bash
python -m pip install -e .[dev] -c constraints.txt
python -m pip install -r docs/requirements-doc.txt -c constraints.txt
```



## Testing and Code Coverage

Pytest remains the primary regression harness. CI now enforces the ADR-019 coverage gate via `make test-cov`, which simply runs `pytest` with coverage flags supplied by `pytest.ini` (`--cov=src/calibrated_explanations --cov-config=.coveragerc --cov-report=term-missing --cov-report=xml --cov-fail-under=90`). The shared `.coveragerc` applies locally as well, so invoking `make test-cov` reproduces the CI behaviour (including generation of `coverage.xml` for Codecov uploads). Coverage results are uploaded to [Codecov](https://app.codecov.io/github/Moffran/calibrated_explanations) for historical tracking, and Codecov enforces ≥90% patch coverage on runtime/calibration paths per the v0.10.0 release plan.

If a change cannot practically meet the 90% package-wide bar (for example, because it touches legacy shims slated for removal), request a coverage waiver:

1. File an issue describing why the threshold cannot be met and outline the follow-up work required.
2. Reference that issue in your pull request description and tick the waiver checkbox in the PR template.
3. Add a brief note in the changelog entry or summary so reviewers can evaluate the trade-off.

Waivers are exceptional and should include a plan with owners/dates so the debt does not linger past the next release.

Additional checks:

- Linting via ruff (style and simple correctness rules).
- Naming guardrails: run `ruff check --select N` locally to preview the CI naming warnings introduced for the v0.7.0 release gate (ADR-017).
- Docstring guardrails: run `pydocstyle --convention=numpy src tests` to surface ADR-018 numpydoc issues; CI runs the same checks in blocking mode, so fix violations before opening a pull request.
- Docstring coverage: `python scripts/check_docstring_coverage.py` prints the current module/class/function/method coverage mix and accepts `--fail-under` for teams that want to experiment with stricter thresholds.
- Type checking via mypy. During Phase 1B, new core modules (e.g., `core/exceptions.py`, `core/validation.py`) are checked with stricter settings.
- Performance guard: see `scripts/check_perf_regression.py` and `tests/benchmarks/perf_thresholds.json`.

## Naming and documentation style quick reference

ADR-017 and ADR-018 define the internal style rules that keep the plugin-first
architecture consistent. The cheat-sheet below summarises what reviewers expect
in day-to-day contributions; consult the ADRs for the full context and
motivation.【F:docs/improvement/adrs/ADR-017-nomenclature-standardization.md†L1-L37】【F:docs/improvement/adrs/ADR-018-code-documentation-standard.md†L1-L62】

### Naming conventions (ADR-017)

| Scope | Requirements | Common pitfalls |
| --- | --- | --- |
| Modules & packages | `snake_case` filenames; transitional shims live in `legacy/` or start with `deprecated_` | CamelCase filenames, silently keeping duplicate module aliases |
| Classes | `PascalCase` with clarifying suffixes when scope is narrow (`...Helper`, `...Mixin`) | Reusing ambiguous names such as `Manager`, `Wrapper` without context |
| Functions & attributes | `snake_case`; boolean values begin with verbs (`is_`, `has_`, `should_`) | Introducing new double-underscore names or mismatching helper prefixes |
| Registry identifiers | Dot-delimited lowercase paths (`core.explanation.factual`) | Forgetting to document aliases when keeping backward-compatible IDs |

### Documentation conventions (ADR-018)

| Area | Minimum expectation | Notes |
| --- | --- | --- |
| Modules | One-paragraph summary describing primary responsibility and notable shims | Keep in sync with ADR names so readers can trace provenance |
| Public callables | Full numpydoc sections (`Parameters`, `Returns`, `Raises`, `Examples` as appropriate) | Summaries start with an imperative verb and fit on one line |
| Internal helpers (`_` prefix) | Single-line summary explaining purpose | Still counted in coverage; these should clarify side-effects/constraints |
| Deprecations | `.. deprecated::` directive or explicit `Warnings` section | Reference the replacement module or helper |
| Coverage tracking | Run `python scripts/check_docstring_coverage.py` before requesting review | Pair with `pydocstyle` output to keep ADR-018 coverage metrics honest |

## Plugin tooling quickstart

Plugin development relies on the shared registry introduced in ADR-006/ADR-013/ADR-015.
Two practical helpers when working on plugins:

- Use the `ce.plugins` console script (packaged via `pyproject.toml`) to inspect
  registered explanation/interval/plot plugins and their trust state:
  `ce.plugins list all`.
- Inline smoke tests live under `tests/integration/plugins/test_cli_smoke.py`.
  Keep them green when adding new commands or metadata fields so the CLI output
  remains stable for operators.

## Local CI pre-checks

We provide a helper to preview and run the repository's CI steps locally. This
is useful to catch CI failures (lint, doc checks, coverage gates, etc.) before
opening a pull request.

- Dry-run (lists CI steps discovered from `.github/workflows`):

  ```pwsh
  python scripts/run_ci_locally.py --dry-run
  ```

- Run a specific workflow (for example `lint` and `test`) using your native
  shell. On Windows the script defaults to PowerShell; on Unix it defaults to
  bash.

  ```pwsh
  # PowerShell on Windows
  python scripts/run_ci_locally.py --shell pwsh --workflow lint --workflow test

  # Bash (recommended when CI steps use bashisms; use WSL/Git-Bash on Windows)
  python scripts/run_ci_locally.py --shell bash --workflow lint --workflow test
  ```

- There is also a Makefile target `ci-local` that invokes the helper in dry-run
  mode for a quick check:

  ```pwsh
  make ci-local
  ```

Notes and safety:

- The helper extracts `run:` blocks from workflows and skips `uses:` steps
  (GitHub Actions) — you must ensure any required setup (Python versions,
  checkout, secrets) is provided locally before executing steps.
- Some CI steps install dependencies and run tests — expect potentially slow
  runs when executing full `test` workflow. Use `--workflow lint` first for a
  quicker pre-check.
- The script defaults to a dry-run; only run commands when you're ready and
  understand their effects.
