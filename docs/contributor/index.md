# Contributor hub

Help shape calibrated explanations by improving the core library, documentation,
and plugin ecosystem. Start with the plugin contract so every change preserves
classification and regression parity, then follow the extending guides to wire
CLI, configuration, and documentation updates.

## Core contributor path

1. Review the {doc}`plugin-contract` to understand the surfaces that must remain
   calibrated-first across explanations, intervals, and PlotSpec outputs.
2. Study the {doc}`extending/plugin-advanced-contract` for advanced plugin wiring
   methods beyond basic parameter passing.
3. Follow the {doc}`../foundations/governance/test_policy` so every Python
   example added to the docs or README gains a matching test in `tests/docs/`.
4. Consult the [Centralized Test Guidance](https://github.com/Moffran/calibrated_explanations/blob/main/.github/tests-guidance.md) for unit and integration testing standards, patterns, and anti-patterns.
5. Set up your environment using the workflow in {doc}`../foundations/governance/pr_guide` and run
   the formatter, tests, and doc builds locally.
6. Use the {doc}`extending/guides/index` collection when adding new plugins or
   command-line tooling.
7. When upgrading between releases, consult the {doc}`../migration/index` for
   terminology changes, breaking changes, and migration guidance.

## Local CI pre-checks

We provide a helper script to inspect and optionally run GitHub Actions
workflow `run:` steps locally. This is handy to catch lint/doc/coverage issues
before opening a PR. The helper lives at `scripts/run_ci_locally.py` and is
invoked like:

```pwsh
# Dry-run (lists runnable steps discovered in .github/workflows)
python scripts/run_ci_locally.py --dry-run

# Run the lint workflow (PowerShell on Windows):
python scripts/run_ci_locally.py --shell pwsh --workflow lint

# Run lint+test under bash (use WSL/Git-Bash on Windows when workflows use bash)
python scripts/run_ci_locally.py --shell bash --workflow lint --workflow test
```

There is a `make ci-local` Makefile target that calls the helperfor a full pre-PR check. The helper intentionally skips `uses:` steps
(GitHub Actions) — ensure any required setup (Python versions, checkout,
dependencies) is present in your environment before executing steps.

## Advanced contributor tooling

Need telemetry hooks or performance instrumentation for plugins? The
{doc}`extending/advanced` page summarises the optional extras and how they tie
back to governance requirements.

```{toctree}
:maxdepth: 1
:hidden:

plugin-contract
extending/plugin-advanced-contract
extending/guides/index
extending/advanced
../migration/index
```
