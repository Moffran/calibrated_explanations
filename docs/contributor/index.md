# Contributor hub

{{ hero_calibrated_explanations }}

Help shape calibrated explanations by improving the core library, documentation, and plugin ecosystem. This hub curates the onboarding guides, coding standards, and quality gates needed to land changes safely.

## Get set up quickly

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pip install -r docs/requirements-doc.txt
```

Run the same checks that back the continuous-integration gates:

```bash
pytest
ruff check .
mypy src tests
make -C docs html  # optional but catches doc regressions
```

## Start contributing

- {doc}`../contributing` – Contribution guidelines, coding standards, and review expectations.
- {doc}`../extending/index` – Architecture notes for plugins, registries, and calibration backends.
- {doc}`../reference/index` – API reference entry point for the calibration and explanation objects.

## Plan and scope work

- {doc}`../governance/nav_crosswalk` – Documentation IA map to keep role-based navigation aligned.
- {doc}`../governance/release_checklist` – Release quality gates, including required doc fragments and quickstart smoke tests.
- {doc}`../overview/index` – Overview narrative that keeps messaging aligned across issues and PRs.

## Keep quality high

- {doc}`../pr_guide` – Pull request checklist for doc and code submissions.
- {doc}`../how-to/index` – Operational guides referenced during review.
- {doc}`../get-started/index` – Quickstarts used by continuous integration smoke tests.

```{toctree}
:maxdepth: 1
:hidden:

../contributing
../extending/index
../reference/index
../governance/nav_crosswalk
../governance/release_checklist
../overview/index
../pr_guide
../how-to/index
../get-started/index
```

{{ optional_extras_template }}
