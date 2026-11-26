# PyPI Release Guide for calibrated-explanations

## Pre-requisites

- Read https://packaging.python.org/en/latest/tutorials/packaging-projects/.
- Install `build` and `twine`.
- Configure a PyPI API token.
- Treat every PyPI version upload as immutable.

Package/runtime versions are exact PEP 440 strings without `v`; tags and
citation display use a leading `v`.

## Release-file contract

`make release-preflight` updates and verifies:

- `pyproject.toml`
- `src/calibrated_explanations/__init__.py` metadata fallback
- `CITATION.cff` version/date
- metadata-derived `docs/conf.py` version/release
- `docs/citing.md` BibTeX version/month/year
- `CHANGELOG.md` release section and compare links
- `METADATA.json`
- the active version plan's declared release/development versions

Do not edit generated `build/`, `dist/`, or `*.egg-info` artifacts manually.

## Steps 1-10: automated preflight

Run:

```bash
make release-preflight
make release-finalize
```

Preflight performs the historical steps 1-10: main-branch/plan readiness, full
tests, changelog and release-file preparation, editable install, governed
notebook execution/save, alignment checks, stale-artifact cleanup, build, Twine
and packaging checks, and clean-wheel install/version smoke. Finalize confirms
that the successful handoff snapshot is still current.

The exact version is inferred from the active plan. For an intentional override:

```bash
make release-preflight VERSION=X.Y.Z RELEASE_DATE=YYYY-MM-DD
```

Do not continue unless both commands exit 0.

## Steps 11-13: maintainer publication

11. Commit/tag/push:

```bash
git add .
git commit -m 'calibrated-explanations vX.Y.Z'
git tag vX.Y.Z
git push
git push --tags
```

12. Publish the tag on Read the Docs:

- confirm the remote tag;
- build/activate it;
- point `stable` to it;
- spot-check the stable docs.

13. Upload the already-validated artifacts:

```bash
python -m twine upload --repository testpypi dist/*
python -m twine upload --repository pypi dist/*
```

These three steps are manual and require explicit maintainer confirmation.

## Steps 14-17: automated postcommit

After steps 11-13 succeed:

```bash
make release-postcommit
```

14. Verify the exact-version PyPI JSON metadata and rendered project page.
15. Install the exact published pin in a clean venv and assert the imported
    version matches.
16. Archive the released plan to `development/finished-work/`. No next
    version plan is created automatically — open one with `ce-release-planner`
    once maintainers have selected the next GitHub milestone.
17. Set `pyproject.toml` and the source fallback to the next development
    version (one patch ahead by default).

For an explicit next-release label instead of the default patch bump:

```bash
make release-postcommit NEXT_VERSION=<version>
```

Postcommit refuses a still-development project version and never uploads,
tags, pushes, or publishes documentation.
