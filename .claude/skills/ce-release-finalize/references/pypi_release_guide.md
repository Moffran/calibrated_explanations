# PyPI Release Guide for calibrated-explanations

Canonical source: `debug/release.md` in the calibrated_explanations repository.

## Pre-requisites (once)

- Python packages: `build`, `twine` (`python -m pip install --upgrade build twine`)
- PyPI account with API token configured in `~/.pypirc`
- Understand PEP 440 versioning: PyPI uses `X.Y.Z`, display/tags use `vX.Y.Z`

## Files requiring version updates

| File | Field | Format | Example |
|---|---|---|---|
| `pyproject.toml` | `[project].version` | `X.Y.Z` | `0.11.0` |
| `src/calibrated_explanations/__init__.py` | `__version__` | `vX.Y.Z` | `v0.11.0` |
| `CITATION.cff` | `version` + `date-released` | `vX.Y.Z` + `YYYY-MM-DD` | `v0.11.0` |
| `docs/conf.py` | `release` + `version` | `X.Y.Z` + `X.Y` | `0.11.0` / `0.11` |
| `docs/citing.md` | BibTeX `version` | `vX.Y.Z` | `v0.11.0` |
| `METADATA.json` | `version` | `X.Y.Z` | `0.11.0` |
| `docs/improvement/RELEASE_PLAN_v1.md` | `Current released version` | `vX.Y.Z` | `v0.11.0` |

## Release steps

1. Checkout and pull `main`
2. Verify CI is green (`make ci-local-new`)
3. Update `CHANGELOG.md` (new version section, move unreleased items, update compare links)
4. Bump version in all files above
5. Sanity-check version consistency
6. Remove build artifacts (`rm -rf dist/ build/ *.egg-info/`)
7. Build (`python -m build`)
8. Validate (`python -m twine check dist/*`)
9. Smoke-test wheel locally (optional but recommended)
10. Commit and tag (`git commit`, `git tag vX.Y.Z`, `git push`, `git push --tags`)
11. Publish RTD docs (verify tag build, activate version, set stable)
12. Upload to PyPI (`python -m twine upload --repository pypi dist/*`)
13. Verify PyPI project page
14. Test installation in clean venv
15. Bump to next dev version

## Critical notes

- PyPI uploads are **immutable** for a given version
- Always run `twine check` before upload
- Never push tags without confirming the build is green
- RTD builds are triggered by pushed tags via `.readthedocs.yaml`
