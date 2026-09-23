Releasing calibrated-explanations on PyPI
=========================================

This is the maintainer runbook for releasing calibrated-explanations on PyPI.

## Pre-requisites (once)

- Read the [official Python packaging guide].
- Install `build` and `twine`: `python -m pip install --upgrade build twine`.
- Configure a PyPI account and API token in `~/.pypirc` (macOS/Linux) or
  `%USERPROFILE%\\.pypirc` (Windows).

```ini
[pypi]
  username = __token__
  password = pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

PyPI uploads are immutable for a version. This repository uses a PEP 440
package/runtime version (`X.Y.Z`, or an exact prerelease such as `1.0.0rc1`)
and a git/citation display version with a leading `v`.

## Files that must be updated for a release

`make release-prepare-files` updates the deterministic fields below while
version-dependent task checklists are still open. `make release-preflight`
repeats the same update idempotently before running the strict gate. The list is
kept here as the contract that the automation and its tests must cover.

1. Packaging/runtime version

- `pyproject.toml`: `[project].version = "X.Y.Z"`
- `src/calibrated_explanations/__init__.py`: metadata fallback set to the
  canonical release version without a leading `v`; postcommit changes it to
  the next canonical development version

2. Citation/docs version

- `CITATION.cff`: `version: vX.Y.Z` and current `date-released`
- `docs/conf.py`: verified as deriving `release` and short `version` from
  installed package metadata
- `docs/citing.md`: software BibTeX version and release month/year

3. Changelog

- `CHANGELOG.md`: move `[Unreleased]` content into the release section and
  update both compare links

4. Repository metadata

- `METADATA.json`: `version = "X.Y.Z"`

5. Planning/release tracking

- the active `development/current-work/vX.Y.Z_plan.md`: version source and
  release-closure checklist. It is archived to `development/finished-work/`
  during step 16.
- the next version plan is not created automatically; open one with the
  `ce-release-planner` skill once maintainers have selected the next GitHub
  milestone.

Generated `*.egg-info`, `build/`, and `dist/` files are not edited manually.
Notebook source code is not version-pinned, but governed outputs are refreshed
by preflight.

## Release steps (every time)

`make release-preflight` is the authoritative implementation of steps 1-10.
It infers the exact version from the active version plan (or accepts
`VERSION=<version>`), uses the UTC date (or `RELEASE_DATE=YYYY-MM-DD`), updates
the release files above, and runs the strict gates. The descriptions below are
the behavioral contract for that one command.

For release tasks that must verify the final version before their checklist can
close, prepare the deterministic files first:

```bash
make release-prepare-files VERSION=X.Y.Z RELEASE_DATE=YYYY-MM-DD
```

This focused command does not run readiness checks, build artifacts, write a
green preflight report, or unlock publication. The complete preflight below is
still mandatory after all prerequisite task checklists are closed.

1. Verify `main`, the version lineage, and release-plan closure readiness.
2. Run the full pytest suite and strict release profile (the local replacement
   for the historical manual CI confidence check).
3. Convert `[Unreleased]` into the dated versioned changelog section.
4. Update and align every deterministic release file listed above.
5. Install the release tree and execute/save all governed release notebooks.
6. Check runtime, package, docs, citation, and repository version alignment.
7. Clean stale distribution artifacts.
8. Build the wheel and source distribution.
9. Validate artifacts with Twine and the packaging inspector.
10. Install the freshly built wheel into a clean environment and verify the
    imported version.

Run steps 1-10:

```bash
make release-preflight
```

For an explicit or nonstandard transition:

```bash
make release-preflight VERSION=X.Y.Z RELEASE_DATE=YYYY-MM-DD
```

Immediately before step 11, run:

```bash
make release-finalize
```

Do not continue unless both commands exit 0. `release-finalize` confirms that
the green preflight report, active plan, branch, and worktree still match.

11. Commit, tag, and push the release manually.

```bash
git add .
git commit -m 'calibrated-explanations vX.Y.Z'
git tag vX.Y.Z
git push
git push --tags
```

12. Publish the docs on Read the Docs manually.

	RTD builds are triggered from the pushed tag using `.readthedocs.yaml` and `docs/conf.py`.
	The public docs live at https://calibrated-explanations.readthedocs.io/.

	1) Confirm the tag is on the remote:

	```bash
	git ls-remote --tags origin vX.Y.Z
	```

	2) Trigger/verify the RTD build for the tag:
	- Builds page: https://readthedocs.org/projects/calibrated-explanations/builds/
	- If the tag build is not present, go to **Versions** and click **Sync versions**.
	- If needed, use **Build version** for `vX.Y.Z`.
	- Wait until the build status is **Passed** (and open the build log if it fails).

	3) Activate the tag and set it as stable:
	- Versions page: https://readthedocs.org/projects/calibrated-explanations/versions/
	- Ensure `vX.Y.Z` is **Active**.
	- Set `vX.Y.Z` as the **stable** version (so `/en/stable/` serves the release tag).
	  (If RTD shows “Default version”, set it to `stable` and make sure `stable` points to `vX.Y.Z`.)

	4) Spot-check the rendered stable docs:
	- https://calibrated-explanations.readthedocs.io/en/stable/
	- Confirm the landing page loads, navigation works, and the two quickstarts render.

	Troubleshooting:
	- If `vX.Y.Z` never appears: make sure tags are enabled in RTD project settings and re-run **Sync versions**.
	- If the build fails: open the RTD build log first; RTD installs `docs/requirements-doc.txt` per `.readthedocs.yaml`.
	  Notebook execution is disabled on RTD to keep builds lightweight; CI should enforce execution gates.

13. Upload the artifacts manually.

```bash
python -m twine upload --repository testpypi dist/*
python -m twine upload --repository pypi dist/*
```

Steps 11-13 are intentionally manual and human-gated. After all three succeed,
run the authoritative post-publish command for steps 14-17:

```bash
make release-postcommit
```

Use `NEXT_VERSION=<version>` only for an explicit next-release label;
otherwise the next patch version is used. This never invents a minor/major
bump — that is a maintainer decision made through a GitHub milestone.

14. Verify the exact version's PyPI JSON metadata and rendered project page.
15. Install `calibrated-explanations==X.Y.Z` from PyPI into a clean temporary
    environment and assert that `ce.__version__` exactly matches.
16. Archive the released plan under `development/finished-work/`. No next
    version plan is created automatically — open one with the
    `ce-release-planner` skill once maintainers have selected the next GitHub
    milestone.
17. Bump `pyproject.toml` and the source fallback to the next development
    version (one patch ahead of the release, unless `NEXT_VERSION` was given).

`make release-postcommit` refuses to run while the project version is still a
development version, and it never performs steps 11-13.

### Optional: conda-forge feedstock auto-update

`release-postcommit` ends with a best-effort, non-blocking call to
`scripts/update_conda_feedstock.py`. When the `CE_CONDA_FEEDSTOCK_DIR`
environment variable points at a local clone of
[`tuvelofstrom/calibrated-explanations-feedstock`](https://github.com/tuvelofstrom/calibrated-explanations-feedstock)
(a fork of `conda-forge/calibrated-explanations-feedstock`), it runs that
clone's `update_meta.py`, which:

- fetches the latest version and sdist sha256 from PyPI;
- if the feedstock recipe is already current, does nothing;
- otherwise branches from `upstream/main`, updates `recipe/meta.yaml`
  (version, sha256, build number reset to 0), commits, pushes the branch to
  the fork, and opens (or reuses) a PR against
  `conda-forge/calibrated-explanations-feedstock` via the `gh` CLI. The fork's
  own default branch is never touched.

This step is unset (skipped) by default and is unrelated to steps 14-17 above
— it never blocks or fails `release-postcommit`. Set the variable once per
machine to enable it:

```powershell
setx CE_CONDA_FEEDSTOCK_DIR "C:\path\to\calibrated-explanations-feedstock"
```

To retry after a failure without rerunning the rest of postcommit:

```bash
make conda-feedstock-update
```

Requires `gh` authenticated with push access to the fork and PR permission on
the upstream feedstock.

## Known environment caveats

- **Windows/Jupyter kernel drift (step 5, notebook execution).** On some
  maintainer machines, `jupyter nbconvert --execute` resolves notebook
  kernels via a globally-registered `python3`/`ce_test_kernel` kernelspec
  (`%APPDATA%\jupyter\kernels\`), which points at the system Python install,
  **not** `venv-release`. `make release-preflight`'s "Editable install
  (release tree)" step only refreshes `venv-release`'s own metadata, so a
  separately-registered global kernel's editable install of
  `calibrated_explanations` can drift stale and silently affect notebook
  execution output. Before trusting notebook execution output, confirm the
  global kernel environment reports the correct version:
  `python -c "import calibrated_explanations as ce; print(ce.__version__)"`
  using the system Python that `%APPDATA%\jupyter\kernels\python3\kernel.json`
  points to; if stale, run `pip install -e ".[dev]"` in that environment. This
  is a local-machine setup issue, not a code defect.

[official Python packaging guide]: https://packaging.python.org/en/latest/tutorials/packaging-projects/
