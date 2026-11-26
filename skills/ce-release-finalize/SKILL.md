---
name: ce-release-finalize
description: >
  Execute the complete calibrated-explanations release handoff: automated
  preflight, human-gated publication, and automated postcommit cleanup.
---

# CE Release Finalize

Finalize a calibrated-explanations release by following `release.md` exactly.
Load `references/pypi_release_guide.md` for command details.

## Use this skill when

- All implementation tasks for the target version are closed.
- The user asks to release, publish, or ship a version.
- A release needs its final metadata, artifacts, publication, or handoff.

## Version contract

- Read the exact PEP 440 release and development versions from the active
  version plan's front matter. Do not assume the next version is
  `X.Y.(Z+1)` — that is a maintainer decision made through a GitHub milestone,
  not something to infer.
- `make release-preflight` updates deterministic release metadata. Do not
  duplicate that work manually before the command.
- A leading `v` is used for tags/citation display, not package/runtime metadata.

## Release workflow

### Steps 1-10: automated preflight

```bash
make release-preflight
make release-finalize
```

`release-preflight` owns release-plan readiness, CHANGELOG conversion, all
deterministic release-file updates, tests, notebook execution/save, version
alignment, clean build, Twine/artifact validation, and clean-wheel smoke.
`release-finalize` verifies that the green snapshot and worktree are unchanged.

For an intentional override:

```bash
make release-preflight VERSION=<exact-version> RELEASE_DATE=YYYY-MM-DD
```

Do not proceed unless both commands exit 0.

### Steps 11-13: maintainer only

The maintainer performs and confirms these immutable/external actions:

11. Commit, tag, and push.
12. Publish/activate/verify the tag on Read the Docs.
13. Upload the validated artifacts to TestPyPI/PyPI.

Do not execute these actions. Pause after `make release-finalize` and wait for
the maintainer to confirm that all three succeeded.

### Steps 14-17: automated postcommit

After the maintainer confirms steps 11-13:

```bash
make release-postcommit
```

The command verifies the exact PyPI metadata and rendered page, installs the
published pin in a clean environment with an exact version assertion, archives
the released plan to `development/finished-work/`, and bumps to the next
development version. Use `NEXT_VERSION=<version>` for an explicit next-release
label; otherwise it advances by one patch. It never invents a minor/major bump
and never creates a next version plan — open one with `ce-release-planner`
only once maintainers have selected the next GitHub milestone.

## Constraints

- Never upload to PyPI, push tags, or publish RTD from this skill; those actions
  belong to the maintainer in steps 11-13.
- Never bypass `twine check`, preflight, or finalize.
- Steps 11-13 are the only human-gated release actions; do not leave steps
  1-10 or 14-17 as undocumented manual follow-ups.
- Follow `release.md` and the canonical guide in
  `references/pypi_release_guide.md`.
