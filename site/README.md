# Calibrated Explanations project website

This is a neutral, static introduction to the independently maintained
[Calibrated Explanations](https://github.com/Moffran/calibrated_explanations)
open-source library. It is not a Grejna corporate website. Links to optional
Grejna commercial services are clearly separated from the BSD 3-Clause project
and the JU-hosted research demo.

## Why there is no publishing workflow

ADR-035 (`development/adrs/ADR-035-ci-workflow-governance.md`) allows
exactly three top-level workflows and forbids documentation deployment and
publishing from CE CI. The project site therefore remains source-controlled
under `site/`, but **must not introduce a deployment workflow** or loosen the
policy validator. Publication is an explicit maintainer action outside CI.

## Initial publication on GitHub Pages

These actions require write access to the **official** upstream repository.
They are not performed automatically when this documentation PR merges.

1. Merge the reviewed PR into `Moffran/calibrated_explanations:main`.
2. In a clean local checkout of the **upstream** repository, update `main`:
   ```bash
   git checkout main
   git pull --ff-only origin main
   ```
3. Create the independent Pages branch from **only** the `site/` subtree:
   ```bash
   PAGES_SHA="$(git subtree split --prefix=site HEAD)"
   git push origin "$PAGES_SHA:refs/heads/gh-pages"
   ```
   Git for Windows (Git Bash) supports these commands. In PowerShell:
   ```powershell
   $pagesSha = git subtree split --prefix=site HEAD
   git push origin "${pagesSha}:refs/heads/gh-pages"
   ```
   Do not use `--force`. If `gh-pages` is already present, first inspect its
   history and ensure that the proposed update is a fast-forward.
4. A repository administrator enables **Settings → Pages → Build and deployment**
   and selects **Deploy from a branch**, branch `gh-pages`, folder `/(root)`.
5. Check the actual deployment link and the responsive layout. The expected
   *default* address is
   `https://moffran.github.io/calibrated_explanations/`, **not verified yet**.
6. Only after a successful deployment, set the repository *About → Website*
   to the verified project-site URL. Keep Read the Docs and the live CE demo as
   separate and independently identifiable destinations.

## Updating the website

Changes to `site/` should be proposed and reviewed via normal PRs to `main`.
After merge, an upstream maintainer repeats the two `git subtree split` and
`git push` commands above. This produces a separate branch with a clean,
purpose-limited static-site history and keeps all publishing steps outside the
CE development CI. No new CI permission or workflow inventory change is needed.

## Local quality checks

Open `site/index.html` in a browser. Test its keyboard-accessible navigation,
mobile layout, every external link, and reduced-motion experience. The
illustrative numbers and visualization are not data or statistical guarantees.

The CE BSD license, PyPI package, Read the Docs technical reference, project
governance and JU demo remain unchanged.
