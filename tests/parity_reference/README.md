# Parity reference harness

`run_parity_reference.py` asserts that `CalibratedExplainer` output for a
fixed `seed=42`, fixed canonical dataset, and fixed model is **byte-for-byte
identical** (within `rtol=1e-6, atol=1e-8`) to the committed golden fixtures
in this directory (`factual*.json`, `alternatives*.json`, `fast*.json`,
`predictions*.json`). It runs nightly via `.github/workflows/ci-nightly.yml`
(`parity-reference` job), not as part of `pytest -q`.

## Why this directory has its own `constraints.txt`

This harness's only purpose is to detect *any* change in explanation output,
including changes caused by upstream dependency upgrades rather than CE code
changes. That makes it inherently sensitive to the exact version of
`scikit-learn` used to fit the underlying `DecisionTreeClassifier` /
`DecisionTreeRegressor` models — tree-splitting tie-breaking is an internal
implementation detail that upstream is free to change between releases
without a deprecation cycle, since it isn't part of sklearn's public
behavioral contract.

**Confirmed 2026-06-16 (see `docs/improvement/v0.11.4_plan.md` Task 7):**
scikit-learn changed `DecisionTreeRegressor` split selection between 1.7.2
and 1.8.0. Bisection results:

| scikit-learn | `--dataset regression` parity | `--dataset classification` parity |
| --- | --- | --- |
| 1.6.1 – 1.7.2 | pass | pass |
| 1.8.0 | **fail** (3,361 diffs) | pass |
| 1.9.0 | **fail** (58,523 diffs) | pass |

`DecisionTreeClassifier` (used by `classification`/`multiclass`) was
unaffected at every tested version. `numpy` version was ruled out as a
factor by the same bisection.

The project-wide `constraints.txt` floor for `scikit-learn` was deliberately
widened to `>=1.3` (no upper bound) in commit `9693c3d8` for general
compatibility — that decision is correct for "does CE work with this
sklearn version" but wrong for "does this harness reproduce byte-identical
output," which are different questions. Re-adding a project-wide upper
bound would have undone the general-compatibility widening for every
consumer to serve one nightly test's narrower need, so the pin lives here
instead: `tests/parity_reference/constraints.txt` is layered on top of the
project-wide `constraints.txt` **only** in the `parity-reference` CI job.

## Updating the pin

If you intentionally want the parity contract to track a newer
`scikit-learn`:

1. Regenerate the affected fixtures: `python tests/parity_reference/run_parity_reference.py --dataset regression --update` (and `probabilistic_regression`, and both `--condition-source` values).
2. Re-run the full harness (all 8 invocations) to confirm parity holds at the new version.
3. Bump the upper bound in `tests/parity_reference/constraints.txt` and document the new verified range, mirroring the table above.
4. Note the change in `CHANGELOG.md` and update `docs/improvement/v0.11.4_plan.md` Task 7 (or its closure record) accordingly.

Do not bump the pin without doing step 1-2 first — the failure mode is
silent (wrong explanations, not a crash), so an unverified bump defeats the
harness's purpose.
