> **Status note (2026-05-12):** Last edited 2026-05-12 · Archive after: Retain indefinitely as architectural record · Implementation window: Begin immediately for new/modified tests. Ratified v0.11.3 - zero-tolerance CI enforcement confirmed via `detect_test_anti_patterns.py` blocking gates in `ci-pr.yml` and `ci-main.yml`; focused local ratification lane added via `python scripts/local_checks.py --adr030-ratification`; marker hygiene taxonomy documented; mutation testing declared optional. Ratified: 2026-05-12.

# ADR-030: Test Quality Priorities and Enforcement

Status: Accepted
Date: 2026-01-11
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

The calibrated_explanations test suite is large (≈2000 tests) and already enforces a
coverage gate (≥90% for `src/calibrated_explanations`) in CI, alongside
anti-pattern scans and private-member usage checks. These controls catch some classes of
test debt, but coverage alone is a weak proxy for confidence and does not prevent
brittle, flaky, or behavior-ambiguous tests. The repo also has an existing
testing guidance document that emphasizes behavior-first pytest tests,
determinism, and avoidance of private helpers, plus fallback-chain restrictions and
explicit opt-in for fallback testing. The ADR should therefore codify the
primary quality criteria (beyond coverage) and define how they are enforced in CI.

### Existing enforced policy and tooling (current state)

- Coverage gate is enforced in CI via `make test-core` (pytest with coverage flags) and
  `scripts/check_coverage_gates.py`, plus the PR checklist and contributor guidance.
- Private-member usage in tests is blocked by the `scan-private-members` workflow
  (`scripts/anti-pattern-analysis/scan_private_usage.py --check`).
- A test anti-pattern report is generated in CI (`scripts/anti-pattern-analysis/detect_test_anti_patterns.py`).
- The test guidance explicitly requires behavior-first pytest tests, determinism,
  and avoiding private helpers, plus explicit fallback-chain opt-in.

## Decision

Coverage remains a **necessary** CI gate, but it is **secondary** to test quality
criteria that measure reliability and behavioral correctness. The primary criteria
(and the ones most aligned with pytest usage in this repo) are:

1. **Determinism & reproducibility (highest priority)**
   - **Definition:** tests should not rely on wall-clock time, nondeterministic
     randomness, network I/O, or test order.
   - **Pros:** eliminates flakiness, improves trust, stabilizes CI.
   - **Cons:** may require dependency injection, seeded RNGs, or more fixtures/mocks.
   - **Pytest fit:** `monkeypatch`, `tmp_path`, explicit seeding, and strict warning modes
     (for fallback detection) are already part of repo guidance.

2. **Public-contract testing (avoid private helpers)**
   - **Definition:** tests should validate observable public behavior, not private
     methods or internal attributes.
   - **Pros:** resilient to refactors; tests act as contract documentation.
   - **Cons:** may need API hooks or integration tests to cover internal edge cases.
   - **Pytest fit:** aligns with current guidance and the private-member scan.

   Public-contract testing also forbids adding production test-helper wrappers
   as a workaround for private-member access in tests.

3. **Assertion strength (behavioral signal)**
   - **Definition:** tests must contain meaningful assertions that would fail for
     plausible regressions; avoid execution-only tests.
   - **Pros:** raises confidence that tests actually validate behavior.
   - **Cons:** overly specific assertions can become brittle if poorly scoped.
   - **Pytest fit:** pytest assert introspection and `pytest.raises(..., match=...)`
     make precise assertions low-cost.

4. **Layering & suite health (unit vs integration vs slow)**
   - **Definition:** tests should be scoped and marked by layer, and runtime budgets
     should be visible and managed.
   - **Pros:** keeps CI fast while preserving integration confidence.
   - **Cons:** requires discipline in markers and CI configuration.
   - **Pytest fit:** existing test tree already uses unit/integration/e2e conventions;
     markers can be enforced with `--strict-markers` as the suite evolves.

5. **Fixture discipline & clarity**
   - **Definition:** fixtures should be minimal, scoped appropriately, and avoid
     deep dependency chains; tests remain readable without chasing multiple fixtures.
   - **Pros:** improves maintainability and debugging speed.
   - **Cons:** can require refactoring fixture-heavy tests.
   - **Pytest fit:** strong fit due to fixture scoping, parametrization, and local
     helpers in conftest files.

6. **Semantic efficiency (avoid meaningful over-testing)**
   - **Definition:** Every test must provide a unique contribution to the suite's
     confidence. A test is considered **redundant** if it has 0 unique lines unless it
     provides a unique parameter combination, return-value assertion, or side-effect check
     that is not covered by any other test.
   - **Pros:** Reduces suite runtime, maintenance burden, and "false confidence" from high test counts.
   - **Cons:** Requires careful design of parameterized tests; may flag valid regression tests if not marked.
   - **Pytest fit:** `pytest.mark.parametrize` is the preferred pattern for variations;
     copy-pasted tests are explicitly discouraged.

## Enforcement

The quality criteria above are enforced through a combination of CI gates and
static audits. Specifically:

- **Keep existing hard gates**:
  - Coverage gate (≥90%) and per-module coverage checks.
  - Private-member usage scan in tests (fail on non-allowlisted usage).
  - Fallback-chain restrictions (tests must opt in when validating fallbacks).
  - Production export leakage guard (fail when `src/` publishes test-helper
    wrappers through `__all__`), via
    `scripts/quality/check_no_test_helper_exports.py`.

- **Redundant test strictness (zero unique lines)**:
  - **Metric:** `Unique Lines` per test (calculated via `pytest --cov-context=test`).
  - **Requirement:** **0 tests with 0 unique lines**, unless:
    1.  The test is a `pytest.mark.parametrize` case where the variation is meaningful (input/output).
    2.  The test is explicitly marked as a regression test for an issue (`@pytest.mark.issue`).
  - **Stronger Goal (Behavioral Uniqueness):** No two non-parameterized tests shall have
    **identical coverage fingerprints** (set of all lines hit). Identical fingerprints
    indicate a purely redundant test that should be deleted or parameterized.
  - **Mechanism:** `scripts/over_testing/over_testing_report.py` computes unique lines and
    fingerprints. CI will warn (advisory) and eventually block (enforced) on violations.

- **Expand test anti-pattern checks** (incremental, non-breaking rollout):
  - Extend the existing anti-pattern detector to flag:
    - tests without assertions (including `pytest.raises` and `pytest.warns` as
      acceptable assertion patterns),
    - use of time/random/network without fixtures/patching,
    - excessive mocking without outcome assertions.
  - Run the detector in CI and treat **new** violations as failures
    (baseline existing debt, then ratchet down).

- **Marker hygiene**:
  - See the formal marker hygiene section below for the binding taxonomy and
    enforcement posture.

### Marker hygiene

ADR-030 adopts a hybrid pytest marker taxonomy that balances explicit review
signals with the repository's existing test layout:

- `unit` and `integration` are inferred from test directory paths by
  `scripts/quality/check_marker_hygiene.py`.
- `slow`, `viz`, and `viz_render` are explicit markers and must be declared
  when the test behavior requires them.
- Every test must be classifiable by this taxonomy either through directory
  inference or an explicit marker.
- Marker registration and marker hygiene are enforced by
  `scripts/quality/check_marker_hygiene.py --check` in the PR/main
  anti-pattern audit jobs and in the local ADR-030 ratification lane.
- Existing marker-hygiene debt is tracked in
  `.github/marker-hygiene-baseline.json`; new violations fail the check rather
  than silently expanding the baseline.

### Mutation testing policy

Mutation testing is recommended for critical-path logic in
`src/calibrated_explanations/calibration/` and
`src/calibrated_explanations/core/`, but it is not a release gate, PR gate, or
required local check for v0.11.x/v1.0.0. Contributors may use `mutmut` for
targeted hardening runs, for example:

```bash
mutmut run --paths-to-mutate src/calibrated_explanations/
```

Mutation testing remains optional because it can be expensive for the full
suite and is most valuable as a focused reviewer or maintainer tool for
deterministic core behavior. Plugins and visualization modules are excluded
from any recommended nightly mutation-testing scope unless a future ADR or
standard changes this policy.

### Focused local ratification lane

The ADR-030 gate stack must remain reproducible without running unrelated
lint, docs, coverage, or packaging checks. The focused local lane is:

```bash
python scripts/local_checks.py --adr030-ratification
```

The corresponding Make target is:

```bash
make adr030-ratification
```

This lane runs the private-member scan, anti-pattern detector, production
test-helper export guard, marker hygiene check, and generated-report local-path
guard, then writes observational timing evidence to
`reports/anti-pattern-analysis/adr030_ratification_timing.json`. Timing
evidence is informational only and must not introduce a duration threshold.

## Alternatives Considered

1. **Keep coverage as the only required gate.**
   - Rejected because it does not address determinism, private-method coupling,
     or assertion strength, and fails to prevent test debt in large suites.

2. **Rely only on documentation (no CI enforcement).**
   - Rejected because it creates aspirational guidance without any guardrails.

## Consequences

Positive:
- Improves test reliability and refactorability without sacrificing coverage goals.
- Aligns test practices with pytest strengths (fixtures, assertions, markers).
- Provides a clear policy that matches existing anti-pattern work and private
  member removal efforts.

Negative/Risks:
- Additional static checks may surface existing debt; must be rolled out
  with baselines and gradual enforcement to avoid blocking all work.
- Some refactors may be needed to expose public seams for complex logic.

## Adoption & Migration

- Phase 1 (immediate): record baseline anti-pattern report and enforce “no new
  violations” in CI.
- Phase 1A (immediate hard blocker): enforce zero tolerance for production
  test-helper wrapper exports via `check_no_test_helper_exports.py`.
- Phase 2: extend the detector to cover assertion presence, determinism, and
  mocking heuristics; add allowlists only with justification.
- Phase 3: enforce marker hygiene and slow-test budgets once tagging is complete.
- Phase 4: introduce the over-testing density gate in advisory mode, then ratchet
  the threshold once baselines are stable.

## Implementation status

- 2026-01-11 – Drafted ADR with priorities and enforcement plan; resolved open questions on marker taxonomy and mutation testing modules; no code changes yet.
- 2026-02-09 – Added over-testing density analysis scripts and proposed a ratcheting
  gate based on per-test coverage contexts (pending CI rollout).
- 2026-02-09 – Completed coverage improvement iteration: added integration tests for plotting style overrides/legacy fallbacks, cache fallback testing, and YAML template loading to increase coverage in low-coverage modules (plotting.py, cache.py, narrative_generator.py).
- 2026-02-15 – Phase 3 (marker hygiene): added `scripts/quality/check_marker_hygiene.py`
  with `--check` / `--rebaseline` modes and committed baseline
  (`.github/marker-hygiene-baseline.json`, 72 existing-debt entries). Wired into
  `ci-pr.yml` and `ci-main.yml` anti-pattern-audit jobs.
- 2026-02-15 – Phase 4 (over-testing density): wired `over_testing_report.py` and
  `detect_redundant_tests.py` into `ci-main.yml` as an advisory
  (`continue-on-error: true`) job with `--cov-context=test` coverage collection.
  Reports published as CI artifacts.
- 2026-02-15 – Anti-pattern audit added to `ci-pr.yml` so PR checks survive
  `test.yml` compat-wrapper decommission (release task 12).
- 2026-02-23 – Added `scripts/quality/check_no_test_helper_exports.py` and
  wired it into PR/main/compat CI plus local checks as a hard blocker to prevent
  production test-helper wrapper exports.
- 2026-05-12 - Ratified ADR-030 for v0.11.3: promoted marker hygiene and mutation
  testing policy into formal sections, confirmed PR/main anti-pattern audit gates
  are blocking, and added focused local ratification with observational timing
  evidence via `python scripts/local_checks.py --adr030-ratification`.
