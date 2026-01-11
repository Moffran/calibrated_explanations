> **Status note (2026-01-11):** Last edited 2026-01-11 · Archive after: Retain indefinitely as architectural record · Implementation window: Begin immediately for new/modified tests.

# ADR-030: Test Quality Priorities and Enforcement

Status: Proposed
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
- A test anti-pattern report is generated in CI (`scripts/detect_test_anti_patterns.py`).
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

## Enforcement

The quality criteria above are enforced through a combination of CI gates and
static audits. Specifically:

- **Keep existing hard gates**:
  - Coverage gate (≥90%) and per-module coverage checks.
  - Private-member usage scan in tests (fail on non-allowlisted usage).
  - Fallback-chain restrictions (tests must opt in when validating fallbacks).

- **Expand test anti-pattern checks** (incremental, non-breaking rollout):
  - Extend the existing anti-pattern detector to flag:
    - tests without assertions (including `pytest.raises` and `pytest.warns` as
      acceptable assertion patterns),
    - use of time/random/network without fixtures/patching,
    - excessive mocking without outcome assertions.
  - Run the detector in CI and treat **new** violations as failures
    (baseline existing debt, then ratchet down).

- **Marker hygiene**:
  - Keep the existing test directory conventions.
  - Enforce marker registration and use for slow/integration tests as part of
    pytest config and CI, starting as advisory and moving to blocking once the
    suite is fully tagged.

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
- Phase 2: extend the detector to cover assertion presence, determinism, and
  mocking heuristics; add allowlists only with justification.
- Phase 3: enforce marker hygiene and slow-test budgets once tagging is complete.

## Open Questions

- What marker taxonomy best fits the current tests (`unit`, `integration`, `e2e`,
  `slow`, `viz`), and should any be mandatory on new files?
- Which modules are “core logic” for optional mutation testing in nightly CI?

## Implementation status

- 2025-03-06 – Drafted ADR with priorities and enforcement plan; no code changes yet.
