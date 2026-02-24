# ADR-030: Test Quality Priorities — Reference Extract

> **Source:** `docs/improvement/adrs/ADR-030-test-quality-priorities-and-enforcement.md`
> Loaded by `ce-test-audit` and `ce-code-review` skills.

---

## Quality Priorities (in order)

1. **Determinism & reproducibility** — no wall-clock, randomness, network without patching.
2. **Public-contract testing** — test observable behavior, not private methods.
   Forbids production test-helper wrappers published through `__all__`.
3. **Assertion strength** — meaningful assertions that would fail on real regressions.
4. **Layering & suite health** — unit/integration/e2e scope + `@pytest.mark.slow`.
5. **Fixture discipline** — minimal, scoped, no deep chains.
6. **Semantic efficiency** — 0 tests with 0 unique lines (unless parametrized or regression-marked).

## Hard Gates (CI failures)

| Gate | Script |
|---|---|
| Coverage ≥ 90% | `make test-core` / `check_coverage_gates.py` |
| No private-member usage in tests | `scan_private_usage.py --check` |
| No fallback triggers without opt-in | `disable_fallbacks` autouse fixture |
| No test-helper exports from `src/` | `check_no_test_helper_exports.py` |

## Redundant Test Rule

A test is **redundant** (zero unique lines) unless:
1. It is a `@pytest.mark.parametrize` case with meaningful variation.
2. It is explicitly marked `@pytest.mark.issue`.

## Anti-Pattern Catalog

| Anti-pattern | Detection | Fix |
|---|---|---|
| Direct `_private_func()` call | `scan_private_usage.py` | Test through public API; or make public |
| `assert "key" in dict` | Anti-pattern detector | Assert semantic invariant instead |
| `assert result is not None` (only) | Anti-pattern detector | Add domain assertion |
| Unpatched `datetime.now()` / `random` | Anti-pattern detector | `monkeypatch` / seed |
| Copy-pasted tests | Over-testing report | Parametrize |
| Missing `enable_fallbacks` + fallback fires | CI `UserWarning` | Fix root cause or opt in |

## Enforcement Tooling

```bash
python scripts/anti-pattern-analysis/detect_test_anti_patterns.py
python scripts/anti-pattern-analysis/scan_private_usage.py --check
python scripts/over_testing/over_testing_report.py
python scripts/over_testing/detect_redundant_tests.py
python scripts/quality/check_marker_hygiene.py --check
python scripts/quality/check_no_test_helper_exports.py
```

## Phase Status (as of 2026-02-23)

- Phase 1: baseline + no-new-violations in CI ✅
- Phase 1A: no-test-helper-export hard blocker ✅
- Phase 2: assertion/determinism/mocking detectors (in progress)
- Phase 3: marker hygiene baseline + CI check ✅
- Phase 4: over-testing density gate (advisory, ratcheting toward enforced)
