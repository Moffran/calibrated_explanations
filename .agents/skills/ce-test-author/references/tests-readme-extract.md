# Test Rubric Reference — tests/README.md Key Extracts

> **Source:** `tests/README.md` — canonical test guidance.
> Loaded by `ce-test-author` and `ce-test-audit` skills.
> When this extract conflicts with the source file, the source file wins.

---

## Framework and Style

- **Framework:** `pytest` + `pytest-mock`. No alternatives.
- **Assertions:** behavior-focused. Refactoring internal implementation must NOT
  break tests.
- **Private API:** avoid `_private` helpers. Test through public callers.

## File Placement

```
tests/unit/<package>/test_<module>.py      — single module, no external I/O
tests/integration/<feature>/test_<feature>.py  — cross-module or pipeline
tests/e2e/<flow>/test_<flow>.py            — full user workflow
```

Create new test files only when: no appropriate file exists, ≤400 lines/50 tests
constraint is breached, or scope differs from all existing candidates.

## Naming

```
test_should_<behavior>_when_<condition>
```

## AAA Structure

```python
def test_should_<behavior>_when_<condition>(fixtures):
    # Arrange
    ...
    # Act
    result = <SUT call>
    # Assert  — one logical assertion block
    assert <semantic domain invariant>
```

## Determinism

- No real network, real clock, unseeded RNG.
- Use `monkeypatch`, `tmp_path`, `np.random.seed(42)`, `freeze_time`.

## Fallback Chain Enforcement

- `disable_fallbacks` is an **autouse** fixture — all tests inherit it.
- `enable_fallbacks` must be requested explicitly if a test validates a fallback.
- Fallback tests MUST assert `pytest.warns(UserWarning)`.

## Fallback Warning Strings

| Fallback | Warning contains |
|---|---|
| Parallel → sequential | "fallback" / "sequential" |
| Cache backend | "Cache backend fallback" |
| Visualization simplified | "Visualization fallback" |
| Plugin execution error | "legacy" / "fallback" |

## Coverage Gate

```bash
pytest --cov=src/calibrated_explanations --cov-config=pyproject.toml --cov-fail-under=90
```

Unit tests: <100 ms. Integration: <2 s. Mark slow tests `@pytest.mark.slow`.

## Behavior vs Implementation Decision Tree

1. What am I testing? → user-facing outcome (not internal mechanism).
2. Would this break on a pure refactor? → if YES, rewrite.
3. Can I state this as a business rule / domain invariant? → if NO, clarify first.
4. Is it isolated from I/O / env-vars? → if NO, add mocks + integration test.

## CE Interval Invariant (always assert for prediction tests)

```python
assert prediction['low'] <= prediction['predict'] <= prediction['high']
```
