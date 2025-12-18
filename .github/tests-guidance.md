# Centralized Test Guidance

This repository uses a single source of truth for all test-related instructions. Whether you are writing tests manually, using Copilot, or invoking automation prompts, follow the guidance below without deviation unless a maintainer signs off in the PR description.

## Scope, Frameworks, and Style
- Python tests use `pytest` with existing fixtures and `pytest-mock` when mocking is required. Do not add alternative frameworks.
- Favor behaviorally focused assertions. A pure refactor that preserves behavior should not break your tests; if it would, rewrite the test.
- Avoid targeting `_private` helpers directly. Either elevate them to a public contract or cover them indirectly through public callers.

## Test File Creation & Grouping Policy
1. Start by identifying the source under test (SUT) and the nearest existing test file that matches it. Extend that file whenever possible.
2. Only create a new test file when **all** of these are true:
   - No appropriate file exists for the SUT and appending elsewhere would mix unrelated concerns.
   - The candidate file would exceed ~400 lines or ~50 test cases after your changes, or it is already flaky/slow.
   - The new tests have a different scope (unit vs integration vs e2e) than the existing file.
   - You can place the new file in the canonical path: `tests/unit/<package>/test_<module>.py`, `tests/integration/<feature>/test_<feature>.py`, or `tests/e2e/<flow>/test_<flow>.py`.
3. Group tests by scope first, then by SUT. Keep fixtures/helpers co-located and reuse shared fixtures rather than duplicating them.
4. Creating a new file requires a **“Why a new test file?”** section in the PR describing scope, justification, and exact path.

## Test Content Rubric
- **Naming:** prefer `should_<behavior>_when_<condition>` style.
- **Structure:** follow Arrange–Act–Assert with one logical assertion block per behavior.
- **Determinism:** no real network, clock, or randomness. Use mocks/fakes, freeze time, or seed RNG, and pair heavily mocked unit tests with at least one integration test.
- **Snapshots/Round-trips:** acceptable only for stable structures; always combine with semantic assertions (e.g., serialize → deserialize → verify invariants).
- **Coverage focus:** emphasize branches and edge cases near recent changes. Link each test to its SUT via a short comment if not obvious.
- **Performance:** keep unit tests <100 ms, integration tests <2 s when feasible. Mark slow tests using existing repo conventions.

## Coverage & Tooling Expectations
- Local gate: `pytest --cov=src/calibrated_explanations --cov-config=.coveragerc --cov-fail-under=90`.
- Lint/mypy requirements still apply to any touched modules; update docs when behavior changes.
- For deep domain context, see the "Detailed Guidelines & Patterns" section below.

## Fallback Chain Enforcement (MANDATORY)

**Policy:** Tests MUST NOT trigger fallback chains unless the test is explicitly validating fallback behavior.

**Rationale:**
- Fallback chains obscure test failures. If a plugin fails and silently falls back to a legacy implementation, the test may pass while hiding real bugs.
- Tests should validate the primary code path, not accidental fallback paths.
- Fallback-dependent tests are fragile and lead to false confidence.

**Implementation:**
1. **Default Behavior:** All tests inherit the `disable_fallbacks` fixture, which automatically sets all fallback chains to empty tuples.
2. **Explicit Opt-In:** If a test is validating fallback behavior, it MUST explicitly opt in by using the `enable_fallbacks` fixture.
3. **Fixture Usage:**
   ```python
   # Normal test (fallbacks disabled by default via autouse fixture)
   def test_explanation_plugin_execution():
       explainer = CalibratedExplainer(...)
       explanation = explainer.explain_factual(x_test)
       assert explanation is not None

   # Test that explicitly validates fallback behavior
   def test_explanation_plugin_fallback_chain(enable_fallbacks):
       """Verify fallback chain handles missing plugin."""
       explainer = CalibratedExplainer(..., _explanation_plugin_override="missing-plugin")
       # Should fall back to default plugin
       explanation = explainer.explain_factual(x_test)
       assert explanation is not None
   ```

4. **CI Enforcement:** The CI pipeline runs with strict fallback detection enabled. Any test that triggers a fallback warning will fail unless explicitly marked with `enable_fallbacks`.

5. **How to Check:** If you see warnings like:
   - `"Execution plugin error; legacy sequential fallback engaged"`
   - `"Parallel failure; forced serial fallback engaged"`
   - `"Cache backend fallback: using minimal in-package LRU/TTL implementation"`
   - `"Visualization fallback: alternative bar simplified due to drawing error"`
   
   Your test is triggering a fallback and will fail in CI. Either fix the underlying issue or explicitly opt in with `enable_fallbacks`.

**When to Use `enable_fallbacks`:**
- Testing that the fallback chain works correctly
- Testing error recovery behavior
- Testing graceful degradation
- Testing compatibility with missing optional dependencies

**When NOT to Use `enable_fallbacks`:**
- Normal feature tests
- Unit tests for specific plugins
- Integration tests that should use the primary code path

Adhering to this document keeps Copilot policies, automation prompts, and human contributors aligned. If a scenario demands deviating, document the reasoning explicitly in the PR.

---

# Detailed Guidelines & Patterns

**Scope:** Detailed decision trees, examples, and patterns for writing behavior-focused tests.

## 📋 Table of Contents

1. [Behavior vs. Implementation: Decision Tree](#behavior-vs-implementation-decision-tree)
2. [Anti-Pattern Catalog](#anti-pattern-catalog-with-refactoring-examples)
3. [Positive Pattern Examples](#positive-pattern-examples)
4. [Private Helpers: Guidelines](#private-helpers-guidelines)
5. [Mocking: Best Practices](#mocking-best-practices)
6. [Snapshots and Serialization](#snapshots-and-serialization)
7. [Test Naming Convention (Extended)](#test-naming-convention-extended)
8. [Integration Testing Checklist](#integration-testing-checklist)
9. [Fallback Chain Testing](#fallback-chain-testing)
10. [Common Pitfalls & Solutions](#common-pitfalls--solutions)

---

## Behavior vs. Implementation: Decision Tree

When writing a test, **always ask these questions in order:**

```
┌─ START: I'm writing a test
│
├─ Q1: What am I testing?
│   ├─ Internal mechanism (e.g., dict keys, private method calls)
│   │   └─ RED FLAG: You're testing implementation, not behavior
│   └─ User-facing outcome (e.g., "data is saved," "model is calibrated")
│       └─ GOOD: You're testing behavior; proceed
│
├─ Q2: Would this test break if I refactored the implementation
│       while keeping the behavior the same?
│   ├─ YES → You're testing implementation
│   │   └─ REFACTOR: Change test to assert on behavior, not mechanism
│   └─ NO → You're testing behavior; keep this test
│
├─ Q3: Can I state this test as a business rule or domain invariant?
│   ├─ NO → You're testing internal detail; clarify the business rule first
│   └─ YES → You're testing behavior; proceed
│
├─ Q4: Is this test isolated from I/O, network, or env vars?
│   ├─ NO → Use mocks; add integration test for the real I/O
│   └─ YES → Good; proceed
│
└─ DECISION: Commit test ✅
```

### Example Application

**Scenario 1: Testing PlotSpec Serialization**

```python
# ❌ ANTI-PATTERN: Testing implementation
def test_plotspec_to_dict_contains_expected_keys():
    spec = PlotSpec(header=IntervalHeaderSpec(...))
    d = plotspec_to_dict(spec)
    assert "plotspec_version" in d  # Q1: Internal detail (dict keys)
    assert "title" in d              # Q2: Breaks if we rename "title" → "figure_title"
    # Q3: No clear business rule (why must these keys exist?)
    # ❌ REFACTOR

# ✅ PATTERN: Testing behavior
def test_plotspec_roundtrip__should_preserve_interval_invariants():
    """Verify that serialization preserves domain constraints."""
    original = PlotSpec(
        header=IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8),
        title="My Plot",
        body=BarHPanelSpec(bars=[...])
    )

    # Roundtrip
    d = plotspec_to_dict(original)
    restored = plotspec_from_dict(d)

    # Q1: User-facing outcome: "restored spec is valid"
    # Q2: Stays true if we change serialization format (JSON, Protobuf, etc.)
    # Q3: Business rule: "Interval bounds must satisfy low ≤ pred ≤ high"
    # Q4: No I/O beyond serialization; local test

    # Assertions on behavior (domain invariants)
    assert restored.header.low <= restored.header.pred, "Interval invariant violated"
    assert restored.header.pred <= restored.header.high, "Interval invariant violated"
    assert restored.body is not None, "Body is mandatory"
    assert len(restored.body.bars) > 0, "Must have at least one bar"
    # ✅ SHIP
```

---

## Anti-Pattern Catalog (with Refactoring Examples)

### Anti-Pattern 1: Direct Private Function Testing

**Problem:** Test directly calls `_private_function()`, tightly coupling the test to internal implementation.

**Why It's Bad:**
- Refactoring private functions breaks tests (even if public behavior is unchanged)
- Makes the test suite a changelog of implementation, not a spec of contracts
- Hard to understand *why* the private function matters

**Examples in Repo:**
```
tests/unit/viz/test_viz_builders_extended.py:10-30
  - test_probability_helper_utilities_and_segments()
  - Calls: _looks_like_probability_values(), _ensure_indexable_length(), _legacy_color_brew()

tests/unit/core/test_helpers.py:30-90
  - test_safe_import_*(), test_safe_isinstance_*()
  - While reasonable, lacks context on usage
```

**Refactoring Decision Tree:**

```
┌─ Is this private helper a reusable, stable domain concept?
│
├─ YES → Make it public
│   ├─ Rename: _func() → func() (or func_internal() if truly internal)
│   ├─ Add docstring with contract
│   ├─ Export from __init__ if user-facing
│   ├─ Update test import: from module import func
│   └─ Add integration test showing real usage
│
├─ NO, it's implementation-specific → Remove direct test
│   ├─ Test through the public function that uses it
│   ├─ Add comment linking to the public API
│   ├─ Create integration test if public API is complex
│   └─ Delete the private helper test
│
└─ UNCERTAIN? → Ask: "Would a plugin developer care about this?"
    └─ If NO → It's implementation; delete the test
```

**Example Refactoring:**

**BEFORE (❌ Anti-pattern):**
```python
# tests/unit/viz/test_viz_builders_extended.py
def test_probability_helper_utilities_and_segments():
    assert builders._looks_like_probability_values(0.0, 0.5, "1.0")
    assert not builders._looks_like_probability_values()
    assert not builders._looks_like_probability_values(0.1, math.inf)
```

**AFTER (✅ Refactored):**

**Option 1: Make Helper Public**
```python
# src/calibrated_explanations/viz/builders.py
def is_valid_probability_values(*values: Any) -> bool:
    """Check if all values are valid probabilities in [0, 1].

    Raises:
        ValueError: If any value cannot be converted to float.
    """
    # Implementation (formerly _looks_like_probability_values)
    for v in values:
        try:
            fv = float(v)
            if not 0.0 <= fv <= 1.0:
                return False
        except (ValueError, TypeError):
            return False
    return len(values) > 0

# tests/unit/viz/test_builders_validation.py (new file)
def test_is_valid_probability_values__should_return_true_for_valid_probabilities():
    from calibrated_explanations.viz.builders import is_valid_probability_values
    assert is_valid_probability_values(0.0, 0.5, 1.0)
    assert is_valid_probability_values("0.5")

def test_is_valid_probability_values__should_return_false_for_invalid_values():
    from calibrated_explanations.viz.builders import is_valid_probability_values
    assert not is_valid_probability_values(1.1)
    assert not is_valid_probability_values(-0.1)
    assert not is_valid_probability_values()  # Empty args
```

**Option 2: Keep Private, Test Through Public API**
```python
# tests/integration/viz/test_plotspec_building.py
def test_build_regression_bars_spec__should_reject_invalid_probabilities():
    """Verify that invalid probability bounds are caught during spec construction."""
    with pytest.raises(ValueError, match="probability"):
        build_regression_bars_spec(
            title="test",
            predict={"predict": 1.5, "low": 0.2, "high": 0.8}  # Invalid: 1.5 > 1.0
        )

def test_build_regression_bars_spec__should_construct_valid_spec_with_valid_bounds():
    """Verify that valid probability specs are constructed correctly."""
    spec = build_regression_bars_spec(
        title="test",
        predict={"predict": 0.5, "low": 0.2, "high": 0.8},
        feature_weights={"predict": np.array([0.1, -0.2])},
        features_to_plot=[0, 1],
        column_names=["f0", "f1"],
        instance=np.array([0.3, 0.4]),
        y_minmax=(0.0, 1.0),
        interval=True,
    )

    # Semantic assertions (not dict keys)
    assert spec.header.low <= spec.header.pred <= spec.header.high
    assert spec.body is not None
    assert len(spec.body.bars) == 2
```

---

### Anti-Pattern 2: Assertion on Dict Keys / Internal Attributes

**Problem:** Test verifies that `dict` has specific keys or object has specific attributes, rather than asserting on domain meaning.

**Why It's Bad:**
- Brittle to refactoring (renaming keys breaks tests)
- Doesn't validate that the data is correct
- Can pass while the data is semantically invalid

**Examples in Repo:**
```
tests/unit/core/test_fast_units.py
  - assert "predict" in d["header"]
  - assert "low" in d["header"]
  - Doesn't check: low ≤ predict ≤ high

tests/unit/viz/test_plotspec_dataclasses.py
  - assert spec.header is not None
  - assert len(spec.body.bars) >= 1
  - (These are OK, but could be stronger)
```

**Refactoring Strategy:**

```
┌─ I'm asserting on dict keys or attribute presence
│
├─ STEP 1: Identify the business rule
│   └─ Example: "Interval bounds must satisfy: low ≤ predict ≤ high"
│
├─ STEP 2: Replace key assertions with domain assertions
│   └─ BEFORE: assert "low" in d["header"]
│   └─ AFTER:  assert d["header"]["low"] <= d["header"]["predict"]
│
├─ STEP 3: Add comment linking to ADR/spec
│   └─ # Invariant per ADR-005 Explanation Envelope
│
└─ Commit refactored test ✅
```

**Example Refactoring:**

**BEFORE (❌ Anti-pattern):**
```python
def test_plotspec_header_has_all_required_fields():
    spec = PlotSpec(header=IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8))
    d = plotspec_to_dict(spec)

    assert "pred" in d["header"]  # Implementation detail!
    assert "low" in d["header"]
    assert "high" in d["header"]
    assert d["header"]["pred"] is not None
```

**AFTER (✅ Refactored):**
```python
def test_interval_header__should_satisfy_ordering_invariant():
    """Verify that interval bounds satisfy low ≤ pred ≤ high.

    Domain Rule: Intervals represent prediction with uncertainty bounds.
    Invariant: The point estimate must lie within the interval.
    Ref: ADR-005 Explanation Envelope
    """
    spec = PlotSpec(
        header=IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8)
    )
    d = plotspec_to_dict(spec)
    restored = plotspec_from_dict(d)

    # Semantic assertions (domain invariants)
    header = restored.header
    assert header.low is not None, "Lower bound is mandatory"
    assert header.pred is not None, "Point estimate is mandatory"
    assert header.high is not None, "Upper bound is mandatory"

    # The core invariant: ordering
    assert header.low <= header.pred, \
        f"Lower bound ({header.low}) should not exceed prediction ({header.pred})"
    assert header.pred <= header.high, \
        f"Prediction ({header.pred}) should not exceed upper bound ({header.high})"
```

---

### Anti-Pattern 3: Snapshot Tests Without Semantic Validation

**Problem:** Test serializes → deserializes → asserts equality, but never validates that the deserialized object is semantically correct.

**Why It's Bad:**
- A broken deserializer might still return an equal (but invalid) object
- Doesn't catch mutations in business logic
- False confidence: roundtrip succeeds while data is wrong

**Example in Repo:**
```
tests/unit/viz/test_viz_serializers.py:16-23
  d = plotspec_to_dict(spec)
  s2 = plotspec_from_dict(d)
  assert s2 == spec  # ← Only checks equality, not correctness
```

**Refactoring Strategy:**

```
Snapshot Test Pattern:
1. Create object
2. Serialize → Dict
3. Deserialize → Object
4. Assert equality (snapshot)
5. ADD: Assert invariants (semantics) ← THIS IS MISSING
```

**Example Refactoring:**

**BEFORE (❌ Anti-pattern):**
```python
def test_plotspec_roundtrip_and_validate():
    spec = PlotSpec(
        header=IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8),
        title="My Plot",
        body=BarHPanelSpec(bars=[BarItem(label="f1", value=0.1)])
    )
    d = plotspec_to_dict(spec)
    s2 = plotspec_from_dict(d)
    assert s2 == spec  # ← That's it!
```

**AFTER (✅ Refactored):**
```python
def test_plotspec_roundtrip__should_preserve_serialization_and_semantics():
    """Verify that roundtrip preserves both structure (snapshot) and semantics (invariants)."""
    original = PlotSpec(
        header=IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8),
        title="My Plot",
        body=BarHPanelSpec(bars=[BarItem(label="f1", value=0.1)])
    )

    # Roundtrip
    d = plotspec_to_dict(original)
    restored = plotspec_from_dict(d)

    # Snapshot check (structure preservation)
    assert restored == original, "Roundtrip should preserve structure"

    # Semantic checks (domain invariants)
    # ← ADD THESE
    assert restored.header is not None, "Header is mandatory"
    assert restored.header.low <= restored.header.pred <= restored.header.high, \
        "Interval invariant violated"
    assert restored.body is not None, "Body is mandatory"
    assert len(restored.body.bars) > 0, "Must have at least one bar"
    assert all(b.label for b in restored.body.bars), "All bars must have labels"
```

**Parametrized Variant (Test Edge Cases):**
```python
@pytest.mark.parametrize("header", [
    IntervalHeaderSpec(pred=0.0, low=0.0, high=0.0),    # All zeros
    IntervalHeaderSpec(pred=1.0, low=0.0, high=1.0),    # Boundaries
    IntervalHeaderSpec(pred=0.5, low=0.5, high=0.5),    # Point estimate
    IntervalHeaderSpec(pred=0.5, low=0.0, high=1.0),    # Wide interval
])
def test_plotspec_roundtrip_with_edge_case_intervals(header):
    """Verify roundtrip handles edge cases."""
    spec = PlotSpec(header=header, body=BarHPanelSpec(bars=[BarItem(label="x", value=0.1)]))

    restored = plotspec_from_dict(plotspec_to_dict(spec))

    # Invariant holds at all edges
    assert restored.header.low <= restored.header.pred <= restored.header.high
```

---

### Anti-Pattern 4: Mock-Heavy Tests Without Behavior Validation

**Problem:** Test mocks >50% of dependencies and verifies mock calls (spy assertions) rather than actual behavior.

**Why It's Bad:**
- Mocks shadow real contracts; tests pass even if the system breaks
- Hard to distinguish real bugs from mock misconfigurations
- Doesn't validate end-to-end correctness

**Example in Repo:**
```
tests/plugins/test_protocols.py:65-85
  - Defines mock plugins
  - Only checks isinstance(plugin, ExplanationPlugin)
  - Never actually registers or uses the plugin

tests/unit/core/test_plugin_registry.py
  - Tests registry with mock DummyPlugin
  - Never verifies real plugins work
```

**Refactoring Strategy:**

```
FOR EACH MOCK-HEAVY TEST:
1. Keep the unit test (validates interface contract)
2. ADD an integration test (validates real system works)
3. In unit test: use OUTCOME assertions, not SPY assertions
4. In integration test: use REAL implementations, not mocks
```

**Example Refactoring:**

**BEFORE (❌ Anti-pattern):**
```python
# Unit test: Mocks too much; only validates structure
class _DummyPlugin:
    plugin_meta = {...}
    def supports_mode(self, mode, *, task): return True
    def explain_batch(self, x, request): return ExplanationBatch(...)

def test_explanation_plugin_runtime_checks():
    plugin = _DummyPlugin()
    assert isinstance(plugin, ExplanationPlugin)  # ← Only checks type!
    # No assertion about behavior; doesn't test registration, discovery, etc.
```

**AFTER (✅ Refactored):**

**Unit Test (Keep, but strengthen):**
```python
# Still mocked, but with outcome assertions
def test_explanation_plugin__should_implement_required_interface():
    """Verify plugin structure and basic contract."""
    plugin = _DummyPlugin()

    # Protocol check
    assert isinstance(plugin, ExplanationPlugin)

    # Outcome assertions (what should the plugin do?)
    ctx = ExplanationContext(...)
    plugin.initialize(ctx)  # Should not raise

    batch = plugin.explain_batch(
        x=np.array([[0.1, 0.2]]),
        request=ExplanationRequest(threshold=0.5)
    )

    # Verify output structure and semantics
    assert isinstance(batch, ExplanationBatch)
    assert len(batch.instances) > 0, "Should produce explanations"
```

**Integration Test (Add):**
```python
# tests/integration/plugins/test_plugin_workflow.py
def test_plugin_registration_discovery_explain_workflow():
    """Verify complete plugin lifecycle with real implementations."""
    # 1. Register real plugin
    plugin = SimpleExplainerPlugin()  # ← NOT a mock!
    registry.clear()
    registry.register(plugin)

    # 2. Discover plugin
    model = RandomForestClassifier()
    model.fit([[0, 0], [1, 1]], [0, 1])

    found = registry.find_for(model)
    assert plugin in found, "Plugin should be discoverable"

    # 3. Initialize plugin
    ctx = ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=("f0", "f1"),
        categorical_features=[],
        predict_bridge=PredictBridge(model),
        ...
    )
    plugin.initialize(ctx)

    # 4. Generate explanations
    x_test = np.array([[0.5, 0.5]])
    request = ExplanationRequest(threshold=0.5)
    batch = plugin.explain_batch(x=x_test, request=request)

    # 5. Verify output is sensible
    assert isinstance(batch, ExplanationBatch)
    assert len(batch.instances) == len(x_test), "One explanation per instance"
    assert batch.container_cls is not None, "Container class should be defined"
```

---

### Anti-Pattern 5: Dataclass Structural Tests (Tied to Implementation)

**Problem:** Test verifies that an object is a dataclass or uses `FrozenInstanceError`, tightly coupling to the implementation mechanism.

**Why It's Bad:**
- If you switch to a different mechanism (e.g., `__slots__`), the test breaks
- Doesn't verify the intended contract (e.g., immutability)
- Test is specific to Python's dataclass implementation

**Example in Repo:**
```
tests/plugins/test_protocols.py:29-40
  assert dataclasses.is_dataclass(ctx)  # ← Implementation detail!
```

**Refactoring Strategy:**

```
Immutability Testing Pattern:
1. Create object
2. Try to modify it
3. Catch generic Exception (not FrozenInstanceError)
4. Verify the modification didn't take effect
```

**Example Refactoring:**

**BEFORE (❌ Anti-pattern):**
```python
def test_explanation_context_is_frozen() -> None:
    ctx = ExplanationContext(...)
    with pytest.raises(FrozenInstanceError):
        ctx.feature_names = ("new",)
```

**AFTER (✅ Refactored):**
```python
def test_explanation_context__should_be_immutable_after_construction() -> None:
    """Verify that context objects cannot be modified after construction.

    Contract: ExplanationContext is a value object passed to plugins.
    It must remain stable during the explanation lifecycle.
    """
    ctx = ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=("f0", "f1"),
        categorical_features=[],
        predict_bridge=None,
        test_data=None,
        target_data=None
    )

    # Attempt modification
    try:
        ctx.feature_names = ("new",)
    except Exception:
        pass  # We expect some exception, don't care which one

    # Verify state is unchanged
    assert ctx.feature_names == ("f0", "f1")
```

---

## Positive Pattern Examples

### Example 1: Behavior-Focused Unit Test

```python
def test_calibrated_explainer__should_require_fit_before_explain():
    """Verify that unfitted explainers cannot generate explanations."""
    explainer = CalibratedExplainer(RandomForestClassifier(), x_train, y_train)

    # Act & Assert
    with pytest.raises(RuntimeError, match="must be fitted"):
        explainer.explain_factual(x_test)
```

**Why It's Good:**
- Tests a user-facing contract ("can't explain before fitting")
- Would survive refactoring if internals change
- Could be stated as: "explainers enforce a fit-before-explain contract"

---

### Example 2: Parametrized Behavior Test

```python
@pytest.mark.parametrize("task,n_classes", [
    ("classification", 2),   # Binary classification
    ("classification", 3),   # Multiclass
    ("regression", 1),        # Regression
])
def test_explainer__should_infer_task_from_model(task, n_classes):
    """Verify that the explainer correctly identifies the task type from the model structure."""
    model = create_mock_model(n_classes=n_classes)
    explainer = CalibratedExplainer(model, x_train, y_train)

    assert explainer.task == task
```

---

## Fallback Chain Testing

### Policy: No Fallbacks by Default

**Rule:** Tests MUST NOT trigger fallback chains unless explicitly testing fallback behavior.

**Mechanism:** The test suite uses fixtures to control fallback chain behavior:
- `disable_fallbacks` (autouse): Automatically disables all fallback chains for every test
- `enable_fallbacks`: Explicitly re-enables fallbacks for tests that validate fallback behavior

### How Fallbacks Are Disabled

The `disable_fallbacks` fixture empties all fallback chains by setting environment variables:

```python
# Automatically applied to all tests via autouse=True
@pytest.fixture(autouse=True)
def disable_fallbacks(monkeypatch):
    """Disable all plugin fallback chains by default.
    
    Tests that explicitly need fallbacks must use the enable_fallbacks fixture.
    """
    # Disable explanation plugin fallbacks
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS", "")
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS", "")
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FAST_FALLBACKS", "")
    
    # Disable interval plugin fallbacks
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FALLBACKS", "")
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FAST_FALLBACKS", "")
    
    # Disable plot style fallbacks
    monkeypatch.setenv("CE_PLOT_STYLE_FALLBACKS", "")
    
    # Parallel execution: no serial fallback
    monkeypatch.setenv("CE_PARALLEL_MIN_BATCH_SIZE", "999999")
```

### Writing Tests That Validate Fallbacks

If your test explicitly validates fallback behavior, use the `enable_fallbacks` fixture:

```python
def test_explanation_plugin__should_fallback_when_primary_fails(enable_fallbacks):
    """Verify fallback chain activates when primary plugin fails."""
    # This test explicitly validates fallback behavior
    explainer = CalibratedExplainer(
        model, x_cal, y_cal,
        _explanation_plugin_override="intentionally-missing"
    )
    
    # Should fall back to default plugin and not raise
    with pytest.warns(UserWarning, match="fallback"):
        explanation = explainer.explain_factual(x_test)
    
    assert explanation is not None
```

### Detecting Fallback Violations in CI

The CI pipeline enforces this policy using pytest's warning capture:

1. **Strict warning mode**: `pytest -Werror::UserWarning` treats warnings as errors
2. **Fallback detection**: All fallback paths emit `UserWarning` (per Fallback Visibility Policy)
3. **Automatic failure**: Tests that trigger unexpected fallbacks fail in CI

### Common Fallback Warning Patterns

If your test fails with one of these warnings, it's triggering a fallback:

```
UserWarning: Execution plugin error; legacy sequential fallback engaged
UserWarning: Parallel failure; forced serial fallback engaged  
UserWarning: Cache backend fallback: using minimal in-package LRU/TTL implementation
UserWarning: Visualization fallback: alternative bar simplified due to drawing error
UserWarning: Perturbation fallback: deterministic swap applied due to degenerate RNG state
UserWarning: Narrative template fallback: default template used because provided path was missing
```

**Fix strategies:**
1. **If the fallback is unexpected**: Fix the root cause (missing plugin, configuration error, etc.)
2. **If the test validates fallback**: Add `enable_fallbacks` fixture parameter
3. **If testing edge cases**: Mock the failure condition instead of triggering actual fallback

### Example: Refactoring a Fallback-Dependent Test

**BEFORE (❌ Fails in CI):**
```python
def test_explanation_with_missing_plugin():
    """Test explanation when plugin is missing."""
    explainer = CalibratedExplainer(model, x_cal, y_cal)
    # This silently falls back to legacy implementation
    explanation = explainer.explain_factual(x_test)
    assert explanation is not None  # False confidence!
```

**AFTER (✅ Fixed):**

**Option 1: Test the primary path**
```python
def test_explanation_with_registered_plugin():
    """Verify explanation uses the registered plugin."""
    explainer = CalibratedExplainer(model, x_cal, y_cal)
    # Fallbacks disabled; will raise if plugin missing
    explanation = explainer.explain_factual(x_test)
    assert explanation is not None
```

**Option 2: Explicitly test fallback**
```python
def test_explanation__should_fallback_to_legacy_when_plugin_fails(enable_fallbacks):
    """Verify graceful degradation when plugin execution fails."""
    explainer = CalibratedExplainer(model, x_cal, y_cal)
    # Explicitly testing fallback behavior
    with pytest.warns(UserWarning, match="fallback"):
        explanation = explainer.explain_factual(x_test)
    assert explanation is not None
```

### Integration with Test Helpers

The test helper module (`tests/helpers/fallback_control.py`) provides utilities for fallback management:

```python
from tests.helpers.fallback_control import (
    assert_no_fallbacks_triggered,
    disable_all_fallbacks,
    enable_specific_fallback,
)

def test_complex_workflow():
    """Verify workflow without fallbacks."""
    # Ensure no fallbacks during test
    with assert_no_fallbacks_triggered():
        explainer = CalibratedExplainer(...)
        result = explainer.explain_factual(x_test)
        assert result is not None
```

### Rationale

**Why this matters:**
- **Test reliability**: Fallbacks obscure real failures
- **Coverage accuracy**: Tests should validate the primary code path
- **Debugging**: Fallback-dependent tests are hard to debug
- **Regression detection**: Silent fallbacks hide regressions

**When fallbacks are acceptable:**
- Testing error recovery paths
- Testing graceful degradation
- Testing compatibility with missing optional dependencies
- Testing the fallback mechanism itself
