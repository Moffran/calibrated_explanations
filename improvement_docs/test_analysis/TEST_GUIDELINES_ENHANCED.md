# Enhanced Test Guidelines

**Status:** Proposed Additions to `.github/instructions/tests.instructions.md`
**Date:** November 18, 2025
**Scope:** Detailed decision trees, examples, and patterns for writing behavior-focused tests

---

## 📋 Table of Contents

1. [Behavior vs. Implementation: Decision Tree](#behavior-vs-implementation-decision-tree)
2. [Anti-Pattern Catalog](#anti-pattern-catalog-with-refactoring-examples)
3. [Positive Pattern Examples](#positive-pattern-examples)
4. [Private Helpers: Guidelines](#private-helpers-guidelines)
5. [Mocking: Best Practices](#mocking-best-practices)
6. [Snapshots and Serialization](#snapshots-and-serialization)
7. [Test Naming Convention (Extended)](#test-naming-convention-extended)
8. [Integration Testing Checklist](#integration-testing-checklist)
9. [Common Pitfalls & Solutions](#common-pitfalls--solutions)

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

    # 6. Verify batch can be collected and plotted
    collection = batch.container_cls(batch.instances)
    collection.plot(show=False)  # Should not raise
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
  with pytest.raises(dataclasses.FrozenInstanceError):  # ← Tied to mechanism!
      ctx.mode = "alternative"
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
    assert dataclasses.is_dataclass(ctx)  # ← Implementation detail!

    with pytest.raises(dataclasses.FrozenInstanceError):  # ← Tied to dataclass
        ctx.mode = "alternative"
```

**AFTER (✅ Refactored):**
```python
def test_explanation_context__should_be_immutable_after_construction() -> None:
    """Verify that context objects cannot be modified after construction.

    Contract: ExplanationContext is immutable to prevent accidental mutations
    during the explanation pipeline.

    Note: Implementation uses @dataclass(frozen=True), but the test verifies
    the contract (immutability), not the mechanism (FrozenInstanceError).
    If implementation changes (e.g., to __slots__), this test remains valid.
    """
    ctx = ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=("f0", "f1"),
        ...
    )

    original_mode = ctx.mode

    # Attempt to modify (generic exception catching)
    try:
        ctx.mode = "alternative"
        # If no exception, the modification succeeded; verify it didn't take effect
        assert ctx.mode == original_mode, "Modification should not have taken effect"
        # But we expect an exception, so reach here = test failure
        pytest.fail("Expected mutation to raise an exception")
    except (TypeError, AttributeError, Exception) as e:
        # ← Generic exception types; not tied to dataclass implementation
        # Expected: the context should be immutable
        pass

    # Verify context still has original state
    assert ctx.mode == original_mode
    assert ctx.feature_names == ("f0", "f1")
```

---

## Positive Pattern Examples

### Example 1: Behavior-Focused Unit Test

```python
def test_calibrated_explainer__should_require_fit_before_explain():
    """Verify that unfitted explainers cannot generate explanations."""
    explainer = WrapCalibratedExplainer(RandomForestClassifier())

    # Not yet fitted
    with pytest.raises(ValueError, match="not fitted"):
        explainer.explain_factual(np.array([[0.1, 0.2]]))

    # After fitting, should work
    x_train = np.array([[0, 0], [1, 1]])
    y_train = np.array([0, 1])
    explainer.fit(x_train, y_train)

    explanation = explainer.explain_factual(np.array([[0.5, 0.5]]))
    assert explanation is not None
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
    ("classification", 5),   # Multiclass classification
    ("regression", 1),        # Regression
])
def test_explainer__should_infer_task_from_model(task, n_classes):
    """Verify that task type is correctly inferred from model."""
    if task == "classification":
        model = LogisticRegression() if n_classes == 2 else OneVsRestClassifier(LogisticRegression())
        explainer = WrapCalibratedExplainer(model)
        explainer.fit(X_train_clf, y_train_clf)

        assert explainer.task == "classification"
        assert explainer.is_multiclass == (n_classes > 2)

    elif task == "regression":
        model = LinearRegression()
        explainer = WrapCalibratedExplainer(model)
        explainer.fit(X_train_reg, y_train_reg)

        assert explainer.task == "regression"
        assert explainer.is_multiclass == False
```

**Why It's Good:**
- Tests multiple scenarios with clear naming
- Parametrization avoids loops
- Each case is independent; failure is clear

---

### Example 3: Integration Test with Real System

```python
def test_classification_workflow_with_custom_calibrator(tmp_path):
    """Verify end-to-end classification with custom interval calibrator.

    Scenario: User wants to use a custom calibrator instead of default.
    Expectation: System should respect custom calibrator and produce valid explanations.
    """
    # 1. Prepare data
    x_train, x_cal, x_test = make_classification_data()
    y_train, y_cal, y_test = make_classification_labels()

    # 2. Train model and calibrator
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    custom_calibrator = CustomIntervalCalibrator()

    # 3. Create explainer with custom calibrator
    explainer = WrapCalibratedExplainer(
        model,
        interval_calibrator=custom_calibrator
    )
    explainer.fit(x_train, y_train)
    explainer.calibrate(x_cal, y_cal)

    # 4. Generate explanation
    explanation = explainer.explain_factual(x_test[0:1])

    # 5. Verify explanation uses custom calibrator
    assert explanation is not None
    assert explanation.calibrator_name == "CustomIntervalCalibrator"

    # 6. Verify explanation can be plotted and saved
    plot_path = tmp_path / "explanation.png"
    explanation.plot(save_path=str(plot_path), show=False)
    assert plot_path.exists(), "Plot should be saved"
    assert plot_path.stat().st_size > 1000, "Plot should not be trivial"

    # 7. Verify explanation is serializable
    serialized = explanation.to_dict()
    assert "prediction" in serialized
    assert "feature_weights" in serialized
    assert "intervals" in serialized
```

**Why It's Good:**
- Tests real user workflow
- Verifies integration between components (model, calibrator, explainer, plotting)
- Catches regressions that unit tests might miss

---

## Private Helpers: Guidelines

### When to Make a Helper Public

**Make a helper public if:**
1. It implements a stable, reusable domain concept (e.g., `is_valid_probability()`)
2. Multiple modules use it
3. Plugin developers might benefit from it
4. It has clear, documentable semantics

**Keep a helper private if:**
1. It's implementation-specific (e.g., `_legacy_color_brew()`)
2. Only one module uses it
3. The interface is likely to change
4. Plugin developers don't need it

### Testing Private Helpers

**Option A: Remove the Test**
- If the private helper is truly internal, remove the direct test
- Test it through the public function that uses it

**Option B: Make It Public**
- If the helper is stable and useful, promote it to public
- Add docstring and type hints
- Export from module's `__init__` if user-facing
- Test as a public API

**Option C: Move to Integration Test**
- If the helper is used in complex workflows, test it through end-to-end scenarios
- Example: test color rendering through actual plot output

---

## Mocking: Best Practices

### Mocking Checklist

- [ ] **Isolation:** Mocks are used to avoid slow I/O, network, or RNG (not to hide implementation)
- [ ] **Outcome Assertions:** Tests assert on the *result*, not on mock calls (no `mock.assert_called_with`)
- [ ] **Integration Pairing:** For every mock-heavy unit test, there's a real integration test
- [ ] **Readability:** Mock implementations are minimal and documented
- [ ] **Reusability:** Mocks are defined in fixtures or `conftest.py`, not copied

### Example: Good Mock Usage

```python
# ✅ GOOD: Mock is for avoiding I/O, outcome is asserted
@pytest.fixture
def mock_filesystem(monkeypatch):
    """Mock filesystem operations to avoid actual file creation."""
    saved_files = {}

    def mock_save(path, data):
        saved_files[path] = data

    monkeypatch.setattr("builtins.open", mock_save)
    return saved_files

def test_explanation_saves_to_file(mock_filesystem):
    """Verify that explanations can be saved without actual I/O."""
    explanation = create_dummy_explanation()

    # Save to (mocked) file
    explanation.save("/tmp/exp.json")

    # Outcome assertion: file was saved with correct data
    assert "/tmp/exp.json" in mock_filesystem
    assert mock_filesystem["/tmp/exp.json"]["prediction"] == explanation.prediction
    # ✅ Not: mock_open.assert_called_with(...), which is implementation-tied

def test_explanation_save_workflow_with_real_filesystem(tmp_path):
    """Integration test: verify save works with real filesystem."""
    explanation = create_dummy_explanation()

    save_path = tmp_path / "explanation.json"
    explanation.save(str(save_path))

    # Outcome assertion: file exists and is readable
    assert save_path.exists()
    with open(save_path) as f:
        data = json.load(f)
    assert data["prediction"] == explanation.prediction
```

---

## Snapshots and Serialization

### Snapshot Testing Rules

1. **Use snapshots for stable, non-algorithmic outputs** (JSON, serialized objects)
2. **Pair snapshots with semantic assertions** (domain invariants)
3. **Keep snapshots small and focused** (one assertion per logical test)
4. **Review snapshot changes carefully** (could hide bugs)

### Pattern

```python
def test_explanation_serialization_snapshot():
    """Verify serialization format is stable (snapshot + semantics)."""
    explanation = create_explanation(pred=0.5, low=0.2, high=0.8)

    # Serialize
    data = explanation.to_dict()

    # Snapshot: the structure is stable
    assert data == {
        "prediction": 0.5,
        "intervals": {"low": 0.2, "high": 0.8},
        "feature_weights": [...],
        "metadata": {...},
    }

    # Semantic checks: the data is correct
    assert data["intervals"]["low"] <= data["prediction"] <= data["intervals"]["high"]
    assert all("label" in feat for feat in data["feature_weights"])
```

---

## Test Naming Convention (Extended)

### Pattern

```
test_<module>_<class>__should_<behavior>_when_<condition>
```

### Examples

**Good Names:**
- ✅ `test_calibrated_explainer__should_require_fit_before_explain()`
- ✅ `test_interval_calibrator__should_update_thresholds_when_calibrate_called()`
- ✅ `test_plotspec_roundtrip__should_preserve_interval_invariants()`
- ✅ `test_plugin_registry__should_discover_plugins_by_model_type()`

**Ambiguous Names (Clarify):**
- ⚠️ `test_explanation_plot()` → `test_explanation_plot__should_render_without_error()`
- ⚠️ `test_calibration()` → `test_calibration__should_reduce_prediction_uncertainty_on_held_out_data()`

**Bad Names (Anti-patterns):**
- ❌ `test_helpers()` — Too vague
- ❌ `test_implementation_detail()` — Red flag; reconsider if this is needed
- ❌ `test_private_function_X()` — Signals direct private testing

---

## Integration Testing Checklist

Use this checklist when deciding to add an integration test:

- [ ] Does this test exercise multiple components together?
- [ ] Does it verify a user-facing workflow or contract?
- [ ] Would this catch bugs that unit tests miss?
- [ ] Does it require I/O, real models, or external dependencies?
- [ ] Is it too slow for unit test suite (mark with `@pytest.mark.slow`)?

**If YES to ≥3 of these → Add as integration test**

### Integration Test Structure

```python
# tests/integration/<feature>/test_<feature>.py

@pytest.mark.integration
def test_<workflow>_with_<scenario>():
    """Comprehensive test of a user-facing workflow.

    Scenario: <Describe the user's intent>
    Steps:
    1. Prepare data
    2. Train/setup components
    3. Execute workflow
    4. Verify end-to-end correctness
    """
    # 1. Prepare
    data = prepare_test_data()

    # 2. Setup
    model = train_model(data)

    # 3. Execute
    result = execute_workflow(model, data)

    # 4. Verify
    assert workflow_is_correct(result)
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Testing Implementation Instead of Behavior

**Symptom:** Test fails when you refactor internals without changing behavior

**Solution:** Rewrite test to assert on observable behavior
```python
# ❌ BEFORE: Tests implementation (keys)
assert "low" in spec_dict["header"]

# ✅ AFTER: Tests behavior (invariant)
assert spec_dict["header"]["low"] <= spec_dict["header"]["predict"]
```

---

### Pitfall 2: Mocking Everything

**Symptom:** Test passes but system breaks in production

**Solution:** Pair mock tests with integration tests
```python
# Add to tests/integration/...:
def test_real_workflow():
    real_obj = RealClass()  # No mocks
    result = real_obj.do_something()
    assert result is valid
```

---

### Pitfall 3: Tests That Are Too Specific

**Symptom:** Test breaks on every minor refactoring

**Solution:** Test the contract, not the implementation
```python
# ❌ BEFORE: Specific to current implementation
assert x.internal_state["counter"] == 5

# ✅ AFTER: Tests the contract
assert x.get_count() == 5
```

---

### Pitfall 4: Large, Monolithic Tests

**Symptom:** Hard to understand what failed; takes long to run

**Solution:** Split into focused tests
```python
# ❌ BEFORE: One test that does everything
def test_full_pipeline():
    # 50 lines of setup, execution, verification

# ✅ AFTER: Focused tests with clear purpose
def test_fit__should_not_raise_with_valid_data(): ...
def test_explain__should_return_explanation_after_fit(): ...
def test_plot__should_save_to_specified_path(): ...
```

---

## Conclusion

**Core Principle:** Write tests that specify *behavior* and *contracts*, not implementation details. A test is good if it survives refactoring and catches real bugs.

**Rule of Thumb:** If you can't explain your test to a non-engineer as a business rule or domain invariant, it's probably testing implementation.

---

**Last Updated:** 2025-11-18
**Status:** Proposed for Integration into `.github/instructions/tests.instructions.md`
**Review:** See `TEST_QUALITY_ANALYSIS.md` for findings and justification
